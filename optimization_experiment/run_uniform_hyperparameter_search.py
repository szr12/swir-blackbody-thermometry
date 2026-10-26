"""等间隔波段的Validation超参搜索；冻结后才评价两组开发留出集。"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_experiment/src"))

import numpy as np
import torch

from convergent_colorimetry import FixedBandColorimetricMLP
from shared_feature_regression import SharedFeatureMLP, SharedLogHelmert
from traditional_utils import (load_data, fold_quality_mask, model_parameters,
                               make_configured_split, regression_metrics, write_json)
from regional_colorimetry import regional_training_masks, soft_region_weights
from run_component_ablations import control_bands, audit_split

DATA_HASH = "81e3a456a1e57bc65d6b7a44e71c21aa4d2dee271927f2173c741f740a3ae2dc"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guarded(path, value):
    from traditional_utils import json_ready
    if path.exists() and read(path) != json_ready(value):
        raise ValueError(f"冻结内容不匹配：{path}")
    write_json(path, value)


def rank_candidates(rows, family=None):
    """仅使用预先规定的收敛条件和等权Validation指标，不读取Test。"""
    candidates = [r for r in rows if r["success"] and np.isfinite(r["validation_rmse_mean"])
                  and (family is None or r["family"] == family)]
    return sorted(candidates, key=lambda r: (r["validation_rmse_mean"], r["band_count"],
        r["validation_rmse_sd"], r["family"], r["overlap_c"], r["routing_width_c"]))


def run(args):
    output, project = args.output_dir.resolve(), args.project_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if project / "artifacts/results" in output.parents:
        raise ValueError("不能覆盖历史证据")
    search = read(ROOT / "optimization_experiment/configs/uniform_hyperparameter_search_20261023.json")
    base = read(ROOT / "optimization_experiment/configs/component_ablation_refrozen_k12_random_20261013.json")
    path = project / base["data_path"]
    if digest(path) != DATA_HASH:
        raise ValueError("输入数据指纹不符")
    protocol = {"search": search, "model_config": base, "data_sha256": DATA_HASH,
        "sources_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted(ROOT.rglob("*.py"))},
        "selection_scope": "random Train280 fits; Validation60 unweighted RMSE averaged over three fixed seeds; no Test in candidates",
        "search_stages": ["coarse K with fixed overlap55/routing15 for all three families",
            "overlap/routing grid for top two K of each regional family",
            "integer K refinement within plus/minus2 of each family winner with its other settings fixed"],
        "training": "original adaptive attention training and stopping unchanged; Validation scoring equally weighted",
        "uniform_rule": "rounded equally spaced positions in Train-QC candidate pool, retain original long-wave constraint; all-QC endpoint explicit",
        "frozen_before_test": True,
        "scope": "development search after viewing historical Test; final frozen configurations evaluated on same-batch random and ordered holdouts; no independent generalization claim",
        "failure_policy": "report all failures; selection requires all three training seeds converged; no seed substitution or Test retries"}
    guarded(output / "protocol.json", protocol)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    write_json(output / "environment.json", {"python": sys.version, "numpy": np.__version__,
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
    data = load_data(path, base["feature_key"], base["uncertainty_key"])
    X, y, wave = data[base["feature_key"]], data["y_temperature_c"], data["wavelengths_nm"]
    boundaries = np.asarray(base["temperature_boundaries_c"])
    train, validation, test = make_configured_split(y, boundaries, base["split"])
    audit_split(y, train, validation, test, False)
    guarded(output / "split_manifest.json", {"train_indices": train, "validation_indices": validation, "test_indices": test})
    quality, qc = fold_quality_mask(X[train], data[base["uncertainty_key"]][train], data["background_mad"][train],
        data["bad_pixel_fraction_by_sample_band"][train], data["saturation_fraction_by_sample_band"][train],
        wave, base["fold_quality_control"])
    available = np.flatnonzero(quality)
    guarded(output / "quality.json", {"available_positions": available, "diagnostics": qc})
    params = model_parameters(base)
    seeds = search["seeds"]
    state = {"pid": os.getpid(), "status": "running", "stage": None, "current": None,
             "test_evaluated": False, "candidates_completed": 0, "models_completed_this_process": 0}
    def update(current=None):
        state["current"], state["updated_epoch"] = current, time.time()
        write_json(output / "queue_state.json", state)
    # 拟合缓存只按训练数据和模型设置确定，不使用评价误差决定复用。
    def fit_predictions(selected, fit, evaluation, family, overlap, seed, cache_root):
        native = family == "native_regional"
        values = X[:, selected]
        transformer = SharedLogHelmert().fit(values[fit], wave[selected])
        transformed = transformer.transform(values)
        global_path = cache_root / f"k_{len(selected)}/seed_{seed}/global"
        roles = [("global", np.ones(len(fit), dtype=bool), global_path, False)]
        if family != "global":
            roles.extend((f"expert_{i}", mask,
                cache_root / f"k_{len(selected)}/seed_{seed}/{family}/overlap_{overlap:g}/expert_{i}", native)
                for i, mask in enumerate(regional_training_masks(y[fit], boundaries, overlap)))
        predictions, diagnostics = [], []
        for role, (label, mask, directory, is_native) in enumerate(roles):
            directory.mkdir(parents=True, exist_ok=True)
            model_path, pred_path, diagnostic_path = directory / "model.pt", directory / "prediction.npy", directory / "diagnostic.json"
            update(str(directory.relative_to(output)))
            fit_values = values[fit][mask] if is_native else transformed[fit][mask]
            eval_values = values[evaluation] if is_native else transformed[evaluation]
            if pred_path.exists() and diagnostic_path.exists():
                prediction, diagnostic = np.load(pred_path), read(diagnostic_path)
            else:
                if int(mask.sum()) < 10:
                    raise ValueError("训练样本不足")
                if model_path.exists():
                    model = torch.load(model_path, weights_only=False)
                else:
                    model = FixedBandColorimetricMLP(seed=seed + role, **params) if is_native else SharedFeatureMLP(seed=seed + role, **params)
                    if is_native:
                        model.fit(fit_values, y[fit][mask], wave[selected])
                    else:
                        model.fit(fit_values, y[fit][mask])
                    torch.save(model, directory / "model.tmp.pt")
                    (directory / "model.tmp.pt").replace(model_path)
                prediction = model.predict(eval_values)
                if not np.isfinite(prediction).all():
                    raise ValueError("预测非有限")
                diagnostic = {"role": label, "seed": seed + role, "n_fit": int(mask.sum()),
                    "fit_indices": fit[mask], "evaluation_indices": evaluation,
                    "temperature_range_c": [float(y[fit][mask].min()), float(y[fit][mask].max())],
                    "success": bool(model.success_), "iterations": model.iterations_,
                    "training_rmse_c": model.training_rmse_c_, "training_adaptive_rmse_c": model.training_adaptive_rmse_c_,
                    "feature_mode": "expert_specific" if is_native else "global_shared",
                    "transformation": {"floors": model.floors_, "offsets": model.offsets_,
                        "feature_mean": model.feature_mean_, "feature_scale": model.feature_scale_} if is_native else transformer.snapshot()}
                write_json(diagnostic_path, diagnostic)
                write_json(directory / "trace.json", {"trace": model.trace_})
                np.save(pred_path, prediction)
                state["models_completed_this_process"] += 1
            predictions.append(prediction); diagnostics.append(diagnostic)
        return np.column_stack(predictions), diagnostics

    candidate_rows = []
    def candidate(family, k, overlap, routing, stage):
        key = f"{family}_k{k}_overlap{overlap:g}_routing{routing:g}"
        folder = output / "candidates" / key
        summary_path = folder / "summary.json"
        if summary_path.exists():
            result = read(summary_path)
        else:
            folder.mkdir(parents=True, exist_ok=True)
            selected = control_bands(available, wave, k, 4242, "uniform_bands")
            guarded(folder / "selection.json", {"original_band_indices": data["original_band_indices"][selected],
                "selected_positions": selected, "wavelengths_nm": wave[selected]})
            records, all_predictions = [], []
            for seed in seeds:
                try:
                    predictions, diagnostic = fit_predictions(selected, train, validation, family, overlap,
                                                               seed, output / "validation_models")
                    prediction = predictions[:, 0] if family == "global" else np.sum(predictions[:, 1:] *
                        soft_region_weights(predictions[:, 0], boundaries, routing), axis=1)
                    metric = regression_metrics(y[validation], prediction)
                    records.append({"seed": seed, "success": all(d["success"] for d in diagnostic),
                                    "metrics": metric, "fit_diagnostics": diagnostic})
                    all_predictions.extend({"seed": seed, "sample_index": int(i), "temperature_c": float(t),
                        "prediction_c": float(p)} for i, t, p in zip(validation, y[validation], prediction))
                except Exception:
                    records.append({"seed": seed, "success": False, "error": traceback.format_exc()})
            values = [r["metrics"]["rmse"] for r in records if "metrics" in r]
            result = {"key": key, "family": family, "band_count": k, "overlap_c": overlap,
                "routing_width_c": routing, "first_stage": stage,
                "success": len(values) == len(seeds) and all(r["success"] for r in records),
                "validation_rmse_mean": float(np.mean(values)) if len(values) == len(seeds) else None,
                "validation_rmse_sd": float(np.std(values, ddof=1)) if len(values) == len(seeds) else None,
                "individual": records, "test_evaluated": False}
            if all_predictions:
                with (folder / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(all_predictions[0])); writer.writeheader(); writer.writerows(all_predictions)
            write_json(summary_path, result)
            print("CANDIDATE", key, result["validation_rmse_mean"], result["success"], flush=True)
        if not any(r["key"] == key for r in candidate_rows):
            candidate_rows.append(result)
        state["candidates_completed"] = len(candidate_rows)
        write_json(output / "candidates.json", {"candidates": candidate_rows, "test_evaluated": False})
        return result

    try:
        coarse = sorted(set(k for k in search["band_counts"] if k <= len(available)) |
                        ({len(available)} if search["include_all_quality_bands"] else set()))
        state["stage"] = "coarse_k"
        for k in coarse:
            for family in search["families"]:
                candidate(family, k, 0 if family == "global" else search["initial_overlap_c"],
                          0 if family == "global" else search["initial_routing_width_c"], "coarse_k")
        state["stage"] = "regional_overlap_routing"
        for family in search["families"][1:]:
            best = rank_candidates(candidate_rows, family)
            if not best:
                raise ValueError(f"{family}没有全部种子收敛的初筛候选")
            top_k = [r["band_count"] for r in best[:search["regional_top_k_count"]]]
            guarded(output / f"{family}_stage2_k.json", {"band_counts": top_k})
            for k in top_k:
                for overlap in search["overlap_values_c"]:
                    for routing in search["routing_width_values_c"]:
                        candidate(family, k, overlap, routing, "regional_overlap_routing")
        state["stage"] = "local_integer_k"
        centers = {family: rank_candidates(candidate_rows, family)[0] for family in search["families"]}
        guarded(output / "refinement_centers.json", {f: {k: row[k] for k in
            ("band_count", "overlap_c", "routing_width_c")} for f, row in centers.items()})
        for family, center in centers.items():
            for k in range(max(2, center["band_count"] - search["final_k_refinement_half_width"]),
                           min(len(available), center["band_count"] + search["final_k_refinement_half_width"]) + 1):
                candidate(family, k, center["overlap_c"], center["routing_width_c"], "local_integer_k")
        chosen = {family: rank_candidates(candidate_rows, family)[0] for family in search["families"]}
        best_overall = rank_candidates(list(chosen.values()))[0]
        frozen = {"status": "frozen_from_validation_only", "test_evaluated": False,
            "selection_metric": search["selection_metric"], "families": chosen,
            "best_family": best_overall["family"], "best_key": best_overall["key"],
            "freeze_epoch": time.time(), "candidate_count": len(candidate_rows),
            "candidate_evidence_sha256": digest(output / "candidates.json")}
        # 续跑时保留第一次冻结时间；科学内容必须保持不变。
        if (output / "frozen_parameters.json").exists():
            frozen["freeze_epoch"] = read(output / "frozen_parameters.json")["freeze_epoch"]
        guarded(output / "frozen_parameters.json", frozen)
        state["stage"] = "frozen_final_evaluation"
        update("frozen_before_test")
        final = []
        for name in search["final_evaluation_protocols"]:
            split_config = read(ROOT / f"optimization_experiment/configs/component_ablation_refrozen_k12_{name}_20261013.json")["split"]
            tr, va, te = make_configured_split(y, boundaries, split_config)
            audit_split(y, tr, va, te, name == "ordered")
            fit = np.sort(np.concatenate([tr, va]))
            q, qdiag = fold_quality_mask(X[tr], data[base["uncertainty_key"]][tr], data["background_mad"][tr],
                data["bad_pixel_fraction_by_sample_band"][tr], data["saturation_fraction_by_sample_band"][tr], wave, base["fold_quality_control"])
            guarded(output / "final" / name / "split_manifest.json", {"train_indices": tr, "validation_indices": va,
                "fit_indices": fit, "test_indices": te, "available_positions": np.flatnonzero(q)})
            for family, selected_config in chosen.items():
                k = selected_config["band_count"]
                if k > int(q.sum()):
                    write_json(output / "final" / name / family / "unavailable.json",
                               {"reason": "frozen K exceeds Train-QC candidate count; no Test-driven modification", "k": k})
                    continue
                bands = control_bands(np.flatnonzero(q), wave, k, 4242, "uniform_bands")
                folder = output / "final" / name / family
                guarded(folder / "selection.json", {"selected_positions": bands,
                    "original_band_indices": data["original_band_indices"][bands], "wavelengths_nm": wave[bands]})
                for seed in seeds:
                    path_summary = folder / f"seed_{seed}.json"
                    if path_summary.exists():
                        info = read(path_summary)
                    else:
                        pred, diagnostic = fit_predictions(bands, fit, te, family, selected_config["overlap_c"],
                            seed + search["final_fit_seed_offset"], output / "final_models" / name)
                        prediction = pred[:, 0] if family == "global" else np.sum(pred[:, 1:] *
                            soft_region_weights(pred[:, 0], boundaries, selected_config["routing_width_c"]), axis=1)
                        info = {"protocol": name, "family": family, "seed": seed,
                            "frozen_key": selected_config["key"], "success": all(d["success"] for d in diagnostic),
                            "metrics": regression_metrics(y[te], prediction), "fit_diagnostics": diagnostic,
                            "evaluated_epoch": time.time(), "frozen_sha256": digest(output / "frozen_parameters.json")}
                        write_json(path_summary, info)
                        with (folder / f"seed_{seed}.csv").open("w", encoding="utf-8", newline="") as handle:
                            writer = csv.writer(handle); writer.writerow(["sample_index", "temperature_c", "prediction_c"])
                            writer.writerows(zip(te.tolist(), y[te].tolist(), prediction.tolist()))
                    final.append(info)
        write_json(output / "final_summary.json", {"individual": final, "best_family_from_validation": frozen["best_family"],
            "test_used_to_choose_family": False})
        state["status"], state["test_evaluated"] = "complete", True
        update("complete")
    except BaseException:
        state["status"], state["error"] = "failed", traceback.format_exc()
        update("failed")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "queue.lock").open("a+b") as lock:
        if os.name == "posix":
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
