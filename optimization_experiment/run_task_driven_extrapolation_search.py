"""任务驱动选带与原生区域回归：按温度顺序进行可续跑超参搜索。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from regional_colorimetry import regional_training_masks, soft_region_weights
from traditional_utils import (load_data, fold_quality_mask, make_configured_split,
    make_selector, model_parameters, regression_metrics, write_json, json_ready)
from run_component_ablations import audit_split
from run_phase7_traditional_hyperparameter_search import _prepare_mi_cache

DATA_HASH = "81e3a456a1e57bc65d6b7a44e71c21aa4d2dee271927f2173c741f740a3ae2dc"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guarded(path, value):
    """缓存恢复必须保持科学内容和源文件指纹完全一致。"""
    if path.exists() and read(path) != json_ready(value):
        raise ValueError(f"冻结内容不匹配：{path}")
    write_json(path, value)


def rank_candidates(rows):
    """外层选优只读取等权Validation；所有固定种子均须收敛。"""
    valid = [r for r in rows if r["success"] and r["validation_rmse_mean"] is not None
             and np.isfinite(r["validation_rmse_mean"])]
    return sorted(valid, key=lambda r: (r["validation_rmse_mean"], r["band_count"],
        r["validation_rmse_sd"], r["redundancy_weight"], r["overlap_c"], r["routing_width_c"]))


def validate_expert_support(y, boundaries, overlaps, minimum):
    """先验证低温Train的专家支持量，禁止借用Validation补齐专家。"""
    counts = {str(float(o)): [int(m.sum()) for m in regional_training_masks(y, boundaries, o)]
              for o in overlaps}
    if any(min(values) < minimum for values in counts.values()):
        raise ValueError(f"搜索含训练样本不足的专家：{counts}")
    return counts


def run(args):
    output, project = args.output_dir.resolve(), args.project_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if project / "artifacts/results" in output.parents:
        raise ValueError("不能覆盖历史论文证据")
    search = read(ROOT / "optimization_experiment/configs/task_driven_extrapolation_search_20261023.json")
    base = read(ROOT / "optimization_experiment/configs/component_ablation_refrozen_k12_ordered_20261013.json")
    base.pop("frozen_search_provenance", None)
    base["evaluate_test_after_search"] = False
    data_path = project / base["data_path"]
    if digest(data_path) != DATA_HASH:
        raise ValueError("输入数据指纹不符")
    guarded(output / "protocol.json", {"search": search, "model_config": base,
        "data_sha256": DATA_HASH,
        "sources_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted(ROOT.rglob("*.py"))},
        "split": "Train50-329C280; Validation330-389C60; final refit50-389C340; Test390-450C61",
        "ranking": "mean of three fixed seeds' equally weighted extrapolation Validation RMSE; all selectors/global/experts must converge",
        "stages": ["coarse K at inherited rho/overlap/routing",
                   "rho grid for two best coarse K",
                   "overlap/routing grid for two best K/rho pairs",
                   "integer K refinement +/-2 at best remaining parameters"],
        "training": "original adaptive training, stopping, and adaptive Validation selector-checkpoint rule unchanged; only outer candidate ranking is unweighted",
        "features": "each native expert estimates floors/offsets/Helmert mean/scale on its own Train subset",
        "replication": "selection and decoder both repeated for each seed; final decoder seed=seed+100000; expert seeds=decoder+1..4",
        "final_evaluation": "freeze winner plus inherited K12/rho0.04/overlap55/routing15 control before Test; only refit successful validation candidates; no Test retries",
        "scope": "same-batch development extrapolation; these high-temperature samples were viewed in earlier experiments; not independent external validation",
        "failure_policy": "all failures retained; nonconverged selector skips decoder; nonconverged final fit skips Test; no seed replacement"})
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    write_json(output / "environment.json", {"python": sys.version, "numpy": np.__version__,
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
    data = load_data(data_path, base["feature_key"], base["uncertainty_key"])
    X, y, wave = data[base["feature_key"]], data["y_temperature_c"], data["wavelengths_nm"]
    boundaries = np.asarray(base["temperature_boundaries_c"])
    train, validation, test = make_configured_split(y, boundaries, base["split"])
    audit_split(y, train, validation, test, True)
    guarded(output / "split_manifest.json", {"train_indices": train, "validation_indices": validation,
        "test_indices": test, "final_fit_indices": np.sort(np.concatenate([train, validation]))})
    support = validate_expert_support(y[train], boundaries, search["overlap_values_c"], search["minimum_expert_training_samples"])
    quality, qc = fold_quality_mask(X[train], data[base["uncertainty_key"]][train], data["background_mad"][train],
        data["bad_pixel_fraction_by_sample_band"][train], data["saturation_fraction_by_sample_band"][train],
        wave, base["fold_quality_control"])
    available = np.flatnonzero(quality)
    guarded(output / "quality.json", {"available_positions": available, "diagnostics": qc,
        "validation_expert_training_counts": support})
    params, seeds = model_parameters(base), search["seeds"]
    state = {"pid": os.getpid(), "status": "running", "stage": "mutual_information", "current": None,
        "test_evaluated": False, "candidates_completed": 0, "models_completed_this_process": 0,
        "selectors_completed_this_process": 0}
    def update(current=None):
        state["current"], state["updated_epoch"] = current, time.time()
        write_json(output / "queue_state.json", state)
    update("Train-only MI")
    mi = _prepare_mi_cache(base, project, output, X[train][:, available], 3, 42)

    def select(k, rho, seed):
        folder = output / "selections" / f"k{k}_rho{rho:g}" / f"seed_{seed}"
        path = folder / "selection.json"
        if path.exists():
            return read(path)
        update(str(folder.relative_to(output)))
        started = time.time()
        try:
            selector = make_selector(base["model"], k, rho, seed, base["adaptive_weighting"])
            result = selector.fit(X[train][:, available], y[train], X[validation][:, available],
                                  y[validation], wave[available], mi)
            info = asdict(result)
            trace = info.pop("trace")
            positions = available[result.selected_indices]
            info.update({"seed": seed, "band_count": k, "redundancy_weight": rho,
                "selected_positions": positions, "original_band_indices": data["original_band_indices"][positions],
                "wavelengths_nm": wave[positions], "train_indices": train, "validation_indices": validation,
                "started_epoch": started, "completed_epoch": time.time(), "test_evaluated": False})
            write_json(folder / "trace.json", {"trace": trace})
        except Exception:
            info = {"seed": seed, "band_count": k, "redundancy_weight": rho, "success": False,
                    "error": traceback.format_exc(), "started_epoch": started, "completed_epoch": time.time()}
        write_json(path, info)
        state["selectors_completed_this_process"] += 1
        print("SELECTION", k, rho, seed, info["success"], info.get("iterations"), flush=True)
        return read(path)

    def fit_components(selected, fit, evaluation, overlap, seed, cache_root, permit_prediction=True):
        # 同一组有序波段及同一初始化可复用；不按误差或方法名称决定缓存。
        band_hash = hashlib.sha256(np.asarray(selected, dtype=np.int64).tobytes()).hexdigest()
        root = cache_root / f"bands_{band_hash[:20]}" / f"seed_{seed}"
        guarded(root / "input.json", {"band_hash": band_hash, "selected_positions": selected,
            "fit_indices": fit, "evaluation_indices": evaluation, "seed": seed})
        masks = [np.ones(len(fit), dtype=bool), *regional_training_masks(y[fit], boundaries, overlap)]
        predictions, diagnostics, models = [], [], []
        for role, mask in enumerate(masks):
            folder = root / ("global" if role == 0 else f"overlap_{overlap:g}/expert_{role-1}")
            folder.mkdir(parents=True, exist_ok=True)
            model_path, info_path = folder / "model.pt", folder / "diagnostic.json"
            pred_path = folder / "prediction.npy"
            update(str(folder.relative_to(output)))
            model = None
            if info_path.exists():
                info = read(info_path)
            else:
                if int(mask.sum()) < search["minimum_expert_training_samples"]:
                    raise ValueError("专家Train样本不足")
                if model_path.exists():
                    model = torch.load(model_path, weights_only=False)
                else:
                    model = FixedBandColorimetricMLP(seed=seed + role, **params)
                    model.fit(X[fit][:, selected][mask], y[fit][mask], wave[selected])
                    torch.save(model, folder / "model.tmp.pt")
                    (folder / "model.tmp.pt").replace(model_path)
                info = {"role": "global" if role == 0 else f"expert_{role-1}", "seed": seed + role,
                    "selected_positions": selected, "fit_indices": fit[mask], "evaluation_indices": evaluation,
                    "overlap_c": overlap if role else None, "n_fit": int(mask.sum()),
                    "temperature_range_c": [float(y[fit][mask].min()), float(y[fit][mask].max())],
                    "success": bool(model.success_), "iterations": model.iterations_,
                    "training_rmse_c": model.training_rmse_c_, "training_adaptive_rmse_c": model.training_adaptive_rmse_c_,
                    "transformation": {"floors": model.floors_, "offsets": model.offsets_,
                        "feature_mean": model.feature_mean_, "feature_scale": model.feature_scale_},
                    "completed_epoch": time.time()}
                write_json(info_path, info)
                write_json(folder / "trace.json", {"trace": model.trace_})
                state["models_completed_this_process"] += 1
            diagnostics.append(info); models.append((folder, model))
        success = all(d["success"] for d in diagnostics)
        # 最终模型先检查全部收敛，再读取Test特征作预测。
        if permit_prediction and (success or cache_root.name == "validation_models"):
            for folder, model in models:
                path = folder / "prediction.npy"
                if path.exists():
                    prediction = np.load(path)
                else:
                    model = model or torch.load(folder / "model.pt", weights_only=False)
                    prediction = model.predict(X[evaluation][:, selected])
                    if not np.isfinite(prediction).all():
                        raise ValueError("预测包含非有限值")
                    np.save(path, prediction)
                predictions.append(prediction)
        return np.column_stack(predictions) if predictions else None, diagnostics, str(root.relative_to(output))

    rows = []
    def candidate(k, rho, overlap, routing, stage):
        key = f"k{k}_rho{rho:g}_overlap{overlap:g}_routing{routing:g}"
        folder = output / "candidates" / key
        path = folder / "summary.json"
        if path.exists():
            row = read(path)
        else:
            individual = []
            for seed in seeds:
                selection = select(k, rho, seed)
                if not selection["success"]:
                    individual.append({"seed": seed, "success": False, "status": "selector_failed"})
                    continue
                try:
                    components, diagnostic, cache = fit_components(selection["selected_positions"], train, validation,
                        overlap, seed, output / "validation_models")
                    weights = soft_region_weights(components[:, 0], boundaries, routing)
                    prediction = (components[:, 1:] * weights).sum(axis=1)
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savez(folder / f"seed_{seed}.npz", sample_indices=validation, temperature_c=y[validation],
                             prediction_c=prediction, components=components, weights=weights)
                    individual.append({"seed": seed, "success": all(d["success"] for d in diagnostic),
                        "metrics": regression_metrics(y[validation], prediction), "model_cache": cache,
                        "model_success": [d["success"] for d in diagnostic]})
                except Exception:
                    individual.append({"seed": seed, "success": False, "error": traceback.format_exc()})
            values = [r["metrics"]["rmse"] for r in individual if "metrics" in r]
            row = {"key": key, "band_count": k, "redundancy_weight": rho, "overlap_c": overlap,
                "routing_width_c": routing, "first_stage": stage, "individual": individual,
                "success": len(values) == len(seeds) and all(r["success"] for r in individual),
                "validation_rmse_mean": float(np.mean(values)) if len(values) == len(seeds) else None,
                "validation_rmse_sd": float(np.std(values, ddof=1)) if len(values) == len(seeds) else None,
                "test_evaluated": False, "completed_epoch": time.time()}
            write_json(path, row)
            print("CANDIDATE", key, row["validation_rmse_mean"], row["success"], flush=True)
        if not any(r["key"] == key for r in rows):
            rows.append(row)
        state["candidates_completed"] = len(rows)
        write_json(output / "candidates.json", {"candidates": rows, "test_evaluated": False})
        return row

    try:
        initial = [search["initial_redundancy_weight"], search["initial_overlap_c"], search["initial_routing_width_c"]]
        state["stage"] = "coarse_k"
        coarse = [candidate(k, *initial, "coarse_k") for k in search["band_counts"]]
        top_k = [r["band_count"] for r in rank_candidates(coarse)[:search["redundancy_top_k_count"]]]
        if not top_k:
            raise ValueError("没有全部固定种子收敛的初筛候选")
        guarded(output / "stage2_k.json", {"band_counts": top_k})
        state["stage"] = "redundancy"
        for k in top_k:
            for rho in search["redundancy_weights"]:
                candidate(k, rho, initial[1], initial[2], "redundancy")
        top_pairs = [{"band_count": r["band_count"], "redundancy_weight": r["redundancy_weight"]}
                     for r in rank_candidates(rows)[:search["regional_top_pair_count"]]]
        guarded(output / "stage3_pairs.json", {"pairs": top_pairs})
        state["stage"] = "overlap_routing"
        for pair in top_pairs:
            for overlap in search["overlap_values_c"]:
                for routing in search["routing_width_values_c"]:
                    candidate(pair["band_count"], pair["redundancy_weight"], overlap, routing, "overlap_routing")
        center = {key: rank_candidates(rows)[0][key] for key in
                  ("band_count", "redundancy_weight", "overlap_c", "routing_width_c")}
        guarded(output / "refinement_center.json", center)
        state["stage"] = "local_k"
        for k in range(max(2, center["band_count"] - search["final_k_refinement_half_width"]),
                       min(len(available), center["band_count"] + search["final_k_refinement_half_width"]) + 1):
            candidate(k, center["redundancy_weight"], center["overlap_c"], center["routing_width_c"], "local_k")
        baseline = next(r for r in rows
            if all(r[key] == value for key, value in search["baseline"].items()))
        frozen = {"status": "frozen_from_extrapolation_validation_only", "test_evaluated": False,
            "winner": rank_candidates(rows)[0], "baseline": baseline, "freeze_epoch": time.time(),
            "candidate_count": len(rows), "candidate_evidence_sha256": digest(output / "candidates.json"),
            "selection_metric": search["selection_metric"]}
        frozen_path = output / "frozen_parameters.json"
        if frozen_path.exists():
            frozen["freeze_epoch"] = read(frozen_path)["freeze_epoch"]
        guarded(frozen_path, frozen)
        state["stage"] = "final_refit_and_test"
        update("frozen_before_test")
        final = []
        fit = np.sort(np.concatenate([train, validation]))
        for label in ["winner", "baseline"]:
            config = frozen[label]
            for seed in seeds:
                folder = output / "final" / label
                path = folder / f"seed_{seed}.json"
                if path.exists():
                    info = read(path)
                elif not config["success"]:
                    info = {"label": label, "seed": seed, "success": False, "status": "validation_pipeline_failed_no_test"}
                    write_json(path, info)
                else:
                    selection = select(config["band_count"], config["redundancy_weight"], seed)
                    guarded(folder / f"selection_seed_{seed}.json", selection)
                    components, diagnostic, cache = fit_components(selection["selected_positions"], fit, test,
                        config["overlap_c"], seed + search["final_fit_seed_offset"], output / "final_models")
                    info = {"label": label, "seed": seed, "frozen_key": config["key"],
                        "success": all(d["success"] for d in diagnostic), "model_cache": cache,
                        "model_success": [d["success"] for d in diagnostic], "evaluated_epoch": time.time(),
                        "frozen_sha256": digest(frozen_path), "test_evaluated": components is not None}
                    if components is not None:
                        weights = soft_region_weights(components[:, 0], boundaries, config["routing_width_c"])
                        prediction = (components[:, 1:] * weights).sum(axis=1)
                        info["metrics"] = regression_metrics(y[test], prediction)
                        info["global_metrics"] = regression_metrics(y[test], components[:, 0])
                        np.savez(folder / f"seed_{seed}.npz", sample_indices=test, temperature_c=y[test],
                                 prediction_c=prediction, components=components, weights=weights)
                    write_json(path, info)
                final.append(info)
        write_json(output / "final_summary.json", {"individual": final, "test_used_for_selection": False})
        state["status"], state["test_evaluated"] = "complete", any(r.get("test_evaluated", False) for r in final)
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
