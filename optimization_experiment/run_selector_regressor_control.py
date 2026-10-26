"""固定K12波段与共享特征，交叉评价OLS、全局MLP和区域MLP。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

RUNTIME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME / "main_experiment/src"))

import numpy as np
import torch
from scipy.linalg import lstsq

from traditional_utils import (load_data, fold_quality_mask, make_configured_split,
                               model_parameters, regression_metrics, write_json)
from regional_colorimetry import regional_training_masks, soft_region_weights
from run_component_ablations import control_bands, audit_split
from shared_feature_regression import SharedLogHelmert, SharedFeatureMLP


BASELINES = ["03_BS_Nets", "05_DRL4BS", "06_HCBH", "07_SGAE", "09_XGBS_Reg", "10_Dropout_CAE"]
METHODS = ["ours", *BASELINES, "uniform", "random"]
DATA_HASH = "81e3a456a1e57bc65d6b7a44e71c21aa4d2dee271927f2173c741f740a3ae2dc"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def guarded(path, payload):
    from traditional_utils import json_ready
    value = json_ready(payload)
    if path.exists() and read(path) != value:
        raise ValueError(f"已有文件与冻结协议不一致，请使用新目录：{path}")
    write_json(path, value)


def fit_ols(features, target):
    design = np.column_stack([np.ones(len(features)), features])
    coef, _, rank, _ = lstsq(design, target, lapack_driver="gelsy")
    # 秩不足时保留最小范数OLS及秩诊断，不根据Test切换为岭回归。
    return coef, {"rank": int(rank), "columns": design.shape[1],
                  "full_rank": rank == design.shape[1],
                  "condition_number": float(np.linalg.cond(design))}


def ols_predict(features, coef):
    return np.column_stack([np.ones(len(features)), features]) @ coef


def select_baseline(method, X, y, train, validation, available, config, directory):
    """各方法使用共享Train质控池，保留原选带算法与既定训练预算。"""
    sys.path.insert(0, str(RUNTIME / "compare" / method))
    settings = config["selector"]
    values = X[train][:, available]
    extra = {}
    started = time.time()
    if method == "03_BS_Nets":
        from bsnets import BSNetFCSelector
        selector = BSNetFCSelector(device="cuda", random_state=settings["random_state"],
                                  profile=settings["profile"]).fit(values)
        local = selector.select(12)
        selector.save(directory / "selector_artifacts")
    elif method == "05_DRL4BS":
        from drl4bs import DRL4BSSelector
        domain = np.all(values >= 0, axis=0)
        extra["nonnegative_domain_excluded_positions"] = available[~domain]
        available = available[domain]
        selector = DRL4BSSelector(device="cuda", random_state=settings["random_state"],
            profile=settings["profile"], horizon=12, episodes=settings["episodes"]).fit(X[train][:, available])
        local = selector.select(12)
        selector.save(directory / "selector_artifacts")
    elif method == "06_HCBH":
        from hcbh import HCBHSelector
        selector = HCBHSelector(device=settings["device"], dtype=settings["dtype"]).fit([values[:, None, :]])
        local = selector.select(12)
        selector.save(directory / "selector_artifacts")
    elif method == "07_SGAE":
        from sgae import SGAESelector
        selector = SGAESelector(device="cuda", random_state=settings["random_state"],
            profile=settings["profile"], learning_rate=settings["learning_rate"],
            sigma=settings["sigma"], stopping_threshold=settings["stopping_threshold"],
            max_epochs=settings["max_epochs"]).fit(values)
        local, diagnostics = selector.select(12, return_diagnostics=True)
        selector.save(12, directory / "selector_artifacts")
        extra["diagnostics"] = diagnostics
    elif method == "10_Dropout_CAE":
        from dropout_cae_reproduction.selector import DropoutCAESelector
        selector = DropoutCAESelector(12, device="cuda", random_state=settings["random_state"],
                                     profile=settings["profile"]).fit(values)
        local = selector.select()
        selector.save_diagnostics(directory / "selector_diagnostics.json")
    elif method == "09_XGBS_Reg":
        from xgbs import XGBSRegressionSelector
        selector = XGBSRegressionSelector(random_state=settings["random_state"]).fit(values, y[train])
        candidates, cache = [], {}
        # 保留原方法的Validation选参规则；选择过程完全不读取Test。
        for alpha in settings["alpha_grid"]:
            for beta in settings["beta_grid"]:
                bands = tuple(selector.select(12, alpha=alpha, beta=beta).selected_bands)
                if bands not in cache:
                    coef, _ = fit_ols(values[:, bands], y[train])
                    pred = ols_predict(X[validation][:, available][:, bands], coef)
                    cache[bands] = regression_metrics(y[validation], pred)
                candidates.append({"alpha": alpha, "beta": beta, "bands": bands, **cache[bands]})
        best = min(candidates, key=lambda row: (row["rmse"], row["mae"], row["alpha"], row["beta"]))
        local = best["bands"]
        extra["validation_choice"] = best
        write_json(directory / "validation_candidates.json", {"candidates": candidates})
    else:
        raise ValueError(method)
    return np.sort(available[np.asarray(local, dtype=int)]), {
        "selection_seconds": time.time() - started, "selector_config": settings,
        "selection_scope": "Train; XGBS alpha/beta chosen by Validation", **extra}


def archived_ours(project, protocol, data, split, available):
    base = project / f"optimization_experiment/outputs/component_ablation_refrozen_k12_{protocol}_20261013"
    archive_protocol = read(base / "protocol.json")
    if archive_protocol["data_pair"]["background_sha256"] != DATA_HASH:
        raise ValueError("历史选带数据指纹不符")
    saved_split = read(base / "split_manifest.json")
    for key, indices in zip(("train_indices", "validation_indices", "test_indices"), split):
        if saved_split[key] != indices.tolist():
            raise ValueError("历史选带划分不符")
    path = base / "seed_4242/baseline/selected_bands.json"
    record = read(path)
    if not record["selection_success"]:
        raise ValueError("历史选带未收敛")
    mapping = {int(original): i for i, original in enumerate(data["original_band_indices"])}
    selected = np.sort([mapping[int(i)] for i in record["original_band_indices"]])
    if not set(selected).issubset(set(available)):
        raise ValueError("历史波段不在相同Train质控池中")
    return selected, {"selection_seed": 4242, "reuse": "bands_only; all downstream models refitted",
        "source": str(path), "source_sha256": digest(path),
        "protocol_sha256": digest(base / "protocol.json"),
        "split_sha256": digest(base / "split_manifest.json")}


def train_models(Z, y, boundaries, parameters, seed, folder, update):
    models, diagnostics = [], []
    masks = [np.ones(len(y), dtype=bool), *regional_training_masks(y, boundaries, 55.0)]
    folder.mkdir(parents=True, exist_ok=True)
    for role, mask in enumerate(masks):
        name = "global" if role == 0 else f"expert_{role-1}"
        update(name)
        path = folder / f"{name}.pt"
        if int(mask.sum()) < 10:
            raise ValueError(f"{name}拟合样本不足")
        if path.exists():
            model = torch.load(path, weights_only=False)
        else:
            model = SharedFeatureMLP(seed=seed + role, **parameters).fit(Z[mask], y[mask])
            torch.save(model, path.with_suffix(".tmp.pt"))
            path.with_suffix(".tmp.pt").replace(path)
        write_json(folder / f"{name}.trace.json", {"trace": model.trace_})
        diagnostics.append({"role": name, "seed": seed + role, "n_fit": int(mask.sum()),
            "temperature_range_c": [float(y[mask].min()), float(y[mask].max())],
            "success": bool(model.success_), "iterations": model.iterations_,
            "training_rmse_c": model.training_rmse_c_,
            "training_adaptive_rmse_c": model.training_adaptive_rmse_c_})
        print(f"  {name}: iter={model.iterations_}, success={model.success_}", flush=True)
        models.append(model)
    write_json(folder / "fit_diagnostics.json", {"models": diagnostics})
    return models, diagnostics


def write_predictions(path, indices, y, predictions):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "temperature_c", *predictions])
        writer.writerows(zip(indices.tolist(), y.tolist(), *[v.tolist() for v in predictions.values()]))


def aggregate(output):
    rows = []
    for protocol in ("random", "ordered"):
        for method in METHODS:
            folder = output / protocol / method
            if not (folder / "ols.json").exists():
                continue
            records = [read(p) for p in sorted(folder.glob("seed_*/summary.json"))]
            entry = {"protocol": protocol, "method": method, "ols": read(folder / "ols.json"),
                     "individual": records}
            # 全部预设种子均纳入描述；未收敛数量单列，禁止仅保留好看的运行。
            for decoder in ("global", "regional"):
                values = [r[decoder]["metrics"]["rmse"] for r in records]
                entry[decoder] = {"n_completed": len(values),
                    "n_converged": sum(r[decoder]["converged"] for r in records),
                    "rmse_mean": float(np.mean(values)) if values else None,
                    "rmse_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
            rows.append(entry)
    write_json(output / "aggregate.json", {"rows": rows})
    return rows


def run(args):
    project, output = args.project_root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if output == project / "artifacts/results" or project / "artifacts/results" in output.parents:
        raise ValueError("禁止覆盖论文历史证据目录")
    config = read(RUNTIME / "optimization_experiment/configs/component_ablation_refrozen_k12_random_20261013.json")
    data_path = project / config["data_path"]
    if digest(data_path) != DATA_HASH:
        raise ValueError("输入数据指纹不符")
    configs = {m: read(RUNTIME / f"artifacts/results/background_refrozen_k12_seed42_20261013/configs/{m}.json") for m in BASELINES}
    protocol = {"version": 1, "data_sha256": DATA_HASH, "model_config": config,
        "baseline_configs": configs, "decoder_seeds": args.seeds,
        "methods": METHODS, "decoders": ["OLS", "global_MLP", "regional_MLP"],
        "band_count": 12, "feature_dimension": 11,
        "candidate_pool": "same Train-only QC; DRL4BS retains its nonnegative input-domain restriction",
        "feature_transform": "one log-radiance Helmert transform per frozen subset fitted on Train+Validation; same features for OLS, global and all experts",
        "selection": "each subset fixed across decoder seeds; ours archived seed4242 bands; baseline selector seed42; uniform deterministic; random seed4242",
        "fit_scope": "Train+Validation (340); selectors Train280 only except XGBS Validation alpha/beta choice",
        "downstream_seed_offset": 100000,
        "uncertainty": "sample SD across three decoder initializations, conditional on fixed selected subset and fixed split; not independent batches",
        "test_scope": "previously viewed same-batch development holdout; no Test-based tuning or retries",
        "failed_policy": "all planned runs reported; nonconvergence flagged, no Test-based exclusion",
        "native_cardinality_methods": "SPA, CARS, STG, Dual-sPLS remain in original pipeline table; not truncated into K12",
        "sources_sha256": {str(p.relative_to(RUNTIME)): digest(p) for p in sorted(RUNTIME.rglob("*.py"))}}
    guarded(output / "protocol.json", protocol)
    write_json(output / "environment.json", {"python": sys.version, "numpy": np.__version__,
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = load_data(data_path, config["feature_key"], config["uncertainty_key"])
    X, y, wavelengths = data[config["feature_key"]], data["y_temperature_c"], data["wavelengths_nm"]
    boundaries = np.asarray(config["temperature_boundaries_c"])
    parameters = model_parameters(config)
    state = {"pid": os.getpid(), "status": "running", "completed": [], "current": None}
    def update(stage):
        state["stage"], state["updated_epoch"] = stage, time.time()
        write_json(output / "queue_state.json", state)
    try:
        for name in ("random", "ordered"):
            split_config = read(RUNTIME / f"optimization_experiment/configs/component_ablation_refrozen_k12_{name}_20261013.json")["split"]
            train, validation, test = split = make_configured_split(y, boundaries, split_config)
            audit = audit_split(y, train, validation, test, name == "ordered")
            fit = np.sort(np.concatenate([train, validation]))
            guarded(output / name / "split_manifest.json", {"train_indices": train,
                "validation_indices": validation, "test_indices": test, "fit_indices": fit, **audit})
            quality, quality_diagnostic = fold_quality_mask(X[train], data[config["uncertainty_key"]][train],
                data["background_mad"][train], data["bad_pixel_fraction_by_sample_band"][train],
                data["saturation_fraction_by_sample_band"][train], wavelengths, config["fold_quality_control"])
            available = np.flatnonzero(quality)
            guarded(output / name / "quality.json", {"available_positions": available, "diagnostics": quality_diagnostic})
            for method in METHODS:
                state["current"] = f"{name}/{method}"
                folder = output / name / method
                folder.mkdir(parents=True, exist_ok=True)
                selection_path = folder / "selection.json"
                update("selecting")
                if selection_path.exists():
                    selected = np.asarray(read(selection_path)["selected_positions"], dtype=int)
                else:
                    if method == "ours":
                        selected, provenance = archived_ours(project, name, data, split, available)
                    elif method in ("uniform", "random"):
                        selected = control_bands(available, wavelengths, 12, 4242, f"{method}_bands")
                        provenance = {"selection_seed": 4242, "required_long_wave": True}
                    else:
                        selected, provenance = select_baseline(method, X, y, train, validation,
                                                              available, configs[method], folder)
                    if len(selected) != 12 or len(np.unique(selected)) != 12:
                        raise ValueError(f"波段预算不符：{name}/{method}")
                    write_json(selection_path, {"selected_positions": selected,
                        "original_band_indices": data["original_band_indices"][selected],
                        "wavelengths_nm": wavelengths[selected], "provenance": provenance})
                transformer = SharedLogHelmert().fit(X[fit][:, selected], wavelengths[selected],
                    fraction=config["model"]["intensity_floor_fraction"], absolute=config["model"]["absolute_intensity_floor"])
                Zfit, Ztest = transformer.transform(X[fit][:, selected]), transformer.transform(X[test][:, selected])
                guarded(folder / "shared_transform.json", {"parameters": transformer.snapshot(),
                    "fit_features_sha256": array_digest(Zfit), "test_features_sha256": array_digest(Ztest),
                    "fit_indices": fit, "test_not_used_to_fit": True})
                if not (folder / "ols.json").exists():
                    coef, diagnostic = fit_ols(Zfit, y[fit])
                    prediction = ols_predict(Ztest, coef)
                    write_json(folder / "ols.json", {"metrics": regression_metrics(y[test], prediction),
                        "diagnostic": diagnostic, "coefficients": coef, "n_fit": len(fit), "n_test": len(test)})
                    write_predictions(folder / "ols_predictions.csv", test, y[test], {"ols": prediction})
                for seed in args.seeds:
                    state["current"] = f"{name}/{method}/seed_{seed}"
                    job = folder / f"seed_{seed}"
                    if not (job / "summary.json").exists():
                        models, diagnostics = train_models(Zfit, y[fit], boundaries, parameters,
                                                          seed + 100000, job, update)
                        coarse = models[0].predict(Ztest)
                        experts = np.column_stack([model.predict(Ztest) for model in models[1:]])
                        weights = soft_region_weights(coarse, boundaries, 15.0)
                        regional = np.sum(experts * weights, axis=1)
                        write_predictions(job / "predictions.csv", test, y[test], {"global": coarse, "regional": regional})
                        write_json(job / "summary.json", {"seed": seed, "fit_seed": seed + 100000,
                            "n_fit": len(fit), "n_test": len(test),
                            "shared_fit_features_sha256": array_digest(Zfit),
                            "global": {"converged": bool(models[0].success_), "metrics": regression_metrics(y[test], coarse)},
                            "regional": {"converged": all(row["success"] for row in diagnostics), "metrics": regression_metrics(y[test], regional)}})
                        print(f"DONE {state['current']}", flush=True)
                    state["completed"].append(state["current"])
                    update("completed_cell")
                aggregate(output)
        state["status"] = "complete"
        update("complete")
        aggregate(output)
    except BaseException:
        state["status"], state["error"] = "failed", traceback.format_exc()
        update("failed")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[4242, 4243, 4244])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 同一输出目录只允许一个训练进程，检查点可在中断后安全续跑。
    with (args.output_dir / "queue.lock").open("a+b") as lock:
        if os.name == "posix":
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
