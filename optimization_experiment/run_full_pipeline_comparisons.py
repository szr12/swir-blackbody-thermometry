"""仅补齐表5六种选带器的完整重复，统一调用本文原生区域回归器。"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


METHODS = ["03_BS_Nets", "05_DRL4BS", "06_HCBH", "07_SGAE", "09_XGBS_Reg", "10_Dropout_CAE"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function_source(path, name):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name),
                    include_attributes=False)


def run(args):
    runtime = args.runtime.resolve()
    sys.path[:0] = [str(runtime / "optimization_experiment"), str(runtime / "main_experiment/src")]
    # XGBS冻结源码位于适配器子目录，仅补齐导入路径，不改动选带算法。
    sys.path.insert(0, str(runtime / "compare/09_XGBS_Reg/adapter_source"))
    import numpy as np
    import torch
    from run_component_ablations import audit_split, fit_ensemble
    from run_selector_regressor_control import select_baseline
    from traditional_utils import fold_quality_mask, load_data, regression_metrics, write_json

    project, output = args.project_root.resolve(), args.output_dir.resolve()
    native = project / "optimization_experiment/outputs/native_regressor_control_20261023/results"
    shared = project / "optimization_experiment/outputs/selector_regressor_control_20261023/results"
    extension = project / "optimization_experiment/outputs/paper_five_repeats_20261024"
    old_runtime = project / "optimization_experiment/outputs/component_ablation_refrozen_k12_bundle_20261013/runtime"
    config = read(native / "protocol.json")["model_config"]
    previous = read(shared / "protocol.json")
    sources = {}
    for name in ["convergent_colorimetry.py", "regional_colorimetry.py", "traditional_utils.py"]:
        current, archived = runtime / "optimization_experiment" / name, old_runtime / name
        assert digest(current) == digest(archived), f"完整流程模型源码不一致：{name}"
        sources[name] = digest(current)
    assert function_source(runtime / "optimization_experiment/run_component_ablations.py", "fit_ensemble") == function_source(
        old_runtime / "run_component_ablations.py", "fit_ensemble"), "原生区域拟合实现不一致"
    sources["fit_ensemble_ast_sha256"] = hashlib.sha256(function_source(
        old_runtime / "run_component_ablations.py", "fit_ensemble").encode()).hexdigest()
    for method in METHODS:
        for source in sorted((runtime / "compare" / method).rglob("*.py")):
            key = source.relative_to(runtime).as_posix()
            assert digest(source) == previous["sources_sha256"][key], f"选带实现发生变化：{key}"
            sources[key] = digest(source)
    data_path = project / config["data_path"]
    assert digest(data_path) == previous["data_sha256"]
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=True)
    assert output != native and output != shared and output != extension
    plan = {"purpose": "table5 full-pipeline repeats only", "methods": METHODS,
            "selector_seeds": list(range(42, 47)), "model_seeds": list(range(4242, 4247)),
            "selector_seed_to_model_seed_offset": 4200, "final_fit_seed_offset": 100000,
            "model_config": config, "data_sha256": digest(data_path), "sources_sha256": sources,
            "baseline_configs": previous["baseline_configs"],
            "regressor": "exact complete-pipeline fit_ensemble; expert-specific transforms; same attention/stopping/overlap/routing",
            "selection": "reselect per stochastic seed; deterministic HCBH unchanged; seed42 selections reused with source evidence",
            "reuse": "reuse prior native predictions only for identical subset, split, model configuration and fitting seed",
            "no_additional_experiments": True,
            "failure_policy": "retain failed runs; no seed replacement, no Test-based tuning",
            "scope": "previously viewed same-batch development holdouts"}
    if (output / "protocol.json").exists():
        assert read(output / "protocol.json") == plan
    write_json(output / "protocol.json", plan)
    write_json(output / "environment.json", {"python": sys.version, "torch": torch.__version__,
               "numpy": np.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0), "tf32": False})
    data = load_data(data_path, config["feature_key"], config["uncertainty_key"])
    X, y, wave = data[config["feature_key"]], data["y_temperature_c"], data["wavelengths_nm"]
    state = {"pid": os.getpid(), "status": "running", "completed": [], "current": None}

    def update(stage):
        state["stage"], state["updated_epoch"] = stage, time.time()
        write_json(output / "queue_state.json", state)

    try:
        for split_name in ["random", "ordered"]:
            split = read(native / split_name / "split_manifest.json")
            train, val, test, fit = [np.asarray(split[k], dtype=int) for k in
                                    ["train_indices", "validation_indices", "test_indices", "fit_indices"]]
            audit_split(y, train, val, test, split_name == "ordered")
            np.testing.assert_array_equal(fit, np.sort(np.r_[train, val]))
            write_json(output / split_name / "split_manifest.json", split)
            quality, diagnostics = fold_quality_mask(X[train], data[config["uncertainty_key"]][train],
                data["background_mad"][train], data["bad_pixel_fraction_by_sample_band"][train],
                data["saturation_fraction_by_sample_band"][train], wave, config["fold_quality_control"])
            available = np.flatnonzero(quality)
            np.testing.assert_array_equal(available, read(shared / split_name / "quality.json")["available_positions"])
            write_json(output / split_name / "quality.json", {"available_positions": available, "diagnostics": diagnostics})
            for method in args.methods:
                old_selection = read(native / split_name / method / "selection.json")
                for selector_seed in range(42, 47):
                    model_seed = selector_seed + 4200
                    key = f"{split_name}/{method}/seed_{model_seed}"
                    job = output / key
                    state["current"] = key
                    update("starting")
                    if (job / "summary.json").exists():
                        state["completed"].append(key)
                        continue
                    job.mkdir(parents=True, exist_ok=True)
                    selection_path = job / "selection.json"
                    if not selection_path.exists():
                        if selector_seed == 42:
                            selected = np.asarray(old_selection["selected_positions"], dtype=int)
                            provenance = {"reuse": "original seed42 selector", "source": str(native / split_name / method / "selection.json"),
                                          "source_sha256": digest(native / split_name / method / "selection.json")}
                        else:
                            settings = copy.deepcopy(previous["baseline_configs"][method])
                            if "random_state" in settings["selector"]:
                                settings["selector"]["random_state"] = selector_seed
                            update("selecting")
                            selected, provenance = select_baseline(method, X, y, train, val, available, settings, job)
                        assert len(selected) == len(set(selected)) == 12
                        assert set(selected).issubset(set(available))
                        write_json(selection_path, {"selector_seed": selector_seed, "selected_positions": selected,
                            "original_band_indices": data["original_band_indices"][selected], "wavelengths_nm": wave[selected],
                            "provenance": provenance})
                    selected = np.asarray(read(selection_path)["selected_positions"], dtype=int)
                    reference_root = native if model_seed <= 4244 else extension / "native"
                    reference_job = reference_root / split_name / method / f"seed_{model_seed}"
                    reuse = np.array_equal(selected, old_selection["selected_positions"])
                    if reuse:
                        old = read(reference_job / "summary.json")
                        assert old["fit_seed"] == model_seed + 100000
                        success = old["native"]["converged"]
                        with (reference_job / "predictions.csv").open(encoding="utf-8") as handle:
                            rows = list(csv.DictReader(handle))
                        np.testing.assert_array_equal([int(float(v["sample_index"])) for v in rows], test)
                        np.testing.assert_allclose([float(v["temperature_c"]) for v in rows], y[test], rtol=0, atol=0)
                        prediction = np.asarray([float(v["regional"]) for v in rows])
                        evidence = {"reuse": True, "source": str(reference_job),
                                    "prediction_sha256": digest(reference_job / "predictions.csv"),
                                    "summary_sha256": digest(reference_job / "summary.json")}
                    else:
                        update("fitting_exact_native_ensemble")
                        ensemble, fit_diagnostics = fit_ensemble(X[fit][:, selected], y[fit], wave[selected], config,
                            model_seed + 100000, job / "native_fit", update)
                        success = all(d["success"] for d in fit_diagnostics)
                        evidence = {"reuse": False, "training_diagnostics": fit_diagnostics}
                        if success:
                            _, experts, weights = ensemble.predict_components(X[test][:, selected])
                            prediction = (experts * weights).sum(axis=1)
                    record = {"seed": model_seed, "selector_seed": selector_seed, "fit_seed": model_seed + 100000,
                              "converged": success, "test_evaluated": success, "k": len(selected), **evidence}
                    if success:
                        record["metrics"] = regression_metrics(y[test], prediction)
                        record["metrics"]["mard_percent"] = float(100 * np.mean(np.abs(prediction - y[test]) / np.abs(y[test])))
                        with (job / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
                            writer = csv.writer(handle)
                            writer.writerow(["sample_index", "temperature_c", "prediction_c"])
                            writer.writerows(zip(test, y[test], prediction))
                    write_json(job / "summary.json", record)
                    state["completed"].append(key)
                    print("DONE", key, "reuse", reuse, "converged", success,
                          record.get("metrics", {}).get("rmse"), flush=True)
                    update("completed_cell")
        state["status"], state["current"] = "complete", None
        update("complete")
    except BaseException:
        state["status"], state["error"] = "failed", traceback.format_exc()
        update("failed")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    with (arguments.output_dir / "queue.lock").open("a+b") as lock:
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(arguments)
