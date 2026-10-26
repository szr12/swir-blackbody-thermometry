"""在固定子集上恢复原生专家独立变换，与共享变换作配对控制。"""

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

from traditional_utils import load_data, write_json, regression_metrics
from run_component_ablations import fit_ensemble, audit_split
from run_selector_regressor_control import read, digest, guarded, write_predictions, METHODS, DATA_HASH


def read_predictions(path):
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}


def aggregate(output, shared):
    rows = []
    for protocol in ("random", "ordered"):
        for method in METHODS:
            records = [read(p) for p in sorted((output / protocol / method).glob("seed_*/summary.json"))]
            if not records:
                continue
            native = np.asarray([r["native"]["metrics"]["rmse"] for r in records])
            common = np.asarray([r["shared"]["metrics"]["rmse"] for r in records])
            delta = native - common
            rows.append({"protocol": protocol, "method": method, "completed": len(records),
                "converged": sum(r["native"]["converged"] for r in records),
                "native_rmse_mean": float(native.mean()),
                "native_rmse_sd": float(native.std(ddof=1)) if len(records) > 1 else None,
                "shared_rmse_mean": float(common.mean()),
                "shared_rmse_sd": float(common.std(ddof=1)) if len(records) > 1 else None,
                "paired_delta_mean": float(delta.mean()),
                "paired_delta_sd": float(delta.std(ddof=1)) if len(records) > 1 else None,
                "individual": records})
    write_json(output / "aggregate.json", {"rows": rows})


def run(args):
    project, shared, output = args.project_root.resolve(), args.shared_results.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if shared == output or shared in output.parents or project / "artifacts/results" in output.parents:
        raise ValueError("必须使用独立结果目录")
    prior = read(shared / "protocol.json")
    if prior["data_sha256"] != DATA_HASH or prior["decoder_seeds"] != [4242, 4243, 4244]:
        raise ValueError("前轮协议不符")
    config = prior["model_config"]
    source_hashes = {str(p.relative_to(RUNTIME)): digest(p) for p in sorted(RUNTIME.rglob("*.py"))}
    for name in ("convergent_colorimetry.py", "traditional_utils.py", "regional_colorimetry.py", "run_component_ablations.py"):
        key = "optimization_experiment/" + name
        if source_hashes[key] != prior["sources_sha256"][key]:
            raise ValueError(f"模型实现与前轮不同：{name}")
    data_path = project / config["data_path"]
    if digest(data_path) != DATA_HASH:
        raise ValueError("数据指纹不符")
    evidence_files = [shared / "protocol.json"]
    for protocol in ("random", "ordered"):
        evidence_files.append(shared / protocol / "split_manifest.json")
        for method in METHODS:
            folder = shared / protocol / method
            evidence_files.append(folder / "selection.json")
            for seed in prior["decoder_seeds"]:
                evidence_files.extend([folder / f"seed_{seed}/summary.json", folder / f"seed_{seed}/predictions.csv"])
    anchors = {name: project / f"optimization_experiment/outputs/component_ablation_refrozen_k12_{name}_20261013/seed_4242/baseline/test.predictions.csv"
               for name in ("random", "ordered")}
    protocol_record = {"version": 1, "design": "paired native-versus-shared expert feature transformation",
        "data_sha256": DATA_HASH, "model_config": config, "seeds": prior["decoder_seeds"], "methods": METHODS,
        "shared_results": str(shared),
        "shared_evidence_sha256": {str(p.relative_to(shared)): digest(p) for p in evidence_files},
        "anchor_predictions": {name: {"path": str(path), "sha256": digest(path)} for name, path in anchors.items()},
        "sources_sha256": source_hashes,
        "changed_factor": "each expert estimates floor, offset and feature mean/scale from its own fitting subset; common Helmert construction unchanged",
        "fixed_factors": "same nine K12 subsets, splits, 340 fitting samples, network, attention, budget, stopping, routing, overlap and paired seeds; no selector refit",
        "scope": "same-batch previously viewed development holdouts; no Test-based tuning or seed replacement",
        "uncertainty": "sample SD of three decoder initializations conditional on each fixed subset and split",
        "reuse": "no model or prediction reused for native fits; shared predictions are frozen paired comparator; OLS unchanged"}
    guarded(output / "protocol.json", protocol_record)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    write_json(output / "environment.json", {"python": sys.version, "torch": torch.__version__,
        "numpy": np.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "tf32": torch.backends.cuda.matmul.allow_tf32})
    data = load_data(data_path, config["feature_key"], config["uncertainty_key"])
    X, y, wavelengths = data[config["feature_key"]], data["y_temperature_c"], data["wavelengths_nm"]
    boundaries = np.asarray(config["temperature_boundaries_c"])
    state = {"pid": os.getpid(), "status": "running", "completed": [], "current": None}
    def update(stage):
        state["stage"], state["updated_epoch"] = stage, time.time()
        write_json(output / "queue_state.json", state)
    try:
        for name in ("random", "ordered"):
            split = read(shared / name / "split_manifest.json")
            train, validation, test, fit = [np.asarray(split[key], dtype=int) for key in
                ("train_indices", "validation_indices", "test_indices", "fit_indices")]
            audit_split(y, train, validation, test, name == "ordered")
            np.testing.assert_array_equal(fit, np.sort(np.concatenate([train, validation])))
            guarded(output / name / "split_manifest.json", split)
            for method in METHODS:
                selection = read(shared / name / method / "selection.json")
                selected = np.asarray(selection["selected_positions"], dtype=int)
                np.testing.assert_array_equal(data["original_band_indices"][selected], selection["original_band_indices"])
                assert len(selected) == len(set(selected)) == 12
                guarded(output / name / method / "selection.json", selection)
                for seed in prior["decoder_seeds"]:
                    key = f"{name}/{method}/seed_{seed}"
                    state["current"] = key
                    job = output / key
                    if not (job / "summary.json").exists():
                        print("START", key, flush=True)
                        ensemble, diagnostics = fit_ensemble(X[fit][:, selected], y[fit], wavelengths[selected],
                            config, seed + 100000, job / "native_fit", update)
                        coarse, experts, weights = ensemble.predict_components(X[test][:, selected])
                        regional = np.sum(experts * weights, axis=1)
                        paired = read_predictions(shared / key / "predictions.csv")
                        np.testing.assert_array_equal(paired["sample_index"], test)
                        np.testing.assert_array_equal(paired["temperature_c"], y[test])
                        max_global_difference = float(np.max(np.abs(coarse - paired["global"])))
                        if max_global_difference > 1e-6:
                            raise ValueError("全局预测发生变化，未保持单因素配对")
                        values = {"global": coarse, "regional": regional,
                            **{f"expert_{i}": experts[:, i] for i in range(4)},
                            **{f"weight_{i}": weights[:, i] for i in range(4)}}
                        job.mkdir(parents=True, exist_ok=True)
                        write_predictions(job / "predictions.csv", test, y[test], values)
                        native_metrics = regression_metrics(y[test], regional)
                        shared_metrics = regression_metrics(y[test], paired["regional"])
                        anchor = None
                        if method == "ours" and seed == 4242:
                            old = read_predictions(anchors[name])
                            np.testing.assert_array_equal(old["sample_index"], test)
                            difference = float(np.max(np.abs(old["regional_c"] - regional)))
                            anchor = {"max_prediction_difference_c": difference,
                                "archived_rmse": regression_metrics(y[test], old["regional_c"])["rmse"]}
                            if difference > 1e-6:
                                raise ValueError("本文seed4242未重现原生基准预测")
                        write_json(job / "summary.json", {"seed": seed, "fit_seed": seed + 100000,
                            "n_fit": len(fit), "n_test": len(test),
                            "native": {"converged": all(d["success"] for d in diagnostics), "metrics": native_metrics},
                            "shared": {"metrics": shared_metrics},
                            "global_metrics": regression_metrics(y[test], coarse),
                            "global_prediction_max_difference_c": max_global_difference,
                            "delta_native_minus_shared_rmse": native_metrics["rmse"] - shared_metrics["rmse"],
                            "anchor_reproduction": anchor})
                        print("DONE", key, "native_rmse", native_metrics["rmse"], flush=True)
                    state["completed"].append(key)
                    update("completed_cell")
                aggregate(output, shared)
        state["status"] = "complete"
        update("complete")
        aggregate(output, shared)
    except BaseException:
        state["status"], state["error"] = "failed", traceback.format_exc()
        update("failed")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--shared-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "queue.lock").open("a+b") as lock:
        if os.name == "posix":
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
