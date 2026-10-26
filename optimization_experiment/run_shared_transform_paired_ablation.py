"""将共享特征变换作为独立消融，与各运行种子的选带结果配对。"""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_experiment/src"))
sys.path.insert(0, str(ROOT / "optimization_experiment"))
runtime = ROOT / "optimization_experiment/outputs/native_regressor_control_20261023/runtime/optimization_experiment"
if runtime.exists():
    sys.path.insert(0, str(runtime))
import numpy as np
import torch
from traditional_utils import load_data, model_parameters, write_json
from shared_feature_regression import SharedLogHelmert
from run_selector_regressor_control import train_models
from regional_colorimetry import soft_region_weights


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, prediction):
    err = prediction - y
    return {"rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
            "r2": float(1 - np.sum(err**2) / np.sum((y - np.mean(y))**2)),
            "mard_percent": float(100 * np.mean(np.abs(err) / np.abs(y)))}


def main(args):
    project, out = args.project_root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_root = args.ablation_root.resolve() if args.ablation_root else project / "optimization_experiment/outputs"
    if not args.ablation_root and not (source_root / "component_ablation_refrozen_k12_random_20261013/protocol.json").exists():
        source_root = project / "artifacts/manuscript_extrapolation_repeat_20261023/ablation"
    def source_dir(split_name):
        remote = source_root / f"component_ablation_refrozen_k12_{split_name}_20261013"
        return remote if remote.exists() else source_root / split_name
    config = read(source_dir("random") / "protocol.json")["config"]
    source = project / config["data_path"]
    if sha(source) != "81e3a456a1e57bc65d6b7a44e71c21aa4d2dee271927f2173c741f740a3ae2dc":
        raise ValueError("数据指纹不一致")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = load_data(source, config["feature_key"], config["uncertainty_key"])
    X, y, wavelengths = data[config["feature_key"]], data["y_temperature_c"], data["wavelengths_nm"]
    parameters = model_parameters(config)
    boundaries = np.asarray(config["temperature_boundaries_c"])
    seeds = list(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("重复种子必须非空且互不重复")
    protocol = {"purpose": "shared-transform single-factor ablation paired to each seed's original bands",
                "data_sha256": sha(source), "seeds": seeds,
                "source_aggregates_sha256": {p: sha(source_dir(p) / "aggregate.json")
                    for p in ("random", "ordered")},
                "selection": "each seed's archived baseline selected_positions; no new selection",
                "model": "original 55C-overlap 15C-routing regional MLP, shared log-Helmert transform",
                "test_boundary": "previously viewed same-batch development holdouts; no Test-based choice"}
    existing = out / "protocol.json"
    if existing.exists() and read(existing) != protocol:
        raise ValueError("协议变动，须使用新目录")
    write_json(existing, protocol)
    state = {"status": "running", "pid": os.getpid(), "completed": [], "current": None}
    write_json(out / "queue_state.json", state)
    try:
        for split_name in ("random", "ordered"):
            archive = source_dir(split_name)
            split = read(archive / "split_manifest.json")
            train, val, test = [np.asarray(split[k], dtype=int) for k in
                                ("train_indices", "validation_indices", "test_indices")]
            fit = np.sort(np.r_[train, val])
            assert len(fit) == 340 and len(test) == 61 and not np.intersect1d(fit, test).size
            for seed in protocol["seeds"]:
                label = f"{split_name}/seed_{seed}"
                state["current"] = label
                write_json(out / "queue_state.json", state)
                baseline_path = archive / f"seed_{seed}/baseline/summary.json"
                baseline = read(baseline_path)
                assert baseline["selection_success"] and baseline["test"]["converged"]
                selected = np.asarray(baseline["selection"]["selected_positions"], dtype=int)
                assert len(selected) == 12 and len(set(selected)) == 12
                job = out / split_name / f"seed_{seed}"
                summary = job / "summary.json"
                if summary.exists():
                    record = read(summary)
                else:
                    transform = SharedLogHelmert().fit(
                        X[fit][:, selected], wavelengths[selected],
                        fraction=config["model"]["intensity_floor_fraction"],
                        absolute=config["model"]["absolute_intensity_floor"])
                    zfit = transform.transform(X[fit][:, selected])
                    ztest = transform.transform(X[test][:, selected])
                    def update(stage):
                        state["stage"] = stage
                        write_json(out / "queue_state.json", state)
                    models, diagnostics = train_models(zfit, y[fit], boundaries,
                                                       parameters, seed + 100000, job, update)
                    converged = all(d["success"] for d in diagnostics)
                    record = {"seed": seed, "split": split_name, "converged": converged,
                              "source_baseline_sha256": sha(baseline_path),
                              "selected_positions": selected.tolist(),
                              "n_fit": len(fit), "n_test": len(test),
                              "training_diagnostics": diagnostics, "test_evaluated": False}
                    if converged:
                        global_pred = models[0].predict(ztest)
                        experts = np.column_stack([m.predict(ztest) for m in models[1:]])
                        weights = soft_region_weights(global_pred, boundaries, 15.0)
                        prediction = np.sum(experts * weights, axis=1)
                        job.mkdir(parents=True, exist_ok=True)
                        with (job / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
                            writer = csv.writer(handle)
                            writer.writerow(["sample_index", "temperature_c", "regional_c"])
                            writer.writerows(zip(test, y[test], prediction))
                        record["metrics"] = metrics(y[test], prediction)
                        record["test_evaluated"] = True
                    write_json(summary, record)
                state["completed"].append(label)
                state["current"] = None
                write_json(out / "queue_state.json", state)
                print("DONE", label, record.get("metrics", {}).get("rmse"), record["converged"], flush=True)
        state["status"] = "complete"
        write_json(out / "queue_state.json", state)
    except BaseException:
        state["status"], state["error"] = "failed", traceback.format_exc()
        write_json(out / "queue_state.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[4242, 4243, 4244])
    main(parser.parse_args())
