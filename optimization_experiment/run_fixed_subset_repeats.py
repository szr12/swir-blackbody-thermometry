"""沿用论文固定波段子集，为原生区域MLP追加预先指定的初始化。"""

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

import numpy as np
import torch

from run_component_ablations import audit_split, fit_ensemble
from traditional_utils import load_data, regression_metrics, write_json


METHODS = ["09_XGBS_Reg", "03_BS_Nets", "05_DRL4BS", "06_HCBH",
           "07_SGAE", "10_Dropout_CAE", "ours"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guarded(path, value):
    from traditional_utils import json_ready
    value = json_ready(value)
    if path.exists() and read(path) != value:
        raise ValueError(f"冻结记录发生变化，不能复用：{path}")
    write_json(path, value)


def main(args):
    project = args.project_root.resolve()
    reference = args.reference_results.resolve()
    out = args.output_dir.resolve()
    if out == reference or reference in out.parents or project / "artifacts/results" in out.parents:
        raise ValueError("追加运行必须使用独立输出目录")
    out.mkdir(parents=True, exist_ok=True)
    old = read(reference / "protocol.json")
    config = old["model_config"]
    seeds = list(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("种子必须非空且互不重复")
    computational_sources = {}
    for name in ["convergent_colorimetry.py", "regional_colorimetry.py",
                 "traditional_utils.py", "run_component_ablations.py"]:
        source = Path(__file__).parent / name
        computational_sources[name] = sha(source)
        if computational_sources[name] != old["sources_sha256"]["optimization_experiment/" + name]:
            raise ValueError(f"计算实现与原生比较归档不一致：{name}")
    data_path = project / config["data_path"]
    if sha(data_path) != old["data_sha256"]:
        raise ValueError("数据指纹不一致")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device != config["model"]["device"]:
        raise ValueError("追加实验的计算设备必须与原协议一致")
    protocol = {"purpose": "additional fixed-subset native regional MLP repeats",
                "seeds": seeds, "methods": METHODS, "data_sha256": sha(data_path),
                "reference_protocol_sha256": sha(reference / "protocol.json"),
                "model_config": config, "sources_sha256": computational_sources,
                "subset_policy": "reuse each method's fixed archived subset; no selector refit",
                "scope": "previously viewed same-batch development holdouts; no Test-based tuning"}
    guarded(out / "protocol.json", protocol)
    write_json(out / "environment.json", {"python": sys.version, "torch": torch.__version__,
               "numpy": np.__version__, "cuda": torch.version.cuda,
               "device": args.device, "gpu": torch.cuda.get_device_name(0)})
    data = load_data(data_path, config["feature_key"], config["uncertainty_key"])
    X, y, wavelengths = data[config["feature_key"]], data["y_temperature_c"], data["wavelengths_nm"]
    state = {"status": "running", "completed": [], "current": None}

    def progress(stage):
        state["stage"], state["updated_epoch"] = stage, time.time()
        write_json(out / "queue_state.json", state)

    try:
        for split_name in ["random", "ordered"]:
            split = read(reference / split_name / "split_manifest.json")
            train, val, test, fit = [np.asarray(split[key], dtype=int) for key in
                                    ["train_indices", "validation_indices", "test_indices", "fit_indices"]]
            audit_split(y, train, val, test, split_name == "ordered")
            np.testing.assert_array_equal(fit, np.sort(np.r_[train, val]))
            guarded(out / split_name / "split_manifest.json", split)
            for method in METHODS:
                selection_path = reference / split_name / method / "selection.json"
                selection = read(selection_path)
                selected = np.asarray(selection["selected_positions"], dtype=int)
                assert len(selected) == len(set(selected)) == 12
                np.testing.assert_array_equal(data["original_band_indices"][selected],
                                              selection["original_band_indices"])
                guarded(out / split_name / method / "selection.json", selection)
                for seed in seeds:
                    key = f"{split_name}/{method}/seed_{seed}"
                    state["current"] = key
                    progress("starting")
                    job = out / key
                    if not (job / "summary.json").exists():
                        print("START", key, flush=True)
                        ensemble, diagnostics = fit_ensemble(
                            X[fit][:, selected], y[fit], wavelengths[selected],
                            config, seed + 100000, job / "native_fit", progress)
                        converged = all(d["success"] for d in diagnostics)
                        record = {"seed": seed, "fit_seed": seed + 100000,
                                  "n_fit": len(fit), "n_test": len(test),
                                  "selection_sha256": sha(selection_path),
                                  "native": {"converged": converged},
                                  "training_diagnostics": diagnostics,
                                  "test_evaluated": False}
                        if converged:
                            coarse, experts, weights = ensemble.predict_components(X[test][:, selected])
                            prediction = np.sum(experts * weights, axis=1)
                            assert np.isfinite(prediction).all()
                            job.mkdir(parents=True, exist_ok=True)
                            with (job / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
                                writer = csv.writer(handle)
                                writer.writerow(["sample_index", "temperature_c", "global", "regional"])
                                writer.writerows(zip(test, y[test], coarse, prediction))
                            record["native"]["metrics"] = regression_metrics(y[test], prediction)
                            record["global_metrics"] = regression_metrics(y[test], coarse)
                            record["test_evaluated"] = True
                        write_json(job / "summary.json", record)
                        print("DONE", key, record["native"], flush=True)
                    state["completed"].append(key)
                    progress("completed")
        state["status"] = "complete"
        progress("complete")
    except BaseException:
        state["status"], state["error"] = "failed", traceback.format_exc()
        progress("failed")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 与后续总队列共享锁，保证提前执行的阶段不会被重复训练。
    with (args.output_dir / "runner.lock").open("a+b") as lock:
        if os.name == "posix":
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        main(args)
