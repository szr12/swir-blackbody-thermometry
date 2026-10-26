"""固定论文划分，对四种有监督原方法预测流程进行指定种子重复。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


METHODS = ("01_SPA", "02_CARS", "04_STG", "08_Dual_sPLS")
PROTOCOLS = ("random", "ordered")
SEEDS = (42, 43, 44)
DATA_HASH = "81e3a456a1e57bc65d6b7a44e71c21aa4d2dee271927f2173c741f740a3ae2dc"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_method(root: Path, method: str):
    source = root / "compare" / method / "run_main_comparison.py"
    spec = importlib.util.spec_from_file_location(f"original_predictor_{method}", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, source


def main(args) -> None:
    root, out = args.project_root.resolve(), args.output_dir.resolve()
    if out == root or root in out.parents and "artifacts/results" in out.as_posix():
        raise ValueError("不得覆盖论文原始归档结果")
    out.mkdir(parents=True, exist_ok=True)
    archive = root / "compare/outputs/background_refrozen_k12_seed42_20261013"
    if not (archive / "configs/01_SPA.json").exists():
        archive = root / "artifacts/results/background_refrozen_k12_seed42_20261013"
    data = root / "preprocessing_validation/preprocessed_swir_background_corrected_v1/swir_roi_background_corrected.npz"
    if sha(data) != DATA_HASH:
        raise ValueError("数据指纹不一致")
    split_paths = {}
    for split_name in PROTOCOLS:
        path = (root / "optimization_experiment/outputs" /
                f"component_ablation_refrozen_k12_{split_name}_20261013/split_manifest.json")
        if not path.exists():
            path = (root / "artifacts/manuscript_extrapolation_repeat_20261023/ablation" /
                    split_name / "split_manifest.json")
        split_paths[split_name] = path
    split_data = {protocol: read(path) for protocol, path in split_paths.items()}
    source_hashes = {method: sha(root / "compare" / method / "run_main_comparison.py") for method in METHODS}
    seeds = tuple(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("重复种子必须非空且互不重复")
    protocol = {"purpose": "original-predictor comparisons on frozen random and ordered splits",
                "data_sha256": DATA_HASH, "split_sha256": {k: sha(v) for k, v in split_paths.items()},
                "source_sha256": source_hashes, "methods": list(METHODS), "protocols": list(PROTOCOLS),
                "selector_seeds": list(seeds), "split_seed": 42,
                "test_boundary": "previously viewed same-batch development holdouts; no Test-based choice",
                "repeat_policy": "SPA is deterministic; its repeated identical outputs are retained with zero SD"}
    p = out / "protocol.json"
    if p.exists() and read(p) != protocol:
        raise ValueError("已有协议变动，须使用新的输出目录")
    write(p, protocol)
    state = {"status": "running", "pid": os.getpid(), "completed": [], "current": None}
    write(out / "queue_state.json", state)
    try:
        for split_name in PROTOCOLS:
            manifest = split_data[split_name]
            train, val, test = [manifest[k] for k in ("train_indices", "validation_indices", "test_indices")]
            assert len(train) == 280 and len(val) == 60 and len(test) == 61
            for method in METHODS:
                module, _ = load_method(root, method)
                # 原方法训练与预测代码保持不变，仅注入已经冻结的论文划分。
                module.make_traditional_split = lambda y, boundaries, split_seed, a=train, b=val, c=test: (
                    module.np.asarray(a, dtype=int), module.np.asarray(b, dtype=int), module.np.asarray(c, dtype=int))
                if split_name == "ordered":
                    # 原方法的分温区报告要求每个区间均有测试样本；顺序外推仅有最高温区。
                    original_relative = module.mean_absolute_relative_deviation

                    def ordered_segments(y_true, prediction, boundaries, _module=module):
                        ids = _module.temperature_bin_ids(y_true, boundaries)
                        return [{"segment": region, "lower_c": float(boundaries[region]),
                                 "upper_c": float(boundaries[region + 1]),
                                 "upper_inclusive": region == len(boundaries) - 2,
                                 "n_samples": int((ids == region).sum()),
                                 **_module.regression_metrics(y_true[ids == region], prediction[ids == region])}
                                for region in range(len(boundaries) - 1) if (ids == region).any()]

                    module.segmented_metrics = ordered_segments
                    module.mean_absolute_relative_deviation = (
                        lambda y_true, prediction, _original=original_relative:
                        _original(y_true, prediction) if len(y_true) else None)
                original_config = read(archive / "configs" / f"{method}.json")
                for seed in seeds:
                    label = f"{split_name}/{method}/seed_{seed}"
                    state["current"] = label
                    write(out / "queue_state.json", state)
                    job = out / split_name / method / f"seed_{seed}"
                    summary = job / "summary.json"
                    if summary.exists():
                        result = read(summary)
                    else:
                        config = copy.deepcopy(original_config)
                        config.pop("comparison_reference", None)
                        config["data_path"] = str(data)
                        config["repeat_seed"] = seed
                        config["split_strategy"] = split_name
                        if "random_state" in config["selector"]:
                            config["selector"]["random_state"] = seed
                        started = time.time()
                        result = module.run_comparison(config, root, job, verify_reference=False)
                        result["repeat_protocol"] = split_name
                        result["repeat_seed"] = seed
                        result["source_sha256"] = source_hashes[method]
                        result["elapsed_wall_seconds"] = time.time() - started
                    if split_name == "ordered":
                        record = read(job / "split_manifest.json")
                        record["strategy"] = "temperature_ordered_70_15_15"
                        write(job / "split_manifest.json", record)
                    result["repeat_protocol"] = split_name
                    result["repeat_seed"] = seed
                    result["source_sha256"] = source_hashes[method]
                    result["split_manifest_sha256"] = sha(job / "split_manifest.json")
                    write(summary, result)
                    assert result["test_used_for_selection_or_stopping"] is False
                    assert result["test_original_indices"] == test
                    state["completed"].append(label)
                    state["current"] = None
                    write(out / "queue_state.json", state)
                    print("DONE", label, result["metrics"]["rmse"], flush=True)
        state["status"] = "complete"
        write(out / "queue_state.json", state)
    except BaseException:
        state["status"] = "failed"
        state["error"] = traceback.format_exc()
        write(out / "queue_state.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 独立阶段可提前执行；同一输出目录使用文件锁避免与总队列并发写入。
    with (args.output_dir / "runner.lock").open("a+b") as lock:
        if os.name == "posix":
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        main(args)
