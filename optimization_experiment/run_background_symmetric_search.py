from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def symmetric_k_values(center: int, step: int, half_steps: int, maximum: int) -> list[int]:
    """在合法K域内生成两侧数量相同的等距窗口。"""
    if not 2 <= center <= maximum or step < 1 or half_steps < 1:
        raise ValueError("K中心、步长或半宽不合法")
    width = min(half_steps, (center - 2) // step, (maximum - center) // step)
    if width < 1:
        raise ValueError("最优K位于物理边界，无法在其两侧对称放置合法候选")
    return [center + offset * step for offset in range(-width, width + 1)]


def symmetric_float_values(center: float, step: float, half_steps: int) -> list[float]:
    """连续参数按中心值两侧等距取点，避免浮点累计误差。"""
    values = [round(center + offset * step, 8) for offset in range(-half_steps, half_steps + 1)]
    if min(values) < 0:
        raise ValueError("对称参数窗口包含负值")
    return values


def fingerprint(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(str((array.shape, array.dtype)).encode() + array.tobytes()).hexdigest()


def selection_signature(parameters: dict, arrays: tuple) -> dict:
    """拟合前冻结配置快照，防止拟合新增属性污染缓存指纹。"""
    return json.loads(json.dumps({
        "parameters": parameters,
        "arrays": [fingerprint(np.asarray(array)) for array in arrays],
    }))


def verify_cached_signature(saved: dict, expected: dict) -> bool:
    """兼容旧缓存中误写的拟合变换，同时严格核对全部配置和输入。"""
    normalized = copy.deepcopy(saved)
    repaired = "transform_" in normalized["parameters"]
    normalized["parameters"].pop("transform_", None)
    if normalized != expected:
        raise RuntimeError("选带缓存指纹不一致")
    return repaired


def run_pipeline(config_path: Path, output: Path, runtime: Path) -> None:
    """先扫描K，再扩展参数，最后围绕新最优K对称核验。"""
    sys.path.insert(0, str(ROOT / "main_experiment/src"))
    sys.path.insert(0, str(runtime))
    import run_phase7_traditional_hyperparameter_search as search
    from hyperparameter_objectives import configured_metric
    from convergent_colorimetry import ConvergentSelectionResult
    from traditional_utils import load_data, make_configured_split, fold_quality_mask, write_json

    config = json.loads(config_path.read_text(encoding="utf-8"))
    metric = configured_metric(config)
    if config.get("evaluate_test_after_search") is not False:
        raise ValueError("扩大超参搜索不允许评价Test")
    data_path = ROOT / config["data_path"]
    with np.load(data_path, allow_pickle=False) as archive:
        if not bool(archive["background_subtracted"].item()):
            raise ValueError("输入必须扣除同帧背景")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.snapshot.json", config)
    settings = config["symmetric_search"]
    data = load_data(data_path, config["feature_key"], config["uncertainty_key"])
    boundaries = np.asarray(config["temperature_boundaries_c"])
    train, validation, test = make_configured_split(data["y_temperature_c"], boundaries, config["split"])
    mask, diagnostics = fold_quality_mask(
        data[config["feature_key"]][train], data[config["uncertainty_key"]][train],
        data["background_mad"][train], data["bad_pixel_fraction_by_sample_band"][train],
        data["saturation_fraction_by_sample_band"][train], data["wavelengths_nm"], config["fold_quality_control"],
    )
    maximum_k = int(mask.sum())
    write_json(output / "preflight.json", {
        "background_subtracted": True, "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "shape": list(data[config["feature_key"]].shape), "quality_control": diagnostics,
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "test_evaluated": False,
        "runtime_sources_sha256": {name: hashlib.sha256((runtime / name).read_bytes()).hexdigest() for name in (
            "convergent_colorimetry.py", "traditional_utils.py", "regional_colorimetry.py",
            "run_phase7_traditional_hyperparameter_search.py",
            "hyperparameter_objectives.py",
        )},
    })
    original_factory = search._selector
    cache_dir = output / "selection_cache"
    cache_dir.mkdir(exist_ok=True)

    class CachedSelector:
        def __init__(self, original):
            self.original = original

        def fit(self, *arrays):
            """只有配置与Train、Validation、波长、互信息全匹配才复用选带。"""
            signature = selection_signature(self.original.__dict__, arrays)
            key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
            path = cache_dir / f"{key}.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if verify_cached_signature(payload["signature"], signature):
                    payload["signature"] = signature
                    payload["signature_repair"] = "移除旧版误写的拟合属性transform_，配置与全部输入指纹完全一致"
                    write_json(path, payload)
                result = payload["result"]
                result["selected_indices"] = np.asarray(result["selected_indices"], dtype=np.int64)
                result["weights"] = np.asarray(result["weights"], dtype=np.float64)
                print(f"复用精确指纹选带缓存 K={self.original.band_count} λ={self.original.redundancy_weight}", flush=True)
                return ConvergentSelectionResult(**result)
            result = self.original.fit(*arrays)
            write_json(path, {"signature": signature, "result": asdict(result)})
            return result

    def cached_factory(*args, **kwargs):
        return CachedSelector(original_factory(*args, **kwargs))

    search._selector = cached_factory
    state = {"status": "running", "test_evaluated": False, "stages": []}
    center = dict(settings["initial_center"])
    started = time.time()
    all_rows = []
    last_cache = None

    def stage(name: str, grid: dict) -> dict:
        nonlocal last_cache
        run_dir = output / name
        stage_config = copy.deepcopy(config)
        stage_config["experiment_name"] = name
        stage_config["search_grid"].update(grid)
        stage_config.pop("mi_cache_source", None)
        if last_cache:
            stage_config["mi_cache_source"] = str(last_cache)
        write_json(output / f"{name}.config.json", stage_config)
        state.update({"current_stage": name, "current_grid": grid, "elapsed_seconds": time.time() - started})
        write_json(output / "pipeline_state.json", state)
        if (run_dir / "summary.json").exists():
            if json.loads((run_dir / "config.snapshot.json").read_text(encoding="utf-8")) != stage_config:
                raise RuntimeError("恢复阶段与保存配置不同")
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        else:
            print(f"开始阶段 {name}: {grid}", flush=True)
            summary = search.run(stage_config, ROOT, run_dir)
        if summary.get("test_evaluated") is not False:
            raise RuntimeError("阶段意外评价了Test")
        rows = json.loads((run_dir / "candidates.json").read_text(encoding="utf-8"))["candidates"]
        for row in rows:
            all_rows.append({**row, "stage": name})
        write_json(output / "all_candidates.json", {"candidates": all_rows})
        last_cache = run_dir / "cache/training_mutual_information.npy"
        state["stages"].append({
            "name": name, "candidate_count": len(rows), "best": summary["best_hyperparameters"],
            "best_validation_rmse_c": summary["best_validation_rmse_c"],
            "selection_metric": metric,
            "best_validation_selection_score_c": summary.get("best_validation_selection_score_c", summary["best_validation_rmse_c"]),
        })
        write_json(output / "pipeline_state.json", state)
        score = summary.get("best_validation_selection_score_c", summary["best_validation_rmse_c"])
        print(f"完成阶段 {name}: 最优={summary['best_hyperparameters']} Validation({metric})={score:.4f} ℃", flush=True)
        return summary

    try:
        preliminary = stage("stage1_k_screen", {
            "band_counts": settings["initial_k_values"],
            "redundancy_weights": [center["redundancy_weight"]],
            "overlap_values_c": [center["overlap_c"]],
            "routing_width_values_c": [center["routing_width_c"]],
        })
        center = preliminary["best_hyperparameters"]
        k_window = symmetric_k_values(center["band_count"], settings["k_step"], settings["k_half_steps"], maximum_k)
        joint = stage("stage2_symmetric_k_lambda", {
            "band_counts": k_window,
            "redundancy_weights": symmetric_float_values(center["redundancy_weight"], settings["lambda_step"], settings["lambda_half_steps"]),
            "overlap_values_c": [center["overlap_c"]],
            "routing_width_values_c": [center["routing_width_c"]],
        })
        center = joint["best_hyperparameters"]
        temperatures = stage("stage3_symmetric_overlap_routing", {
            "band_counts": [center["band_count"]], "redundancy_weights": [center["redundancy_weight"]],
            "overlap_values_c": symmetric_float_values(center["overlap_c"], settings["temperature_step_c"], settings["temperature_half_steps"]),
            "routing_width_values_c": symmetric_float_values(center["routing_width_c"], settings["temperature_step_c"], settings["temperature_half_steps"]),
        })
        center = temperatures["best_hyperparameters"]
        confirmation_number = 0
        while True:
            k_window = symmetric_k_values(center["band_count"], settings["k_step"], settings["k_half_steps"], maximum_k)
            confirmation_number += 1
            summary = stage(f"stage4_k_confirmation_{confirmation_number:02d}", {
                "band_counts": k_window, "redundancy_weights": [center["redundancy_weight"]],
                "overlap_values_c": [center["overlap_c"]], "routing_width_values_c": [center["routing_width_c"]],
            })
            new_center = summary["best_hyperparameters"]
            if new_center["band_count"] == center["band_count"]:
                break
            center = new_center
        frozen = json.loads((output / state["stages"][-1]["name"] / "frozen_parameters.json").read_text(encoding="utf-8"))
        frozen.update({
            "selection_metric": metric,
            "data_path": config["data_path"], "background_subtracted": True,
            "search_design": "先K扫描、再K与λ条件联合搜索、再重叠与路由条件搜索、最后最优K对称核验",
            "k_symmetry": {"center": center["band_count"], "values": k_window, "step": settings["k_step"]},
        })
        write_json(output / "frozen_parameters.json", frozen)
        write_json(output / "summary.json", {
            "selection_metric": metric,
            "best_validation_selection_score_c": summary.get("best_validation_selection_score_c", summary["best_validation_rmse_c"]),
            "best_validation_unweighted_rmse_c": summary.get("best_validation_unweighted_rmse_c"),
            "status": "search_complete", "test_evaluated": False,
            "test_used_for_hyperparameter_selection": False,
            "best_hyperparameters": summary["best_hyperparameters"],
            "best_validation_rmse_c": summary["best_validation_rmse_c"],
            "best_validation_relative_deviation": summary["best_validation_relative_deviation"],
            "k_symmetry": frozen["k_symmetry"], "stages": state["stages"],
            "total_candidate_evaluations": len(all_rows),
            "search_design": frozen["search_design"], "elapsed_seconds": time.time() - started,
        })
        state.update({"status": "search_complete", "best_hyperparameters": summary["best_hyperparameters"], "k_symmetry": frozen["k_symmetry"]})
        write_json(output / "pipeline_state.json", state)
    except Exception as error:
        state.update({"status": "failed", "error": str(error), "elapsed_seconds": time.time() - started})
        write_json(output / "pipeline_state.json", state)
        raise
    finally:
        search._selector = original_factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run_pipeline(args.config.resolve(), args.run_dir.resolve(), args.runtime_dir.resolve())


if __name__ == "__main__":
    main()
