"""在 v3 数据上独立运行论文忠实的 Dual-sPLS_l 对比实验。"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from dual_spls_reproduction.selector import DualSPLSSelector, _fit_lasso  # noqa: E402


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "r2": float(r2_score(y_true, prediction)),
        "bias": float(np.mean(prediction - y_true)),
        "max_absolute_error": float(np.max(np.abs(prediction - y_true))),
    }


def temperature_bin_ids(y: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=np.float64)
    edges = np.asarray(boundaries, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 3 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("温区边界必须严格递增且至少包含三个点")
    if np.any(values < edges[0]) or np.any(values > edges[-1]):
        raise ValueError("温度超出配置的温区范围")
    return np.searchsorted(edges[1:-1], values, side="right")


def segmented_metrics(y_true: np.ndarray, prediction: np.ndarray, boundaries: np.ndarray) -> list[dict]:
    ids = temperature_bin_ids(y_true, boundaries)
    rows = []
    for segment_id in range(len(boundaries) - 1):
        mask = ids == segment_id
        if not mask.any():
            raise ValueError(f"温区 {segment_id} 没有样本")
        rows.append({
            "segment": segment_id,
            "lower_c": float(boundaries[segment_id]),
            "upper_c": float(boundaries[segment_id + 1]),
            "upper_inclusive": segment_id == len(boundaries) - 2,
            "n_samples": int(mask.sum()),
            **regression_metrics(y_true[mask], prediction[mask]),
        })
    return rows


def mean_absolute_relative_deviation(y_true: np.ndarray, prediction: np.ndarray) -> float:
    true = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    if true.shape != predicted.shape or np.any(true == 0.0):
        raise ValueError("相对偏差要求预测形状一致且真实值非零")
    return float(np.mean(np.abs(predicted - true) / np.abs(true)))


def _load_data(path: Path, feature_key: str, uncertainty_key: str) -> dict[str, np.ndarray]:
    """读取 v3 数据；只应用数据文件自带的静态坏波段掩码。"""
    with np.load(path, allow_pickle=False) as archive:
        required = {
            feature_key,
            uncertainty_key,
            "background_mad",
            "bad_pixel_fraction_by_sample_band",
            "saturation_fraction_by_sample_band",
            "y_temperature_c",
            "wavelengths_nm",
            "group_ids",
            "bad_band_mask",
            "sensor_names",
            "standardized",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"v3 数据缺少字段: {missing}")
        if bool(np.asarray(archive["standardized"]).item()):
            raise ValueError("输入不能预先标准化")
        data = {key: np.asarray(archive[key]) for key in required}
    static_good = ~data["bad_band_mask"].astype(bool)
    for key in (
        feature_key,
        uncertainty_key,
        "background_mad",
        "bad_pixel_fraction_by_sample_band",
        "saturation_fraction_by_sample_band",
    ):
        data[key] = np.asarray(data[key], dtype=np.float64)[:, static_good]
    data["wavelengths_nm"] = np.asarray(data["wavelengths_nm"], dtype=np.float64)[static_good]
    data["sensor_names"] = np.asarray(data["sensor_names"]).astype(str)[static_good]
    data["original_band_indices"] = np.flatnonzero(static_good)
    data["y_temperature_c"] = np.asarray(data["y_temperature_c"], dtype=np.float64)
    data["group_ids"] = np.asarray(data["group_ids"]).astype(str)
    if not np.isfinite(data[feature_key]).all():
        raise ValueError("v3 特征包含非有限值")
    return data


def make_traditional_split(
    temperatures_c: np.ndarray,
    boundaries_c: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """复现当前主流程的温区分层 70%/15%/15% 划分。"""
    indices = np.arange(len(temperatures_c), dtype=np.int64)
    bins = temperature_bin_ids(temperatures_c, boundaries_c)
    train_index, remaining_index = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        shuffle=True,
        stratify=bins,
    )
    validation_index, test_index = train_test_split(
        remaining_index,
        test_size=0.50,
        random_state=seed + 1,
        shuffle=True,
        stratify=bins[remaining_index],
    )
    return np.sort(train_index), np.sort(validation_index), np.sort(test_index)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _guard_json(path: Path, payload: dict) -> None:
    normalized = _json_ready(payload)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise RuntimeError(f"已有快照与当前 Dual-sPLS 对比实验不一致: {path}")
        return
    _write_json(path, normalized)


def _reference_paths(config: dict, project_root: Path) -> tuple[Path, Path]:
    reference = config.get("comparison_reference", {})
    config_path = (project_root / reference.get(
        "config_snapshot",
        "optimization_experiment/outputs/swir_phase7_lambda_overlap_current_test_seed42/config.snapshot.json",
    )).resolve()
    summary_path = (project_root / reference.get(
        "summary",
        "optimization_experiment/outputs/swir_phase7_lambda_overlap_current_test_seed42/summary.json",
    )).resolve()
    return config_path, summary_path


def _verify_reference(config: dict, project_root: Path) -> dict:
    """只核对共享数据、划分和评价目标，不继承主方法模型组件。"""
    config_path, summary_path = _reference_paths(config, project_root)
    reference_config = json.loads(config_path.read_text(encoding="utf-8"))
    reference_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ("data_path", "feature_key", "uncertainty_key", "split_seed", "temperature_boundaries_c"):
        if reference_config.get(key) != config.get(key):
            raise RuntimeError(f"Dual-sPLS 对比配置未与最新主流程对齐: {key}")
    return {
        "config_snapshot": str(config_path),
        "config_sha256": _sha256(config_path),
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "reference_regional_metrics": reference_summary["regional_metrics"],
        "reference_regional_relative_deviation": reference_summary["regional_relative_deviation"],
    }


def _load_and_split(config: dict, project_root: Path) -> tuple[dict, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    data = _load_data(
        (project_root / config["data_path"]).resolve(),
        str(config["feature_key"]),
        str(config["uncertainty_key"]),
    )
    boundaries = np.asarray(config["temperature_boundaries_c"], dtype=np.float64)
    split = make_traditional_split(data["y_temperature_c"], boundaries, int(config["split_seed"]))
    train_index, validation_index, test_index = split
    if (
        np.intersect1d(train_index, validation_index).size
        or np.intersect1d(train_index, test_index).size
        or np.intersect1d(validation_index, test_index).size
    ):
        raise RuntimeError("Dual-sPLS 对比实验划分发生重叠")
    return data, split


def prepare_comparison(
    config: dict,
    project_root: Path,
    run_dir: Path,
    *,
    verify_reference: bool = True,
) -> dict:
    """校验 v3 数据和共享划分，不启动选带或测试评价。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    reference = _verify_reference(config, project_root) if verify_reference else None
    data, (train_index, validation_index, test_index) = _load_and_split(config, project_root)
    data_path = (project_root / config["data_path"]).resolve()
    manifest = {
        "strategy": "temperature_stratified_random_70_15_15",
        "seed": int(config["split_seed"]),
        "train_indices": train_index,
        "validation_indices": validation_index,
        "test_indices": test_index,
        "train_size": int(train_index.size),
        "validation_size": int(validation_index.size),
        "test_size": int(test_index.size),
    }
    metadata = {
        "status": "prepared_not_trained",
        "method": config["method"],
        "data_version": config.get("data_version", "v3_foldsafe"),
        "data_path": str(data_path),
        "data_sha256": _sha256(data_path),
        "feature_key": config["feature_key"],
        "n_samples": int(data[config["feature_key"]].shape[0]),
        "n_bands_after_static_bad_band_mask": int(data[config["feature_key"]].shape[1]),
        "split_sizes": {
            "train": int(train_index.size),
            "validation": int(validation_index.size),
            "test": int(test_index.size),
        },
        "comparison_reference": reference,
        "selection_started": False,
        "test_evaluated": False,
    }
    _guard_json(run_dir / "config.snapshot.json", config)
    _guard_json(run_dir / "dataset.json", metadata)
    _guard_json(run_dir / "split_manifest.json", manifest)
    return _json_ready(metadata)


def run_comparison(
    config: dict,
    project_root: Path,
    run_dir: Path,
    *,
    verify_reference: bool = True,
) -> dict:
    """运行原生 Dual-sPLS_l；Test 只在模型配置冻结后评价一次。"""
    prepared = prepare_comparison(config, project_root, run_dir, verify_reference=verify_reference)
    data, (train_index, validation_index, test_index) = _load_and_split(config, project_root)
    X = data[config["feature_key"]]
    y = data["y_temperature_c"]
    wavelengths = data["wavelengths_nm"]
    selector_config = config["selector"]
    started = time.perf_counter()

    # 论文和作者包用重复随机校准/验证选择潜变量数；整个过程严格限制在 Train 内。
    selector = DualSPLSSelector(
        profile=str(selector_config["profile"]),
        random_state=int(selector_config["random_state"]),
    ).fit(X[train_index], y[train_index])
    best_components = int(selector.diagnostics_["best_components"])
    train_selected = selector.select()
    selection_seconds = time.perf_counter() - started

    # 潜变量数冻结后，仿照作者流程在全部可用于拟合的数据上重拟合原生模型。
    fit_index = np.sort(np.concatenate([train_index, validation_index]))
    model = _fit_lasso(
        X[fit_index],
        y[fit_index],
        n_components=best_components,
        ppnu=float(selector.diagnostics_["ppnu"]),
    )
    final_coefficients = model.coefficients[:, best_components - 1]
    selected = np.flatnonzero(final_coefficients != 0.0).astype(np.int64)
    if selected.size == 0:
        raise RuntimeError("Train+Validation 重拟合后没有非零回归系数")
    prediction = np.asarray(model.predict(X[test_index], best_components), dtype=np.float64).reshape(-1)
    metrics = regression_metrics(y[test_index], prediction)
    relative_deviation = mean_absolute_relative_deviation(y[test_index], prediction)
    boundaries = np.asarray(config["temperature_boundaries_c"], dtype=np.float64)
    test_bins = temperature_bin_ids(y[test_index], boundaries)
    reference = prepared.get("comparison_reference")
    result = {
        "status": "success",
        "method": config["method"],
        "evaluation_scope": "current_seed42_development_test",
        "independent_generalization_claim_allowed": False,
        "test_used_for_selection_or_stopping": False,
        "split_sizes": {
            "train": int(train_index.size),
            "validation": int(validation_index.size),
            "final_fit_train_plus_validation": int(fit_index.size),
            "test": int(test_index.size),
        },
        "selector": {
            "component_selection_data_roles": ["train"],
            "profile": selector.profile,
            "ppnu": float(selector.diagnostics_["ppnu"]),
            "support_size_policy": "natural_nonzero_coefficients",
            "best_components": best_components,
            "train_natural_band_count": int(train_selected.size),
            "train_selected_available_band_indices": train_selected,
            "natural_band_count": int(selected.size),
            "selected_available_band_indices": selected,
            "selected_band_indices": data["original_band_indices"][selected],
            "selected_wavelengths_nm": wavelengths[selected],
            "selection_seconds": selection_seconds,
            "diagnostics": selector.diagnostics_,
        },
        "final_model": {
            "name": "dual_spls_l",
            "algorithm": "paper_algorithm_2_pseudo_lasso",
            "fit_scope": "train_plus_validation_after_component_selection",
            "best_components": best_components,
            "ppnu": float(selector.diagnostics_["ppnu"]),
            "mean_center_X_and_y": True,
            "uses_native_coefficients": True,
            "coefficients": final_coefficients,
            "intercept": float(model.intercepts[best_components - 1]),
        },
        "test_original_indices": test_index,
        "test_temperatures_c": y[test_index],
        "predictions_c": prediction,
        "metrics": metrics,
        "relative_deviation": relative_deviation,
        "segment_metrics": segmented_metrics(y[test_index], prediction, boundaries),
        "segment_relative_deviations": [
            mean_absolute_relative_deviation(
                y[test_index][test_bins == region_id],
                prediction[test_bins == region_id],
            )
            for region_id in range(boundaries.size - 1)
        ],
        "comparison_reference": reference,
        "delta_dual_spls_minus_reference": None if reference is None else {
            "rmse": float(metrics["rmse"] - reference["reference_regional_metrics"]["rmse"]),
            "mae": float(metrics["mae"] - reference["reference_regional_metrics"]["mae"]),
            "r2": float(metrics["r2"] - reference["reference_regional_metrics"]["r2"]),
            "relative_deviation": float(
                relative_deviation - reference["reference_regional_relative_deviation"]
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(run_dir / "summary.json", result)
    return _json_ready(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-sPLS_l 2023 原生回归独立对比实验")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("prepare", "train"), default="prepare")
    parser.add_argument("--run-dir")
    arguments = parser.parse_args()
    config = json.loads(Path(arguments.config).resolve().read_text(encoding="utf-8"))
    if arguments.run_dir:
        run_dir = Path(arguments.run_dir).resolve()
    else:
        name = config["experiment_name"] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (PROJECT_ROOT / config["output_root"] / name).resolve()
    function = prepare_comparison if arguments.mode == "prepare" else run_comparison
    result = function(config, PROJECT_ROOT, run_dir)
    if arguments.mode == "prepare":
        displayed = result
    else:
        displayed = {
            "status": result["status"],
            "method": result["method"],
            "evaluation_scope": result["evaluation_scope"],
            "natural_band_count": result["selector"]["natural_band_count"],
            "selected_band_indices": result["selector"]["selected_band_indices"],
            "selected_wavelengths_nm": result["selector"]["selected_wavelengths_nm"],
            "metrics": result["metrics"],
            "relative_deviation": result["relative_deviation"],
            "delta_dual_spls_minus_reference": result["delta_dual_spls_minus_reference"],
            "run_dir": str(run_dir),
        }
    print(json.dumps(displayed, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
