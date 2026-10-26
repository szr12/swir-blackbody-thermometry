from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from convergent_colorimetry import ConvergentColorimetricSelector


def json_ready(value):
    """将NumPy对象转换为可写入JSON的Python对象。"""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict) -> None:
    """原子写入实验产物。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def regression_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """计算回归指标；提供权重时按冻结权重计算加权指标。"""
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    return {
        "mse": float(mean_squared_error(y_true, prediction, sample_weight=weights)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction, sample_weight=weights))),
        "mae": float(mean_absolute_error(y_true, prediction, sample_weight=weights)),
        "r2": float(r2_score(y_true, prediction, sample_weight=weights)),
        "bias": float(np.average(prediction - y_true, weights=weights)),
        "max_absolute_error": float(np.max(np.abs(prediction - y_true))),
    }


def temperature_bin_ids(y: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """生成左闭右开、末段右闭的温区编号。"""
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


def observed_segmented_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
    boundaries: np.ndarray,
) -> list[dict]:
    """仅统计当前数据实际覆盖的温区，适用于连续高温外推测试。"""
    ids = temperature_bin_ids(y_true, boundaries)
    rows = []
    for segment_id in range(len(boundaries) - 1):
        mask = ids == segment_id
        if not mask.any():
            continue
        rows.append({
            "segment": segment_id,
            "lower_c": float(boundaries[segment_id]),
            "upper_c": float(boundaries[segment_id + 1]),
            "upper_inclusive": segment_id == len(boundaries) - 2,
            "n_samples": int(mask.sum()),
            **regression_metrics(y_true[mask], prediction[mask]),
        })
    return rows


def load_data(path: Path, feature_key: str, uncertainty_key: str) -> dict[str, np.ndarray]:
    """读取SWIR V3数据并应用静态坏波段掩码。"""
    with np.load(path, allow_pickle=False) as archive:
        required = {
            feature_key, uncertainty_key, "background_mad", "bad_pixel_fraction_by_sample_band",
            "saturation_fraction_by_sample_band", "y_temperature_c", "wavelengths_nm",
            "group_ids", "bad_band_mask", "sensor_names", "standardized",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"V3数据缺少字段: {missing}")
        if bool(np.asarray(archive["standardized"]).item()):
            raise ValueError("正式输入不能预先标准化")
        data = {key: np.asarray(archive[key]) for key in required}
    static_good = ~data["bad_band_mask"].astype(bool)
    for key in (
        feature_key, uncertainty_key, "background_mad",
        "bad_pixel_fraction_by_sample_band", "saturation_fraction_by_sample_band",
    ):
        data[key] = np.asarray(data[key], dtype=np.float64)[:, static_good]
    data["wavelengths_nm"] = np.asarray(data["wavelengths_nm"], dtype=np.float64)[static_good]
    data["sensor_names"] = np.asarray(data["sensor_names"]).astype(str)[static_good]
    data["original_band_indices"] = np.flatnonzero(static_good)
    data["y_temperature_c"] = np.asarray(data["y_temperature_c"], dtype=np.float64)
    data["group_ids"] = np.asarray(data["group_ids"]).astype(str)
    if not np.isfinite(data[feature_key]).all() or not np.isfinite(data[uncertainty_key]).all():
        raise ValueError("V3特征或不确定度包含非有限值")
    return data


def fold_quality_mask(
    X_train: np.ndarray,
    uncertainty_train: np.ndarray,
    background_mad_train: np.ndarray,
    bad_pixel_train: np.ndarray,
    saturation_train: np.ndarray,
    wavelengths_nm: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, dict]:
    """只用训练集估计统计型波段质控规则。"""
    finite = (
        np.all(np.isfinite(X_train), axis=0)
        & np.all(np.isfinite(uncertainty_train), axis=0)
        & np.all(np.isfinite(background_mad_train), axis=0)
    )
    nonconstant = np.ptp(X_train, axis=0) > 1e-8
    relative = np.median(uncertainty_train / np.maximum(np.abs(X_train), 1.0), axis=0)
    relative_ok = relative <= float(config["maximum_median_relative_uncertainty"])
    background = np.median(background_mad_train, axis=0)
    center = float(np.median(background))
    spread = float(1.4826 * np.median(np.abs(background - center)))
    background_ok = background <= center + float(config["background_noise_mad_multiplier"]) * max(spread, 1e-8)
    bad_pixel_ok = np.median(bad_pixel_train, axis=0) <= float(config["maximum_median_bad_pixel_fraction"])
    saturation_ok = np.max(saturation_train, axis=0) <= float(config["maximum_saturation_fraction"])
    mask = finite & nonconstant & relative_ok & background_ok & bad_pixel_ok & saturation_ok
    protected = []
    for target in config.get("protected_wavelengths_nm", []):
        index = int(np.argmin(np.abs(wavelengths_nm - float(target))))
        if np.isclose(wavelengths_nm[index], float(target), atol=0.02) and finite[index] and nonconstant[index]:
            mask[index] = True
            protected.append(index)
    if mask.sum() < 2:
        raise RuntimeError("训练集质控后不足两个有效波段")
    return mask, {
        "retained_band_count": int(mask.sum()),
        "rejected_band_count": int((~mask).sum()),
        "protected_band_indices": protected,
        "rejected_relative_uncertainty": np.flatnonzero(~relative_ok),
        "rejected_background_noise": np.flatnonzero(~background_ok),
        "rejected_bad_pixel_fraction": np.flatnonzero(~bad_pixel_ok),
        "rejected_saturation": np.flatnonzero(~saturation_ok),
    }


def make_traditional_split(
    temperatures_c: np.ndarray,
    boundaries_c: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按温区分层生成一次70%/15%/15%随机划分。"""
    indices = np.arange(len(temperatures_c), dtype=np.int64)
    bins = temperature_bin_ids(temperatures_c, boundaries_c)
    train_index, remaining_index = train_test_split(
        indices, test_size=0.30, random_state=seed, shuffle=True, stratify=bins,
    )
    validation_index, test_index = train_test_split(
        remaining_index, test_size=0.50, random_state=seed + 1,
        shuffle=True, stratify=bins[remaining_index],
    )
    return np.sort(train_index), np.sort(validation_index), np.sort(test_index)


def make_temperature_ordered_split(
    temperatures_c: np.ndarray,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按温度升序连续划分，最高温部分只用于外推测试。"""
    values = np.asarray(temperatures_c, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("温度必须是一维有限数组")
    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("训练集和验证集比例必须位于0到1之间")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("训练集与验证集比例之和必须小于1")
    order = np.argsort(values, kind="stable")
    train_end = int(np.floor(values.size * train_fraction))
    validation_end = train_end + int(np.floor(values.size * validation_fraction))
    if train_end < 1 or validation_end <= train_end or validation_end >= values.size:
        raise ValueError("连续温度划分产生了空数据集")
    train = order[:train_end]
    validation = order[train_end:validation_end]
    test = order[validation_end:]
    if not (values[train].max() < values[validation].min() <= values[validation].max() < values[test].min()):
        raise ValueError("划分边界存在重复温度，无法形成严格温度外推测试")
    return np.sort(train), np.sort(validation), np.sort(test)


def make_configured_split(
    temperatures_c: np.ndarray,
    boundaries_c: np.ndarray,
    split_config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据实验配置选择传统随机划分或连续温度外推划分。"""
    strategy = str(split_config["strategy"])
    if strategy == "temperature_stratified_random_70_15_15":
        return make_traditional_split(
            temperatures_c, boundaries_c, int(split_config.get("seed", 42)),
        )
    if strategy == "temperature_ordered_70_15_15":
        return make_temperature_ordered_split(
            temperatures_c,
            float(split_config.get("train_fraction", 0.70)),
            float(split_config.get("validation_fraction", 0.15)),
        )
    raise ValueError(f"不支持的数据集划分策略: {strategy}")


def mean_absolute_relative_deviation(y_true: np.ndarray, prediction: np.ndarray) -> float:
    """计算平均绝对相对偏差，返回0到1之间的比例值。"""
    true = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    if true.shape != predicted.shape or np.any(true == 0.0):
        raise ValueError("相对偏差要求预测形状一致且真实值非零")
    return float(np.mean(np.abs(predicted - true) / np.abs(true)))


def make_selector(
    config: dict,
    band_count: int,
    redundancy_weight: float,
    seed: int,
    adaptive_weighting: dict | None = None,
):
    """构造真实温度误差驱动的Top-K选带器。"""
    adaptive = dict(adaptive_weighting or {})
    freeze_gate_after = config.get("freeze_gate_after")
    return ConvergentColorimetricSelector(
        band_count=band_count,
        target_training_rmse_c=float(config["target_training_rmse_c"]),
        learning_rate=float(config["learning_rate"]),
        gate_learning_rate=float(config["gate_learning_rate"]),
        freeze_gate_after=None if freeze_gate_after is None else int(freeze_gate_after),
        max_iter=int(config["max_iter"]),
        evaluation_interval=int(config["evaluation_interval"]),
        required_consecutive_successes=int(config["required_consecutive_successes"]),
        redundancy_weight=redundancy_weight,
        binarization_weight=float(config["binarization_weight"]),
        hidden_units=tuple(config["hidden_units"]),
        intensity_floor_fraction=float(config["intensity_floor_fraction"]),
        absolute_intensity_floor=float(config["absolute_intensity_floor"]),
        adaptive_weighting=bool(adaptive.get("enabled", False)),
        attention_hidden_units=int(adaptive.get("attention_hidden_units", 16)),
        attention_loss_weight=float(adaptive.get("attention_loss_weight", 0.1)),
        adaptive_weight_power=float(adaptive.get("power", 0.5)),
        adaptive_weight_minimum=float(adaptive.get("minimum", 0.5)),
        adaptive_weight_maximum=float(adaptive.get("maximum", 2.0)),
        adaptive_weight_warmup_iterations=int(adaptive.get("warmup_iterations", 100)),
        error_teacher_blend=float(adaptive.get("error_teacher_blend", 0.5)),
        required_wavelength_ranges_nm=tuple(
            tuple(float(bound) for bound in interval)
            for interval in config.get("required_wavelength_ranges_nm", [])
        ),
        device=str(config["device"]),
        seed=seed,
        selector_feature_mode=str(config.get("selector_feature_mode", "centered")),
    )


def model_parameters(config: dict) -> dict:
    """提取分温区模型所需参数。"""
    model = config["model"]
    adaptive = dict(config.get("adaptive_weighting", {}))
    return {
        "target_training_rmse_c": float(model["target_training_rmse_c"]),
        "learning_rate": float(model["learning_rate"]),
        "max_iter": int(model["max_iter"]),
        "evaluation_interval": int(model["evaluation_interval"]),
        "required_consecutive_successes": int(model["required_consecutive_successes"]),
        "hidden_units": tuple(int(value) for value in model["hidden_units"]),
        "intensity_floor_fraction": float(model["intensity_floor_fraction"]),
        "absolute_intensity_floor": float(model["absolute_intensity_floor"]),
        "adaptive_weighting": bool(adaptive.get("enabled", False)),
        "attention_hidden_units": int(adaptive.get("attention_hidden_units", 16)),
        "attention_loss_weight": float(adaptive.get("attention_loss_weight", 0.1)),
        "adaptive_weight_power": float(adaptive.get("power", 0.5)),
        "adaptive_weight_minimum": float(adaptive.get("minimum", 0.5)),
        "adaptive_weight_maximum": float(adaptive.get("maximum", 2.0)),
        "adaptive_weight_warmup_iterations": int(adaptive.get("warmup_iterations", 100)),
        "error_teacher_blend": float(adaptive.get("error_teacher_blend", 0.5)),
        "device": str(model["device"]),
        "regression_feature_mode": str(model.get("regression_feature_mode", "helmert")),
    }
