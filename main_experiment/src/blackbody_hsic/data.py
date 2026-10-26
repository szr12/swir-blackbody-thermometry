from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SpectralDataset:
    X: np.ndarray
    y: np.ndarray
    wavelengths: np.ndarray
    group_ids: np.ndarray
    quality_tier: np.ndarray
    original_indices: np.ndarray
    band_indices: np.ndarray
    sensor_names: np.ndarray
    feature_names: np.ndarray
    feature_types: np.ndarray


def load_dataset(path: str | Path, sample_policy: str = "all_samples") -> SpectralDataset:
    """加载预处理矩阵并执行正式训练前的数据契约检查。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"数据文件不存在: {source}")

    with np.load(source, allow_pickle=False) as archive:
        feature_key = "X_raw_features" if "X_raw_features" in archive.files else "X_raw_difference"
        required = {
            feature_key, "y_temperature_c", "wavelengths_nm",
            "group_ids", "quality_tier", "formal_use_mask",
            "bad_band_mask", "standardized",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"NPZ 缺少字段: {missing}")
        if bool(np.asarray(archive["standardized"]).item()):
            raise ValueError("输入必须是未标准化矩阵，标准化只能在训练折内执行")

        X = np.asarray(archive[feature_key], dtype=np.float64)
        y = np.asarray(archive["y_temperature_c"], dtype=np.float64)
        wavelengths = np.asarray(archive["wavelengths_nm"], dtype=np.float64)
        groups = np.asarray(archive["group_ids"]).astype(str)
        quality = np.asarray(archive["quality_tier"]).astype(str)
        formal_mask = np.asarray(archive["formal_use_mask"], dtype=bool)
        bad_band_mask = np.asarray(archive["bad_band_mask"], dtype=bool)
        sensor_names = (
            np.asarray(archive["sensor_names"]).astype(str)
            if "sensor_names" in archive.files else np.asarray(["SWIR"] * X.shape[1])
        )
        feature_names = (
            np.asarray(archive["feature_names"]).astype(str)
            if "feature_names" in archive.files else np.asarray([f"band_{index:03d}" for index in range(X.shape[1])])
        )
        feature_types = (
            np.asarray(archive["feature_types"]).astype(str)
            if "feature_types" in archive.files else np.asarray(["roi_background_difference_dn"] * X.shape[1])
        )

    if X.ndim != 2 or y.shape != (X.shape[0],):
        raise ValueError(f"矩阵形状错误: X={X.shape}, y={y.shape}")
    if wavelengths.shape != (X.shape[1],) or bad_band_mask.shape != (X.shape[1],):
        raise ValueError("波长或坏波段掩码与 X 的列数不一致")
    if sensor_names.shape != wavelengths.shape or feature_names.shape != wavelengths.shape or feature_types.shape != wavelengths.shape:
        raise ValueError("特征元数据与 X 的列数不一致")
    if groups.shape != y.shape or quality.shape != y.shape or formal_mask.shape != y.shape:
        raise ValueError("样本元数据与 X 的行数不一致")
    if not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isfinite(wavelengths).all():
        raise ValueError("输入包含 NaN 或 Inf")
    if np.unique(y).size < 2:
        raise ValueError("温度标签必须至少包含两个不同值")

    if sample_policy == "all_samples":
        sample_mask = np.ones(X.shape[0], dtype=bool)
    elif sample_policy == "formal_mask":
        sample_mask = formal_mask
    else:
        raise ValueError(f"不支持的 sample_policy: {sample_policy}")

    good_band_mask = ~bad_band_mask
    if not good_band_mask.any():
        raise ValueError("没有可用波段")
    original_indices = np.flatnonzero(sample_mask)
    band_indices = np.flatnonzero(good_band_mask)
    return SpectralDataset(
        X=X[sample_mask][:, good_band_mask],
        y=y[sample_mask],
        wavelengths=wavelengths[good_band_mask],
        group_ids=groups[sample_mask],
        quality_tier=quality[sample_mask],
        original_indices=original_indices,
        band_indices=band_indices,
        sensor_names=sensor_names[good_band_mask],
        feature_names=feature_names[good_band_mask],
        feature_types=feature_types[good_band_mask],
    )
