from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_background_corrected_data(input_path: Path, output_dir: Path) -> Path:
    """用同一帧环形背景扣除中心圆DN，生成独立的预处理产物。"""
    with np.load(input_path, allow_pickle=True) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

    required = {
        "X_raw_features",
        "background_median_audit_only",
        "X_uncertainty",
        "y_temperature_c",
        "wavelengths_nm",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"输入数据缺少必要字段: {missing}")

    roi = np.asarray(arrays["X_raw_features"], dtype=np.float32)
    background = np.asarray(arrays["background_median_audit_only"], dtype=np.float32)
    if roi.shape != background.shape or roi.ndim != 2:
        raise ValueError("中心圆DN与环形背景的形状必须相同且为二维矩阵")
    if not np.isfinite(roi).all() or not np.isfinite(background).all():
        raise ValueError("中心圆DN或环形背景包含非有限值")

    corrected = roi - background
    arrays["X_raw_features"] = corrected.astype(np.float32)
    arrays["X_roi_dn"] = corrected.astype(np.float32)
    arrays["feature_types"] = np.asarray(
        ["roi_gaussian_dn_minus_same_frame_annulus_median"] * roi.shape[1]
    )
    arrays["background_subtracted"] = np.asarray(True)
    arrays["bad_nonfinite"] = ~np.isfinite(corrected).all(axis=0)
    arrays["bad_constant"] = np.ptp(corrected, axis=0) <= 1e-8
    arrays["bad_band_mask"] = arrays["bad_nonfinite"] | arrays["bad_constant"]

    # 温度标签只用于生成审计指标，不参与任何像素校正或特征生成。
    temperatures = np.asarray(arrays["y_temperature_c"], dtype=np.float64)
    unique_temperatures = np.unique(temperatures)
    before_means = np.asarray(
        [float(np.mean(roi[temperatures == value])) for value in unique_temperatures]
    )
    after_means = np.asarray(
        [float(np.mean(corrected[temperatures == value])) for value in unique_temperatures]
    )
    odd = (unique_temperatures.astype(np.int64) % 2) == 1
    even = ~odd

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "swir_roi_background_corrected.npz"
    np.savez_compressed(destination, **arrays)
    summary = {
        "status": "preprocessing_complete",
        "input_path": str(input_path.resolve()),
        "shape": list(corrected.shape),
        "background_subtracted": True,
        "formal_feature": "中心圆高斯加权DN减同帧环形背景中位数",
        "temperature_labels_used_in_feature_generation": False,
        "source_roi_field": "X_raw_features",
        "source_background_field": "background_median_audit_only",
        "static_invalid_band_count": int(arrays["bad_band_mask"].sum()),
        "negative_value_fraction": float(np.mean(corrected < 0.0)),
        "audit_only_zigzag_metrics": {
            "adjacent_temperature_mean_dn_median_before": float(
                np.median(np.abs(np.diff(before_means)))
            ),
            "adjacent_temperature_mean_dn_median_after": float(
                np.median(np.abs(np.diff(after_means)))
            ),
            "odd_even_mean_dn_difference_before": float(
                np.mean(before_means[odd]) - np.mean(before_means[even])
            ),
            "odd_even_mean_dn_difference_after": float(
                np.mean(after_means[odd]) - np.mean(after_means[even])
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="生成同帧背景校正的SWIR预处理数据")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(build_background_corrected_data(args.input_path, args.output_dir), flush=True)


if __name__ == "__main__":
    main()
