from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from analyze_recapture_swir import (
    build_reference,
    detect_swir_geometry,
    open_cube,
    scan_records,
    swir_proxy,
)
from spatial_uncertainty import RECAPTURE_RANGES, in_ranges, process_cube


def _quality_tier(temperature: float) -> str:
    """质量标签只描述采集情况，不决定样本是否进入正式实验。"""
    if in_ranges(temperature, RECAPTURE_RANGES):
        return "validated_recapture"
    if temperature < 150:
        return "low_signal_user_included"
    if temperature <= 198:
        return "conditional_low_signal"
    return "acquisition_zigzag"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成无全局统计删带的SWIR折安全预处理产物")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = sorted(
        [row for row in scan_records(args.dataset_root) if "chongxin" not in row.group_id.lower()],
        key=lambda row: row.temperature_c,
    )
    if len(records) != 401 or len({row.temperature_c for row in records}) != 401:
        raise ValueError("SWIR主序列必须包含401个唯一温度扫描")
    reference = build_reference(records)
    features: list[np.ndarray] = []
    uncertainties: list[np.ndarray] = []
    background_mads: list[np.ndarray] = []
    outlier_fractions: list[np.ndarray] = []
    saturation_fractions: list[np.ndarray] = []
    manifest: list[dict[str, object]] = []

    for index, record in enumerate(records):
        cube = open_cube(record)
        geometry = detect_swir_geometry(swir_proxy(cube, record.wavelengths_nm), reference)
        result = process_cube(cube, geometry)
        features.append(np.asarray(result["difference_annulus"], dtype=np.float32))
        uncertainties.append(np.asarray(result["uncertainty_annulus"], dtype=np.float32))
        background_mads.append(np.asarray(result["background_mad"], dtype=np.float32))
        outlier_fractions.append(np.asarray(result["outlier_fraction_by_band"], dtype=np.float32))
        saturation_fractions.append(np.asarray(result["saturation_fraction_by_band"], dtype=np.float32))
        manifest.append({
            "sample_index": index,
            "temperature_c": record.temperature_c,
            "group_id": record.group_id,
            "quality_tier": _quality_tier(record.temperature_c),
            "center_x": geometry.center_x,
            "center_y": geometry.center_y,
            "radius_px": geometry.radius,
            "dark_current_corrected": record.dark_current_corrected,
        })
        print(f"[{index + 1:3d}/{len(records)}] {record.group_id}", flush=True)

    X = np.vstack(features)
    uncertainty = np.vstack(uncertainties)
    background_mad = np.vstack(background_mads)
    outlier_fraction = np.vstack(outlier_fractions)
    saturation_fraction = np.vstack(saturation_fractions)
    wavelengths = records[0].wavelengths_nm.astype(np.float64)

    # 划分前只删除确定无效的仪器列；统计型质控必须在每个训练折内部计算。
    bad_nonfinite = ~(
        np.all(np.isfinite(X), axis=0)
        & np.all(np.isfinite(uncertainty), axis=0)
        & np.all(np.isfinite(background_mad), axis=0)
    )
    bad_constant = np.ptp(X, axis=0) <= 1e-8
    bad_band_mask = bad_nonfinite | bad_constant
    if bad_band_mask[0]:
        raise RuntimeError("858.45 nm锚点被确定无效规则删除，请检查原始数据")

    y = np.asarray([row.temperature_c for row in records], dtype=np.float64)
    groups = np.asarray([row.group_id for row in records])
    quality = np.asarray([row["quality_tier"] for row in manifest])
    np.savez_compressed(
        args.output_dir / "swir_preprocessed_foldsafe_v3.npz",
        X_raw_features=X,
        X_uncertainty=uncertainty,
        background_mad=background_mad,
        bad_pixel_fraction_by_sample_band=outlier_fraction,
        saturation_fraction_by_sample_band=saturation_fraction,
        y_temperature_c=y,
        wavelengths_nm=wavelengths,
        group_ids=groups,
        quality_tier=quality,
        formal_use_mask=np.ones(X.shape[0], dtype=bool),
        bad_band_mask=bad_band_mask,
        bad_nonfinite=bad_nonfinite,
        bad_constant=bad_constant,
        sensor_names=np.asarray(["SWIR"] * X.shape[1]),
        feature_names=np.asarray([f"swir_{value:.2f}nm" for value in wavelengths]),
        feature_types=np.asarray(["roi_gaussian_minus_annulus_median_dn"] * X.shape[1]),
        standardized=np.asarray(False),
        radiometric_calibrated=np.asarray(False),
    )
    _write_csv(args.output_dir / "swir_v3_sample_manifest.csv", manifest)
    summary = {
        "status": "preprocessing_complete",
        "shape": list(X.shape),
        "feature_key": "X_raw_features",
        "uncertainty_key": "X_uncertainty",
        "global_data_derived_band_filtering": False,
        "fixed_spectral_edge_filtering": False,
        "anchor_858_45_nm_retained": bool(np.isclose(wavelengths[0], 858.45) and not bad_band_mask[0]),
        "static_invalid_band_count": int(bad_band_mask.sum()),
        "all_401_samples_retained": True,
        "radiometric_calibrated": False,
        "temperature_labels_used_in_feature_generation": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "preprocessing_config.json").write_text(
        json.dumps({
            "dataset_root": str(args.dataset_root.resolve()),
            "formal_feature": "ROI高斯加权均值减同帧环形背景中位数",
            "uncertainty": "ROI与环形背景稳健标准误的平方和开方",
            "data_derived_qc": "deferred_to_each_training_fold",
            "standardization": "deferred_to_each_training_fold",
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
