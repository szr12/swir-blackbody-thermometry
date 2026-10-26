from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_recapture_swir import Geometry, open_cube, scan_records
from spatial_uncertainty import process_cube


def build(source_path: Path, dataset_root: Path, output_dir: Path) -> None:
    """按已有圆心逐帧提取背景，生成共用样本和几何的两套实验输入。"""
    with np.load(source_path, allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]) for key in archive.files}
    records = {record.group_id: record for record in scan_records(dataset_root)}
    groups = source["group_ids"].astype(str)
    roi = source["X_center_roi_dn"].astype(np.float32)
    y = source["y_temperature_c"].astype(np.float64)
    wavelengths = source["wavelengths_nm"].astype(np.float64)
    if roi.shape != (401, 264) or len(records) != 401 or len(set(groups)) != 401:
        raise ValueError("自然光数据必须是身份唯一的401样本、264波段")
    rows: dict[str, list[np.ndarray]] = {
        "background_median": [], "background_mad": [], "roi_uncertainty": [],
        "difference_uncertainty": [], "bad_pixel_fraction": [], "saturation_fraction": [],
    }
    max_roi_difference = 0.0
    for index, group in enumerate(groups):
        record = records[group]
        if record.temperature_c != y[index] or not np.array_equal(record.wavelengths_nm, wavelengths):
            raise ValueError(f"样本身份或波长不一致：{group}")
        geometry = Geometry(
            float(source["center_x"][index]), float(source["center_y"][index]),
            float(source["radius_px"][index]), 1.0, "复用自然光圆心",
        )
        result = process_cube(open_cube(record), geometry, include_alternative_backgrounds=False)
        difference = float(np.max(np.abs(result["roi_mean"] - roi[index])))
        max_roi_difference = max(max_roi_difference, difference)
        if difference > 1e-3:
            raise ValueError(f"{group} 的中心圆DN无法复现：{difference}")
        for key, result_key in (
            ("background_median", "background_median"),
            ("background_mad", "background_mad"),
            ("roi_uncertainty", "roi_standard_error"),
            ("difference_uncertainty", "uncertainty_annulus"),
            ("bad_pixel_fraction", "outlier_fraction_by_band"),
            ("saturation_fraction", "saturation_fraction_by_band"),
        ):
            rows[key].append(np.asarray(result[result_key], dtype=np.float32))
        if (index + 1) % 10 == 0 or index + 1 == 401:
            print(f"背景提取 {index + 1}/401", flush=True)

    stacked = {key: np.stack(values) for key, values in rows.items()}
    background = stacked["background_median"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for mode, X, uncertainty in (
        ("no_background", roi, stacked["roi_uncertainty"]),
        ("background_subtracted", roi - background, stacked["difference_uncertainty"]),
    ):
        bad_band = (~np.isfinite(X).all(axis=0)) | (np.ptp(X, axis=0) <= 1e-8)
        destination = output_dir / f"sunroom_60cm_{mode}.npz"
        np.savez_compressed(
            destination,
            X_raw_features=X.astype(np.float32),
            X_roi_dn=X.astype(np.float32),
            X_uncertainty=uncertainty,
            background_median_audit_only=background,
            background_mad=stacked["background_mad"],
            bad_pixel_fraction_by_sample_band=stacked["bad_pixel_fraction"],
            saturation_fraction_by_sample_band=stacked["saturation_fraction"],
            y_temperature_c=y,
            wavelengths_nm=wavelengths,
            group_ids=groups,
            bad_band_mask=bad_band,
            sensor_names=np.full(wavelengths.shape, "SWIR"),
            standardized=np.asarray(False),
            background_subtracted=np.asarray(mode == "background_subtracted"),
        )
        print(f"写入 {destination}", flush=True)
    summary = {
        "source": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "dataset_root": str(dataset_root),
        "sample_count": 401,
        "band_count": 264,
        "maximum_roi_reproduction_difference_dn": max_roi_difference,
        "background_rule": "同帧目标圆外1.55R至2.40R环带中位数",
        "geometry_rule": "两种输入复用同一自然光圆心、半径与样本顺序",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.dataset_root, args.output_dir)


if __name__ == "__main__":
    main()
