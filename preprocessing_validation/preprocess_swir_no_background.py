from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from analyze_recapture_swir import Geometry, open_cube, scan_records
from spatial_uncertainty import process_cube


def build_no_background_data(
    dataset_root: Path, baseline_path: Path, manifest_path: Path, output_dir: Path,
) -> Path:
    """复用原中心圆几何提取未扣背景DN，并逐样本核对原差分结果。"""
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(baseline_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    records = {
        record.group_id: record for record in scan_records(dataset_root)
        if "chongxin" not in record.group_id.lower()
    }
    n_samples, n_bands = arrays["X_raw_features"].shape
    if n_samples != 401 or len(manifest) != n_samples or len(records) != n_samples:
        raise ValueError("原始扫描、中心圆清单与基线必须对应同一401样本主序列")
    roi_rows, uncertainty_rows, background_rows = [], [], []
    maximum_difference = 0.0
    for index, row in enumerate(manifest):
        group_id = str(arrays["group_ids"][index])
        if int(row["sample_index"]) != index or row["group_id"] != group_id:
            raise ValueError(f"第{index}个样本的身份或顺序与基线不一致")
        record = records[group_id]
        if record.temperature_c != float(arrays["y_temperature_c"][index]):
            raise ValueError(f"{group_id}的温度标签与基线不一致")
        if not np.array_equal(record.wavelengths_nm, arrays["wavelengths_nm"]):
            raise ValueError(f"{group_id}的波长轴与基线不一致")
        geometry = Geometry(
            float(row["center_x"]), float(row["center_y"]),
            float(row["radius_px"]), 1.0, "复用基线中心圆",
        )
        result = process_cube(
            open_cube(record), geometry, include_alternative_backgrounds=False,
        )
        # 背景仅供一致性审计；训练输入直接取中心圆均值，不执行相减。
        difference = float(np.max(np.abs(
            result["difference_annulus"] - arrays["X_raw_features"][index]
        )))
        maximum_difference = max(maximum_difference, difference)
        if difference > 1e-3:
            raise ValueError(f"{group_id}无法复现原差分DN，最大差值={difference}")
        roi_rows.append(result["roi_mean"])
        uncertainty_rows.append(result["roi_standard_error"])
        background_rows.append(result["background_median"])
        print(f"[{index + 1:3d}/{n_samples}] 未扣背景中心圆DN：{group_id}", flush=True)
    arrays["X_raw_features"] = np.vstack(roi_rows).astype(np.float32)
    arrays["X_roi_dn"] = arrays["X_raw_features"].copy()
    arrays["X_uncertainty"] = np.vstack(uncertainty_rows).astype(np.float32)
    arrays["background_median_audit_only"] = np.vstack(background_rows).astype(np.float32)
    arrays["feature_types"] = np.asarray(["roi_gaussian_dn_no_background_subtraction"] * n_bands)
    arrays["background_subtracted"] = np.asarray(False)
    arrays["bad_nonfinite"] = ~(
        np.isfinite(arrays["X_raw_features"]).all(axis=0)
        & np.isfinite(arrays["X_uncertainty"]).all(axis=0)
        & np.isfinite(arrays["background_mad"]).all(axis=0)
    )
    arrays["bad_constant"] = np.ptp(arrays["X_raw_features"], axis=0) <= 1e-8
    arrays["bad_band_mask"] = arrays["bad_nonfinite"] | arrays["bad_constant"]
    destination = output_dir / "swir_roi_no_background.npz"
    np.savez_compressed(destination, **arrays)
    summary = {
        "status": "preprocessing_complete",
        "shape": [n_samples, n_bands],
        "background_subtracted": False,
        "formal_feature": "同一中心圆的异常点剔除后高斯加权DN均值",
        "uncertainty": "仅中心圆稳健标准误",
        "background_fields": "仅保留审计元数据，不参与冻结模型的输入或权重",
        "temperature_labels_used_in_feature_generation": False,
        "baseline_difference_max_absolute_dn": maximum_difference,
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "geometry_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "dataset_root": str(dataset_root.resolve()),
        "baseline_path": str(baseline_path.resolve()),
        "static_invalid_band_count": int(arrays["bad_band_mask"].sum()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="提取与原实验几何一致的未扣背景中心圆DN")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(build_no_background_data(
        args.dataset_root, args.baseline, args.manifest, args.output_dir,
    ), flush=True)


if __name__ == "__main__":
    main()
