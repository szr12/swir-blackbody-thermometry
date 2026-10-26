from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from analyze_recapture_swir import (
    Geometry,
    build_reference,
    detect_swir_geometry,
    open_cube,
    scan_records,
    swir_proxy,
)


INNER_FRACTION = 0.40
SIGMA_FRACTION = 0.75
OUTLIER_SIGMA = 8.0
ANNULUS_INNER = 1.55
ANNULUS_OUTER = 2.40
PLANE_MAX_POINTS = 3000
PLANE_ITERATIONS = 3
RECAPTURE_RANGES = ((199, 207), (223, 255), (309, 315))


def in_ranges(temperature: float, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(low <= temperature <= high for low, high in ranges)


def legacy_quality_tier(temperature: float) -> str:
    if in_ranges(temperature, RECAPTURE_RANGES):
        return "formal_validated_recapture"
    if 150 <= temperature <= 198:
        return "formal_conditional_low_signal"
    if temperature < 150:
        return "formal_low_signal_user_included"
    return "formal_acquisition_zigzag"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _weighted_mean(values: np.ndarray, weights: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    effective_weights = weights[:, None] * keep
    denominator = effective_weights.sum(axis=0)
    mean = np.divide(
        (values * effective_weights).sum(axis=0),
        denominator,
        out=np.zeros(values.shape[1], dtype=np.float64),
        where=denominator > 0,
    )
    effective_n = np.divide(
        denominator**2,
        (effective_weights**2).sum(axis=0),
        out=np.ones_like(denominator),
        where=(effective_weights**2).sum(axis=0) > 0,
    )
    return mean, effective_n


def _robust_scale(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0)
    return 1.4826 * np.median(np.abs(values - median), axis=0)


def _fit_robust_plane(design: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对所有波段并行执行Huber迭代加权二维平面拟合。"""
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0].T
    weights = np.ones_like(values, dtype=np.float64)
    scale = np.ones(values.shape[1], dtype=np.float64)
    for _ in range(PLANE_ITERATIONS):
        residual = values - design @ coefficients.T
        center = np.median(residual, axis=0)
        scale = np.maximum(1.4826 * np.median(np.abs(residual - center), axis=0), 1e-6)
        ratio = np.abs(residual - center) / (1.345 * scale[None, :])
        weights = np.ones_like(ratio)
        large = ratio > 1.0
        weights[large] = 1.0 / ratio[large]
        normal = np.einsum("ni,nb,nj->bij", design, weights, design)
        rhs = np.einsum("ni,nb->bi", design, weights * values)
        normal += np.eye(3)[None, :, :] * 1e-9
        # 显式保留末尾单列，避免波段数恰好等于3时被NumPy解释为矩阵右端项。
        coefficients = np.linalg.solve(normal, rhs[..., None])[..., 0]
    effective_n = (weights.sum(axis=0) ** 2) / np.maximum((weights**2).sum(axis=0), 1e-12)
    return coefficients, scale, effective_n


def _sector_plane(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """用东南西北四区中位数拟合背景平面。"""
    sectors = [
        x >= np.abs(y),
        x <= -np.abs(y),
        y >= np.abs(x),
        y <= -np.abs(x),
    ]
    points, medians = [], []
    for mask in sectors:
        if not np.any(mask):
            raise ValueError("四方向背景分区为空")
        points.append([float(np.median(x[mask])), float(np.median(y[mask])), 1.0])
        medians.append(np.median(values[mask], axis=0))
    design = np.asarray(points, dtype=np.float64)
    coefficients = (np.linalg.pinv(design) @ np.asarray(medians)).T
    residual = values - np.column_stack([x, y, np.ones_like(x)]) @ coefficients.T
    return coefficients, _robust_scale(residual)


def process_cube(
    cube: np.ndarray,
    geometry: Geometry,
    *,
    include_alternative_backgrounds: bool = True,
) -> dict[str, np.ndarray | float | int]:
    """提取中心圆与环形背景；可跳过仅用于诊断的两种平面背景。"""
    outer = ANNULUS_OUTER * geometry.radius
    x0 = max(0, int(math.floor(geometry.center_x - outer)))
    x1 = min(cube.shape[1], int(math.ceil(geometry.center_x + outer)) + 1)
    y0 = max(0, int(math.floor(geometry.center_y - outer)))
    y1 = min(cube.shape[0], int(math.ceil(geometry.center_y + outer)) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = (xx - geometry.center_x) / geometry.radius
    dy = (yy - geometry.center_y) / geometry.radius
    distance = np.hypot(dx, dy)
    target_mask = distance <= INNER_FRACTION
    annulus_mask = (distance >= ANNULUS_INNER) & (distance <= ANNULUS_OUTER)
    block = np.asarray(cube[y0:y1, x0:x1, :], dtype=np.float64)
    target_values = block[target_mask]
    annulus_values = block[annulus_mask]

    target_median = np.median(target_values, axis=0)
    target_mad = np.median(np.abs(target_values - target_median), axis=0)
    target_scale = 1.4826 * target_mad
    limit = np.maximum(OUTLIER_SIGMA * target_scale, 1.0)
    sensor_max = float(np.iinfo(cube.dtype).max)
    keep = (np.abs(target_values - target_median) <= limit) & (target_values < sensor_max)
    outlier_fraction_by_band = 1.0 - keep.mean(axis=0)
    saturation_fraction_by_band = np.mean(target_values >= sensor_max, axis=0)
    gaussian = np.exp(-distance[target_mask] ** 2 / (2.0 * SIGMA_FRACTION**2))
    roi_mean, roi_effective_n = _weighted_mean(target_values, gaussian, keep)
    roi_robust_std = target_scale
    roi_standard_error = roi_robust_std / np.sqrt(np.maximum(roi_effective_n, 1.0))

    background_median = np.median(annulus_values, axis=0)
    background_mad = np.median(np.abs(annulus_values - background_median), axis=0)
    background_scale = 1.4826 * background_mad
    background_standard_error = background_scale / math.sqrt(max(annulus_values.shape[0], 1))
    difference_annulus = roi_mean - background_median
    uncertainty_annulus = np.sqrt(roi_standard_error**2 + background_standard_error**2)

    result = {
        "roi_mean": roi_mean.astype(np.float32),
        "roi_robust_std": roi_robust_std.astype(np.float32),
        "roi_standard_error": roi_standard_error.astype(np.float32),
        "background_median": background_median.astype(np.float32),
        "background_mad": background_mad.astype(np.float32),
        "difference_annulus": difference_annulus.astype(np.float32),
        "uncertainty_annulus": uncertainty_annulus.astype(np.float32),
        "outlier_fraction_by_band": outlier_fraction_by_band.astype(np.float32),
        "saturation_fraction_by_band": saturation_fraction_by_band.astype(np.float32),
        "roi_pixel_count": int(target_values.shape[0]),
        "background_pixel_count": int(annulus_values.shape[0]),
        "outlier_fraction": float(np.mean(outlier_fraction_by_band)),
        "zero_fraction": float(np.mean(target_values == 0)),
        "saturation_fraction": float(np.mean(saturation_fraction_by_band)),
    }
    if not include_alternative_backgrounds:
        return result

    annulus_x = dx[annulus_mask]
    annulus_y = dy[annulus_mask]
    if annulus_values.shape[0] > PLANE_MAX_POINTS:
        sample_index = np.linspace(0, annulus_values.shape[0] - 1, PLANE_MAX_POINTS).round().astype(int)
        fit_x, fit_y = annulus_x[sample_index], annulus_y[sample_index]
        fit_values = annulus_values[sample_index]
    else:
        fit_x, fit_y, fit_values = annulus_x, annulus_y, annulus_values
    fit_design = np.column_stack([fit_x, fit_y, np.ones_like(fit_x)])
    plane_coef, plane_scale, plane_effective_n = _fit_robust_plane(fit_design, fit_values)
    target_design = np.column_stack(
        [dx[target_mask], dy[target_mask], np.ones(int(target_mask.sum()))]
    )
    plane_background_pixels = target_design @ plane_coef.T
    plane_corrected = target_values - plane_background_pixels
    difference_plane, _ = _weighted_mean(plane_corrected, gaussian, keep)
    plane_background_roi, _ = _weighted_mean(plane_background_pixels, gaussian, keep)
    plane_background_se = plane_scale / np.sqrt(np.maximum(plane_effective_n, 1.0))
    plane_roi_scale = _robust_scale(plane_corrected)
    uncertainty_plane = np.sqrt(
        (plane_roi_scale / np.sqrt(np.maximum(roi_effective_n, 1.0))) ** 2
        + plane_background_se**2
    )

    sector_coef, sector_scale = _sector_plane(annulus_x, annulus_y, annulus_values)
    sector_background_pixels = target_design @ sector_coef.T
    sector_corrected = target_values - sector_background_pixels
    difference_sector, _ = _weighted_mean(sector_corrected, gaussian, keep)
    sector_background_roi, _ = _weighted_mean(sector_background_pixels, gaussian, keep)
    uncertainty_sector = np.sqrt(
        (_robust_scale(sector_corrected) / np.sqrt(np.maximum(roi_effective_n, 1.0))) ** 2
        + (sector_scale / 2.0) ** 2
    )

    result.update({
        "background_plane": plane_background_roi.astype(np.float32),
        "difference_plane": difference_plane.astype(np.float32),
        "uncertainty_plane": uncertainty_plane.astype(np.float32),
        "background_sector": sector_background_roi.astype(np.float32),
        "difference_sector": difference_sector.astype(np.float32),
        "uncertainty_sector": uncertainty_sector.astype(np.float32),
    })
    return result


def method_metrics(X: np.ndarray, y: np.ndarray, source_batch: np.ndarray) -> dict[str, object]:
    signal = X.mean(axis=1)
    order = np.argsort(y, kind="stable")
    step = np.diff(signal[order])
    batch_boundaries = np.flatnonzero(source_batch[order][1:] != source_batch[order][:-1])
    return {
        "nonpositive_adjacent_steps": int(np.sum(step <= 0)),
        "largest_negative_step_dn": float(step.min()),
        "median_absolute_adjacent_step_dn": float(np.median(np.abs(step))),
        "batch_boundary_jumps_dn": [float(step[index]) for index in batch_boundaries],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成带不确定度和稳健空间背景的SWIR v2产物")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_records = scan_records(args.dataset_root)
    primary = sorted(
        [record for record in all_records if "chongxin" not in record.group_id.lower()],
        key=lambda record: record.temperature_c,
    )
    if len(primary) != 401 or len({record.temperature_c for record in primary}) != 401:
        raise ValueError("SWIR主序列必须包含401个唯一温度扫描")
    reference = build_reference(primary)
    keys = [
        "roi_mean", "roi_robust_std", "roi_standard_error", "background_median",
        "background_mad", "difference_annulus", "uncertainty_annulus",
        "background_plane", "difference_plane", "uncertainty_plane",
        "background_sector", "difference_sector", "uncertainty_sector",
        "outlier_fraction_by_band", "saturation_fraction_by_band",
    ]
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    rows: list[dict[str, object]] = []
    acquisition_times = np.asarray([record.data_path.stat().st_mtime for record in primary])
    acquisition_order = np.argsort(np.argsort(acquisition_times, kind="stable"), kind="stable")

    for index, record in enumerate(primary):
        cube = open_cube(record)
        geometry = detect_swir_geometry(swir_proxy(cube, record.wavelengths_nm), reference)
        result = process_cube(cube, geometry)
        for key in keys:
            arrays[key].append(result[key])
        tier = legacy_quality_tier(record.temperature_c)
        rows.append({
            "sample_index": index,
            "temperature_c": record.temperature_c,
            "group_id": record.group_id,
            "source_batch": "recapture_replacement" if in_ranges(record.temperature_c, RECAPTURE_RANGES) else "original_main",
            "acquisition_order": int(acquisition_order[index]),
            "acquisition_timestamp_unix": float(acquisition_times[index]),
            "quality_tier": tier,
            "center_x": geometry.center_x,
            "center_y": geometry.center_y,
            "radius_px": geometry.radius,
            "roi_pixel_count": result["roi_pixel_count"],
            "background_pixel_count": result["background_pixel_count"],
            "outlier_fraction": result["outlier_fraction"],
            "zero_fraction": result["zero_fraction"],
            "saturation_fraction": result["saturation_fraction"],
            "dark_current_corrected": record.dark_current_corrected,
            "exposure_time": "unavailable",
            "gain": "unavailable",
            "detector_temperature": "unavailable",
        })
        print(f"[{index + 1:3d}/{len(primary)}] {record.group_id}", flush=True)

    stacked = {key: np.vstack(value) for key, value in arrays.items()}
    y = np.asarray([record.temperature_c for record in primary], dtype=np.float64)
    wavelengths = primary[0].wavelengths_nm.astype(np.float64)
    group_ids = np.asarray([record.group_id for record in primary])
    source_batch = np.asarray([row["source_batch"] for row in rows])
    tiers = np.asarray([row["quality_tier"] for row in rows])

    signal = stacked["difference_plane"]
    uncertainty = stacked["uncertainty_plane"]
    relative_uncertainty = np.median(
        uncertainty / np.maximum(np.abs(signal), 1.0), axis=0
    )
    background_noise = np.median(stacked["background_mad"], axis=0)
    background_center = np.median(background_noise)
    background_spread = 1.4826 * np.median(np.abs(background_noise - background_center))
    bad_nonfinite = ~np.all(np.isfinite(signal), axis=0)
    bad_constant = np.std(signal, axis=0, ddof=1) <= 1e-8
    bad_relative_uncertainty = relative_uncertainty > 0.50
    bad_background_noise = background_noise > background_center + 6.0 * max(background_spread, 1e-8)
    median_outlier_fraction = np.median(stacked["outlier_fraction_by_band"], axis=0)
    maximum_saturation_fraction = np.max(stacked["saturation_fraction_by_band"], axis=0)
    bad_pixel_fraction = median_outlier_fraction > 0.01
    bad_saturation = maximum_saturation_fraction > 1e-4
    bad_spectral_edge = (wavelengths < 900.0) | (wavelengths > 2450.0)
    # 当前数据没有独立暗场、平场和足量重复参考，相关规则仅声明为不可评估，不能擅自判坏。
    bad_dark_noise = np.zeros(signal.shape[1], dtype=bool)
    bad_flat_response = np.zeros(signal.shape[1], dtype=bool)
    bad_repeat_cv = np.zeros(signal.shape[1], dtype=bool)
    bad_band_mask = (
        bad_nonfinite | bad_constant | bad_relative_uncertainty
        | bad_background_noise | bad_pixel_fraction | bad_saturation | bad_spectral_edge
    )

    np.savez_compressed(
        args.output_dir / "swir_preprocessed_v2.npz",
        X_raw_difference=stacked["difference_annulus"],
        X_background_corrected=stacked["difference_plane"],
        X_four_sector_corrected=stacked["difference_sector"],
        X_radiometric_proxy=stacked["difference_plane"],
        X_uncertainty=uncertainty,
        roi_mean=stacked["roi_mean"],
        roi_robust_std=stacked["roi_robust_std"],
        roi_standard_error=stacked["roi_standard_error"],
        background_median=stacked["background_median"],
        background_mad=stacked["background_mad"],
        bad_pixel_fraction_by_sample_band=stacked["outlier_fraction_by_band"],
        saturation_fraction_by_sample_band=stacked["saturation_fraction_by_band"],
        difference_uncertainty_annulus=stacked["uncertainty_annulus"],
        difference_uncertainty_plane=stacked["uncertainty_plane"],
        difference_uncertainty_sector=stacked["uncertainty_sector"],
        y_temperature_c=y,
        wavelengths_nm=wavelengths,
        group_ids=group_ids,
        source_batch=source_batch,
        acquisition_order=acquisition_order,
        acquisition_timestamp_unix=acquisition_times,
        exposure_time=np.full(401, np.nan),
        gain=np.full(401, np.nan),
        detector_temperature=np.full(401, np.nan),
        quality_tier=tiers,
        formal_use_mask=np.ones(401, dtype=bool),
        bad_band_mask=bad_band_mask,
        bad_nonfinite=bad_nonfinite,
        bad_constant=bad_constant,
        bad_relative_uncertainty=bad_relative_uncertainty,
        bad_background_noise=bad_background_noise,
        bad_pixel_fraction=bad_pixel_fraction,
        bad_saturation=bad_saturation,
        bad_spectral_edge=bad_spectral_edge,
        bad_dark_noise=bad_dark_noise,
        bad_flat_response=bad_flat_response,
        bad_repeat_cv=bad_repeat_cv,
        bad_dark_noise_rule_available=np.asarray(False),
        bad_flat_response_rule_available=np.asarray(False),
        bad_repeat_cv_rule_available=np.asarray(False),
        radiometric_calibrated=np.asarray(False),
        standardized=np.asarray(False),
    )
    write_csv(args.output_dir / "swir_v2_sample_manifest.csv", rows)
    band_rows = [{
        "band_index_zero_based": index,
        "wavelength_nm": float(wavelengths[index]),
        "median_relative_uncertainty": float(relative_uncertainty[index]),
        "median_background_mad_dn": float(background_noise[index]),
        "median_bad_pixel_fraction": float(median_outlier_fraction[index]),
        "maximum_saturation_fraction": float(maximum_saturation_fraction[index]),
        "bad_nonfinite": bool(bad_nonfinite[index]),
        "bad_constant": bool(bad_constant[index]),
        "bad_relative_uncertainty": bool(bad_relative_uncertainty[index]),
        "bad_background_noise": bool(bad_background_noise[index]),
        "bad_pixel_fraction": bool(bad_pixel_fraction[index]),
        "bad_saturation": bool(bad_saturation[index]),
        "bad_spectral_edge": bool(bad_spectral_edge[index]),
        "bad_band": bool(bad_band_mask[index]),
    } for index in range(signal.shape[1])]
    write_csv(args.output_dir / "swir_v2_band_quality.csv", band_rows)

    summary = {
        "status": "preprocessing_complete",
        "X_shape": [401, 264],
        "standardized": False,
        "radiometric_calibrated": False,
        "temperature_labels_used_in_feature_generation": False,
        "temperature_labels_used_for_posthoc_qa": True,
        "acquisition_order_source": "file_mtime_proxy_not_instrument_timestamp",
        "all_dark_current_corrected": all(row["dark_current_corrected"] is True for row in rows),
        "unavailable_metadata": ["exposure_time", "gain", "detector_temperature", "flat_field", "dark_file"],
        "all_401_samples_retained": True,
        "formal_sample_count": 401,
        "all_264_bands_retained_in_arrays": True,
        "bad_band_count": int(bad_band_mask.sum()),
        "bad_band_rule_counts": {
            "nonfinite": int(bad_nonfinite.sum()),
            "constant": int(bad_constant.sum()),
            "relative_uncertainty": int(bad_relative_uncertainty.sum()),
            "background_noise": int(bad_background_noise.sum()),
            "bad_pixel_fraction": int(bad_pixel_fraction.sum()),
            "saturation": int(bad_saturation.sum()),
            "spectral_edge": int(bad_spectral_edge.sum()),
        },
        "unavailable_bad_band_rules": ["dark_noise", "flat_response", "repeat_reference_cv"],
        "methods": {
            "annulus_median": method_metrics(stacked["difference_annulus"], y, source_batch),
            "robust_plane": method_metrics(stacked["difference_plane"], y, source_batch),
            "four_sector": method_metrics(stacked["difference_sector"], y, source_batch),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    config = {
        "dataset_root": str(args.dataset_root.resolve()),
        "inner_fraction": INNER_FRACTION,
        "sigma_fraction": SIGMA_FRACTION,
        "outlier_sigma": OUTLIER_SIGMA,
        "annulus": [ANNULUS_INNER, ANNULUS_OUTER],
        "robust_plane": {"max_points": PLANE_MAX_POINTS, "iterations": PLANE_ITERATIONS, "loss": "Huber"},
        "standardization": "deferred_to_each_training_fold",
        "batch_correction": "not_applied_no_cross_batch_anchor",
    }
    (args.output_dir / "preprocessing_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
