from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr


NAME_PATTERN = re.compile(
    r"^(?P<temperature>[+-]?\d+(?:\.\d+)?)_(?P<tag>.*?)FS[^()]+\(900-2500\)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Geometry:
    center_x: float
    center_y: float
    radius: float
    score: float
    method: str


@dataclass(frozen=True)
class Record:
    header_path: Path
    data_path: Path
    temperature_c: float
    group_id: str
    samples: int
    lines: int
    bands: int
    header_offset: int
    byte_order: int
    wavelengths_nm: np.ndarray
    dark_current_corrected: bool | None


def gaussian_circle_weights(
    height: int,
    width: int,
    center_x: float,
    center_y: float,
    radius: float,
    *,
    inner_fraction: float,
    sigma_fraction: float,
) -> tuple[slice, slice, np.ndarray]:
    """构造圆形硬掩膜内的高斯空间权重。"""
    inner_radius = radius * inner_fraction
    x0 = max(0, int(math.floor(center_x - inner_radius)))
    x1 = min(width, int(math.ceil(center_x + inner_radius)) + 1)
    y0 = max(0, int(math.floor(center_y - inner_radius)))
    y1 = min(height, int(math.ceil(center_y + inner_radius)) + 1)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("ROI 不与立方体相交")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
    sigma = max(radius * sigma_fraction, 1e-6)
    weights = np.exp(-distance_sq / (2.0 * sigma**2))
    weights[distance_sq > inner_radius**2] = 0.0
    if not np.any(weights > 0):
        raise ValueError("ROI 权重为空")
    return slice(y0, y1), slice(x0, x1), weights


def aggregate_roi_spectrum(
    cube: np.ndarray,
    geometry: Geometry,
    *,
    inner_fraction: float = 0.8,
    sigma_fraction: float = 0.75,
    outlier_sigma: float = 8.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """提取坏点稳健剔除后的圆内高斯加权光谱。"""
    ys, xs, weight_map = gaussian_circle_weights(
        cube.shape[0],
        cube.shape[1],
        geometry.center_x,
        geometry.center_y,
        geometry.radius,
        inner_fraction=inner_fraction,
        sigma_fraction=sigma_fraction,
    )
    block = np.asarray(cube[ys, xs, :], dtype=np.float64)
    weights = weight_map.reshape(-1, 1)
    values = block.reshape(-1, block.shape[-1])
    inside = weights[:, 0] > 0
    values = values[inside]
    weights = weights[inside]

    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    robust_sigma = 1.4826 * mad
    limit = np.maximum(outlier_sigma * robust_sigma, 1.0)
    keep = np.abs(values - median) <= limit
    sensor_max = float(np.iinfo(cube.dtype).max)
    keep &= np.isfinite(values) & (values < sensor_max)
    weighted_keep = weights * keep
    denominator = weighted_keep.sum(axis=0)
    fallback = denominator <= 0
    numerator = (values * weighted_keep).sum(axis=0)
    spectrum = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=~fallback)
    if np.any(fallback):
        raw_denominator = weights.sum(axis=0)
        spectrum[fallback] = ((values * weights).sum(axis=0) / raw_denominator)[fallback]

    diagnostics = {
        "roi_pixel_count": int(values.shape[0]),
        "outlier_fraction": float(1.0 - keep.mean()),
        "zero_fraction": float(np.mean(values == 0)),
        "saturation_fraction": float(np.mean(values >= sensor_max)),
    }
    return spectrum.astype(np.float32), diagnostics


def swir_proxy(cube: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    """使用长波端多波段平均图增强黑体圆对比度。"""
    indices = np.where((wavelengths_nm >= 2100.0) & (wavelengths_nm <= 2450.0))[0]
    if indices.size < 5:
        indices = np.arange(max(0, cube.shape[-1] - 48), cube.shape[-1])
    step = max(1, indices.size // 16)
    return np.asarray(cube[:, :, indices[::step]], dtype=np.float32).mean(axis=2)


def detect_swir_geometry(proxy: np.ndarray, prior: Geometry | None = None) -> Geometry:
    """逐扫描定位 SWIR 黑体圆，并保持物理半径稳定。"""
    smooth = ndimage.gaussian_filter(np.asarray(proxy, dtype=np.float64), sigma=2.5)
    height, width = smooth.shape
    has_reference = prior is not None
    if prior is None:
        prior = Geometry(0.61 * width, 0.48 * height, 0.075 * width, 0.0, "粗略先验")
    x_margin = int(max(100, round(prior.radius * 2.6)))
    x0 = max(0, int(round(prior.center_x)) - x_margin)
    x1 = min(width, int(round(prior.center_x)) + x_margin + 1)
    y0 = max(0, int(math.floor(0.15 * height)))
    y1 = min(height, int(math.ceil(0.75 * height)))
    local = smooth[y0:y1, x0:x1]
    background = float(np.percentile(local, 35.0))
    peak = float(np.percentile(local, 99.8))
    contrast = peak - background
    if contrast <= 0:
        raise ValueError("SWIR 图像没有正对比度")
    binary = local >= background + 0.50 * contrast
    labels, count = ndimage.label(binary)
    if count == 0:
        raise ValueError("未检测到 SWIR 黑体圆")
    peak_position = np.unravel_index(np.argmax(local), local.shape)
    label_id = int(labels[peak_position])
    if label_id == 0:
        sizes = ndimage.sum(binary, labels, np.arange(1, count + 1))
        label_id = int(np.argmax(sizes) + 1)
    component = labels == label_id
    area = int(component.sum())
    if area < 100:
        raise ValueError(f"SWIR 圆连通区域过小：{area}px")
    yy, xx = np.nonzero(component)
    signal_weights = np.maximum(local[component] - background, 1e-6)
    center_x = x0 + float(np.average(xx, weights=signal_weights))
    center_y = y0 + float(np.average(yy, weights=signal_weights))
    measured_radius = math.sqrt(area / math.pi)
    radius = prior.radius if has_reference else measured_radius
    score = contrast / max(abs(background), 1.0)
    return Geometry(center_x, center_y, radius, float(score), "长波半峰连通区域")


def parse_header(path: Path) -> dict[str, str]:
    """完整读取可能跨行的 ENVI 头字段。"""
    lines = path.read_text(encoding="utf-8-sig", errors="strict").splitlines()
    values: dict[str, str] = {}
    index = 1
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        while value.count("{") > value.count("}") and index < len(lines):
            value += " " + lines[index].strip()
            index += 1
        values[key.lower()] = value
    return values


def parse_wavelengths(value: str) -> np.ndarray:
    """将 ENVI 波长列表转换为数组。"""
    text = value.strip().strip("{}")
    return np.asarray([float(item.strip()) for item in text.split(",") if item.strip()])


def scan_records(root: Path) -> list[Record]:
    """扫描新命名格式中的 SWIR 记录。"""
    records: list[Record] = []
    for header_path in sorted(root.glob("*.hdr"), key=lambda item: item.name.lower()):
        match = NAME_PATTERN.match(header_path.stem)
        if match is None:
            continue
        metadata = parse_header(header_path)
        dark_text = metadata.get("deducting dark current", "").strip().lower()
        dark_state = True if dark_text == "true" else False if dark_text == "false" else None
        record = Record(
            header_path=header_path,
            data_path=header_path.with_suffix(".spe"),
            temperature_c=float(match.group("temperature")),
            group_id=header_path.stem,
            samples=int(metadata["samples"]),
            lines=int(metadata["lines"]),
            bands=int(metadata["bands"]),
            header_offset=int(metadata.get("header offset", "0")),
            byte_order=int(metadata.get("byte order", "0")),
            wavelengths_nm=parse_wavelengths(metadata["wavelength"]),
            dark_current_corrected=dark_state,
        )
        expected = record.header_offset + record.lines * record.samples * record.bands * 2
        if record.data_path.stat().st_size != expected:
            raise ValueError(f"SPE 文件大小不匹配: {record.data_path}")
        records.append(record)
    if not records:
        raise ValueError(f"没有找到 SWIR 数据: {root}")
    return records


def open_cube(record: Record) -> np.ndarray:
    """以只读内存映射打开 BIL 数据，并返回 lines-samples-bands 视图。"""
    dtype = np.dtype("<u2" if record.byte_order == 0 else ">u2")
    raw = np.memmap(record.data_path, dtype=dtype, mode="r", offset=record.header_offset)
    return raw.reshape(record.lines, record.bands, record.samples).transpose(0, 2, 1)


def build_reference(records: list[Record]) -> Geometry:
    """使用中等温区建立稳定半径，避免高温光晕放大圆轮廓。"""
    ordered = sorted(records, key=lambda item: (item.temperature_c, item.group_id))
    selected = [record for record in ordered if 250.0 <= record.temperature_c <= 315.0]
    if len(selected) < 3:
        selected = ordered[-min(12, len(ordered)) :]
    elif len(selected) > 15:
        indices = np.linspace(0, len(selected) - 1, 15).round().astype(int)
        selected = [selected[index] for index in indices]
    detections: list[Geometry] = []
    for record in selected:
        cube = open_cube(record)
        detections.append(detect_swir_geometry(swir_proxy(cube, record.wavelengths_nm)))
    return Geometry(
        center_x=float(np.median([item.center_x for item in detections])),
        center_y=float(np.median([item.center_y for item in detections])),
        radius=float(np.median([item.radius for item in detections])),
        score=float(np.median([item.score for item in detections])),
        method="中等温区样本中位参考",
    )


def annulus_spectrum(cube: np.ndarray, geometry: Geometry) -> np.ndarray:
    """提取目标周围同帧背景，避开目标光晕和图像边界。"""
    outer = 2.4 * geometry.radius
    inner = 1.55 * geometry.radius
    x0 = max(0, int(math.floor(geometry.center_x - outer)))
    x1 = min(cube.shape[1], int(math.ceil(geometry.center_x + outer)) + 1)
    y0 = max(0, int(math.floor(geometry.center_y - outer)))
    y1 = min(cube.shape[0], int(math.ceil(geometry.center_y + outer)) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance = np.hypot(xx - geometry.center_x, yy - geometry.center_y)
    mask = (distance >= inner) & (distance <= outer)
    values = np.asarray(cube[y0:y1, x0:x1, :], dtype=np.float64)[mask]
    return np.median(values, axis=0)


def analyze(records: list[Record], reference: Geometry) -> tuple[list[dict[str, object]], np.ndarray]:
    """逐扫描提取目标、背景以及差分光谱。"""
    rows: list[dict[str, object]] = []
    difference_spectra: list[np.ndarray] = []
    for index, record in enumerate(sorted(records, key=lambda item: (item.temperature_c, item.group_id))):
        cube = open_cube(record)
        proxy = swir_proxy(cube, record.wavelengths_nm)
        geometry = detect_swir_geometry(proxy, reference)
        target, diagnostic = aggregate_roi_spectrum(cube, geometry)
        background = annulus_spectrum(cube, geometry)
        difference = target.astype(np.float64) - background
        difference_spectra.append(difference)
        rows.append(
            {
                "temperature_c": record.temperature_c,
                "group_id": record.group_id,
                "dark_current_corrected": record.dark_current_corrected,
                "center_x": geometry.center_x,
                "center_y": geometry.center_y,
                "radius": geometry.radius,
                "target_mean_dn": float(np.mean(target)),
                "background_mean_dn": float(np.mean(background)),
                "difference_mean_dn": float(np.mean(difference)),
                "difference_median_dn": float(np.median(difference)),
                "zero_fraction": diagnostic["zero_fraction"],
                "saturation_fraction": diagnostic["saturation_fraction"],
            }
        )
        print(f"[{index + 1:3d}/{len(records)}] {record.group_id}", flush=True)
    return rows, np.vstack(difference_spectra)


def summarize(rows: list[dict[str, object]], spectra: np.ndarray) -> dict[str, object]:
    """汇总差分光谱的单调性、异常跳变和传感器质量。"""
    temperatures = np.asarray([float(row["temperature_c"]) for row in rows])
    signal = np.asarray([float(row["difference_mean_dn"]) for row in rows])
    order = np.argsort(temperatures, kind="stable")
    temperatures = temperatures[order]
    signal = signal[order]
    step = np.diff(signal)
    duplicate_mask = np.diff(temperatures) == 0
    valid_step = step[~duplicate_mask]
    correlations = np.asarray(
        [spearmanr(spectra[:, index], np.asarray([float(row["temperature_c"]) for row in rows])).statistic
         for index in range(spectra.shape[1])]
    )
    return {
        "record_count": len(rows),
        "temperature_min": float(np.min(temperatures)),
        "temperature_max": float(np.max(temperatures)),
        "all_dark_current_corrected": all(row["dark_current_corrected"] is True for row in rows),
        "max_saturation_fraction": float(max(float(row["saturation_fraction"]) for row in rows)),
        "median_zero_fraction": float(np.median([float(row["zero_fraction"]) for row in rows])),
        "difference_mean_dn_min": float(np.min(signal)),
        "difference_mean_dn_max": float(np.max(signal)),
        "nonpositive_temperature_steps": int(np.sum(valid_step <= 0)),
        "temperature_steps": int(valid_step.size),
        "largest_negative_step_dn": float(np.min(valid_step)) if valid_step.size else None,
        "max_abs_band_spearman": float(np.nanmax(np.abs(correlations))),
        "bands_abs_spearman_ge_0_8": int(np.sum(np.abs(correlations) >= 0.8)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """写出逐扫描诊断表。"""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="比较主数据和 SWIR 重采集数据的预处理质量")
    parser.add_argument("main_root", type=Path)
    parser.add_argument("recapture_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    recapture_records = scan_records(args.recapture_root)
    recapture_temperatures = {record.temperature_c for record in recapture_records}
    main_records = [
        record for record in scan_records(args.main_root) if record.temperature_c in recapture_temperatures
    ]

    recapture_reference = build_reference(recapture_records)
    main_reference = build_reference(main_records)
    main_rows, main_spectra = analyze(main_records, main_reference)
    recapture_rows, recapture_spectra = analyze(recapture_records, recapture_reference)
    write_csv(args.output_dir / "main_overlap_swir.csv", main_rows)
    write_csv(args.output_dir / "recapture_swir.csv", recapture_rows)
    np.savez_compressed(
        args.output_dir / "recapture_comparison_swir.npz",
        main_difference=main_spectra,
        recapture_difference=recapture_spectra,
    )
    summary = {
        "main_overlap": summarize(main_rows, main_spectra),
        "recapture": summarize(recapture_rows, recapture_spectra),
        "main_reference": main_reference.__dict__,
        "recapture_reference": recapture_reference.__dict__,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
