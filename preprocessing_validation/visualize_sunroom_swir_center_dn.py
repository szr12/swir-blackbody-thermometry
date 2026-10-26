from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from analyze_recapture_swir import (
    Record,
    build_reference,
    detect_swir_geometry,
    gaussian_circle_weights,
    open_cube,
    scan_records,
    swir_proxy,
)


INNER_FRACTION = 0.40
SIGMA_FRACTION = 0.75
OUTLIER_SIGMA = 8.0
MIN_CENTER_DETECTION_SCORE = 0.10
QC_TEMPERATURES = (70.0, 250.0, 450.0)
SPECTRUM_TEMPERATURES = (50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0)


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def save_figure(fig: mpl.figure.Figure, destination: Path) -> list[Path]:
    """按统一规格导出科研图，不改变图中的数据。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix, dpi in ((".png", 300), (".pdf", 300), (".svg", 300), (".tiff", 600)):
        path = destination.with_suffix(suffix)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(fig)
    return outputs


def aggregate_roi_spectrum_sequential(
    cube: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """按原始BIL存储顺序连续读取ROI所在行，避免机械盘随机寻道。"""
    ys, xs, weight_map = gaussian_circle_weights(
        cube.shape[0],
        cube.shape[1],
        center_x,
        center_y,
        radius,
        inner_fraction=INNER_FRACTION,
        sigma_fraction=SIGMA_FRACTION,
    )
    # cube是lines-samples-bands视图；转回lines-bands-samples后按整行连续读取。
    bil_view = cube.transpose(0, 2, 1)
    bil_rows = np.array(bil_view[ys, :, :], copy=True)
    block = bil_rows[:, :, xs].transpose(0, 2, 1).astype(np.float64, copy=False)
    weights = weight_map.reshape(-1, 1)
    values = block.reshape(-1, block.shape[-1])
    inside = weights[:, 0] > 0
    values = values[inside]
    weights = weights[inside]

    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    robust_sigma = 1.4826 * mad
    limit = np.maximum(OUTLIER_SIGMA * robust_sigma, 1.0)
    keep = np.abs(values - median) <= limit
    sensor_max = float(np.iinfo(cube.dtype).max)
    keep &= np.isfinite(values) & (values < sensor_max)
    weighted_keep = weights * keep
    denominator = weighted_keep.sum(axis=0)
    numerator = (values * weighted_keep).sum(axis=0)
    spectrum = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    fallback = denominator <= 0
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


def aggregate_record_roi_spectrum_sequential(
    record: Record,
    center_x: float,
    center_y: float,
    radius: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """以一次连续文件读取载入ROI所在BIL行块，再执行相同的圆心聚合。"""
    ys, _, _ = gaussian_circle_weights(
        record.lines,
        record.samples,
        center_x,
        center_y,
        radius,
        inner_fraction=INNER_FRACTION,
        sigma_fraction=SIGMA_FRACTION,
    )
    dtype = np.dtype("<u2" if record.byte_order == 0 else ">u2")
    row_values = record.bands * record.samples
    row_count = ys.stop - ys.start
    offset = record.header_offset + ys.start * row_values * dtype.itemsize
    with record.data_path.open("rb") as handle:
        handle.seek(offset)
        flat = np.fromfile(handle, dtype=dtype, count=row_count * row_values)
    if flat.size != row_count * row_values:
        raise ValueError(f"{record.group_id}的ROI行块读取不完整")
    bil_rows = flat.reshape(row_count, record.bands, record.samples)
    cube_rows = bil_rows.transpose(0, 2, 1)
    # 使用局部行坐标调用同一聚合函数，避免重新实现数值逻辑。
    return aggregate_roi_spectrum_sequential(
        cube_rows,
        center_x,
        center_y - ys.start,
        radius,
    )


def plot_heatmap(
    temperatures: np.ndarray,
    wavelengths: np.ndarray,
    spectra: np.ndarray,
    destination: Path,
) -> list[Path]:
    """展示全部温度和全部SWIR波段的圆心原始DN。"""
    display_max = float(np.percentile(spectra, 99.5))
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    image = ax.pcolormesh(
        temperatures,
        wavelengths,
        spectra.T,
        shading="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=display_max,
        rasterized=True,
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(f"Raw center-ROI DN (display capped at P99.5 = {display_max:.0f})")
    ax.set_xlabel("Blackbody temperature (°C)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title("Sunroom 60 cm · SWIR center-ROI DN · no background subtraction")
    return save_figure(fig, destination)


def plot_mean_curve(
    temperatures: np.ndarray,
    spectra: np.ndarray,
    destination: Path,
) -> list[Path]:
    """展示每个扫描在264个波段上的圆心原始DN均值。"""
    mean_dn = spectra.mean(axis=1)
    fig, ax = plt.subplots(figsize=(7.2, 3.9), layout="constrained")
    ax.plot(temperatures, mean_dn, color="#176b87", linewidth=1.15)
    ax.scatter(temperatures, mean_dn, color="#176b87", s=5, linewidths=0)
    ax.set_xlabel("Blackbody temperature (°C)")
    ax.set_ylabel("Mean raw DN across 264 SWIR bands")
    ax.set_xlim(float(temperatures.min()), float(temperatures.max()))
    ax.grid(axis="y", color="#e3e8eb", linewidth=0.6)
    ax.set_title("Sunroom 60 cm · mean SWIR center-ROI DN · no background subtraction")
    return save_figure(fig, destination)


def plot_selected_spectra(
    temperatures: np.ndarray,
    wavelengths: np.ndarray,
    spectra: np.ndarray,
    destination: Path,
) -> list[Path]:
    """展示固定温度节点的圆心原始光谱，不沿温度轴平滑。"""
    fig, ax = plt.subplots(figsize=(7.2, 4.3), layout="constrained")
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(SPECTRUM_TEMPERATURES)))
    for requested, color in zip(SPECTRUM_TEMPERATURES, colors):
        index = int(np.argmin(np.abs(temperatures - requested)))
        ax.plot(
            wavelengths,
            spectra[index],
            color=color,
            linewidth=1.0,
            label=f"{temperatures[index]:.0f} °C",
        )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Raw center-ROI DN")
    ax.grid(axis="y", color="#e3e8eb", linewidth=0.5)
    ax.legend(ncol=3, loc="upper left")
    ax.set_title("Sunroom 60 cm · selected raw SWIR spectra · no background subtraction")
    return save_figure(fig, destination)


def plot_geometry_qc(
    proxy: np.ndarray,
    temperature: float,
    center_x: float,
    center_y: float,
    radius: float,
    destination: Path,
) -> list[Path]:
    """在原始长波代理图上标出检测圆和实际0.4R取值区域。"""
    low, high = np.percentile(proxy, [1.0, 99.5])
    fig, ax = plt.subplots(figsize=(5.0, 4.2), layout="constrained")
    image = ax.imshow(proxy, origin="upper", cmap="gray", vmin=low, vmax=high)
    ax.add_patch(Circle((center_x, center_y), radius, fill=False, color="#56b4e9", linewidth=1.3))
    ax.add_patch(Circle(
        (center_x, center_y), radius * INNER_FRACTION,
        fill=False, color="#f0e442", linewidth=1.5,
    ))
    ax.scatter([center_x], [center_y], marker="+", s=45, color="#d55e00", linewidth=1.2)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Raw 2100–2450 nm proxy DN")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Line index")
    ax.set_title(f"{temperature:.0f} °C · detected circle (blue) and 0.4R ROI (yellow)")
    return save_figure(fig, destination)


def write_matrix_csv(
    path: Path,
    temperatures: np.ndarray,
    wavelengths: np.ndarray,
    spectra: np.ndarray,
) -> None:
    """写出可追溯的温度×波段原始DN矩阵。"""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["temperature_c", *[f"{value:.6f}_nm" for value in wavelengths]])
        for temperature, spectrum in zip(temperatures, spectra):
            writer.writerow([f"{temperature:.6f}", *[f"{value:.6f}" for value in spectrum]])


def run(dataset_root: Path, output_dir: Path) -> Path:
    """处理日光室60 cm的401个SWIR扫描并生成未扣背景可视化。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    records = sorted(scan_records(dataset_root), key=lambda item: (item.temperature_c, item.group_id))
    if len(records) != 401 or len({record.temperature_c for record in records}) != 401:
        raise ValueError("日光室SWIR数据必须包含50–450 ℃的401个唯一温度")
    expected_temperatures = np.arange(50.0, 451.0)
    temperatures = np.asarray([record.temperature_c for record in records], dtype=np.float64)
    if not np.array_equal(temperatures, expected_temperatures):
        raise ValueError("日光室SWIR温度序列不是完整的50–450 ℃逐1 ℃采样")
    wavelengths = records[0].wavelengths_nm.astype(np.float64)
    if wavelengths.shape != (264,):
        raise ValueError("日光室SWIR波段数不是264")

    reference = build_reference(records)
    spectra: list[np.ndarray] = []
    centers_x: list[float] = []
    centers_y: list[float] = []
    radii: list[float] = []
    scores: list[float] = []
    zero_fractions: list[float] = []
    saturation_fractions: list[float] = []
    outlier_fractions: list[float] = []
    used_reference_centers: list[bool] = []
    qc_payload: dict[float, tuple[np.ndarray, float, float, float]] = {}

    for index, record in enumerate(records):
        if not np.array_equal(record.wavelengths_nm, wavelengths):
            raise ValueError(f"{record.group_id}的波长轴与首个样本不一致")
        cube = open_cube(record)
        proxy = swir_proxy(cube, wavelengths)
        detected_geometry = detect_swir_geometry(proxy, reference)
        use_reference = detected_geometry.score < MIN_CENTER_DETECTION_SCORE
        geometry = reference if use_reference else detected_geometry
        spectrum, diagnostics = aggregate_record_roi_spectrum_sequential(
            record,
            geometry.center_x,
            geometry.center_y,
            geometry.radius,
        )
        spectra.append(spectrum)
        centers_x.append(geometry.center_x)
        centers_y.append(geometry.center_y)
        radii.append(geometry.radius)
        scores.append(detected_geometry.score)
        zero_fractions.append(diagnostics["zero_fraction"])
        saturation_fractions.append(diagnostics["saturation_fraction"])
        outlier_fractions.append(diagnostics["outlier_fraction"])
        used_reference_centers.append(use_reference)
        if record.temperature_c in QC_TEMPERATURES:
            qc_payload[record.temperature_c] = (
                np.asarray(proxy, dtype=np.float32),
                geometry.center_x,
                geometry.center_y,
                geometry.radius,
            )
        print(f"[{index + 1:3d}/401] {record.group_id}", flush=True)

    X = np.vstack(spectra).astype(np.float32)
    if X.shape != (401, 264) or not np.isfinite(X).all():
        raise ValueError("圆心DN矩阵的形状或数值异常")
    output_npz = output_dir / "sunroom_60cm_swir_center_dn_no_background.npz"
    np.savez_compressed(
        output_npz,
        X_center_roi_dn=X,
        y_temperature_c=temperatures,
        wavelengths_nm=wavelengths,
        group_ids=np.asarray([record.group_id for record in records]),
        center_x=np.asarray(centers_x),
        center_y=np.asarray(centers_y),
        radius_px=np.asarray(radii),
        detection_score=np.asarray(scores),
        zero_fraction=np.asarray(zero_fractions),
        saturation_fraction=np.asarray(saturation_fractions),
        outlier_fraction=np.asarray(outlier_fractions),
        used_reference_center=np.asarray(used_reference_centers),
        sensor=np.asarray("SWIR_900_2500_nm"),
        inner_fraction=np.asarray(INNER_FRACTION),
        sigma_fraction=np.asarray(SIGMA_FRACTION),
        background_subtracted=np.asarray(False),
        standardized=np.asarray(False),
    )
    write_matrix_csv(output_dir / "sunroom_60cm_swir_center_dn_no_background.csv", temperatures, wavelengths, X)

    exported: list[Path] = []
    exported.extend(plot_heatmap(temperatures, wavelengths, X, figure_dir / "01_center_dn_heatmap"))
    exported.extend(plot_mean_curve(temperatures, X, figure_dir / "02_center_mean_dn_vs_temperature"))
    exported.extend(plot_selected_spectra(
        temperatures, wavelengths, X, figure_dir / "03_selected_temperature_spectra",
    ))
    for temperature in QC_TEMPERATURES:
        proxy, center_x, center_y, radius = qc_payload[temperature]
        exported.extend(plot_geometry_qc(
            proxy,
            temperature,
            center_x,
            center_y,
            radius,
            figure_dir / f"04_geometry_qc_{int(temperature)}c",
        ))

    summary = {
        "status": "complete",
        "dataset_root": str(dataset_root.resolve()),
        "sample_count": 401,
        "band_count": 264,
        "temperature_range_c": [50.0, 450.0],
        "sensor": "SWIR 900–2500 nm",
        "background_subtracted": False,
        "standardized": False,
        "spatial_aggregation": {
            "inner_fraction": INNER_FRACTION,
            "sigma_fraction": SIGMA_FRACTION,
            "outlier_sigma": OUTLIER_SIGMA,
        },
        "center_policy": {
            "minimum_detection_score": MIN_CENTER_DETECTION_SCORE,
            "low_confidence_action": "use_medium_temperature_reference_center",
            "reference_fallback_count": int(np.sum(used_reference_centers)),
        },
        "center_x_range": [float(np.min(centers_x)), float(np.max(centers_x))],
        "center_y_range": [float(np.min(centers_y)), float(np.max(centers_y))],
        "radius_px": float(np.median(radii)),
        "maximum_zero_fraction": float(np.max(zero_fractions)),
        "maximum_saturation_fraction": float(np.max(saturation_fractions)),
        "maximum_outlier_fraction": float(np.max(outlier_fractions)),
        "dn_min": float(np.min(X)),
        "dn_max": float(np.max(X)),
        "dn_median": float(np.median(X)),
        "npz_sha256": hashlib.sha256(output_npz.read_bytes()).hexdigest(),
        "figure_files": [str(path.relative_to(output_dir)) for path in exported],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_npz


def repair_low_confidence_centers(dataset_root: Path, output_dir: Path) -> Path:
    """仅重算低置信度帧，并用可靠中温样本恢复稳定参考圆心。"""
    output_npz = output_dir / "sunroom_60cm_swir_center_dn_no_background.npz"
    if not output_npz.is_file():
        raise FileNotFoundError(f"缺少可修复的NPZ：{output_npz}")
    records = sorted(scan_records(dataset_root), key=lambda item: (item.temperature_c, item.group_id))
    if len(records) != 401:
        raise ValueError("修复时仍要求存在401个SWIR样本")
    with np.load(output_npz, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]).copy() for key in archive.files}

    temperatures = payload["y_temperature_c"].astype(np.float64)
    wavelengths = payload["wavelengths_nm"].astype(np.float64)
    X = payload["X_center_roi_dn"].astype(np.float32)
    scores = payload["detection_score"].astype(np.float64)
    centers_x = payload["center_x"].astype(np.float64)
    centers_y = payload["center_y"].astype(np.float64)
    radii = payload["radius_px"].astype(np.float64)
    zero_fractions = payload["zero_fraction"].astype(np.float64)
    saturation_fractions = payload["saturation_fraction"].astype(np.float64)
    outlier_fractions = payload["outlier_fraction"].astype(np.float64)
    reliable_reference = (
        (temperatures >= 250.0)
        & (temperatures <= 315.0)
        & (scores >= MIN_CENTER_DETECTION_SCORE)
    )
    if int(np.sum(reliable_reference)) < 3:
        raise ValueError("没有足够的可靠中温样本来恢复参考圆心")
    reference_x = float(np.median(centers_x[reliable_reference]))
    reference_y = float(np.median(centers_y[reliable_reference]))
    reference_radius = float(np.median(radii[reliable_reference]))
    fallback_mask = scores < MIN_CENTER_DETECTION_SCORE

    for index in np.flatnonzero(fallback_mask):
        record = records[int(index)]
        spectrum, diagnostics = aggregate_record_roi_spectrum_sequential(
            record,
            reference_x,
            reference_y,
            reference_radius,
        )
        X[index] = spectrum
        centers_x[index] = reference_x
        centers_y[index] = reference_y
        radii[index] = reference_radius
        zero_fractions[index] = diagnostics["zero_fraction"]
        saturation_fractions[index] = diagnostics["saturation_fraction"]
        outlier_fractions[index] = diagnostics["outlier_fraction"]
        print(f"[修复 {int(index) + 1:3d}/401] {record.group_id}", flush=True)

    payload.update({
        "X_center_roi_dn": X,
        "center_x": centers_x,
        "center_y": centers_y,
        "radius_px": radii,
        "zero_fraction": zero_fractions,
        "saturation_fraction": saturation_fractions,
        "outlier_fraction": outlier_fractions,
        "used_reference_center": fallback_mask,
    })
    np.savez_compressed(output_npz, **payload)
    write_matrix_csv(
        output_dir / "sunroom_60cm_swir_center_dn_no_background.csv",
        temperatures,
        wavelengths,
        X,
    )

    figure_dir = output_dir / "figures"
    exported: list[Path] = []
    exported.extend(plot_heatmap(temperatures, wavelengths, X, figure_dir / "01_center_dn_heatmap"))
    exported.extend(plot_mean_curve(temperatures, X, figure_dir / "02_center_mean_dn_vs_temperature"))
    exported.extend(plot_selected_spectra(
        temperatures, wavelengths, X, figure_dir / "03_selected_temperature_spectra",
    ))
    record_by_temperature = {record.temperature_c: record for record in records}
    for temperature in QC_TEMPERATURES:
        index = int(np.flatnonzero(temperatures == temperature)[0])
        record = record_by_temperature[temperature]
        proxy = swir_proxy(open_cube(record), wavelengths)
        exported.extend(plot_geometry_qc(
            proxy,
            temperature,
            float(centers_x[index]),
            float(centers_y[index]),
            float(radii[index]),
            figure_dir / f"04_geometry_qc_{int(temperature)}c",
        ))

    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({
        "center_policy": {
            "minimum_detection_score": MIN_CENTER_DETECTION_SCORE,
            "low_confidence_action": "use_medium_temperature_reference_center",
            "reference_fallback_count": int(np.sum(fallback_mask)),
            "reference_center_xy": [reference_x, reference_y],
        },
        "center_x_range": [float(np.min(centers_x)), float(np.max(centers_x))],
        "center_y_range": [float(np.min(centers_y)), float(np.max(centers_y))],
        "maximum_zero_fraction": float(np.max(zero_fractions)),
        "maximum_saturation_fraction": float(np.max(saturation_fractions)),
        "maximum_outlier_fraction": float(np.max(outlier_fractions)),
        "dn_min": float(np.min(X)),
        "dn_max": float(np.max(X)),
        "dn_median": float(np.median(X)),
        "npz_sha256": hashlib.sha256(output_npz.read_bytes()).hexdigest(),
        "figure_files": [str(path.relative_to(output_dir)) for path in exported],
    })
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化日光室60 cm的SWIR圆心原始DN")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repair-low-confidence",
        action="store_true",
        help="复用现有NPZ，仅重算检测分数低于阈值的圆心光谱",
    )
    args = parser.parse_args()
    if args.repair_low_confidence:
        print(repair_low_confidence_centers(args.dataset_root, args.output_dir), flush=True)
    else:
        print(run(args.dataset_root, args.output_dir), flush=True)


if __name__ == "__main__":
    main()

