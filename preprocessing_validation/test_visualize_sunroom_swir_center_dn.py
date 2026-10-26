from __future__ import annotations

import numpy as np
from pathlib import Path

from analyze_recapture_swir import Geometry, Record, aggregate_roi_spectrum
from visualize_sunroom_swir_center_dn import (
    aggregate_record_roi_spectrum_sequential,
    aggregate_roi_spectrum_sequential,
)


def test_center_roi_dn_does_not_subtract_background() -> None:
    """圆外背景变化不应被算法主动从圆心DN中相减。"""
    height, width, bands = 100, 100, 4
    yy, xx = np.mgrid[:height, :width]
    target = np.hypot(xx - 50.0, yy - 50.0) <= 8.0
    cube = np.full((height, width, bands), 1000, dtype=np.uint16)
    cube[target] = np.asarray([1100, 1200, 1300, 1400], dtype=np.uint16)
    spectrum, diagnostics = aggregate_roi_spectrum(
        cube,
        Geometry(50.0, 50.0, 20.0, 1.0, "test"),
        inner_fraction=0.4,
        sigma_fraction=0.75,
        outlier_sigma=8.0,
    )
    assert np.allclose(spectrum, [1100, 1200, 1300, 1400], atol=1e-5)
    assert diagnostics["saturation_fraction"] == 0.0


def test_center_roi_rejects_isolated_hot_pixel() -> None:
    cube = np.full((100, 100, 3), 1000, dtype=np.uint16)
    yy, xx = np.mgrid[:100, :100]
    target = np.hypot(xx - 50.0, yy - 50.0) <= 8.0
    cube[target] = 1500
    cube[50, 50] = 65000
    spectrum, diagnostics = aggregate_roi_spectrum(
        cube,
        Geometry(50.0, 50.0, 20.0, 1.0, "test"),
        inner_fraction=0.4,
        sigma_fraction=0.75,
        outlier_sigma=8.0,
    )
    assert np.allclose(spectrum, 1500.0, atol=1e-5)
    assert diagnostics["outlier_fraction"] > 0.0


def test_sequential_reader_matches_reference_aggregation() -> None:
    """连续BIL读取必须与既有聚合函数逐值一致。"""
    rng = np.random.default_rng(42)
    cube = rng.integers(700, 1800, size=(110, 120, 7), dtype=np.uint16)
    geometry = Geometry(61.3, 54.8, 22.0, 1.0, "test")
    expected, expected_diagnostics = aggregate_roi_spectrum(
        cube,
        geometry,
        inner_fraction=0.4,
        sigma_fraction=0.75,
        outlier_sigma=8.0,
    )
    actual, actual_diagnostics = aggregate_roi_spectrum_sequential(
        cube,
        geometry.center_x,
        geometry.center_y,
        geometry.radius,
    )
    assert np.array_equal(actual, expected)
    assert actual_diagnostics == expected_diagnostics


def test_contiguous_file_reader_matches_reference(tmp_path: Path) -> None:
    """单次连续文件读取必须与内存中的既有聚合结果一致。"""
    rng = np.random.default_rng(7)
    lines, samples, bands = 105, 116, 6
    bil = rng.integers(300, 2200, size=(lines, bands, samples), dtype=np.uint16)
    data_path = tmp_path / "synthetic.spe"
    bil.tofile(data_path)
    record = Record(
        header_path=tmp_path / "synthetic.hdr",
        data_path=data_path,
        temperature_c=200.0,
        group_id="synthetic",
        samples=samples,
        lines=lines,
        bands=bands,
        header_offset=0,
        byte_order=0,
        wavelengths_nm=np.linspace(900.0, 2500.0, bands),
        dark_current_corrected=None,
    )
    cube = bil.transpose(0, 2, 1)
    geometry = Geometry(58.2, 51.4, 21.0, 1.0, "test")
    expected, expected_diagnostics = aggregate_roi_spectrum(
        cube,
        geometry,
        inner_fraction=0.4,
        sigma_fraction=0.75,
        outlier_sigma=8.0,
    )
    actual, actual_diagnostics = aggregate_record_roi_spectrum_sequential(
        record,
        geometry.center_x,
        geometry.center_y,
        geometry.radius,
    )
    assert np.array_equal(actual, expected)
    assert actual_diagnostics == expected_diagnostics

