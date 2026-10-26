from __future__ import annotations

import numpy as np

from analyze_recapture_swir import Geometry
from spatial_uncertainty import process_cube


def test_robust_plane_removes_spatial_gradient() -> None:
    height, width, bands = 120, 140, 5
    yy, xx = np.mgrid[:height, :width]
    cube = np.empty((height, width, bands), dtype=np.uint16)
    center_x, center_y, radius = 70.0, 60.0, 20.0
    target = np.hypot(xx - center_x, yy - center_y) <= 0.4 * radius
    for band in range(bands):
        background = 1000.0 + (band + 1) * 0.6 * xx + (band + 1) * 0.4 * yy
        signal = background + target * (80.0 + 10.0 * band)
        cube[:, :, band] = np.round(signal).astype(np.uint16)
    result = process_cube(cube, Geometry(center_x, center_y, radius, 1.0, "test"))
    expected = np.asarray([80.0 + 10.0 * band for band in range(bands)])
    assert np.allclose(result["difference_plane"], expected, atol=1.5)
    assert np.all(np.asarray(result["uncertainty_plane"]) >= 0)
    assert np.asarray(result["outlier_fraction_by_band"]).shape == (bands,)
    assert np.asarray(result["saturation_fraction_by_band"]).shape == (bands,)


def test_outlier_does_not_dominate_roi_mean() -> None:
    cube = np.full((100, 100, 3), 1000, dtype=np.uint16)
    yy, xx = np.mgrid[:100, :100]
    target = np.hypot(xx - 50.0, yy - 50.0) <= 8.0
    cube[target] += 100
    cube[50, 50, :] = 65000
    result = process_cube(cube, Geometry(50.0, 50.0, 20.0, 1.0, "test"))
    assert np.allclose(result["difference_annulus"], 100.0, atol=1.0)
    assert float(result["outlier_fraction"]) > 0


def test_no_background_roi_keeps_background_and_matches_full_extraction() -> None:
    """未扣背景输入保留背景DN，快速路径与原提取结果逐项一致。"""
    cube = np.full((100, 100, 3), 1000, dtype=np.uint16)
    yy, xx = np.mgrid[:100, :100]
    cube[np.hypot(xx - 50.0, yy - 50.0) <= 8.0] += 100
    geometry = Geometry(50.0, 50.0, 20.0, 1.0, "测试")
    full = process_cube(cube, geometry)
    fast = process_cube(cube, geometry, include_alternative_backgrounds=False)
    assert np.allclose(fast["roi_mean"], 1100.0)
    assert np.allclose(fast["background_median"], 1000.0)
    assert np.allclose(fast["difference_annulus"], 100.0)
    for key in fast:
        np.testing.assert_array_equal(fast[key], full[key])

