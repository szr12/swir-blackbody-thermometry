from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

import preprocess_swir_no_background as preprocessing


@pytest.mark.parametrize("audit_mismatch", [False, True])
def test_no_background_archive_contains_roi_dn_and_checks_baseline(
    tmp_path, monkeypatch, audit_mismatch,
) -> None:
    """核实训练字段保存原ROI，并拒绝与原数据不一致的提取结果。"""
    y = np.arange(50, 451, dtype=np.float64)
    wavelengths = np.asarray([858.45, 2359.37])
    groups = np.asarray([f"sample_{index}" for index in range(401)])
    baseline_features = np.repeat(np.arange(100, 501)[:, None], 2, axis=1).astype(np.float32)
    baseline_path = tmp_path / "baseline.npz"
    np.savez_compressed(
        baseline_path, X_raw_features=baseline_features,
        group_ids=groups, y_temperature_c=y, wavelengths_nm=wavelengths,
        background_mad=np.ones_like(baseline_features),
    )
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "group_id", "center_x", "center_y", "radius_px"])
        for index, group in enumerate(groups):
            writer.writerow([index, group, 50, 50, 20])
    records = [
        SimpleNamespace(group_id=group, temperature_c=temperature, wavelengths_nm=wavelengths, index=index)
        for index, (group, temperature) in enumerate(zip(groups, y))
    ]
    monkeypatch.setattr(preprocessing, "scan_records", lambda root: records)
    monkeypatch.setattr(preprocessing, "open_cube", lambda record: record.index)

    def extract(index, geometry, *, include_alternative_backgrounds):
        assert include_alternative_backgrounds is False
        return {
            "difference_annulus": baseline_features[index] + (1 if audit_mismatch else 0),
            "roi_mean": baseline_features[index] + 1000,
            "roi_standard_error": np.asarray([2., 3.], dtype=np.float32),
            "background_median": np.asarray([1000., 1000.], dtype=np.float32),
        }

    monkeypatch.setattr(preprocessing, "process_cube", extract)
    output_dir = tmp_path / "prepared"
    if audit_mismatch:
        with pytest.raises(ValueError, match="无法复现原差分DN"):
            preprocessing.build_no_background_data(tmp_path, baseline_path, manifest_path, output_dir)
        assert not (output_dir / "swir_roi_no_background.npz").exists()
        return
    destination = preprocessing.build_no_background_data(tmp_path, baseline_path, manifest_path, output_dir)
    with np.load(destination, allow_pickle=False) as prepared:
        np.testing.assert_array_equal(prepared["X_raw_features"], baseline_features + 1000)
        np.testing.assert_array_equal(prepared["X_roi_dn"], prepared["X_raw_features"])
        np.testing.assert_array_equal(prepared["y_temperature_c"], y)
        assert not prepared["background_subtracted"].item()
        assert not prepared["bad_band_mask"].any()
        np.testing.assert_array_equal(prepared["X_uncertainty"][0], [2., 3.])
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["baseline_difference_max_absolute_dn"] == 0.
