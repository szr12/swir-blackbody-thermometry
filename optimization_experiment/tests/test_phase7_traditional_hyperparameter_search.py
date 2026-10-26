from __future__ import annotations

from pathlib import Path
import sys

import json
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_phase7_traditional_hyperparameter_search import rank_candidates  # noqa: E402
import run_phase7_traditional_hyperparameter_search as search  # noqa: E402


def test_rank_candidates_prefers_adaptive_rmse_then_relative_deviation() -> None:
    rows = [
        {"success": True, "validation_relative_deviation": 0.08, "validation_rmse_c": 10.0,
         "band_count": 10, "redundancy_weight": 0.01, "overlap_c": 25.0, "routing_width_c": 35.0},
        {"success": True, "validation_relative_deviation": 0.07, "validation_rmse_c": 20.0,
         "band_count": 9, "redundancy_weight": 0.01, "overlap_c": 25.0, "routing_width_c": 35.0},
        {"success": True, "validation_relative_deviation": 0.07, "validation_rmse_c": 18.0,
         "band_count": 8, "redundancy_weight": 0.01, "overlap_c": 25.0, "routing_width_c": 35.0},
        {"success": False, "validation_relative_deviation": 0.01, "validation_rmse_c": 1.0,
         "band_count": 8, "redundancy_weight": 0.01, "overlap_c": 25.0, "routing_width_c": 35.0},
    ]
    ranked = rank_candidates(rows)
    assert ranked[0]["band_count"] == 10
    assert len(ranked) == 3


def test_rank_candidates_enforces_rmse_constraint_and_balanced_score() -> None:
    common = {"success": True, "validation_relative_deviation": 0.05,
              "redundancy_weight": 0.01, "overlap_c": 25.0, "routing_width_c": 20.0}
    rows = [
        {**common, "band_count": 10, "validation_rmse_c": 11.0,
         "validation_rmse_feasible": True, "validation_balanced_score": 0.10},
        {**common, "band_count": 11, "validation_rmse_c": 10.0,
         "validation_rmse_feasible": True, "validation_balanced_score": 0.08},
        {**common, "band_count": 12, "validation_rmse_c": 9.0,
         "validation_rmse_feasible": False, "validation_balanced_score": 0.01},
    ]
    ranked = rank_candidates(rows)
    assert [row["band_count"] for row in ranked] == [11, 10]


def test_mi_cache_does_not_copy_unverified_background_cache(tmp_path, monkeypatch) -> None:
    """相同尺寸不足以证明缓存属于当前Train，缺少指纹时必须重新计算。"""
    source = tmp_path / "old.npy"
    np.save(source, np.full((2, 2), 99., dtype=np.float32))
    calls = []

    def compute(path, features, neighbors, seed):
        assert not path.exists()
        calls.append(features.copy())
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.eye(2, dtype=np.float32)
        np.save(path, matrix)
        return matrix

    monkeypatch.setattr(search, "load_or_compute_mi", compute)
    features = np.arange(12).reshape(6, 2)
    result = search._prepare_mi_cache({"mi_cache_source": "old.npy"}, tmp_path, tmp_path / "new", features, 3, 42)
    assert len(calls) == 1
    np.testing.assert_array_equal(result, np.eye(2))
    metadata = json.loads((tmp_path / "new/cache/training_mutual_information.json").read_text())
    assert metadata["shape"] == [6, 2]


def test_mi_cache_rejects_changed_train_features(tmp_path, monkeypatch) -> None:
    """即使形状相同，DN值改变后不能复用先前互信息矩阵。"""
    def compute(path, features, neighbors, seed):
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.eye(features.shape[1], dtype=np.float32)
        np.save(path, matrix)
        return matrix

    monkeypatch.setattr(search, "load_or_compute_mi", compute)
    features = np.arange(12).reshape(6, 2)
    search._prepare_mi_cache({}, tmp_path, tmp_path / "run", features, 3, 42)
    with pytest.raises(RuntimeError, match="指纹"):
        search._prepare_mi_cache({}, tmp_path, tmp_path / "run", features + 1000, 3, 42)
