from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_background_symmetric_search import (
    fingerprint, selection_signature, verify_cached_signature,
    symmetric_float_values, symmetric_k_values,
)


def test_k_window_recenters_on_new_best_and_has_equal_sides() -> None:
    """最优K改变后必须跟随新中心，两侧等距且同数量。"""
    for center in (14, 20, 26):
        values = symmetric_k_values(center, 2, 5, 264)
        assert values[len(values) // 2] == center
        assert len(values) == 11
        assert all(left + right == 2 * center for left, right in zip(values, reversed(values)))
        assert np.all(np.diff(values) == 2)


def test_k_domain_limits_never_create_one_sided_window() -> None:
    """靠近物理边界只能双侧同时缩短，位于边界时必须报告不可对称。"""
    assert symmetric_k_values(4, 2, 5, 264) == [2, 4, 6]
    assert symmetric_k_values(262, 2, 5, 264) == [260, 262, 264]
    with pytest.raises(ValueError, match="物理边界"):
        symmetric_k_values(2, 2, 5, 264)


def test_lambda_and_temperature_windows_are_symmetric() -> None:
    for center, step, width in ((0.03, 0.005, 4), (55.0, 10.0, 3), (45.0, 10.0, 3)):
        values = symmetric_float_values(center, step, width)
        assert np.allclose(np.asarray(values) + values[::-1], 2 * center)
        assert np.allclose(np.diff(values), step)
    with pytest.raises(ValueError, match="负值"):
        symmetric_float_values(0.01, 0.005, 4)


def test_cache_fingerprint_changes_with_values_and_shape() -> None:
    """同尺寸不同输入不能复用选带缓存。"""
    values = np.arange(12, dtype=np.float64).reshape(4, 3)
    assert fingerprint(values) != fingerprint(values + 1)
    assert fingerprint(values) != fingerprint(values.reshape(3, 4))
    assert fingerprint(values) == fingerprint(np.array(values, order="F"))


def test_selection_signature_survives_fit_mutation() -> None:
    """复现拟合新增变换属性及嵌套配置变更，缓存快照必须保持不变。"""
    parameters = {"band_count": 8, "hidden_units": (256, 128), "ranges": [[1800, 2500]]}
    arrays = (np.arange(6, dtype=np.float64),)
    expected = selection_signature(parameters, arrays)
    parameters["transform_"] = {"floors": np.ones(6)}
    parameters["ranges"][0][0] = 1900
    assert "transform_" not in expected["parameters"]
    assert expected["parameters"]["ranges"] == [[1800, 2500]]
    assert verify_cached_signature(expected, expected) is False


def test_legacy_cache_repair_rejects_changed_input_or_configuration() -> None:
    """只允许移除已确认的拟合属性，真实配置或输入改变仍必须拒绝。"""
    expected = selection_signature({"band_count": 8, "redundancy_weight": 0.03}, (np.arange(6),))
    import copy
    saved = copy.deepcopy(expected)
    saved["parameters"]["transform_"] = {"floors": [1, 2]}
    assert verify_cached_signature(saved, expected) is True
    assert "transform_" in saved["parameters"]
    for altered in (
        selection_signature({"band_count": 10, "redundancy_weight": 0.03}, (np.arange(6),)),
        selection_signature({"band_count": 8, "redundancy_weight": 0.03}, (np.arange(6) + 1,)),
    ):
        with pytest.raises(RuntimeError, match="指纹不一致"):
            verify_cached_signature(saved, altered)
