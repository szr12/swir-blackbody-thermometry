"""SPA 原始 Phase 1-2 的公共接口。"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from .core import phase1_chains, phase2_press


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _validate_selection_role(role: str, argument_name: str) -> str:
    normalized = str(role).strip().lower()
    if not normalized:
        raise ValueError(f"{argument_name} 不能为空")
    if "test" in normalized or "测试" in normalized:
        raise ValueError("测试数据不得进入 SPA 选带")
    return normalized


class SPASelector:
    """2001 SPA 投影与 PRESS 验证选择器。"""

    def __init__(self, *, m_min: int = 1, m_max: int | None = None, autoscaling: bool = False) -> None:
        self.m_min = int(m_min)
        self.m_max = None if m_max is None else int(m_max)
        self.autoscaling = bool(autoscaling)
        self._fitted = False

    @staticmethod
    def _xy(X: np.ndarray, y: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError(f"{name} 必须为 X:[N,D]、y:[N]")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"{name} 的样本数不一致")
        if X.shape[0] < 3 or X.shape[1] < 1:
            raise ValueError(f"{name} 样本或波段不足")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(f"{name} 包含 NaN 或 Inf")
        if np.ptp(y) == 0:
            raise ValueError(f"{name} 的响应为常量")
        return X, y

    def fit(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        *,
        calibration_role: str = "calibration",
        validation_role: str = "validation",
    ) -> "SPASelector":
        """拟合 SPA；角色参数用于阻止调用方把 Test 传入选带流程。"""
        calibration_role = _validate_selection_role(calibration_role, "calibration_role")
        X_cal, y_cal = self._xy(X_cal, y_cal, "校准集")
        separate_validation = X_val is not None or y_val is not None
        roles = [calibration_role]
        if separate_validation:
            validation_role = _validate_selection_role(validation_role, "validation_role")
            if X_val is None or y_val is None:
                raise ValueError("X_val 与 y_val 必须同时提供")
            X_val, y_val = self._xy(X_val, y_val, "验证集")
            if X_val.shape[1] != X_cal.shape[1]:
                raise ValueError("校准集与验证集波段数不同")
            roles.append(validation_role)
        default_m_max = min(X_cal.shape[0] - (1 if separate_validation else 2), X_cal.shape[1])
        m_max = default_m_max if self.m_max is None else self.m_max
        if not 1 <= self.m_min <= m_max <= min(X_cal.shape[0] - 1, X_cal.shape[1]):
            raise ValueError("m_min 或 m_max 超出作者源码允许范围")

        started = time.perf_counter()
        chains, means, scales = phase1_chains(X_cal, m_max, self.autoscaling)
        press, best_initial, best_size = phase2_press(X_cal, y_cal, chains, self.m_min, X_val, y_val)
        selected_order = chains[:best_size, best_initial].astype(np.int64)
        arrays = [X_cal, y_cal]
        if separate_validation:
            arrays.extend([X_val, y_val])
        self.selected_order_ = selected_order
        self.selected_indices_ = np.sort(selected_order)
        self.diagnostics_ = {
            "method": "SPA 2001 Phase 1-2 author-source-aligned reproduction",
            "source_file": "official_materials/source_2007/spa.m",
            "phase3_applied": False,
            "validation_mode": "separate" if separate_validation else "leave_one_out",
            "selection_data_roles": roles,
            "m_min": self.m_min,
            "m_max": m_max,
            "autoscaling": self.autoscaling,
            "best_initial_index": best_initial,
            "best_size": best_size,
            "selected_order": selected_order.tolist(),
            "selected_indices": self.selected_indices_.tolist(),
            "phase1_chains": chains.tolist(),
            "press": press.tolist(),
            "calibration_means": means.tolist(),
            "normalization_scales": scales.tolist(),
            "data_hash": _array_hash(*arrays),
            "elapsed_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        }
        self._fitted = True
        return self

    def select(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("必须先调用 fit")
        return self.selected_indices_.copy()

    def save_diagnostics(self, path: str | Path) -> None:
        if not self._fitted:
            raise RuntimeError("必须先调用 fit")
        Path(path).write_text(
            json.dumps(self.diagnostics_, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )


def band_selection_regression(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    *,
    m_min: int = 1,
    m_max: int | None = None,
    autoscaling: bool = False,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    selector = SPASelector(m_min=m_min, m_max=m_max, autoscaling=autoscaling).fit(
        X_cal, y_cal, X_val, y_val
    )
    selected = selector.select()
    if return_diagnostics:
        return selected, selector.diagnostics_
    return selected
