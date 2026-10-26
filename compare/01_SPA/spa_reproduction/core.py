"""SPA Phase 1-2 的数值核心。"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lstsq, qr


def projections_qr(X: np.ndarray, initial_index: int, length: int) -> np.ndarray:
    """逐语句对应作者 ``projections_qr.m`` 的列主元 QR 链。"""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X 必须是二维数组")
    if not 0 <= initial_index < X.shape[1]:
        raise ValueError("初始变量索引越界")
    if not 1 <= length <= X.shape[1]:
        raise ValueError("投影链长度越界")
    projected = X.copy()
    norms = np.sum(projected * projected, axis=0)
    if norms[initial_index] <= 0:
        raise ValueError("初始变量的中心化范数为零，原文投影公式无定义")
    norm_max = float(np.max(norms))
    projected[:, initial_index] *= 2.0 * norm_max / norms[initial_index]
    _, _, order = qr(projected, mode="economic", pivoting=True, check_finite=True)
    return np.asarray(order[:length], dtype=np.int64)


def _design_matrix(X: np.ndarray, selected: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(X.shape[0]), X[:, selected]))


def mlr_predict(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_eval: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    """使用含截距的最小二乘 MLR，与 MATLAB 反斜杠的 QR 路径对齐。"""
    cal = _design_matrix(X_cal, selected)
    evaluation = _design_matrix(X_eval, selected)
    coefficients = lstsq(cal, y_cal, lapack_driver="gelsy", check_finite=True)[0]
    return evaluation @ coefficients


def validation_errors(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    selected: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """对应作者 ``validation.m`` 的独立验证或显式留一交叉验证。"""
    if X_val is not None and y_val is not None:
        prediction = mlr_predict(X_cal, y_cal, X_val, selected)
        return prediction, y_val - prediction
    n_samples = X_cal.shape[0]
    prediction = np.empty(n_samples, dtype=np.float64)
    for held_out in range(n_samples):
        keep = np.arange(n_samples) != held_out
        prediction[held_out] = mlr_predict(
            X_cal[keep], y_cal[keep], X_cal[held_out : held_out + 1], selected
        )[0]
    return prediction, y_cal - prediction


def loo_press(X_cal: np.ndarray, y_cal: np.ndarray, selected: np.ndarray) -> float:
    """用 PRESS 恒等式精确计算满秩 MLR 的留一误差平方和。

    对秩亏或杠杆值接近 1 的候选退回作者源码等价的显式留一流程，
    避免用不同的广义逆约定悄悄改变候选排序。
    """
    design = _design_matrix(X_cal, selected)
    coefficients, _, rank, _ = lstsq(
        design, y_cal, lapack_driver="gelsy", check_finite=True
    )
    if rank < design.shape[1]:
        _, errors = validation_errors(X_cal, y_cal, selected)
        return float(errors @ errors)
    q, _ = qr(design, mode="economic", pivoting=False, check_finite=True)
    leverage = np.sum(q * q, axis=1)
    denominator = 1.0 - leverage
    if np.any(np.abs(denominator) <= 100.0 * np.finfo(np.float64).eps):
        _, errors = validation_errors(X_cal, y_cal, selected)
        return float(errors @ errors)
    residual = y_cal - design @ coefficients
    loo_errors = residual / denominator
    return float(loo_errors @ loo_errors)


def phase1_chains(
    X_cal: np.ndarray, m_max: int, autoscaling: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成以每个波段为起点的候选投影链。"""
    means = X_cal.mean(axis=0)
    scales = X_cal.std(axis=0, ddof=1) if autoscaling else np.ones(X_cal.shape[1])
    centered_norms = np.sum((X_cal - means) ** 2, axis=0)
    if np.any(centered_norms <= 0):
        bad = np.flatnonzero(centered_norms <= 0).tolist()
        raise ValueError(f"存在常量波段，原文投影公式无定义：{bad}")
    if autoscaling and np.any(scales <= 0):
        raise ValueError("自动缩放遇到零标准差")
    normalized = (X_cal - means) / scales
    chains = np.column_stack(
        [projections_qr(normalized, initial, m_max) for initial in range(X_cal.shape[1])]
    )
    return chains, means, scales


def phase2_press(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    chains: np.ndarray,
    m_min: int,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
) -> tuple[np.ndarray, int, int]:
    """计算候选子集 PRESS，并按 MATLAB ``min`` 的首次并列规则选优。"""
    m_max, n_features = chains.shape
    press = np.full((m_max, n_features), np.inf, dtype=np.float64)
    use_loo = X_val is None and y_val is None
    for initial in range(n_features):
        for size in range(m_min, m_max + 1):
            selected = chains[:size, initial]
            if use_loo:
                press[size - 1, initial] = loo_press(X_cal, y_cal, selected)
            else:
                _, errors = validation_errors(X_cal, y_cal, selected, X_val, y_val)
                press[size - 1, initial] = float(errors @ errors)
    best_size_per_initial = np.argmin(press, axis=0)
    best_press_per_initial = press[best_size_per_initial, np.arange(n_features)]
    best_initial = int(np.argmin(best_press_per_initial))
    best_size = int(best_size_per_initial[best_initial] + 1)
    return press, best_initial, best_size
