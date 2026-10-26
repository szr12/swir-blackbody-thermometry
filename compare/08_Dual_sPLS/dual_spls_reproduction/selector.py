"""Dual-sPLS pseudo-lasso（Dual-sPLS_l）复现。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import platform
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DualSPLSModel:
    """保存从1到M个潜变量对应的全部模型系数。"""

    x_mean: np.ndarray
    y_mean: float
    scores: np.ndarray
    loadings: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    thresholds: np.ndarray
    lambdas: np.ndarray
    zero_counts: np.ndarray

    def predict(self, x: np.ndarray, n_components: int) -> np.ndarray:
        """使用指定数量潜变量的系数预测连续响应。"""
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.coefficients.shape[0]:
            raise ValueError("X必须为二维数组，且波段数必须与拟合数据一致")
        if not np.isfinite(values).all():
            raise ValueError("X包含NaN或Inf")
        if not 1 <= n_components <= self.coefficients.shape[1]:
            raise ValueError("n_components超出已拟合范围")
        column = n_components - 1
        return values @ self.coefficients[:, column] + self.intercepts[column]


def _validate_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("X必须为二维数组[N,D]")
    if target.ndim != 1:
        raise ValueError("y必须为一维数组[N]")
    if values.shape[0] != target.shape[0]:
        raise ValueError("X与y的样本数不一致")
    if values.shape[0] < 4 or values.shape[1] < 2:
        raise ValueError("至少需要4个样本和2个波段")
    if not np.isfinite(values).all() or not np.isfinite(target).all():
        raise ValueError("输入包含NaN或Inf")
    if np.ptp(target) == 0:
        raise ValueError("y不能为常量")
    if np.any(np.ptp(values, axis=0) == 0):
        raise ValueError("原文算法未定义常量波段，请先核查数据")
    return values, target


def _fit_lasso(x: np.ndarray, y: np.ndarray, n_components: int, ppnu: float) -> DualSPLSModel:
    """逐式复现作者R函数d.spls.lasso。"""
    if not 0 < ppnu < 1:
        raise ValueError("ppnu必须位于(0,1)")
    n_samples, n_features = x.shape
    if not 1 <= n_components <= min(n_samples - 1, n_features):
        raise ValueError("潜变量数量超出数据秩允许范围")

    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    x_centered = x - x_mean
    y_centered = y - y_mean
    x_deflated = x_centered.copy()

    loadings = np.zeros((n_features, n_components), dtype=np.float64)
    scores = np.zeros((n_samples, n_components), dtype=np.float64)
    coefficients = np.zeros((n_features, n_components), dtype=np.float64)
    intercepts = np.zeros(n_components, dtype=np.float64)
    thresholds = np.zeros(n_components, dtype=np.float64)
    lambdas = np.zeros(n_components, dtype=np.float64)
    zero_counts = np.zeros(n_components, dtype=np.int64)

    empirical = np.arange(1, n_features + 1, dtype=np.float64) / n_features
    threshold_index = int(np.argmin(np.abs(empirical - ppnu)))

    for component in range(n_components):
        z = x_deflated.T @ y_centered
        nu = float(np.sort(np.abs(z))[threshold_index])
        z_nu = np.sign(z) * np.maximum(np.abs(z) - nu, 0.0)
        mu = float(np.linalg.norm(z_nu))
        if not np.isfinite(mu) or mu == 0:
            raise ValueError("当前ppnu使全部载荷为零，无法继续原文算法")
        denominator = nu * np.abs(z_nu).sum() + mu * mu
        w = (mu / denominator) * z_nu
        score = x_deflated @ w
        score_norm = float(np.linalg.norm(score))
        if not np.isfinite(score_norm) or score_norm == 0:
            raise ValueError("潜变量得分退化为零")
        score /= score_norm

        loadings[:, component] = w
        scores[:, component] = score
        x_deflated -= np.outer(score, score @ x_deflated)

        current_scores = scores[:, : component + 1]
        current_loadings = loadings[:, : component + 1]
        r_matrix = current_scores.T @ x_centered @ current_loadings
        r_matrix[np.tril_indices(component + 1, k=-1)] = 0.0
        try:
            inverse_r = np.linalg.solve(r_matrix, np.eye(component + 1))
        except np.linalg.LinAlgError as exc:
            raise ValueError("原文上三角系统奇异，无法增加潜变量") from exc
        beta = current_loadings @ (inverse_r @ (current_scores.T @ y_centered))

        coefficients[:, component] = beta
        intercepts[component] = y_mean - float(x_mean @ beta)
        thresholds[component] = nu
        lambdas[component] = nu / mu
        zero_counts[component] = int(np.count_nonzero(beta == 0.0))

    return DualSPLSModel(
        x_mean=x_mean,
        y_mean=y_mean,
        scores=scores,
        loadings=loadings,
        coefficients=coefficients,
        intercepts=intercepts,
        thresholds=thresholds,
        lambdas=lambdas,
        zero_counts=zero_counts,
    )


class DualSPLSSelector:
    """论文 pseudo-lasso 变体的连续回归波段选择器。"""

    _PROFILES = {
        "paper": {"ppnu": 0.99, "max_components": 10, "n_repeats": 30, "calibration_percent": 70},
        "quick": {"ppnu": 0.80, "max_components": 3, "n_repeats": 3, "calibration_percent": 70},
    }

    def __init__(self, profile: str = "paper", random_state: int = 0) -> None:
        if profile not in self._PROFILES:
            raise ValueError("profile必须为paper或quick")
        self.profile = profile
        self.random_state = int(random_state)
        self.model_: DualSPLSModel | None = None
        self.selected_indices_: np.ndarray | None = None
        self.diagnostics_: dict[str, Any] | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "DualSPLSSelector":
        """仅使用外层训练数据选择潜变量数量并拟合最终模型。"""
        started = time.perf_counter()
        x, y = _validate_xy(x_train, y_train)
        config = dict(self._PROFILES[self.profile])
        max_components = min(config["max_components"], x.shape[0] - 1, x.shape[1])
        candidates = np.arange(1, max_components + 1, dtype=np.int64)
        n_calibration = int(np.floor(x.shape[0] * config["calibration_percent"] / 100))
        if n_calibration < 3 or x.shape[0] - n_calibration < 1:
            raise ValueError("样本数不足以执行原文校准/验证流程")

        rng = np.random.default_rng(self.random_state)
        squared_errors = np.zeros((config["n_repeats"], max_components), dtype=np.float64)
        all_indices = np.arange(x.shape[0])
        for repeat in range(config["n_repeats"]):
            calibration = np.sort(rng.choice(all_indices, size=n_calibration, replace=False))
            validation = np.setdiff1d(all_indices, calibration, assume_unique=True)
            fold_model = _fit_lasso(x[calibration], y[calibration], max_components, config["ppnu"])
            for column, n_components in enumerate(candidates):
                residual = fold_model.predict(x[validation], int(n_components)) - y[validation]
                squared_errors[repeat, column] = float(residual @ residual)

        mean_sse = squared_errors.mean(axis=0)
        best_components = int(candidates[int(np.argmin(mean_sse))])
        model = _fit_lasso(x, y, best_components, config["ppnu"])
        beta = model.coefficients[:, best_components - 1]
        selected = np.flatnonzero(beta != 0.0).astype(np.int64)
        if selected.size == 0:
            raise RuntimeError("最终模型没有非零回归系数")

        data_hash = hashlib.sha256(
            np.ascontiguousarray(x).tobytes() + np.ascontiguousarray(y).tobytes()
        ).hexdigest()
        self.model_ = model
        self.selected_indices_ = selected
        self.diagnostics_ = {
            "method": "Dual-sPLS_l",
            "profile": self.profile,
            "ppnu": config["ppnu"],
            "component_candidates": candidates.tolist(),
            "best_components": best_components,
            "cv_sum_squared_errors": squared_errors.tolist(),
            "cv_mean_sum_squared_error": mean_sse.tolist(),
            "selected_indices": selected.tolist(),
            "selected_count": int(selected.size),
            "coefficients": beta.tolist(),
            "thresholds": model.thresholds.tolist(),
            "lambdas": model.lambdas.tolist(),
            "zero_counts": model.zero_counts.tolist(),
            "random_state": self.random_state,
            "data_sha256": data_hash,
            "elapsed_seconds": time.perf_counter() - started,
            "device": "cpu",
            "dtype": "float64",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        }
        return self

    def select(self) -> np.ndarray:
        """返回原文自然产生的非零系数波段，不强制固定数量。"""
        if self.selected_indices_ is None:
            raise RuntimeError("必须先调用fit")
        return self.selected_indices_.copy()

    def diagnostics_json(self) -> str:
        """返回不含NaN或Inf的机器可读诊断。"""
        if self.diagnostics_ is None:
            raise RuntimeError("必须先调用fit")
        return json.dumps(self.diagnostics_, ensure_ascii=False, allow_nan=False, indent=2)


def band_selection_regression(
    x_2d: np.ndarray,
    y_1d: np.ndarray,
    *,
    profile: str = "paper",
    random_state: int = 0,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """便捷入口；不接收测试集，也不接收固定波段数。"""
    selector = DualSPLSSelector(profile=profile, random_state=random_state).fit(x_2d, y_1d)
    indices = selector.select()
    if return_diagnostics:
        return indices, dict(selector.diagnostics_ or {})
    return indices

