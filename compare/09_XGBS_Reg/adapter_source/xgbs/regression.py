"""官方 XGBS 的连续温度回归适配。"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np
from xgboost import XGBRegressor

from .tools import gauss_weighting, normalize, parse_trees


PAPER_MODEL_DEFAULTS: dict[str, Any] = {
    "n_estimators": 128,
    "max_depth": 10,
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "gamma": 1,
    "min_child_weight": 1,
    "subsample": 1,
    "colsample_bytree": 1,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "importance_type": "gain",
    "device": "cpu",
    "n_jobs": 1,
}


def _normalize_array(values: np.ndarray) -> np.ndarray:
    normalized = normalize(values)
    if np.isscalar(normalized):
        return np.zeros_like(values, dtype=np.float64)
    return np.asarray(normalized, dtype=np.float64)


def _validate_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("X 必须为 [N,D]，y 必须为同样本数的 [N]")
    if X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("至少需要两个样本和一个波段")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X 和 y 必须只包含有限数值")
    if float(np.std(y)) == 0.0:
        raise ValueError("y 不能是常量")
    return X, y


def _redundancy_matrix_2d(X: np.ndarray) -> np.ndarray:
    result = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    standard_deviation = np.std(X, axis=0)
    for i in range(X.shape[1]):
        result[i, i] = 1.0
        for j in range(i + 1, X.shape[1]):
            if standard_deviation[i] == 0 or standard_deviation[j] == 0:
                correlation = 0.0
            else:
                correlation = float(abs(np.corrcoef(X[:, i], X[:, j])[0, 1]))
            result[i, j] = correlation
            result[j, i] = correlation
    return result


@dataclass(frozen=True)
class RegressionSelectionResult:
    selected_bands: tuple[int, ...]
    alpha: float
    beta: float
    feature_gain: np.ndarray
    sequential_dependence: np.ndarray
    complementary_dependence: np.ndarray
    redundancy: np.ndarray
    target_mean: float
    target_std: float
    elapsed_seconds: float
    model_params: Mapping[str, Any]
    random_state: int

    def to_dict(self, include_matrices: bool = False) -> dict[str, Any]:
        output: dict[str, Any] = {
            "selected_bands": list(self.selected_bands),
            "alpha": self.alpha,
            "beta": self.beta,
            "feature_gain": self.feature_gain.tolist(),
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "elapsed_seconds": self.elapsed_seconds,
            "model_params": dict(self.model_params),
            "random_state": self.random_state,
            "band_index_base": 0,
        }
        if include_matrices:
            output.update(
                sequential_dependence=self.sequential_dependence.tolist(),
                complementary_dependence=self.complementary_dependence.tolist(),
                redundancy=self.redundancy.tolist(),
            )
        return output


class XGBSRegressionSelector:
    """拟合一次回归 XGBoost，并复用解释量搜索 alpha、beta。"""

    def __init__(self, *, random_state: int = 0, model_params: Mapping[str, Any] | None = None):
        self.random_state = int(random_state)
        self.model_params = dict(PAPER_MODEL_DEFAULTS)
        if model_params:
            self.model_params.update(model_params)
        self.model_params["random_state"] = self.random_state
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBSRegressionSelector":
        X, y = _validate_xy(X, y)
        started = perf_counter()
        X_normalized = _normalize_array(X)
        target_mean = float(np.mean(y))
        target_std = float(np.std(y))
        y_standardized = (y - target_mean) / target_std
        model = XGBRegressor(**self.model_params)
        model.fit(X_normalized, y_standardized)
        feature_gain = np.asarray(model.feature_importances_, dtype=np.float64)
        same_level, same_path = parse_trees(
            model.get_booster().get_dump(with_stats=True), X.shape[1]
        )
        self.X_shape_ = X.shape
        self.feature_gain_ = feature_gain
        self.sequential_dependence_ = np.asarray(same_path, dtype=np.float64)
        self.complementary_dependence_ = np.asarray(same_level, dtype=np.float64)
        self.redundancy_ = gauss_weighting(
            _redundancy_matrix_2d(X_normalized), det=20, clip=0.1
        )
        self.target_mean_ = target_mean
        self.target_std_ = target_std
        self.elapsed_seconds_ = perf_counter() - started
        self.model_ = model
        self._is_fitted = True
        return self

    def select(self, nbs: int, *, alpha: float = 0.1, beta: float = 0.1) -> RegressionSelectionResult:
        if not self._is_fitted:
            raise RuntimeError("必须先调用 fit")
        if isinstance(nbs, bool) or int(nbs) != nbs or not 1 <= int(nbs) <= self.X_shape_[1]:
            raise ValueError("nbs 超出有效范围")
        if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(beta) or beta < 0:
            raise ValueError("alpha 和 beta 必须为有限非负数")
        selected = [int(np.argmax(self.feature_gain_))]
        for _ in range(int(nbs) - 1):
            sequential_score = _normalize_array(
                np.sum(self.sequential_dependence_[selected, :], axis=0)
            )
            complementary_score = _normalize_array(
                np.sum(self.complementary_dependence_[selected, :], axis=0)
            )
            dependency_score = sequential_score + float(beta) * complementary_score
            maximum_redundancy = np.max(self.redundancy_[selected, :], axis=0)
            individual_score = self.feature_gain_ * (1.0 - maximum_redundancy)
            combined_score = individual_score + float(alpha) * dependency_score
            combined_score[selected] = -np.inf
            selected.append(int(np.argmax(combined_score)))
        return RegressionSelectionResult(
            selected_bands=tuple(selected), alpha=float(alpha), beta=float(beta),
            feature_gain=self.feature_gain_.copy(),
            sequential_dependence=self.sequential_dependence_.copy(),
            complementary_dependence=self.complementary_dependence_.copy(),
            redundancy=self.redundancy_.copy(), target_mean=self.target_mean_,
            target_std=self.target_std_, elapsed_seconds=self.elapsed_seconds_,
            model_params=dict(self.model_params), random_state=self.random_state,
        )


def band_selection_regression(
    X_2d: np.ndarray, y_1d: np.ndarray, nbs: int, alpha: float = 0.1,
    beta: float = 0.1, *, random_state: int = 0,
    model_params: Mapping[str, Any] | None = None, return_diagnostics: bool = False,
):
    """从连续温度任务中选择论文启发式定义的光谱波段。"""
    result = XGBSRegressionSelector(
        random_state=random_state, model_params=model_params
    ).fit(X_2d, y_1d).select(nbs, alpha=alpha, beta=beta)
    return result if return_diagnostics else list(result.selected_bands)
