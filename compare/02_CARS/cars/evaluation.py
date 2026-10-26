"""Shared fair-comparison regression metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np
from sklearn.linear_model import LinearRegression


class Regressor(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...
    def predict(self, X: np.ndarray) -> Any: ...


RegressorFactory = Callable[[], Regressor]


def mean_pairwise_jaccard(selections: Sequence[Sequence[int]]) -> float:
    sets = [set(map(int, selection)) for selection in selections]
    if len(sets) < 2 or any(not value for value in sets):
        raise ValueError("at least two non-empty selections are required")
    values = [
        len(sets[i] & sets[j]) / len(sets[i] | sets[j])
        for i in range(len(sets) - 1)
        for j in range(i + 1, len(sets))
    ]
    return float(np.mean(values))


def variance_inflation_factors(X_selected: Any) -> list[float]:
    X = np.asarray(X_selected, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 2 or not np.isfinite(X).all():
        raise ValueError("X_selected must be finite, 2D, and contain at least two bands")
    result: list[float] = []
    for index in range(X.shape[1]):
        target = X[:, index]
        others = np.delete(X, index, axis=1)
        if np.std(target) == 0:
            result.append(float("inf"))
            continue
        prediction = LinearRegression().fit(others, target).predict(others)
        total = float(np.sum((target - target.mean()) ** 2))
        residual = float(np.sum((target - prediction) ** 2))
        r_squared = 1.0 - residual / total
        result.append(float("inf") if r_squared >= 1.0 else 1.0 / (1.0 - r_squared))
    return result


def evaluate_selected_bands(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    *,
    selected_bands: Sequence[int],
    evaluator_factory: RegressorFactory,
) -> dict[str, Any]:
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("train and test features must be 2D")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("train and test band counts must match")
    if X_train.shape[0] != y_train.size or X_test.shape[0] != y_test.size:
        raise ValueError("feature and target row counts must match")
    if not all(np.isfinite(value).all() for value in (X_train, X_test, y_train, y_test)):
        raise ValueError("evaluation arrays must be finite")
    bands = np.asarray(selected_bands, dtype=int)
    if bands.ndim != 1 or bands.size == 0 or np.unique(bands).size != bands.size:
        raise ValueError("selected_bands must be a non-empty unique sequence")
    if bands.min() < 0 or bands.max() >= X_train.shape[1]:
        raise ValueError("selected_bands contains an out-of-range index")
    model = evaluator_factory()
    model.fit(X_train[:, bands], y_train)
    prediction = np.asarray(model.predict(X_test[:, bands]), dtype=np.float64).reshape(-1)
    residual = y_test - prediction
    total = float(np.sum((y_test - y_test.mean()) ** 2))
    residual_ss = float(np.sum(residual**2))
    vif = variance_inflation_factors(X_train[:, bands])
    return {
        "selected_bands": bands.tolist(),
        "band_index_base": 0,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": None if total == 0 else 1.0 - residual_ss / total,
        "vif": [None if not np.isfinite(value) else value for value in vif],
        "vif_has_infinite": any(not np.isfinite(value) for value in vif),
    }


def write_result_json(path: str | Path, result: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
