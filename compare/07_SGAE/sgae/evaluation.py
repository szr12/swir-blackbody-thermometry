from __future__ import annotations

from itertools import combinations
from typing import Any, Callable

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    if y_true_array.ndim != 1 or y_pred_array.shape != y_true_array.shape:
        raise ValueError("y_true and y_pred must be matching one-dimensional arrays")
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "r2": float(r2_score(y_true_array, y_pred_array)),
    }


def vif_values(X: Any) -> list[float]:
    x = np.asarray(X, dtype=float)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("X must be a non-empty two-dimensional array")
    values: list[float] = []
    for column in range(x.shape[1]):
        y = x[:, column]
        others = np.delete(x, column, axis=1)
        if others.shape[1] == 0:
            values.append(1.0)
            continue
        design = np.column_stack([np.ones(x.shape[0]), others])
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        total = np.sum((y - y.mean()) ** 2)
        residual = np.sum((y - fitted) ** 2)
        if total == 0 or residual <= np.finfo(float).eps * max(total, 1.0):
            values.append(float("inf"))
        else:
            r2 = 1.0 - residual / total
            values.append(float(1.0 / max(1.0 - r2, np.finfo(float).eps)))
    return values


def mean_pairwise_jaccard(selections: list[Any]) -> float:
    sets = [set(np.asarray(item, dtype=int).tolist()) for item in selections]
    if len(sets) < 2:
        raise ValueError("at least two selections are required")
    scores = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
    return float(np.mean(scores))


def evaluate_with_regressor(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    selected_bands: Any,
    regressor_factory: Callable[[], Any],
) -> dict[str, Any]:
    selected = np.asarray(selected_bands, dtype=int)
    train = np.asarray(X_train)[:, selected]
    test = np.asarray(X_test)[:, selected]
    regressor = regressor_factory()
    regressor.fit(train, y_train)
    prediction = regressor.predict(test)
    return {
        "metrics": regression_metrics(y_test, prediction),
        "vif": vif_values(train),
    }

