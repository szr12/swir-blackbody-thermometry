from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


def vif_values(X: np.ndarray) -> list[float]:
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] == 1:
        return [1.0]
    output: list[float] = []
    for column in range(X.shape[1]):
        target = X[:, column]
        others = np.delete(X, column, axis=1)
        if float(np.var(target)) <= 1e-15:
            output.append(float("inf"))
            continue
        r2 = float(LinearRegression().fit(others, target).score(others, target))
        output.append(float("inf") if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2))
    return output


def mean_pairwise_jaccard(selections: list[list[int]]) -> float:
    values: list[float] = []
    for left in range(len(selections)):
        for right in range(left + 1, len(selections)):
            a, b = set(selections[left]), set(selections[right])
            values.append(len(a & b) / len(a | b))
    return float(np.mean(values)) if values else 1.0
