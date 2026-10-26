"""Core CARS equations and sampling operations from Li et al. (2009)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class EDFSchedule:
    a: float
    k: float
    ratios: np.ndarray
    retained_counts: np.ndarray


def exponentially_decreasing_schedule(n_variables: int, n_runs: int) -> EDFSchedule:
    """Paper Eqs. 4-6: first run keeps all variables, last keeps two."""

    if n_variables < 2:
        raise ValueError("CARS requires at least two variables")
    if n_runs < 2:
        raise ValueError("CARS requires at least two sampling runs")
    k = float(np.log(n_variables / 2.0) / (n_runs - 1))
    a = float(np.exp(k))
    positions = np.arange(1, n_runs + 1, dtype=np.float64)
    ratios = a * np.exp(-k * positions)
    counts = np.floor(n_variables * ratios + 1e-12).astype(np.int64)
    counts = np.clip(counts, 2, n_variables)
    counts[0] = n_variables
    counts[-1] = 2
    return EDFSchedule(a, k, ratios, counts)


def normalized_coefficient_weights(coefficients: Any) -> np.ndarray:
    values = np.abs(np.asarray(coefficients, dtype=np.float64).reshape(-1))
    if not np.isfinite(values).all():
        raise ValueError("PLS coefficients must be finite")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("all PLS coefficient weights are zero")
    return values / total


def adaptive_reweighted_sample(
    weights: Any,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Paper ARS: weighted sampling with replacement, then retain unique indices."""

    probabilities = np.asarray(weights, dtype=np.float64).reshape(-1)
    if n_draws < 1 or probabilities.size < 2:
        raise ValueError("invalid ARS dimensions")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("ARS weights must be finite and non-negative")
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("ARS weights must have positive sum")
    probabilities = probabilities / total
    draws = rng.choice(probabilities.size, size=n_draws, replace=True, p=probabilities)
    return np.unique(draws).astype(np.int64)


def enforce_edf(weights: Any, retained_count: int) -> np.ndarray:
    """Zero all but the largest weights, with lower-index deterministic ties."""

    values = np.asarray(weights, dtype=np.float64).reshape(-1).copy()
    if retained_count < 2 or retained_count > values.size:
        raise ValueError("retained_count must be between 2 and the variable count")
    order = np.lexsort((np.arange(values.size), -values))
    values[order[retained_count:]] = 0.0
    return values


def _interleaved_splits(y: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    order = np.argsort(y, kind="stable")
    fold_id = np.empty(y.size, dtype=np.int64)
    fold_id[order] = np.arange(y.size) % n_splits
    all_indices = np.arange(y.size)
    return [(all_indices[fold_id != fold], all_indices[fold_id == fold]) for fold in range(n_splits)]


def make_cv_splits(
    y: Any,
    *,
    n_splits: int,
    groups: Any | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if n_splits < 2 or n_splits > target.size:
        raise ValueError("invalid cross-validation fold count")
    if groups is None:
        return _interleaved_splits(target, n_splits)
    group_array = np.asarray(groups)
    if group_array.ndim != 1 or group_array.size != target.size:
        raise ValueError("groups must have shape [N]")
    if np.unique(group_array).size < n_splits:
        raise ValueError("groups contain fewer unique values than CV folds")
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(target.size), target, group_array))


def pls_cross_validated_rmse(
    X: Any,
    y: Any,
    *,
    max_components: int,
    n_splits: int,
    scale: bool,
    groups: Any | None = None,
) -> tuple[float, int, np.ndarray]:
    """Return minimum RMSECV, selected component count, and component curve."""

    features = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    splits = make_cv_splits(target, n_splits=n_splits, groups=groups)
    min_train = min(train.size for train, _ in splits)
    upper = min(int(max_components), features.shape[1], min_train - 1)
    if upper < 1:
        raise ValueError("not enough samples/features for PLS cross-validation")
    curve = np.empty(upper, dtype=np.float64)
    for component in range(1, upper + 1):
        prediction = np.empty(target.size, dtype=np.float64)
        for train, test in splits:
            model = PLSRegression(n_components=component, scale=scale, max_iter=1000, tol=1e-10)
            model.fit(features[train], target[train])
            prediction[test] = model.predict(features[test]).reshape(-1)
        curve[component - 1] = float(np.sqrt(np.mean((target - prediction) ** 2)))
    best = int(np.argmin(curve)) + 1
    return float(curve[best - 1]), best, curve


def validate_disjoint_group_splits(
    splits: Iterable[tuple[np.ndarray, np.ndarray]], groups: Any
) -> bool:
    values = np.asarray(groups)
    return all(not (set(values[train]) & set(values[test])) for train, test in splits)
