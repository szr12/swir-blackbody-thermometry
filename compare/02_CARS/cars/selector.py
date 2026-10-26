"""Full paper CARS: Monte Carlo + EDF + ARS + PLS + RMSECV."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from sklearn.cross_decomposition import PLSRegression

from .core import (
    adaptive_reweighted_sample,
    enforce_edf,
    exponentially_decreasing_schedule,
    normalized_coefficient_weights,
    pls_cross_validated_rmse,
)


METHOD_NAME = "CARS-PLS (Li et al. 2009, natural subset size)"
IMPLEMENTATION_VERSION = 1


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _hash_config(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate(X: Any, y: Any, groups: Any | None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    features = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    if features.ndim != 2 or target.ndim != 1:
        raise ValueError("X and y must have shapes [N,D] and [N]")
    if features.shape[0] != target.size:
        raise ValueError("X and y must contain the same number of samples")
    if features.shape[0] < 12 or features.shape[1] < 2:
        raise ValueError("CARS requires at least 12 samples and two bands")
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("X and y must contain only finite values")
    if np.ptp(target) == 0:
        raise ValueError("y must be nonconstant")
    group_array = None if groups is None else np.asarray(groups)
    if group_array is not None and (group_array.ndim != 1 or group_array.size != target.size):
        raise ValueError("groups must have shape [N]")
    return features, target, group_array


@dataclass(frozen=True)
class CARSResult:
    selected_bands: np.ndarray
    diagnostics: dict[str, Any]


class CARSSelector:
    def __init__(
        self,
        *,
        n_components: int,
        n_sampling_runs: int = 100,
        sample_ratio: float = 0.8,
        cv_folds: int = 10,
        scale: bool = False,
        random_state: int = 42,
        device: str = "cpu",
        checkpoint_path: str | Path | None = None,
    ) -> None:
        if not isinstance(n_components, int) or isinstance(n_components, bool) or n_components < 1:
            raise ValueError("n_components must be a positive integer")
        if not isinstance(n_sampling_runs, int) or n_sampling_runs < 2:
            raise ValueError("n_sampling_runs must be at least two")
        if not 0.0 < sample_ratio < 1.0:
            raise ValueError("sample_ratio must be between zero and one")
        if not isinstance(cv_folds, int) or cv_folds < 2:
            raise ValueError("cv_folds must be at least two")
        if device != "cpu":
            raise ValueError("the reference CARS-PLS implementation is CPU-only")
        self.n_components = n_components
        self.n_sampling_runs = n_sampling_runs
        self.sample_ratio = float(sample_ratio)
        self.cv_folds = cv_folds
        self.scale = bool(scale)
        self.random_state = int(random_state)
        self.device = device
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self._is_fitted = False

    def fit(self, X: Any, y: Any, *, groups: Any | None = None) -> "CARSSelector":
        started = time.perf_counter()
        features, target, group_array = _validate(X, y, groups)
        self.n_samples_, self.n_bands_ = features.shape
        self.cv_partition_ = "GroupKFold" if group_array is not None else "interleaved after stable y sort"
        self.constant_bands_ = np.flatnonzero(np.ptp(features, axis=0) == 0).astype(np.int64)
        self.eligible_bands_ = np.setdiff1d(np.arange(self.n_bands_), self.constant_bands_)
        if self.eligible_bands_.size < 2:
            raise ValueError("fewer than two nonconstant bands remain")
        features_work = features[:, self.eligible_bands_]
        p = features_work.shape[1]
        n_calibration = int(np.floor(self.n_samples_ * self.sample_ratio))
        if n_calibration < self.cv_folds:
            raise ValueError("Monte Carlo calibration subset is smaller than cv_folds")
        if group_array is not None:
            min_groups = max(2, self.cv_folds)
            if np.unique(group_array).size < min_groups:
                raise ValueError("groups contain fewer unique values than cv_folds")

        group_hash_array = np.asarray([], dtype=np.uint8) if group_array is None else np.asarray(
            [str(value) for value in group_array], dtype="U"
        )
        self.data_hash_ = _hash_arrays(features, target, group_hash_array)
        config = {
            "implementation_version": IMPLEMENTATION_VERSION,
            "n_components": self.n_components,
            "n_sampling_runs": self.n_sampling_runs,
            "sample_ratio": self.sample_ratio,
            "cv_folds": self.cv_folds,
            "scale": self.scale,
            "random_state": self.random_state,
            "device": self.device,
            "ars_draws": "original full variable count",
            "subset_size": "natural minimum-RMSECV",
        }
        self.config_hash_ = _hash_config(config)
        schedule = exponentially_decreasing_schedule(p, self.n_sampling_runs)
        self.edf_a_, self.edf_k_ = schedule.a, schedule.k
        self.edf_ratios_ = schedule.ratios
        self.edf_counts_ = schedule.retained_counts
        self.coefficient_path_ = np.full((self.n_sampling_runs, self.n_bands_), np.nan)
        self.rmsecv_ = np.full(self.n_sampling_runs, np.nan)
        self.cv_component_curves_: list[list[float] | None] = [None] * self.n_sampling_runs
        self.best_components_ = np.full(self.n_sampling_runs, -1, dtype=np.int64)
        self.subsets_: list[np.ndarray | None] = [None] * self.n_sampling_runs
        self.calibration_indices_: list[np.ndarray | None] = [None] * self.n_sampling_runs
        self.completed_runs_ = 0
        self._rng = np.random.default_rng(self.random_state)
        current = np.arange(p, dtype=np.int64)
        self.checkpoint_loaded_ = False
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            current = self._load_checkpoint()

        for run in range(self.completed_runs_, self.n_sampling_runs):
            calibration = np.sort(
                self._rng.choice(self.n_samples_, size=n_calibration, replace=False)
            ).astype(np.int64)
            active = current.copy()
            X_cal = features_work[np.ix_(calibration, active)]
            y_cal = target[calibration]
            groups_cal = None if group_array is None else group_array[calibration]
            effective = min(self.n_components, active.size, X_cal.shape[0] - 1)
            if effective < 1:
                raise RuntimeError("CARS subset cannot support a PLS component")

            rmse, best_component, curve = pls_cross_validated_rmse(
                X_cal,
                y_cal,
                max_components=effective,
                n_splits=self.cv_folds,
                scale=self.scale,
                groups=groups_cal,
            )
            model = PLSRegression(
                n_components=effective, scale=self.scale, max_iter=1000, tol=1e-10
            ).fit(X_cal, y_cal)
            local_coef = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
            full_work_coef = np.zeros(p, dtype=np.float64)
            full_work_coef[active] = local_coef
            full_coef = np.zeros(self.n_bands_, dtype=np.float64)
            full_coef[self.eligible_bands_] = full_work_coef

            self.subsets_[run] = self.eligible_bands_[active].copy()
            self.calibration_indices_[run] = calibration
            self.coefficient_path_[run] = full_coef
            self.rmsecv_[run] = rmse
            self.best_components_[run] = best_component
            self.cv_component_curves_[run] = curve.tolist()
            self.completed_runs_ = run + 1

            weights = normalized_coefficient_weights(full_work_coef)
            forced = enforce_edf(weights, int(self.edf_counts_[run]))
            current = adaptive_reweighted_sample(forced, p, self._rng)
            if current.size < 2:
                # The paper terminates at two wavelengths; preserve that lower bound.
                current = np.lexsort((np.arange(p), -forced))[:2].astype(np.int64)
            self._save_checkpoint(current)

        self.optimal_iteration_ = int(np.nanargmin(self.rmsecv_))
        optimal_subset = self.subsets_[self.optimal_iteration_]
        assert optimal_subset is not None
        self.selected_bands_ = np.sort(optimal_subset.astype(np.int64))
        self.fit_seconds_ = time.perf_counter() - started
        self._is_fitted = True
        return self

    def select(self, n_bands: int | None = None) -> CARSResult:
        if not self._is_fitted:
            raise RuntimeError("fit must be called before select")
        if n_bands is not None:
            raise ValueError("original CARS chooses a natural subset size; fixed K is not supported")
        return CARSResult(self.selected_bands_.copy(), self.diagnostics())

    def diagnostics(self) -> dict[str, Any]:
        if not self._is_fitted:
            raise RuntimeError("fit must be called before diagnostics")
        subsets = [None if value is None else value.tolist() for value in self.subsets_]
        calibration = [None if value is None else value.tolist() for value in self.calibration_indices_]
        return {
            "method": METHOD_NAME,
            "implementation_version": IMPLEMENTATION_VERSION,
            "paper_doi": "10.1016/j.aca.2009.06.046",
            "band_index_base": 0,
            "n_samples": self.n_samples_,
            "n_bands": self.n_bands_,
            "constant_bands": self.constant_bands_.tolist(),
            "n_components_upper_bound": self.n_components,
            "n_sampling_runs": self.n_sampling_runs,
            "sample_ratio": self.sample_ratio,
            "cv_folds": self.cv_folds,
            "cv_partition": self.cv_partition_,
            "scale_X": self.scale,
            "edf_a": self.edf_a_,
            "edf_k": self.edf_k_,
            "edf_ratios": self.edf_ratios_.tolist(),
            "edf_retained_counts": self.edf_counts_.tolist(),
            "candidate_subsets": subsets,
            "calibration_indices": calibration,
            "coefficient_path": self.coefficient_path_.tolist(),
            "rmsecv": self.rmsecv_.tolist(),
            "best_components": self.best_components_.tolist(),
            "cv_component_curves": self.cv_component_curves_,
            "optimal_iteration_zero_based": self.optimal_iteration_,
            "selected_bands": self.selected_bands_.tolist(),
            "selected_band_count": int(self.selected_bands_.size),
            "selection_policy": "global minimum RMSECV; first iteration breaks exact ties",
            "fixed_k_supported": False,
            "actual_device": "cpu",
            "random_state": self.random_state,
            "fit_seconds": self.fit_seconds_,
            "checkpoint_loaded": self.checkpoint_loaded_,
            "data_sha256": self.data_hash_,
            "config_sha256": self.config_hash_,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "scikit_learn_version": sklearn.__version__,
            "processor": platform.processor() or platform.machine(),
        }

    def _save_checkpoint(self, current: np.ndarray) -> None:
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.checkpoint_path.with_name(self.checkpoint_path.name + ".tmp")
        subset_matrix = np.zeros((self.n_sampling_runs, self.n_bands_), dtype=np.uint8)
        for index, subset in enumerate(self.subsets_):
            if subset is not None:
                subset_matrix[index, subset] = 1
        calibration_matrix = np.full((self.n_sampling_runs, max(len(x) for x in self.calibration_indices_ if x is not None)), -1, dtype=np.int64)
        for index, values in enumerate(self.calibration_indices_):
            if values is not None:
                calibration_matrix[index, : values.size] = values
        with temp.open("wb") as handle:
            np.savez_compressed(
                handle,
                data_hash=np.asarray(self.data_hash_),
                config_hash=np.asarray(self.config_hash_),
                completed=np.asarray(self.completed_runs_),
                current=current,
                coefficients=self.coefficient_path_,
                rmsecv=self.rmsecv_,
                best_components=self.best_components_,
                subsets=subset_matrix,
                calibration=calibration_matrix,
                curves=np.asarray(json.dumps(self.cv_component_curves_)),
                rng_state=np.asarray(json.dumps(self._rng.bit_generator.state)),
            )
        os.replace(temp, self.checkpoint_path)

    def _load_checkpoint(self) -> np.ndarray:
        assert self.checkpoint_path is not None
        try:
            with np.load(self.checkpoint_path, allow_pickle=False) as saved:
                if str(saved["data_hash"].item()) != self.data_hash_:
                    raise ValueError("checkpoint data hash does not match")
                if str(saved["config_hash"].item()) != self.config_hash_:
                    raise ValueError("checkpoint configuration hash does not match")
                completed = int(saved["completed"].item())
                if not 0 <= completed <= self.n_sampling_runs:
                    raise ValueError("checkpoint completed-run count is invalid")
                self.completed_runs_ = completed
                self.coefficient_path_ = saved["coefficients"].copy()
                self.rmsecv_ = saved["rmsecv"].copy()
                self.best_components_ = saved["best_components"].copy()
                subset_matrix = saved["subsets"]
                calibration = saved["calibration"]
                self.subsets_ = [
                    None if index >= completed else np.flatnonzero(subset_matrix[index]).astype(np.int64)
                    for index in range(self.n_sampling_runs)
                ]
                self.calibration_indices_ = [
                    None if index >= completed else calibration[index][calibration[index] >= 0].astype(np.int64)
                    for index in range(self.n_sampling_runs)
                ]
                self.cv_component_curves_ = json.loads(str(saved["curves"].item()))
                self._rng.bit_generator.state = json.loads(str(saved["rng_state"].item()))
                current = saved["current"].astype(np.int64)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("checkpoint"):
                raise
            raise ValueError(f"invalid checkpoint: {self.checkpoint_path}") from exc
        self.checkpoint_loaded_ = True
        return current


def band_selection_regression(
    X_2d: Any,
    y_1d: Any,
    *,
    groups: Any | None = None,
    return_diagnostics: bool = False,
    **selector_kwargs: Any,
) -> np.ndarray | CARSResult:
    result = CARSSelector(**selector_kwargs).fit(X_2d, y_1d, groups=groups).select()
    return result if return_diagnostics else result.selected_bands
