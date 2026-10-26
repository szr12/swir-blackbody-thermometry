"""Public HCBH selector API."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .geometry import IMPLEMENTATION_VERSION, build_hierarchy, origin_distance, representatives
from .statistics import (
    StreamingStatistics,
    atomic_write_json,
    atomic_write_npz,
    collect_streaming_statistics,
    runtime_environment,
)


def _configuration_hash(configuration: dict[str, Any]) -> str:
    payload = json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stats_diagnostics(stats: StreamingStatistics) -> dict[str, Any]:
    layout = stats.layout.to_dict()
    return {
        "layout": layout,
        "global_minimum": stats.global_minimum,
        "global_maximum": stats.global_maximum,
        "global_value_range": stats.global_maximum - stats.global_minimum,
        "normalization": "training-fold global min-max followed by one common unit-ball scale",
        "ball_scale": stats.ball_scale,
        "preprojection_band_norms": stats.preprojection_norms.tolist(),
        "projected_band_norms": stats.projected_norms.tolist(),
        "raw_band_ranges": stats.raw_band_ranges.tolist(),
        "constant_bands": stats.constant_bands.astype(int).tolist(),
        "zero_vector_bands": stats.zero_vector_bands.astype(int).tolist(),
        "data_sha256": stats.data_hash,
        "source_fingerprint_sha256": stats.source_fingerprint,
        "statistics_checkpoint_restored": stats.checkpoint_restored,
    }


class HCBHSelector:
    """Paper-driven HCBH for one homogeneous group of training HSI cubes."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        dtype: str = "float64",
        checkpoint_path: str | Path | None = None,
        chunk_lines: int = 32,
        epsilon: float = 1e-12,
        ball_margin: float = 1e-12,
    ) -> None:
        if device != "cpu":
            raise ValueError("the canonical HCBH implementation supports device='cpu' only")
        if dtype != "float64":
            raise ValueError("the canonical HCBH implementation requires dtype='float64'")
        if isinstance(chunk_lines, bool) or not isinstance(chunk_lines, int) or chunk_lines <= 0:
            raise ValueError("chunk_lines must be a positive integer")
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if not np.isfinite(ball_margin) or not 0.0 < ball_margin < 1.0:
            raise ValueError("ball_margin must be in (0, 1)")
        self.device = device
        self.dtype = dtype
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self.chunk_lines = chunk_lines
        self.epsilon = float(epsilon)
        self.ball_margin = float(ball_margin)
        self._configuration = {
            "implementation_version": IMPLEMENTATION_VERSION,
            "device": device,
            "dtype": dtype,
            "chunk_lines": chunk_lines,
            "epsilon_1": self.epsilon,
            "epsilon_2": self.epsilon,
            "ball_margin": self.ball_margin,
            "normalization": "global_minmax_then_common_unit_ball_scale",
            "cluster_feature": "arithmetic_mean_of_member_bands",
            "merge_tie_break": "eta_then_left_original_boundary",
            "representative_tie_break": "smaller_original_index",
        }
        self.configuration_hash = _configuration_hash(self._configuration)
        self._stats: StreamingStatistics | None = None
        self._hierarchy: dict[str, Any] | None = None
        self._origin_distances: np.ndarray | None = None
        self._fit_seconds: float | None = None
        self._hierarchy_checkpoint_restored = False

    def fit(self, cubes_or_records: Sequence[Any]) -> "HCBHSelector":
        started = time.perf_counter()
        stats = collect_streaming_statistics(
            cubes_or_records,
            chunk_lines=self.chunk_lines,
            ball_margin=self.ball_margin,
            checkpoint_path=self.checkpoint_path,
            config_hash=self.configuration_hash,
        )
        distances = np.asarray([
            origin_distance(float(norm), margin=self.ball_margin)
            for norm in stats.projected_norms
        ], dtype=np.float64)

        hierarchy_path = None if self.checkpoint_path is None else self.checkpoint_path / "hierarchy.json"
        hierarchy: dict[str, Any] | None = None
        if hierarchy_path is not None and hierarchy_path.is_file():
            cached = json.loads(hierarchy_path.read_text(encoding="utf-8"))
            if (
                cached.get("configuration_sha256") == self.configuration_hash
                and cached.get("data_sha256") == stats.data_hash
                and cached.get("source_fingerprint_sha256") == stats.source_fingerprint
            ):
                hierarchy = cached["hierarchy"]
                self._hierarchy_checkpoint_restored = True
            else:
                raise ValueError("hierarchy checkpoint does not match current data or configuration")
        if hierarchy is None:
            hierarchy = build_hierarchy(stats.scaled_gram, epsilon=self.epsilon)
            if hierarchy_path is not None:
                atomic_write_json(
                    hierarchy_path,
                    {
                        "configuration_sha256": self.configuration_hash,
                        "data_sha256": stats.data_hash,
                        "source_fingerprint_sha256": stats.source_fingerprint,
                        "hierarchy": hierarchy,
                    },
                )

        self._stats = stats
        self._origin_distances = distances
        self._hierarchy = hierarchy
        self._fit_seconds = time.perf_counter() - started
        return self

    def _require_fit(self) -> tuple[StreamingStatistics, dict[str, Any], np.ndarray]:
        if self._stats is None or self._hierarchy is None or self._origin_distances is None:
            raise RuntimeError("fit must be called before selection")
        return self._stats, self._hierarchy, self._origin_distances

    def select(self, n_bands: int) -> list[int]:
        _stats, hierarchy, distances = self._require_fit()
        selected, _order, _clusters = representatives(hierarchy, distances, n_bands)
        return selected

    def diagnostics(self, n_bands: int | None = None) -> dict[str, Any]:
        stats, hierarchy, distances = self._require_fit()
        result: dict[str, Any] = {
            "method": "HCBH",
            "paper": {
                "title": "Novel hyperbolic clustering-based band hierarchy (HCBH) for effective unsupervised band selection of hyperspectral images",
                "doi": "10.1016/j.patcog.2022.108788",
                "implementation": "unofficial paper-driven reproduction",
            },
            "configuration": self._configuration,
            "configuration_sha256": self.configuration_hash,
            "band_index_base": 0,
            "fit_seconds": self._fit_seconds,
            "environment": runtime_environment(),
            "origin_distances": distances.tolist(),
            "hierarchy_checkpoint_restored": self._hierarchy_checkpoint_restored,
            "hierarchy": hierarchy,
            **_stats_diagnostics(stats),
        }
        if n_bands is not None:
            selected, order, clusters = representatives(hierarchy, distances, n_bands)
            result["selection"] = {
                "n_bands": int(n_bands),
                "selected_bands": selected,
                "selection_order": order,
                "clusters_in_spectral_order": clusters,
            }
        return result

    def save(self, output_directory: str | Path) -> None:
        stats, _hierarchy, distances = self._require_fit()
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_npz(
            output / "model.npz",
            scaled_gram=stats.scaled_gram,
            origin_distances=distances,
            projected_band_norms=stats.projected_norms,
        )
        atomic_write_json(output / "diagnostics.json", self.diagnostics())


def band_selection_hsi(
    cubes_or_records: Sequence[Any],
    n_bands: int,
    *,
    return_diagnostics: bool = False,
    **selector_kwargs: Any,
) -> list[int] | tuple[list[int], dict[str, Any]]:
    selector = HCBHSelector(**selector_kwargs).fit(cubes_or_records)
    selected = selector.select(n_bands)
    return (selected, selector.diagnostics(n_bands)) if return_diagnostics else selected
