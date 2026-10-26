"""Paper equations and deterministic hierarchy construction for HCBH."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


IMPLEMENTATION_VERSION = "hcbh-1.0.1"


def _finite_nonnegative(value: float, name: str, *, tolerance: float = 1e-12) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < -tolerance:
        raise ValueError(f"{name} must be non-negative")
    return max(value, 0.0)


def origin_distance(norm: float, *, margin: float = 1e-12) -> float:
    """HCBH Eq. (1), ``z = 2 atanh(||x||_2)``."""

    norm = _finite_nonnegative(norm, "norm")
    if not 0.0 <= norm < 1.0:
        raise ValueError("a Poincare-ball norm must be in [0, 1)")
    if not 0.0 < margin < 1.0:
        raise ValueError("margin must be in (0, 1)")
    return float(2.0 * np.arctanh(min(norm, 1.0 - margin)))


def hyperbolic_distance(
    norm_left_sq: float,
    norm_right_sq: float,
    squared_euclidean: float,
    *,
    epsilon: float = 1e-12,
    domain_tolerance: float = 1e-12,
) -> float:
    """HCBH Eq. (3) for two points in the open Poincare ball.

    The paper prints ``epsilon_2`` (epsilon with subscript 2) as an additive
    denominator stabilizer, not epsilon squared.
    ``domain_tolerance`` is used only for floating-point round-off at acosh(1).
    """

    a = _finite_nonnegative(norm_left_sq, "norm_left_sq")
    b = _finite_nonnegative(norm_right_sq, "norm_right_sq")
    d2 = _finite_nonnegative(squared_euclidean, "squared_euclidean")
    if a >= 1.0 or b >= 1.0:
        raise ValueError("hyperbolic distance requires points inside the open unit ball")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    denominator = (1.0 - a) * (1.0 - b) + epsilon
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError("invalid hyperbolic-distance denominator")
    argument = 1.0 + 2.0 * d2 / denominator
    if not np.isfinite(argument):
        raise ValueError("hyperbolic-distance acosh argument is not finite")
    if argument < 1.0 - domain_tolerance:
        raise ValueError("hyperbolic-distance acosh argument is outside its domain")
    return float(np.arccosh(max(argument, 1.0)))


@dataclass(frozen=True)
class Cluster:
    cluster_id: int
    members: tuple[int, ...]

    @property
    def start(self) -> int:
        return self.members[0]

    @property
    def stop(self) -> int:
        return self.members[-1] + 1

    @property
    def size(self) -> int:
        return len(self.members)


def _coefficient(cluster: Cluster, n_bands: int) -> np.ndarray:
    coefficient = np.zeros(n_bands, dtype=np.float64)
    coefficient[np.asarray(cluster.members, dtype=np.int64)] = 1.0 / cluster.size
    return coefficient


def _cluster_measurements(
    left: Cluster,
    right: Cluster,
    gram: np.ndarray,
    *,
    epsilon: float,
) -> tuple[float, float, float, float, float, float]:
    n_bands = gram.shape[0]
    left_c = _coefficient(left, n_bands)
    right_c = _coefficient(right, n_bands)
    left_norm_sq = _finite_nonnegative(left_c @ gram @ left_c, "left cluster norm squared")
    right_norm_sq = _finite_nonnegative(right_c @ gram @ right_c, "right cluster norm squared")
    delta = left_c - right_c
    distance_sq = _finite_nonnegative(delta @ gram @ delta, "cluster distance squared")
    h_value = hyperbolic_distance(
        left_norm_sq,
        right_norm_sq,
        distance_sq,
        epsilon=epsilon,
    )
    mu_left = left.size * np.sqrt(left_norm_sq)
    mu_right = right.size * np.sqrt(right_norm_sq)
    eta = float(h_value * mu_left * mu_right)
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError("adaptive hyperbolic distance must be finite and non-negative")
    return eta, h_value, float(mu_left), float(mu_right), left_norm_sq, right_norm_sq


def _edge_table(
    clusters: list[Cluster], gram: np.ndarray, *, epsilon: float
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for index in range(len(clusters) - 1):
        left, right = clusters[index], clusters[index + 1]
        eta, h_value, mu_left, mu_right, left_norm_sq, right_norm_sq = (
            _cluster_measurements(left, right, gram, epsilon=epsilon)
        )
        edges.append(
            {
                "index": index,
                "left_id": left.cluster_id,
                "right_id": right.cluster_id,
                "left_boundary": left.start,
                "eta": eta,
                "hyperbolic_distance": h_value,
                "mu_left": mu_left,
                "mu_right": mu_right,
                "left_norm_sq": left_norm_sq,
                "right_norm_sq": right_norm_sq,
            }
        )
    return edges


def _mutual_edge_indices(clusters: list[Cluster], edges: list[dict[str, Any]]) -> list[int]:
    nearest: list[int] = []
    for cluster_index in range(len(clusters)):
        candidates: list[int] = []
        if cluster_index > 0:
            candidates.append(cluster_index - 1)
        if cluster_index < len(clusters) - 1:
            candidates.append(cluster_index)
        nearest.append(
            min(candidates, key=lambda edge_index: (edges[edge_index]["eta"], edges[edge_index]["left_boundary"]))
        )
    return [
        edge_index
        for edge_index in range(len(edges))
        if nearest[edge_index] == edge_index and nearest[edge_index + 1] == edge_index
    ]


def _snapshot(clusters: list[Cluster]) -> list[list[int]]:
    return [list(cluster.members) for cluster in clusters]


def build_hierarchy(gram: Any, *, epsilon: float = 1e-12) -> dict[str, Any]:
    """Build the complete deterministic HCBH hierarchy from a scaled Gram matrix."""

    matrix = np.asarray(gram, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("gram must be a square matrix with at least two bands")
    if not np.isfinite(matrix).all():
        raise ValueError("gram must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("gram must be symmetric")
    matrix = (matrix + matrix.T) * 0.5
    diagonal = np.diag(matrix)
    if np.any(diagonal < -1e-12) or np.any(diagonal >= 1.0):
        raise ValueError("gram diagonal must describe points inside the open unit ball")

    n_bands = matrix.shape[0]
    clusters = [Cluster(index, (index,)) for index in range(n_bands)]
    snapshots: dict[str, list[list[int]]] = {str(n_bands): _snapshot(clusters)}
    linkage: list[list[float]] = []
    merges: list[dict[str, Any]] = []
    next_id = n_bands

    while len(clusters) > 1:
        edges = _edge_table(clusters, matrix, epsilon=epsilon)
        mutual = _mutual_edge_indices(clusters, edges)
        candidate_edges = mutual or [
            min(range(len(edges)), key=lambda index: (edges[index]["eta"], edges[index]["left_boundary"]))
        ]
        candidate_ids = [
            (edges[index]["left_id"], edges[index]["right_id"]) for index in sorted(
                candidate_edges,
                key=lambda index: (edges[index]["eta"], edges[index]["left_boundary"]),
            )
        ]
        used_fallback = not bool(mutual)

        for left_id, right_id in candidate_ids:
            position = next(
                (
                    index
                    for index in range(len(clusters) - 1)
                    if clusters[index].cluster_id == left_id
                    and clusters[index + 1].cluster_id == right_id
                ),
                None,
            )
            if position is None:
                continue
            current_edges = _edge_table(clusters, matrix, epsilon=epsilon)
            edge = current_edges[position]
            left, right = clusters[position], clusters[position + 1]
            merged = Cluster(next_id, left.members + right.members)
            event = {
                "step": len(merges) + 1,
                "clusters_before": len(clusters),
                "left_id": left.cluster_id,
                "right_id": right.cluster_id,
                "new_id": next_id,
                "left_members": list(left.members),
                "right_members": list(right.members),
                "members": list(merged.members),
                "eta": edge["eta"],
                "hyperbolic_distance": edge["hyperbolic_distance"],
                "mu_left": edge["mu_left"],
                "mu_right": edge["mu_right"],
                "fallback": used_fallback,
            }
            merges.append(event)
            linkage.append(
                [float(left.cluster_id), float(right.cluster_id), edge["eta"], float(merged.size)]
            )
            clusters[position : position + 2] = [merged]
            next_id += 1
            snapshots[str(len(clusters))] = _snapshot(clusters)
            if len(clusters) == 1:
                break

    return {
        "implementation_version": IMPLEMENTATION_VERSION,
        "n_bands": n_bands,
        "epsilon": float(epsilon),
        "linkage": linkage,
        "merges": merges,
        "snapshots": snapshots,
    }


def representatives(
    hierarchy: dict[str, Any], origin_distances: Any, n_bands: int
) -> tuple[list[int], list[int], list[list[int]]]:
    distances = np.asarray(origin_distances, dtype=np.float64)
    total_bands = int(hierarchy["n_bands"])
    if distances.shape != (total_bands,) or not np.isfinite(distances).all():
        raise ValueError("origin_distances has an invalid shape or values")
    if isinstance(n_bands, bool) or not isinstance(n_bands, (int, np.integer)):
        raise ValueError("n_bands must be an integer")
    n_bands = int(n_bands)
    if not 1 <= n_bands <= total_bands:
        raise ValueError("n_bands must be between one and the fitted band count")
    clusters = hierarchy["snapshots"][str(n_bands)]
    order = [
        min(members, key=lambda index: (float(distances[index]), int(index)))
        for members in clusters
    ]
    return sorted(order), order, clusters
