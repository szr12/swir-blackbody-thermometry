from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
from sklearn.feature_selection import mutual_info_regression


def _rbf_kernel(x: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    norm = (x * x).sum(dim=1, keepdim=True)
    distance = torch.clamp(norm + norm.T - 2.0 * (x @ x.T), min=0.0)
    if sigma is None:
        valid = distance[torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)]
        valid = valid[valid > 1e-8]
        sigma = float(torch.sqrt(torch.median(valid)).item()) if valid.numel() else 1.0
    sigma = max(float(sigma), 1e-8)
    return torch.exp(-distance / (2.0 * sigma * sigma))


def _centered_hsic(kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
    """计算参考实现采用的中心化 HSIC 估计量。"""
    n = kx.shape[0]
    h = torch.eye(n, device=kx.device, dtype=kx.dtype) - torch.ones(
        (n, n), device=kx.device, dtype=kx.dtype
    ) / n
    return torch.trace((h @ kx @ h) @ (h @ ky @ h)) / ((n - 1) ** 2)


def _project_capped_simplex(values: torch.Tensor, target: int) -> torch.Tensor:
    """投影到 0<=xi<=1 且 sum(xi)=K 的截断单纯形。"""
    lower = float((values - 1.0).min().item())
    upper = float(values.max().item())
    for _ in range(80):
        middle = (lower + upper) / 2.0
        projected = torch.clamp(values - middle, 0.0, 1.0)
        if float(projected.sum().item()) > target:
            lower = middle
        else:
            upper = middle
    return torch.clamp(values - (lower + upper) / 2.0, 0.0, 1.0)


def mutual_information_matrix(X: np.ndarray, neighbors: int, seed: int) -> np.ndarray:
    """只用当前训练折估计波段间冗余矩阵。"""
    dimension = X.shape[1]
    matrix = np.empty((dimension, dimension), dtype=np.float32)
    for target in range(dimension):
        matrix[:, target] = mutual_info_regression(
            X, X[:, target], n_neighbors=neighbors, random_state=seed
        ).astype(np.float32)
    return (matrix + matrix.T) * 0.5


def _redundancy_from_reference(mi: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """按参考实现排除自互信息，并用 K(K-1) 归一化冗余项。"""
    dimension = weights.numel()
    off_diagonal = 1.0 - torch.eye(dimension, device=weights.device, dtype=weights.dtype)
    total = weights @ (mi * off_diagonal) @ weights
    estimated_k = weights.sum()
    denominator = estimated_k * (estimated_k - 1.0)
    if float(denominator.detach().item()) < 1e-8:
        return torch.zeros((), device=weights.device, dtype=weights.dtype)
    return total / denominator


@dataclass(frozen=True)
class SelectionResult:
    selected_indices: list[int]
    weights: np.ndarray
    iterations: int
    objective: float
    sigma_x: float
    elapsed_seconds: float


class HsicMrnrSelector:
    def __init__(
        self,
        *,
        band_count: int,
        lam: float,
        eta: float = 0.02,
        max_iter: int = 300,
        min_iter: int = 10,
        tolerance: float = 1e-8,
        mi_neighbors: int = 3,
        device: str = "cuda",
        seed: int = 42,
    ) -> None:
        self.band_count = int(band_count)
        self.lam = float(lam)
        self.eta = float(eta)
        self.max_iter = int(max_iter)
        self.min_iter = int(min_iter)
        self.tolerance = float(tolerance)
        self.mi_neighbors = int(mi_neighbors)
        self.device = device
        self.seed = int(seed)

    def fit(self, X: np.ndarray, y: np.ndarray, mi_matrix: np.ndarray | None = None) -> SelectionResult:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n, dimension = X.shape
        if y.shape != (n,) or n < 3 or not 1 <= self.band_count <= dimension:
            raise ValueError("选择器输入形状或 K 不合法")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("选择器输入包含非有限值")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("配置要求 CUDA，但当前 PyTorch 无法使用 CUDA")

        started = time.perf_counter()
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        if mi_matrix is None:
            mi_matrix = mutual_information_matrix(X, self.mi_neighbors, self.seed)
        mi = torch.as_tensor(mi_matrix, dtype=torch.float32, device=self.device)
        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        target = torch.as_tensor(y[:, None], dtype=torch.float32, device=self.device)
        ky = _rbf_kernel(target)
        xi = torch.full((dimension,), self.band_count / dimension, device=self.device)

        with torch.no_grad():
            weighted = x * xi[None, :]
            norm = (weighted * weighted).sum(dim=1, keepdim=True)
            distance = torch.clamp(norm + norm.T - 2.0 * (weighted @ weighted.T), min=0.0)
            triangle = distance[torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)]
            valid = triangle[triangle > 1e-8]
            sigma_x = float(torch.sqrt(torch.median(valid)).item()) if valid.numel() else 1.0
            sigma_x = max(sigma_x, 1e-8)

        objective_value = float("nan")
        completed = 0
        for iteration in range(self.max_iter):
            xi = xi.detach().requires_grad_(True)
            kx = _rbf_kernel(x * xi[None, :], sigma_x)
            redundancy = _redundancy_from_reference(mi, xi)
            objective = _centered_hsic(kx, ky) - self.lam * redundancy
            objective.backward()
            with torch.no_grad():
                updated = _project_capped_simplex(xi + self.eta * xi.grad, self.band_count)
                new_kx = _rbf_kernel(x * updated[None, :], sigma_x)
                new_redundancy = _redundancy_from_reference(mi, updated)
                new_objective = _centered_hsic(new_kx, ky) - self.lam * new_redundancy
                difference = abs(float(new_objective.item()) - float(objective.item()))
                objective_value = float(new_objective.item())
            xi = updated.detach()
            completed = iteration + 1
            if completed >= self.min_iter and difference < self.tolerance:
                break

        weights = xi.cpu().numpy()
        selected = np.argsort(-weights, kind="stable")[: self.band_count].astype(int).tolist()
        return SelectionResult(
            selected_indices=selected,
            weights=weights,
            iterations=completed,
            objective=objective_value,
            sigma_x=sigma_x,
            elapsed_seconds=time.perf_counter() - started,
        )
