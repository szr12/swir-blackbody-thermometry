from __future__ import annotations

from pathlib import Path

import numpy as np

from .selector import HsicMrnrSelector, mutual_information_matrix


def make_selector(selector_cfg: dict, band_count: int, lam: float, seed: int) -> HsicMrnrSelector:
    """根据配置创建HSIC-mRNR选带器。"""
    return HsicMrnrSelector(
        band_count=band_count,
        lam=lam,
        eta=selector_cfg["eta"],
        max_iter=selector_cfg["max_iter"],
        min_iter=selector_cfg["min_iter"],
        tolerance=selector_cfg["tolerance"],
        mi_neighbors=selector_cfg["mi_neighbors"],
        device=selector_cfg["device"],
        seed=seed,
    )


def load_or_compute_mi(path: Path, X: np.ndarray, neighbors: int, seed: int) -> np.ndarray:
    """缓存并复用训练集的两两互信息矩阵。"""
    if path.exists():
        matrix = np.load(path, allow_pickle=False)
        if matrix.shape != (X.shape[1], X.shape[1]) or not np.isfinite(matrix).all():
            raise RuntimeError(f"互信息缓存不合法: {path}")
        return np.asarray(matrix, dtype=np.float32)
    matrix = mutual_information_matrix(X, neighbors, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, matrix, allow_pickle=False)
    temporary.replace(path)
    return matrix
