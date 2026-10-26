"""Dropout CAE 的训练与选择接口。"""

from __future__ import annotations

import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .core import PAPER_PROFILES, DropoutCAE, paper_loss, paper_temperature_rate


class DropoutCAESelector:
    """论文优先的 Dropout CAE 波段选择器。"""

    def __init__(
        self,
        n_bands: int,
        *,
        profile: str = "T1",
        device: str = "auto",
        random_state: int = 100,
    ) -> None:
        if profile not in PAPER_PROFILES:
            raise ValueError(f"未知配置：{profile}")
        self.n_bands = int(n_bands)
        self.profile = PAPER_PROFILES[profile]
        self.device_request = device
        self.random_state = int(random_state)
        self._fitted = False

    @staticmethod
    def _validate(X: np.ndarray, n_bands: int, batch_size: int) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X 必须是二维数组 [样本, 波段]")
        if X.shape[0] < batch_size:
            raise ValueError("样本数小于论文配置的批量大小")
        if not 0 < n_bands <= X.shape[1]:
            raise ValueError("目标波段数越界")
        if not np.isfinite(X).all():
            raise ValueError("X 包含 NaN 或 Inf")
        return X

    def _resolve_device(self) -> torch.device:
        if self.device_request == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(self.device_request)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA，但当前环境不可用")
        return device

    def fit(self, X_train: np.ndarray) -> "DropoutCAESelector":
        """只用训练数据拟合固定 K 的论文模型。"""

        X = self._validate(X_train, self.n_bands, self.profile.batch_size)
        device = self._resolve_device()
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_state)
        torch.use_deterministic_algorithms(True)

        minimum = X.min(axis=0)
        maximum = X.max(axis=0)
        span = maximum - minimum
        constant = np.flatnonzero(span == 0)
        safe_span = np.where(span == 0, 1.0, span)
        normalized = (X - minimum) / safe_span
        normalized[:, constant] = 0.0

        tensor = torch.from_numpy(normalized).to(device)
        model = DropoutCAE(X.shape[1], self.n_bands).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(self.profile.milestones), gamma=0.1
        )
        temperature = self.profile.start_temperature
        temperature_rate = paper_temperature_rate(
            self.profile.start_temperature,
            self.profile.end_temperature,
            self.profile.batch_size,
            X.shape[0],
            self.profile.epochs,
        )
        order_generator = torch.Generator(device="cpu").manual_seed(self.random_state)
        gate_generator = torch.Generator(device=device).manual_seed(self.random_state + 1)
        history: list[dict[str, float | int]] = []
        started = time.perf_counter()

        for epoch in range(self.profile.epochs):
            order = torch.randperm(X.shape[0], generator=order_generator)
            total_loss = total_rec = total_reg = 0.0
            batches = 0
            # 官方实现 drop_last=True；不足一个完整批次的尾部样本不进入该轮。
            stop = X.shape[0] - X.shape[0] % self.profile.batch_size
            for begin in range(0, stop, self.profile.batch_size):
                indices = order[begin : begin + self.profile.batch_size].to(device)
                batch = tensor[indices]
                optimizer.zero_grad(set_to_none=True)
                reconstruction, mask, _ = model(batch, temperature, gate_generator)
                loss, rec, reg = paper_loss(batch, reconstruction, mask)
                loss.backward()
                optimizer.step()
                temperature *= temperature_rate
                total_loss += float(loss.detach().cpu())
                total_rec += float(rec.detach().cpu())
                total_reg += float(reg.detach().cpu())
                batches += 1
            scheduler.step()
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": total_loss / batches,
                    "reconstruction": total_rec / batches,
                    "regularization": total_reg / batches,
                    "temperature": float(temperature),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )

        elapsed = time.perf_counter() - started
        theta = model.gate.theta.detach().cpu().numpy().astype(float)
        ranking = np.lexsort((np.arange(theta.size), theta)).astype(int)
        self.model_ = model
        self.selected_order_ = ranking[: self.n_bands]
        self.selected_indices_ = np.sort(self.selected_order_)
        self.diagnostics_ = {
            "method": "Dropout CAE paper-formula reproduction",
            "profile": self.profile.to_dict(),
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_selected": self.n_bands,
            "training_min": minimum.astype(float).tolist(),
            "training_max": maximum.astype(float).tolist(),
            "constant_bands": constant.astype(int).tolist(),
            "paper_theta": theta.tolist(),
            "ranking": ranking.tolist(),
            "selected_order": self.selected_order_.tolist(),
            "selected_indices": self.selected_indices_.tolist(),
            "history": history,
            "elapsed_seconds": elapsed,
            "actual_device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "random_state": self.random_state,
            "reconstruction_gradient_to_gate": False,
            "fidelity_note": "硬 Top-K 与官方代码一致；重建项不向门参数传播梯度，未添加任何替代梯度。",
        }
        self._fitted = True
        return self

    def select(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("必须先调用 fit")
        return self.selected_indices_.copy()

    def save_diagnostics(self, path: str | Path) -> None:
        if not self._fitted:
            raise RuntimeError("必须先调用 fit")
        Path(path).write_text(json.dumps(self.diagnostics_, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def band_selection_tabular(
    X_2d: np.ndarray,
    n_bands: int,
    *,
    profile: str = "T1",
    device: str = "auto",
    random_state: int = 100,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    selector = DropoutCAESelector(
        n_bands, profile=profile, device=device, random_state=random_state
    ).fit(X_2d)
    selected = selector.select()
    if return_diagnostics:
        return selected, selector.diagnostics_
    return selected
