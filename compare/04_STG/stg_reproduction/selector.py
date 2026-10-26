"""连续回归 STG 选带器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import platform
import time
from typing import Any

import numpy as np
import torch

from .core import STGRegressor


@dataclass(frozen=True)
class STGConfig:
    """官方回归示例中公开的固定训练配置。"""

    hidden_dims: tuple[int, ...] = (500, 50, 10)
    activation: str = "tanh"
    sigma: float = 0.5
    regularization: float = 0.1
    learning_rate: float = 0.1
    weight_decay: float = 1e-3
    epochs: int = 3000
    selection_threshold: float = 0.5


class STGSelector:
    """只用训练数据拟合的 STG 连续回归特征选择器。"""

    def __init__(
        self,
        *,
        device: str = "auto",
        random_state: int = 1,
        config: STGConfig | None = None,
    ) -> None:
        self.random_state = int(random_state)
        self.config = config or STGConfig()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("device 必须为 auto、cpu 或 cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求了 CUDA，但当前环境不可用")
        self.device = torch.device(device)
        self.model_: STGRegressor | None = None
        self.diagnostics_: dict[str, Any] | None = None

    @staticmethod
    def _validate(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(X, dtype=np.float32)
        target = np.asarray(y, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError("X 必须为二维数组 [N,D]")
        if target.ndim != 1:
            raise ValueError("y 必须为一维连续目标")
        if features.shape[0] != target.shape[0] or features.shape[0] < 2:
            raise ValueError("X 与 y 样本数必须一致且至少包含两个样本")
        if features.shape[1] < 1:
            raise ValueError("X 至少需要一个特征")
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise ValueError("X 和 y 不得包含 NaN 或 Inf")
        if np.ptp(target) == 0:
            raise ValueError("连续目标 y 不能为常量")
        return features, target

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "STGSelector":
        X, y = self._validate(X_train, y_train)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_state)

        model = STGRegressor(
            input_dim=X.shape[1],
            hidden_dims=self.config.hidden_dims,
            sigma=self.config.sigma,
            activation=self.config.activation,
        ).to(self.device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        x_tensor = torch.from_numpy(X).to(self.device)
        y_tensor = torch.from_numpy(y).to(self.device)
        history: list[dict[str, float | int]] = []
        started = time.perf_counter()
        for epoch in range(1, self.config.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total, mse, l0_surrogate = model.objective(
                x_tensor,
                y_tensor,
                self.config.regularization,
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"第 {epoch} 个 epoch 出现非有限损失")
            total.backward()
            optimizer.step()
            history.append(
                {
                    "epoch": epoch,
                    "objective": float(total.detach().cpu()),
                    "mse": float(mse.detach().cpu()),
                    "l0_surrogate": float(l0_surrogate.detach().cpu()),
                }
            )

        model.eval()
        gates = model.gate.deterministic_gates().detach().cpu().numpy()
        active_probabilities = model.gate.active_probabilities().detach().cpu().numpy()
        selected = np.flatnonzero(gates >= self.config.selection_threshold).astype(np.int64)
        elapsed = time.perf_counter() - started
        self.model_ = model
        self.diagnostics_ = {
            "method": "STG regression",
            "paper_profile": "official regression example",
            "config": asdict(self.config),
            "random_state": self.random_state,
            "device": str(self.device),
            "selected_indices": selected.tolist(),
            "deterministic_gates": gates.tolist(),
            "active_probabilities": active_probabilities.tolist(),
            "mu": model.gate.mu.detach().cpu().numpy().tolist(),
            "history": history,
            "elapsed_seconds": elapsed,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
        return self

    def select(self) -> np.ndarray:
        if self.diagnostics_ is None:
            raise RuntimeError("必须先调用 fit")
        selected = np.asarray(self.diagnostics_["selected_indices"], dtype=np.int64)
        if selected.size == 0:
            raise RuntimeError("论文阈值未选出任何特征；不得擅自回退为 Top-K")
        return selected.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("必须先调用 fit")
        features = np.asarray(X, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.model_.gate.mu.numel():
            raise ValueError("X 必须为与训练特征数一致的二维数组")
        if not np.isfinite(features).all():
            raise ValueError("X 不得包含 NaN 或 Inf")
        self.model_.eval()
        with torch.no_grad():
            prediction = self.model_(torch.from_numpy(features).to(self.device))
        return prediction.detach().cpu().numpy().astype(np.float64, copy=False)


def band_selection_regression(
    X_2d: np.ndarray,
    y_1d: np.ndarray,
    *,
    device: str = "auto",
    random_state: int = 1,
    config: STGConfig | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    selector = STGSelector(device=device, random_state=random_state, config=config).fit(X_2d, y_1d)
    selected = selector.select()
    if return_diagnostics:
        assert selector.diagnostics_ is not None
        return selected, selector.diagnostics_
    return selected
