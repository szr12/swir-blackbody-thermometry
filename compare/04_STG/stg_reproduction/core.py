"""Yamada 等（ICML 2020）STG 回归模型的核心公式。"""

from __future__ import annotations

import math

import torch
from torch import nn


class StochasticGate(nn.Module):
    """论文式高斯随机门：z = clip(mu + epsilon, 0, 1)。"""

    def __init__(self, input_dim: int, sigma: float = 0.5) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim 必须为正整数")
        if sigma <= 0:
            raise ValueError("sigma 必须大于 0")
        self.mu = nn.Parameter(torch.full((input_dim,), 0.5))
        self.sigma = float(sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = torch.randn_like(self.mu) * self.sigma
            gates = torch.clamp(self.mu + noise, 0.0, 1.0)
        else:
            gates = self.deterministic_gates()
        return x * gates

    def deterministic_gates(self) -> torch.Tensor:
        """按论文去除随机性后的门值。"""
        return torch.clamp(self.mu, 0.0, 1.0)

    def active_probabilities(self) -> torch.Tensor:
        """计算 P(z_d > 0) = Phi(mu_d / sigma)。"""
        scaled = self.mu / (self.sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + torch.erf(scaled))


class STGRegressor(nn.Module):
    """带随机门的全连接回归网络。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (500, 50, 10),
        sigma: float = 0.5,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.gate = StochasticGate(input_dim=input_dim, sigma=sigma)
        if activation == "tanh":
            activation_factory = nn.Tanh
        elif activation == "relu":
            activation_factory = nn.ReLU
        elif activation == "sigmoid":
            activation_factory = nn.Sigmoid
        else:
            raise ValueError("activation 必须为 tanh、relu 或 sigmoid")

        dimensions = (input_dim, *hidden_dims, 1)
        layers: list[nn.Module] = []
        for position, (source, target) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(source, target))
            if position < len(dimensions) - 2:
                layers.append(activation_factory())
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """按补充材料使用 N(0, 0.1) 权重和零偏置。"""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(self.gate(x)).squeeze(-1)

    def objective(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        regularization: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction = self(x)
        mse = torch.mean((prediction - y) ** 2)
        # 补充材料说明实际实现按特征数 D 归一化正则项。
        l0_surrogate = self.gate.active_probabilities().mean()
        total = mse + float(regularization) * l0_surrogate
        return total, mse, l0_surrogate
