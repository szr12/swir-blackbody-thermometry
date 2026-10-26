"""Dropout CAE 的论文核心组件。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TrainingProfile:
    """论文退火配置。"""

    name: str
    start_temperature: float
    end_temperature: float
    epochs: int
    batch_size: int
    milestones: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PAPER_PROFILES = {
    "T1": TrainingProfile("T1", 1.0, 0.001, 40, 1, (15, 30)),
    "T2": TrainingProfile("T2", 1.0, 0.001, 200, 256, (15, 30)),
    "T3": TrainingProfile("T3", 1.0, 0.01, 200, 32, (15, 30)),
    # 快速配置只用于单元测试和流程冒烟，不属于论文实验配置。
    "quick": TrainingProfile("quick", 1.0, 0.1, 4, 8, (2, 3)),
}


def paper_temperature_rate(
    start_temperature: float,
    end_temperature: float,
    batch_size: int,
    n_samples: int,
    epochs: int,
) -> float:
    """按论文式(6)计算每个批次的温度衰减倍率。"""

    if not 0 < end_temperature <= start_temperature:
        raise ValueError("温度必须满足 0 < 终止温度 <= 初始温度")
    if min(batch_size, n_samples, epochs) <= 0:
        raise ValueError("批量、样本数和轮数必须为正数")
    return float((end_temperature / start_temperature) ** (batch_size / (n_samples * epochs)))


class BinaryConcreteGate(nn.Module):
    """论文式(3)的二元 Concrete 门。"""

    def __init__(self, n_features: int, init: float = 0.99, implicit_temperature: float = 0.2):
        super().__init__()
        if n_features <= 0 or not 0 < init < 1 or implicit_temperature <= 0:
            raise ValueError("门参数非法")
        initial_logit = -torch.log(1.0 / torch.tensor(init) - 1.0) * implicit_temperature
        self.logits = nn.Parameter(torch.full((n_features,), float(initial_logit)))
        self.implicit_temperature = float(implicit_temperature)

    @property
    def theta(self) -> torch.Tensor:
        """返回论文和官方代码中用于排序的可学习率。"""

        return torch.sigmoid(self.logits / self.implicit_temperature)

    def sample(self, batch_size: int, temperature: float, generator: torch.Generator | None = None) -> torch.Tensor:
        """采样形状为 ``[batch_size, n_features]`` 的 Concrete 掩码。"""

        if batch_size <= 0 or temperature <= 0:
            raise ValueError("批量和温度必须为正数")
        u = torch.rand(
            (batch_size, self.logits.numel()),
            dtype=self.logits.dtype,
            device=self.logits.device,
            generator=generator,
        )
        eps = torch.finfo(u.dtype).eps
        u = u.clamp(eps, 1.0 - eps)
        logistic_noise = torch.log(u) - torch.log1p(-u)
        return torch.sigmoid((self.logits / self.implicit_temperature + logistic_noise) / temperature)

    def selected_indices(self, n_bands: int) -> torch.Tensor:
        """按论文伪代码取最小的 ``n_bands`` 个率，并稳定处理并列。"""

        if not 0 < n_bands <= self.logits.numel():
            raise ValueError("目标波段数越界")
        return torch.argsort(self.theta, stable=True)[:n_bands]


class Decoder(nn.Module):
    """论文所述的两层全连接解码器。"""

    def __init__(self, n_selected: int, n_features: int, hidden_dim: int = 128):
        super().__init__()
        self.hidden = nn.Linear(n_selected, hidden_dim)
        self.output = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.output(torch.relu(self.hidden(x))))


class DropoutCAE(nn.Module):
    """严格保留论文硬 Top-K 数据通路的 Dropout CAE。"""

    def __init__(self, n_features: int, n_selected: int, hidden_dim: int = 128):
        super().__init__()
        self.n_selected = int(n_selected)
        self.gate = BinaryConcreteGate(n_features)
        self.decoder = Decoder(n_selected, n_features, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        temperature: float,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = self.gate.sample(x.shape[0], temperature, generator)
        selected = self.gate.selected_indices(self.n_selected)
        reconstruction = self.decoder(x[:, selected])
        return reconstruction, mask, selected


def paper_loss(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    mask: torch.Tensor,
    regularization: float = 0.005,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """实现论文式(5)：单边交叉熵与 Concrete 掩码正则。"""

    if target.shape != reconstruction.shape or mask.shape != target.shape:
        raise ValueError("目标、重建和掩码形状必须一致")
    if regularization < 0:
        raise ValueError("正则系数不能为负数")
    eps = torch.finfo(reconstruction.dtype).eps
    reconstruction_term = -(target * torch.log(reconstruction.clamp_min(eps))).sum(dim=1).mean()
    regularization_term = mask.sum(dim=1).mean()
    total = reconstruction_term + regularization * regularization_term
    return total, reconstruction_term, regularization_term
