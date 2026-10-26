from __future__ import annotations

import math

import torch
from torch import nn


def stable_topk_indices(values: torch.Tensor, k: int) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not 0 < k <= values.numel():
        raise ValueError("k must be in [1, number of values]")
    # Stable descending order makes a smaller band index win exact ties.
    return torch.argsort(values, descending=True, stable=True)[:k]


def hard_topk_straight_through(values: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary Top-K in the forward pass and identity gradient in backward.

    The paper specifies a binary Top-K gate but does not disclose the gradient
    estimator for that discontinuous operation.  This explicit STE is the sole
    non-paper optimization assumption used by this reproduction.
    """

    indices = stable_topk_indices(values, k)
    hard = torch.zeros_like(values)
    hard.scatter_(0, indices, 1.0)
    mask = hard + values - values.detach()
    return mask, indices


def gaussian_open_probability(mu: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return 0.5 * (1.0 + torch.erf(mu / (sigma * math.sqrt(2.0))))


def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if x_hat.shape != x.shape or x.ndim != 2:
        raise ValueError("x_hat and x must have the same two-dimensional shape")
    return 0.5 * torch.sum((x - x_hat) ** 2) / x.shape[0]


class SGAEDecoder(nn.Module):
    def __init__(self, n_bands: int, target_bands: int) -> None:
        super().__init__()
        if not 0 < target_bands <= n_bands:
            raise ValueError("target_bands must be in [1, n_bands]")
        hidden_1 = target_bands + (n_bands - target_bands) // 3
        hidden_2 = target_bands + (2 * (n_bands - target_bands)) // 3
        self.layers = nn.Sequential(
            nn.Linear(target_bands, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, n_bands),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SGAEModel(nn.Module):
    def __init__(self, n_bands: int, target_bands: int, sigma: float) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.target_bands = target_bands
        self.sigma = float(sigma)
        # "Treating each atom equally" and Fig. 2's initial 0.5 regularizer
        # together imply zero initialization (Phi(0) = 0.5).
        self.mu = nn.Parameter(torch.zeros(n_bands))
        self.decoder = SGAEDecoder(n_bands, target_bands)

    def forward(
        self, x: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        noise = torch.randn(
            self.mu.shape,
            dtype=self.mu.dtype,
            device=self.mu.device,
            generator=generator,
        ) * self.sigma
        z = self.mu + noise
        mask, indices = hard_topk_straight_through(z, self.target_bands)
        selected = (x * mask.unsqueeze(0)).index_select(1, indices)
        return self.decoder(selected), z, mask, indices

    def regularization(self) -> torch.Tensor:
        # Eq. 6 is typeset as mean(sum(Phi)); Fig. 2 starts at 0.5, proving
        # that the operational statistic is mean(Phi(mu/sigma)).
        return gaussian_open_probability(self.mu, self.sigma).mean()

    @torch.no_grad()
    def deterministic_indices(self) -> torch.Tensor:
        return stable_topk_indices(self.mu, self.target_bands)

