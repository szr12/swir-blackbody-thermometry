from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class SGAEConfig:
    """Training configuration.

    The paper reports lr=1e-4, max_epochs=50 and tau=1e-3.  Sigma was
    dataset-specific (0.01 for three datasets and 0.02 for Botswana), so the
    tabular adapter uses 0.01 unless explicitly overridden.
    """

    learning_rate: float = 1e-4
    sigma: float = 0.01
    stopping_threshold: float = 1e-3
    max_epochs: int = 50
    optimizer: str = "adam"
    gradient_estimator: str = "straight_through_topk"

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.stopping_threshold < 0:
            raise ValueError("stopping_threshold must be non-negative")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.optimizer != "adam":
            raise ValueError("only the documented reproduction assumption 'adam' is supported")
        if self.gradient_estimator != "straight_through_topk":
            raise ValueError("only straight_through_topk is supported")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_profile(name: str = "paper", **overrides: Any) -> SGAEConfig:
    if name == "paper":
        config = SGAEConfig()
    elif name == "quick":
        config = SGAEConfig(max_epochs=8, stopping_threshold=0.0)
    else:
        raise ValueError("profile must be 'paper' or 'quick'")
    if overrides:
        unknown = set(overrides) - set(config.to_dict())
        if unknown:
            raise TypeError(f"unknown configuration fields: {sorted(unknown)}")
        config = replace(config, **overrides)
    config.validate()
    return config

