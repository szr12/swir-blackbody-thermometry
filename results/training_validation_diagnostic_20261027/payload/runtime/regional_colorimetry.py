from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convergent_colorimetry import FixedBandColorimetricMLP


def regional_training_masks(
    temperatures_c: np.ndarray,
    boundaries_c: np.ndarray,
    overlap_c: float,
) -> list[np.ndarray]:
    """生成带边界重叠的温区训练掩码。"""
    temperatures = np.asarray(temperatures_c, dtype=np.float64)
    boundaries = np.asarray(boundaries_c, dtype=np.float64)
    if boundaries.ndim != 1 or boundaries.size < 3 or np.any(np.diff(boundaries) <= 0.0):
        raise ValueError("温区边界必须严格递增且至少包含三个点")
    if overlap_c < 0.0:
        raise ValueError("温区重叠宽度不能为负数")
    masks = []
    for region_id in range(boundaries.size - 1):
        lower = boundaries[region_id] - (overlap_c if region_id > 0 else 0.0)
        upper = boundaries[region_id + 1] + (overlap_c if region_id < boundaries.size - 2 else 0.0)
        masks.append((temperatures >= lower) & (temperatures <= upper))
    return masks


def soft_region_weights(
    coarse_temperature_c: np.ndarray,
    boundaries_c: np.ndarray,
    routing_width_c: float,
) -> np.ndarray:
    """仅根据粗预测温度计算各温区专家的软路由权重。"""
    coarse = np.asarray(coarse_temperature_c, dtype=np.float64).reshape(-1)
    boundaries = np.asarray(boundaries_c, dtype=np.float64)
    if routing_width_c <= 0.0:
        raise ValueError("软路由宽度必须大于零")
    centers = 0.5 * (boundaries[:-1] + boundaries[1:])
    logits = -0.5 * ((coarse[:, None] - centers[None, :]) / routing_width_c) ** 2
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / np.sum(weights, axis=1, keepdims=True)


@dataclass(frozen=True)
class RegionalFitDiagnostic:
    region_id: int
    lower_c: float
    upper_c: float
    n_training_samples: int
    training_rmse_c: float
    training_adaptive_rmse_c: float
    success: bool
    iterations: int
    attention_weight_min: float
    attention_weight_max: float
    attention_weight_mean: float


class RegionalColorimetricEnsemble:
    """用全局粗预测对多个独立温区比色回归器进行软路由。"""

    def __init__(
        self,
        *,
        boundaries_c: np.ndarray,
        overlap_c: float,
        routing_width_c: float,
        model_parameters: dict,
        seed: int,
    ) -> None:
        self.boundaries_c = np.asarray(boundaries_c, dtype=np.float64)
        self.overlap_c = float(overlap_c)
        self.routing_width_c = float(routing_width_c)
        self.model_parameters = dict(model_parameters)
        self.seed = int(seed)

    def _model(self, seed: int) -> FixedBandColorimetricMLP:
        return FixedBandColorimetricMLP(seed=seed, **self.model_parameters)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        wavelengths_nm: np.ndarray,
    ) -> RegionalColorimetricEnsemble:
        values = np.asarray(X, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        self.global_model_ = self._model(self.seed).fit(values, target, wavelengths_nm)
        self.experts_: list[FixedBandColorimetricMLP] = []
        self.diagnostics_: list[RegionalFitDiagnostic] = []
        masks = regional_training_masks(target, self.boundaries_c, self.overlap_c)
        for region_id, mask in enumerate(masks):
            if int(mask.sum()) < 10:
                raise ValueError(f"温区 {region_id} 的训练样本不足")
            expert = self._model(self.seed + region_id + 1).fit(
                values[mask], target[mask], wavelengths_nm
            )
            self.experts_.append(expert)
            self.diagnostics_.append(RegionalFitDiagnostic(
                region_id=region_id,
                lower_c=float(self.boundaries_c[region_id]),
                upper_c=float(self.boundaries_c[region_id + 1]),
                n_training_samples=int(mask.sum()),
                training_rmse_c=float(expert.training_rmse_c_),
                training_adaptive_rmse_c=float(expert.training_adaptive_rmse_c_),
                success=bool(expert.success_),
                iterations=int(expert.iterations_),
                attention_weight_min=float(np.min(expert.training_attention_weights_)),
                attention_weight_max=float(np.max(expert.training_attention_weights_)),
                attention_weight_mean=float(np.mean(expert.training_attention_weights_)),
            ))
        return self

    @property
    def success_(self) -> bool:
        return bool(self.global_model_.success_ and all(row.success for row in self.diagnostics_))

    def predict_components(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coarse = self.global_model_.predict(X)
        expert_predictions = np.column_stack([expert.predict(X) for expert in self.experts_])
        weights = soft_region_weights(coarse, self.boundaries_c, self.routing_width_c)
        return coarse, expert_predictions, weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, expert_predictions, weights = self.predict_components(X)
        return np.sum(expert_predictions * weights, axis=1)
