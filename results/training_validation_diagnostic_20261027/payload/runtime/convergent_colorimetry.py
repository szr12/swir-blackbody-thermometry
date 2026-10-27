from __future__ import annotations

from dataclasses import dataclass
import copy
import time

import numpy as np
import torch

from blackbody_hsic.selector import _project_capped_simplex, _redundancy_from_reference


@dataclass(frozen=True)
class ConvergentSelectionResult:
    selected_indices: np.ndarray
    weights: np.ndarray
    success: bool
    stop_reason: str
    iterations: int
    training_rmse_c: float
    training_adaptive_rmse_c: float
    validation_rmse_c: float
    validation_adaptive_rmse_c: float
    trace: list[dict[str, float]]
    elapsed_seconds: float


class _TemperatureMLP(torch.nn.Module):
    def __init__(self, input_dimension: int, hidden_units: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        previous = input_dimension
        for width in hidden_units:
            layers.extend([torch.nn.Linear(previous, width), torch.nn.SiLU()])
            previous = width
        layers.append(torch.nn.Linear(previous, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


class _SampleAttention(torch.nn.Module):
    """由比色特征生成每个样本的正注意力权重。"""

    def __init__(self, input_dimension: int, hidden_units: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, hidden_units),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_units, 1),
            torch.nn.Softplus(),
        )

    def forward(
        self,
        values: torch.Tensor,
        minimum_weight: float,
        maximum_weight: float,
    ) -> torch.Tensor:
        raw = self.network(values).squeeze(1) + 1e-6
        normalized = raw / torch.mean(raw).clamp_min(1e-12)
        clipped = torch.clamp(normalized, min=minimum_weight, max=maximum_weight)
        return clipped / torch.mean(clipped).clamp_min(1e-12)


def attention_error_targets(
    squared_errors_c: torch.Tensor,
    *,
    power: float,
    minimum_weight: float,
    maximum_weight: float,
) -> torch.Tensor:
    """把当前训练误差转换成注意力网络的困难样本监督目标。"""
    difficulty = torch.sqrt(squared_errors_c.detach() + 1e-12)
    relative = difficulty / torch.mean(difficulty).clamp_min(1e-12)
    targets = torch.clamp(relative ** power, min=minimum_weight, max=maximum_weight)
    return targets / torch.mean(targets).clamp_min(1e-12)


def blend_adaptive_weights(
    attention_weights: torch.Tensor,
    error_teacher_weights: torch.Tensor,
    *,
    teacher_blend: float,
) -> torch.Tensor:
    """融合注意力预测权重与真实误差教师权重，并保持均值为1。"""
    if not 0.0 <= teacher_blend <= 1.0:
        raise ValueError("真实误差教师融合比例必须位于0到1之间")
    blended = (1.0 - teacher_blend) * attention_weights + teacher_blend * error_teacher_weights
    return blended / torch.mean(blended).clamp_min(1e-12)


def constrained_topk_indices(
    weights: torch.Tensor,
    band_count: int,
    required_masks: tuple[torch.Tensor, ...] = (),
) -> torch.Tensor:
    """选择Top-K波段，并保证每个物理波段区间至少保留一个波段。"""
    if weights.ndim != 1 or not 1 <= band_count <= weights.numel():
        raise ValueError("Top-K权重或波段数量不合法")
    selected = torch.zeros_like(weights, dtype=torch.bool)
    for mask in required_masks:
        current = mask.to(device=weights.device, dtype=torch.bool)
        if current.shape != weights.shape or not torch.any(current):
            raise ValueError("必选波段区间在当前候选波段中为空")
        if torch.any(selected & current):
            continue
        candidates = torch.nonzero(current & ~selected, as_tuple=False).flatten()
        chosen = candidates[torch.argmax(weights[candidates])]
        selected[chosen] = True
    remaining = band_count - int(selected.sum().item())
    if remaining < 0:
        raise ValueError("必选波段区间数量超过K")
    if remaining:
        scores = weights.masked_fill(selected, -torch.inf)
        selected[torch.topk(scores, remaining, sorted=False).indices] = True
    return torch.nonzero(selected, as_tuple=False).flatten()


def _prepare_log_radiance(
    train: np.ndarray,
    evaluation: np.ndarray,
    wavelengths_nm: np.ndarray,
    intensity_floor_fraction: float,
    absolute_intensity_floor: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """仅由训练数据估计偏移和标准化参数。"""
    scale = np.maximum(np.median(np.abs(train), axis=0), absolute_intensity_floor)
    floors = np.maximum(scale * intensity_floor_fraction, absolute_intensity_floor)
    offsets = np.maximum(floors - np.min(train, axis=0), 0.0)

    def transform(values: np.ndarray) -> np.ndarray:
        stabilized = np.maximum(values + offsets[None, :], floors[None, :])
        return np.log(stabilized) + 5.0 * np.log(wavelengths_nm[None, :])

    train_log = transform(train)
    evaluation_log = transform(evaluation)
    mean = train_log.mean(axis=0)
    std = train_log.std(axis=0, ddof=0)
    std[std < 1e-8] = 1.0
    return (train_log - mean) / std, (evaluation_log - mean) / std, {
        "floors": floors,
        "offsets": offsets,
        "log_mean": mean,
        "log_scale": std,
    }


class ConvergentColorimetricSelector:
    """直接以摄氏温度误差训练Top-K波段门控，达到目标才算成功。"""

    def __init__(
        self,
        *,
        band_count: int,
        target_training_rmse_c: float,
        learning_rate: float = 1e-3,
        gate_learning_rate: float = 1e-4,
        freeze_gate_after: int | None = 1000,
        max_iter: int = 5000,
        evaluation_interval: int = 10,
        required_consecutive_successes: int = 3,
        redundancy_weight: float = 0.01,
        binarization_weight: float = 0.01,
        hidden_units: tuple[int, ...] = (128, 64),
        intensity_floor_fraction: float = 1e-6,
        absolute_intensity_floor: float = 1e-9,
        adaptive_weighting: bool = False,
        attention_hidden_units: int = 16,
        attention_loss_weight: float = 0.1,
        adaptive_weight_power: float = 0.5,
        adaptive_weight_minimum: float = 0.5,
        adaptive_weight_maximum: float = 2.0,
        adaptive_weight_warmup_iterations: int = 100,
        error_teacher_blend: float = 0.5,
        required_wavelength_ranges_nm: tuple[tuple[float, float], ...] = (),
        device: str = "cpu",
        seed: int = 42,
        selector_feature_mode: str = "centered",
    ) -> None:
        self.band_count = int(band_count)
        self.target_training_rmse_c = float(target_training_rmse_c)
        self.learning_rate = float(learning_rate)
        self.gate_learning_rate = float(gate_learning_rate)
        self.freeze_gate_after = None if freeze_gate_after is None else int(freeze_gate_after)
        self.max_iter = int(max_iter)
        self.evaluation_interval = int(evaluation_interval)
        self.required_consecutive_successes = int(required_consecutive_successes)
        self.redundancy_weight = float(redundancy_weight)
        self.binarization_weight = float(binarization_weight)
        self.hidden_units = tuple(int(value) for value in hidden_units)
        self.intensity_floor_fraction = float(intensity_floor_fraction)
        self.absolute_intensity_floor = float(absolute_intensity_floor)
        self.adaptive_weighting = bool(adaptive_weighting)
        self.attention_hidden_units = int(attention_hidden_units)
        self.attention_loss_weight = float(attention_loss_weight)
        self.adaptive_weight_power = float(adaptive_weight_power)
        self.adaptive_weight_minimum = float(adaptive_weight_minimum)
        self.adaptive_weight_maximum = float(adaptive_weight_maximum)
        self.adaptive_weight_warmup_iterations = int(adaptive_weight_warmup_iterations)
        self.error_teacher_blend = float(error_teacher_blend)
        self.required_wavelength_ranges_nm = tuple(
            (float(lower), float(upper)) for lower, upper in required_wavelength_ranges_nm
        )
        self.device = str(device)
        self.seed = int(seed)
        if selector_feature_mode not in ("centered", "uncentered_log"):
            raise ValueError("选带特征模式必须为centered或uncentered_log")
        self.selector_feature_mode = selector_feature_mode
        if self.adaptive_weighting and not 0 <= self.adaptive_weight_warmup_iterations < self.max_iter:
            raise ValueError("注意力预热轮数必须小于最大训练轮数")
        if self.attention_hidden_units < 1 or self.attention_loss_weight < 0.0:
            raise ValueError("注意力网络参数不合法")
        if not 0.0 <= self.error_teacher_blend <= 1.0:
            raise ValueError("真实误差教师融合比例必须位于0到1之间")
        if len(self.required_wavelength_ranges_nm) > self.band_count:
            raise ValueError("必选波段区间数量不能超过K")

    @staticmethod
    def _hard_gate(
        weights: torch.Tensor,
        band_count: int,
        required_masks: tuple[torch.Tensor, ...] = (),
    ) -> torch.Tensor:
        indices = constrained_topk_indices(weights, band_count, required_masks)
        hard = torch.zeros_like(weights)
        hard[indices] = 1.0
        return hard + weights - weights.detach()

    @staticmethod
    def _color_features(log_values: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        common = (log_values * gate[None, :]).sum(dim=1, keepdim=True) / gate.sum()
        return (log_values - common) * gate[None, :]

    def _selection_features(self, log_values: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """按配置保留共同强度，或沿用原跨波段中心化表示。"""
        if self.selector_feature_mode == "uncentered_log":
            return log_values * gate[None, :]
        return self._color_features(log_values, gate)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_validation: np.ndarray,
        y_validation: np.ndarray,
        wavelengths_nm: np.ndarray,
        mi_matrix: np.ndarray,
    ) -> ConvergentSelectionResult:
        started = time.perf_counter()
        train = np.asarray(X_train, dtype=np.float64)
        validation = np.asarray(X_validation, dtype=np.float64)
        train_y = np.asarray(y_train, dtype=np.float64)
        validation_y = np.asarray(y_validation, dtype=np.float64)
        wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
        if train.ndim != 2 or validation.shape[1] != train.shape[1]:
            raise ValueError("收敛选带训练集和验证集的波段数不一致")
        if train_y.shape != (train.shape[0],) or validation_y.shape != (validation.shape[0],):
            raise ValueError("收敛选带温度标签形状不一致")
        if wavelengths.shape != (train.shape[1],) or not 2 <= self.band_count <= train.shape[1]:
            raise ValueError("收敛选带的波长或K不合法")
        matrix = np.asarray(mi_matrix, dtype=np.float32)
        if matrix.shape != (train.shape[1], train.shape[1]):
            raise ValueError("收敛选带互信息矩阵形状不一致")
        required_masks = tuple(
            torch.as_tensor(
                (wavelengths >= lower) & (wavelengths <= upper),
                dtype=torch.bool,
            )
            for lower, upper in self.required_wavelength_ranges_nm
        )
        if any(not torch.any(mask) for mask in required_masks):
            raise ValueError("必选波段区间在质控后的波段中没有候选")

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        train_log, validation_log, transform = _prepare_log_radiance(
            train, validation, wavelengths,
            self.intensity_floor_fraction, self.absolute_intensity_floor,
        )
        self.transform_ = transform
        y_mean = float(train_y.mean())
        y_scale = max(float(train_y.std(ddof=0)), 1.0)
        device = torch.device(self.device)
        train_tensor = torch.as_tensor(train_log, dtype=torch.float32, device=device)
        validation_tensor = torch.as_tensor(validation_log, dtype=torch.float32, device=device)
        train_target = torch.as_tensor((train_y - y_mean) / y_scale, dtype=torch.float32, device=device)
        validation_target_c = torch.as_tensor(validation_y, dtype=torch.float32, device=device)
        train_target_c = torch.as_tensor(train_y, dtype=torch.float32, device=device)
        mi = torch.as_tensor(matrix, dtype=torch.float32, device=device)
        positive = mi[(mi > 0.0) & torch.isfinite(mi)]
        if positive.numel():
            mi = mi / torch.median(positive)

        gate = torch.nn.Parameter(torch.full(
            (train.shape[1],), self.band_count / train.shape[1],
            dtype=torch.float32, device=device,
        ))
        model = _TemperatureMLP(train.shape[1], self.hidden_units).to(device)
        attention = _SampleAttention(train.shape[1], self.attention_hidden_units).to(device)
        optimizer = torch.optim.Adam([
            {"params": [gate], "lr": self.gate_learning_rate},
            {"params": model.parameters(), "lr": self.learning_rate},
            {"params": attention.parameters(), "lr": self.learning_rate},
        ])
        trace: list[dict[str, float]] = []
        consecutive_successes = 0
        best_validation = float("inf")
        best_weights = None
        best_train_rmse = float("inf")
        best_train_adaptive_rmse = float("inf")
        best_validation_rmse = float("inf")
        best_validation_adaptive_rmse = float("inf")
        successful_validation = float("inf")
        completed = 0
        success = False
        frozen_gate: torch.Tensor | None = None
        adaptive_sample_weights = torch.ones(train.shape[0], device=device)

        for iteration in range(1, self.max_iter + 1):
            optimizer.zero_grad(set_to_none=True)
            if (
                frozen_gate is None
                and self.freeze_gate_after is not None
                and iteration > self.freeze_gate_after
            ):
                with torch.no_grad():
                    frozen_gate = self._hard_gate(gate, self.band_count, required_masks).detach()
            hard_gate = (
                frozen_gate
                if frozen_gate is not None
                else self._hard_gate(gate, self.band_count, required_masks)
            )
            color_features = self._selection_features(train_tensor, hard_gate)
            prediction_normalized = model(color_features)
            squared_error_normalized = (prediction_normalized - train_target) ** 2
            if self.adaptive_weighting and iteration > self.adaptive_weight_warmup_iterations:
                prediction_c = prediction_normalized * y_scale + y_mean
                adaptive_sample_weights = attention(
                    color_features.detach(),
                    self.adaptive_weight_minimum,
                    self.adaptive_weight_maximum,
                )
                attention_targets = attention_error_targets(
                    (prediction_c.detach() - train_target_c) ** 2,
                    power=self.adaptive_weight_power,
                    minimum_weight=self.adaptive_weight_minimum,
                    maximum_weight=self.adaptive_weight_maximum,
                )
                prediction_weights = blend_adaptive_weights(
                    adaptive_sample_weights.detach(),
                    attention_targets,
                    teacher_blend=self.error_teacher_blend,
                )
                prediction_loss = torch.mean(prediction_weights * squared_error_normalized)
                attention_loss = torch.mean((adaptive_sample_weights - attention_targets) ** 2)
            else:
                prediction_loss = torch.mean(squared_error_normalized)
                attention_loss = torch.zeros((), device=device)
            redundancy = _redundancy_from_reference(mi, gate) if frozen_gate is None else torch.zeros((), device=device)
            binarization = torch.mean(gate * (1.0 - gate)) if frozen_gate is None else torch.zeros((), device=device)
            loss = (
                prediction_loss
                + self.attention_loss_weight * attention_loss
                + self.redundancy_weight * redundancy
                + self.binarization_weight * binarization
            )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if frozen_gate is None:
                    gate.copy_(_project_capped_simplex(gate, self.band_count))
            completed = iteration
            if iteration % self.evaluation_interval != 0:
                continue
            with torch.no_grad():
                evaluation_gate = (
                    frozen_gate
                    if frozen_gate is not None
                    else self._hard_gate(gate, self.band_count, required_masks)
                )
                train_prediction_c = model(
                    self._selection_features(train_tensor, evaluation_gate)
                ) * y_scale + y_mean
                validation_prediction_c = model(
                    self._selection_features(validation_tensor, evaluation_gate)
                ) * y_scale + y_mean
                train_squared_error = (train_prediction_c - train_target_c) ** 2
                validation_squared_error = (validation_prediction_c - validation_target_c) ** 2
                adaptive_ready = not self.adaptive_weighting or iteration > self.adaptive_weight_warmup_iterations
                if self.adaptive_weighting and adaptive_ready:
                    train_attention_weights = attention(
                        self._selection_features(train_tensor, evaluation_gate),
                        self.adaptive_weight_minimum,
                        self.adaptive_weight_maximum,
                    )
                    validation_attention_weights = attention(
                        self._selection_features(validation_tensor, evaluation_gate),
                        self.adaptive_weight_minimum,
                        self.adaptive_weight_maximum,
                    )
                else:
                    train_attention_weights = torch.ones_like(train_squared_error)
                    validation_attention_weights = torch.ones_like(validation_squared_error)
                train_rmse = float(torch.sqrt(torch.mean(train_squared_error)).item())
                train_adaptive_rmse = float(torch.sqrt(torch.mean(
                    train_attention_weights * train_squared_error
                )).item())
                validation_rmse = float(torch.sqrt(torch.mean(validation_squared_error)).item())
                validation_adaptive_rmse = float(torch.sqrt(torch.mean(
                    validation_attention_weights * validation_squared_error
                )).item())
                trace.append({
                    "iteration": float(iteration),
                    "training_rmse_c": train_rmse,
                    "training_adaptive_rmse_c": train_adaptive_rmse,
                    "validation_rmse_c": validation_rmse,
                    "validation_adaptive_rmse_c": validation_adaptive_rmse,
                    "selected_band_count": float(self.band_count),
                    "temperature_mse_normalized": float(prediction_loss.item()),
                    "redundancy": float(redundancy.item()),
                    "total_loss": float(loss.item()),
                    "attention_weight_min": float(torch.min(adaptive_sample_weights).item()),
                    "attention_weight_max": float(torch.max(adaptive_sample_weights).item()),
                    "attention_loss": float(attention_loss.item()),
                })
                validation_objective = validation_adaptive_rmse if self.adaptive_weighting else validation_rmse
                training_objective = train_adaptive_rmse if self.adaptive_weighting else train_rmse
                if adaptive_ready and validation_objective < best_validation:
                    best_validation = validation_objective
                    best_weights = gate.detach().cpu().numpy().copy()
                    best_train_rmse = train_rmse
                    best_train_adaptive_rmse = train_adaptive_rmse
                    best_validation_rmse = validation_rmse
                    best_validation_adaptive_rmse = validation_adaptive_rmse
                if adaptive_ready and training_objective <= self.target_training_rmse_c:
                    consecutive_successes += 1
                    # 达标后优先保存验证误差较小的权重，避免只记录偶然一次达标。
                    if validation_objective < successful_validation:
                        successful_validation = validation_objective
                        best_weights = gate.detach().cpu().numpy().copy()
                        best_train_rmse = train_rmse
                        best_train_adaptive_rmse = train_adaptive_rmse
                        best_validation_rmse = validation_rmse
                        best_validation_adaptive_rmse = validation_adaptive_rmse
                else:
                    consecutive_successes = 0
            if consecutive_successes >= self.required_consecutive_successes:
                success = True
                break

        if best_weights is None:
            raise RuntimeError("收敛选带没有生成有效波段权重")
        selected = constrained_topk_indices(
            torch.as_tensor(best_weights), self.band_count, required_masks,
        ).cpu().numpy().astype(np.int64)
        return ConvergentSelectionResult(
            selected_indices=selected,
            weights=best_weights,
            success=success,
            stop_reason="target_training_rmse_reached" if success else "maximum_iterations_reached",
            iterations=completed,
            training_rmse_c=best_train_rmse,
            training_adaptive_rmse_c=best_train_adaptive_rmse,
            validation_rmse_c=best_validation_rmse,
            validation_adaptive_rmse_c=best_validation_adaptive_rmse,
            trace=trace,
            elapsed_seconds=time.perf_counter() - started,
        )


class FixedBandColorimetricMLP:
    """在已经选定的比色波段上训练最终非线性温度模型。"""

    def __init__(
        self,
        *,
        target_training_rmse_c: float,
        learning_rate: float,
        max_iter: int,
        evaluation_interval: int,
        required_consecutive_successes: int,
        hidden_units: tuple[int, ...],
        intensity_floor_fraction: float,
        absolute_intensity_floor: float,
        adaptive_weighting: bool,
        attention_hidden_units: int,
        attention_loss_weight: float,
        adaptive_weight_power: float,
        adaptive_weight_minimum: float,
        adaptive_weight_maximum: float,
        adaptive_weight_warmup_iterations: int,
        error_teacher_blend: float = 0.5,
        device: str,
        seed: int,
        regression_feature_mode: str = "helmert",
    ) -> None:
        self.target_training_rmse_c = float(target_training_rmse_c)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.evaluation_interval = int(evaluation_interval)
        self.required_consecutive_successes = int(required_consecutive_successes)
        self.hidden_units = hidden_units
        self.intensity_floor_fraction = float(intensity_floor_fraction)
        self.absolute_intensity_floor = float(absolute_intensity_floor)
        self.adaptive_weighting = bool(adaptive_weighting)
        self.attention_hidden_units = int(attention_hidden_units)
        self.attention_loss_weight = float(attention_loss_weight)
        self.adaptive_weight_power = float(adaptive_weight_power)
        self.adaptive_weight_minimum = float(adaptive_weight_minimum)
        self.adaptive_weight_maximum = float(adaptive_weight_maximum)
        self.adaptive_weight_warmup_iterations = int(adaptive_weight_warmup_iterations)
        self.error_teacher_blend = float(error_teacher_blend)
        self.device = str(device)
        self.seed = int(seed)
        if regression_feature_mode not in ("helmert", "log_spectrum"):
            raise ValueError("回归特征模式必须为helmert或log_spectrum")
        self.regression_feature_mode = regression_feature_mode
        if self.adaptive_weighting and not 0 <= self.adaptive_weight_warmup_iterations < self.max_iter:
            raise ValueError("注意力预热轮数必须小于最大训练轮数")
        if self.attention_hidden_units < 1 or self.attention_loss_weight < 0.0:
            raise ValueError("注意力网络参数不合法")
        if not 0.0 <= self.error_teacher_blend <= 1.0:
            raise ValueError("真实误差教师融合比例必须位于0到1之间")

    @staticmethod
    def _helmert_basis(dimension: int) -> np.ndarray:
        basis = np.zeros((dimension - 1, dimension), dtype=np.float64)
        for row in range(1, dimension):
            denominator = np.sqrt(row * (row + 1.0))
            basis[row - 1, :row] = 1.0 / denominator
            basis[row - 1, row] = -row / denominator
        return basis

    def _transform(self, X: np.ndarray) -> np.ndarray:
        stabilized = np.maximum(X + self.offsets_[None, :], self.floors_[None, :])
        log_values = np.log(stabilized) + 5.0 * np.log(self.wavelengths_nm_[None, :])
        features = self._regression_features(log_values)
        return (features - self.feature_mean_) / self.feature_scale_

    def _regression_features(self, log_values: np.ndarray) -> np.ndarray:
        """完整对数谱保留K维；原Helmert表示保留K减1维。"""
        if self.regression_feature_mode == "log_spectrum":
            return log_values
        return log_values @ self.helmert_basis_.T

    def fit(self, X: np.ndarray, y: np.ndarray, wavelengths_nm: np.ndarray) -> FixedBandColorimetricMLP:
        values = np.asarray(X, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        self.wavelengths_nm_ = np.asarray(wavelengths_nm, dtype=np.float64)
        scale = np.maximum(np.median(np.abs(values), axis=0), self.absolute_intensity_floor)
        self.floors_ = np.maximum(scale * self.intensity_floor_fraction, self.absolute_intensity_floor)
        self.offsets_ = np.maximum(self.floors_ - np.min(values, axis=0), 0.0)
        self.helmert_basis_ = self._helmert_basis(values.shape[1])
        stabilized = np.maximum(values + self.offsets_[None, :], self.floors_[None, :])
        log_values = np.log(stabilized) + 5.0 * np.log(self.wavelengths_nm_[None, :])
        features = self._regression_features(log_values)
        self.feature_mean_ = features.mean(axis=0)
        self.feature_scale_ = features.std(axis=0, ddof=0)
        self.feature_scale_[self.feature_scale_ < 1e-8] = 1.0
        features = (features - self.feature_mean_) / self.feature_scale_
        self.y_mean_ = float(target.mean())
        self.y_scale_ = max(float(target.std(ddof=0)), 1.0)

        torch.manual_seed(self.seed)
        device = torch.device(self.device)
        tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        target_tensor = torch.as_tensor((target - self.y_mean_) / self.y_scale_, dtype=torch.float32, device=device)
        target_c_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
        self.model_ = _TemperatureMLP(features.shape[1], self.hidden_units).to(device)
        self.attention_ = _SampleAttention(features.shape[1], self.attention_hidden_units).to(device)
        optimizer = torch.optim.Adam(
            list(self.model_.parameters()) + list(self.attention_.parameters()),
            lr=self.learning_rate,
        )
        consecutive = 0
        self.trace_: list[dict[str, float]] = []
        self.success_ = False
        best_state = copy.deepcopy(self.model_.state_dict())
        best_attention_state = copy.deepcopy(self.attention_.state_dict())
        best_objective_rmse = float("inf")
        best_unweighted_rmse = float("inf")
        best_adaptive_rmse = float("inf")
        adaptive_sample_weights = torch.ones(target.shape[0], device=device)
        for iteration in range(1, self.max_iter + 1):
            optimizer.zero_grad(set_to_none=True)
            normalized_prediction = self.model_(tensor)
            squared_error_normalized = (normalized_prediction - target_tensor) ** 2
            if self.adaptive_weighting and iteration > self.adaptive_weight_warmup_iterations:
                prediction_c = normalized_prediction * self.y_scale_ + self.y_mean_
                adaptive_sample_weights = self.attention_(
                    tensor.detach(),
                    self.adaptive_weight_minimum,
                    self.adaptive_weight_maximum,
                )
                attention_targets = attention_error_targets(
                    (prediction_c.detach() - target_c_tensor) ** 2,
                    power=self.adaptive_weight_power,
                    minimum_weight=self.adaptive_weight_minimum,
                    maximum_weight=self.adaptive_weight_maximum,
                )
                prediction_weights = blend_adaptive_weights(
                    adaptive_sample_weights.detach(),
                    attention_targets,
                    teacher_blend=self.error_teacher_blend,
                )
                prediction_loss = torch.mean(prediction_weights * squared_error_normalized)
                attention_loss = torch.mean((adaptive_sample_weights - attention_targets) ** 2)
                loss = prediction_loss + self.attention_loss_weight * attention_loss
            else:
                loss = torch.mean(squared_error_normalized)
                attention_loss = torch.zeros((), device=device)
            loss.backward()
            optimizer.step()
            if iteration % self.evaluation_interval != 0:
                continue
            with torch.no_grad():
                prediction = self.model_(tensor) * self.y_scale_ + self.y_mean_
                squared_error_c = (prediction - target_c_tensor) ** 2
                evaluation_weights = (
                    self.attention_(tensor, self.adaptive_weight_minimum, self.adaptive_weight_maximum)
                    if self.adaptive_weighting and iteration > self.adaptive_weight_warmup_iterations
                    else torch.ones_like(squared_error_c)
                )
                rmse = float(torch.sqrt(torch.mean(squared_error_c)).item())
                adaptive_rmse = float(torch.sqrt(torch.mean(evaluation_weights * squared_error_c)).item())
                self.trace_.append({
                    "iteration": float(iteration),
                    "training_rmse_c": rmse,
                    "training_adaptive_rmse_c": adaptive_rmse,
                    "attention_weight_min": float(torch.min(adaptive_sample_weights).item()),
                    "attention_weight_max": float(torch.max(adaptive_sample_weights).item()),
                    "attention_loss": float(attention_loss.item()),
                })
                adaptive_ready = not self.adaptive_weighting or iteration > self.adaptive_weight_warmup_iterations
                objective_rmse = adaptive_rmse if self.adaptive_weighting else rmse
                if adaptive_ready and objective_rmse < best_objective_rmse:
                    best_objective_rmse = objective_rmse
                    best_unweighted_rmse = rmse
                    best_adaptive_rmse = adaptive_rmse
                    best_state = copy.deepcopy(self.model_.state_dict())
                    best_attention_state = copy.deepcopy(self.attention_.state_dict())
                consecutive = (
                    consecutive + 1
                    if adaptive_ready and objective_rmse <= self.target_training_rmse_c
                    else 0
                )
            if consecutive >= self.required_consecutive_successes:
                self.success_ = True
                break
        self.model_.load_state_dict(best_state)
        self.attention_.load_state_dict(best_attention_state)
        self.training_rmse_c_ = best_unweighted_rmse
        self.training_adaptive_rmse_c_ = best_adaptive_rmse
        self.iterations_ = iteration
        self.training_attention_weights_ = self.attention_weights(values)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        device = torch.device(self.device)
        tensor = torch.as_tensor(self._transform(np.asarray(X, dtype=np.float64)), dtype=torch.float32, device=device)
        with torch.no_grad():
            return (self.model_(tensor) * self.y_scale_ + self.y_mean_).cpu().numpy().astype(np.float64)

    def attention_weights(self, X: np.ndarray) -> np.ndarray:
        """用冻结的注意力网络生成评价样本权重，不读取真实温度。"""
        if not self.adaptive_weighting:
            return np.ones(np.asarray(X).shape[0], dtype=np.float64)
        device = torch.device(self.device)
        tensor = torch.as_tensor(self._transform(np.asarray(X, dtype=np.float64)), dtype=torch.float32, device=device)
        with torch.no_grad():
            weights = self.attention_(
                tensor,
                self.adaptive_weight_minimum,
                self.adaptive_weight_maximum,
            )
        return weights.cpu().numpy().astype(np.float64)
