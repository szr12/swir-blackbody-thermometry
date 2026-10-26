"""统一特征接口下的选带与回归器交叉控制；不改变历史实验实现。"""

from __future__ import annotations

import copy
import numpy as np
import torch

from convergent_colorimetry import (
    FixedBandColorimetricMLP, _TemperatureMLP, _SampleAttention,
    attention_error_targets, blend_adaptive_weights,
)


class SharedLogHelmert:
    """只用最终拟合集估计一次参数，OLS与全部神经网络共享结果。"""

    def fit(self, X, wavelengths_nm, *, fraction=1e-6, absolute=1e-9):
        values = np.asarray(X, dtype=np.float64)
        self.wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float64)
        self.floors = np.maximum(np.median(np.abs(values), axis=0) * fraction, absolute)
        self.offsets = np.maximum(self.floors - values.min(axis=0), 0.0)
        self.basis = FixedBandColorimetricMLP._helmert_basis(values.shape[1])
        raw = self._raw(values)
        self.mean = raw.mean(axis=0)
        self.scale = raw.std(axis=0, ddof=0)
        self.scale[self.scale < 1e-8] = 1.0
        return self

    def _raw(self, X):
        stabilized = np.maximum(np.asarray(X) + self.offsets, self.floors)
        return (np.log(stabilized) + 5.0 * np.log(self.wavelengths_nm)) @ self.basis.T

    def transform(self, X):
        return (self._raw(X) - self.mean) / self.scale

    def snapshot(self):
        return {key: getattr(self, key) for key in
                ("wavelengths_nm", "floors", "offsets", "basis", "mean", "scale")}


class SharedFeatureMLP(FixedBandColorimetricMLP):
    """复用原网络、注意力和训练准则，接收已固定的11维特征。"""

    def _transform(self, X):
        return np.asarray(X, dtype=np.float64)

    def fit(self, X, y):
        values = features = np.asarray(X, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        if values.ndim != 2 or len(values) != len(target) or not np.isfinite(values).all():
            raise ValueError("共享特征形状或数值非法")
        self.y_mean_ = float(target.mean())
        self.y_scale_ = max(float(target.std(ddof=0)), 1.0)
        # 训练主体与历史固定波段回归器一致，禁止在专家内部重新估计输入变换。
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

