from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


CELSIUS_TO_KELVIN = 273.15


@dataclass(frozen=True)
class ColorimetricResult:
    metrics: dict[str, float]
    predictions_c: np.ndarray
    diagnostics: dict


class SupervisedColorimetricRegressor:
    """使用真实温度标定多组光谱比值，并融合各波段对的温度反演结果。"""

    def __init__(
        self,
        *,
        alpha: float = 1e-8,
        intensity_floor_fraction: float = 1e-6,
        absolute_intensity_floor: float = 1e-9,
        minimum_pair_rmse_c: float = 0.1,
        extrapolation_margin_c: float = 25.0,
        pair_mode: str = "all_selected",
    ) -> None:
        self.alpha = float(alpha)
        self.intensity_floor_fraction = float(intensity_floor_fraction)
        self.absolute_intensity_floor = float(absolute_intensity_floor)
        self.minimum_pair_rmse_c = float(minimum_pair_rmse_c)
        self.extrapolation_margin_c = float(extrapolation_margin_c)
        self.pair_mode = pair_mode

    @staticmethod
    def _validate_X(
        X: np.ndarray,
        expected_columns: int | None = None,
        minimum_rows: int = 1,
    ) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < minimum_rows or values.shape[1] < 2:
            raise ValueError(f"比色反演至少需要两个波段和 {minimum_rows} 个样本")
        if expected_columns is not None and values.shape[1] != expected_columns:
            raise ValueError("预测数据的波段数与标定数据不一致")
        if not np.isfinite(values).all():
            raise ValueError("比色反演输入包含 NaN 或 Inf")
        return values

    def _make_pairs(self, sensor_names: np.ndarray) -> list[tuple[int, int]]:
        all_pairs = list(combinations(range(sensor_names.size), 2))
        if self.pair_mode == "all_selected":
            return all_pairs
        if self.pair_mode == "within_sensor":
            return [(left, right) for left, right in all_pairs if sensor_names[left] == sensor_names[right]]
        raise ValueError(f"不支持的 pair_mode: {self.pair_mode}")

    def _stabilize(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        adjusted = X + self.offsets_[None, :]
        clipped = adjusted < self.floors_[None, :]
        return np.maximum(adjusted, self.floors_[None, :]), clipped

    def _pair_features(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        stabilized, clipped = self._stabilize(X)
        # 维恩形式中的 lambda^(-5) 项在对数域转化为加法项。
        log_radiance_proxy = np.log(stabilized) + 5.0 * np.log(self.wavelengths_nm_[None, :])
        features = np.column_stack([
            log_radiance_proxy[:, left] - log_radiance_proxy[:, right]
            for left, right in self.pairs_
        ])
        return features, clipped

    def _temperatures_from_reciprocal(self, reciprocal: np.ndarray) -> np.ndarray:
        bounded = np.clip(reciprocal, self.reciprocal_lower_, self.reciprocal_upper_)
        return 1.0 / bounded - CELSIUS_TO_KELVIN

    def fit(
        self,
        X: np.ndarray,
        y_temperature_c: np.ndarray,
        wavelengths_nm: np.ndarray,
        sensor_names: np.ndarray,
    ) -> SupervisedColorimetricRegressor:
        X = self._validate_X(X, minimum_rows=2)
        y = np.asarray(y_temperature_c, dtype=np.float64)
        wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
        sensors = np.asarray(sensor_names).astype(str)
        if y.shape != (X.shape[0],) or wavelengths.shape != (X.shape[1],) or sensors.shape != wavelengths.shape:
            raise ValueError("比色标定数据与温度、波长或传感器元数据形状不一致")
        if not np.isfinite(y).all() or not np.isfinite(wavelengths).all() or np.any(wavelengths <= 0):
            raise ValueError("温度或波长包含非法值")
        if np.any(y <= -CELSIUS_TO_KELVIN):
            raise ValueError("摄氏温度必须高于绝对零度")

        self.wavelengths_nm_ = wavelengths.copy()
        self.sensor_names_ = sensors.copy()
        self.pairs_ = self._make_pairs(sensors)
        if not self.pairs_:
            raise ValueError("所选波段无法形成合法的比色波段对")

        # 偏移量和强度下限只能由训练折估计，随后原样应用于验证折和测试折。
        scale = np.maximum(np.median(np.abs(X), axis=0), self.absolute_intensity_floor)
        self.floors_ = np.maximum(
            scale * self.intensity_floor_fraction,
            self.absolute_intensity_floor,
        )
        self.offsets_ = np.maximum(self.floors_ - np.min(X, axis=0), 0.0)
        pair_features, train_clipped = self._pair_features(X)

        temperature_kelvin = y + CELSIUS_TO_KELVIN
        reciprocal_target = 1.0 / temperature_kelvin
        lower_temperature = max(float(temperature_kelvin.min()) - self.extrapolation_margin_c, 1.0)
        upper_temperature = float(temperature_kelvin.max()) + self.extrapolation_margin_c
        self.reciprocal_lower_ = 1.0 / upper_temperature
        self.reciprocal_upper_ = 1.0 / lower_temperature

        self.models_: list[Ridge] = []
        pair_train_rmse: list[float] = []
        pair_predictions: list[np.ndarray] = []
        for column in range(pair_features.shape[1]):
            model = Ridge(alpha=self.alpha)
            model.fit(pair_features[:, column : column + 1], reciprocal_target)
            reciprocal_prediction = np.asarray(
                model.predict(pair_features[:, column : column + 1]), dtype=np.float64
            )
            temperature_prediction = self._temperatures_from_reciprocal(reciprocal_prediction)
            rmse = float(np.sqrt(mean_squared_error(y, temperature_prediction)))
            self.models_.append(model)
            pair_train_rmse.append(rmse)
            pair_predictions.append(temperature_prediction)

        precision = 1.0 / np.maximum(np.asarray(pair_train_rmse), self.minimum_pair_rmse_c) ** 2
        self.pair_weights_ = precision / precision.sum()
        self.pair_weight_source_ = "training_rmse"
        self.pair_train_rmse_c_ = np.asarray(pair_train_rmse, dtype=np.float64)
        self.train_clip_fraction_ = np.mean(train_clipped, axis=0)
        fused = np.average(np.column_stack(pair_predictions), axis=1, weights=self.pair_weights_)
        self.train_fused_rmse_c_ = float(np.sqrt(mean_squared_error(y, fused)))
        return self

    def predict_pairs(self, X: np.ndarray) -> np.ndarray:
        """返回每个波段对的独立温度预测，供训练折内稳健融合使用。"""
        if not hasattr(self, "models_"):
            raise RuntimeError("比色模型尚未完成标定")
        X = self._validate_X(X, expected_columns=self.wavelengths_nm_.size)
        pair_features, clipped = self._pair_features(X)
        predictions = []
        for column, model in enumerate(self.models_):
            reciprocal = np.asarray(model.predict(pair_features[:, column : column + 1]), dtype=np.float64)
            predictions.append(self._temperatures_from_reciprocal(reciprocal))
        self.last_predict_clip_fraction_ = np.mean(clipped, axis=0)
        return np.column_stack(predictions)

    def set_pair_weights(self, weights: np.ndarray, source: str = "external") -> SupervisedColorimetricRegressor:
        """设置仅由训练数据得到的外部融合权重。"""
        if not hasattr(self, "models_"):
            raise RuntimeError("比色模型尚未完成标定")
        values = np.asarray(weights, dtype=np.float64)
        if values.shape != (len(self.pairs_),) or not np.isfinite(values).all():
            raise ValueError("融合权重的形状或数值不合法")
        if np.any(values < 0.0) or float(values.sum()) <= 0.0:
            raise ValueError("融合权重必须非负且总和大于零")
        self.pair_weights_ = values / values.sum()
        self.pair_weight_source_ = str(source)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pair_predictions = self.predict_pairs(X)
        return np.average(pair_predictions, axis=1, weights=self.pair_weights_)

    def diagnostics(self) -> dict:
        if not hasattr(self, "models_"):
            raise RuntimeError("比色模型尚未完成标定")
        pair_rows = []
        for pair_id, ((left, right), model) in enumerate(zip(self.pairs_, self.models_)):
            pair_rows.append({
                "pair_id": pair_id,
                "left_selected_index": left,
                "right_selected_index": right,
                "left_wavelength_nm": float(self.wavelengths_nm_[left]),
                "right_wavelength_nm": float(self.wavelengths_nm_[right]),
                "left_sensor": str(self.sensor_names_[left]),
                "right_sensor": str(self.sensor_names_[right]),
                "cross_sensor": bool(self.sensor_names_[left] != self.sensor_names_[right]),
                "reciprocal_temperature_coefficient": float(model.coef_[0]),
                "reciprocal_temperature_intercept": float(model.intercept_),
                "train_rmse_c": float(self.pair_train_rmse_c_[pair_id]),
                "fusion_weight": float(self.pair_weights_[pair_id]),
            })
        return {
            "name": "supervised_colorimetric_pair_fusion",
            "supervision_target": "reciprocal_absolute_temperature",
            "pair_mode": self.pair_mode,
            "pair_count": len(self.pairs_),
            "pair_weight_source": self.pair_weight_source_,
            "alpha": self.alpha,
            "intensity_floor_fraction": self.intensity_floor_fraction,
            "absolute_intensity_floor": self.absolute_intensity_floor,
            "minimum_pair_rmse_c": self.minimum_pair_rmse_c,
            "extrapolation_margin_c": self.extrapolation_margin_c,
            "training_only_offsets": self.offsets_.copy(),
            "training_only_floors": self.floors_.copy(),
            "train_clip_fraction_by_selected_band": self.train_clip_fraction_.copy(),
            "predict_clip_fraction_by_selected_band": getattr(
                self, "last_predict_clip_fraction_", np.zeros_like(self.train_clip_fraction_)
            ).copy(),
            "train_fused_rmse_c": self.train_fused_rmse_c_,
            "pairs": pair_rows,
            "interpretation": (
                "这是由真实温度监督标定的经验比色模型；输入为未完成绝对辐射定标的 DN 特征，"
                "不得表述为无标定的第一性原理普朗克反演。"
            ),
        }


def evaluate_supervised_colorimetry(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    wavelengths_nm: np.ndarray,
    sensor_names: np.ndarray,
    model_config: dict,
) -> ColorimetricResult:
    """仅用训练折标定比色模型，并在给定测试折计算温度误差。"""
    model = SupervisedColorimetricRegressor(
        alpha=float(model_config.get("alpha", 1e-8)),
        intensity_floor_fraction=float(model_config.get("intensity_floor_fraction", 1e-6)),
        absolute_intensity_floor=float(model_config.get("absolute_intensity_floor", 1e-9)),
        minimum_pair_rmse_c=float(model_config.get("minimum_pair_rmse_c", 0.1)),
        extrapolation_margin_c=float(model_config.get("extrapolation_margin_c", 25.0)),
        pair_mode=str(model_config.get("pair_mode", "all_selected")),
    ).fit(X_train, y_train, wavelengths_nm, sensor_names)
    prediction = np.asarray(model.predict(X_test), dtype=np.float64)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
        "mae": float(mean_absolute_error(y_test, prediction)),
        "r2": float(r2_score(y_test, prediction)),
    }
    return ColorimetricResult(metrics=metrics, predictions_c=prediction, diagnostics=model.diagnostics())
