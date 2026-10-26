from __future__ import annotations

import json
import os
import platform
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import SGAEConfig, get_profile
from .model import SGAEModel, reconstruction_loss


def _actual_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _validate_x(x: Any) -> np.ndarray:
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("X must be a two-dimensional [samples, bands] array")
    if array.shape[0] < 2:
        raise ValueError("X must contain at least two samples")
    if array.shape[1] < 2:
        raise ValueError("X must contain at least two bands")
    if not np.isfinite(array).all():
        raise ValueError("X must contain only finite values")
    return array


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    # A mathematically constant column can acquire a tiny non-zero std after a
    # large affine offset because its floating-point mean is not exact.
    constant = np.flatnonzero(np.ptp(x, axis=0) == 0.0)
    if constant.size == x.shape[1]:
        raise ValueError("all bands are constant")
    safe_scale = scale.copy()
    safe_scale[constant] = 1.0
    normalized = (x - mean) / safe_scale
    normalized[:, constant] = 0.0
    return normalized, mean, scale, constant


def early_stop(regularization: float, threshold: float) -> bool:
    return regularization < threshold


@dataclass
class _Run:
    selected: np.ndarray
    diagnostics: dict[str, Any]
    model_state: dict[str, torch.Tensor]


class SGAESelector:
    """Paper-guided, non-official SGAE adaptation for spectral tables.

    ``fit`` stores and standardizes outer-training features only.  Since the
    published network and binary gate both depend on K, each new K passed to
    ``select`` trains and caches a separate model.
    """

    def __init__(
        self,
        *,
        device: str = "auto",
        random_state: int = 0,
        profile: str = "paper",
        config: SGAEConfig | None = None,
        **config_overrides: Any,
    ) -> None:
        if not isinstance(random_state, int) or random_state < 0:
            raise ValueError("random_state must be a non-negative integer")
        self.requested_device = device
        self.random_state = random_state
        self.profile = profile
        if config is not None and config_overrides:
            raise ValueError("pass either config or config overrides, not both")
        self.config = config or get_profile(profile, **config_overrides)
        self.config.validate()
        self._device = _actual_device(device)
        self._runs: dict[int, _Run] = {}
        self._fitted = False

    def fit(self, X: Any) -> "SGAESelector":
        x = _validate_x(X)
        constant = np.flatnonzero(np.ptp(x, axis=0) == 0.0)
        if constant.size == x.shape[1]:
            raise ValueError("all bands are constant")
        self.n_samples_, self.n_features_in_ = x.shape
        self.mean_ = np.zeros(self.n_features_in_, dtype=np.float64)
        self.scale_ = np.ones(self.n_features_in_, dtype=np.float64)
        self.constant_bands_ = constant
        # 论文直接使用投影后的二维HSI矩阵，未报告逐波段Z-score。
        self._x_train = x.astype(np.float32, copy=True)
        self._runs.clear()
        self._fitted = True
        return self

    def _train(self, n_bands: int) -> _Run:
        if not self._fitted:
            raise RuntimeError("fit must be called before select")
        if not isinstance(n_bands, int) or isinstance(n_bands, bool):
            raise TypeError("n_bands must be an integer")
        if not 0 < n_bands <= self.n_features_in_:
            raise ValueError("n_bands must be in [1, number of input bands]")
        if n_bands in self._runs:
            return self._runs[n_bands]

        _seed_everything(self.random_state)
        x = torch.as_tensor(self._x_train, device=self._device)
        model = SGAEModel(self.n_features_in_, n_bands, self.config.sigma).to(self._device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        generator = torch.Generator(device=self._device)
        generator.manual_seed(self.random_state)
        history: list[dict[str, float | int]] = []
        stop_reason = "max_epochs"
        last_indices: torch.Tensor | None = None
        started = time.perf_counter()

        for epoch in range(1, self.config.max_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            x_hat, _, _, indices = model(x, generator=generator)
            last_indices = indices.detach().clone()
            rec = reconstruction_loss(x_hat, x)
            reg = model.regularization()
            loss = rec + reg
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite SGAE loss")
            loss.backward()
            if model.mu.grad is None or not torch.isfinite(model.mu.grad).all():
                raise FloatingPointError("non-finite or missing gate gradient")
            optimizer.step()
            with torch.no_grad():
                post_reg = float(model.regularization().item())
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach().item()),
                    "reconstruction": float(rec.detach().item()),
                    "regularization": float(reg.detach().item()),
                    "post_update_regularization": post_reg,
                }
            )
            if early_stop(post_reg, self.config.stopping_threshold):
                stop_reason = "regularization_below_threshold"
                break

        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        elapsed = time.perf_counter() - started
        with torch.no_grad():
            mu = model.mu.detach().cpu().numpy().astype(float)
            probabilities = model.regularization().new_tensor(0)  # device anchor
            del probabilities
            probs = (0.5 * (1.0 + torch.erf(model.mu / (self.config.sigma * np.sqrt(2.0))))).cpu().numpy().astype(float)
            ranking = np.lexsort((np.arange(self.n_features_in_), -mu)).astype(int)
            if last_indices is None:
                raise RuntimeError("SGAE training completed without a stochastic gate")
            chosen_rank_order = last_indices.cpu().numpy().astype(int)
            selected = np.sort(chosen_rank_order).astype(int)

        gpu = None
        if self._device.type == "cuda":
            gpu = torch.cuda.get_device_name(self._device)
        diagnostics: dict[str, Any] = {
            "method": "SGAE-Tabular",
            "implementation": "non-official paper-guided adaptation",
            "target_bands": n_bands,
            "selected_bands": selected.tolist(),
            "selected_rank_order": chosen_rank_order.tolist(),
            "last_stochastic_gate_rank_order": chosen_rank_order.tolist(),
            "mu_ranking": ranking.tolist(),
            "ranking": ranking.tolist(),
            "gate_mu": mu.tolist(),
            "gate_open_probability": probs.tolist(),
            "loss_history": history,
            "epochs_run": len(history),
            "stop_reason": stop_reason,
            "training_mean": self.mean_.astype(float).tolist(),
            "training_scale": self.scale_.astype(float).tolist(),
            "input_preprocessing": "none_paper_does_not_report_zscore",
            "constant_bands": self.constant_bands_.astype(int).tolist(),
            "config": self.config.to_dict(),
            "profile": self.profile,
            "random_state": self.random_state,
            "elapsed_seconds": elapsed,
            "requested_device": self.requested_device,
            "actual_device": str(self._device),
            "gpu": gpu,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "paper_ambiguities": [
                "binary Top-K backward estimator is not disclosed; straight-through Top-K is used",
                "optimizer is not disclosed; Adam is used",
                "Eq. 6/7 typography conflicts with Fig. 2; mean Gaussian CDF is used",
                "sigma is dataset-specific; tabular default is 0.01",
                "the final subset is the stochastic Top-K gate used in the stopping epoch",
            ],
        }
        model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        run = _Run(selected=selected, diagnostics=diagnostics, model_state=model_state)
        self._runs[n_bands] = run
        return run

    def select(self, n_bands: int, *, return_diagnostics: bool = False):
        run = self._train(n_bands)
        selected = run.selected.copy()
        if return_diagnostics:
            return selected, deepcopy(run.diagnostics)
        return selected

    def diagnostics(self, n_bands: int) -> dict[str, Any]:
        return deepcopy(self._train(n_bands).diagnostics)

    def save(self, n_bands: int, output_directory: str | Path) -> None:
        run = self._train(n_bands)
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(run.model_state, output / "model.pt")
        write_json_strict(output / "diagnostics.json", run.diagnostics)


def band_selection_tabular(
    X_2d: Any,
    n_bands: int,
    *,
    return_diagnostics: bool = False,
    **selector_kwargs: Any,
):
    selector = SGAESelector(**selector_kwargs).fit(X_2d)
    return selector.select(n_bands, return_diagnostics=return_diagnostics)


def write_json_strict(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
