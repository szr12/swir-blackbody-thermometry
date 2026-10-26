"""BS-Net-FC-Tabular.

The paper is primary where it conflicts with the official TensorFlow 1.6
source. See PROVENANCE.md for the evidence table and explicit adaptations.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import scipy
import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 2e-3
    l1_lambda: float = 1e-2
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    batch_norm_epsilon: float = 1e-3
    batch_norm_decay: float = 0.99
    degenerate_score_tolerance: float = 1e-12


@dataclass
class SelectionResult:
    selected_bands: np.ndarray
    selection_order: np.ndarray
    diagnostics: dict[str, Any]


def validate_k(n_bands: int, n_available: int | None = None) -> int:
    if isinstance(n_bands, (bool, np.bool_)) or not isinstance(n_bands, (int, np.integer)):
        raise ValueError("n_bands must be a positive integer")
    value = int(n_bands)
    if value < 1:
        raise ValueError("n_bands must be a positive integer")
    if n_available is not None and value > n_available:
        raise ValueError("n_bands cannot exceed the available nonconstant bands")
    return value


def prepare_training_data(X: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw = np.asarray(X)
    if raw.dtype.kind not in "fiu":
        raise ValueError("X must be a real numeric array")
    data = np.array(raw, dtype=np.float64, order="C", copy=True)
    if data.ndim != 2:
        raise ValueError("X must be two-dimensional [samples, bands]")
    if data.shape[0] < Config.batch_size:
        raise ValueError("at least 64 training samples are required by the paper profile")
    if not np.isfinite(data).all():
        raise ValueError("X must contain only finite values")
    minima = data.min(axis=0)
    maxima = data.max(axis=0)
    constant = maxima == minima
    retained = np.flatnonzero(~constant)
    if retained.size < 6:
        raise ValueError("at least six nonconstant training bands are required")
    scaled = (data[:, retained] - minima[retained]) / (maxima[retained] - minima[retained])
    metadata = {
        "input_shape": list(data.shape),
        "input_sha256": hashlib.sha256(str(data.shape).encode() + data.tobytes()).hexdigest(),
        "retained_bands": retained.tolist(),
        "excluded_constant_bands": np.flatnonzero(constant).tolist(),
        "training_minima": minima.tolist(),
        "training_maxima": maxima.tolist(),
        "preprocessing": "per-band [0,1] min-max fitted on X_train only",
    }
    return scaled.astype(np.float32), retained.astype(np.int64), metadata


class TensorFlowBatchNorm1d(nn.Module):
    """TensorFlow 1.6 tf.layers.batch_normalization semantics for 2-D input."""

    def __init__(self, features: int, *, epsilon: float = 1e-3, decay: float = 0.99):
        super().__init__()
        self.epsilon = epsilon
        self.decay = decay
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.register_buffer("moving_mean", torch.zeros(features))
        self.register_buffer("moving_variance", torch.ones(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.weight.numel():
            raise ValueError("batch normalization expects [batch, features]")
        if self.training:
            mean = x.mean(dim=0)
            variance = (x - mean).square().mean(dim=0)
            with torch.no_grad():
                self.moving_mean.mul_(self.decay).add_(mean.detach(), alpha=1 - self.decay)
                self.moving_variance.mul_(self.decay).add_(variance.detach(), alpha=1 - self.decay)
        else:
            mean, variance = self.moving_mean, self.moving_variance
        return (x - mean) * torch.rsqrt(variance + self.epsilon) * self.weight + self.bias


class BSNetFC(nn.Module):
    def __init__(self, bands: int, config: Config):
        super().__init__()
        bn = lambda size: TensorFlowBatchNorm1d(
            size, epsilon=config.batch_norm_epsilon, decay=config.batch_norm_decay)
        self.input_norm = bn(bands)
        self.attention_1 = nn.Linear(bands, 64)
        self.attention_norm = bn(64)
        self.bottleneck = nn.Linear(64, 128)
        self.channel_weight = nn.Linear(128, bands)
        self.rec_1, self.rec_norm_1 = nn.Linear(bands, 64), bn(64)
        self.rec_2, self.rec_norm_2 = nn.Linear(64, 128), bn(128)
        self.rec_3, self.rec_norm_3 = nn.Linear(128, 256), bn(256)
        self.rec_4, self.output_norm = nn.Linear(256, bands), bn(bands)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.input_norm(x)
        # This double ReLU order is deliberate and matches the official source.
        attention = torch.relu(self.attention_1(normalized))
        attention = torch.relu(self.attention_norm(attention))
        attention = torch.relu(self.bottleneck(attention))
        weights = torch.sigmoid(self.channel_weight(attention))
        reconstruction = torch.relu(self.rec_norm_1(self.rec_1(weights * normalized)))
        reconstruction = torch.relu(self.rec_norm_2(self.rec_2(reconstruction)))
        reconstruction = torch.relu(self.rec_norm_3(self.rec_3(reconstruction)))
        reconstruction = torch.sigmoid(self.output_norm(self.rec_4(reconstruction)))
        return weights, reconstruction


def paper_loss(
    target: torch.Tensor, reconstruction: torch.Tensor, weights: torch.Tensor, l1_lambda: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equation 10 with S equal to the current batch size."""
    if target.shape != reconstruction.shape or target.shape != weights.shape or target.ndim != 2:
        raise ValueError("target, reconstruction and weights must share a 2-D shape")
    reconstruction_loss = 0.5 * (target - reconstruction).square().sum() / target.shape[0]
    regularization_loss = l1_lambda * weights.abs().sum()
    return reconstruction_loss + regularization_loss, reconstruction_loss, regularization_loss


class RollingBatchDataset:
    """Exact batching behavior of the official Helper.Dataset with a local RNG."""

    def __init__(self, data: np.ndarray, random_state: int):
        self.data = np.array(data, copy=True)
        self.index = 0
        self.completed = 0
        self.rng = np.random.default_rng(random_state)
        self._initial_shuffle = False

    def next_batch(self, batch_size: int) -> np.ndarray:
        start = self.index
        if start == 0 and self.completed == 0 and not self._initial_shuffle:
            self.data = self.data[self.rng.permutation(len(self.data))]
            self._initial_shuffle = True
        if start + batch_size > len(self.data):
            rest = self.data[start:]
            self.completed += 1
            self.data = self.data[self.rng.permutation(len(self.data))]
            self.index = batch_size - len(rest)
            return np.concatenate((rest, self.data[: self.index]), axis=0)
        self.index += batch_size
        return self.data[start : self.index]


@torch.no_grad()
def mean_attention(
    model: BSNetFC, data: np.ndarray, device: torch.device, batch_size: int = 512
) -> np.ndarray:
    was_training = model.training
    model.eval()
    total = torch.zeros(data.shape[1], dtype=torch.float64, device=device)
    count = 0
    for start in range(0, len(data), batch_size):
        x = torch.as_tensor(data[start : start + batch_size], device=device)
        weights, _ = model(x)
        total += weights.double().sum(dim=0)
        count += len(x)
    model.train(was_training)
    return (total / count).cpu().numpy()


def rank_bands(scores: np.ndarray, original_indices: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    original_indices = np.asarray(original_indices, dtype=np.int64)
    if scores.ndim != 1 or scores.shape != original_indices.shape or not np.isfinite(scores).all():
        raise ValueError("scores and indices must be finite one-dimensional arrays of equal size")
    # Primary key: descending score. Secondary key: smaller original index.
    order = np.lexsort((original_indices, -scores))
    return original_indices[order]


def score_is_degenerate(scores: np.ndarray, tolerance: float = 1e-12) -> bool:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("scores must be a non-empty finite vector")
    return float(np.ptp(scores)) <= tolerance


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    paths = sorted((root / "bsnets").glob("*.py")) + sorted(root.glob("*.py"))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class BSNetFCSelector:
    def __init__(self, device: str = "auto", random_state: int = 133, profile: str = "paper"):
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        if profile not in ("paper", "quick"):
            raise ValueError("profile must be paper or quick")
        if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
            raise ValueError("random_state must be an integer")
        if not 0 <= int(random_state) < 2**32:
            raise ValueError("random_state must be in [0, 2**32)")
        self.device = device
        self.random_state = int(random_state)
        self.profile = profile

    def fit(self, X_train: Any) -> "BSNetFCSelector":
        for name in ("model_", "diagnostics_", "selection_order_", "retained_bands_"):
            self.__dict__.pop(name, None)
        started = time.perf_counter()
        data, retained, metadata = prepare_training_data(X_train)
        actual = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        actual = "cpu" if actual == "auto" else actual
        if actual == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        base = Config()
        config = base if self.profile == "paper" else Config(epochs=2)
        device = torch.device(actual)
        prior_deterministic = torch.are_deterministic_algorithms_enabled()
        prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        prior_threads = torch.get_num_threads()
        prior_benchmark = torch.backends.cudnn.benchmark
        prior_tf32 = torch.backends.cuda.matmul.allow_tf32
        cuda_devices = [torch.cuda.current_device()] if actual == "cuda" else []
        try:
            torch.use_deterministic_algorithms(True)
            torch.set_num_threads(1)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.random.fork_rng(devices=cuda_devices):
                torch.random.default_generator.manual_seed(self.random_state)
                if actual == "cuda":
                    torch.cuda.manual_seed(self.random_state)
                model = BSNetFC(data.shape[1], config).to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=config.learning_rate, betas=config.adam_betas,
                    eps=config.adam_eps, weight_decay=0, foreach=False)
                dataset = RollingBatchDataset(data, self.random_state)
                batches_per_epoch = len(data) // config.batch_size
                history: list[dict[str, Any]] = []
                for epoch in range(1, config.epochs + 1):
                    model.train()
                    for _ in range(batches_per_epoch):
                        x = torch.as_tensor(dataset.next_batch(config.batch_size), device=device)
                        weights, reconstruction = model(x)
                        loss, _, _ = paper_loss(x, reconstruction, weights, config.l1_lambda)
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError(f"non-finite loss at epoch {epoch}")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        if any(
                            parameter.grad is not None
                            and not bool(torch.isfinite(parameter.grad).all())
                            for parameter in model.parameters()
                        ):
                            raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
                        optimizer.step()
                    # Paper Eq. 11: average weights across the complete training set.
                    scores = mean_attention(model, data, device)
                    model.eval()
                    full_total = full_reconstruction = full_l1 = 0.0
                    with torch.no_grad():
                        for start in range(0, len(data), 512):
                            x = torch.as_tensor(data[start : start + 512], device=device)
                            weights, reconstruction = model(x)
                            _, rec, reg = paper_loss(x, reconstruction, weights, config.l1_lambda)
                            # Accumulate chunk sums, then form the full-data Eq. 10.
                            n = len(x)
                            full_reconstruction += float(rec.cpu()) * n
                            full_l1 += float(reg.cpu())
                    full_reconstruction /= len(data)
                    full_total = full_reconstruction + full_l1
                    if not np.isfinite(full_total) or not np.isfinite(scores).all():
                        raise FloatingPointError(f"non-finite diagnostics at epoch {epoch}")
                    history.append({
                        "epoch": epoch, "loss": full_total,
                        "reconstruction_loss": full_reconstruction,
                        "l1_loss": full_l1, "mean_attention": scores.tolist(),
                    })
                if actual == "cuda":
                    torch.cuda.synchronize()
                training_ended = time.perf_counter()
                final_scores = mean_attention(model, data, device)
                if actual == "cuda":
                    torch.cuda.synchronize()
                inference_ended = time.perf_counter()
                model.eval()
        finally:
            torch.use_deterministic_algorithms(prior_deterministic, warn_only=prior_warn_only)
            torch.set_num_threads(prior_threads)
            torch.backends.cudnn.benchmark = prior_benchmark
            torch.backends.cuda.matmul.allow_tf32 = prior_tf32
        order = rank_bands(final_scores, retained)
        score_span = float(np.ptp(final_scores))
        degenerate = score_is_degenerate(final_scores, config.degenerate_score_tolerance)
        original_scores: list[float | None] = [None] * metadata["input_shape"][1]
        for local, original in enumerate(retained):
            original_scores[int(original)] = float(final_scores[local])
        self.model_ = model
        self.retained_bands_ = retained.copy()
        self.selection_order_ = order
        self.diagnostics_ = {
            **metadata,
            "method": "BS-Net-FC-Tabular (non-official paper-first PyTorch adaptation)",
            "profile": self.profile,
            "config": asdict(config),
            "random_state": self.random_state,
            "epochs_completed": config.epochs,
            "batches_per_epoch": batches_per_epoch,
            "optimizer_steps": config.epochs * batches_per_epoch,
            "stop_reason": "maximum_epochs_reached",
            "loss_history": history,
            "mean_attention_scores": original_scores,
            "selection_order": order.tolist(),
            "attention_score_span": score_span,
            "degenerate_ranking": degenerate,
            "warnings": ["all retained mean attention scores are tied within tolerance"]
            if degenerate else [],
            "training_seconds": training_ended - started,
            "final_inference_seconds": inference_ended - training_ended,
            "runtime": {
                "device": actual,
                "gpu": torch.cuda.get_device_name() if actual == "cuda" else None,
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "platform": platform.platform(),
                "cpu": platform.processor(),
                "training_cpu_threads": 1,
                "deterministic_algorithms": True,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            },
            "source_sha256": source_hashes(),
            "official_commit": "bbe8dc258b70fbf727bd3d577ea4d516d095bd35",
        }
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "selection_order_"):
            raise RuntimeError("fit must complete before select, result or save")

    def select(self, n_bands: int) -> np.ndarray:
        self._check_fitted()
        k = validate_k(n_bands, len(self.selection_order_))
        return np.sort(self.selection_order_[:k]).copy()

    def result(self, n_bands: int) -> SelectionResult:
        selected = self.select(n_bands)
        return SelectionResult(
            selected_bands=selected,
            selection_order=self.selection_order_[: len(selected)].copy(),
            diagnostics=copy.deepcopy(self.diagnostics_),
        )

    def save(self, output_dir: str | Path) -> None:
        self._check_fitted()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        model_path = output / "model.pt"
        torch.save({key: value.detach().cpu() for key, value in self.model_.state_dict().items()}, model_path)
        diagnostics = copy.deepcopy(self.diagnostics_)
        diagnostics["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        diagnostics["selection_rule"] = "top-k mean attention according to paper Eq. 11"
        (output / "result.json").write_text(
            json.dumps(diagnostics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def band_selection_tabular(
    X_2d: Any,
    n_bands: int,
    *,
    device: str = "auto",
    random_state: int = 133,
    profile: str = "paper",
    return_diagnostics: bool = False,
) -> np.ndarray | SelectionResult:
    validate_k(n_bands)
    selector = BSNetFCSelector(device=device, random_state=random_state, profile=profile).fit(X_2d)
    return selector.result(n_bands) if return_diagnostics else selector.select(n_bands)
