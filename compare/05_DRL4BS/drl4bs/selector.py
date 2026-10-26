"""Paper-first DRL4BS; see PROVENANCE.md for departures from historical code."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Required by deterministic CUDA GEMM; do not override a compatible user setting.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import scipy
import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    episodes: int = 10000
    horizon: int = 6
    replay_capacity: int = 50000
    batch_size: int = 100
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.95
    epsilon_min: float = 0.01
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    momentum_decay: float = 0.004
    weight_decay: float = 0.0
    entropy_range_tolerance_bits: float = 1e-12


def validate_k(n_bands: int, maximum: int | None = None) -> int:
    if isinstance(n_bands, (bool, np.bool_)) or not isinstance(n_bands, (int, np.integer)):
        raise ValueError("n_bands must be a positive integer")
    if n_bands < 1 or (maximum is not None and n_bands > maximum):
        suffix = "" if maximum is None else f" not greater than the trained horizon {maximum}"
        raise ValueError(f"n_bands must be a positive integer{suffix}")
    return int(n_bands)


def prepare_entropy(X: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw = np.asarray(X)
    if raw.dtype.kind not in "fiu":
        raise ValueError("X must be a real numeric array")
    data = np.array(raw, dtype=np.float64, order="C", copy=True)
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError("X must be 2D with at least two samples")
    if not np.isfinite(data).all() or np.any(data < 0):
        raise ValueError("X must be finite and nonnegative; no automatic shifting")
    keep = np.flatnonzero(np.any(data > 0, axis=0))
    if keep.size < 1:
        raise ValueError("at least one nonzero band is required")
    values = data[:, keep]
    # Algebraically identical to x/sum(x), without overflowing a large finite sum.
    scaled = values / values.max(axis=0)
    probabilities = scaled / scaled.sum(axis=0)
    logs = np.zeros_like(probabilities)
    np.log2(probabilities, out=logs, where=probabilities > 0)
    entropies = -(probabilities * logs).sum(axis=0)
    span = float(np.ptp(entropies))
    degenerate = span <= Config.entropy_range_tolerance_bits
    normalized = np.zeros_like(entropies) if degenerate else (entropies - entropies.min()) / span
    raw_entropy: list[float | None] = [None] * data.shape[1]
    normalized_entropy: list[float | None] = [None] * data.shape[1]
    for idx, original in enumerate(keep):
        raw_entropy[int(original)] = float(entropies[idx])
        normalized_entropy[int(original)] = float(normalized[idx])
    metadata = {
        "input_shape": list(data.shape),
        "input_sha256": hashlib.sha256(str(data.shape).encode() + data.tobytes()).hexdigest(),
        "retained_bands": keep.tolist(),
        "excluded_zero_bands": np.flatnonzero(~np.any(data > 0, axis=0)).tolist(),
        "positive_constant_bands": keep[np.all(values == values[0], axis=0)].tolist(),
        "raw_entropy_bits": raw_entropy,
        "normalized_entropy": normalized_entropy,
        "entropy_range_bits": span,
        "degenerate_reward": degenerate,
        "warnings": ["all retained band entropies equal within tolerance; rewards cannot distinguish bands"]
        if degenerate else [],
        "preprocessing": "nonnegative observations / per-band sum; no z-score or shift",
    }
    return keep, normalized, metadata


def step_reward(entropies: np.ndarray, selected: list[int], action: int) -> float:
    old_mean = float(np.mean(entropies[selected])) if selected else 0.0
    return float(np.mean(entropies[selected + [action]])) - old_mean


class QNetwork(nn.Sequential):
    def __init__(self, bands: int):
        super().__init__(nn.Linear(bands, 2 * bands), nn.ReLU(),
                         nn.Linear(2 * bands, 2 * bands), nn.ReLU(),
                         nn.Linear(2 * bands, bands))
        for layer in self:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)


def greedy_action(q: torch.Tensor, state: torch.Tensor) -> int:
    if bool(torch.all(state.bool())):
        raise ValueError("no valid actions remain")
    # Retained band indices are ascending, so argmax ties use the original index.
    return int(q.masked_fill(state.bool(), -torch.inf).argmax().item())


@torch.no_grad()
def bellman_targets(rewards: torch.Tensor, done: torch.Tensor,
                    next_q: torch.Tensor, next_states: torch.Tensor,
                    gamma: float = 0.99) -> torch.Tensor:
    target = rewards.clone()
    active = ~done
    if bool(active.any()):
        valid_q = next_q[active].masked_fill(next_states[active].bool(), -torch.inf)
        target[active] += gamma * valid_q.max(dim=1).values
    return target


def decay_epsilon(epsilon: float, config: Config) -> float:
    return max(config.epsilon_min, epsilon * config.epsilon_decay)


class ReplayBuffer:
    """Bounded ring buffer; uniform sampling without copying all stored states."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items: list = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.items)

    def append(self, item: Any) -> None:
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            self.items[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, count: int, rng: random.Random) -> list:
        return [self.items[i] for i in rng.sample(range(len(self.items)), count)]


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    paths = sorted((root / "drl4bs").glob("*.py")) + sorted(root.glob("*.py"))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


@dataclass
class SelectionResult:
    selected_bands: np.ndarray
    selection_order: np.ndarray
    diagnostics: dict[str, Any]


class DRL4BSSelector:
    def __init__(self, device: str = "auto", random_state: int = 0, profile: str = "paper",
                 horizon: int = 6, episodes: int | None = None):
        if profile not in ("paper", "quick"):
            raise ValueError("profile must be paper or quick")
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
            raise ValueError("random_state must be an integer")
        if not 0 <= random_state < 2**32:
            raise ValueError("random_state must be in [0, 2**32)")
        validate_k(horizon)
        if episodes is not None and (
            isinstance(episodes, bool) or not isinstance(episodes, (int, np.integer)) or episodes < 1
        ):
            raise ValueError("episodes must be a positive integer or None")
        self.device = device
        self.random_state = int(random_state)
        self.profile = profile
        self.horizon = int(horizon)
        self.episodes = None if episodes is None else int(episodes)

    def fit(self, X_train: Any) -> "DRL4BSSelector":
        # A failed refit must not leave stale, seemingly fitted results.
        for name in ("model_", "diagnostics_", "selection_order_", "retained_bands_"):
            self.__dict__.pop(name, None)
        start = time.perf_counter()
        keep, entropies, metadata = prepare_entropy(X_train)
        actual = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        actual = "cpu" if actual == "auto" else actual
        if actual == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if keep.size < self.horizon:
            raise ValueError(f"at least {self.horizon} nonzero bands are required")
        episode_count = self.episodes if self.episodes is not None else (
            10000 if self.profile == "paper" else 50
        )
        config = Config(episodes=episode_count, horizon=self.horizon)
        device = torch.device(actual)
        rng = random.Random(self.random_state)
        previous_determinism = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        previous_threads = torch.get_num_threads()
        previous_benchmark = torch.backends.cudnn.benchmark
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        devices = [torch.cuda.current_device()] if actual == "cuda" else []
        try:
            torch.use_deterministic_algorithms(True)
            torch.set_num_threads(1)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(self.random_state)
                if actual == "cuda":
                    torch.cuda.manual_seed(self.random_state)
                model = QNetwork(len(keep)).to(device)
                optimizer = torch.optim.NAdam(
                    model.parameters(), lr=config.learning_rate, betas=config.betas,
                    eps=config.adam_eps, momentum_decay=config.momentum_decay,
                    weight_decay=config.weight_decay, foreach=False,
                )
                memory = ReplayBuffer(config.replay_capacity)
                epsilon = config.epsilon_start
                losses, epsilon_history, trajectory = [], [], []
                model.train()
                for episode in range(1, config.episodes + 1):
                    state = np.zeros(len(keep), dtype=np.float32)
                    chosen: list[int] = []
                    for step in range(config.horizon):
                        if rng.random() < epsilon:
                            action = rng.choice(np.flatnonzero(state == 0).tolist())
                        else:
                            with torch.no_grad():
                                tensor = torch.as_tensor(state, device=device)
                                action = greedy_action(model(tensor), tensor)
                        reward = step_reward(entropies, chosen, action)
                        next_state = state.copy()
                        next_state[action] = 1
                        memory.append((state.copy(), action, reward, next_state.copy(),
                                       step == config.horizon - 1))
                        state = next_state
                        chosen.append(action)
                    if len(memory) >= config.batch_size:
                        sample = memory.sample(config.batch_size, rng)
                        states = torch.as_tensor(np.stack([v[0] for v in sample]), device=device)
                        actions = torch.tensor([v[1] for v in sample], device=device)
                        rewards = torch.tensor([v[2] for v in sample], dtype=torch.float32, device=device)
                        next_states = torch.as_tensor(np.stack([v[3] for v in sample]), device=device)
                        done = torch.tensor([v[4] for v in sample], dtype=torch.bool, device=device)
                        with torch.no_grad():
                            target = bellman_targets(rewards, done, model(next_states),
                                                     next_states, config.gamma)
                        prediction = model(states).gather(1, actions[:, None]).squeeze(1)
                        loss = (prediction - target).square().mean()
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError(f"non-finite Bellman loss at episode {episode}")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                               for p in model.parameters()):
                            raise FloatingPointError("non-finite gradient")
                        optimizer.step()
                        losses.append({"episode": episode, "loss": float(loss.detach().cpu())})
                        epsilon = decay_epsilon(epsilon, config)
                    epsilon_history.append(epsilon)
                    if episode % 100 == 0:
                        order = self._infer(model, device, keep, config.horizon)
                        trajectory.append({"episode": episode, "selection_order": order.tolist()})
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise FloatingPointError("non-finite final model")
                if actual == "cuda":
                    torch.cuda.synchronize()
                train_end = time.perf_counter()
                order = self._infer(model, device, keep, config.horizon)
                if actual == "cuda":
                    torch.cuda.synchronize()
                infer_end = time.perf_counter()
                model.eval()
        finally:
            torch.use_deterministic_algorithms(previous_determinism, warn_only=previous_warn_only)
            torch.set_num_threads(previous_threads)
            torch.backends.cudnn.benchmark = previous_benchmark
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        self.model_ = model
        self.retained_bands_ = keep.copy()
        self.selection_order_ = order
        self.diagnostics_ = {
            **metadata, "method": "DRL4BS-Tabular (non-official paper-first adaptation)",
            "profile": self.profile, "config": asdict(config), "random_state": self.random_state,
            "episodes_completed": config.episodes, "stop_reason": "budget_exhausted",
            "gradient_updates": len(losses), "replay_size": len(memory),
            "loss_history": losses, "epsilon_history": epsilon_history,
            "greedy_trajectory": trajectory, "selection_order_horizon": order.tolist(),
            "training_seconds": train_end - start, "final_inference_seconds": infer_end - train_end,
            "training_timing_includes": "entropy preprocessing, initialization, replay, periodic greedy diagnostics",
            "runtime": {"device": actual, "gpu": torch.cuda.get_device_name() if actual == "cuda" else None,
                        "python": platform.python_version(), "torch": str(torch.__version__),
                        "cuda": torch.version.cuda, "numpy": np.__version__, "scipy": scipy.__version__,
                        "platform": platform.platform(), "cpu": platform.processor(),
                        "training_cpu_threads": 1, "deterministic_algorithms": True,
                        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
            "source_sha256": source_hashes(),
            "official_head": "c952228074cddbbefb9f0c60c5026b54644acc50",
            "official_snapshot": "9ab5c9ce7bb03f58559ebbc91e0f7ad50f376130",
        }
        return self

    @staticmethod
    @torch.no_grad()
    def _infer(model: QNetwork, device: torch.device, keep: np.ndarray, horizon: int) -> np.ndarray:
        state = torch.zeros(len(keep), device=device)
        chosen = []
        for _ in range(horizon):
            action = greedy_action(model(state), state)
            chosen.append(int(keep[action]))
            state[action] = 1
        return np.asarray(chosen, dtype=np.int64)

    def _check_fitted(self) -> None:
        if not hasattr(self, "selection_order_"):
            raise RuntimeError("fit must complete before select or save")

    def select(self, n_bands: int) -> np.ndarray:
        self._check_fitted()
        k = validate_k(n_bands, int(self.diagnostics_["config"]["horizon"]))
        return np.sort(self.selection_order_[:k]).copy()

    def result(self, n_bands: int) -> SelectionResult:
        bands = self.select(n_bands)
        return SelectionResult(bands, self.selection_order_[:len(bands)].copy(),
                               copy.deepcopy(self.diagnostics_))

    def save(self, output_dir: str | Path) -> None:
        """Save final weights and strict JSON, not a resumable training checkpoint."""
        self._check_fitted()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        model_path = output / "model.pt"
        torch.save({k: v.detach().cpu() for k, v in self.model_.state_dict().items()}, model_path)
        metadata = copy.deepcopy(self.diagnostics_)
        metadata["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        horizon = int(self.diagnostics_["config"]["horizon"])
        reported = sorted({k for k in (4, 5, 6, horizon) if k <= horizon})
        metadata["selections"] = {str(k): self.select(k).tolist() for k in reported}
        (output / "result.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def band_selection_tabular(X_2d: Any, n_bands: int, *, device: str = "auto",
                           random_state: int = 0, profile: str = "paper",
                           return_diagnostics: bool = False) -> np.ndarray | SelectionResult:
    validate_k(n_bands)
    selector = DRL4BSSelector(device, random_state, profile, horizon=n_bands).fit(X_2d)
    return selector.result(n_bands) if return_diagnostics else selector.select(n_bands)
