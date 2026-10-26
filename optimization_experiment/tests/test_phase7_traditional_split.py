from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from traditional_utils import (  # noqa: E402
    make_traditional_split,
    make_temperature_ordered_split,
    mean_absolute_relative_deviation,
    observed_segmented_metrics,
)
from convergent_colorimetry import (  # noqa: E402
    _SampleAttention,
    FixedBandColorimetricMLP,
    attention_error_targets,
    blend_adaptive_weights,
    constrained_topk_indices,
)


def test_traditional_split_has_expected_sizes_and_no_overlap() -> None:
    temperatures = np.linspace(50.0, 450.0, 401)
    boundaries = np.asarray([50.0, 150.0, 250.0, 350.0, 450.0])
    train, validation, test = make_traditional_split(temperatures, boundaries, 42)
    assert (train.size, validation.size, test.size) == (280, 60, 61)
    assert np.unique(np.concatenate([train, validation, test])).size == 401


def test_temperature_ordered_split_reserves_highest_temperatures_for_test() -> None:
    temperatures = np.linspace(50.0, 450.0, 401)
    train, validation, test = make_temperature_ordered_split(temperatures)
    assert (train.size, validation.size, test.size) == (280, 60, 61)
    assert temperatures[train].max() == 329.0
    assert temperatures[validation].min() == 330.0
    assert temperatures[validation].max() == 389.0
    assert temperatures[test].min() == 390.0
    assert temperatures[test].max() == 450.0


def test_observed_segmented_metrics_skips_uncovered_temperature_regions() -> None:
    temperatures = np.asarray([390.0, 400.0, 450.0])
    metrics = observed_segmented_metrics(
        temperatures,
        temperatures + 1.0,
        np.asarray([50.0, 150.0, 250.0, 350.0, 450.0]),
    )
    assert len(metrics) == 1
    assert metrics[0]["segment"] == 3
    assert metrics[0]["n_samples"] == 3


def test_mean_absolute_relative_deviation_matches_formula() -> None:
    true = np.asarray([100.0, 200.0])
    prediction = np.asarray([110.0, 180.0])
    assert np.isclose(mean_absolute_relative_deviation(true, prediction), 0.1)


def test_attention_targets_raise_harder_samples() -> None:
    targets = attention_error_targets(
        torch.tensor([100.0, 100.0, 900.0, 900.0]),
        power=0.5,
        minimum_weight=0.5,
        maximum_weight=2.0,
    )
    assert targets[2] > targets[0]
    assert torch.isclose(targets.mean(), torch.tensor(1.0))


def test_attention_network_outputs_positive_normalized_weights() -> None:
    network = _SampleAttention(input_dimension=4, hidden_units=3)
    weights = network(torch.randn(8, 4), minimum_weight=0.5, maximum_weight=2.0)
    assert torch.all(weights > 0.0)
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-6)


def test_error_teacher_is_blended_into_prediction_weights() -> None:
    attention = torch.tensor([0.8, 1.2])
    error_teacher = torch.tensor([1.6, 0.4])
    blended = blend_adaptive_weights(attention, error_teacher, teacher_blend=0.5)
    assert torch.allclose(blended, torch.tensor([1.2, 0.8]))
    assert torch.isclose(blended.mean(), torch.tensor(1.0))


def test_constrained_topk_keeps_a_longwave_band() -> None:
    weights = torch.tensor([0.9, 0.8, 0.7, 0.1])
    longwave = torch.tensor([False, False, False, True])
    selected = constrained_topk_indices(weights, band_count=2, required_masks=(longwave,))
    assert set(selected.tolist()) == {0, 3}


def test_fixed_band_model_trains_with_attention_weighted_mse() -> None:
    rng = np.random.default_rng(7)
    values = rng.uniform(10.0, 100.0, size=(32, 4))
    temperatures = np.linspace(50.0, 450.0, 32)
    model = FixedBandColorimetricMLP(
        target_training_rmse_c=1000.0,
        learning_rate=0.003,
        max_iter=10,
        evaluation_interval=1,
        required_consecutive_successes=1,
        hidden_units=(8,),
        intensity_floor_fraction=1e-6,
        absolute_intensity_floor=1e-9,
        adaptive_weighting=True,
        attention_hidden_units=4,
        attention_loss_weight=0.1,
        adaptive_weight_power=0.5,
        adaptive_weight_minimum=0.5,
        adaptive_weight_maximum=2.0,
        adaptive_weight_warmup_iterations=2,
        device="cpu",
        seed=7,
    ).fit(values, temperatures, np.asarray([900.0, 1000.0, 1100.0, 1200.0]))
    weights = model.attention_weights(values)
    assert model.success_ is True
    assert np.isfinite(model.training_adaptive_rmse_c_)
    assert np.isclose(weights.mean(), 1.0)
