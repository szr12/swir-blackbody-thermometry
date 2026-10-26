"""验证共同特征无评价集依赖，以及训练适配器保持原全局模型行为。"""

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "optimization_experiment"), str(ROOT / "main_experiment/src")]
from convergent_colorimetry import FixedBandColorimetricMLP
from traditional_utils import model_parameters
from shared_feature_regression import SharedLogHelmert, SharedFeatureMLP
from run_selector_regressor_control import fit_ols, ols_predict


class SharedFeatureControlTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(91)
        self.X = rng.normal(20, 3, (45, 12))
        self.X[0, 0] = -2
        self.y = np.linspace(50, 450, 45)
        self.wave = np.linspace(860, 2400, 12)

    def test_frozen_transform_ignores_test_distribution(self):
        transform = SharedLogHelmert().fit(self.X[:30], self.wave)
        before = {key: value.copy() for key, value in transform.snapshot().items()}
        original = transform.transform(self.X[:30])
        transform.transform(self.X[30:] * 1e5 - 1e4)
        np.testing.assert_array_equal(original, transform.transform(self.X[:30]))
        for key in before:
            np.testing.assert_array_equal(before[key], transform.snapshot()[key])
        np.testing.assert_allclose(original.mean(axis=0), 0, atol=1e-12)

    def test_precomputed_training_matches_original_global_model(self):
        config = json.loads((ROOT / "optimization_experiment/configs/component_ablation_refrozen_k12_random_20261013.json").read_text("utf-8"))
        params = model_parameters(config)
        params.update(device="cpu", max_iter=140, hidden_units=(8, 8))
        original = FixedBandColorimetricMLP(seed=17, **params).fit(self.X, self.y, self.wave)
        transform = SharedLogHelmert().fit(self.X, self.wave)
        np.testing.assert_array_equal(original._transform(self.X), transform.transform(self.X))
        adapted = SharedFeatureMLP(seed=17, **params).fit(transform.transform(self.X), self.y)
        np.testing.assert_array_equal(original.predict(self.X), adapted.predict(transform.transform(self.X)))
        self.assertEqual(original.trace_, adapted.trace_)

    def test_rank_deficiency_is_reported_without_changing_ols(self):
        Z = np.column_stack([self.y, self.y])
        coef, diagnostic = fit_ols(Z, self.y)
        self.assertFalse(diagnostic["full_rank"])
        np.testing.assert_allclose(ols_predict(Z, coef), self.y, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
