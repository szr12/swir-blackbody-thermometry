"""验证等间隔预算和Validation选优边界。"""

from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "optimization_experiment"), str(ROOT / "main_experiment/src")]
from run_uniform_hyperparameter_search import rank_candidates
from run_component_ablations import control_bands


class UniformSearchTest(unittest.TestCase):
    def test_selection_ignores_test_and_rejects_nonconvergence(self):
        base = {"family": "global", "band_count": 12, "overlap_c": 0,
                "routing_width_c": 0, "validation_rmse_sd": 0.2, "success": True}
        rows = [{**base, "validation_rmse_mean": 4, "test_rmse": 100},
                {**base, "validation_rmse_mean": 5, "test_rmse": 1},
                {**base, "validation_rmse_mean": 1, "success": False}]
        self.assertEqual(rank_candidates(rows)[0]["validation_rmse_mean"], 4)
        rows[0]["test_rmse"], rows[1]["test_rmse"] = 1, 100
        self.assertEqual(rank_candidates(rows)[0]["validation_rmse_mean"], 4)

    def test_uniform_budget_and_all_band_endpoint(self):
        wave = np.linspace(858, 2497, 264)
        available = np.delete(np.arange(264), [43, 89, 145])
        for k in [4, 12, 17, 128, len(available)]:
            bands = control_bands(available, wave, k, 4242, "uniform_bands")
            self.assertEqual(len(np.unique(bands)), k)
            self.assertTrue(set(bands).issubset(set(available)))
            self.assertTrue(np.any((wave[bands] >= 1800) & (wave[bands] <= 2500)))
        np.testing.assert_array_equal(bands, available)


if __name__ == "__main__":
    unittest.main()
