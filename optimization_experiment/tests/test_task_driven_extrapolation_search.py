"""检验外推选优边界，以及高温专家无法借用Validation补齐的约束。"""

from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "optimization_experiment"), str(ROOT / "main_experiment/src")]
from run_task_driven_extrapolation_search import rank_candidates, validate_expert_support


class ExtrapolationSearchTest(unittest.TestCase):
    def test_validation_only_and_failure_exclusion(self):
        base = {"band_count": 12, "redundancy_weight": .04, "overlap_c": 55,
                "routing_width_c": 15, "success": True, "validation_rmse_sd": 1.0}
        rows = [{**base, "validation_rmse_mean": 4, "test_rmse": 100},
                {**base, "validation_rmse_mean": 5, "test_rmse": 1},
                {**base, "validation_rmse_mean": 1, "success": False},
                {**base, "validation_rmse_mean": None, "success": False}]
        self.assertEqual(rank_candidates(rows)[0]["validation_rmse_mean"], 4)
        rows[0]["test_rmse"], rows[1]["test_rmse"] = 1, 100
        self.assertEqual(rank_candidates(rows)[0]["validation_rmse_mean"], 4)

    def test_high_temperature_expert_support_uses_train_only(self):
        train = np.arange(50, 330)
        boundaries = np.array([50, 150, 250, 350, 450])
        counts = validate_expert_support(train, boundaries, [35,55,75,95], 10)
        self.assertEqual([counts[str(float(o))][-1] for o in [35,55,75,95]], [15,35,55,75])
        with self.assertRaises(ValueError):
            validate_expert_support(train, boundaries, [15], 10)
        with self.assertRaises(ValueError):
            validate_expert_support(train, boundaries, [25], 10)


if __name__ == "__main__":
    unittest.main()
