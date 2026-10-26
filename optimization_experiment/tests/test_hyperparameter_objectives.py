from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hyperparameter_objectives import configured_metric, rank_candidates, selection_objective


def candidate(k, equal, adaptive, success=True, feasible=True, balanced=0.1):
    return {"band_count": k, "success": success, "redundancy_weight": 0.03,
            "overlap_c": 55, "routing_width_c": 45, "validation_rmse_c": adaptive,
            "validation_unweighted_metrics": {"rmse": equal},
            "validation_relative_deviation": 0.03, "validation_balanced_score": balanced,
            "validation_rmse_feasible": feasible}


class ObjectiveTests(unittest.TestCase):
    def test_equal_and_adaptive_choose_different_candidates(self):
        """故意令两种指标排序相反，防止等权选参仍读取加权字段。"""
        rows = [candidate(12, 4.0, 9.0), candidate(18, 6.0, 3.0)]
        self.assertEqual(rank_candidates(rows, "unweighted_rmse", 12.2)[0]["band_count"], 12)
        self.assertEqual(rank_candidates(rows)[0]["band_count"], 18)

    def test_equal_constraint_ignores_old_weighted_feasibility(self):
        rows = [candidate(12, 4.0, 20.0, feasible=False), candidate(18, 6.0, 3.0)]
        self.assertEqual(rank_candidates(rows, "unweighted_rmse", 12.2)[0]["band_count"], 12)
        self.assertEqual(rank_candidates(rows, "unweighted_rmse")[0]["band_count"], 12)
        self.assertEqual(rank_candidates(rows)[0]["band_count"], 18)

    def test_nonconverged_and_nonfinite_cannot_win(self):
        rows = [candidate(2, 0.1, 0.1, success=False), candidate(4, float("nan"), 2), candidate(12, 5.0, 6)]
        self.assertEqual([row["band_count"] for row in rank_candidates(rows, "unweighted_rmse", 12.2)], [12])

    def test_fallback_and_ties_follow_declared_rules(self):
        rows = [candidate(18, 15, 3, balanced=0.2), candidate(12, 15, 4, balanced=0.1)]
        self.assertEqual(rank_candidates(rows, "unweighted_rmse", 12.2)[0]["band_count"], 12)

    def test_missing_equal_metrics_cannot_fall_back_to_adaptive(self):
        row = candidate(12, 4, 5)
        del row["validation_unweighted_metrics"]
        with self.assertRaises(KeyError):
            rank_candidates([row], "unweighted_rmse")

    def test_metric_configuration_is_explicit_and_legacy_compatible(self):
        self.assertEqual(configured_metric({}), "adaptive_rmse")
        self.assertEqual(configured_metric({"validation_objective": {"metric": "unweighted_rmse"}}), "unweighted_rmse")
        self.assertIn("unweighted_rmse", selection_objective("unweighted_rmse"))
        with self.assertRaises(ValueError):
            configured_metric({"validation_objective": {"metric": "test_rmse"}})


if __name__ == "__main__":
    unittest.main()
