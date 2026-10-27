"""独立核验学习曲线的样本边界、端点误差和归档结果，不执行训练。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main(root):
    manifest = json.loads((root / "payload/manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "results/summary.json").read_text(encoding="utf-8"))
    guard = json.loads((root / "results/observer_verification.json").read_text(encoding="utf-8"))
    assert guard["training_weights_bitwise_identical"] and guard["validation_label_perturbation_ignored_by_training"]
    train, val, test = [set(manifest[k]) for k in ["train_indices", "validation_indices", "test_indices"]]
    assert (len(train), len(val), len(test)) == (280, 60, 61)
    assert not train & val and not test & (train | val)
    assert summary["scientific_fits"] == 25 and len(summary["models"]) == 25
    assert summary["all_converged"] and not summary["test_evaluated"]
    trace_points = 0
    endpoint_tolerance = 0.0
    for row in summary["models"]:
        job = root / "results" / f"seed_{row['seed']}" / row["role"]
        trace = json.loads((job / "trace.json").read_text(encoding="utf-8"))["trace"]
        assert len(trace) == row["iterations"] // 10
        assert [int(v["iteration"]) for v in trace] == list(range(10, row["iterations"] + 1, 10))
        returned = next(v for v in trace if int(v["iteration"]) == row["restored_iteration"])
        with np.load(job / "predictions.npz") as data:
            for prefix, allowed in [("train", train), ("validation", val)]:
                ids = set(data[prefix + "_indices"].tolist())
                assert ids <= allowed and not ids & test
                target, prediction = data[prefix + "_target_c"], data[prefix + "_prediction_c"]
                actual = float(np.sqrt(np.mean((prediction - target) ** 2)))
                key = "training_rmse_c" if prefix == "train" else "validation_rmse_c"
                assert abs(actual - row[key]) < 1e-10
                if prefix == "validation":
                    difference = abs(actual - returned[key])
                    endpoint_tolerance = max(endpoint_tolerance, difference)
                    assert difference < 1e-10
        trace_points += len(trace)
    for row in summary["ensembles"]:
        for subset, allowed in [("training", train), ("validation", val)]:
            path = root / "results" / f"seed_{row['seed']}" / (subset + "_ensemble.npz")
            with np.load(path) as data:
                assert set(data["indices"].tolist()) == allowed
                actual = float(np.sqrt(np.mean((data["prediction_c"] - data["target_c"]) ** 2)))
                assert abs(actual - row[subset + "_rmse_c"]) < 1e-10
        assert row["difference_from_archived_validation_c"] == 0.0
    with (root / "results/learning_curves.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == trace_points
    statistics = {}
    for subset in ["training", "validation"]:
        values = np.asarray([v[subset + "_rmse_c"] for v in summary["ensembles"]])
        statistics[subset] = {"mean_rmse_c": float(values.mean()), "sample_sd_rmse_c": float(values.std(ddof=1))}
    outputs = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((root / "results").rglob("*")) if p.is_file()}
    report = {"all_passed": True, "diagnostic_models": 25, "seed_count": 5, "trace_points": trace_points,
              "train_validation_disjoint": True, "test_indices_excluded": True,
              "validation_did_not_change_training": True, "max_returned_validation_recompute_difference_c": endpoint_tolerance,
              "archived_validation_predictions_reproduced_exactly": True, "statistics": statistics,
              "result_files_sha256": outputs}
    (root / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "result_files_sha256"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root.resolve())
