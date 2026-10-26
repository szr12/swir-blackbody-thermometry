"""仅用标准库重算公开结果，不训练、不调参、不合并不同实验口径。"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def measures(rows):
    target = [float(v["temperature_c"]) for v in rows]
    errors = [float(v["prediction_c"]) - y for v, y in zip(rows, target)]
    mean = statistics.fmean(target)
    return {"rmse": math.sqrt(statistics.fmean(e*e for e in errors)),
            "mae": statistics.fmean(abs(e) for e in errors),
            "r2": 1 - sum(e*e for e in errors)/sum((y-mean)**2 for y in target),
            "mard_percent": 100*statistics.fmean(abs(e)/abs(y) for e, y in zip(errors, target))}


def main():
    metrics = json.loads((ROOT / "results/paper_metrics.json").read_text(encoding="utf-8"))
    total = 0
    for table, group, label in [("table4", "ablation", "configuration"), ("table5", "comparison", "method")]:
        rows = read_rows(ROOT / ("results/" + table + "_predictions.csv"))
        for protocol, value in metrics["protocols"].items():
            split = json.loads((ROOT / f"results/splits/{protocol}.json").read_text(encoding="utf-8"))
            assert not set(split["test_indices"]) & set(split["fit_indices"])
            for method, summary in value[group].items():
                runs = []
                for run in summary["runs"]:
                    subset = [v for v in rows if v["protocol"] == protocol and v[label] == method and int(v["seed"]) == run["seed"]]
                    assert len(subset) == 61
                    assert {int(float(v["sample_index"])) for v in subset} == set(split["test_indices"])
                    actual = measures(subset)
                    assert all(abs(actual[k]-run[k]) < 1e-8 for k in actual), (table, protocol, method, run["seed"])
                    runs.append(actual)
                    total += 1
                for key, aggregate in summary["aggregate"].items():
                    assert abs(statistics.fmean(v[key] for v in runs)-aggregate["mean"]) < 1e-8
                    assert abs(statistics.stdev(v[key] for v in runs)-aggregate["sd"]) < 1e-8
        print(table, "PASS", len(rows), "scan-level predictions")
    for protocol in metrics["protocols"].values():
        assert protocol["comparison"]["Task-driven"]["aggregate"] == protocol["ablation"]["complete"]["aggregate"]
    print(json.dumps({"status": "PASS", "valid_runs": total, "training_runs": 0,
                      "same_complete_pipeline_in_both_tables": True}))


if __name__ == "__main__":
    main()
