"""公开版可移植入口，直接复用冻结选带器和原生区域回归实现。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
BASELINES = {"BS-Net-FC": "03_BS_Nets", "DRL4BS": "05_DRL4BS", "HCBH": "06_HCBH",
             "SGAE": "07_SGAE", "XGBS-Reg": "09_XGBS_Reg", "Dropout-CAE": "10_Dropout_CAE"}


def main(args):
    sys.path[:0] = [str(ROOT / "optimization_experiment"), str(ROOT / "main_experiment/src"),
                   str(ROOT / "compare/09_XGBS_Reg/adapter_source")]
    import numpy as np
    import torch
    from run_component_ablations import audit_split, fit_ensemble
    from run_selector_regressor_control import select_baseline
    from traditional_utils import fold_quality_mask, load_data, make_selector, regression_metrics, write_json
    from blackbody_hsic.selector_runtime import load_or_compute_mi
    cfg = json.loads((ROOT / f"configs/benchmark_{args.protocol}.json").read_text())
    cfg["model"]["device"] = args.device
    path = ROOT / cfg["data_path"]
    public = json.loads((ROOT / "results/protocol.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == public["data_sha256"]
    for name in ["convergent_colorimetry.py", "regional_colorimetry.py", "traditional_utils.py"]:
        assert hashlib.sha256((ROOT / "optimization_experiment" / name).read_bytes()).hexdigest() == public["fitting_source_identity"][name]
    split = json.loads((ROOT / f"results/splits/{args.protocol}.json").read_text())
    train, val, test, fit = [np.asarray(split[k], dtype=int) for k in ["train_indices", "validation_indices", "test_indices", "fit_indices"]]
    data = load_data(path, cfg["feature_key"], cfg["uncertainty_key"])
    X, y, wave = data[cfg["feature_key"]], data["y_temperature_c"], data["wavelengths_nm"]
    audit_split(y, train, val, test, args.protocol == "ordered")
    quality, diagnostic = fold_quality_mask(X[train], data[cfg["uncertainty_key"]][train], data["background_mad"][train],
        data["bad_pixel_fraction_by_sample_band"][train], data["saturation_fraction_by_sample_band"][train], wave, cfg["fold_quality_control"])
    available = np.flatnonzero(quality)
    if args.prepare_only:
        print(json.dumps({"status": "PREPARED", "protocol": args.protocol, "method": args.method,
                          "shape": list(X.shape), "candidate_bands": len(available), "split": [len(train), len(val), len(test)],
                          "model_source_hashes_match": True, "training_runs": 0}))
        return
    output = args.output.resolve()
    assert not output.exists() or not any(output.iterdir()), "拒绝覆盖已有结果"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    for seed in args.seeds:
        job = output / f"seed_{seed}"
        job.mkdir(parents=True, exist_ok=True)
        if args.method == "ours":
            mi = load_or_compute_mi(output / "train_mutual_information.npy", X[train][:, quality], 3, 42)
            selected_run = make_selector(cfg["model"], 12, 0.04, seed, cfg["adaptive_weighting"]).fit(
                X[train][:, quality], y[train], X[val][:, quality], y[val], wave[quality], mi)
            write_json(job / "selection.json", asdict(selected_run))
            if not selected_run.success:
                write_json(job / "summary.json", {"seed": seed, "converged": False, "stage": "selection", "test_evaluated": False})
                continue
            selected = np.sort(available[selected_run.selected_indices])
        else:
            method = BASELINES[args.method]
            settings = json.loads((ROOT / f"configs/selectors/{method}.json").read_text())
            if "random_state" in settings["selector"]:
                settings["selector"]["random_state"] = seed - 4200
            selected, provenance = select_baseline(method, X, y, train, val, available, settings, job)
            write_json(job / "selection.json", {"selected_positions": selected, "selector_seed": seed - 4200, "provenance": provenance})
        ensemble, diagnostics = fit_ensemble(X[fit][:, selected], y[fit], wave[selected], cfg, seed + 100000,
                                             job / "native_fit", lambda _: None)
        success = all(v["success"] for v in diagnostics)
        record = {"seed": seed, "converged": success, "test_evaluated": success}
        if success:
            _, experts, weights = ensemble.predict_components(X[test][:, selected])
            prediction = (experts*weights).sum(axis=1)
            record["metrics"] = regression_metrics(y[test], prediction)
            record["metrics"]["mard_percent"] = float(100*np.mean(np.abs(prediction-y[test])/np.abs(y[test])))
            import csv
            with (job / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["sample_index", "temperature_c", "prediction_c"])
                writer.writerows(zip(test, y[test], prediction))
        write_json(job / "summary.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["random", "ordered"], required=True)
    parser.add_argument("--method", choices=["ours", *BASELINES], required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(4242, 4247)))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduction")
    main(parser.parse_args())
