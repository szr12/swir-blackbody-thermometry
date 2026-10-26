from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_and_report(results: Path) -> None:
    """根据逐样本预测独立重算指标，并核对冻结记录及测试划分。"""
    output = results / "final_test_evaluation"
    summary = read_json(output / "summary.json")
    if summary["status"] != "success" or summary["test_evaluated"] is not True:
        raise ValueError("最终模型未成功完成测试")
    for name, expected in summary["frozen_sources_sha256"].items():
        if sha256(results / name) != expected:
            raise ValueError(f"冻结记录已改变：{name}")
    search_config = read_json(results / "config.snapshot.json")
    if sha256(ROOT / search_config["data_path"]) != summary["source_data_sha256"]:
        raise ValueError("本地数据与测试输入不一致")
    frozen = read_json(results / "frozen_parameters.json")
    if summary["parameters"] != frozen["parameters"]:
        raise ValueError("测试超参数与冻结参数不一致")
    manifest = read_json(output / "split_manifest.json")
    fit_indices = manifest["fit_indices"]
    test_indices = manifest["test_indices"]
    if set(fit_indices) & set(test_indices) or len(set(fit_indices + test_indices)) != 401:
        raise ValueError("拟合与测试划分有重叠或样本遗漏")
    if len(fit_indices) != 340 or len(test_indices) != 61:
        raise ValueError("拟合与测试数量不符合冻结协议")
    with np.load(output / "test_predictions.npz", allow_pickle=False) as archive:
        actual_indices = archive["test_indices"]
        actual_bands = archive["selected_band_indices"]
        truth = archive["y_true_c"]
        prediction = archive["prediction_c"]
        attention = archive["attention_weights"]
        routing = archive["routing_weights"]
        experts = archive["expert_predictions_c"]
    if actual_indices.tolist() != test_indices or actual_bands.tolist() != frozen["selected_band_indices"]:
        raise ValueError("预测记录与冻结波段或Test索引不一致")
    if not np.isfinite(prediction).all() or not np.isfinite(attention).all() or np.any(attention <= 0):
        raise ValueError("预测或注意力权重不合法")
    if not np.allclose(np.sum(routing, axis=1), 1.0) or not np.allclose(np.sum(routing * experts, axis=1), prediction):
        raise ValueError("软路由权重与最终预测不一致")
    errors = prediction - truth
    metrics = {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "r2": float(1 - np.sum(errors ** 2) / np.sum((truth - truth.mean()) ** 2)),
        "bias": float(np.mean(errors)),
        "max_absolute_error": float(np.max(np.abs(errors))),
    }
    for key, value in metrics.items():
        if not np.isclose(value, summary["regional_test_metrics"][key], rtol=0, atol=1e-10):
            raise ValueError(f"独立重算不一致：{key}")
    weighted_rmse = float(np.sqrt(np.sum(attention * errors ** 2) / np.sum(attention)))
    relative = float(np.mean(np.abs(errors) / np.abs(truth)))
    if not np.isclose(weighted_rmse, summary["regional_adaptive_test_metrics"]["rmse"], rtol=0, atol=1e-10):
        raise ValueError("加权RMSE独立重算不一致")
    if not np.isclose(relative, summary["regional_test_relative_deviation"], rtol=0, atol=1e-10):
        raise ValueError("相对偏差独立重算不一致")
    for row in summary["regional_test_segment_metrics"]:
        mask = (truth >= row["lower_c"]) & (
            truth <= row["upper_c"] if row["upper_inclusive"] else truth < row["upper_c"]
        )
        if int(mask.sum()) != row["n_samples"] or not np.isclose(
            np.sqrt(np.mean(errors[mask] ** 2)), row["rmse"], rtol=0, atol=1e-10,
        ):
            raise ValueError("分温区指标独立重算不一致")
    parameters = summary["parameters"]
    lines = [
        "# 扣背景冻结参数的开发性Test评价", "",
        f"固定参数：K={parameters['band_count']}，λ={parameters['redundancy_weight']}，"
        f"重叠{parameters['overlap_c']:g} ℃，路由{parameters['routing_width_c']:g} ℃。",
        "暗环境60 cm；401样本×264波段输入。仅使用搜索冻结的12个波段，Helmert回归输入为11维。",
        "搜索采用Train/Validation/Test=280/60/61，seed=42；冻结后按原流程使用Train+Validation共340个样本重训。",
        f"最终模型训练seed={summary['training_seed']}，训练达到既定收敛门槛后评价61个Test样本。",
        "本轮Test未用于选带、选择超参数或选择训练种子。原seed42 Test此前开发中已查看，因此本结果标为开发性测试。",
        "随机温区分层划分检验同一采集条件下的插值表现；本轮未检验高温外推、不同采集批次或不同光照条件。", "",
        "| Test指标 | 数值 |", "|---|---:|",
        f"| 等权RMSE | {metrics['rmse']:.6f} ℃ |",
        f"| 自适应加权RMSE | {weighted_rmse:.6f} ℃ |",
        f"| 等权MAE | {metrics['mae']:.6f} ℃ |",
        f"| R² | {metrics['r2']:.6f} |",
        f"| 平均绝对相对偏差 | {100 * relative:.6f}% |",
        f"| 平均有符号误差 | {metrics['bias']:.6f} ℃ |",
        f"| 最大绝对误差 | {metrics['max_absolute_error']:.6f} ℃ |", "",
        "## 分温区Test结果", "",
        "| 温区 | 样本数 | 等权RMSE (℃) | MAE (℃) | 最大绝对误差 (℃) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["regional_test_segment_metrics"]:
        end = "]" if row["upper_inclusive"] else ")"
        lines.append(f"| [{row['lower_c']:g}, {row['upper_c']:g}{end} ℃ | {row['n_samples']} | "
                     f"{row['rmse']:.6f} | {row['mae']:.6f} | {row['max_absolute_error']:.6f} |")
    lines += ["", "## 核验", "",
              "已核对数据和运行代码SHA-256、冻结参数及波段、所有阶段一致的划分、拟合与Test索引无重叠。",
              "已从保存的逐样本预测独立重算等权指标、加权RMSE、相对偏差与分温区RMSE，全部匹配。",
              "此前的Validation结果（等权RMSE=5.300973 ℃，加权RMSE=5.560304 ℃）来自280样本训练的模型；本次Test来自340样本重训的模型。",
              "结果与预测文件位于artifacts/results/swir_background_symmetric_search_20261013/final_test_evaluation/。", "",
              "可随Git提交的结果副本见 [实验结果证据](../evidence/swir_background_symmetric_search_20261013/README.md)，"
              "[61个Test逐样本预测](../../artifacts/results/swir_background_symmetric_search_20261013/final_test_evaluation/test_predictions.csv)。", ""]
    report = ROOT / "optimization_experiment/reports/swir_background_symmetric_search_20261013_test.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    (output / "verification.json").write_text(json.dumps({
        "status": "passed", "n_test": len(truth),
        "predictions_sha256": sha256(output / "test_predictions.npz"),
        "verified_metrics": metrics, "verified_adaptive_rmse_c": weighted_rmse,
        "verified_relative_deviation": relative,
        "frozen_sources_unchanged": True, "fit_and_test_disjoint": True,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verification": "passed", "report": str(report), "test_metrics": metrics,
                      "adaptive_rmse_c": weighted_rmse, "relative_deviation": relative}, ensure_ascii=False, indent=2))


def evaluate(results: Path, runtime: Path) -> None:
    """冻结后按既有340样本重训流程进行一次开发性Test评价。"""
    output = results / "final_test_evaluation"
    if (output / "summary.json").exists():
        raise ValueError("已有最终测试结果，请读取已有结果，避免重复评价")
    preflight = read_json(results / "preflight.json")
    config = read_json(results / "config.snapshot.json")
    frozen = read_json(results / "frozen_parameters.json")
    search_summary = read_json(results / "summary.json")
    data_path = ROOT / config["data_path"]
    if sha256(data_path) != preflight["data_sha256"]:
        raise ValueError("测试输入与搜索输入不一致")
    for name, expected in preflight["runtime_sources_sha256"].items():
        if sha256(runtime / name) != expected:
            raise ValueError(f"运行代码指纹不一致：{name}")
    if frozen["status"] != "frozen_from_validation_only" or search_summary["status"] != "search_complete":
        raise ValueError("尚未完成搜索与冻结")
    if frozen["parameters"] != search_summary["best_hyperparameters"] or frozen["background_subtracted"] is not True:
        raise ValueError("冻结参数或背景处理不匹配")
    if len(set(frozen["selected_band_indices"])) != frozen["parameters"]["band_count"]:
        raise ValueError("冻结波段数量不一致")
    source_hashes = {name: sha256(results / name) for name in (
        "frozen_parameters.json", "summary.json", "config.snapshot.json", "preflight.json",
    )}
    sys.path.insert(0, str(ROOT / "main_experiment/src"))
    sys.path.insert(0, str(runtime))
    from regional_colorimetry import RegionalColorimetricEnsemble
    from traditional_utils import (
        load_data, make_configured_split, model_parameters, regression_metrics,
        mean_absolute_relative_deviation, segmented_metrics, write_json,
    )

    data = load_data(data_path, config["feature_key"], config["uncertainty_key"])
    X = data[config["feature_key"]]
    y = data["y_temperature_c"]
    wavelengths = data["wavelengths_nm"]
    boundaries = np.asarray(config["temperature_boundaries_c"], dtype=np.float64)
    train, validation, test = make_configured_split(y, boundaries, config["split"])
    for stage in search_summary["stages"]:
        manifest = read_json(results / stage["name"] / "split_manifest.json")
        if any(manifest[f"{part}_indices"] != indices.tolist() for part, indices in (
            ("train", train), ("validation", validation), ("test", test),
        )):
            raise ValueError("测试划分与搜索阶段不一致")
    fit = np.sort(np.concatenate([train, validation]))
    if np.intersect1d(fit, test).size or [len(fit), len(test)] != [340, 61]:
        raise ValueError("最终拟合与Test划分错误")
    positions = {int(index): i for i, index in enumerate(data["original_band_indices"])}
    selected = np.asarray([positions[int(index)] for index in frozen["selected_band_indices"]])
    if not np.allclose(wavelengths[selected], frozen["selected_wavelengths_nm"], rtol=0, atol=1e-10):
        raise ValueError("冻结中心波长与数据不匹配")
    parameters = frozen["parameters"]
    training_seed = int(config["model"]["seed"]) + 100000
    output.mkdir(exist_ok=True)
    write_json(output / "split_manifest.json", {
        "train_indices": train, "validation_indices": validation, "test_indices": test,
        "fit_indices": fit, "split_seed": config["split"]["seed"],
    })
    write_json(output / "progress.json", {"status": "fitting_train_plus_validation", "test_evaluated": False})
    print(f"冻结参数已核验，Train+Validation={len(fit)}，Test={len(test)}，训练seed={training_seed}", flush=True)
    model = RegionalColorimetricEnsemble(
        boundaries_c=boundaries, overlap_c=parameters["overlap_c"],
        routing_width_c=parameters["routing_width_c"],
        model_parameters=model_parameters(config), seed=training_seed,
    ).fit(X[fit][:, selected], y[fit], wavelengths[selected])
    if not model.success_:
        write_json(output / "progress.json", {"status": "failed_to_converge", "test_evaluated": False})
        raise RuntimeError("最终模型未达到既定收敛门槛，Test尚未评价")
    write_json(output / "progress.json", {"status": "evaluating_test", "test_evaluated": True})
    coarse, experts, routing = model.predict_components(X[test][:, selected])
    prediction = np.sum(experts * routing, axis=1)
    attention = model.global_model_.attention_weights(X[test][:, selected])
    np.savez_compressed(output / "test_predictions.npz", test_indices=test, y_true_c=y[test],
                        prediction_c=prediction, coarse_c=coarse, expert_predictions_c=experts,
                        routing_weights=routing, attention_weights=attention,
                        selected_band_indices=frozen["selected_band_indices"])
    with (output / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "reference_temperature_c", "prediction_c", "error_c", "absolute_error_c", "attention_weight"])
        writer.writerows((int(index), float(actual), float(predicted), float(predicted - actual),
                          float(abs(predicted - actual)), float(weight))
                         for index, actual, predicted, weight in zip(test, y[test], prediction, attention))
    summary = {
        "status": "success", "test_evaluated": True, "test_used_for_hyperparameter_selection": False,
        "test_used_for_training_or_stopping": False, "historical_test_seen": True,
        "evidence_classification": "development_test", "background_subtracted": True,
        "fit_scope": "train_plus_validation_after_freeze", "n_fit": len(fit), "n_test": len(test),
        "training_seed": training_seed, "parameters": parameters,
        "selected_band_indices": frozen["selected_band_indices"],
        "selected_wavelengths_nm": frozen["selected_wavelengths_nm"],
        "source_data_sha256": preflight["data_sha256"], "runtime_sources_sha256": preflight["runtime_sources_sha256"],
        "frozen_sources_sha256": source_hashes, "training_converged": bool(model.success_),
        "global_training_rmse_c": model.global_model_.training_rmse_c_,
        "global_training_adaptive_rmse_c": model.global_model_.training_adaptive_rmse_c_,
        "regional_training": [row.__dict__ for row in model.diagnostics_],
        "regional_test_metrics": regression_metrics(y[test], prediction),
        "regional_adaptive_test_metrics": regression_metrics(y[test], prediction, attention),
        "regional_test_relative_deviation": mean_absolute_relative_deviation(y[test], prediction),
        "regional_test_segment_metrics": segmented_metrics(y[test], prediction, boundaries),
        "coarse_test_metrics": regression_metrics(y[test], coarse),
        "adaptive_weight_source": "frozen_training_attention_network",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if any(sha256(results / name) != expected for name, expected in source_hashes.items()):
        raise ValueError("测试期间冻结记录发生改变")
    write_json(output / "summary.json", summary)
    write_json(output / "progress.json", {"status": "test_complete", "test_evaluated": True})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_and_report(args.results_dir.resolve())
    else:
        if args.runtime_dir is None:
            parser.error("进行最终测试需要--runtime-dir")
        evaluate(args.results_dir.resolve(), args.runtime_dir.resolve())


if __name__ == "__main__":
    main()
