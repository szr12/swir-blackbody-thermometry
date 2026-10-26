from __future__ import annotations

import argparse
from dataclasses import asdict
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "baseline", "no_background", "no_attention_fixed",
    "no_attention_pipeline", "no_redundancy", "no_overlap",
    "uniform_bands", "random_bands",
)
LABELS = {
    "baseline": "完整流程", "no_background": "固定波段不扣背景",
    "no_attention_fixed": "固定波段等权回归",
    "no_attention_pipeline": "选带与回归均等权", "no_redundancy": "无冗余惩罚重新选带",
    "no_overlap": "固定波段无温区重叠", "uniform_bands": "等间隔20波段",
    "random_bands": "随机20波段", "global_only": "仅全局模型", "hard_routing": "硬路由",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_parameters(config: dict) -> dict:
    """只接受每个超参均已冻结为单值的配置，不在消融入口搜索。"""
    grid = config["search_grid"]
    keys = {"band_count": "band_counts", "redundancy_weight": "redundancy_weights",
            "overlap_c": "overlap_values_c", "routing_width_c": "routing_width_values_c"}
    if any(len(grid[key]) != 1 for key in keys.values()):
        raise ValueError("消融入口要求全部超参已冻结为单值")
    values = {name: grid[key][0] for name, key in keys.items()}
    if int(values["band_count"]) != values["band_count"] or values["band_count"] < 2:
        raise ValueError("波段数量必须为至少2的整数")
    if values["redundancy_weight"] < 0 or values["overlap_c"] < 0 or values["routing_width_c"] <= 0:
        raise ValueError("冻结超参取值不合法")
    return values


def variant_label(variant: str, count: int) -> str:
    """选带对照名称按当前冻结预算显示，避免沿用旧20波段标签。"""
    if variant == "uniform_bands":
        return f"等间隔{count}波段"
    if variant == "random_bands":
        return f"随机{count}波段"
    return LABELS[variant]


def validate_split_config(split: dict) -> bool:
    """仅接受预先规定的随机或温度顺序划分，不从评价误差调整比例。"""
    if split == {"strategy": "temperature_stratified_random_70_15_15", "seed": 42}:
        return False
    if split == {"strategy": "temperature_ordered_70_15_15", "seed": 42,
                 "train_fraction": 0.70, "validation_fraction": 0.15}:
        return True
    raise ValueError("需要固定seed42随机划分或70/15/15温度顺序划分")


def audit_split(y, train, validation, test, ordered: bool) -> dict:
    """严格审计样本互斥及温度边界，最终拟合不包含任何Test样本。"""
    groups = [np.asarray(values, dtype=int) for values in (train, validation, test)]
    all_indices = np.concatenate(groups)
    if len(all_indices) != len(y) or len(np.unique(all_indices)) != len(y):
        raise ValueError("划分样本重复或不完整")
    if np.any(all_indices < 0) or np.any(all_indices >= len(y)):
        raise ValueError("划分索引越界")
    ranges = [[float(y[values].min()), float(y[values].max())] for values in groups]
    if ordered and ranges != [[50.0, 329.0], [330.0, 389.0], [390.0, 450.0]]:
        raise ValueError(f"本轮401样本顺序划分范围不匹配: {ranges}")
    return {"train_temperature_range_c": ranges[0], "validation_temperature_range_c": ranges[1],
            "test_temperature_range_c": ranges[2], "final_fit_temperature_range_c": [min(ranges[0][0], ranges[1][0]), max(ranges[0][1], ranges[1][1])],
            "test_in_fit": False, "temperature_ordered": ordered}


def control_bands(available: np.ndarray, wavelengths: np.ndarray, count: int, seed: int, method: str) -> np.ndarray:
    """两种对照均使用相同Train质控候选池，并满足原长波区约束。"""
    if method == "uniform_bands":
        positions = np.round(np.linspace(0, len(available) - 1, count)).astype(int)
    elif method == "random_bands":
        positions = np.random.default_rng(seed).choice(len(available), count, replace=False)
    else:
        raise ValueError(method)
    selected = available[positions].copy()
    long = (wavelengths[selected] >= 1800.0) & (wavelengths[selected] <= 2500.0)
    if not np.any(long):
        eligible = available[(wavelengths[available] >= 1800.0) & (wavelengths[available] <= 2500.0)]
        if not len(eligible):
            raise ValueError("质控后没有原协议要求的长波候选")
        selected[-1] = np.random.default_rng(seed + 1).choice(eligible)
    selected = np.sort(selected)
    if len(np.unique(selected)) != count:
        raise ValueError("对照波段集合数量或唯一性错误")
    return selected


def hard_region_ids(coarse: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """仅按粗预测划分专家，边界处使用右侧温区，越界预测截至端部温区。"""
    return np.searchsorted(boundaries[1:-1], coarse, side="right")


def metrics(y: np.ndarray, prediction: np.ndarray, boundaries: np.ndarray) -> dict:
    """等权指标为主；边界指标使用预先固定的各内部边界正负15℃范围。"""
    from traditional_utils import regression_metrics, temperature_bin_ids
    ids = temperature_bin_ids(y, boundaries)
    rows = []
    for region in range(len(boundaries) - 1):
        mask = ids == region
        if mask.any():
            rows.append({"lower_c": float(boundaries[region]), "upper_c": float(boundaries[region + 1]),
                         "n": int(mask.sum()), **regression_metrics(y[mask], prediction[mask]),
                         "mard_percent": float(np.mean(np.abs(prediction[mask] - y[mask]) / np.abs(y[mask])) * 100)})
    boundary_mask = np.min(np.abs(y[:, None] - boundaries[None, 1:-1]), axis=1) <= 15.0
    return {**regression_metrics(y, prediction), "n": len(y),
            "mard_percent": float(np.mean(np.abs(prediction - y) / np.abs(y)) * 100),
            "segments": rows,
            "boundary_15c": {"n": int(boundary_mask.sum()),
                             **regression_metrics(y[boundary_mask], prediction[boundary_mask])} if boundary_mask.any() else None}


def validate_pair(background: Path, raw: Path) -> dict:
    """核对同帧同像素的两种输入，避免把采集条件差异引入背景消融。"""
    with np.load(background, allow_pickle=False) as bg, np.load(raw, allow_pickle=False) as un:
        for key in ("group_ids", "y_temperature_c", "wavelengths_nm", "bad_band_mask"):
            if not np.array_equal(bg[key], un[key]):
                raise ValueError(f"背景消融的{key}不一致")
        if not bool(bg["background_subtracted"]) or bool(un["background_subtracted"]):
            raise ValueError("背景扣除标记错误")
        expected = un["X_raw_features"] - un["background_median_audit_only"]
        difference = float(np.max(np.abs(bg["X_raw_features"] - expected)))
        if difference > 1e-5 or expected.shape != (401, 264):
            raise ValueError(f"同帧相减或样本形状核对失败: {difference}, {expected.shape}")
        return {"shape": list(expected.shape), "same_frame_subtraction_max_error_dn": difference,
                "background_sha256": sha256(background), "raw_sha256": sha256(raw)}


def atomic_torch_save(value, path: Path) -> None:
    import torch
    temporary = path.with_suffix(".tmp.pt")
    torch.save(value, temporary)
    temporary.replace(path)


def fit_ensemble(X, y, wavelengths, config, seed, stage_dir, progress, allow_missing_experts=False):
    """逐模型保存完成的检查点；重启只重跑被中断的模型。"""
    import torch
    from convergent_colorimetry import FixedBandColorimetricMLP
    from regional_colorimetry import RegionalColorimetricEnsemble, regional_training_masks
    from traditional_utils import model_parameters, write_json
    parameters = model_parameters(config)
    overlap = float(config["search_grid"]["overlap_values_c"][0])
    boundaries = np.asarray(config["temperature_boundaries_c"])
    ensemble = RegionalColorimetricEnsemble(boundaries_c=boundaries, overlap_c=overlap,
        routing_width_c=float(config["search_grid"]["routing_width_values_c"][0]), model_parameters=parameters, seed=seed)
    masks = [np.ones(len(y), dtype=bool), *regional_training_masks(y, boundaries, overlap)]
    stage_dir.mkdir(parents=True, exist_ok=True)
    models, diagnostics = [], []
    for role, mask in enumerate(masks):
        name = "global" if role == 0 else f"expert_{role - 1}"
        progress(f"{stage_dir.name}/{name}")
        if mask.sum() < 10 and role > 0 and allow_missing_experts:
            diagnostics.append({"role": name, "seed": seed + role, "n_fit": int(mask.sum()),
                                "success": False, "status": "unavailable_training_samples", "iterations": 0})
            models.append(None)
            print(f"  {stage_dir.name}/{name}: 仅{mask.sum()}个拟合样本，记录不可用，不使用评价集补齐", flush=True)
            continue
        checkpoint = stage_dir / f"{name}.pt"
        if checkpoint.exists():
            model = torch.load(checkpoint, weights_only=False)
        else:
            if mask.sum() < 10:
                raise ValueError(f"{name}拟合样本不足")
            model = FixedBandColorimetricMLP(seed=seed + role, **parameters).fit(X[mask], y[mask], wavelengths)
            write_json(stage_dir / f"{name}.trace.json", {"trace": model.trace_})
            atomic_torch_save(model, checkpoint)
        diagnostics.append({"role": name, "seed": model.seed, "n_fit": int(mask.sum()), "success": bool(model.success_),
            "status": "success" if model.success_ else "diagnostic_nonconverged",
            "fit_temperature_range_c": [float(y[mask].min()), float(y[mask].max())],
            "raw_feature_min": X[mask].min(axis=0), "raw_feature_max": X[mask].max(axis=0),
            "iterations": int(model.iterations_), "training_rmse_c": model.training_rmse_c_,
            "training_adaptive_rmse_c": model.training_adaptive_rmse_c_,
            "input_dimension": model.model_.network[0].in_features,
            "floors": model.floors_, "offsets": model.offsets_,
            "feature_mean": model.feature_mean_, "feature_scale": model.feature_scale_,
            "target_mean": model.y_mean_, "target_scale": model.y_scale_})
        models.append(model)
        print(f"  {stage_dir.name}/{name}: iter={model.iterations_}, RMSE={model.training_rmse_c_:.3f}, success={model.success_}", flush=True)
    ensemble.global_model_, ensemble.experts_ = models[0], models[1:]
    write_json(stage_dir / "fit_diagnostics.json", {"models": diagnostics})
    return ensemble, diagnostics


def evaluate(ensemble, X, y, sample_indices, boundaries, stage, job_dir, diagnostics):
    """全局与硬路由复用相同模型，不允许真实评价温度进入路由。"""
    from traditional_utils import write_json, regression_metrics
    missing = [i for i, model in enumerate(ensemble.experts_) if model is None]
    if missing:
        from regional_colorimetry import soft_region_weights
        coarse = ensemble.global_model_.predict(X)
        experts = np.column_stack([model.predict(X) if model is not None else np.full(len(y), np.nan)
                                   for model in ensemble.experts_])
        weights = soft_region_weights(coarse, boundaries, ensemble.routing_width_c)
    else:
        coarse, experts, weights = ensemble.predict_components(X)
    hard = hard_region_ids(coarse, boundaries)
    predictions = {"regional": np.sum(experts * weights, axis=1), "global_only": coarse,
                   "hard_routing": experts[np.arange(len(y)), hard]}
    result = {}
    for name, prediction in predictions.items():
        if missing and name != "global_only":
            result[name] = {"converged": False, "metrics": None, "status": "unavailable_training_samples",
                            "missing_expert_ids": missing}
            continue
        roles = diagnostics[:1] if name == "global_only" else diagnostics
        success = all(row["success"] for row in roles)
        result[name] = {"converged": success, "metrics": metrics(y, prediction, boundaries),
            "temperature_range_c": [float(y.min()), float(y.max())],
            "prediction_range_c": [float(prediction.min()), float(prediction.max())],
            "status": "success" if success else "diagnostic_nonconverged"}
    # 自适应评价权重来自冻结网络，等权结果始终作为跨消融比较口径。
    adaptive = ensemble.global_model_.attention_weights(X)
    if not missing:
        result["regional"]["adaptive_metrics"] = regression_metrics(y, predictions["regional"], adaptive)
    # 域外与截断计数只用于评价诊断，绝不回流到拟合或选带。
    domain = []
    for model, diagnostic in zip([ensemble.global_model_, *ensemble.experts_], diagnostics):
        if model is None:
            continue
        if "raw_feature_min" in diagnostic:
            outside = (X < np.asarray(diagnostic["raw_feature_min"])) | (X > np.asarray(diagnostic["raw_feature_max"]))
            clipped = X + model.offsets_[None, :] < model.floors_[None, :]
            domain.append({"role": diagnostic["role"], "feature_entries": int(X.size),
                           "outside_fit_range_entries": int(outside.sum()), "floor_clipped_entries": int(clipped.sum())})
    write_json(job_dir / f"{stage}.domain_audit.json", {"models": domain})
    write_json(job_dir / f"{stage}.evaluation.json", result)
    with (job_dir / f"{stage}.predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "temperature_c", "regional_c", "global_c", "hard_routing_c",
                         "hard_expert", "adaptive_weight", *[f"expert_{i}_c" for i in range(4)],
                         *[f"soft_weight_{i}" for i in range(4)]])
        for i in range(len(y)):
            writer.writerow([sample_indices[i], y[i], predictions["regional"][i], coarse[i], predictions["hard_routing"][i],
                             hard[i], adaptive[i], *experts[i], *weights[i]])
    return result


def aggregate(output: Path, seeds: list[int]) -> dict:
    """失败拟合只进入诊断表，不进入成功结果平均值和配对差值。"""
    from traditional_utils import write_json
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else {}
    parameters = fixed_parameters(protocol["config"]) if "config" in protocol else {"band_count":20,"redundancy_weight":0.06,"overlap_c":35.0,"routing_width_c":35.0}
    count = int(parameters["band_count"])
    ordered = protocol.get("config", {}).get("split", {}).get("strategy") == "temperature_ordered_70_15_15"
    scope = ("同批次温度顺序外推：Train50–329℃、Validation330–389℃、最终拟合50–389℃、Test390–450℃；"
             "这批高温样本此前已被查看，属于开发性外推评价；重复初始化不构成独立重复采集") if ordered else "seed42已查看的开发性Test；重复初始化不构成独立重复采集"
    results = []
    for seed in seeds:
        for variant in VARIANTS:
            path = output / f"seed_{seed}" / variant / "summary.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            results.append(result)
            if variant == "baseline" and "test" in result:
                for derived in ("global_only", "hard_routing"):
                    row = {**result, "variant": derived, "test": result["derived_test"][derived],
                           "status": result["derived_test"][derived]["status"]}
                    results.append(row)
            elif variant == "baseline":
                for derived in ("global_only", "hard_routing"):
                    results.append({**result, "variant": derived})
    rows = []
    for variant in (*VARIANTS, "global_only", "hard_routing"):
        entries = [r for r in results if r["variant"] == variant]
        successful = [r for r in entries if r.get("test", {}).get("converged", False) and r.get("selection_success", False)]
        row = {"variant": variant, "label": variant_label(variant,count), "completed": len(entries), "successful": len(successful),
               "diagnostic_or_failed": len(entries) - len(successful), "seeds": seeds}
        for key in ("rmse", "mae", "mard_percent", "r2"):
            values = [r["test"]["metrics"][key] for r in successful]
            row[key + "_mean"] = float(np.mean(values)) if values else None
            row[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        paired = []
        for result in successful:
            baseline = next((r for r in results if r["seed"] == result["seed"] and r["variant"] == "baseline"), None)
            if baseline and baseline.get("test", {}).get("converged", False) and baseline["selection_success"]:
                paired.append({"seed": result["seed"], "delta_rmse_c": result["test"]["metrics"]["rmse"] - baseline["test"]["metrics"]["rmse"]})
        row["paired_deltas"] = paired
        row["paired_delta_rmse_mean"] = float(np.mean([r["delta_rmse_c"] for r in paired])) if paired else None
        rows.append(row)
    summary = {"test_scope": scope,
               "rows": rows, "individual": results}
    write_json(output / "aggregate.json", summary)
    with (output / "aggregate.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["variant", "label", "completed", "successful", "diagnostic_or_failed", "rmse_mean", "rmse_std",
                  "mae_mean", "mard_percent_mean", "r2_mean", "paired_delta_rmse_mean"]
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    description = ("温度顺序280/60/61划分，Train50–329℃、Validation330–389℃、最终拟合50–389℃、Test390–450℃") if ordered else "seed42固定280/60/61划分"
    lines = [f"# 扣背景K{count}组件消融" + ("：温度顺序外推" if ordered else ""), "", f"协议：暗环境60 cm、401样本、{description}，K={count}、ρ={parameters['redundancy_weight']}、重叠{parameters['overlap_c']}℃、路由{parameters['routing_width_c']}℃。",
             "训练初始化：" + "、".join(map(str, seeds)) + "。均值和标准差仅统计收敛成功的重复；失败结果另列。",
             scope, "",
             "| 方案 | 完成 | 成功 | RMSE均值±标准差（℃） | MAE（℃） | MARD（%） | 配对RMSE变化（℃） |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "—" if value is None else f"{value:.3f}"
    for row in rows:
        lines.append(f"| {row['label']} | {row['completed']} | {row['successful']} | {fmt(row['rmse_mean'])} ± {fmt(row['rmse_std'])} | {fmt(row['mae_mean'])} | {fmt(row['mard_percent_mean'])} | {fmt(row['paired_delta_rmse_mean'])} |")
    segment_label = "390–450℃RMSE" if ordered else "50–150℃RMSE"
    lines.extend(["", "## 各次结果与失败诊断", "", f"| 种子 | 方案 | 状态 | Test RMSE（℃） | {segment_label}（℃） |", "|---|---|---|---:|---:|"])
    for row in results:
        test = row.get("test", {})
        value = test.get("metrics", {})
        low = value.get("rmse") if ordered else next((r["rmse"] for r in value.get("segments", []) if r["lower_c"] == 50.0), None)
        lines.append(f"| {row['seed']} | {variant_label(row['variant'],count)} | {row['status']} | {fmt(value.get('rmse'))} | {fmt(low)} |")
    lines.extend(["", "完整分温区、边界±15℃、逐样本预测及训练参数见各任务目录和aggregate.json。",
                  "固定波段消融使用对应初始化的基线Train波段；无冗余及全流程等权组重新选带；对照选带使用相同Train质控池和长波约束。",
                  "无重叠组保留原实现的闭区间掩码，因此精确边界样本仍可被相邻专家共享；此实验仅取消向邻区扩展。",
                  "非收敛组的Test数值仅用于诊断，不用于成功均值或性能改善结论。"])
    if ordered:
        lines.extend(["无重叠组Train阶段没有高温专家样本，Validation区域预测标记不可用；固定方案后以50–389℃最终拟合数据重训全部专家。",
                      "Test仅覆盖390–450℃，不报告未覆盖低温区指标；本组也未覆盖内部温区边界正负15℃评价范围。",
                      "成功表示预设训练收敛门槛通过，不代表外推误差已达到可接受水平。"])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run(args):
    import torch
    from traditional_utils import (write_json, load_data, fold_quality_mask, make_configured_split, make_selector)
    from blackbody_hsic.selector_runtime import load_or_compute_mi
    output, project = args.output_dir.resolve(), args.project_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["model"]["device"] = args.device
    seeds = args.seeds
    parameters = fixed_parameters(config)
    count = int(parameters["band_count"])
    ordered = validate_split_config(config["split"])
    raw_path = project / "preprocessing_validation/preprocessed_swir_no_background/swir_roi_no_background.npz"
    bg_path = project / config["data_path"]
    paired = validate_pair(bg_path, raw_path)
    provenance = config.get("frozen_search_provenance")
    if provenance:
        reference = project / provenance["path"]
        if not reference.exists():
            reference = project / provenance["remote_fallback_path"]
        if sha256(reference) != provenance["sha256"]:
            raise ValueError("最新冻结超参文件哈希不一致")
        frozen = json.loads(reference.read_text(encoding="utf-8"))
        if frozen["status"] != "frozen_from_validation_only" or frozen["test_evaluated"] or frozen["parameters"] != parameters:
            raise ValueError("配置与Validation冻结结果不一致")
    runtime = args.runtime_dir.resolve() if args.runtime_dir else Path(__file__).resolve().parent
    sources = {str(p.relative_to(runtime)): sha256(p) for p in sorted(runtime.rglob("*.py"))} if args.runtime_dir else {
        name: sha256(runtime / name) for name in ("run_component_ablations.py", "convergent_colorimetry.py", "regional_colorimetry.py", "traditional_utils.py")}
    if provenance:
        if paired["background_sha256"] != provenance["data_sha256"]:
            raise ValueError("输入与最新超参搜索数据不一致")
        for name, expected in provenance["runtime_sources_sha256"].items():
            if name == "run_phase7_traditional_hyperparameter_search.py":
                continue
            if sha256(runtime / name) != expected:
                raise ValueError(f"模型源码与最新超参搜索不一致: {name}")
    protocol = {"config": config, "seeds": seeds, "variants": VARIANTS, "data_pair": paired, "sources": sources,
                "final_fit_seed_offset": 100000,
                "test_scope": "同批次温度顺序外推开发评价；所有组预先确定，不依据Test调参" if ordered else "固定seed42开发性Test；所有组预先确定，不依据Test调参",
                "unavailable_expert_policy": "Train阶段记录不可用专家，不补入评价样本；Train+Validation固定配置重训全部专家",
                "boundary_half_width_c": 15.0, "checkpoint_resume": "复用已完成模型；中断模型从头重训"}
    path = output / "protocol.json"
    # 源码、输入和配置改变时要求新目录，避免复用不匹配的检查点。
    encoded = json.loads(json.dumps(protocol, default=list))
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != encoded:
        raise ValueError("目录已有不同协议，请使用新的输出目录")
    write_json(path, protocol)
    write_json(output / "environment.json", {"torch": torch.__version__, "numpy": np.__version__,
        "cuda": torch.version.cuda, "device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None})
    data = load_data(bg_path, config["feature_key"], config["uncertainty_key"])
    raw_data = load_data(raw_path, config["feature_key"], config["uncertainty_key"])
    X, raw_X = data[config["feature_key"]], raw_data[config["feature_key"]]
    y, wavelengths = data["y_temperature_c"], data["wavelengths_nm"]
    boundaries = np.asarray(config["temperature_boundaries_c"])
    train, validation, test = make_configured_split(y, boundaries, config["split"])
    fit_indices = np.sort(np.concatenate([train, validation]))
    if [len(train), len(validation), len(test)] != [280, 60, 61]:
        raise ValueError("划分数量不匹配")
    split_audit = audit_split(y, train, validation, test, ordered)
    write_json(output / "split_manifest.json", {"train_indices": train, "validation_indices": validation, "test_indices": test, **split_audit})
    quality, diagnostic = fold_quality_mask(X[train], data[config["uncertainty_key"]][train], data["background_mad"][train],
        data["bad_pixel_fraction_by_sample_band"][train], data["saturation_fraction_by_sample_band"][train], wavelengths, config["fold_quality_control"])
    available = np.flatnonzero(quality)
    write_json(output / "quality.json", {"diagnostics": diagnostic, "available_positions": available})
    print(f"准备完成：样本={X.shape}, 候选={len(available)}, 初始化={seeds}, 训练任务={len(seeds) * len(VARIANTS)}", flush=True)
    if args.prepare_only:
        return
    jobs = [f"seed_{seed}/{variant}" for seed in seeds for variant in VARIANTS]
    state = {"status": "running", "pid": os.getpid(), "total": len(jobs), "completed": {}, "current": None}
    def progress(stage):
        state["stage"], state["updated_epoch"] = stage, time.time()
        write_json(output / "queue_state.json", state)
    (output / "queue.pid").write_text(str(os.getpid()), encoding="utf-8")
    try:
        for seed in seeds:
            selection_records = {}
            for variant in VARIANTS:
                key = f"seed_{seed}/{variant}"
                job_dir = output / key
                job_dir.mkdir(parents=True, exist_ok=True)
                state["current"] = key
                summary_path = job_dir / "summary.json"
                if summary_path.exists():
                    old = json.loads(summary_path.read_text(encoding="utf-8"))
                    if "selection" in old:
                        selection_records[variant] = old["selection"]
                    state["completed"][key] = old["status"]
                    continue
                started = time.time()
                print(f"开始 {key}", flush=True)
                progress("preparing")
                variant_config = copy.deepcopy(config)
                if variant in ("no_attention_fixed", "no_attention_pipeline"):
                    variant_config["adaptive_weighting"]["enabled"] = False
                if variant == "no_overlap":
                    variant_config["search_grid"]["overlap_values_c"] = [0.0]
                if variant == "no_redundancy":
                    variant_config["search_grid"]["redundancy_weights"] = [0.0]
                if variant == "no_background":
                    variant_config["data_path"] = str(raw_path.relative_to(project))
                variant_config["model"]["seed"] = seed
                write_json(job_dir / "config.snapshot.json", variant_config)
                try:
                    selection_path = job_dir / "selection.json"
                    if selection_path.exists():
                        record = json.loads(selection_path.read_text(encoding="utf-8"))
                    elif variant in ("baseline", "no_attention_pipeline", "no_redundancy"):
                        progress("selecting_bands")
                        cache = output / "cache/training_mutual_information.npy"
                        if not cache.exists():
                            # 匹配当前输入的历史Train缓存；矩阵和输入指纹均验证后再复用。
                            reference_name = "darkroom_background_subtracted_extrapolation" if ordered else "darkroom_background_subtracted_main"
                            source = project / f"artifacts/results/sunroom_darkroom_60cm_fixed_protocol/{reference_name}/cache/training_mutual_information.npy"
                            if not source.exists():
                                source = project / f"optimization_experiment/outputs/sunroom_darkroom_60cm_fixed_protocol/{reference_name}/cache/training_mutual_information.npy"
                            meta = source.with_suffix(".json")
                            expected = {"training_features_sha256": hashlib.sha256(np.ascontiguousarray(X[train][:, quality], dtype=np.float64).tobytes()).hexdigest(),
                                        "shape": [len(train), len(available)], "neighbors": 3, "seed": 42}
                            if source.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8")) == expected:
                                cache.parent.mkdir(parents=True, exist_ok=True)
                                np.save(cache, np.load(source, allow_pickle=False), allow_pickle=False)
                        mi = load_or_compute_mi(cache, X[train][:, quality], 3, 42)
                        rho = 0.0 if variant == "no_redundancy" else float(parameters["redundancy_weight"])
                        variant_config["search_grid"]["redundancy_weights"] = [rho]
                        write_json(job_dir / "config.snapshot.json", variant_config)
                        selection = make_selector(variant_config["model"], count, rho, seed, variant_config["adaptive_weighting"]).fit(
                            X[train][:, quality], y[train], X[validation][:, quality], y[validation], wavelengths[quality], mi)
                        record = asdict(selection)
                        record["selected_positions"] = available[selection.selected_indices]
                        record["method"] = variant
                        write_json(selection_path, record)
                    elif variant in ("uniform_bands", "random_bands"):
                        record = {"selected_positions": control_bands(available, wavelengths, count, seed, variant), "success": True, "method": variant}
                        write_json(selection_path, record)
                    else:
                        record = copy.deepcopy(selection_records["baseline"])
                        record["method"] = "fixed_baseline_bands"
                        write_json(selection_path, record)
                    selection_records[variant] = record
                    selected = np.asarray(record["selected_positions"], dtype=int)
                    selected = np.sort(selected)
                    write_json(job_dir / "selected_bands.json", {"original_band_indices": data["original_band_indices"][selected],
                        "wavelengths_nm": wavelengths[selected], "selection_success": record["success"]})
                    if not record["success"]:
                        result = {"seed": seed, "variant": variant, "status": "failed_selection", "selection": record,
                                  "selection_success": False, "test_evaluated": False}
                    else:
                        values = (raw_X if variant == "no_background" else X)[:, selected]
                        ensemble, fit_diagnostics = fit_ensemble(values[train], y[train], wavelengths[selected], variant_config, seed, job_dir / "train_only", progress, allow_missing_experts=ordered)
                        validation_result = evaluate(ensemble, values[validation], y[validation], validation, boundaries, "validation", job_dir, fit_diagnostics)
                        del ensemble
                        ensemble, final_diagnostics = fit_ensemble(values[fit_indices], y[fit_indices], wavelengths[selected], variant_config, seed + 100000, job_dir / "train_plus_validation", progress)
                        progress("evaluating_test")
                        test_result = evaluate(ensemble, values[test], y[test], test, boundaries, "test", job_dir, final_diagnostics)
                        available_diagnostics = [r for r in fit_diagnostics if r.get("status") != "unavailable_training_samples"]
                        pipeline_success = all(row["success"] for row in available_diagnostics + final_diagnostics)
                        for name, evaluation in test_result.items():
                            evaluation["converged"] = bool(evaluation["converged"] and all(row["success"] for row in (available_diagnostics[:1] if name == "global_only" else available_diagnostics)))
                            evaluation["status"] = "success" if evaluation["converged"] else "diagnostic_nonconverged"
                        write_json(job_dir / "test.evaluation.json", test_result)
                        result = {"seed": seed, "variant": variant,
                            "status": "success" if pipeline_success else "diagnostic_nonconverged",
                            "selection_success": True, "selection": record, "validation": validation_result["regional"],
                            "test": test_result["regional"], "derived_test": test_result,
                            "train_diagnostics": fit_diagnostics, "final_diagnostics": final_diagnostics,
                            "validation_regional_available": validation_result["regional"]["status"] != "unavailable_training_samples",
                            "split_audit": split_audit,
                            "test_evaluated": True, "elapsed_seconds": time.time() - started}
                        del ensemble
                    write_json(summary_path, result)
                    if provenance and not ordered and variant == "baseline" and seed == 4242 and "test" in result:
                        selected_match = np.array_equal(data["original_band_indices"][selected], frozen["selected_band_indices"])
                        write_json(output / "frozen_selection_reproduction.json", {
                            "frozen_sha256": provenance["sha256"], "same_selected_bands": selected_match,
                            "validation_metrics": result["validation"].get("metrics"),
                            "validation_adaptive_metrics": validation_result["regional"].get("adaptive_metrics"),
                        })
                        if not selected_match:
                            raise RuntimeError("新基线波段未复现最新超参冻结结果")
                    if not provenance and variant == "baseline" and seed == 4242 and "test" in result:
                        reference_name = "darkroom_background_subtracted_extrapolation" if ordered else "darkroom_background_subtracted_main"
                        reference = project / f"artifacts/results/sunroom_darkroom_60cm_fixed_protocol/{reference_name}/summary.json"
                        if not reference.exists():
                            reference = project / f"optimization_experiment/outputs/sunroom_darkroom_60cm_fixed_protocol/{reference_name}/summary.json"
                        if reference.exists():
                            prior = json.loads(reference.read_text(encoding="utf-8"))
                            new_rmse = result["test"]["metrics"]["rmse"]
                            old_rmse = prior["regional_test_metrics"]["rmse"]
                            write_json(output / "baseline_reproduction.json", {
                                "reference_path": str(reference), "reference_sha256": sha256(reference),
                                "same_selected_bands": np.array_equal(data["original_band_indices"][selected], prior["selected_bands"]["selected_band_indices"]),
                                "reference_rmse_c": old_rmse, "current_rmse_c": new_rmse,
                                "rmse_difference_c": new_rmse - old_rmse,
                            })
                    state["completed"][key] = result["status"]
                except Exception as error:
                    write_json(job_dir / "failure.json", {"error": str(error), "traceback": traceback.format_exc()})
                    state["completed"][key] = "execution_error"
                    print(traceback.format_exc(), flush=True)
                aggregate(output, seeds)
                progress("job_complete")
                print(f"结束 {key}: {state['completed'][key]}, {len(state['completed'])}/{len(jobs)}", flush=True)
        state["status"] = "complete" if not any(s == "execution_error" for s in state["completed"].values()) else "complete_with_execution_errors"
        state["current"] = None
        progress("complete")
        aggregate(output, seeds)
    except BaseException:
        state["status"] = "interrupted"
        progress("interrupted")
        raise


def main():
    parser = argparse.ArgumentParser(description="冻结超参协议的可恢复组件消融")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[4242])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.project_root / "main_experiment/src"))
    if args.runtime_dir:
        sys.path.insert(0, str(args.runtime_dir.resolve()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 操作系统级锁防止两个队列写入同一个输出目录；退出时自动释放。
    with (args.output_dir / "queue.lock").open("a+b") as lock:
        if os.name == "posix":
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        run(args)


if __name__ == "__main__":
    main()
