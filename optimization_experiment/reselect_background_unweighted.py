from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

from hyperparameter_objectives import configured_metric, rank_candidates, selection_objective, validation_score
from run_background_symmetric_search import symmetric_float_values, symmetric_k_values


ROOT = Path(__file__).resolve().parents[1]
NAME = "swir_background_unweighted_selection_20261015"
KEYS = ("band_count", "redundancy_weight", "overlap_c", "routing_width_c")
METRIC = "unweighted_rmse"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(source: Path, output: Path, config_path: Path) -> dict:
    """严格沿等权择优路径重排已保存候选；路径改变时报告缺少的训练候选。"""
    if output.exists() and any(output.iterdir()):
        raise ValueError("结果目录已存在，请使用新目录以保留审计记录")
    old_config = read(source / "config.snapshot.json")
    config = read(config_path)
    if configured_metric(config) != METRIC or config["evaluate_test_after_search"]:
        raise ValueError("本轮必须按等权Validation选参且不评价Test")
    preserved = copy.deepcopy(config)
    for key in ("experiment_name", "validation_objective"):
        preserved[key] = old_config[key]
    if preserved != old_config:
        raise ValueError("训练、数据或搜索窗口配置与源实验不同，不能只重排")
    preflight = read(source / "preflight.json")
    expected_source_hashes = read(source / "evidence_manifest.json")["files_sha256"]
    source_files = ["config.snapshot.json", "preflight.json", "summary.json", "frozen_parameters.json"]
    settings = config["symmetric_search"]
    stages = []
    all_rows = []
    reference_split = None
    best_record = None

    def stage(name: str, grid: dict) -> dict:
        nonlocal reference_split, best_record
        stage_source = source / name
        saved_config = read(stage_source / "config.snapshot.json")
        if any(saved_config["search_grid"][key] != values for key, values in grid.items()):
            raise ValueError(f"{name}的等权择优路径改变，需要补跑新窗口：{grid}")
        manifest = read(stage_source / "split_manifest.json")
        split = [manifest[f"{part}_indices"] for part in ("train", "validation", "test")]
        if list(map(len, split)) != [280, 60, 61] or len(set(sum(split, []))) != 401:
            raise ValueError("划分数量或互斥核验失败")
        if reference_split is None:
            reference_split = split
        if split != reference_split:
            raise ValueError("各阶段划分不一致")
        saved_summary = read(stage_source / "summary.json")
        if saved_summary["test_evaluated"] or saved_config["evaluate_test_after_search"]:
            raise ValueError("源候选阶段评价过Test")
        rows = read(stage_source / "candidates.json")["candidates"]
        expected_count = 1
        for values in grid.values():
            expected_count *= len(values)
        if len(rows) != expected_count or len({tuple(row[key] for key in KEYS) for row in rows}) != expected_count:
            raise ValueError("源候选数量或唯一性与窗口不符")
        for row in rows:
            for key, axis in zip(KEYS, ("band_counts", "redundancy_weights", "overlap_values_c", "routing_width_values_c")):
                if row[key] not in grid[axis]:
                    raise ValueError("源候选参数不在等权搜索窗口内")
            row["selection_metric"] = METRIC
            row["validation_selection_score_c"] = validation_score(row, METRIC)
            row["validation_rmse_feasible"] = row["validation_selection_score_c"] <= config["validation_objective"]["maximum_rmse_c"]
        ranked = rank_candidates(rows, METRIC, config["validation_objective"]["maximum_rmse_c"])
        if not ranked:
            raise ValueError("没有成功收敛的候选")
        best_record = ranked[0]
        parameters = {key: best_record[key] for key in KEYS}
        result = {
            "status": "search_complete", "selection_metric": METRIC,
            "selection_objective": selection_objective(METRIC),
            "test_evaluated": False, "test_used_for_hyperparameter_selection": False,
            "candidate_count": len(rows), "best_hyperparameters": parameters,
            "best_validation_selection_score_c": validation_score(best_record, METRIC),
            "best_validation_unweighted_rmse_c": validation_score(best_record, METRIC),
            "best_validation_rmse_c": best_record["validation_rmse_c"],
            "best_validation_relative_deviation": best_record["validation_relative_deviation"],
            "selected_bands": best_record["selection"],
            "execution": "仅用保存的Validation候选重排，未新增训练或Test评价",
        }
        snapshot = copy.deepcopy(saved_config)
        snapshot["validation_objective"] = config["validation_objective"]
        snapshot["experiment_name"] = NAME + "_" + name
        for filename, payload in (("config.snapshot.json", snapshot), ("split_manifest.json", manifest),
                                  ("candidates.json", {"candidates": rows}), ("summary.json", result)):
            write(output / name / filename, payload)
            source_files.append(f"{name}/{filename}")
        stages.append({"name": name, "candidate_count": len(rows), "best": parameters,
                       "best_validation_unweighted_rmse_c": validation_score(best_record, METRIC),
                       "best_validation_adaptive_rmse_c": best_record["validation_rmse_c"],
                       "same_best_as_weighted_search": parameters == saved_summary["best_hyperparameters"]})
        all_rows.extend({**row, "stage": name} for row in rows)
        return parameters

    center = settings["initial_center"]
    center = stage("stage1_k_screen", {
        "band_counts": settings["initial_k_values"], "redundancy_weights": [center["redundancy_weight"]],
        "overlap_values_c": [center["overlap_c"]], "routing_width_values_c": [center["routing_width_c"]],
    })
    maximum = preflight["quality_control"]["retained_band_count"]
    center = stage("stage2_symmetric_k_lambda", {
        "band_counts": symmetric_k_values(center["band_count"], settings["k_step"], settings["k_half_steps"], maximum),
        "redundancy_weights": symmetric_float_values(center["redundancy_weight"], settings["lambda_step"], settings["lambda_half_steps"]),
        "overlap_values_c": [center["overlap_c"]], "routing_width_values_c": [center["routing_width_c"]],
    })
    center = stage("stage3_symmetric_overlap_routing", {
        "band_counts": [center["band_count"]], "redundancy_weights": [center["redundancy_weight"]],
        "overlap_values_c": symmetric_float_values(center["overlap_c"], settings["temperature_step_c"], settings["temperature_half_steps"]),
        "routing_width_values_c": symmetric_float_values(center["routing_width_c"], settings["temperature_step_c"], settings["temperature_half_steps"]),
    })
    window = symmetric_k_values(center["band_count"], settings["k_step"], settings["k_half_steps"], maximum)
    confirmed = stage("stage4_k_confirmation_01", {
        "band_counts": window, "redundancy_weights": [center["redundancy_weight"]],
        "overlap_values_c": [center["overlap_c"]], "routing_width_values_c": [center["routing_width_c"]],
    })
    if confirmed != center:
        raise ValueError("等权确认改变最优K，必须追加新中心的对称候选")
    source_hashes = {}
    for filename in source_files:
        actual = sha(source / filename)
        if actual != expected_source_hashes[filename]:
            raise ValueError(f"源证据哈希不一致：{filename}")
        source_hashes[filename] = actual
    selected = best_record["selection"]
    if not selected["success"] or len(set(selected["selected_band_indices"])) != center["band_count"]:
        raise ValueError("最佳候选选带收敛或波段数量不合法")
    frozen = {
        "status": "frozen_from_validation_only", "selection_metric": METRIC,
        "test_evaluated": False, "background_subtracted": True, "parameters": center,
        "best_validation_selection_score_c": validation_score(best_record, METRIC),
        "selected_band_indices": selected["selected_band_indices"],
        "selected_wavelengths_nm": selected["selected_wavelengths_nm"],
        "data_path": config["data_path"],
        "k_symmetry": {"center": center["band_count"], "values": window, "step": settings["k_step"]},
    }
    old_frozen = read(source / "frozen_parameters.json")
    result = {
        "status": "search_complete", "selection_metric": METRIC, "selection_objective": selection_objective(METRIC),
        "test_evaluated": False, "test_used_for_hyperparameter_selection": False,
        "historical_test_seen": True, "execution": "validation_candidate_reanalysis",
        "new_training_runs": 0, "reused_candidate_evaluations": len(all_rows),
        "unique_parameter_combinations": len({tuple(row[key] for key in KEYS) for row in all_rows}),
        "best_hyperparameters": center,
        "best_validation_selection_score_c": validation_score(best_record, METRIC),
        "best_validation_unweighted_rmse_c": validation_score(best_record, METRIC),
        "best_validation_adaptive_rmse_c": best_record["validation_rmse_c"],
        "best_validation_relative_deviation": best_record["validation_relative_deviation"],
        "same_parameters_and_bands_as_weighted_search": center == old_frozen["parameters"] and selected["selected_band_indices"] == old_frozen["selected_band_indices"],
        "stages": stages, "k_symmetry": frozen["k_symmetry"],
        "training_adaptive_weighting_enabled": config["adaptive_weighting"]["enabled"],
        "source_data_sha256": preflight["data_sha256"],
        "scope": "超参按等权Validation择优；原选带内部检查点规则及自适应加权训练保持原实验定义",
    }
    write(output / "config.snapshot.json", config)
    write(output / "frozen_parameters.json", frozen)
    write(output / "summary.json", result)
    write(output / "all_candidates.json", {"candidates": all_rows})
    write(output / "source_provenance.json", {
        "source_evidence": source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else str(source),
        "files_sha256": source_hashes, "source_data_sha256": preflight["data_sha256"],
        "source_training_code_sha256": preflight["runtime_sources_sha256"],
        "reanalysis_code_sha256": {name: sha(Path(__file__).parent / name) for name in (
            "reselect_background_unweighted.py", "hyperparameter_objectives.py", "run_background_symmetric_search.py",
        )},
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/results/swir_background_symmetric_search_20261013")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/results" / NAME)
    parser.add_argument("--config", type=Path, default=ROOT / "optimization_experiment/configs" / f"{NAME}.json")
    args = parser.parse_args()
    print(json.dumps(run(args.source.resolve(), args.output.resolve(), args.config.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
