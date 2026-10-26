from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "main_experiment" / "src"))

from blackbody_hsic.selector_runtime import load_or_compute_mi  # noqa: E402
from blackbody_hsic.selector import HsicMrnrSelector  # noqa: E402
from regional_colorimetry import RegionalColorimetricEnsemble, soft_region_weights  # noqa: E402
from hyperparameter_objectives import (  # noqa: E402
    configured_metric, rank_candidates, selection_objective, validation_score,
)
from traditional_utils import (  # noqa: E402
    fold_quality_mask,
    json_ready as _json_ready,
    load_data as _load_data,
    make_configured_split,
    make_selector as _selector,
    mean_absolute_relative_deviation,
    model_parameters as _model_parameters,
    observed_segmented_metrics,
    regression_metrics,
    segmented_metrics,
    temperature_bin_ids,
    write_json as _write_json,
)


def _build_hsic_prefilter_pool(
    config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    wavelengths_nm: np.ndarray,
    original_band_indices: np.ndarray,
    mi_matrix: np.ndarray,
    band_count: int,
) -> tuple[np.ndarray, dict]:
    """只用Train执行HSIC-mRNR预筛选，并保留后续监督选带的候选池。"""
    prefilter = dict(config.get("hsic_prefilter", {}))
    multiplier = float(prefilter.get("candidate_multiplier", 3.0))
    if not np.isfinite(multiplier) or multiplier < 1.0:
        raise ValueError("HSIC候选池倍数必须大于或等于1")
    candidate_count = min(
        X_train.shape[1],
        max(int(band_count), int(np.ceil(multiplier * band_count))),
    )
    selector = HsicMrnrSelector(
        band_count=candidate_count,
        lam=float(prefilter.get("redundancy_weight", 0.06)),
        eta=float(prefilter.get("eta", 0.02)),
        max_iter=int(prefilter.get("max_iter", 300)),
        min_iter=int(prefilter.get("min_iter", 10)),
        tolerance=float(prefilter.get("tolerance", 1e-8)),
        mi_neighbors=int(config["search_grid"]["mi_neighbors"]),
        device=str(config["model"]["device"]),
        seed=int(config["model"]["seed"]),
    )
    result = selector.fit(X_train, y_train, mi_matrix=mi_matrix)
    weights = np.asarray(result.weights, dtype=np.float64)
    pool = np.asarray(result.selected_indices, dtype=np.int64)
    required_ranges = tuple(
        tuple(float(bound) for bound in interval)
        for interval in config["model"].get("required_wavelength_ranges_nm", [])
    )
    added_required: list[int] = []
    protected: set[int] = set()
    for lower, upper in required_ranges:
        candidates = np.flatnonzero(
            (wavelengths_nm >= lower) & (wavelengths_nm <= upper)
        )
        if candidates.size == 0:
            raise RuntimeError(f"HSIC预筛选的必选波段区间没有候选: {lower}-{upper} nm")
        in_pool = [int(index) for index in pool if index in set(candidates.tolist())]
        if in_pool:
            protected.add(in_pool[0])
            continue
        chosen = int(candidates[np.argmax(weights[candidates])])
        replaceable = [int(index) for index in pool if int(index) not in protected]
        if not replaceable:
            raise RuntimeError("HSIC预筛选无法为必选波段区间腾出候选位置")
        replace = min(replaceable, key=lambda index: weights[index])
        pool[pool == replace] = chosen
        protected.add(chosen)
        added_required.append(chosen)
    pool = np.unique(pool)
    if pool.size != candidate_count:
        raise RuntimeError("HSIC预筛选候选池出现重复波段")
    record = {
        "enabled": True,
        "source": "HSIC-mRNR",
        "candidate_multiplier": multiplier,
        "candidate_count": int(pool.size),
        "hsic_redundancy_weight": float(prefilter.get("redundancy_weight", 0.06)),
        "iterations": int(result.iterations),
        "objective": float(result.objective),
        "sigma_x": float(result.sigma_x),
        "hsic_band_weights": weights,
        "uses_rbf_kernel_and_centering_h": True,
        "uses_training_labels_only": True,
        "required_range_added_positions": added_required,
        "candidate_band_indices": original_band_indices[pool],
        "candidate_wavelengths_nm": wavelengths_nm[pool],
    }
    return pool, record


def _prepare_mi_cache(
    config: dict,
    project_root: Path,
    run_dir: Path,
    training_features: np.ndarray,
    neighbors: int,
    seed: int,
) -> np.ndarray:
    """只复用特征值、样本顺序、邻居数与种子完全一致的冗余矩阵。"""
    destination = run_dir / "cache" / "training_mutual_information.npy"
    metadata_path = destination.with_suffix(".json")
    features = np.ascontiguousarray(training_features, dtype=np.float64)
    expected = {
        "training_features_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
        "shape": list(features.shape), "neighbors": int(neighbors), "seed": int(seed),
    }
    if destination.exists():
        if not metadata_path.exists() or json.loads(metadata_path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("当前目录互信息缓存缺少匹配的Train特征指纹，请使用新的实验目录")
    source_value = config.get("mi_cache_source")
    if source_value and not destination.exists():
        source = (project_root / str(source_value)).resolve()
        source_metadata = source.with_suffix(".json")
        if source.exists() and source_metadata.exists() and json.loads(source_metadata.read_text(encoding="utf-8")) == expected:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            _write_json(metadata_path, expected)
    print("读取或计算当前Train的互信息矩阵", flush=True)
    matrix = load_or_compute_mi(destination, training_features, neighbors, seed)
    _write_json(metadata_path, expected)
    return matrix


def run(config: dict, project_root: Path, run_dir: Path) -> dict:
    metric = configured_metric(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.snapshot.json", config)
    data = _load_data(
        (project_root / config["data_path"]).resolve(),
        config["feature_key"],
        config["uncertainty_key"],
    )
    X = data[config["feature_key"]]
    y = data["y_temperature_c"]
    wavelengths = data["wavelengths_nm"]
    boundaries = np.asarray(config["temperature_boundaries_c"], dtype=np.float64)
    split_config = dict(config["split"])
    split_seed = int(split_config.get("seed", 42))
    train_index, validation_index, test_index = make_configured_split(
        y, boundaries, split_config,
    )
    _write_json(run_dir / "split_manifest.json", {
        "strategy": split_config["strategy"],
        "seed": split_seed,
        "train_indices": train_index,
        "validation_indices": validation_index,
        "test_indices": test_index,
        "train_temperature_range_c": [float(y[train_index].min()), float(y[train_index].max())],
        "validation_temperature_range_c": [
            float(y[validation_index].min()), float(y[validation_index].max()),
        ],
        "test_temperature_range_c": [float(y[test_index].min()), float(y[test_index].max())],
    })

    quality_mask, quality_diagnostics = fold_quality_mask(
        X[train_index],
        data[config["uncertainty_key"]][train_index],
        data["background_mad"][train_index],
        data["bad_pixel_fraction_by_sample_band"][train_index],
        data["saturation_fraction_by_sample_band"][train_index],
        wavelengths,
        config["fold_quality_control"],
    )
    available = np.flatnonzero(quality_mask)
    grid = config["search_grid"]
    mi = _prepare_mi_cache(
        config,
        project_root,
        run_dir,
        X[train_index][:, quality_mask],
        int(grid["mi_neighbors"]),
        split_seed,
    )
    candidates: list[dict] = []
    selection_cache: dict[tuple[int, float], tuple[np.ndarray, dict]] = {}
    prefilter_cache: dict[int, tuple[np.ndarray, dict]] = {}
    hsic_prefilter_enabled = bool(config.get("hsic_prefilter", {}).get("enabled", False))
    if hsic_prefilter_enabled:
        for band_count in grid["band_counts"]:
            _write_json(run_dir / "progress.json", {
                "status": "hsic_prefiltering",
                "completed_band_counts": len(prefilter_cache),
                "total_band_counts": len(grid["band_counts"]),
                "band_count": band_count,
                "test_evaluated": False,
            })
            pool, prefilter_record = _build_hsic_prefilter_pool(
                config,
                X[train_index][:, quality_mask],
                y[train_index],
                wavelengths[quality_mask],
                data["original_band_indices"][quality_mask],
                mi,
                int(band_count),
            )
            prefilter_cache[int(band_count)] = (pool, prefilter_record)
            _write_json(run_dir / "prefilter" / f"k{band_count}.json", prefilter_record)
            print(
                f"HSIC-mRNR预筛选完成 K={band_count}，候选池={pool.size}个波段",
                flush=True,
            )
    started = time.perf_counter()
    trial_id = 0
    total = len(grid["band_counts"]) * len(grid["redundancy_weights"]) * len(grid["overlap_values_c"]) * len(grid["routing_width_values_c"])
    print(f"开始搜索：Train={train_index.size}，Validation={validation_index.size}，候选={total}，Test不参与选择", flush=True)
    for band_count in grid["band_counts"]:
        for redundancy_weight in grid["redundancy_weights"]:
            _write_json(run_dir / "progress.json", {
                "status": "selecting_bands", "completed_candidates": trial_id,
                "total_candidates": total, "band_count": band_count,
                "redundancy_weight": redundancy_weight, "test_evaluated": False,
            })
            print(f"开始选带 K={band_count} lambda={redundancy_weight}", flush=True)
            if hsic_prefilter_enabled:
                pool_positions, prefilter_record = prefilter_cache[int(band_count)]
            else:
                pool_positions = np.arange(available.size, dtype=np.int64)
                prefilter_record = {
                    "enabled": False,
                    "candidate_count": int(pool_positions.size),
                }
            pool_original_positions = available[pool_positions]
            pool_mi = mi[np.ix_(pool_positions, pool_positions)]
            direct_hsic = bool(config.get("hsic_direct_selection", False))
            if direct_hsic:
                # 直接选带对照不执行误差驱动门控，λ只作用于HSIC-mRNR。
                direct_config = dict(config)
                direct_config["hsic_prefilter"] = {
                    **config.get("hsic_prefilter", {}),
                    "candidate_multiplier": 1.0,
                    "redundancy_weight": float(redundancy_weight),
                }
                direct_indices, direct_record = _build_hsic_prefilter_pool(
                    direct_config, X[train_index][:, quality_mask], y[train_index],
                    wavelengths[quality_mask], data["original_band_indices"][quality_mask],
                    mi, int(band_count),
                )
                pool_original_positions = available
                prefilter_record = direct_record
                selection = SimpleNamespace(
                    selected_indices=direct_indices, success=True,
                    iterations=direct_record["iterations"], training_rmse_c=None,
                    training_adaptive_rmse_c=None, validation_rmse_c=None,
                    validation_adaptive_rmse_c=None, weights=None, trace=[],
                    stop_reason="hsic_direct_selection_completed",
                )
            else:
                selection = _selector(
                    config["model"],
                    int(band_count),
                    float(redundancy_weight),
                    int(config["model"]["seed"]),
                    adaptive_weighting=config.get("adaptive_weighting"),
                ).fit(
                    X[train_index][:, pool_original_positions],
                    y[train_index],
                    X[validation_index][:, pool_original_positions],
                    y[validation_index],
                    wavelengths[pool_original_positions],
                    pool_mi,
                )
            selected = pool_original_positions[selection.selected_indices]
            selection_record = {
                "success": bool(selection.success),
                "iterations": int(selection.iterations),
                "training_rmse_c": selection.training_rmse_c,
                "training_adaptive_rmse_c": selection.training_adaptive_rmse_c,
                "validation_rmse_c": selection.validation_rmse_c,
                "validation_adaptive_rmse_c": selection.validation_adaptive_rmse_c,
                "selection_method": "hsic_direct" if direct_hsic else "error_driven",
                "selected_band_indices": data["original_band_indices"][selected],
                "selected_wavelengths_nm": wavelengths[selected],
                "hsic_prefilter": prefilter_record,
            }
            selection_cache[(int(band_count), float(redundancy_weight))] = (selected, selection_record)
            _write_json(run_dir / "selections" / f"k{band_count}_lambda{redundancy_weight}.json", {
                **selection_record, "trace": selection.trace,
                "weights": selection.weights, "stop_reason": selection.stop_reason,
            })
            print(
                f"选带完成：{selection.iterations}轮，成功={selection.success}，"
                f"训练RMSE={selection.training_rmse_c if selection.training_rmse_c is not None else '不适用'}",
                flush=True,
            )
            for overlap_c in grid["overlap_values_c"]:
                _write_json(run_dir / "progress.json", {
                    "status": "training_regional_models", "completed_candidates": trial_id,
                    "total_candidates": total, "band_count": band_count,
                    "redundancy_weight": redundancy_weight, "overlap_c": overlap_c,
                    "test_evaluated": False,
                })
                ensemble = RegionalColorimetricEnsemble(
                    boundaries_c=boundaries,
                    overlap_c=float(overlap_c),
                    routing_width_c=float(grid["routing_width_values_c"][0]),
                    model_parameters=_model_parameters(config),
                    seed=int(config["model"]["seed"]),
                ).fit(X[train_index][:, selected], y[train_index], wavelengths[selected])
                coarse, expert_predictions, _ = ensemble.predict_components(X[validation_index][:, selected])
                for routing_width_c in grid["routing_width_values_c"]:
                    weights = soft_region_weights(coarse, boundaries, float(routing_width_c))
                    prediction = np.sum(expert_predictions * weights, axis=1)
                    validation_bins = temperature_bin_ids(y[validation_index], boundaries)
                    segment_relative_deviations = [
                        mean_absolute_relative_deviation(
                            y[validation_index][validation_bins == region_id],
                            prediction[validation_bins == region_id],
                        )
                        for region_id in range(boundaries.size - 1)
                        if np.any(validation_bins == region_id)
                    ]
                    relative_deviation = mean_absolute_relative_deviation(y[validation_index], prediction)
                    adaptive_validation_weights = ensemble.global_model_.attention_weights(
                        X[validation_index][:, selected]
                    )
                    validation_rmse = regression_metrics(
                        y[validation_index], prediction, adaptive_validation_weights,
                    )["rmse"]
                    objective = config.get("validation_objective", {})
                    worst_weight = float(objective.get("worst_segment_relative_deviation_weight", 0.0))
                    rmse_limit = float(objective.get("maximum_rmse_c", float("inf")))
                    row = {
                        "trial_id": trial_id,
                        "band_count": int(band_count),
                        "redundancy_weight": float(redundancy_weight),
                        "overlap_c": float(overlap_c),
                        "routing_width_c": float(routing_width_c),
                        "success": bool(selection.success and ensemble.success_),
                        "validation_relative_deviation": relative_deviation,
                        "validation_segment_relative_deviations": segment_relative_deviations,
                        "validation_worst_segment_relative_deviation": max(segment_relative_deviations),
                        "validation_balanced_score": relative_deviation + worst_weight * max(segment_relative_deviations),
                        "validation_rmse_c": validation_rmse,
                        "validation_unweighted_metrics": regression_metrics(y[validation_index], prediction),
                        "validation_segment_metrics": (
                            observed_segmented_metrics(y[validation_index], prediction, boundaries)
                            if split_config["strategy"] == "temperature_ordered_70_15_15"
                            else segmented_metrics(y[validation_index], prediction, boundaries)
                        ),
                        "validation_rmse_feasible": validation_rmse <= rmse_limit,
                        "selection": selection_record,
                        "regional_training": [diagnostic.__dict__ for diagnostic in ensemble.diagnostics_],
                    }
                    row["selection_metric"] = metric
                    row["validation_selection_score_c"] = validation_score(row, metric)
                    row["validation_rmse_feasible"] = row["validation_selection_score_c"] <= rmse_limit
                    candidates.append(_json_ready(row))
                    trial_id += 1
                    print(
                        f"候选{trial_id}/{len(grid['band_counts']) * len(grid['redundancy_weights']) * len(grid['overlap_values_c']) * len(grid['routing_width_values_c'])} "
                        f"K={band_count} lambda={redundancy_weight} overlap={overlap_c} width={routing_width_c} "
                        f"验证相对偏差={row['validation_relative_deviation'] * 100:.3f}% "
                        f"择优指标({metric})={row['validation_selection_score_c']:.3f}℃",
                        flush=True,
                    )
                    _write_json(run_dir / "candidates.json", {"candidates": candidates})
                    _write_json(run_dir / "progress.json", {
                        "status": "candidate_complete", "completed_candidates": trial_id,
                        "total_candidates": total, "test_evaluated": False,
                    })

    ranked = rank_candidates(candidates, metric, config.get("validation_objective", {}).get("maximum_rmse_c"))
    if not ranked:
        _write_json(run_dir / "progress.json", {
            "status": "failed_no_converged_candidate", "completed_candidates": trial_id,
            "total_candidates": total, "test_evaluated": False,
        })
        raise RuntimeError("没有成功收敛的超参候选")
    best = ranked[0]
    selected, selection_record = selection_cache[(best["band_count"], best["redundancy_weight"])]
    if not bool(config.get("evaluate_test_after_search", True)):
        summary = {
            "status": "search_complete",
            "selection_objective": selection_objective(metric),
            "selection_metric": metric,
            "best_validation_selection_score_c": validation_score(best, metric),
            "best_validation_unweighted_rmse_c": best["validation_unweighted_metrics"]["rmse"],
            "test_evaluated": False,
            "test_used_for_hyperparameter_selection": False,
            "candidate_count": len(candidates),
            "best_hyperparameters": {
                "band_count": best["band_count"],
                "redundancy_weight": best["redundancy_weight"],
                "overlap_c": best["overlap_c"],
                "routing_width_c": best["routing_width_c"],
            },
            "best_validation_relative_deviation": best["validation_relative_deviation"],
            "best_validation_worst_segment_relative_deviation": best["validation_worst_segment_relative_deviation"],
            "best_validation_segment_relative_deviations": best["validation_segment_relative_deviations"],
            "best_validation_balanced_score": best["validation_balanced_score"],
            "best_validation_rmse_c": best["validation_rmse_c"],
            "hsic_prefilter": config.get("hsic_prefilter", {"enabled": False}),
            "best_hsic_candidate_pool": selection_record.get("hsic_prefilter"),
            "selected_bands": selection_record,
            "quality_control": quality_diagnostics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "frozen_parameters.json", {
            "status": "frozen_from_validation_only",
            "test_evaluated": False,
            "parameters": summary["best_hyperparameters"],
            "selection_metric": metric,
            "best_validation_selection_score_c": validation_score(best, metric),
            "selected_band_indices": selection_record["selected_band_indices"],
            "selected_wavelengths_nm": selection_record["selected_wavelengths_nm"],
        })
        _write_json(run_dir / "progress.json", {
            "status": "search_complete", "completed_candidates": trial_id,
            "total_candidates": total, "test_evaluated": False,
        })
        return _json_ready(summary)
    fit_index = np.sort(np.concatenate([train_index, validation_index]))
    final = RegionalColorimetricEnsemble(
        boundaries_c=boundaries,
        overlap_c=float(best["overlap_c"]),
        routing_width_c=float(best["routing_width_c"]),
        model_parameters=_model_parameters(config),
        seed=int(config["model"]["seed"]) + 100000,
    ).fit(X[fit_index][:, selected], y[fit_index], wavelengths[selected])
    coarse, expert_predictions, weights = final.predict_components(X[test_index][:, selected])
    prediction = np.sum(expert_predictions * weights, axis=1)
    adaptive_test_weights = final.global_model_.attention_weights(X[test_index][:, selected])
    summary = {
        "status": "success" if final.success_ else "failed_to_converge",
        "selection_objective": selection_objective(metric),
        "selection_metric": metric,
        "best_validation_selection_score_c": validation_score(best, metric),
        "best_validation_unweighted_rmse_c": best["validation_unweighted_metrics"]["rmse"],
        "test_used_for_hyperparameter_selection": False,
        "candidate_count": len(candidates),
        "split_sizes": {
            "train": int(train_index.size),
            "validation": int(validation_index.size),
            "test": int(test_index.size),
        },
        "split_strategy": split_config["strategy"],
        "temperature_ranges_c": {
            "train": [float(y[train_index].min()), float(y[train_index].max())],
            "validation": [float(y[validation_index].min()), float(y[validation_index].max())],
            "test": [float(y[test_index].min()), float(y[test_index].max())],
        },
        "best_hyperparameters": {
            "band_count": best["band_count"],
            "redundancy_weight": best["redundancy_weight"],
            "overlap_c": best["overlap_c"],
            "routing_width_c": best["routing_width_c"],
        },
        "best_validation_relative_deviation": best["validation_relative_deviation"],
        "best_validation_rmse_c": best["validation_rmse_c"],
        "hsic_prefilter": config.get("hsic_prefilter", {"enabled": False}),
        "best_hsic_candidate_pool": selection_record.get("hsic_prefilter"),
        "selected_bands": selection_record,
        "quality_control": quality_diagnostics,
        "final_global_training_rmse_c": final.global_model_.training_rmse_c_,
        "final_regional_training": [row.__dict__ for row in final.diagnostics_],
        "test_temperatures_c": y[test_index],
        "coarse_predictions_c": coarse,
        "predictions_c": prediction,
        "coarse_test_metrics": regression_metrics(y[test_index], coarse),
        "regional_test_metrics": regression_metrics(y[test_index], prediction),
        "regional_adaptive_test_metrics": regression_metrics(
            y[test_index], prediction, adaptive_test_weights,
        ),
        "primary_mse": "regional_adaptive_test_metrics.mse",
        "adaptive_weight_source": "frozen_training_attention_network",
        "coarse_test_relative_deviation": mean_absolute_relative_deviation(y[test_index], coarse),
        "regional_test_relative_deviation": mean_absolute_relative_deviation(y[test_index], prediction),
        "regional_test_segment_metrics": (
            observed_segmented_metrics(y[test_index], prediction, boundaries)
            if split_config["strategy"] == "temperature_ordered_70_15_15"
            else segmented_metrics(y[test_index], prediction, boundaries)
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(run_dir / "summary.json", summary)
    return _json_ready(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir")
    arguments = parser.parse_args()
    config = json.loads(Path(arguments.config).resolve().read_text(encoding="utf-8"))
    if arguments.run_dir:
        run_dir = Path(arguments.run_dir).resolve()
    else:
        name = config["experiment_name"] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (PROJECT_ROOT / config["output_root"] / name).resolve()
    print(json.dumps(run(config, PROJECT_ROOT, run_dir), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
