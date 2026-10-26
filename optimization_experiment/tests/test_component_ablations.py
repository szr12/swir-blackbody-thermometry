from pathlib import Path
import json
import os
import sys

import numpy as np


ROOT = Path(os.environ.get("WORK2_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "main_experiment/src"))
sys.path.insert(0, str(ROOT / "optimization_experiment"))
if os.environ.get("WORK2_ABLATION_RUNTIME"):
    sys.path.insert(0, os.environ["WORK2_ABLATION_RUNTIME"])

from run_component_ablations import aggregate, control_bands, hard_region_ids, metrics, validate_pair, audit_split, validate_split_config, fit_ensemble, evaluate, fixed_parameters, variant_label


def test_control_bands_obey_common_pool_budget_and_longwave_constraint():
    """选带对照保持相同候选池、预算和长波约束，不读取温度标签。"""
    wavelengths = np.linspace(850.0, 2500.0, 264)
    available = np.delete(np.arange(264), [1, 7, 13])
    for method in ("uniform_bands", "random_bands"):
        first = control_bands(available, wavelengths, 20, 4242, method)
        assert np.array_equal(first, control_bands(available, wavelengths, 20, 4242, method))
        assert len(np.unique(first)) == 20
        assert set(first).issubset(set(available))
        assert np.any(wavelengths[first] >= 1800.0)
    assert not np.array_equal(control_bands(available, wavelengths, 20, 4242, "random_bands"),
                              control_bands(available, wavelengths, 20, 4243, "random_bands"))


def test_hard_routing_uses_prediction_boundaries_and_clips_outside_range():
    """精确边界、越界粗预测均可路由，函数不接收真实温度。"""
    ids = hard_region_ids(np.array([-100.0, 149.99, 150.0, 250.0, 350.0, 1000.0]),
                          np.array([50.0, 150.0, 250.0, 350.0, 450.0]))
    assert np.array_equal(ids, [0, 0, 1, 2, 3, 3])


def test_equal_metrics_include_low_temperature_and_prespecified_boundaries():
    """整体指标与各温区独立核对，边界范围在评价前固定。"""
    y = np.array([60.0, 140.0, 200.0, 240.0, 300.0, 360.0, 400.0])
    prediction = y + np.array([2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0])
    result = metrics(y, prediction, np.array([50.0, 150.0, 250.0, 350.0, 450.0]))
    assert np.isclose(result["rmse"], np.sqrt(np.mean((y - prediction) ** 2)))
    assert np.isclose(result["mard_percent"], np.mean(np.abs(y - prediction) / y) * 100)
    assert [r["n"] for r in result["segments"]] == [2, 2, 1, 2]
    assert result["boundary_15c"]["n"] == 3


def test_nonconverged_results_are_excluded_from_success_means(tmp_path):
    """失败拟合保留诊断数值，但不得进入成功均值或配对改善结果。"""
    for seed, converged, rmse in ((4242, True, 6.0), (4243, False, 1.0)):
        directory = tmp_path / f"seed_{seed}/baseline"
        directory.mkdir(parents=True)
        evaluation = {"converged": converged, "status": "success" if converged else "diagnostic_nonconverged",
                      "metrics": {"rmse": rmse, "mae": rmse, "r2": 0.9, "mard_percent": 3.0}}
        record = {"seed": seed, "variant": "baseline", "status": evaluation["status"],
                  "selection_success": True, "test": evaluation,
                  "derived_test": {"global_only": evaluation, "hard_routing": evaluation}}
        (directory / "summary.json").write_text(json.dumps(record), encoding="utf-8")
    result = aggregate(tmp_path, [4242, 4243])
    row = result["rows"][0]
    assert row["successful"] == 1 and row["diagnostic_or_failed"] == 1
    assert row["rmse_mean"] == 6.0 and row["rmse_std"] is None
    assert len(row["paired_deltas"]) == 1


def test_actual_background_pair_is_same_frame_subtraction():
    """对真实401×264输入执行同帧相减和样本身份一致性核验。"""
    result = validate_pair(
        ROOT / "preprocessing_validation/preprocessed_swir_background_corrected_v1/swir_roi_background_corrected.npz",
        ROOT / "preprocessing_validation/preprocessed_swir_no_background/swir_roi_no_background.npz")
    assert result["shape"] == [401, 264]
    assert result["same_frame_subtraction_max_error_dn"] == 0.0


def test_ordered_split_is_disjoint_and_test_outside_final_fit():
    """打乱输入顺序也按温度划分；最高温Test不进入最终拟合。"""
    from traditional_utils import make_configured_split
    y = np.random.default_rng(9).permutation(np.arange(50.0, 451.0))
    split = {"strategy": "temperature_ordered_70_15_15", "seed": 42, "train_fraction": 0.70, "validation_fraction": 0.15}
    assert validate_split_config(split)
    train, validation, test = make_configured_split(y, np.array([50.,150.,250.,350.,450.]), split)
    audit = audit_split(y, train, validation, test, True)
    assert (len(train),len(validation),len(test)) == (280,60,61)
    assert audit["final_fit_temperature_range_c"] == [50.0,389.0]
    assert audit["test_temperature_range_c"] == [390.0,450.0]
    try:
        audit_split(y, train, validation, np.append(test[:-1], train[0]), True)
    except ValueError:
        pass
    else:
        raise AssertionError("重复或泄漏的Test索引应被拒绝")


def test_high_temperature_metrics_do_not_invent_low_temperature_or_boundary_data():
    """最高温Test只报告实际覆盖温区，不虚构低温及内部边界指标。"""
    y = np.arange(390.0,451.0)
    result = metrics(y,y+2.0,np.array([50.,150.,250.,350.,450.]))
    assert len(result["segments"]) == 1
    assert result["segments"][0]["lower_c"] == 350.0
    assert result["segments"][0]["n"] == 61
    assert result["boundary_15c"] is None
    assert result["rmse"] == 2.0


def test_no_overlap_missing_train_expert_is_explicit_and_final_fit_is_available(tmp_path):
    """Train不补入高温样本；仅按规定合并Validation后训练第四专家。"""
    config_path = Path(os.environ.get("WORK2_ABLATION_CONFIG", ROOT/'optimization_experiment/configs/component_ablation_k20_20261012.json'))
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config["model"].update({"max_iter": 1, "evaluation_interval": 1, "required_consecutive_successes": 1,
                            "hidden_units": [4], "device": "cpu", "target_training_rmse_c": 1000.0})
    config["adaptive_weighting"]["enabled"] = False
    config["search_grid"]["overlap_values_c"] = [0.0]
    y = np.arange(50.0,390.0)
    X = np.column_stack([y+20.0, y*2.0+40.0, y*3.0+60.0])
    wave = np.array([900.,1500.,2400.])
    ensemble, diagnostic = fit_ensemble(X[:280],y[:280],wave,config,7,tmp_path/'train_only',lambda stage:None,True)
    assert ensemble.experts_[3] is None
    assert diagnostic[4]["status"] == "unavailable_training_samples"
    assert diagnostic[4]["n_fit"] == 0
    assert all(r.get("fit_temperature_range_c",[0,0])[1] <= 329 for r in diagnostic)
    result = evaluate(ensemble,X[280:],y[280:],np.arange(280,340),np.array([50.,150.,250.,350.,450.]),'validation',tmp_path,diagnostic)
    assert result["regional"]["metrics"] is None
    assert result["regional"]["status"] == "unavailable_training_samples"
    assert result["global_only"]["metrics"]["n"] == 60
    final, final_diagnostic = fit_ensemble(X,y,wave,config,100007,tmp_path/'final',lambda stage:None)
    assert final.routing_width_c == float(config["search_grid"]["routing_width_values_c"][0])
    assert all(model is not None for model in final.experts_)
    assert all(row["success"] for row in final_diagnostic)
    assert final_diagnostic[4]["fit_temperature_range_c"] == [350.,389.]


def test_new_frozen_parameters_propagate_to_selector_and_labels():
    """最新K、冗余系数和连续门控配置必须到达模型，拒绝意外网格搜索。"""
    from traditional_utils import make_selector
    config_path = Path(os.environ.get("WORK2_ABLATION_CONFIG", ROOT/'optimization_experiment/configs/component_ablation_refrozen_k12_random_20261013.json'))
    config = json.loads(config_path.read_text(encoding='utf-8'))
    parameters = fixed_parameters(config)
    assert parameters == {"band_count":12,"redundancy_weight":0.04,"overlap_c":55.0,"routing_width_c":15.0}
    selector=make_selector(config['model'],12,0.04,4242,config['adaptive_weighting'])
    assert selector.band_count==12 and selector.redundancy_weight==0.04
    assert selector.freeze_gate_after is None
    assert variant_label('uniform_bands',12)=='等间隔12波段'
    assert variant_label('random_bands',12)=='随机12波段'
    config['search_grid']['band_counts']=[12,14]
    try:
        fixed_parameters(config)
    except ValueError:
        pass
    else:
        raise AssertionError('未冻结的网格配置必须拒绝')
