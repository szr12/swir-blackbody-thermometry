from __future__ import annotations

import math


METRICS = ("adaptive_rmse", "unweighted_rmse")


def configured_metric(config: dict) -> str:
    """读取显式选参指标；历史配置默认沿用原加权规则。"""
    metric = config.get("validation_objective", {}).get("metric", "adaptive_rmse")
    if metric not in METRICS:
        raise ValueError(f"不支持的超参择优指标：{metric}")
    return metric


def validation_score(row: dict, metric: str) -> float:
    """等权选参必须读取等权指标，禁止误用历史加权字段。"""
    if metric == "unweighted_rmse":
        return float(row["validation_unweighted_metrics"]["rmse"])
    if metric == "adaptive_rmse":
        return float(row["validation_rmse_c"])
    raise ValueError(f"不支持的超参择优指标：{metric}")


def selection_objective(metric: str) -> str:
    if metric not in METRICS:
        raise ValueError(f"不支持的超参择优指标：{metric}")
    return f"minimum_validation_{metric}_then_relative_deviation_balance"


def rank_candidates(candidates: list[dict], metric: str = "adaptive_rmse",
                    maximum_rmse_c: float | None = None) -> list[dict]:
    """按指定Validation指标择优，门槛与排序使用同一指标。"""
    if metric not in METRICS:
        raise ValueError(f"不支持的超参择优指标：{metric}")
    successful = [row for row in candidates if row["success"] and math.isfinite(validation_score(row, metric))]
    if maximum_rmse_c is not None:
        feasible = [row for row in successful if validation_score(row, metric) <= maximum_rmse_c]
    elif metric == "adaptive_rmse":
        feasible = [row for row in successful if row.get("validation_rmse_feasible", True)]
    else:
        # 旧字段的可行性来自加权指标，不得限制等权重排。
        feasible = successful
    return sorted(feasible or successful, key=lambda row: (
        validation_score(row, metric),
        float(row.get("validation_balanced_score", row["validation_relative_deviation"])),
        int(row["band_count"]), float(row["redundancy_weight"]),
        float(row["overlap_c"]), float(row["routing_width_c"]),
    ))
