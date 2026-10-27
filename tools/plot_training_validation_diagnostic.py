"""绘制原生区域MLP的逐种子学习曲线，不平滑、不补齐已停止的轨迹。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def main(args):
    sys.path.insert(0, str(args.qa_scripts.resolve()))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    source, out = args.results.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (source / "learning_curves.csv").open(encoding="utf-8") as handle:
        curves = list(csv.DictReader(handle))
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    assert summary["scientific_fits"] == 25 and not summary["test_evaluated"]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    train_color, val_color = "#205A89", "#B25D19"
    # 固定为180毫米乘125毫米，使用英寸字面值便于自动审计识别。
    fig, axes = plt.subplots(2, 3, figsize=(7.08661417, 4.92125984))
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.13, top=0.86, wspace=0.44, hspace=0.66)
    roles = ["global", "expert_0", "expert_1", "expert_2", "expert_3"]
    titles = ["Global model", "Expert 1", "Expert 2", "Expert 3", "Expert 4", "Fused predictions"]
    for index, (ax, title) in enumerate(zip(axes.flat, titles, strict=True)):
        ax.text(-0.19, 1.08, chr(97 + index), transform=ax.transAxes, weight="bold", fontsize=8,
                ha="left", va="bottom")
        ax.set_title(title, loc="left", pad=8)
        ax.set_ylim(0, 35)
        ax.set_yticks([0, 10, 20, 30])
        ax.set_ylabel("RMSE (°C)")
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
        ax.set_axisbelow(True)
    for ax, role in zip(axes.flat, roles):
        for seed in range(4242, 4247):
            rows = [v for v in curves if int(v["seed"]) == seed and v["role"] == role]
            step = np.asarray([float(v["iteration"]) for v in rows])
            train = np.asarray([float(v["training_rmse_c"]) for v in rows])
            val = np.asarray([float(v["validation_rmse_c"]) for v in rows])
            assert len(step) and np.all(np.diff(step) > 0)
            assert np.isfinite(train).all() and np.isfinite(val).all()
            assert np.max(np.r_[train, val]) < 35
            ax.plot(step, train, color=train_color, linewidth=0.8, alpha=0.6)
            ax.plot(step, val, color=val_color, linestyle="--", linewidth=0.8, alpha=0.6)
            meta = next(v for v in summary["models"] if v["seed"] == seed and v["role"] == role)
            restored = np.flatnonzero(step == meta["restored_iteration"])
            assert len(restored) == 1
            k = int(restored[0])
            for values, color in [(train, train_color), (val, val_color)]:
                ax.plot(step[k], values[k], marker="o", markersize=2.7,
                        markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.65)
        ax.set_xlabel("Iteration")
        ax.set_xlim(left=0)
    ax = axes.flat[-1]
    rows = summary["ensembles"]
    for jitter, row in zip(np.linspace(-0.07, 0.07, len(rows)), rows, strict=True):
        values = [row["training_rmse_c"], row["validation_rmse_c"]]
        positions = np.asarray([0, 1]) + jitter
        ax.plot(positions, values, color="#A6A6A6", linewidth=0.65, zorder=1)
        ax.scatter(positions, values, c=[train_color, val_color], s=16,
                   edgecolors="white", linewidths=0.45, zorder=2)
    ax.set_xticks([0, 1], ["Train", "Validation"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_xlabel("Returned ensemble")
    handles = [Line2D([], [], color=train_color, lw=1.2, label="Train"),
               Line2D([], [], color=val_color, lw=1.2, ls="--", label="Validation"),
               Line2D([], [], color="#666666", marker="o", mfc="white", markersize=3,
                      linestyle="none", label="Returned checkpoint")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.99), ncol=3,
               frameon=False, fontsize=7, handlelength=2, columnspacing=1.7)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=out / "learning_curves.alignment.json",
                                      overlay_svg=out / "learning_curves.alignment.svg",
                                      require_panel_labels=True, tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(out / "learning_curves.pdf", facecolor="white")
    fig.savefig(out / "learning_curves.svg", facecolor="white")
    fig.savefig(out / "learning_curves.png", dpi=600, facecolor="white")
    plt.close(fig)
    contract = {"question": "Do native regional MLP fits show a training-validation gap under the existing stopping rule?",
                "panels": {"a-e": "five individual seed trajectories per native model; raw values, no smoothing or extrapolation",
                           "f": "five paired endpoint RMSE values for the fused ensemble"},
                "n": "five algorithm seeds on one fixed acquisition-batch split; no independent acquisition repeats",
                "uncertainty": "individual trajectories and points; no aggregate confidence intervals or error bars",
                "colors": {"train": train_color, "validation": val_color},
                "test_evaluated": False, "source_data": "learning_curves.csv and ensemble_summary.csv",
                "axis_limits_checked_against_all_values": True}
    (out / "figure_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Six-panel learning-curve figure exported with alignment checks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qa-scripts", type=Path, required=True)
    main(parser.parse_args())
