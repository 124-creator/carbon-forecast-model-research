#!/usr/bin/env python3
"""Render report-only figures/tables from the three completed independent scores.

Reads aggregate/cell/CI/gate JSON and CSV receipts only. Does not import model or
training code, open prediction NPZs, or access confirmation data. All writes are
restricted to this study's reporting directory. Run with the research Python.
"""
from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import json
import math
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Keep Matplotlib's temporary font cache inside the authorized reporting scope.
_MPL_CACHE = tempfile.TemporaryDirectory(prefix="matplotlib_cache_", dir=Path(__file__).resolve().parent)
os.environ["MPLCONFIGDIR"] = _MPL_CACHE.name
atexit.register(_MPL_CACHE.cleanup)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import numpy as np
from PIL import Image


R = "primary_equal_cell_mse_ratio_to_rw"
Q = "secondary_equal_cell_crps_ratio_to_rw"
YEARS = (2021, 2022, 2023)
HORIZONS = (1, 5, 20)
BLUE = "#0072B2"
ORANGE = "#D55E00"
PURPLE = "#8C5E8F"
INK = "#243442"
GREY = "#637582"
LIGHT = "#EEF2F5"
SOURCES = ("AGGREGATE.csv", "FULL_CELLS.csv", "PAIRED_CI.csv",
           "PAIRED_CI.json", "GATES.json", "INDEPENDENT_METRICS.json")
STAGES = {
    "original_v1": ("", "原始 v1（输入解释缺陷；历史参照）", "SDMC"),
    "corrected_v1": ("calendar_correction_20261018/rerun", "纠正 v1（主 v1 比较）", "SDMC"),
    "fdmc_v2": ("scale_separated_revision", "FDMC v2（纠正数据）", "FDMC"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(row, field):
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite report field: {field}")
    return value


def close(a, b, name, errors, atol=1e-11):
    error = abs(float(a) - float(b))
    errors[name] = max(errors.get(name, 0.0), error)
    if error > atol * max(1.0, abs(float(a)), abs(float(b))):
        raise AssertionError(f"{name}: {a} != {b}")


def load_and_validate(root):
    stages, inputs, errors = {}, [], {}
    for stage, (rel, title, candidate) in STAGES.items():
        directory = root / rel / "results/independent_metrics"
        for filename in SOURCES:
            path = directory / filename
            inputs.append({"stage": stage, "path": str(path.relative_to(root)),
                           "sha256": sha(path), "bytes": path.stat().st_size})
        aggregates = csv_rows(directory / "AGGREGATE.csv")
        cells = csv_rows(directory / "FULL_CELLS.csv")
        cis = csv_rows(directory / "PAIRED_CI.csv")
        cj = json.loads((directory / "PAIRED_CI.json").read_text())
        gates = json.loads((directory / "GATES.json").read_text())
        receipt = json.loads((directory / "INDEPENDENT_METRICS.json").read_text())
        by_model = {x["model"]: x for x in aggregates}
        assert len(by_model) == len(aggregates)
        assert len(aggregates) == (11 if stage == "fdmc_v2" else 12)
        assert receipt["confirmation_data_read"] is False
        assert receipt["training_runs_by_this_script"] == 0
        for row in aggregates:
            model = row["model"]
            subset = [c for c in cells if c["model"] == model]
            keys = {(int(x["issue_year"]), int(x["horizon"])) for x in subset}
            assert len(subset) == 9 and keys == {(y, h) for y in YEARS for h in HORIZONS}
            for target, source in [(R, "mse_ratio_to_rw"), (Q, "crps_ratio_to_rw"),
                    ("equal_cell_price_mse", "price_mse"),
                    ("equal_cell_price_mae", "price_mae"),
                    ("equal_cell_price_crps", "price_crps"),
                    ("equal_cell_interval80_coverage", "interval80_coverage"),
                    ("equal_cell_interval80_width", "interval80_mean_width")]:
                close(num(row, target), np.mean([num(c, source) for c in subset]),
                      "aggregate_from_cells_" + target, errors)
            for field in ("n_issued", "n_price_scored", "n_price_unscored"):
                assert int(row[field]) == sum(int(c[field]) for c in subset)
            assert int(row["n_issued"]) == 1933
            assert int(row["n_price_scored"]) == 1927
            assert int(row["n_price_unscored"]) == 6
            assert int(row["cells_beating_rw_mse"]) == sum(num(c, "mse_ratio_to_rw") < 1 for c in subset)
            for c in subset:
                close(num(c, "mse_ratio_to_rw"), num(c, "price_mse") / num(c, "rw_price_mse"),
                      "cell_MSE_ratio_identity", errors)
            for year in YEARS:
                close(num(row, f"year_{year}_mse_ratio_to_rw"),
                      np.mean([num(c, "mse_ratio_to_rw") for c in subset if int(c["issue_year"]) == year]),
                      "year_R_identity", errors)
        summary = {x["model"]: x for x in receipt["primary_aggregate"]}
        for model, row in by_model.items():
            close(num(row, R), summary[model][R], "CSV_vs_JSON_R", errors)
        assert cj["draws"] == 2000 and cj["block_length_origins"] == 20 and cj["seed"] == 20261018
        cij = {x["fixed_comparator"]: x for x in cj["primary_comparisons"]}
        for row in cis:
            comparator = row["fixed_comparator"]
            assert row["model"] == candidate
            assert row["comparator_reselected_in_bootstrap"] == "False"
            for field in ("primary_R_ratio", "lower_95", "upper_95"):
                close(num(row, field), cij[comparator][field], "CI_CSV_vs_JSON_" + field, errors)
            close(num(row, "primary_R_ratio"), num(by_model[candidate], R) / num(by_model[comparator], R),
                  "CI_point_from_aggregate_R", errors)
            assert num(row, "lower_95") <= num(row, "upper_95")
        checks = gates["checks"]
        ci_map = {x["fixed_comparator"]: x for x in cis}
        best = gates["best_matched_comparator_fixed_from_original_sample"]
        close(checks["MSE_vs_RW"]["value"], num(by_model[candidate], R), "gate_RW_value", errors)
        close(checks["MSE_vs_best_matched_predictor"]["value"], num(ci_map[best], "primary_R_ratio"), "gate_best_value", errors)
        close(checks["uncertainty"]["upper_95"], num(ci_map[best], "upper_95"), "gate_CI_upper", errors)
        assert checks["MSE_vs_RW"]["pass"] == (checks["MSE_vs_RW"]["value"] <= .98)
        assert checks["MSE_vs_best_matched_predictor"]["pass"] == (checks["MSE_vs_best_matched_predictor"]["value"] <= .98)
        for key, comparator in [("mean_demand_integration", candidate + "_plugin"),
                                ("supply_operator", candidate + "_no_supply")]:
            close(checks[key]["value"], num(ci_map[comparator], "primary_R_ratio"), "gate_" + key, errors)
            assert checks[key]["pass"] == (checks[key]["value"] <= .99)
        stability = checks["stability"]
        assert stability["cells_beating_RW"] == int(by_model[candidate]["cells_beating_rw_mse"])
        close(stability["worst_year_R"], max(num(by_model[candidate], f"year_{y}_mse_ratio_to_rw") for y in YEARS), "gate_worst_year", errors)
        assert stability["pass"] == (stability["cells_beating_RW"] >= 6 and stability["worst_year_R"] <= 1.10)
        assert checks["uncertainty"]["pass"] == (checks["uncertainty"]["upper_95"] < 1)
        if "scale_separation" in checks:
            close(checks["scale_separation"]["value"], num(ci_map["FDMC_unseparated"], "primary_R_ratio"), "gate_scale_value", errors)
            assert checks["scale_separation"]["pass"] == (checks["scale_separation"]["value"] < 1)
        assert not gates["all_development_gates_pass"]
        assert gates["decision"] == "STOP_LEAVE_CONFIRMATION_UNSCORED"
        stages[stage] = {"title": title, "candidate": candidate, "aggregate": aggregates,
                         "models": by_model, "cells": cells, "ci": cis, "ci_map": ci_map,
                         "gates": gates, "population": receipt["population"], "ci_metadata": cj,
                         "source_dir": directory}
    first = stages["original_v1"]
    for stage in stages.values():
        assert stage["population"] == first["population"]
        rw_cells = sorted((int(c["issue_year"]), int(c["horizon"]), num(c, "price_mse"),
                           int(c["n_price_scored"])) for c in stage["cells"] if c["model"] == "RW")
        if stage is first:
            reference = rw_cells
        assert rw_cells == reference
    return stages, inputs, {"status": "PASS", "max_absolute_differences": errors,
                            "all_three_stages_same_population_and_RW_cell_MSE": True,
                            "all_gate_values_and_booleans_reconciled": True,
                            "formal_model_counts": {k: len(v["aggregate"]) for k, v in stages.items()}}


def base_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": "#B2BDC5", "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "svg.fonttype": "none", "svg.hashsalt": "carbon-auction-report-20261018",
        "savefig.facecolor": "white", "savefig.dpi": 320})


def save_figure(fig, output, stem, registry, caption):
    paths = []
    for extension in ("png", "svg"):
        path = output / f"{stem}.{extension}"
        fig.savefig(path, dpi=320, bbox_inches="tight", pad_inches=.16)
        meta = {"path": path.name, "sha256": sha(path), "bytes": path.stat().st_size}
        if extension == "png":
            with Image.open(path) as im:
                meta["pixel_dimensions"] = list(im.size)
                meta["dpi"] = list(im.info.get("dpi", (320, 320)))
        paths.append(meta)
    plt.close(fig)
    registry.append({"id": stem, "files": paths, "caption_zh": caption})


def figure_stages(stages, output, registry):
    fig, ax = plt.subplots(figsize=(11.2, 6.1))
    fig.subplots_adjust(left=.31, right=.95, top=.81, bottom=.24)
    centers = [2.6, 1.3, 0.0]
    ax.axhspan(2.02, 3.18, color="#F2F2F2", zorder=0)
    ax.axvline(1, color=INK, linestyle="--", linewidth=1.2, alpha=.65)
    for y, (key, stage) in zip(centers, stages.items()):
        candidate = stage["candidate"]
        for offset, model, color, marker in [(.27, "RW", INK, "s"),
                (0, "GRU_full", GREY, "o"), (-.27, candidate, BLUE if key == "fdmc_v2" else ORANGE, "D")]:
            value = num(stage["models"][model], R)
            ax.scatter(value, y + offset, s=52, color=color, marker=marker, zorder=4)
            ax.annotate(f"{value:.6f}", (value, y + offset), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=10, color=color)
    ax.set_yticks(centers, ["Original SDMC v1\nInput interpretation defect\nHistorical record only",
                           "Corrected SDMC v1\nPrimary v1 comparison",
                           "FDMC v2\nCorrected inputs"])
    ax.tick_params(axis="y", length=0, pad=13)
    ax.set_ylim(-.7, 3.28)
    ax.set_xlim(.973, 1.283)
    ax.set_xticks([1.00, 1.05, 1.10, 1.15, 1.20, 1.25])
    ax.grid(axis="x", color="#DCE3E8", linewidth=.7)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Primary aggregate R: equal-cell MSE / RW MSE (lower is better)", labelpad=10)
    fig.suptitle("Three completed stages: forecasting performance", x=.055, ha="left", y=.96,
                 fontsize=16, fontweight="bold")
    fig.text(.055, .902, "Point estimates; each stage uses its own matched-information GRU_full baseline.", fontsize=10.5)
    handles = [Line2D([], [], marker="s", color=INK, linestyle="", label="RW"),
               Line2D([], [], marker="o", color=GREY, linestyle="", label="GRU_full"),
               Line2D([], [], marker="D", color=ORANGE, linestyle="", label="SDMC"),
               Line2D([], [], marker="D", color=BLUE, linestyle="", label="FDMC")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.05, .085), ncol=4, frameon=False)
    fig.text(.055, .045, "Cross-stage differences combine input correction, context and model changes; they do not isolate one modification.", fontsize=9)
    fig.text(.055, .014, "All stages: 9 equal year x horizon cells; 1,927 scored records. Uncertainty is shown separately in Figure 4.", fontsize=9)
    save_figure(fig, output, "FIG01_THREE_STAGES", registry,
        "三阶段主指标。原始 v1 存在日历脚注的输入解释缺陷，仅保留为历史记录；主 v1 比较使用纠正后的完整重跑。"
        "阶段内使用各自的同信息 GRU_full 对照。跨阶段差值同时包含数据纠正、context 和参数化变化，不能解释为单一修改的净效果。"
        "本图只呈现点估计，没有把阶段差异当作独立的显著性检验；FDMC 的 R=1.004985 仍高于 RW。")


def figure_all_models(stages, output, registry):
    stage = stages["fdmc_v2"]
    order = ["RW", "FDMC", "FactorDirect", "GRU_full", "FDMC_plugin", "FDMC_no_supply",
             "LastBidMedian", "FDMC_unseparated", "MLP_full", "Ridge", "HistGB"]
    fig, ax = plt.subplots(figsize=(10.8, 7.1))
    fig.subplots_adjust(left=.235, right=.845, top=.80, bottom=.19)
    ys = np.arange(len(order))[::-1]
    ax.axvline(1.0, color=INK, linestyle="--", linewidth=1.4, label="RW = 1")
    ax.axvline(.98, color=BLUE, linestyle=":", linewidth=1.5, label="Candidate gate: R <= 0.98")
    transform = blended_transform_factory(ax.transAxes, ax.transData)
    for y, model in zip(ys, order):
        value = num(stage["models"][model], R)
        color = BLUE if model == "FDMC" else (INK if model == "RW" else GREY)
        marker = "D" if model == "FDMC" else ("s" if model == "RW" else "o")
        if model == "FDMC":
            ax.axhspan(y - .4, y + .4, color="#EEF5FA", zorder=0)
        ax.scatter(value, y, color=color, marker=marker, s=58, zorder=3)
        ax.text(1.025, y, f"{value:.6f}", transform=transform, va="center", color=color,
                fontweight="bold" if model == "FDMC" else "normal")
    ax.text(1.025, len(order) - .1, "R", transform=transform, fontweight="bold")
    ax.set_yticks(ys, order)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlim(.955, 1.53)
    ax.set_xticks([.98, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50])
    ax.set_xticklabels(["0.98", "1.00", "1.10", "1.20", "1.30", "1.40", "1.50"], rotation=35, ha="right")
    ax.set_ylim(-.7, len(order) - .1)
    ax.grid(axis="x", color="#DCE3E8", linewidth=.7)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Primary aggregate R (lower is better)", labelpad=9)
    fig.suptitle("FDMC v2: all frozen model comparisons", x=.05, ha="left", y=.955,
                 fontsize=16, fontweight="bold")
    fig.text(.05, .903, "Nine equal year x horizon cells; three-seed ensembles for neural models.", fontsize=10.5)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(.045, .074),
               frameon=False, ncol=2, fontsize=9.5)
    fig.text(.05, .037, "FDMC R = 1.004985; the predeclared RW threshold (0.98) was not met. Dots are point estimates, not significance marks.", fontsize=9)
    save_figure(fig, output, "FIG02_FDMC_ALL_MODELS", registry,
        "FDMC v2 全部 11 个冻结模型的主指标 R。黑色虚线为 RW=1，蓝色点线为候选模型的预定门槛 R≤0.98。"
        "所有模型按同一九格定义汇总。FDMC R=1.004985，未达到 RW 优势门槛；点估计图不表示统计显著性。")


def matrix(stage, model):
    index = {(int(c["issue_year"]), int(c["horizon"])): num(c, "mse_ratio_to_rw")
             for c in stage["cells"] if c["model"] == model}
    return np.asarray([[index[y, h] for h in HORIZONS] for y in YEARS])


def figure_heatmaps(stages, output, registry):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.4))
    fig.subplots_adjust(left=.09, right=.83, top=.835, bottom=.18, wspace=.28, hspace=.54)
    cmap = LinearSegmentedColormap.from_list("blue_white_orange", [BLUE, "#FAFAFA", ORANGE])
    norm = Normalize(vmin=.15, vmax=1.85)
    specifications = [(stages["fdmc_v2"], "FDMC", "A  FDMC v2"),
                      (stages["fdmc_v2"], "FactorDirect", "B  FactorDirect v2"),
                      (stages["fdmc_v2"], "GRU_full", "C  GRU_full v2"),
                      (stages["corrected_v1"], "SDMC", "D  Corrected SDMC v1")]
    for ax, (stage, model, title) in zip(axes.flat, specifications):
        values = matrix(stage, model)
        assert values.min() >= norm.vmin and values.max() <= norm.vmax
        im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        for (i, j), value in np.ndenumerate(values):
            color = "white" if abs(value - 1) > .60 else INK
            ax.text(j, i, f"{value:.4f}", ha="center", va="center", fontsize=12, color=color)
        ax.set_title(f"{title}\nAggregate R = {num(stage['models'][model], R):.6f}", loc="left", pad=11, fontsize=11.5)
        ax.set_xticks(range(3), ["H = 1", "H = 5", "H = 20"])
        ax.set_yticks(range(3), YEARS)
        ax.tick_params(length=0, pad=7)
        ax.set_ylabel("Issue year")
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([.87, .255, .027, .49])
    bar = fig.colorbar(im, cax=cax, ticks=[.2, .6, 1.0, 1.4, 1.8])
    bar.set_label("Cell MSE / same-record RW MSE", labelpad=12)
    fig.suptitle("Where the forecast errors occur", x=.06, ha="left", y=.965,
                 fontsize=16, fontweight="bold")
    fig.text(.06, .915, "Shared linear colour scale centred at 1; blue < 1, orange > 1. Values are descriptive, without cellwise tests.", fontsize=10)
    fig.text(.06, .105, "Each panel is a 3 x 3 issue-year / forecast-horizon grid. The four panels use the same RW denominators.", fontsize=9)
    fig.text(.06, .069, "FDMC beats RW in 5/9 cells, below the predeclared 6/9 requirement. Near-white cells indicate ratios near one.", fontsize=9)
    fig.text(.06, .033, "Corrected SDMC v1 is a cross-stage reference; its contrast with v2 does not isolate one model change.", fontsize=9)
    save_figure(fig, output, "FIG03_YEAR_HORIZON_HEATMAPS", registry,
        "FDMC、FactorDirect、GRU_full 的 v2 结果以及纠正 SDMC v1 的九格 MSE/RW 热图。横轴 H=1、5、20，纵轴为预测起点年份。"
        "四面板共用以 1 为中点的线性色标 [0.15,1.85]，无截断、无逐格显著性标记。"
        "v2 比值接近 1，必须结合单元格数字解读，不能把颜色微差当作优势证据。FDMC 仅 5/9 格胜过 RW，未达到 6/9 稳定性门槛。")


def figure_forest(stages, output, registry):
    stage = stages["fdmc_v2"]
    order = ["RW", "GRU_full", "FactorDirect", "FDMC_plugin", "FDMC_no_supply", "FDMC_unseparated"]
    labels = ["RW", "GRU_full (fixed best matched)", "FactorDirect", "FDMC_plugin", "FDMC_no_supply", "FDMC_unseparated"]
    fig, ax = plt.subplots(figsize=(12, 6.7))
    fig.subplots_adjust(left=.275, right=.695, top=.77, bottom=.265)
    ys = np.arange(len(order))[::-1]
    ax.axvline(1, color=INK, linestyle="--", linewidth=1.4)
    transform = blended_transform_factory(ax.transAxes, ax.transData)
    for y, comparator in zip(ys, order):
        row = stage["ci_map"][comparator]
        point, lo, hi = (num(row, k) for k in ("primary_R_ratio", "lower_95", "upper_95"))
        color = PURPLE if comparator == "FDMC_unseparated" else BLUE
        marker = "D" if comparator == "FDMC_unseparated" else "o"
        ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt=marker,
                    color=color, ecolor=color, elinewidth=1.8, capsize=4, markersize=6.5, zorder=3)
        ax.text(1.04, y, f"{point:.6f}  [{lo:.6f}, {hi:.6f}]", transform=transform,
                va="center", fontsize=9.8, color=INK, clip_on=False)
    ax.text(1.04, len(order) - .07, "R ratio  [95% paired CI]", transform=transform,
            fontsize=10, fontweight="bold", clip_on=False)
    ax.set_yticks(ys, labels)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlim(.77, 1.055)
    ax.set_xticks([.8, .85, .9, .95, 1.0, 1.05])
    ax.set_ylim(-.75, len(order) - .1)
    ax.set_xlabel("R(FDMC) / R(fixed comparator); lower is better", labelpad=10)
    ax.grid(axis="x", color="#DCE3E8", linewidth=.7)
    ax.spines["left"].set_visible(False)
    fig.suptitle("FDMC: paired uncertainty against the required controls", x=.045, ha="left", y=.95,
                 fontsize=16, fontweight="bold")
    fig.text(.045, .897, "Statistic: ratio of aggregate R, where R is the mean of nine within-cell MSE / RW MSE ratios.", fontsize=10.5)
    fig.text(.045, .851, "The first five intervals include 1. An interval containing 1 does not establish equivalence or inferiority.", fontsize=10.5)
    fig.text(.045, .157, "95% paired percentile intervals: 2,000 draws; noncircular 20-origin blocks sampled separately within each year.", fontsize=9)
    fig.text(.045, .119, "All horizons/platforms at an issue date share the same draw weight. Comparators and fitted forecasts remain fixed.", fontsize=9)
    fig.text(.045, .081, "Intervals omit retraining and comparator-selection uncertainty; they are not simultaneous confidence guarantees.", fontsize=9)
    fig.text(.045, .043, "The unseparated contrast supports this scale parameterization only; it does not replace the six failed primary gates.", fontsize=9)
    save_figure(fig, output, "FIG04_FDMC_PAIRED_CI", registry,
        "FDMC 相对固定对照的配对 95% 百分位区间。统计量 B=R_FDMC/R_comp，R 为九格等权的 MSE/RW；"
        "不是 pooled MSE 比，也不是九格 FDMC/对照 MSE 比的均值。每年独立按 20 个连续 unique issue_date 构成非循环移动块，"
        "2,000 次抽样、seed=20261018；同日起点的所有 horizon/platform 共用权重，每个 draw 内重算 RW 分母。"
        "对照身份和已拟合预测固定，不包含重新训练、最佳对照选择的不确定性，不提供同时置信保证。"
        "相对 RW、GRU、FactorDirect、plugin、no_supply 的区间均包含 1，不能解释为等价或劣势证明。"
        "相对 unseparated 的优势仅支持本次尺度参数化，不能替代原六项推进门槛。")


def md_table(headers, rows):
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    output.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return "\n".join(output)


def make_tables(stages, output, registry):
    lines = ["# 正式报告数表与图注", "", "所有数值由三轮正式独立评分收据自动生成；本步骤不重算预测、不重训、不读取确认数据。",
        "", "## 1. 统计口径与适用范围", "",
        r"主指标 $R_m=\frac{1}{9}\sum_{c=(year,H)}\mathrm{MSE}_{m,c}/\mathrm{MSE}_{RW,c}$，年份为 2021、2022、2023，H 为 1、5、20。格内按正式评分记录计算 MSE，九格等权。R<1 表示点估计优于 RW，推进门槛为候选 R≤0.98。R 不是 pooled MSE 比。",
        "", "每个模型均发出 1,933 条预测，1,927 条有正有限价格并进入评分；6 条无价格预测保留，涉及 2 个取消目标事件在多个 horizon 上的重复。共有 652 个 unique issue_date。三 seed 的点预测先等权平均、分布先等权混合，再评分；seed 不是独立市场样本。",
        "", "CRPS/RW 也按九格等权计算，为次要分布指标；覆盖率和宽度为 80% 预测区间的九格均值。MSE、MAE、CRPS 和区间宽度均在原始价格尺度上计算。表格显示位数经过舍入，原始 CSV/JSON 保留完整精度。",
        "", "原始 v1 存在日历脚注未应用的输入解释缺陷，只作历史记录；正式 v1 判断使用纠正 v1。v2 与纠正 v1 使用同一纠正日历，但其 origin-known context 和模型参数化不同，跨阶段差异不能解释为单一修改的净效果。",
        "", "## 2. 三阶段关键比较", ""]
    rows = []
    for key, stage in stages.items():
        candidate = stage["candidate"]
        rw_ci = stage["ci_map"]["RW"]
        best_ci = stage["ci_map"]["GRU_full"]
        rows.append([stage["title"], candidate, f"{num(stage['models'][candidate], R):.6f}",
                     f"{num(stage['models']['GRU_full'], R):.6f}",
                     f"[{num(rw_ci, 'lower_95'):.6f}, {num(rw_ci, 'upper_95'):.6f}]",
                     f"{num(best_ci, 'primary_R_ratio'):.6f}", "STOP"])
    lines += [md_table(["阶段", "候选", "候选 R", "同阶段 GRU R", "候选/RW 的 95% CI", "候选/GRU 的 R 比", "决定"], rows), ""]
    for section, (key, stage) in enumerate(stages.items(), 3):
        lines += [f"## {section}. {stage['title']}：完整主表", ""]
        rows = []
        for row in stage["aggregate"]:
            rows.append([row["model"], f"{num(row,R):.6f}", f"{num(row,Q):.6f}",
                         f"{num(row,'equal_cell_price_mse'):.4f}", f"{num(row,'equal_cell_price_mae'):.4f}",
                         f"{num(row,'equal_cell_price_crps'):.4f}",
                         f"{100*num(row,'equal_cell_interval80_coverage'):.2f}%",
                         f"{num(row,'equal_cell_interval80_width'):.4f}", f"{int(row['cells_beating_rw_mse'])}/9"])
        lines += [md_table(["模型", "R（主）", "CRPS/RW（次）", "MSE", "MAE", "CRPS", "80% 覆盖率", "80% 宽度", "胜 RW 格数"], rows), ""]
    lines += ["## 6. 完整配对置信区间", "",
        r"所有区间的主统计量均为 $B=R_{candidate}/R_{fixed\ comparator}$。每个 bootstrap draw 内重算各格模型 MSE 与同记录 RW MSE，再取九格等权 R，最后求 R 之比。该量既不是 pooled MSE 比，也不是逐格 candidate/comparator 比的均值。",
        "", "每年对排序的 unique issue_date 抽取非循环 20-origin 移动块，抽至原日期数并截断；同日所有 horizon/platform/模型共用 multiplicity。2,000 draws，固定 seed=20261018，百分位 [2.5%,97.5%]。对照身份固定，不在每次 draw 重新选最佳；区间不含重训或选择不确定性，也不提供所有比较同时成立的 95% 保证。", ""]
    for stage in stages.values():
        lines += [f"### {stage['title']}", ""]
        rows = []
        for row in stage["ci"]:
            lo, hi = num(row, "lower_95"), num(row, "upper_95")
            relation = "包含 1" if lo <= 1 <= hi else ("整体低于 1" if hi < 1 else "整体高于 1")
            rows.append([row["model"], row["fixed_comparator"], f"{num(row,'primary_R_ratio'):.6f}",
                         f"{lo:.6f}", f"{hi:.6f}", relation])
        lines += [md_table(["候选", "固定对照", "B 点估计", "95% 下界", "95% 上界", "与 1 的关系"], rows), ""]
    lines += ["包含 1 表示在这套固定抽样方案下未取得相应方向的区间证据，不能证明统计等价或劣势。FDMC 相对 RW、GRU_full、FactorDirect、plugin 和 no_supply 的区间均包含 1。相对 unseparated 的区间低于 1 只支持本次尺度参数化，不替代原六项推进门槛。", "", "## 7. 冻结门槛逐项验收", ""]
    labels = {"MSE_vs_RW": "相对 RW", "MSE_vs_best_matched_predictor": "相对最佳同信息对照",
              "mean_demand_integration": "随机积分 / plugin", "supply_operator": "供给算子 / no_supply",
              "stability": "稳定性", "uncertainty": "最佳同信息对照的区间", "scale_separation": "附加尺度门槛"}
    for stage in stages.values():
        lines += [f"### {stage['title']}", ""]
        rows = []
        for key, gate in stage["gates"]["checks"].items():
            if key == "stability":
                value = f"{gate['cells_beating_RW']}/9；最差年度 R={gate['worst_year_R']:.6f}"
            elif key == "uncertainty":
                value = f"[{gate['lower_95']:.6f}, {gate['upper_95']:.6f}]"
            else:
                value = f"{gate['value']:.6f}"
            rows.append([labels[key], gate["criterion"], value, "PASS" if gate["pass"] else "FAIL"])
        lines += [md_table(["门槛", "冻结标准（原文）", "实际值", "结果"], rows), "",
                  f"固定最佳同信息对照：`{stage['gates']['best_matched_comparator_fixed_from_original_sample']}`。正式决定：`{stage['gates']['decision']}`。", ""]
    lines += ["纠正 v1 原六项仅 plugin 比较通过；FDMC 原六项全部未通过，仅附加尺度门槛通过。两种已试实现均停止进入确认阶段。", "", "## 8. 逐年 / 逐 horizon 描述表", "",
              "以下为相应三格的等权 R 均值，属于既有九格的描述性分解，不替代主指标或建立新的检验。", ""]
    for stage in stages.values():
        lines += [f"### {stage['title']}", ""]
        rows = []
        for model, aggregate in stage["models"].items():
            cells = [c for c in stage["cells"] if c["model"] == model]
            rows.append([model] + [f"{num(aggregate, f'year_{y}_mse_ratio_to_rw'):.6f}" for y in YEARS]
                        + [f"{np.mean([num(c,'mse_ratio_to_rw') for c in cells if int(c['horizon'])==h]):.6f}" for h in HORIZONS])
        lines += [md_table(["模型", "2021", "2022", "2023", "H=1", "H=5", "H=20"], rows), ""]
    lines += ["## 9. 热图精确数表", ""]
    for stage_key, model in [("fdmc_v2", "FDMC"), ("fdmc_v2", "FactorDirect"), ("fdmc_v2", "GRU_full"), ("corrected_v1", "SDMC")]:
        values = matrix(stages[stage_key], model)
        lines += [f"### {stages[stage_key]['title']} / {model}", "",
                  md_table(["起点年份", "H=1", "H=5", "H=20"], [[y] + [f"{v:.8f}" for v in values[i]] for i,y in enumerate(YEARS)]), ""]
    lines += ["## 10. 图件与可直接粘贴的图注", ""]
    for figure in registry:
        lines += [f"### {figure['id']}", "", figure["caption_zh"], "",
                  f"PNG：`{figure['id']}.png`；SVG：`{figure['id']}.svg`。", ""]
    lines += ["## 11. 正式来源与复现", "",
              "本文件与四图全部由 `reporting/make_report_figures.py` 从以下目录的 `AGGREGATE.csv`、`FULL_CELLS.csv`、`PAIRED_CI.csv`、`PAIRED_CI.json`、`GATES.json`、`INDEPENDENT_METRICS.json` 生成：", ""]
    for key, stage in stages.items():
        rel = Path(STAGES[key][0]) / "results/independent_metrics"
        lines.append(f"- {stage['title']}：`{rel.as_posix()}/`。")
    lines += ["", "全部 18 个正式输入的 SHA256、交叉核对结果和导出文件的 SHA256 / 像素尺寸见 `FIGURES_MANIFEST.json`。图和数表只消费正式收据；未读取原始行情或预测 NPZ，没有改变模型、输入、冻结标准和既有统计结果。", "",
              "复现命令（从研究根目录执行）：", "", "```bash", "python reporting/make_report_figures.py --study-dir .", "```", ""]
    path = output / "REPORT_TABLES_ZH.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.study_dir.resolve()
    output = root / "reporting"
    output.mkdir(parents=True, exist_ok=True)
    stages, inputs, validation = load_and_validate(root)
    base_style()
    figures = []
    figure_stages(stages, output, figures)
    figure_all_models(stages, output, figures)
    figure_heatmaps(stages, output, figures)
    figure_forest(stages, output, figures)
    tables = make_tables(stages, output, figures)
    manifest = {"status": "PASS_DATA_VALIDATED_RENDERED", "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": {"path": "make_report_figures.py", "sha256": sha(__file__)},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "scope": {"source_summary_files_only": True, "prediction_NPZ_read": False,
                  "confirmation_data_read": False, "training_performed": False,
                  "all_writes_in_reporting": True, "no_PDF_generated": True},
        "inputs": inputs, "validation": validation,
        "visual_review": {"status": "PENDING_HUMAN_MODEL_IMAGE_INSPECTION"},
        "figures": figures,
        "tables": {"path": tables.name, "sha256": sha(tables), "bytes": tables.stat().st_size},
        "statistical_interpretation": {
            "R": "Equal mean of nine year/horizon MSE_model / same-record MSE_RW ratios",
            "CI_statistic": "B = R_candidate / R_fixed_comparator, recalculated within each paired draw",
            "CI": "2000 year-stratified paired noncircular moving-block draws; 20 unique issue dates; seed 20261018",
            "not_pooled_MSE_ratio": True, "not_mean_direct_comparator_cell_ratios": True,
            "fixed_comparators_and_forecasts": True, "not_selection_adjusted_or_simultaneous": True,
            "original_v1_input_interpretation_defect_marked": True,
            "cross_stage_difference_not_single_change_effect": True,
            "FDMC_significant_advantage_over_RW_claimed": False,
            "contains_one_not_proof_of_equivalence_or_inferiority": True},
        "plotted_values": {
            "three_stages": {k: {m: num(v["models"][m], R) for m in ("RW", "GRU_full", v["candidate"])} for k,v in stages.items()},
            "v2_all_R": {m: num(v,R) for m,v in stages["fdmc_v2"]["models"].items()},
            "heatmaps": {f"{k}/{m}": matrix(stages[k], m).tolist() for k,m in [("fdmc_v2","FDMC"),("fdmc_v2","FactorDirect"),("fdmc_v2","GRU_full"),("corrected_v1","SDMC")]},
            "v2_forest": {m: {f: num(stages['fdmc_v2']['ci_map'][m], f) for f in ('primary_R_ratio','lower_95','upper_95')}
                          for m in ('RW','GRU_full','FactorDirect','FDMC_plugin','FDMC_no_supply','FDMC_unseparated')}}}
    path = output / "FIGURES_MANIFEST.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "figures": len(figures), "formats": ["PNG", "SVG"],
                      "tables": str(tables), "manifest": str(path), "data_validation": validation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
