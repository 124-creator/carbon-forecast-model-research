"""Export fixed public D1 aggregate JSON to CSV and scientific PNG/SVG figures.

Only the explicit public results JSON allowlist below is read. No CSV input,
local_only file, checkpoint, confirmation, model or training module is opened.
Existing selections and source row order are retained; no selector is run.
Requires NumPy and Matplotlib. Default outputs are results/tables and figures.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


PROTOCOL_SHA = "b10409de072efaba5fce1d570c15e6e0b4128f5234f037c25afeecfd90efc503"
MARKETS = ["EEX_EUA_T3PA", "GUANGZHOU", "HUBEI", "SHANGHAI", "EU_REPORTED"]
PRIMARY_H = [1, 5, 20]
PRIMARY = ["gru_native", "gru_normalized", "gru_aux", "lstm_native",
           "patchtst_native", "ridge", "ridge_val", "markov_mle"]
SMP = ["smp_full", "smp_no_clock", "smp_no_sign", "smp_pooled_mark",
       "smp_iid_state", "smp_one_step", "smp_no_aux"]
LABELS = {
    "rw": "Random walk", "smp_full": "SMP full", "smp_no_clock": "No clock",
    "smp_no_sign": "No sign", "smp_pooled_mark": "Pooled marks",
    "smp_iid_state": "IID state", "smp_one_step": "One-step training",
    "smp_no_aux": "No auxiliary loss", "gru_native": "GRU (native loss)",
    "gru_normalized": "GRU (normalized loss)", "gru_aux": "GRU + auxiliary loss",
    "lstm_native": "LSTM", "patchtst_native": "PatchTST",
    "ridge": "Ridge (global choice)", "ridge_val": "Ridge (local validation)",
    "markov_mle": "Markov MLE", "project_reference": "Project reference",
}
SHORT_MARKETS = ["EEX EUA", "Guangzhou", "Hubei", "Shanghai", "EU reported"]
DECOMP_FIELDS = [
    "relative_mse", "prediction_energy_over_rw", "twice_alignment_over_rw",
    "prediction_rms_over_rw", "mean_prediction_over_rw_rms",
    "mean_truth_over_rw_rms", "mean_error_over_rw_rms", "correlation",
]
TOP_JSON = {"D1_ALL_CONFIGS.json", "D1_SELECTION.json", "D1_FAILURE_DECOMPOSITION.json",
            "D1/STARTED.json", "D1/COMPLETED.json"}
JOB_JSON = re.compile(r"D1/jobs/[A-Za-z0-9_]+/(FIT|SUPPORT|METRICS)[.]json")
RED, BLUE, GREEN, GOLD, PURPLE, GRAY = "#B23A48", "#245C89", "#237A57", "#AF7D20", "#7963A6", "#6B7280"


def sha(body):
    return hashlib.sha256(body).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message):
    require(np.allclose(a, b, atol=2e-12, rtol=2e-12, equal_nan=False), message)


class PublicResults:
    def __init__(self, study):
        self.root = (Path(study).resolve() / "results").resolve()
        self.records = {}

    def read(self, relative):
        require(relative in TOP_JSON or JOB_JSON.fullmatch(relative), "JSON outside public allowlist: " + relative)
        path = self.root / relative
        require(not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(self.root),
                "Missing or unsafe public JSON: " + relative)
        body = path.read_bytes()
        self.records[relative] = {"path": "results/" + relative, "sha256": sha(body), "bytes": len(body)}
        return json.loads(body)


def scalar(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if isinstance(value, float):
        require(math.isfinite(value), "Nonfinite CSV scalar")
    return value


def csv_body(rows):
    require(bool(rows), "Empty export table")
    columns = list(rows[0])
    require(all(list(row) == columns for row in rows), "CSV schema changed within a table")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: scalar(value) for key, value in row.items()} for row in rows)
    return stream.getvalue().encode("utf-8"), columns


def aggregate_tables(rows, collection, source_name, source_hash):
    summaries, long_rows = [], []
    for order, row in enumerate(rows, 1):
        require(list(row["by_market"]) == MARKETS, "Market order or support inventory changed")
        require(len(row["by_market"]) == 5, "Five series required")
        summary = dict(source_order=order, collection=collection, year=2022,
                       family=row["family"], config_id=row["config_id"], config_value=row["value"],
                       score=row["score"], aggregation=row["aggregation"])
        for horizon in PRIMARY_H:
            summary["mean_relative_mse_h" + str(horizon)] = float(np.mean([
                cell["relative_mse"][horizon - 1] for cell in row["by_market"].values()]))
        for market, cell in row["by_market"].items():
            relative = np.asarray(cell["relative_mse"], dtype=float)
            require(relative.shape == (20,) and np.isfinite(relative).all(), "Incomplete all20 aggregate")
            close(cell["score"], relative[np.asarray(PRIMARY_H) - 1].mean(), "Market score differs from h1/5/20")
            summary[market + "_score"] = cell["score"]
            for horizon in range(1, 21):
                long_rows.append(dict(
                    source_order=order, collection=collection, year=2022, family=row["family"],
                    config_id=row["config_id"], config_value=row["value"], market=market,
                    horizon=horizon, is_primary_horizon=horizon in PRIMARY_H,
                    relative_mse=cell["relative_mse"][horizon - 1],
                    excess_relative_mse=cell["relative_mse"][horizon - 1] - 1.0,
                    market_primary_score=cell["score"], evaluation_n=cell["n"],
                    seeds=cell["seeds"], job_ids=cell["job_ids"],
                    local_ridge_selected_value=cell.get("selected_value"),
                    local_ridge_validation_score=cell.get("selected_validation_score"),
                    source_json=source_name, source_sha256=source_hash))
        close(row["score"], np.mean([cell["score"] for cell in row["by_market"].values()]),
              "Global score differs from equal-series mean")
        summary["source_json"], summary["source_sha256"] = source_name, source_hash
        summaries.append(summary)
    return summaries, long_rows


def job_tables(reader, started, completed):
    ids = started["job_ids"]
    require(len(ids) == 160 and len(set(ids)) == 160, "Expected 160 unique D1 jobs")
    completion = {row["id"]: row for row in completed["jobs"]}
    require(set(completion) == set(ids), "Completion job inventory differs")
    summaries, horizons, metadata = [], [], {}
    for order, job_id in enumerate(ids, 1):
        base = "D1/jobs/" + job_id + "/"
        metric = reader.read(base + "METRICS.json")
        support = reader.read(base + "SUPPORT.json")
        fit = reader.read(base + "FIT.json") if metric["kind"] in {"neural", "ridge", "markov_mle"} else {}
        require(metric["id"] == job_id and metric["phase"] == "D1" and metric["year"] == 2022, "Job metadata mismatch")
        require(metric["protocol_sha256"] == PROTOCOL_SHA, "Job protocol mismatch")
        require(metric["status"] == completion[job_id]["status"] == "completed", "Incomplete job")
        require(metric["n"] == support["evaluation_n"], "Job support count mismatch")
        close(np.asarray(metric["mse"]) / np.asarray(metric["rw_mse"]), metric["relative_mse"],
              "Public job numerator/denominator mismatch")
        close(metric["score"], np.asarray(metric["relative_mse"])[[0, 4, 19]].mean(), "Public job score mismatch")
        summaries.append(dict(
            plan_order=order, job_id=job_id, year=metric["year"], market=metric["market"],
            family=metric["family"], config_id=metric["config_id"], kind=metric["kind"],
            config_value=metric["value"], seed=metric["seed"], status=metric["status"],
            fit_receipt_present=bool(fit), eligible_for_selection=fit.get("eligible_for_selection"),
            stop_reason=fit.get("stop_reason"), selected_epoch=fit.get("selected_epoch"),
            executed_epochs=fit.get("executed_epochs"), optimizer_steps=fit.get("optimizer_steps", 0),
            parameters=fit.get("parameters"), trainable_parameters=fit.get("trainable_parameters"),
            initial_validation_score=fit.get("initial_validation_score"),
            selected_validation_score=metric.get("selected_validation_score"),
            evaluation_score=metric["score"], train_n=support["train_n"],
            validation_n=support["validation_n"], evaluation_n=support["evaluation_n"],
            input_channel_count=len(support["input_channels"]),
            train_first_origin=support["train_first_origin"], train_last_origin=support["train_last_origin"],
            max_train_label=support["max_train_label"], first_validation_origin=support["first_validation_origin"],
            max_validation_label=support["max_validation_label"],
            evaluation_first_origin=support["evaluation_first_origin"],
            evaluation_last_origin=support["evaluation_last_origin"],
            evaluation_last_label=support["evaluation_last_label"],
            evaluation_first_date=support["evaluation_first_date"],
            evaluation_last_date=support["evaluation_last_date"],
            training_state_counts=support["training_state_counts"],
            wall_seconds=metric["wall_seconds"], fit_wall_seconds=fit.get("wall_seconds"),
            initial_state_sha256=fit.get("initial_state_sha256"),
            selected_state_sha256=fit.get("selected_state_sha256"),
            public_metric_json="results/" + base + "METRICS.json",
            public_metric_sha256=reader.records[base + "METRICS.json"]["sha256"],
            public_support_sha256=reader.records[base + "SUPPORT.json"]["sha256"],
            public_fit_sha256=reader.records.get(base + "FIT.json", {}).get("sha256")))
        for horizon, relative in enumerate(metric["relative_mse"], 1):
            horizons.append(dict(plan_order=order, job_id=job_id, year=2022, market=metric["market"],
                                 family=metric["family"], config_id=metric["config_id"], config_value=metric["value"],
                                 seed=metric["seed"], horizon=horizon, is_primary_horizon=horizon in PRIMARY_H,
                                 relative_mse=relative, evaluation_n=metric["n"],
                                 source_json="results/" + base + "METRICS.json",
                                 source_sha256=reader.records[base + "METRICS.json"]["sha256"]))
        metadata[job_id] = metric
    return summaries, horizons, metadata


def decomposition_table(obj, selected, source_hash):
    rows = []
    maximum_error = 0.0
    require(len(obj["rows"]) == 85, "Expected 17 selected models by five series")
    for order, record in enumerate(obj["rows"], 1):
        selected_row = selected[record["family"]]
        require(selected_row["config_id"] == record["config_id"], "Decomposition uses a different selected config")
        cell = selected_row["by_market"][record["market"]]
        require(record["job_id"] in cell["job_ids"] and record["n"] == cell["n"], "Decomposition job/support mismatch")
        close(record["relative_mse"], cell["relative_mse"], "Decomposition MSE differs")
        error = np.asarray(record["relative_mse"]) - (
            1 + np.asarray(record["prediction_energy_over_rw"]) - np.asarray(record["twice_alignment_over_rw"]))
        maximum_error = max(maximum_error, float(np.max(np.abs(error))))
        require(maximum_error < 2e-11, "Decomposition algebra mismatch")
        for horizon in range(1, 21):
            row = dict(source_order=order, year=2022, family=record["family"], config_id=record["config_id"],
                       market=record["market"], horizon=horizon, evaluation_n=record["n"], job_id=record["job_id"])
            for field in DECOMP_FIELDS:
                require(len(record[field]) == 20, "Incomplete decomposition")
                row[field] = record[field][horizon - 1]
            row["excess_relative_mse"] = record["relative_mse"][horizon - 1] - 1
            row["identity_error"] = float(error[horizon - 1])
            row["source_json"] = "results/D1_FAILURE_DECOMPOSITION.json"
            row["source_sha256"] = source_hash
            rows.append(row)
    return rows, maximum_error


def plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 12,
        "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#374151", "text.color": "#202938",
        "axes.labelcolor": "#202938", "axes.grid": False,
        "svg.fonttype": "none", "svg.hashsalt": "smp-d1-public-export-v1",
        "savefig.facecolor": "white",
    })


def dot_panel(ax, names, source, title):
    values = [source[name]["score"] for name in names]
    y = np.arange(len(names))
    ax.axvline(1, color=GRAY, linestyle="--", linewidth=1.2, zorder=1)
    for index, (name, value) in enumerate(zip(names, values)):
        color = RED if name == "smp_full" else (GRAY if name == "rw" else BLUE)
        ax.scatter(value, index, s=58 if name == "smp_full" else 42,
                   color=color, edgecolors="white", linewidths=.5, zorder=3)
        ax.annotate(f"{value:.4f}", (value, index), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=color)
    ax.set_yticks(y, [LABELS[name] for name in names])
    ax.set_ylim(len(names) - .5, -.65)
    ax.set_xlim(min(1, min(values)) - .015, max(values) + .060)
    ax.set_title(title, loc="left", pad=16)
    ax.set_xlabel("Relative MSE (RW = 1; lower is better)")
    ax.grid(axis="x", color="#E5E7EB", linewidth=.65)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def figures(selection, decomp):
    plot_style()
    selected, matched = selection["selected"], selection["matched_to_smp"]
    output = []
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.9))
    dot_panel(axes[0], ["rw", "smp_full"] + PRIMARY, selected, "(a) Eight primary baselines, RW and SMP")
    dot_panel(axes[1], SMP, matched,
              f"(b) SMP ablations at matched shrink alpha = {matched['smp_full']['value']:g}")
    fig.suptitle("D1 2022 development results", fontsize=16, x=.06, ha="left", y=.97)
    fig.text(.06, .918, "Neural seed 17; deterministic statistical references; no confidence intervals.", fontsize=10.5, color=GRAY)
    fig.text(.06, .027, "Equal weight across five series and horizons 1, 5, 20. Original D1 gate: FAILED.", fontsize=10.5)
    fig.subplots_adjust(left=.18, right=.97, bottom=.13, top=.82, wspace=.84)
    output.append(("D1_main_and_ablations", fig))

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.0))
    h = np.arange(1, 21)
    lines = [
        ("smp_full", selected, RED, "-", "SMP full"),
        ("smp_one_step", matched, PURPLE, "--", "SMP one-step (matched)"),
        ("gru_native", selected, BLUE, "-", "GRU (native loss)"),
        ("markov_mle", selected, GREEN, "-.", "Markov MLE"),
        ("patchtst_native", selected, GOLD, ":", "PatchTST"),
    ]
    for name, src, color, style, label in lines:
        axes[0].plot(h, src[name]["by_market"]["EEX_EUA_T3PA"]["relative_mse"],
                     color=color, linestyle=style, linewidth=2 if name == "smp_full" else 1.65, label=label)
    axes[0].axhline(1, color=GRAY, linestyle="--", linewidth=1, label="RW")
    axes[0].set_title("(a) Relative MSE by observation horizon", loc="left", pad=12)
    axes[0].set_ylabel("MSE / same-horizon RW MSE")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9.2)
    record = next(row for row in decomp["rows"] if row["family"] == "smp_full" and row["market"] == "EEX_EUA_T3PA")
    axes[1].plot(h, record["prediction_energy_over_rw"], color=GOLD, linewidth=2, label="Prediction energy  E[f²] / E[y²]")
    axes[1].plot(h, record["twice_alignment_over_rw"], color=GREEN, linewidth=2, label="Twice alignment  2 E[fy] / E[y²]")
    axes[1].plot(h, np.asarray(record["relative_mse"]) - 1, color=RED, linewidth=2, linestyle="--", label="Excess relative MSE  (energy − twice alignment)")
    axes[1].axhline(0, color=GRAY, linewidth=.8)
    axes[1].set_title("(b) SMP full: exact error decomposition", loc="left", pad=12)
    axes[1].set_ylabel("Units of same-horizon RW MSE")
    axes[1].legend(loc="upper left", frameon=False, fontsize=9.0)
    for ax in axes:
        ax.set_xlabel("Forecast horizon (observation steps)")
        ax.set_xlim(1, 20)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(color="#E5E7EB", linewidth=.6)
    fig.suptitle("EEX EUA primary auctions: horizon risk and SMP decomposition", fontsize=15, x=.065, ha="left", y=.975)
    fig.text(.065, .922, "2022 development; 193 common forecast origins; neural seed 17; no confidence intervals.", fontsize=10.2, color=GRAY)
    fig.text(.065, .03, "Relative MSE = 1 + energy − twice alignment. f and y are cumulative price changes; all plotted quantities are unitless.", fontsize=9.8)
    fig.subplots_adjust(left=.07, right=.98, bottom=.15, top=.82, wspace=.27)
    output.append(("D1_EEX_horizons_and_decomposition", fig))

    names = ["smp_full", "smp_one_step", "gru_native", "markov_mle", "patchtst_native"]
    matrix = np.array([[selected[name]["by_market"][market]["score"] for name in names] for market in MARKETS])
    fig, ax = plt.subplots(figsize=(9.7, 5.9))
    color_extent = max(float(matrix.max()), 1 / float(matrix.min()))
    normalizer = LogNorm(vmin=1 / color_extent, vmax=color_extent)
    shown = ax.imshow(matrix, cmap="RdBu_r", norm=normalizer, aspect="auto")
    for row in range(5):
        for col in range(5):
            color = "white" if normalizer(matrix[row, col]) > .83 or normalizer(matrix[row, col]) < .12 else "#18212F"
            ax.text(col, row, f"{matrix[row, col]:.4f}", ha="center", va="center", color=color, fontsize=11)
    ax.set_xticks(np.arange(5), ["SMP full", "SMP one-step", "GRU native", "Markov MLE", "PatchTST"])
    ax.set_yticks(np.arange(5), SHORT_MARKETS)
    ax.tick_params(length=0, axis="both", pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(shown, ax=ax, fraction=.035, pad=.04)
    cbar.set_ticks([.5, 1.0, 1.5, 2.0])
    cbar.set_ticklabels(["0.5", "1.0", "1.5", "2.0"])
    cbar.set_label("Cell relative MSE (log color scale)")
    fig.suptitle("D1 2022 development: selected models by series", x=.06, ha="left", fontsize=15, y=.96)
    fig.text(.06, .90, "Neural seed 17; no CI. Mean over h = 1, 5, 20. Log color is centered at RW = 1.", fontsize=10.1, color=GRAY)
    fig.text(.06, .025, "EEX EUA and EU reported are related EU ETS targets, not independent markets.", fontsize=10.2)
    fig.subplots_adjust(left=.17, right=.90, bottom=.14, top=.82)
    output.append(("D1_selected_series_heatmap", fig))
    return output


def explanation(selection, tables, decomp_error):
    full = selection["selected"]["smp_full"]
    one = selection["matched_to_smp"]["smp_one_step"]
    table_lines = "\n".join(f"| \u0060{name}\u0060 | {len(rows)} |" for name, rows in tables.items())
    return f"""# D1 公开聚合结果图表说明

本交付只读取本轮公开的 results/ JSON，保留已生成的全部配置、选择结果与任务记录，不运行模型、选择器或新的统计检验。全部图表对应 **2022 年 development、神经模型 seed 17、无置信区间**；统计参考中的 seed=0 是任务标识，不是额外随机重复。D1 固定门槛未通过，图表不改变该停止决定。

## 图 1：主结果与匹配消融

文件：figures/D1_main_and_ablations.png 与同名 SVG。

左图显示八项主要基线、RW 和 SMP-full。八项基线是原生风险 GRU、旧归一化损失 GRU、带相同辅助似然的 GRU、LSTM、PatchTST、全局已选 ridge、局部验证已选 ridge 和 Markov MLE。它们读取 D1_SELECTION.json 中的既定选择；图中使用固定模型顺序，不按分数重新排序。项目原有参考模型完整保留在表中，由于其历史支持不同，不混入这八项主要基线图。

右图显示 full 加六个 SMP 消融，均取 matched_to_smp 的同一输出收缩系数 alpha={full['value']:g}。每点是五个序列、h=1/5/20 等权的相对价格 MSE，RW=1，数值越低越好。SMP-full={full['score']:.12f}，单步训练消融={one['score']:.12f}。此处没有把消融优于 full 包装成完整递推机制的实证支持。

## 图 2：EEX 各期限风险及精确误差分解

文件：figures/D1_EEX_horizons_and_decomposition.png 与同名 SVG。

左图显示 EEX 同一组 193 个合法预测起点下的 h=1..20 相对 MSE，横轴是观察步，不是日历天。SMP one-step 使用与 full 匹配的 alpha；其他曲线使用既定已选配置。每条曲线来自已公开的全期限聚合结果，不读取或绘制逐日报价或逐起点预测。

右图来自 D1_FAILURE_DECOMPOSITION.json 中已计算的 SMP-full 聚合量。令 f 为已发行的累计价格变化预测，y 为相同起点和期限的实际累计变化。对同一批起点取平均，有

$$
\u005cfrac{{\u005cmathbb{{E}}[(f-y)^2]}}{{\u005cmathbb{{E}}[y^2]}}
=1+
\u005cunderbrace{{\u005cfrac{{\u005cmathbb{{E}}[f^2]}}{{\u005cmathbb{{E}}[y^2]}}}}_{{\u005ctext{{prediction energy}}}}
-
\u005cunderbrace{{\u005cfrac{{2\u005cmathbb{{E}}[fy]}}{{\u005cmathbb{{E}}[y^2]}}}}_{{\u005ctext{{twice alignment}}}}.
$$

预测能量不是预测误差的方差，图中的 twice alignment 项也不是相关系数；两者在图中均除以同期限 RW MSE。较大的正 twice alignment 会抵消部分预测能量。虚线显示 energy−twice alignment，即相对 MSE 超出 RW 的部分；若要还原相对 MSE，需要加上 RW 基准 1。图中只有单位化聚合数值，没有原始价格、尺度或逐点预测。对全部 85 个已选模型×序列记录、1700 个期限单元核对恒等式，最大绝对偏差为 {decomp_error:.3g}。

这一分解是选择完成后的代数描述，不是碳价变化的因果解释，也不能单独证明某个网络组件造成失败。

## 图 3：五个序列的小型热图

文件：figures/D1_selected_series_heatmap.png 与同名 SVG。

各单元是该序列 h=1/5/20 的等权相对 MSE。显示 full、one-step、原生损失 GRU、Markov MLE 与 PatchTST 的既定已选结果。颜色使用以 RW=1 为中心、上下界互为倒数的对数色标，并标注实际数值；接近 1 的小幅差异保持浅色。EEX 与 EU_REPORTED 都涉及 EU ETS，不能当作统计独立市场；本图没有把市场或期限当成独立重复进行显著性检验。

## 完整导出表

所有表均为 UTF-8 CSV，保留 Python float 的可往返精度，未按图中的四位小数截断。空字段代表不适用或原记录缺少该项，不替换为数值 0。每个表保留来源路径和 SHA256，输出清单记录完整输入身份。

| results/tables 文件 | 数据行数（不含表头） |
| --- | ---: |
{table_lines}

33 个全局配置中，ridge_val 是从已有 ridge 任务借用其各序列局部验证选择后的汇总视图，不是第 161 次拟合。17 个 selected 与 12 个 matched 均原样继承公开选择 JSON。三份 market_horizons 表分别完整保留 all-config、selected 和 matched 的五序列×20期限支持。job_execution 保留原任务计划顺序、160 项完成状态、拟合轮数、选中轮数、梯度步数、参数量、支持和时间；job_horizons 保留全部 160×20 的单位化误差。statistical reference 和 RW 没有神经训练 FIT 记录，相关字段保留为空，梯度步数为 0。

## 重算导出与范围

从研究目录执行：

\u0060\u0060\u0060bash
python -B source/export_results.py
\u0060\u0060\u0060

默认写入 results/tables、figures 和 figures/FIGURES_ZH.md，并生成 results/TABLE_FIGURE_EXPORT.json。已存在的交付文件会被拒绝覆盖。若要再次验证导出，在新的输出根目录运行：

\u0060\u0060\u0060bash
python -B source/export_results.py --output-root /absolute/path/new_export
\u0060\u0060\u0060

输入仍只来自 --study-root 指向的公开 JSON。脚本不导入训练、数据处理或选择模块，不打开 CSV 输入、local_only、检查点或确认文件，不重新训练、不补实验、不计算新选择或置信区间。图形依赖 NumPy 和 Matplotlib。公开数据 JSON 的指标恒等式检查属于导出一致性验证，不替代已完成的逐起点独立结果审核。
"""


def export(study_root, output_root):
    reader = PublicResults(study_root)
    all_configs = reader.read("D1_ALL_CONFIGS.json")
    selection = reader.read("D1_SELECTION.json")
    decomp = reader.read("D1_FAILURE_DECOMPOSITION.json")
    started = reader.read("D1/STARTED.json")
    completed = reader.read("D1/COMPLETED.json")
    require(all(obj["protocol_sha256"] == PROTOCOL_SHA for obj in [all_configs, selection, started, completed]),
            "Different frozen study protocol")
    require(selection["year"] == 2022 and selection["stage"] == "D1" and not selection["advance_to_D2"],
            "This fixed D1 export expects the recorded failed gate")
    require(not selection["confirmation_access"] and not decomp["confirmation_access"] and decomp["new_fits"] == 0,
            "Unexpected confirmation or new-fit scope")
    require(decomp["selection_sha256"] == reader.records["D1_SELECTION.json"]["sha256"],
            "Decomposition is not bound to this selection")
    require(len(all_configs["rows"]) == 33 and len(selection["selected"]) == 17
            and len(selection["matched_to_smp"]) == 12, "Fixed aggregate inventory changed")
    require(completed["completed_jobs"] == 160 and completed["failed_jobs"] == 0, "D1 did not complete")
    tables = {}
    for name, rows, source_name in [
        ("all_configs", all_configs["rows"], "D1_ALL_CONFIGS.json"),
        ("selected_models", list(selection["selected"].values()), "D1_SELECTION.json"),
        ("matched_models", list(selection["matched_to_smp"].values()), "D1_SELECTION.json"),
    ]:
        summary, long_rows = aggregate_tables(rows, name, "results/" + source_name,
                                             reader.records[source_name]["sha256"])
        tables["D1_" + name + ".csv"] = summary
        tables["D1_" + name + "_market_horizons.csv"] = long_rows
    jobs, job_horizons, metrics = job_tables(reader, started, completed)
    tables["D1_job_execution.csv"] = jobs
    tables["D1_job_horizons.csv"] = job_horizons
    for row in all_configs["rows"]:
        for cell in row["by_market"].values():
            values = [metrics[job_id]["relative_mse"] for job_id in cell["job_ids"]]
            close(np.mean(values, axis=0), cell["relative_mse"], "Aggregate differs from referenced public jobs")
    decomposition, max_error = decomposition_table(decomp, selection["selected"],
                                                    reader.records["D1_FAILURE_DECOMPOSITION.json"]["sha256"])
    tables["D1_selected_decomposition_horizons.csv"] = decomposition
    expected = [33, 3300, 17, 1700, 12, 1200, 160, 3200, 1700]
    require([len(rows) for rows in tables.values()] == expected, "Export row counts differ")

    output_root = Path(output_root).resolve()
    table_dir, figure_dir = output_root / "results/tables", output_root / "figures"
    figure_names = ["D1_main_and_ablations", "D1_EEX_horizons_and_decomposition", "D1_selected_series_heatmap"]
    note_path = figure_dir / "FIGURES_ZH.md"
    manifest_path = output_root / "results/TABLE_FIGURE_EXPORT.json"
    outputs = [table_dir / name for name in tables] + [
        figure_dir / (name + extension) for name in figure_names for extension in [".png", ".svg"]
    ] + [note_path, manifest_path]
    require(not any(path.exists() or path.is_symlink() for path in outputs), "An export output already exists; use a new --output-root")
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for name, rows in tables.items():
        body, columns = csv_body(rows)
        with (table_dir / name).open("xb") as stream:
            stream.write(body)
        artifacts.append(dict(path="results/tables/" + name, rows=len(rows), columns=columns, bytes=len(body), sha256=sha(body)))
    for name, fig in figures(selection, decomp):
        for extension in [".png", ".svg"]:
            stream = io.BytesIO()
            kwargs = {"metadata": {"Date": None}} if extension == ".svg" else {}
            fig.savefig(stream, format=extension[1:], dpi=200, **kwargs)
            body = stream.getvalue()
            with (figure_dir / (name + extension)).open("xb") as target:
                target.write(body)
            artifacts.append(dict(path="figures/" + name + extension, bytes=len(body), sha256=sha(body)))
        plt.close(fig)
    note = explanation(selection, tables, max_error).encode("utf-8")
    with note_path.open("xb") as stream:
        stream.write(note)
    artifacts.append(dict(path="figures/FIGURES_ZH.md", bytes=len(note), sha256=sha(note)))
    # Only already-read public JSON is rehashed to ensure source immutability.
    for relative, record in reader.records.items():
        require(sha((reader.root / relative).read_bytes()) == record["sha256"], "Public result changed during export")
    manifest = dict(
        status="EXPORTED_PUBLIC_AGGREGATES", created_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(Path(__file__).read_bytes()), protocol_sha256=PROTOCOL_SHA,
        input_root=str(reader.root), output_root=str(output_root), inputs=list(reader.records.values()),
        artifacts=artifacts, input_json_count=len(reader.records), original_result_bytes_unchanged=True,
        table_rows={name: len(rows) for name, rows in tables.items()},
        selection_rerun=False, bootstrap_or_ci_computed=False, neural_seed=17, evaluation_year=2022,
        original_advance_to_D2=selection["advance_to_D2"], figure_count=3, formats=["PNG", "SVG"],
        decomposition_max_abs_error=max_error, csv_inputs_opened=0, local_only_files_opened=0,
        confirmation_files_opened=0, new_fits=0, new_optimizer_steps=0,
        runtime=dict(python=sys.version, numpy=np.__version__, matplotlib=matplotlib.__version__),
        visual_inspection="Pending separate human/model image review after rendering.",
        scope="Existing public JSON aggregates only; fixed original row/selection identities. No private-data numerical replay.")
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=manifest["status"], table_rows=manifest["table_rows"], figure_count=3,
                          input_json_count=len(reader.records), decomposition_max_abs_error=max_error,
                          new_fits=0, csv_inputs_opened=0, local_only_files_opened=0,
                          confirmation_files_opened=0, script_sha256=manifest["script_sha256"]), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    export(args.study_root, args.output_root or args.study_root)


if __name__ == "__main__":
    main()
