"""Publish aggregate-only RRC D1 report, agreement receipt and scientific plots.

Reads completed public aggregate files only. No market CSV, NPZ, source auction
archive, confirmation file, training code or model is opened. Requires a new
artifact directory and a new report path; never changes a selection or gate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import datetime as dt
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform

PROTOCOL_SHA = "404c14867dcc5b43f456075d6b9edd4a5cd75cd99ad943a034e8da7172ef8a70"
METRICS_SHA = "85612d7365034ae85b34cbbf94e04b1bb2f5e2a649980f52c579604d295428e8"
SERIES = ("EEX_EUA_T3PA", "GUANGZHOU", "HUBEI", "SHANGHAI", "EU_REPORTED")
ZH = {"EEX_EUA_T3PA": "EEX EUA 拍卖", "GUANGZHOU": "广州", "HUBEI": "湖北", "SHANGHAI": "上海", "EU_REPORTED": "EU_REPORTED"}
EN = {"EEX_EUA_T3PA": "EEX EUA auction*", "GUANGZHOU": "Guangzhou", "HUBEI": "Hubei", "SHANGHAI": "Shanghai", "EU_REPORTED": "EU reported*"}
FAMILY_ZH = {"rrc": "RRC", "rw": "随机游走", "dsof": "DSOF 同骨干对照", "er": "经验回放 ER", "masked": "成熟部分监督", "self_td": "学生自身 TD",
             "static_gru": "静态 GRU", "static_teacher_student": "静态教师＋学生", "online_gru": "在线 GRU",
             "no_prefix": "无前缀校准", "intercept": "仅截距校准", "output_only": "仅输出修正", "old_target": "错误旧目标诊断"}
FAMILY_EN = {"rrc": "RRC", "rw": "Random walk", "dsof": "DSOF rule", "er": "Experience replay", "masked": "Mature-prefix supervision", "self_td": "Student self-TD",
             "static_gru": "Static GRU", "static_teacher_student": "Static teacher + student", "online_gru": "Online GRU",
             "no_prefix": "No-prefix calibration", "intercept": "Intercept-only calibration", "output_only": "Output-only correction", "old_target": "Old-target sanity control"}
FAMILY_ORDER = ("rrc", "rw", "dsof", "er", "masked", "self_td", "static_gru", "static_teacher_student", "online_gru", "no_prefix", "intercept", "output_only", "old_target")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def exact_mean(values):
    return float(sum((Fraction(str(v)) for v in values), Fraction(0)) / len(values))


def cross_check(root, primary, independent, families):
    errors, scalar_count, exact_count = [], 0, 0
    def same(a, b):
        nonlocal exact_count
        if a != b: raise ValueError("Primary/independent discrete choice or gate mismatch")
        exact_count += 1
    def close(a, b):
        nonlocal scalar_count
        a, b = float(a), float(b)
        if not math.isclose(a, b, rel_tol=5e-12, abs_tol=1e-12):
            raise ValueError("Primary/independent aggregate mismatch")
        errors.append(abs(a-b)); scalar_count += 1
    for name in ("selected_rrc", "selected_baseline", "selected_baseline_family", "selected_no_prefix", "selected_output_only", "selected_dsof", "metric_cells", "configurations", "advancement_year"):
        same(primary[name], independent[name])
    same(primary["selection_only_year"], independent["selection_year"])
    same(sorted(primary["breadth_better_series"]), sorted(independent["breadth_series_won"]))
    same(primary["advance_to_confirmation"], independent["advance"])
    for p, q in (("primary", "primary"), ("breadth", "breadth"), ("mechanism", "mechanism"), ("EEX_relevance", "EEX_relevance"), ("finite_complete", "finite")):
        same(primary["gate_flags"][p], independent["gate_parts"][q])
    for p, q in (("rrc_2023", "rrc_score_2023"), ("baseline_2023", "baseline_score_2023"),
                 ("relative_improvement_2023", "primary_relative_improvement"), ("EEX_rrc_2023", "EEX_rrc_2023"), ("EEX_dsof_2023", "EEX_dsof_2023")):
        close(primary[p], independent[q])
    close(primary["no_prefix_2023"] - primary["rrc_2023"], independent["no_prefix_minus_rrc_2023"])
    close(primary["output_only_2023"] - primary["rrc_2023"], independent["output_only_minus_rrc_2023"])
    primary_families = {r["family"]: r for r in primary["family_comparison"]}
    same(set(primary_families), set(families))
    for family, row in families.items():
        other = primary_families[family]
        same(other["selected_config"], row["selected_config_2022"])
        close(other["selection_2022"], row["selection_score_2022"])
        close(other["advancement_2023"], row["unchanged_config_score_2023"])
    all_root = read_csv(root / "results/D1/ALL_CONFIGS.csv")
    all_ind = read_csv(root / "results/D1/independent_selection/CONFIG_YEAR_SCORES.csv")
    ind_map = {(r["config_id"], int(r["year"])): r for r in all_ind}
    same(len(ind_map), len(all_ind))
    same(len(all_root)*2, len(ind_map))
    for row in all_root:
        for year in (2022, 2023):
            close(row[f"score_{year}"], ind_map[row["config_id"], year]["equal_cell_score"])
        expected_selected = families[row["family"]]["selected_config_2022"] == row["config_id"]
        same(row["selected_within_family"] == "True", expected_selected)
    primary_series = read_csv(root / "results/D1/ALL_MARKET_SCORES.csv")
    independent_series = read_csv(root / "results/D1/independent_selection/CONFIG_YEAR_SERIES_SCORES.csv")
    series_map = {(r["config_id"], int(r["year"]), r["series"]): r for r in independent_series}
    same(len(series_map), len(independent_series)); same(len(primary_series), len(series_map))
    for row in primary_series:
        close(row["score"], series_map[row["config_id"], int(row["year"]), row["market"]]["equal_seed_horizon_score"])
    return {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "passed": True,
            "exact_choice_gate_or_structure_comparisons": exact_count, "numeric_score_comparisons": scalar_count,
            "maximum_absolute_numeric_difference": max(errors), "all_family_choices_agree": True,
            "all_gate_flags_and_stop_decision_agree": True,
            "all_38_config_scores_both_years_and_all_series_compared": True,
            "independence_record": "The independent selector was frozen and executed without reading the primary selector or its output; this comparison is post-execution.",
            "primary_selector_sha256": sha(root / "source/select_rrc.py"),
            "independent_selector_sha256": sha(root / "source/independent_selection.py"),
            "primary_selection_sha256": sha(root / "results/D1/SELECTION_AND_GATE.json"),
            "independent_selection_sha256": sha(root / "results/D1/independent_selection/INDEPENDENT_SELECTION.json"),
            "protocol_sha256": PROTOCOL_SHA, "metrics_sha256": METRICS_SHA,
            "market_fit": False, "confirmation_data_read": False}


def horizon_table(metrics, selection):
    configs = {"rrc": selection["selected_rrc"], "dsof": selection["selected_dsof"],
               "rw": selection["selected_baseline"], "no_prefix": selection["selected_no_prefix"],
               "output_only": selection["selected_output_only"]}
    rows = []
    for name in SERIES:
        for horizon in (1, 5, 20):
            row = {"series": name, "year": 2023, "horizon": horizon}
            for label, config in configs.items():
                values = [r for r in metrics if r["config_id"] == config and r["market"] == name and int(r["year"]) == 2023 and int(r["horizon"]) == horizon]
                if len(values) != 2 or {int(r["seed"]) for r in values} != {17, 29}:
                    raise ValueError("Incomplete seed pair for a plotted cell")
                counts = {int(r["origins"]) for r in values}
                if len(counts) != 1: raise ValueError("Seed origin supports differ")
                row["origins_per_seed"] = counts.pop()
                row[label + "_relative_mse"] = exact_mean([r["relative_mse"] for r in values])
            row["rrc_minus_dsof_relative_mse"] = row["rrc_relative_mse"] - row["dsof_relative_mse"]
            rows.append(row)
    return rows


def figures(output, horizon, families, selection):
    import os
    import tempfile
    plotting_cache = tempfile.TemporaryDirectory(prefix="rrc_figure_cache_")
    os.environ["MPLCONFIGDIR"] = plotting_cache.name
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 10, "svg.fonttype": "none", "svg.hashsalt": "rrc_d1_frozen_20261017",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    outputs = []
    def save(fig, stem, title):
        for suffix in ("svg", "png", "pdf"):
            path = output / f"{stem}.{suffix}"
            metadata = {"Title": title}
            if suffix == "svg": metadata["Date"] = None
            if suffix == "pdf": metadata.update(CreationDate=None, ModDate=None)
            fig.savefig(path, format=suffix, dpi=300, bbox_inches="tight", metadata=metadata)
            outputs.append(path.name)
        plt.close(fig)
    matrix = np.array([[next(r["rrc_relative_mse"] for r in horizon if r["series"] == m and r["horizon"] == h) for h in (1, 5, 20)] for m in SERIES])
    differences = np.array([[next(r["rrc_minus_dsof_relative_mse"]*100 for r in horizon if r["series"] == m and r["horizon"] == h) for h in (1, 5, 20)] for m in SERIES])
    span = max(.02, math.ceil(float(np.abs(matrix-1).max())/.02)*.02)
    delta_span = max(.05, math.ceil(float(np.abs(differences).max())/.05)*.05)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.8), constrained_layout=True)
    for j, (ax, values, center, extent, title) in enumerate(zip(axes, (matrix, differences), (1., 0.), (span, delta_span),
                          ("a  Selected RRC / random walk", "b  Selected RRC minus selected DSOF"))):
        im = ax.imshow(values, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=center-extent, vcenter=center, vmax=center+extent), aspect="auto")
        ax.set_xticks(range(3), ("h = 1", "h = 5", "h = 20"))
        ax.set_yticks(range(5), [EN[m] for m in SERIES] if j == 0 else [""]*5)
        ax.tick_params(length=0, pad=7)
        ax.set_title(title, loc="left", pad=12)
        ax.set_xlabel("Observed quote steps")
        for y in range(5):
            for x in range(3):
                value = values[y, x]
                text = f"{value:.3f}" if j == 0 else f"{value:+.2f}"
                color = "white" if abs(value-center) / extent > .64 else "#151515"
                ax.text(x, y, text, ha="center", va="center", color=color, fontsize=10)
        for edge in ax.spines.values(): edge.set_visible(False)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=.08, shrink=.95, aspect=24)
        cb.set_label("Normalized MSE (1 = random walk)" if j == 0 else "100 × normalized-MSE difference; negative favors RRC", fontsize=9)
        cb.ax.tick_params(labelsize=8)
    fig.suptitle("2023 development evaluation — identities fixed using 2022", fontsize=12)
    fig.supxlabel("Each cell averages two seed-specific MSE / RW-MSE ratios. * Both EU price targets share the EU ETS system.", fontsize=8)
    save(fig, "figure_1_rrc_series_horizons", "2023 RRC by series and horizon; configurations fixed in 2022")
    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    fig.subplots_adjust(left=.30, right=.95, top=.88, bottom=.20)
    scores = [float(families[f]["unchanged_config_score_2023"]) for f in FAMILY_ORDER]
    colors = ["#087E8B" if f == "rrc" else "#5E6572" if f in ("rw", "dsof", "er", "masked", "self_td", "static_gru", "static_teacher_student", "online_gru") else "#AD5D32" for f in FAMILY_ORDER]
    ax.axvspan(.99, 1., color="#EEF1F4", zorder=0)
    ax.axvline(1., color="#6B7280", linewidth=1.1, zorder=1)
    ax.axvline(.99, color="#087E8B", linestyle="--", linewidth=1.2, zorder=1)
    for y, (score, color, family) in enumerate(zip(scores, colors, FAMILY_ORDER)):
        ax.plot(score, y, "D" if family == "rrc" else "o", color=color, markersize=7 if family == "rrc" else 5.5, zorder=3)
        ax.annotate(f"{score:.6f}", (score, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=color)
    ax.set_yticks(range(len(FAMILY_ORDER)), [FAMILY_EN[f] for f in FAMILY_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(min(.99, min(scores))-.004, max(1., max(scores))+.012)
    ax.set_ylim(len(FAMILY_ORDER)-.3, -.7)
    ax.set_xlabel("2023 normalized MSE; lower is better")
    ax.tick_params(axis="y", length=0, pad=8)
    ax.grid(axis="y", alpha=.14)
    for edge in ("top", "right", "left"): ax.spines[edge].set_visible(False)
    ax.spines["bottom"].set_color("#B0B0B0")
    fig.suptitle("Fixed family choices evaluated on 2023", fontsize=13, y=.975)
    fig.text(.30, .915, "Every configuration and the strongest baseline were selected on 2022.\nRRC gain over the selected baseline: 0.817%; required: at least 1%.", fontsize=9)
    legend = [Line2D([0], [0], color="#087E8B", linestyle="--", label="Required primary score ≤ 0.99"),
              Line2D([0], [0], color="#6B7280", label="Selected random-walk baseline = 1")]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.30, .112), frameon=False, fontsize=8, borderaxespad=0)
    fig.text(.30, .016, "Equal weights: 5 series × 2 seeds × 3 horizons. No IID market confidence interval.\nOld-target is a sanity control only. The full advancement gate fails.", fontsize=8)
    save(fig, "figure_2_rrc_frozen_family_comparison", "2023 family comparison under 2022-only selection")
    plotting_cache.cleanup()
    return outputs, {"python": platform.python_version(), **{k: importlib.metadata.version(k) for k in ("matplotlib", "numpy", "pillow")}}, {"rrc_heatmap_symmetric_span_around_one": span, "difference_heatmap_symmetric_span_percentage_points": delta_span}


def report_text(root, output, family, horizons, selection, agreement, completed, matched):
    relative = output.relative_to(root).as_posix()
    rrc = selection["rrc_score_2023"]
    rows = []
    for name in SERIES:
        cells = [r for r in horizons if r["series"] == name]
        value = {int(r["horizon"]): r for r in cells}
        avg = exact_mean([r["rrc_relative_mse"] for r in cells])
        dsof = exact_mean([r["dsof_relative_mse"] for r in cells])
        rows.append(f"| {ZH[name]} | {value[1]['origins_per_seed']} | {value[1]['rrc_relative_mse']:.6f} | {value[5]['rrc_relative_mse']:.6f} | {value[20]['rrc_relative_mse']:.6f} | {avg:.6f} | {dsof:.6f} |")
    family_rows = [f"| {FAMILY_ZH[f]} | `{family[f]['selected_config_2022']}` | {float(family[f]['selection_score_2022']):.8f} | {float(family[f]['unchanged_config_score_2023']):.8f} |" for f in FAMILY_ORDER]
    descriptions = {"output_same_calibrator": "同模式、同 α 的仅输出修正", "dsof_lr_matched": "同学生学习率的 DSOF", "full_vs_linear_alpha_lr_matched": "同 α、同学习率的线性校准", "no_prefix_alpha_lr_matched": "同 α、同学习率的无前缀校准", "intercept_alpha_lr_matched": "同 α、同学习率的仅截距", "old_target_sanity_only": "错误旧目标诊断（α 不匹配）"}
    matched_rows = [f"| {descriptions[r['comparison']]} | `{r['reference_config']}` | {float(r['reference_score']):.8f} | {float(r['reference_minus_rrc']):+.8f} |" for r in matched]
    return rf"""# RRC 冻结阶段结果：未通过晋级，保留新确认区

**本轮固定候选停止。** 按 2022 年选择配置、在 2023 年评价的预定程序，RRC 的总体相对 MSE 为 **{rrc:.10f}**，随机游走为 1，改善 **{selection['primary_relative_improvement']*100:.6f}%**，未达到至少 1% 的门槛。EEX 子序列相对 MSE 为 **{selection['EEX_rrc_2023']:.10f}**，仍高于随机游走。两个独立选择实现的所有选择与 gate 一致，逐预测指标核验通过。EEX 2024、2025 年没有用于模型选择、机制成绩检查或确认评分。[独立选择结果](results/D1/independent_selection/INDEPENDENT_SELECTION.json) · [失败决策证据]({relative}/RRC_STOP_DECISION.json)

## 1. 本轮评价对象与冻结边界

本轮评价的是同骨干在线教师／学生系统中的 RRC 修订残差校准候选。RRC 主族固定为 full/linear 两种校准形式、两种正则参数、两种学生学习率，共 8 个设置；整体含 38 个配置，覆盖随机游走、静态/在线 GRU、ER、成熟部分监督、DSOF 学习路径、自身 TD，以及无前缀、截距、仅输出修正和错误旧目标等对照。表中的 DSOF 是 D1 共同骨干中的学习路径适配对照，不能直接解释为完整原作者方案的跨数据集性能结论。[D1 协议](provenance/D1_PROTOCOL.json) · [冻结实现](source/rrc_core.py)

2021 年用于在线学习预热。各模型族用 **2022 年**的等权分数选配置；指定基线族也先各自在 2022 年选，再仅用 2022 年选出总体最强基线。将所有选定身份原样用于 **2023 年**，禁止再按 2023 年挑选学习率、校准形式或基线族。2023 年承担本轮机制开发和晋级筛选，因此已经是开发证据，不能在后续论文中重新标为从未用于开发的最终独立测试集。[完整配置表](results/D1/ALL_CONFIGS.csv)

每个系列、年份、期限、种子先计算：

$$r_{{m,y,h,s}} = \frac{{\operatorname{{MSE}}_{{m,y,h,s}}}}{{\operatorname{{MSE}}^{{RW}}_{{m,y,h,s}}}}.$$

再对 5 条价格序列、2 个种子、h=1/5/20 三个期限等权平均，每个配置每年 30 格。先对每个种子评分，再平均分数；不先平均预测。所有期限使用共同的、完整 H20 且目标仍在同一年内的起点。h 表示未来观察到的报价记录步数，不是固定自然日。跨货币/市场的原价 MSE 不直接混合。[独立指标程序](source/independent_metric_audit.py)

## 2. 新增 EEX 数据及仍封存的确认区

本轮新增来源为 [EEX 官方 2012–2025 拍卖报表归档](https://www.eex.com/en/market-data/market-data-hub/environmentals/eex-eua-primary-auction-spot-download)。主序列只保留 T3PA EUA 成功拍卖结果，价格为同日按拍卖量加权的一级拍卖清算价；它不是二级市场连续收盘价，也不是期货结算价。旧无 Status 报表按追加冻结的正值、收入恒等式和无取消文字规则标为推断完成，而不伪造原表明确状态。[数据方法与许可](data_reproduction/EEX_DATA_METHODS_ZH.md)

| 分区 | 实际日期 | 日价格记录 | 原拍卖场次 | 本轮用途 |
|---|---|---:|---:|---|
| EEX 开发 | 2012-10-26—2023-12-18 | 2,255 | 2,269 | D1 开发与评分 |
| EEX 2024 | 2024-01-15—2024-12-16 | 213 | 213 | 仍封存，未用于模型评价 |
| EEX 2025 | 2025-01-07—2025-12-15 | 213 | 213 | 仍封存，未用于模型评价 |
| 新增合计 | — | **2,681** | **2,695** | 开发 2,255 日＋确认 426 日 |

D1 的 EEX 输入除历史归一化价格增量、已发生的日期间隔外，还使用过去已经公布的拍卖量、覆盖倍数、投标人数。未来实现的拍卖结果与未来实际间隔未作为已知输入。源文件保存了更完整字段，但保存字段不等于所有字段进入模型。[冻结特征实现](source/rrc_core.py)

数据准备者曾为按预先固定边界拆分、核验来源而机械解析 2024/2025 单元格；这些价格未向根研究者展示，也未用于模型评分。这里的“仍封存”指保持研究使用隔离，并非声称数据从未被计算机下载或解析。EEX 原始与派生报价均留在本地，公开包只含代码、元数据和聚合研究结果；不把其他来源的 CC BY 许可套到 EEX。[来源审计](data_reproduction/provenance/EEX_PREPARATION_AUDIT.json)

## 3. 五条价格序列 × 三个期限的 2023 年结果

以下所有 RRC 数值均来自同一个由 2022 年选出的 `full_a0p01_lr0p0003`。DSOF 同样固定为 2022 年选出的 `dsof_lr0p0003`。每个 h 格是两个种子的相对 MSE 平均；RW 每格等于 1。数值越低越好，N 是每个种子的完整 H20 起点数。[可复算表]({relative}/selected_2023_series_horizon.csv)

| 价格序列 | N/种子 | RRC h1 | RRC h5 | RRC h20 | RRC 三期限平均 | 选定 DSOF 三期限平均 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

![2023年系列与期限结果]({relative}/figure_1_rrc_series_horizons.png)

**图 1.** 左：选定 RRC 的相对 MSE，1 为随机游走。右：选定 RRC 减选定 DSOF 的相对 MSE 差乘以 100，负值有利于 RRC。每格先按种子计算比率，再平均；未将种子当作独立市场构造误差条。EEX 与 EU_REPORTED 均连接同一 EU ETS 体系。矢量版：[SVG]({relative}/figure_1_rrc_series_horizons.svg)、[PDF]({relative}/figure_1_rrc_series_horizons.pdf)。

湖北、上海、EU_REPORTED 的系列平均优于随机游走，EEX 与广州没有。不能把“3/5 胜出”解释成五个独立市场中的显著多数；EEX 和 EU_REPORTED 的依赖关系必须保留。

## 4. 强对照与各族固定选择

各行配置均由 2022 年独立选择。总体基线也由 2022 年决定，为随机游走；不会因看见 2023 年其他配置更低而替换它。`old_target` 仅是错误目标的合理性检查，不计为创新证据。

| 模型族 | 2022 年选定配置 | 2022 选择分数 | 原配置 2023 分数 |
|---|---|---:|---:|
{chr(10).join(family_rows)}

![固定模型族比较]({relative}/figure_2_rrc_frozen_family_comparison.png)

**图 2.** 2022 年固定的各族配置在 2023 年的总体分数。实线为选定 RW 基线 1；虚线为本轮至少改善 1% 对应的 0.99。RRC 位于 0.99 与 1 之间，虽略低于 RW，仍未达到原定主条件。图中没有显著性误差条；该门槛是事前研究推进规则。矢量版：[SVG]({relative}/figure_2_rrc_frozen_family_comparison.svg)、[PDF]({relative}/figure_2_rrc_frozen_family_comparison.pdf)。

## 5. 匹配消融揭示的机制限制

晋级协议中的机制比较使用各消融族在 2022 年独立选出的配置。按这个规定，RRC 比选定无前缀和选定仅输出修正分别低 {selection['no_prefix_minus_rrc_2023']:.10f}、{selection['output_only_minus_rrc_2023']:.10f}，均超过 0.0001；这一项通过。

但是，**同 α、同学生学习率的无前缀版本和线性版本在 2023 年略优于选中的 full RRC**。因此，不能把前述通过扩大成前缀信息或交互项已得到稳健消融支持。以下固定匹配比较完整保留，正分差表示对照更差、RRC 更好，负分差表示对照更好。[完整匹配表](results/D1/independent_selection/MATCHED_ABLATIONS.csv)

| 与选中 RRC 的比较 | 对照配置 | 对照 2023 分数 | 对照 − RRC |
|---|---|---:|---:|
{chr(10).join(matched_rows)}

选中 RRC 的分数为 {rrc:.10f}。严格匹配无前缀的优势约 0.000202、严格匹配线性的优势约 0.000149，幅度很小，也不是独立确认的效果。错误旧目标只有一个 α=0.1，与选中 RRC 的 α=0.01 不匹配；它的结果仅保留为诊断。未围绕这些已观察差异追加超参数或重新训练。

## 6. 晋级失败决定与两套独立实现核对

| 原定条件 | 结果 | 判定 |
|---|---|---|
| 相对 2022 年选定总体基线改善至少 1%，且分数 < 1 | 改善 {selection['primary_relative_improvement']*100:.6f}%，低于 1% | 失败 |
| 至少 3/5 条价格序列优于该基线 | 湖北、上海、EU_REPORTED，共 3 条 | 通过 |
| 比 2022 年分别选定的无前缀、仅输出修正均好至少 0.0001 | 两个分差均满足 | 通过 |
| EEX 分数 < 1 且低于选定 DSOF | {selection['EEX_rrc_2023']:.10f}，仍 > 1；虽略低于 DSOF | 失败 |
| 固定网格完整，所有结果有限 | 38 配置、2,280 格，无失败任务 | 通过 |

总体为 **停止该固定候选，不打开 EEX 确认区**。不把 1% 改成 0.8%，不删除 EEX 条件，不按 2023 年回选更有利的参数，也不把机制子条件通过改写为总体通过。[决策证据]({relative}/RRC_STOP_DECISION.json)

根研究者与独立审核者分别实现选择规则，独立版本在读取任何 D1 数值、根选择源码或根选择输出前冻结。执行后逐项比较：{agreement['exact_choice_gate_or_structure_comparisons']} 项选择/gate/结构相等；{agreement['numeric_score_comparisons']} 项数值比较通过，最大绝对差 {agreement['maximum_absolute_numeric_difference']:.3g}，仅为算术尾差。全部 13 个族内选择、总体基线、所有 gate、38 个配置的两年及各系列分数一致。[交叉核对回执]({relative}/SELECTOR_AGREEMENT.json)

另一个不导入训练器或 PyTorch 的实现，从批准开发 CSV 和全部 380 份预测 NPZ 复算 2,280 个格，对局部/总体表及每个配置分数完成 32,300 次比较，全部通过。这个审核确认分数计算与预测文件一致，不增加关于创新或可发表性的统计保证。[逐预测审核](results/D1/INDEPENDENT_METRIC_AUDIT.json)

## 7. 实际执行数量与报告边界

| 执行类别 | D1 实际数量 |
|---|---:|
| 系列×种子任务 | {completed['jobs']} |
| 联合教师/学生离线拟合（两种 shrink） | {completed['joint_pretraining_fits']} |
| 选定教师在线流 | {completed['teacher_online_streams']} |
| 学生在线流 | {completed['student_online_streams']} |
| 岭校准在线流 | {completed['calibration_online_streams']} |
| 保存的配置预测系列 | 380 |
| 年份×期限×种子的指标格 | {completed['metric_rows']} |
| 失败任务 | 0 |
| EEX 确认拟合 | 0 |

380 是 38 配置×10 任务的预测系列数，不能写成 380 次完整独立训练；多个配置共享教师路径和选定初始状态。计时运行与原作者实现锚定运行在主协议中另列，不混入 D1 计数。[完成回执](results/D1/COMPLETE.json)

本报告只交付冻结 RRC 阶段的结果。2023 年已经用于机制开发筛选，两个种子、三个期限和重叠标签不是独立重复市场；EEX/EU 相依，未采用 iid 跨市场置信区间或宣称显著性。现有结果没有证明达到发表标准，也没有把任何备用方案说成已经完成了实验。

## 8. 图表和报告复现

`source/make_rrc_results.py` 只读取 D1 公开聚合表、选择结果、完成回执和协议；不打开原始价格、逐点预测、源 ZIP 或确认文件。图 1、图 2 提供 SVG、300 dpi PNG 和嵌入字体 PDF；图中只含聚合误差。运行时记录版本与全部输入哈希，输出必须为新路径：

```bash
python3 -B source/make_rrc_results.py --study-root . --output results/D1/new_report_artifacts --report RESULTS_RRC_REPRODUCED_ZH.md
```

本次绘图环境为 Python 3.12.14、matplotlib 3.10.8、NumPy 2.3.5、Pillow 12.3.0；这是聚合绘图环境，模型训练的依赖版本仍以 D1 运行记录为准。没有为绘图重新安装或更改训练环境。[制图回执]({relative}/FIGURES_RECEIPT.json)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    root = args.study_root.resolve()
    output = args.output.resolve()
    report = args.report.resolve()
    if output.exists() or output.is_symlink() or report.exists():
        raise FileExistsError("New report/artifact paths required; no overwrite")
    if not output.is_relative_to(root) or not report.is_relative_to(root):
        raise ValueError("Keep report and figures under the chosen study root for portable relative links")
    if sha(root / "provenance/D1_PROTOCOL.json") != PROTOCOL_SHA or sha(root / "results/D1/ALL_METRICS.csv") != METRICS_SHA:
        raise ValueError("Frozen protocol/aggregate identity changed")
    primary = read_json(root / "results/D1/SELECTION_AND_GATE.json")
    independent = read_json(root / "results/D1/independent_selection/INDEPENDENT_SELECTION.json")
    completed = read_json(root / "results/D1/COMPLETE.json")
    family_rows = read_csv(root / "results/D1/independent_selection/FAMILY_SELECTION_2022.csv")
    families = {r["family"]: r for r in family_rows}
    agreement = cross_check(root, primary, independent, families)
    if primary["advance_to_confirmation"] or independent["advance"] or completed["failures"]:
        raise ValueError("This report requires the verified stopped, complete RRC stage")
    horizon = horizon_table(read_csv(root / "results/D1/ALL_METRICS.csv"), independent)
    matched = [r for r in read_csv(root / "results/D1/independent_selection/MATCHED_ABLATIONS.csv")
               if r["rrc_config"] == independent["selected_rrc"] and int(r["year"]) == 2023]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "SELECTOR_AGREEMENT.json", agreement)
    decision = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "decision": "STOP_FIXED_RRC_CANDIDATE", "advance_to_confirmation": False,
                "failed_prespecified_conditions": [k for k, v in independent["gate_parts"].items() if not v],
                "gate_parts": independent["gate_parts"], "thresholds_unchanged": True,
                "selection_year": 2022, "advancement_year": 2023,
                "selected_rrc": independent["selected_rrc"], "selected_baseline": independent["selected_baseline"],
                "primary_required_relative_improvement": .01,
                "primary_observed_relative_improvement": independent["primary_relative_improvement"],
                "EEX_required_score_below": 1., "EEX_observed_rrc_score": independent["EEX_rrc_2023"],
                "2023_used_for_development_advancement": True,
                "EEX_2024_2025_model_selection_or_evaluation_opened": False,
                "source_preparation_mechanical_confirmation_parse_previously_performed": True,
                "source_mechanical_parse_did_not_compute_or_disclose_confirmation_scores": True,
                "no_posthoc_threshold_change_or_reselection": True,
                "matched_ablation_limit": "At matched alpha and student learning rate, no-prefix and additive-linear controls are slightly better on 2023. The prespecified selected-family mechanism gate does not establish robust prefix or interaction value.",
                "series_dependence": "EEX and EU_REPORTED share EU ETS; five series are not five independent markets",
                "publication_ready_demonstrated": False, "new_backup_model_experiment_claimed": False,
                "protocol_sha256": PROTOCOL_SHA, "metrics_sha256": METRICS_SHA,
                "selector_agreement_sha256": sha(output / "SELECTOR_AGREEMENT.json"),
                "no_new_market_fit": True, "no_new_confirmation_read": True}
    write_json(output / "RRC_STOP_DECISION.json", decision)
    write_csv(output / "selected_2023_series_horizon.csv", horizon)
    write_csv(output / "family_selected_2022_evaluated_2023.csv", family_rows)
    write_csv(output / "matched_ablations_selected_rrc.csv", matched)
    plot_files, environment, rendering = figures(output, horizon, families, independent)
    text = report_text(root, output, families, horizon, independent, agreement, completed, matched)
    with report.open("x", encoding="utf-8") as stream: stream.write(text)
    receipt = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "environment": environment,
               "plot_code_sha256": sha(Path(__file__)), "rendering": rendering,
               "public_aggregate_inputs_only": True, "raw_price_or_forecast_files_read": False,
               "confirmation_files_read": False, "new_market_fits": 0,
               "plot_files": [{"path": name, "bytes": (output/name).stat().st_size, "sha256": sha(output/name)} for name in plot_files],
               "protocol_sha256": PROTOCOL_SHA, "metrics_sha256": METRICS_SHA,
               "report_sha256": sha(report), "selector_agreement_passed": True}
    write_json(output / "FIGURES_RECEIPT.json", receipt)
    public = [Path(__file__).resolve(), report, *sorted(output.iterdir())]
    manifest = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "files": [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in public],
                "scope": "RRC stopped-stage report, scientific aggregate figures, source and consistency evidence; no quotations or per-origin predictions"}
    write_json(output / "REPORT_DELIVERY_MANIFEST.json", manifest)
    print(json.dumps({"report": str(report), "artifacts": str(output), "selector_agreement": agreement,
                      "stop_decision": decision["decision"], "figures": plot_files,
                      "delivery_manifest_sha256": sha(output / "REPORT_DELIVERY_MANIFEST.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
