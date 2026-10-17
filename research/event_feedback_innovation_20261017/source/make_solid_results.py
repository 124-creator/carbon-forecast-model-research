"""Publish aggregate-only SOLID B1 results and post-execution selector agreement.

No market CSV, forecast NPZ, fitted state or confirmation data are opened.
This reporting code does not change the frozen independent selector or runner.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

PROTOCOL_SHA256 = "b44357f30ce5c7f50f2023ebb0fd3fcc17e1598038724738b0b64b56c7ca76fd"
METRIC_SHA256 = "b0461f48d8de33ff3ba16b2eef80d0fa10c13275ab1e664b98831d7b370c2adc"
INDEPENDENT_SOURCE_SHA256 = "9460620d200245e6c7976bd2576abb93fc5743c80e9cc7ab6599595a836362b6"
ROOT_SOURCE_SHA256 = "dd6596ddb98acf713e82987367dfdd6b8b0f9c694527ed537b31b082ddc91e30"
MARKETS = ("EEX_EUA_T3PA", "GUANGZHOU", "HUBEI", "SHANGHAI", "EU_REPORTED")
LABELS = {"EEX_EUA_T3PA": "EEX EUA 一级拍卖", "GUANGZHOU": "广州作者报价", "HUBEI": "湖北作者报价",
          "SHANGHAI": "上海研究数据", "EU_REPORTED": "EU_REPORTED 作者报价"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, obj):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])


def build(root, output, report):
    if output.exists() or output.is_symlink() or report.exists() or report.is_symlink():
        raise FileExistsError("Reporting destination exists; refusing overwrite")
    checks = {"provenance/SOLID_B1_PROTOCOL.json": PROTOCOL_SHA256,
              "results/SOLID_B1/ALL_METRICS.csv": METRIC_SHA256,
              "source/independent_solid_b1.py": INDEPENDENT_SOURCE_SHA256,
              "source/select_solid_reference.py": ROOT_SOURCE_SHA256}
    for rel, expected in checks.items():
        if sha(root / rel) != expected:
            raise ValueError("Frozen reporting input identity differs: " + rel)
    base = root / "results/SOLID_B1"
    independent = base / "independent_audit"
    own = read_json(independent / "INDEPENDENT_SELECTION.json")
    audit = read_json(independent / "INDEPENDENT_METRIC_AUDIT.json")
    primary = read_json(base / "root_selection/SELECTION.json")
    primary_configs = read_csv(base / "root_selection/ALL_CONFIGS.csv")
    overall = read_csv(independent / "CONFIG_YEAR_SCORES.csv")
    series = read_csv(independent / "CONFIG_YEAR_SERIES_SCORES.csv")
    horizons = read_csv(independent / "CONFIG_YEAR_SERIES_HORIZON_SCORES.csv")
    matched = read_csv(independent / "MATCHED_DISTANCE_RECENCY.csv")
    cfg = read_json(root / "provenance/SOLID_B1_PROTOCOL.json")
    complete = read_json(base / "COMPLETE.json")
    started = read_json(base / "STARTED.json")
    if not audit["passed"] or complete["failures"] or audit["new_confirmation_data_read"]:
        raise ValueError("Formal execution/audit did not pass at the required information boundary")
    global_scores = {(r["config_id"], int(r["year"])): float(r["equal_series_seed_horizon_score"]) for r in overall}
    series_scores = {(r["config_id"], int(r["year"]), r["market"]): float(r["equal_seed_horizon_score"]) for r in series}
    horizon_scores = {(r["config_id"], int(r["year"]), r["market"], int(r["horizon"])):
                      float(r["mean_seed_relative_mse"]) for r in horizons}
    exact, numeric = [], []

    def eq(label, left, right):
        if left != right:
            raise ValueError("Independent selector mismatch: " + label)
        exact.append(label)

    def close(label, left, right):
        left, right = float(left), float(right)
        if not math.isclose(left, right, rel_tol=5e-13, abs_tol=5e-15):
            raise ValueError("Independent aggregate mismatch: " + label)
        numeric.append({"field": label, "primary": left, "independent": right, "absolute_difference": abs(left-right)})

    for family in ("solid_distance", "solid_recency"):
        eq("selected." + family, primary["selected"][family], own["selected"][family])
    for field in ("selection_year", "evaluation_year", "protocol_sha256"):
        eq(field, primary[field], own[field])
    eq("metric_table_identity", primary["metric_sha256"], own["B1_metric_table_sha256"])
    eq("no_new_algorithm_claim", primary["new_algorithm_claim"], own["new_algorithm_claim"])
    eq("confirmation_not_read", primary["confirmation_data_read"], own["new_confirmation_data_read"])
    eq("no_confirmation_advancement", primary["advance_to_confirmation"], own["confirmation_opening_authorized"])
    own_meta = {r["config_id"]: r for r in overall if int(r["year"]) == 2022}
    eq("all_configuration_identities", sorted(r["config_id"] for r in primary_configs), sorted(own_meta))
    for row in primary_configs:
        c = row["config_id"]
        eq(c + ".family", row["family"], own_meta[c]["family"])
        eq(c + ".selected_2022", row["selected_2022"], own_meta[c]["selected_in_family_on_2022"])
        for year in (2022, 2023):
            close(f"all_configs.{c}.{year}", row[f"score_{year}"], global_scores[c, year])
    expected_comparison = {**own["selected"], "static_gru": "static_gru", "rw": "rw"}
    eq("comparison_families", sorted(r["family"] for r in primary["family_comparison"]), sorted(expected_comparison))
    for row in primary["family_comparison"]:
        c = row["config_id"]
        eq("comparison." + row["family"], c, expected_comparison[row["family"]])
        for year in (2022, 2023):
            close(f"family_comparison.{c}.{year}", row[f"score_{year}"], global_scores[c, year])
    expected_series_keys = {(c, m) for c in expected_comparison.values() for m in MARKETS}
    eq("selected_series_cube", sorted((r["config_id"], r["market"]) for r in primary["selected_series_2023"]), sorted(expected_series_keys))
    for row in primary["selected_series_2023"]:
        close("series." + row["config_id"] + "." + row["market"], row["score"], series_scores[row["config_id"], 2023, row["market"]])
    agreement = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "passed": True,
        "exact_choice_structure_scope_comparisons": len(exact), "numeric_score_comparisons": len(numeric),
        "maximum_absolute_numeric_difference": max(v["absolute_difference"] for v in numeric),
        "selected": own["selected"], "exact_comparison_fields": exact, "numeric_comparisons": numeric,
        "primary_selector_sha256": ROOT_SOURCE_SHA256, "independent_selector_sha256": INDEPENDENT_SOURCE_SHA256,
        "primary_selection_sha256": sha(base / "root_selection/SELECTION.json"),
        "independent_selection_sha256": sha(independent / "INDEPENDENT_SELECTION.json"),
        "independence": "Independent implementation frozen before its first B1 value read and executed before opening primary selector/output. Cross-comparison performed only afterwards.",
        "numerical_gate_in_protocol": False, "market_fit": False, "confirmation_data_read": False}

    distance, recency = own["selected"]["solid_distance"], own["selected"]["solid_recency"]
    selected_evaluations = {r["family"]: r for r in own["selected_family_evaluations"]}
    d, r = selected_evaluations["solid_distance"], selected_evaluations["solid_recency"]
    decision = {"created_utc": agreement["created_utc"], "decision": "CLOSE_FIXED_EXISTING_METHOD_REFERENCE_B1",
        "formal_budget_completed": True, "performance_directed_extra_grid": False, "new_algorithm_claim": False,
        "publication_readiness_established": False, "open_EEX_confirmation": False, "reverse_RRC_stop": False,
        "numerical_advancement_gate_was_preregistered_for_B1": False,
        "selected_on_2022": own["selected"], "unchanged_config_score_2023": {
            "solid_distance": global_scores[distance, 2023], "solid_recency": global_scores[recency, 2023],
            "static_gru": global_scores["static_gru", 2023], "rw": 1.0},
        "all_eight_2023_configuration_scores_above_rw": all(global_scores[s["id"], 2023] > 1 for s in cfg["configs"]),
        "all_four_matched_2023_pairs_favor_recency": all(float(m["recency_minus_distance"]) < 0 for m in matched if int(m["year"]) == 2023),
        "interpretation": "The fixed existing-method reference completed correctly. The selected configurations did not improve overall MSE versus RW; tiny static-GRU differences and EEX subseries gains are development diagnostics, not significant or novel-method evidence.",
        "sealed_scope": "EEX 2024 and 2025 remain unused for model selection, scores or adaptation. Mechanical source preparation already occurred under the predownload source scope; this is not a claim that no software ever parsed those source cells.",
        "development_exposure": "2022/2023 were already used during RRC development before this exploratory backup. Previously evaluated Shanghai confirmation is not reclassified as untouched.",
        "no_independent_market_assumption": "EEX and EU_REPORTED share EU ETS exposure; seeds/horizons/overlapping targets are not independent market replications.",
        "protocol_sha256": PROTOCOL_SHA256, "B1_metric_table_sha256": METRIC_SHA256,
        "independent_audit_sha256": sha(independent / "INDEPENDENT_METRIC_AUDIT.json")}

    config_rows = []
    for spec in cfg["configs"]:
        c = spec["id"]
        selected = "**族内选中**" if c in own["selected"].values() else ""
        execution = "等价派生" if spec.get("equivalent_prediction_from") else "实际执行"
        config_rows.append([f"`{c}`", execution, f"{global_scores[c,2022]:.10f}", f"{global_scores[c,2023]:.10f}", selected])
    for c, label in (("static_gru", "固定同状态对照"), ("rw", "固定对照")):
        config_rows.append([f"`{c}`", label, f"{global_scores[c,2022]:.10f}", f"{global_scores[c,2023]:.10f}", ""])
    config_table = markdown_table(["配置", "计算身份", "2022 选择分数", "2023 原配置分数", "选择"], config_rows)
    series_rows = []
    for m in MARKETS:
        n = next(row["origins_per_seed"] for row in series if row["config_id"] == distance and row["market"] == m and int(row["year"]) == 2023)
        series_rows.append([LABELS[m], n, f"{series_scores[distance,2023,m]:.8f}", f"{series_scores[recency,2023,m]:.8f}",
                            f"{series_scores['static_gru',2023,m]:.8f}", "1.00000000"])
    series_table = markdown_table(["序列", "2023 起点/seed", "选中 distance", "选中 recency", "同状态 static GRU", "RW"], series_rows)
    horizon_rows = []
    for m in MARKETS:
        for c, label in ((distance, "distance"), (recency, "recency"), ("static_gru", "static GRU")):
            horizon_rows.append([LABELS[m], label] + [f"{horizon_scores[c,2023,m,h]:.8f}" for h in (1,5,20)])
    horizon_table = markdown_table(["序列", "固定配置", "h=1", "h=5", "h=20"], horizon_rows)
    matched_rows = []
    for m in matched:
        matched_rows.append([m["context_samples"], m["learning_rate"], m["year"], f"{float(m['distance_score']):.10f}",
                             f"{float(m['recency_score']):.10f}", f"{float(m['recency_minus_distance']):+.10f}"])
    matched_table = markdown_table(["候选池 C", "学习率", "年份", "distance", "recency", "recency−distance"], matched_rows)
    duration = (dt.datetime.fromisoformat(complete["finished_utc"]) - dt.datetime.fromisoformat(started["started_utc"])).total_seconds()
    report_text = rf"""# SOLID B1 结果：既有方法参考完成，未证明总体预测优势

**按 2022 年分别选择 distance 和 recency 配置、仅在 2023 年评价同一身份，两个族的总体相对 MSE 为 {global_scores[distance,2023]:.10f}、{global_scores[recency,2023]:.10f}；随机游走为 1，同状态静态 GRU 为 {global_scores['static_gru',2023]:.10f}。** 两者相对静态 GRU 的改善只有 {100*d['relative_improvement_vs_static_gru_2023']:.6f}% 和 {100*r['relative_improvement_vs_static_gru_2023']:.6f}%，相对随机游走分别差 {100*(global_scores[distance,2023]-1):.6f}% 和 {100*(global_scores[recency,2023]-1):.6f}%。逐预测复算、等价派生核验和两个选择实现的交叉比较全部通过；执行正确没有转化为总体优于随机游走的证据。[独立选择](results/SOLID_B1/independent_audit/INDEPENDENT_SELECTION.json) · [完整性及指标审核](results/SOLID_B1/independent_audit/INDEPENDENT_METRIC_AUDIT.json)

这是 RRC 已停止之后，对**已发表 SOLID 方法**的有限、后验适配性研究。本轮不提出新算法，不恢复 RRC，不以小幅差异宣称统计显著或达到发表标准，不依成绩追加配置，也不打开 EEX 2024/2025 确认区。B1 协议没有新增数值晋级 gate，本文也没有事后另造一个。[固定协议](provenance/SOLID_B1_PROTOCOL.json) · [完成决策证据](results/SOLID_B1/report_artifacts/B1_REFERENCE_DECISION.json)

## 1. 已发表方法身份与本次执行范围

来源为 Chen 等的 *Calibration of Time-Series Forecasting: Detecting and Adapting Context-Driven Distribution Shift*，KDD 2024。固定作者仓库为 [HALF111/calibration_CDS，commit 3c74e5f8](https://github.com/HALF111/calibration_CDS/tree/3c74e5f8b3c921a2e7cda30f36117c295bf8c112)，MIT。归档 [README](backup_review/author_solid/README.md) 给出论文与 SOLID 执行入口；本次实际调用的 `Exp_Main.select_with_distance` 算法函数 AST 与固定作者文件一致。[作者执行前置审核](backup_review/SOLID_EXECUTION_AUDIT_ZH.md)

本次骨干为 D1 已保存的离线 GRU teacher：**没有新增骨干预训练**。数据装载、因果尺度、GRU 前向、内存 checkpoint 和诊断输出由本任务桥接。每个 query 重置到同一个既定离线模型，冻结编码器，只调整线性预测头。作者非 AMP 路径在共同初始头部计算 10 个成熟邻居的梯度，求和后做一次更新；没有按邻居数平均，也没有逐邻居顺序更新。本轮没有重现作者原 benchmark 的全部模型与数据处理，没有运行 Reconditionor 去重新选择市场或决定哪个 query 适应。[适配源码](source/solid_author_adapter.py) · [数学范围与局限](backup_review/SOLID_THEORETICAL_REVIEW_ZH.md)

对当前起点 q，候选起点从 max(60, q−20−C+1) 连续取到 q−20，完整标签均已成熟。C 为 128 或 512，固定邻居数 k=10，学习率为 0.001/0.01。distance 使用作者输入距离检索，recency 使用最近成熟起点；period=1 保留作者金融/未知数据的默认规则。查询的未来真实标签未传给作者适配函数，未来真实价格、拍卖结果和未来实现的日期间隔不进入当前特征。[正式运行器](source/run_solid_reference.py) · [冻结的独立因果/梯度审核](backup_review/SOLID_EXECUTION_INDEPENDENT_AUDIT.json)

## 2. 选择、评分与观察单位

两个族各有四个预声明配置，**只按 2022 年分数各自选优**；完全并列时按配置名升序。2023 年仅评价固定身份，所有未选配置也完整披露。recency 的 C128/C512 在同一学习率下精确等价，因此 C128 的选择来自事先规定的并列排序，不能解释成较短候选池获胜。

每格使用相同系列 m、年份 y、seed s、期限 h 的随机游走误差作为分母。每格相对误差 R = 模型 MSE ÷ 同起点 RW MSE；年份总体分数 S 为该年 5 条序列 × 2 个 seed × 3 个期限，共 30 个 R 的算术平均。RW 每格与总体分数均为 1。

系列、两个 seed 和 h=1/5/20 等权；不按价格尺度或起点数加权，不平均两个 seed 的预测后再算分。每个 query 的完整 H20 目标必须全部位于同一年，三个展示期限使用相同起点集合。h 指下一个报价/拍卖观察位置，不保证等于固定日历天数。static GRU 和 RW 使用 D1 中同状态、相同起点的历史预测；独立审核重新抽取并复算这两项，未把 D1 其他年份或年末不完整目标混入。[全部 480 格](results/SOLID_B1/ALL_METRICS.csv) · [每系列/期限聚合](results/SOLID_B1/independent_audit/CONFIG_YEAR_SERIES_HORIZON_SCORES.csv)

2022/2023 在 RRC 及之前研究中已经用于开发；B1 是其后的有限备选研究，2023 **不是未用于开发的独立最终测试集**。五条价格序列涉及四个关联市场体系；EEX 与 EU_REPORTED 共享 EU ETS 暴露，不把它们当两个完全独立市场，也不把 seed、期限或重叠 H20 标签当独立市场重复。本文没有据此构造 iid 跨市场置信区间或显著性结论。

## 3. 全部八配置及固定对照

数值越低越好。选中身份在表中标明；2023 较优但未被 2022 选中的配置不替换主结果。

{config_table}

全八个配置的 2023 总体分数都高于 RW。较大学习率在部分 2023 展示分数上更低，但 2022 族内选择没有选中它；因此没有据此追加学习率、改选参数或重跑。[全部配置聚合](results/SOLID_B1/independent_audit/CONFIG_YEAR_SCORES.csv)

## 4. 2023 五条序列结果

以下 distance 固定为 `{distance}`，recency 固定为 `{recency}`；每行均值包含两个 seed 和三个期限。

{series_table}

distance 在 4/5 条序列上略优于静态 GRU、3/5 上优于 RW；recency 对应为 4/5 和 2/5。多个值极接近 1，例如上海 distance 为 0.99998796，不能把这类微小差异写成稳定或有实际意义的普遍优势。广州的较高误差使两个选中配置的总体分数仍超过 1；这不是通过删除该序列可以修正的事后分析自由度。[系列聚合原表](results/SOLID_B1/independent_audit/CONFIG_YEAR_SERIES_SCORES.csv)

EEX 上两个选中配置分别为 {series_scores[distance,2023,'EEX_EUA_T3PA']:.10f}、{series_scores[recency,2023,'EEX_EUA_T3PA']:.10f}，静态 GRU 已是 {series_scores['static_gru',2023,'EEX_EUA_T3PA']:.10f}。本轮可报告该子序列低于 RW，不能把其大部分继承自静态骨干的表现全部归因于 SOLID，更不能凭该子序列打开确认期。

### 4.1 各期限展开

表内仍为两个 seed 的误差比均值，所有 RW 格均为 1。期限分解用于公开误差结构，不承担另选目标期限的功能。

{horizon_table}

## 5. 相同候选池与学习率下的 distance / recency 对照

最后一列为 `recency − distance`，负值表示 recency 较好。两年分别展示，禁止把两年合并成一个选择指标。

{matched_table}

四组匹配对照在 2022 年均由 distance 略占优，在 2023 年均由 recency 较好；选中 C128、lr=0.001 的差为 {global_scores[recency,2023]-global_scores[distance,2023]:+.10f}。这一开发观察没有支持“距离检索在本任务中提供额外总体优势”的主张，也不等于证明 SOLID 在其他数据上普遍无效。未围绕这种反转新增度量学习、加权、ridge、窗口或更新规则实验。[匹配对照](results/SOLID_B1/independent_audit/MATCHED_DISTANCE_RECENCY.csv) · [分系列匹配表](results/SOLID_B1/independent_audit/MATCHED_DISTANCE_RECENCY_SERIES.csv)

## 6. 精确派生、运行量与独立核验

正式运行开始于 `{started['started_utc']}`，完成于 `{complete['finished_utc']}`，约 {duration:.1f} 秒，0 个失败。唯一先前 pilot 为 20 个查询的计时，不算正式成绩。实际计数如下，预测序列数不冒充骨干拟合数。

| 项目 | 实际数量 |
| --- | ---: |
| 系列×seed 任务 | 10 |
| 实际作者适配流 | 60 |
| 明确等价派生预测序列 | 20 |
| 保存的配置预测序列 | 80 |
| 实际 query 头部更新 | 25,332 |
| 实际邻居反向传播调用 | 253,320 |
| 全部保存预测的 query 行，含等价派生 | 33,776 |
| 新增骨干预训练 | 0 |
| 正式指标格 | 480 |
| 新确认期拟合/评价 | 0 |

period=1、k=10 时，两个 recency 候选池均选 q−20、q−21、…、q−29 的同一有序样本集；初始头、目标、步长及求和顺序相同。因此正式运行前的等价附录允许复用 C128 的结果作为 C512 预测，保留全部配置身份与明确 ALIAS。原协议字节和 pilot 版本另行保留。独立审核逐位核验了 20 个派生预测，均一致；没有把它们记成重复的真实反向传播。[正式完成回执](results/SOLID_B1/COMPLETE.json) · [执行前等价附录](provenance/SOLID_B1_PROTOCOL.json)

独立 selector/recompute 在 10:48:54 UTC 冻结，当时 B1 尚无完整 `ALL_METRICS.csv`，审核者尚未读取任何 B1 数值；正式运行已经开始，这里准确称为**读取结果前的实现冻结**，不冒称外部预注册。根 selector 在 10:50:01 UTC、其首次读分前冻结；独立版本于 10:50:29 UTC 执行完成后，才打开根的选择源码/输出作交叉比较。[独立冻结](provenance/SOLID_B1_INDEPENDENT_FREEZE.json) · [根选择冻结](provenance/SOLID_ROOT_SELECTOR_FREEZE.json)

独立审核不导入 runner、模型或 PyTorch，实际从 5 个批准开发 CSV 重建目标和起点，复算 **80 份 B1 预测＋20 份 D1 固定基线预测**，核对 **480＋120 个指标格**，完成 **{audit['reported_metric_scalar_and_label_comparisons']:,} 项指标值/标签比较**及 20 项等价预测检查；选择身份在独立复算指标上也完全一致。它确认评分、支持与保存预测相符；实际梯度和因果访问由另一个合成前置审核负责，不能混称此处已独立重演全部市场训练。[数值审核](results/SOLID_B1/independent_audit/INDEPENDENT_METRIC_AUDIT.json) · [15 组合成检查](provenance/SOLID_B1_INDEPENDENT_SYNTHETIC.json)

两个 selector 的 **{len(exact)} 项选择/结构/范围比较**、**{len(numeric)} 项分数比较**全部通过，最大绝对尾差为 {agreement['maximum_absolute_numeric_difference']:.2e}。范围包含全部八配置和两项基线的两年分数、两个族内选择及选中模型/基线的全部五系列分数。[交叉核对证据](results/SOLID_B1/report_artifacts/SELECTOR_AGREEMENT.json)

## 7. 数据封存与研究结论

新 EEX 来源仍保持原固定划分：2012–2023 开发 2,255 日；2024 确认 213 日；2025 确认 213 日；2026 排除。EEX 目标为 T3PA EUA 成功一级拍卖的按拍卖量加权日清算价，与 EU_REPORTED 的作者报价口径分别保留。本次没有取得新来源、读取 EEX 确认期作拟合/评分，或公开逐日价格/预测。[EEX 数据方法](data_reproduction/EEX_DATA_METHODS_ZH.md)

源准备阶段曾按预先固定边界机械解析确认单元格以拆分与核验来源；“封存”指研究使用隔离，不声称数据从未下载或被程序解析。旧上海确认期此前已评价，不会因本轮改称未见。EEX 原始/派生报价及逐点预测均为本地文件，公开内容限本报告的聚合误差、来源元数据与复现代码。

**本轮完成现有方案参考后收束：总体没有胜过 RW，距离检索也没有在四组 2023 匹配比较中胜过 recency。** 所有配置、计算身份和微小差异如实保留；没有性能驱动追加、没有把普通改造重新命名为创新、没有凭已有方法的适配实验认证发表。后续若提出新机制，仍需要独立的新颖性依据、明确可识别条件及在方法冻结后的验证；本报告不声称已经完成此类新实验。

## 8. 复核与报告生成入口

完整研究结果在本目录存在时，可向新输出目录复算：

```bash
python -B source/independent_solid_b1.py run \
  --protocol provenance/SOLID_B1_PROTOCOL.json \
  --output /absolute/path/to/new_independent_audit
```

该指标审核需要本地批准开发数据和私有预测。公开报告生成器 `source/make_solid_results.py` 仅读取公开聚合、协议、审核及选择回执，不打开市场 CSV、NPZ 或确认数据；保留本次冻结输入身份，并拒绝覆盖已有输出：

```bash
python -B source/make_solid_results.py \
  --study /absolute/path/to/event_feedback_innovation_20261017 \
  --output /absolute/path/to/new_report_artifacts \
  --report /absolute/path/to/RESULTS_SOLID_ZH.md
```

此报告引用的本地相对链接以 study 根目录为基准，移动报告时应一并保留对应研究目录结构。全部交付身份见 [SOLID 结果交付清单](results/SOLID_B1/report_artifacts/SOLID_RESULTS_DELIVERY_MANIFEST.json)。
"""
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "SELECTOR_AGREEMENT.json", agreement)
    write_json(output / "B1_REFERENCE_DECISION.json", decision)
    with report.open("x", encoding="utf-8") as stream:
        stream.write(report_text)
    included = [Path(__file__).resolve(), root / "source/independent_solid_b1.py", report,
                root / "provenance/SOLID_B1_INDEPENDENT_FREEZE.json", root / "provenance/SOLID_B1_INDEPENDENT_SYNTHETIC.json",
                *sorted(independent.glob("*.json")), *sorted(independent.glob("*.csv")),
                output / "SELECTOR_AGREEMENT.json", output / "B1_REFERENCE_DECISION.json"]
    manifest = {"created_utc": agreement["created_utc"], "aggregate_only_public_results": True,
        "market_prices_or_individual_predictions_included": False, "new_market_fits": 0, "confirmation_data_read": False,
        "files": [{"path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                   "bytes": path.stat().st_size, "sha256": sha(path)} for path in included]}
    write_json(output / "SOLID_RESULTS_DELIVERY_MANIFEST.json", manifest)
    return {"report": str(report), "report_sha256": sha(report), "agreement_exact_checks": len(exact),
            "agreement_numeric_checks": len(numeric), "maximum_absolute_difference": agreement["maximum_absolute_numeric_difference"],
            "manifest_sha256": sha(output / "SOLID_RESULTS_DELIVERY_MANIFEST.json"), "manifest_files": len(included)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.study.resolve(), args.output.resolve(), args.report.resolve()), indent=2))


if __name__ == "__main__":
    main()
