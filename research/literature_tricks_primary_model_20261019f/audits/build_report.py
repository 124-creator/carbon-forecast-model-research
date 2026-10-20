"""Build the phase-f narrative tables and decision from completed aggregate evidence.

This delivery script does not read canonical price rows, fit models, select
hyperparameters, or recalculate prediction scores.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |",
         "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


stats = read("audits/ENHANCEMENT_STATISTICS.json")
equations = read("audits/FITTED_EQUATIONS_REPLAY.json")
public = read("audits/PUBLIC_STATE_REPLAY_AUDIT.json")
same = read("audits/SAME_PRIMARY_ONLINE_STRATEGY_DIAGNOSTIC.json")
freeze = read("provenance/LOCAL_REMOTE_FREEZE.json")
assert stats["status"] == "COMPLETE_EXPOSED_DEVELOPMENT_STATISTICS"
assert equations["status"] == "PASS_INDEPENDENT_NUMPY_FITTED_EQUATIONS"
assert public["status"] == "PASS_ALL_PUBLIC_NUMERIC_STATE_REPLAYS"
assert equations["response_saved_head_equations_replayed"] == 544
assert equations["ridge_equations_replayed"] == 54
assert public["total_prediction_panels"] == 90
assert public["online_matured_readout_solves"] == 13632
assert len(stats["metrics"]) == 98 and len(stats["paired_intervals"]) == 112

markets = ["TIANJIN_OFFICIAL_MEAN", "CEA_OFFICIAL_MEAN"]
market_names = dict(zip(markets, ["天津 TJEA", "全国 CEA"]))
settings = [
    "raw320", "window_past", "train_joint_test_window", "full_series_joint",
    "dense_path", "validation_selected_matured_readout",
]
setting_names = {
    "raw320": "原始主模型 max320",
    "window_past": "逐历史窗口 EWT",
    "train_joint_test_window": "训练前缀联合／评价窗口 EWT",
    "full_series_joint": "全序列联合 EWT（回顾设置）",
    "dense_path": "H1–20 密集路径监督",
    "validation_selected_matured_readout": "验证所选成熟标签更新",
}
families = [
    "signed_context_response", "static_response", "amplitude_context",
    "signed_shock_only", "ridge_price",
]
metric = {(x["market"], x["setting"], x["family"]): x for x in stats["metrics"]}
interval = {(x["market"], x["block_length"], x["comparison_id"]): x
            for x in stats["paired_intervals"]}


def q(market, setting, family="signed_context_response"):
    return metric[(market, setting, family)]["Q"]


main_rows = []
primary_records = []
for setting in settings:
    row = [setting_names[setting]]
    for market in markets:
        value = q(market, setting)
        relative = 100 * (1 - value / q(market, "raw320"))
        row += [f"{value:.6f}", "—" if setting == "raw320" else f"{relative:+.4f}%"]
        if setting != "raw320":
            x20 = interval[(market, 20, "enhancement/" + setting)]
            x40 = interval[(market, 40, "enhancement/" + setting)]
            primary_records.append({
                "market": market, "setting": setting, "raw_primary_Q": q(market, "raw320"),
                "enhanced_primary_Q": value, "relative_Q_improvement_percent": relative,
                "effect_rw_percentage_points": x20["effect_rw_percentage_points"],
                "enhancement_family_interval_block20":
                    x20["familywise_95_interval_rw_percentage_points"],
                "enhancement_family_interval_block40":
                    x40["familywise_95_interval_rw_percentage_points"],
                "retrospective_full_series": setting == "full_series_joint",
            })
    main_rows.append(row)
main_table = table(
    ["主模型设置", "天津 Q", "天津相对 Q 改善", "CEA Q", "CEA 相对 Q 改善"],
    main_rows,
)

comparison_rows = []
native_rows = []
for market in markets:
    for setting in settings:
        values = []
        for family in families:
            x = metric.get((market, setting, family))
            values.append("—" if x is None else f"{x['Q']:.6f}")
        comparison_rows.append([market_names[market], setting_names[setting]] + values)
        x = metric[(market, setting, families[0])]
        native_rows.append([market_names[market], setting_names[setting]]
                           + [f"{v:.4f}" for v in x["rmse_by_horizon"]])
comparison_table = table(
    ["市场", "设置", "主模型", "静态", "幅度门", "无成交量门", "价格 Ridge"],
    comparison_rows,
)
native_table = table(["市场", "主模型设置", "H1 RMSE", "H5 RMSE", "H20 RMSE"], native_rows)

interval_rows = []
for market in markets:
    for setting in settings[1:]:
        a = interval[(market, 20, "enhancement/" + setting)]
        b = interval[(market, 40, "enhancement/" + setting)]
        fmt = lambda x: "[" + ", ".join(f"{v:.4f}" for v in x) + "]"
        interval_rows.append([
            market_names[market], setting_names[setting],
            f"{a['effect_rw_percentage_points']:+.4f}",
            fmt(a["familywise_95_interval_rw_percentage_points"]),
            fmt(b["familywise_95_interval_rw_percentage_points"]),
        ])
interval_table = table(
    ["市场", "增强设置", "增益 Δ（RW 归一化百分点）",
     "块长 20：五项家族调整区间", "块长 40：五项家族调整区间"],
    interval_rows,
)

online_rows = []
for strategy in ["frozen", "expanding_equal", "expanding_half_life126", "rolling126", "rolling252"]:
    row = [strategy]
    for market in markets:
        chosen = metric[(market, settings[-1], families[0])]["strategy"]
        row += [f"{q(market, 'matured_readout_' + strategy):.6f}"
                + ("（验证所选）" if strategy == chosen else "")]
    online_rows.append(row)
online_table = table(["固定策略", "天津主模型 Q", "CEA 主模型 Q"], online_rows)

same_rows = []
for x in same["markets"]:
    same_rows.append([
        market_names[x["market"]], x["primary_validation_selected_strategy"],
        f"{x['primary_Q']:.6f}",
    ] + [f"{v['same_policy_reference_Q']:.6f}" for v in x["comparators"]])
same_table = table(["市场", "主模型验证所选的同一策略", "主模型", "静态", "幅度门", "无成交量门"], same_rows)

report = r"""# 阶段 f 完成报告：文献技巧如何改善现有条件持续时间主模型

本轮已经把文献中的分量输入、密集多步监督和成熟标签后的读出更新加入现有 signed_context_response，完成全部 598 次声明拟合、13,632 次在线读出求解、配对统计和完整预测重放。主模型出现真实提效：天津的验证所选在线更新使 Q 降低 4.9901%；全国 CEA 的历史窗口 EWT 使 Q 降低 1.9618%。全序列联合 EWT 则分别降低 28.2806% 和 27.5368%，该结论对应完整序列可用的回顾实验。

同样增强后的机制对照显示，当前证据仍不足以把这些改善归因于条件持续时间门的独立创新。全序列设置下静态模型、无成交量门和价格 Ridge 的点分数更好；在线更新也存在策略选择与机制归因的区别。因此本轮完成实验与可复现交付，保留增强候选及全部结果，尚未形成“条件持续时间创新稳定优于充分增强对照”的正面论文结论。

本报告的全部项目数值来自 [独立统计结果](audits/ENHANCEMENT_STATISTICS.json)、[拟合方程复放](audits/FITTED_EQUATIONS_REPLAY.json) 与 [公开状态完整重放](audits/PUBLIC_STATE_REPLAY_AUDIT.json)。机器可读决定见 [FINAL_DECISION.json](FINAL_DECISION.json)。

## 1. 文献究竟怎样做，我们怎样落地

本轮针对已读取的一手论文和作者代码执行具体处理规则；没有把“轻微泄露”作为排除某个实验的统一标签。处理取得哪些数据、在哪个阶段使用、文献给它什么解释，都与成绩一起保存。

| 文献做法 | 本轮实现 | 对应范围与改编 |
| --- | --- | --- |
| SWEWT–GRU 将原始序列和 EWT 分量一起作为输入，并比较不同分解边界 [S1] | 保留原价及全部三个 EWT 分析分量，分别形成状态，共享现有持续时间门 | 主模型由 9 个参数增为 15 个；保持原始官方价格目标。网络主体使用我们的响应模型，未冒称复制作者 GRU |
| 同文比较全序列分解、训练联合分解及历史窗口分解 [S1] | 实跑逐窗口、训练前缀联合／评价窗口、全序列联合三套设置 | 全序列结果确实报告；原文将它作为揭示未来信息影响的对照，没有把它作为实时可发行预测来推荐 |
| DLinear 作者代码在整个 pred_len 上计算 MSE [S4] | 同一模型训练 H1…20 完整轨迹，仍按 H1/5/20 验证和评价 | 增加的是训练标签约束，推理输入和参数量不变；没有实现教师蒸馏 |
| Onlineforecast 的固定输入变换、动态回归与遗忘因子 [S2]；碳价预测中的扩展估计窗口 [S3] | 固定原非线性状态参数，合格标签成熟后更新三个线性系数 | 冻结、扩展等权、半衰期 126、滚动 126／252 五策略均留存；具体长度是我们的事前实现选择 |

SWEWT–GRU 的 Table 6 报告全序列联合设置 RMSE=1.594、全部历史窗口设置 RMSE=2.317；它们来自原论文的广东收盘价任务，不能与本项目的天津／全国总成交均价多期限分数直接横比。[S1]

碳价文献也会明确讨论数据构造的适用范围。S3 将年度核证排放插值为月度，并通过排除该变量进行敏感性检查；这说明应核对具体构造、用途与影响，不能从某一个已发表例子推导出任意未来信息处理都能支持同一种预测结论。[S3] 完整文献对应表与阅读位置见 [PUBLISHED_TRICKS_ZH.md](literature/PUBLISHED_TRICKS_ZH.md)。

## 2. 我们的主体模型和创新候选保留在哪里

主体保留有符号冲击响应、两个时间尺度、发行时上下文调节持续时间，以及精确线性读出求解。以四路 EWT 版本为例：

\[
v_{c,k,t}=(1-\rho_k)\sum_{\ell=0}^{63}\rho_k^\ell r_{c,t-\ell},
\qquad
\widehat{\Delta P}_{t,h}
=s_P\left[\mu h+\sum_{c=0}^{3}\sum_{k=1}^{2}
b_{c,k}v_{c,k,t}\sum_{j=1}^{h}\lambda_{k,t}^{j}\right].
\]

这里 \(r_{0,t}=\Delta P_t/s_P\)，\(r_{c,t}=\Delta \mathrm{EWT}_{c,t}/s_P\)（\(c=1,2,3\)）；\(s_P\) 是固定初始训练段价格变动 RMS。四路为归一化后的原价变化及全部三个 EWT 分量变化，有序历史速率和上下文门在通道间共享。持续率由当前原始价格冲击幅度与原定义成交量 surprise 调节，读出系数允许正负。EWT 只作为分析特征前端，不把分析系数相加当作价格重构。密集监督与在线更新两块仍使用原始单路、9 参数主模型。

创新候选位于这个可学习的条件持续时间机制及其紧凑共享结构。EWT、滚动回归、完整多步 MSE 本身是已有方法。把它们接到主体上是必要的性能检验；对整体组合是否具有新的方法贡献，还需要同处理对照与更广的相关工作核实。本轮并未宣称已证明全球新颖性。

| 家族 | 原始输入参数数 | 四路 EWT 参数数 | 本轮机制比较回答的问题 |
| --- | --- | --- | --- |
| 条件持续时间主模型 | 9 | 15 | 上下文是否需要改变响应持续时间 |
| 静态持续时间 | 5 | 11 | 固定时间尺度是否已经足够 |
| 条件幅度门 | 9 | 15 | 上下文只调幅度是否足够 |
| 无成交量持续时间门 | 7 | 13 | 成交量信息是否提供额外贡献 |

同输入 EWT Ridge 使用四路末 20 个变化，共 80 个特征，三期限独立读出含截距共 243 个拟合系数。它是较高容量的强线性参考；其 20 步历史与主体 64 步不同，不能称为完全匹配的参数量或历史跨度。具体实现和损失归一化见 [事前设计](methods/ENHANCEMENT_DESIGN_ZH.md)。

## 3. 数据、冻结与实际工作量

本轮复用已经下载和核对的官方数据，未增加行情下载。两序列均为官方公布的每日总成交均价，保留原价精度、全部观测记录及原切分，不补自然日或重算价格。数据来源与 SHA 见 [RUN_DATASETS.json](provenance/RUN_DATASETS.json)；获取与重放入口见 [REPRODUCE_ZH.md](REPRODUCE_ZH.md)。

| 数据 | 全部记录 | 初始训练原点 | 验证原点 | 最终拟合原点 | 评价原点 | 评价发行日范围 |
| --- | --- | --- | --- | --- | --- | --- |
| 天津 TJEA | 1,077 | 562 | 195 | 777 | 196 | 2023-04-19 至 2026-07-02 |
| 全国 CEA | 1,251 | 666 | 230 | 916 | 231 | 2025-09-12 至 2026-09-01 |

H1/5/20 指随后第 1、5、20 条观测，最晚标签均为 2026-09-30。天津原始记录最大日历间隔为 319 天，因此不能把 H20 一律写成未来 20 个自然日。所有发行原点按当前报告已知处理；现有数据是历史快照，并非逐日留存的首次公布版本。两个评价尾段在阶段 e 已评分，本轮全为开发实验。[原点记录](audits/ENHANCEMENT_STATISTICS.json)；[天津输入清单及日历间隔](results/ewt/TIANJIN_OFFICIAL_MEAN_fixed602020/INPUT_MANIFEST.json)。

方案在本轮拟合前于 2026-10-19 14:49:14 UTC 推至 GitHub，远端冻结提交为 137d58e78ddf28c0163ab8d1cadf50b74e79ef67。三个块均在远端头与内容树回读核对后启动，执行前后四个拟合源与两份冻结计划 SHA 一致。该记录证明本轮的评分前固定，不能把早已使用过的数据重新称为未见数据预注册。[LOCAL_REMOTE_FREEZE.json](provenance/LOCAL_REMOTE_FREEZE.json)

| 实验块 | 选择拟合 | 最终拟合 | 合计 | 其他求解 |
| --- | --- | --- | --- | --- |
| EWT 响应家族 | 384 | 24 | 408 | 24 次初始动力学读出诊断 |
| EWT Ridge | 48 | 6 | 54 | — |
| 密集路径响应家族 | 128 | 8 | 136 | — |
| 成熟标签读出 | 0 | 0 | 0 | 13,632 次三系数读出求解 |
| 合计 | 560 | 38 | **598** | **13,656** |

非线性实际梯度更新共 80,398 次。每个响应单元按八个 alpha、两个学习率选择，最高 320 次更新，并遵守固定 patience 和验证容差；最终重新初始化，执行所选确切 epoch。原始 max320、价格 Ridge 与 RW 直接引用既有结果，不把引用算成新拟合。本轮三类技巧分别运行，没有在看见结果后追加组合网格。

## 4. 主模型确实提升了多少

\[
Q=\frac{1}{3}\sum_{h\in\{1,5,20\}}
\frac{\operatorname{MSE}_{model,h}}{\operatorname{MSE}_{RW,h}}.
\]

Q 越低越好，RW=1。“相对 Q 改善”为 100×(1−Q增强/Q原模型)，不是 RMSE 的同幅度下降。下表完整列出五个事前增强协议；未按两个市场测试分数拼接出一个事后选定的新模型。

@@MAIN_TABLE@@

全序列处理给主模型带来的大幅改善是真实运行结果；训练前缀联合、评价时转为窗口的处理却在两个市场都变差。历史窗口 EWT 在两市场点值改善，天津密集监督和成熟标签更新也有提升。密集监督和成熟标签更新的改善未跨市场保持；窗口 EWT 的双市场点值改善仍有较大不确定性。

### 4.1 原始单位的三期限误差

以下 RMSE 单位为元／吨。全部 294 个逐期限指标行（包括 MAE、MSE、偏差及比率）保存在 [ENHANCEMENT_NATIVE_METRICS.csv](audits/ENHANCEMENT_NATIVE_METRICS.csv)，与统计 JSON 一一对应。

@@NATIVE_TABLE@@

天津密集监督的 H1 RMSE 比原模型略高，H5 与 H20 较低；因此不能把其 Q 改善描述为所有期限一致改善。天津在线主模型三个期限 RMSE 均降低；CEA 的在线主模型三个期限均升高。完整路径监督是否适合当前动力学形式，需要依据期限结构判断。

## 5. 同样增强后，主机制是否更好

@@COMPARISON_TABLE@@

表内“验证所选成熟标签更新”为各家族获得相同五策略选择机会后，分别按各自验证分数选定的完整流程。其他行共享相同的数据处理与响应训练预算。Ridge 没有密集路径和在线新增行，是因为本轮没有声明这些 Ridge 实验；原始 Ridge 仍保留为参考。

全序列 EWT 下，静态与无成交量门在两个市场的点分数均略优于主模型；它们之间的机制家族区间跨零。Ridge 的 Q 为天津 0.436594、CEA 0.340286，比同设置主模型更低。主模型相对该 Ridge 的增益分别为 −22.4838、−29.1175 个 RW 归一化百分点；块长 20 的次级点态 95% 区间分别为 [−26.8895, −14.5158] 与 [−36.6063, −22.3584]。这些线性参考比较未进行机制家族调整，且容量不同，解释时保留这两个事实。

天津历史窗口 EWT 中，主模型比同输入静态模型差 7.9145 个 RW 归一化百分点；十五项机制家族调整后区间在块长 20、40 下分别为 [−11.8207, −1.5347]、[−11.3766, −2.0681]。它说明保留复杂门控并不会自动把分量信息转化为更好的预测。

### 5.1 在线策略必须把选择流程和门机制分开解释

@@ONLINE_TABLE@@

天津测试上 rolling126 的主模型 Q 更低，但验证选择的是 rolling252，因此主比较始终使用 rolling252。全国 CEA 始终使用其验证选择的 rolling126。五种策略全部留存，测试表现没有回写验证选择。

天津静态与无成交量模型各自验证选择 frozen，幅度门选择 rolling252；这使“各自验证所选流程”的对照同时包含策略差异。为检查归因，我们在看到在线结果后，补充以下纯描述性表：将主模型已保存的验证选择策略，同样用于三个对照，直接读取早已完成的路径。它没有新拟合、新预测、新置信区间，也不属于事前十五项机制家族。[SAME_PRIMARY_ONLINE_STRATEGY_DIAGNOSTIC.json](audits/SAME_PRIMARY_ONLINE_STRATEGY_DIAGNOSTIC.json)

@@SAME_TABLE@@

天津在相同 rolling252 下，静态与无成交量门均优于主模型。因此，主模型相对“各自验证选择的对照流程”的点值优势，不能单独归因于条件持续时间门。CEA 的四家族原本就都选择 rolling126，主模型也没有对三个对照形成点值优势。

## 6. 不确定性与本轮证据强度

按拟合前冻结的规则，对每个市场、每个块长各执行 50,000 次配对循环块 bootstrap，块长 20 为主配置，40 为敏感性。所有模型共享抽样原点，每次重抽重新计算各期限 RW 分母。下表为五项增强比较的 Bonferroni 家族调整 95% 区间，增益 Δ=100×(Q原模型−Q增强)，单位为 RW 归一化百分点。正值支持增强。

@@INTERVAL_TABLE@@

全序列 EWT 是唯一在两个市场、两个块长下，增强家族区间均保持正值的设置，其适用范围仍是完整序列回顾实验。天津在线更新的逐项 95% 区间在两个块长下为正，但五项家族调整后主块长 20 跨零；不能只取块长 40 的正区间称其稳健通过。历史窗口的双市场点值改善都存在较大不确定性。密集监督在 CEA 的调整区间为负，训练联合／评价窗口在两个市场均为负。

十五项机制比较没有支持跨两个市场的持续时间门独立增量。存在一个应完整披露的小正区间：CEA 训练联合／评价窗口主模型对无成交量门增益为 0.01024 个百分点，两个块长调整区间均正。但这一主模型选中 epoch 0、门矩阵全零，并与静态和幅度门的预测完全相同，Q≈0.99808，显著劣于原始主模型。因此它并非学到条件持续时间优势的证据。

所有区间条件于固定路径、固定超参数与已经暴露的开发过程；重抽不重训模型，也不重演每个 bootstrap 样本的在线策略选择。两个市场分别调整，没有声称联合跨市场或所有历史研究搜索的覆盖。这些范围见 [冻结统计方案](audits/STATISTICAL_PLAN_BEFORE_RUN.json) 和 [统计回放](audits/ENHANCEMENT_STATISTICS_REPLAY.json)。

## 7. 拟合、方程和公开重放均已实际核验

独立 NumPy 程序逐一重建响应基底、标准化 Ridge 读出、验证损失和最终预测，覆盖 544 个响应 checkpoint、54 个 Ridge checkpoint，共 598 个拟合 checkpoint。它未调用 Torch 模型 forward 或 sklearn 估计器。原始特征与标签独立重建；EWT 分量在该方程审查中读取已经核 SHA 的缓存，EWT 变换则在另一公开重放入口中重新执行。[FITTED_EQUATIONS_REPLAY_ZH.md](audits/FITTED_EQUATIONS_REPLAY_ZH.md)

| 核验 | 实际结果 |
| --- | --- |
| 独立拟合方程审计 | 598 个 checkpoint、38 组最终预测，0 失败 |
| 独立响应读出系数最大差 | 8.6686×10⁻¹² |
| 保存响应头的正规方程残差 | 8.9928×10⁻¹⁵ |
| 独立原始单位最终预测最大差 | 3.0241×10⁻¹¹ |
| 独立最终 Q 最大差 | 1.9412×10⁻¹³ |
| 独立统计重算与运行汇总 | 全部指标最大差 0 |
| 公开数值状态完整重放 | 90 个预测面板；公开评价指标最大差 0 |
| 在线成熟读出重放 | 13,632 次求解，非线性参数不更新 |
| 公开结果导出 | 752 份正式 JSON 逐字保留 |

32 个响应最终 checkpoint 中，25 个存在非零非线性参数变化，全部 32 个线性读出都实际拟合。16 个最终 checkpoint 选到 320 上限。两市场训练联合／评价窗口的主模型均选 epoch 0，它们仍有已拟合线性读出；其他六个新主模型单元的非线性参数确实变化。达到预算上限不等于证明全局最优或充分收敛。

752 份正式结果 JSON 包含 557 份 EWT、157 份密集监督、38 份成熟读出。统计中的 98 行包含 8 个所选在线策略别名；90 个预测路径角色还包含 8 条重放既有 raw 的 frozen 在线路径。因此这些数量不是相互独立的实验样本数。原始价格行、逐原点预测、在线逐期头与二进制缓存留在运行目录，公开完整汇总、自行训练参数、选择轨迹与重建入口。[RESULT_EXPORT_RECEIPT.json](provenance/RESULT_EXPORT_RECEIPT.json)

公开入口用同 SHA 官方数据和公开数值 JSON 生成全部 90 个面板，无需旧 PT／NPZ 的字节或原临时目录。它是已完成的预测数值复现，不是从头重训 598 次的通用封装。重新从头训练仍需建立新的复现运行身份，具体依赖与入口在 [REPRODUCE_ZH.md](REPRODUCE_ZH.md)。此限制已写明，未用语法检查替代实际重放。

## 8. 论文决定与研究记忆

本轮可以支持的陈述是：文献技巧能够改变并在特定设置中改善当前主模型，完整序列分解带来的误差降低尤其明显；在历史输入可得的设置中，天津在线读出及 CEA 窗口分解也出现点值提升。方法、数据和复现检查均已完成。

当前尚未支持的核心陈述是：有符号条件持续时间门在同样增强的静态、幅度门、无成交量门之上获得稳定的跨市场贡献。这一判断来自匹配对照和实际区间，不是套用一个“所有期刊都要求提高 10% 或 p<0.05”的阈值，也不是因某种处理被预先称为泄露而拒绝实验。

按“实验支持创新后再完成正面模型论文”的任务顺序，本轮交付完整实验章节材料、文献映射与可审阅结果；暂不把它包装成已证实持续时间创新的成稿。现有证据更适合支持处理方式、在线适应与紧凑响应模型的比较分析。后续若继续修改主机制，应针对已观察到的门增量问题提出新的明确假设并另外固定比较，而不把本轮测试较优的技巧和市场事后拼为已确认的最终系统。

用户已授权继续实验和推送 GitHub。本轮结果分支为 research/literature-tricks-primary-model-20261019f，目标基线为阶段 e 分支；交付使用草稿 PR，保留旧阶段 d/e。当前完整接续记录见 [HANDOFF_CONTINUATION_ZH.md](HANDOFF_CONTINUATION_ZH.md)。最终推送与读回身份另外记录在 provenance/GITHUB_DELIVERY.json。

## 参考来源

- [S1] Zhang et al. (2024), Carbon Price Forecasting Using Optimized Sliding Window Empirical Wavelet Transform and Gated Recurrent Unit Network to Mitigate Data Leakage, Energies 17(17), 4358. [DOI](https://doi.org/10.3390/en17174358)。已读取 MDPI 提供的全文镜像，重点为输入构造和 §3.4.2 / Table 6。
- [S2] Onlineforecast (2023), The R Journal 15(1), 173–194. [正式全文与附录公式](https://journal.r-project.org/articles/RJ-2023-031/RJ-2023-031.pdf)，DOI 10.32614/RJ-2023-031。
- [S3] Bastianin et al. (2026), Forecasting the Price of Carbon with Macroeconomic and Financial variables. [2026-10-03 作者版本 v3](https://arxiv.org/html/2402.04828v3)，JEDC DOI 10.1016/j.jedc.2026.105435。使用 §3.2 的扩展估计和排放变量敏感性讨论；未将其月度任务与本项目观测步长混为一谈。
- [S4] Zeng et al. (2023), Are Transformers Effective for Time Series Forecasting?, AAAI 37(9). [论文](https://ojs.aaai.org/index.php/AAAI/article/view/26317)；[作者训练代码](https://github.com/cure-lab/LTSF-Linear/blob/main/exp/exp_main.py)。本轮只借用完整多步监督做法。
"""
for token, value in {
    "@@MAIN_TABLE@@": main_table, "@@NATIVE_TABLE@@": native_table,
    "@@COMPARISON_TABLE@@": comparison_table, "@@ONLINE_TABLE@@": online_table,
    "@@SAME_TABLE@@": same_table, "@@INTERVAL_TABLE@@": interval_table,
}.items():
    report = report.replace(token, value)
assert "@@" not in report
(ROOT / "REPORT_ZH.md").write_text(report.strip() + "\n", encoding="utf-8")

decision = {
    "status": "ENHANCEMENTS_COMPLETED_PRIMARY_PERFORMANCE_IMPROVED_DURATION_INCREMENT_UNSUPPORTED",
    "phase": "literature_tricks_primary_model_20261019f",
    "statistics_created_utc": stats["created_utc"],
    "all_declared_experiment_blocks_completed": True,
    "new_model_fits": 598,
    "new_response_fits": 544,
    "new_ridge_fits": 54,
    "new_nonlinear_gradient_updates": 80398,
    "matured_readout_solves": 13632,
    "initial_dynamics_diagnostic_solves": 24,
    "public_run_json_count": 752,
    "evaluation_path_roles": 90,
    "aggregate_metric_identity_rows_including_aliases": 98,
    "all_results_are_exposed_development": True,
    "known_literature_tricks_implemented_on_existing_primary": True,
    "primary_point_improvement_observed": True,
    "history_window_ewt_improves_primary_point_score_in_both_markets": True,
    "history_window_ewt_family_intervals_exclude_zero_in_both_markets": False,
    "full_series_ewt_improvement_supported_within_declared_conditional_intervals": True,
    "full_series_results_are_retrospective": True,
    "consistent_cross_market_conditional_duration_increment_supported": False,
    "positive_duration_innovation_manuscript_ready": False,
    "publication_decision_is_not_a_universal_journal_threshold": True,
    "manuscript_action": "Deliver completed experimental material; do not claim that duration-gate novelty has already been established.",
    "no_cross_protocol_test_selected_winner_declared": True,
    "online_primary_policy": {"TIANJIN_OFFICIAL_MEAN": "rolling252", "CEA_OFFICIAL_MEAN": "rolling126"},
    "same_policy_diagnostic_is_post_result_point_only": True,
    "all_five_online_policies_retained": True,
    "primary_results": primary_records,
    "audits": {
        "independent_fitted_equations": equations["status"],
        "all_public_state_replays": public["status"],
        "statistics": stats["status"],
        "independent_final_prediction_max_gap":
            equations["numeric_checks"]["native_test_predictions"]["max_absolute_gap"],
        "public_metric_max_gap": public["maximum_public_metric_absolute_gap"],
    },
    "evidence_sha256": {
        rel: sha(rel) for rel in [
            "audits/EXPERIMENT_PLAN_BEFORE_RUN.json",
            "audits/STATISTICAL_PLAN_BEFORE_RUN.json",
            "audits/ENHANCEMENT_STATISTICS.json",
            "audits/FITTED_EQUATIONS_REPLAY.json",
            "audits/PUBLIC_STATE_REPLAY_AUDIT.json",
            "audits/SAME_PRIMARY_ONLINE_STRATEGY_DIAGNOSTIC.json",
            "provenance/RESULT_EXPORT_RECEIPT.json",
        ]
    },
    "freeze_commit_remote": freeze["remote_freeze_commit"],
    "github_branch": freeze["branch"],
    "draft_pr_base": "research/signed-response-search-20261019e",
    "merge_action": "No merge requested or performed",
}
(ROOT / "FINAL_DECISION.json").write_text(
    json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps({"report": "REPORT_ZH.md", "decision": decision["status"],
                  "derived_tables_only": True, "new_fits": 0, "new_scores": 0}))
