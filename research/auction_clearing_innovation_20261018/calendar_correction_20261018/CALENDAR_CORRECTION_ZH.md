# 2020 EEX 日历脚注修正：来源、规则与输入变更

## 结论及处置

原 v1 日历解析器遗漏了一项当时已经写入官方工作簿的数量调整。2019 年 12 月 12 日发布的 2020 年拍卖日历 v2.1 保留原 `Volume` 数字，同时用 `***` 脚注通知延后每场 EU 拍卖中的 373,000 EUA Innovation Fund 数量。只取数值单元格会高估相应的已知计划供给。

本修正仅应用于这一份 SHA 固定的工作簿中，产品为 T3PA、平台为 CAP、拍卖名称明确带 `***` 的 96 个源行。原冻结文件、原训练和原预测全部保留；修正数据位于 `rerun/`，新实验必须先重新冻结。该问题是在查看 v1 开发结果后的诊断中发现，故应公开记录为事后发现的输入缺陷，不能称为预先未知的新供给冲击，也不能隐去原实验的失败。

## 原始证据与日期

原件路径：

`/workspace/scratch/be2216688c09/research_planning/data/source_documents/calendar_vintages_2012_2023/20260217_History_EUA_Primary_Auction_Calendar_2011-2025/History_EUA_Primary_Auction_Calendar_2020/EEX Auction Calendar_exIF_12.12.2019.xlsx`

原件 SHA256：`22820d037bac47c38264732b87b0ec39f648259df2051d2bdbf63f6008daf383`。

`Calendar!A14:C14` 给出 2019-12-12、版本 2.1，以及 Innovation Fund 数量延后的修订说明。`Calendar!A34:A35` 给出数量、适用拍卖日及生效条件；关键数量短语为 “373,000 allowances in each EU Auction on Mondays, Tuesdays and Thursdays”。说明同时引用 2019 年 12 月 12 日 Customer Information，延后持续至必要实际安排到位。完整单元格文本保存于 `CALENDAR_CORRECTION_RECEIPT.json`，独立来源审查见 `INDEPENDENT_FOOTNOTE_AUDIT.json`。

历史日期仍按原协议处理：公告日期加 1 自然日后才视为已知。此版本的可用窗口为 **[2019-12-13, 2020-03-18)**，版本 ID 为 `2020_2019-12-12_22820d037bac`。这些是官方 Document History 对历史发布日期的陈述，当前 ZIP 的封装日期不提供当年在线发布时点的独立证明。

## 可复算的最小解释规则

对固定源 SHA 的 T3PA / CAP 行，只有原始拍卖名称带独立 `***` 标记才执行：

\[
S_{\text{known plan}}=S_{\text{numeric cell}}-373000.
\]

新 CSV 保留 `plannedvolume_numeric_raw`、`plannedvolume_text_adjustment`、`plannedvolume_rule_id`、`plannedvolume_evidence`、`volume_interpretation`，原 `plannedvolume` 列供未改动的 example builder 使用。脚注明确标记的 96 行覆盖 2020 年 1 月至 8 月，包括 8 月半量行；这是逐行文字标记的字面解释。9—12 月另有 44 个 CAP 行，仅标 `*` 并说明 MSR indicative，本次没有把 `***` 外推到这些行。现有冻结 H=1/5/20 样本在该版可见期间实际只选到 1 月 7 日至 4 月 20 日，因此后半年的适用范围不影响本次重跑。

本规则不读取最终拍卖结果。日期、平台、版本、版本可用时间、原版失效区间、排序和目标事件身份均保持不变。修正不是使用后来实际结果把计划量强行补成最终量。

## 相邻版本与同类文字检查

| 工作簿版本 | 已知日期 | 检查结论 |
|---|---|---|
| 2019-07-11 初版 | 2019-07-12 | 无延后脚注，不回溯施加扣除。 |
| 2019-11-04 v2.0 | 2019-11-05 | 无延后脚注；独立审查确认 210 个数值计划行与 v2.1 一致，排除已经数值扣除后再扣一次。 |
| 2019-12-12 v2.1 | 2019-12-13 | `***` 明确通知延后，96 个 CAP 源行应用规则。 |
| 2020-03-17、文件名 2020-04-03 的两版 | 未建立唯一可靠日期 | 同样保留脚注，但同一历史日期对应不同内容，沿用 v1 的整组排除；不恢复既有信息缺口。 |
| 2020-07-01 v4.0 | 2020-07-02 | 历史行保留延后记录，原数字表已经更新，并新宣布 7 月 16 日开始拍卖 Innovation Fund；不把这项后来知识回填到 2019 年 12 月。 |
| 2020-09-07、2020-09-09 | 各自公告次日 | 同样属于后来已更新数字表，未应用固定 SHA 规则。 |

本次还扫描了全部 34 份 2017—2023 开发 XLSX 概览的前 100 行，保存涉及延期、扣减、重分配和 indicative 的来源单元格。2022 年扣减/取消后重分配文字位于相应历史修订描述，没有发现与当前固定源相同的、要求在数值单元格外执行一个明确未兑现扣除的额外证据；这不是对所有自由文本语义的穷尽认证。本次不扩大修正规则。完整来源扫描与 8 个相邻 2020 原件元数据分别在 `local_only/SOURCE_FOOTNOTE_AUDIT.json`、`local_only/SURROUNDING_2020_VINTAGES.json`。

## 重建与逐值差异

先运行冻结版 parser 从官方原件重新生成数值版 CSV，得到与 v1 完全相同的字节 SHA；随后才应用上述源证据规则。修正后仍为 **6,608 行、32 个版本**。然后复制原事件 CSV，用原字节 `prepare_examples.py` 重建，仍为 **4,141 个样本**。

| 变更项 | 结果 |
|---|---|
| 受影响日历源行 | 96 |
| 受影响预测样本 / 不同目标事件 | 96 / 44 |
| 2019 发出的跨年预测 | H1: 1；H5: 2；H20: 3 |
| 2020 发出的预测 | H1/H5/H20 各 30 |
| 输入数组 | `context` 第 0、1 列共 192 个值；`base` 第 1 列共 96 个值 |
| 不变数组 | `x`、`y`、`price`、`anchor`、`scale`、`bid_ratio`、`last_bid_median` 及所有状态标签逐值一致 |
| 样本身份、目标日期/平台、历史边界、版本时钟 | 全部一致 |
| 2021—2023 直接评价输入 | 全部一致 |
| 原计划量与最终量不一致计数 | 170 → 81；89 个标记由 False 转 True，另外 7 个受影响例仍不一致 |

最后一项只用于描述输入修正后的结果，未参与规则建立或范围选择。7 个仍不一致例不按后来实际数量继续补改。

成熟标签范围内，2021 折初始训练无变更，验证有 87 个输入样本变更，最终训练有 93 个；2022、2023 折的初始与最终训练各有 93 个变更，验证无变更。其余 3 个受影响例为未完成拍卖，仍保留预测身份，但不成为成熟价格标签。因此即使 2021—2023 直接评价输入相同，拟合、选择和预测仍可能受此前错误输入影响，必须完整重跑才能评估。

所有 13 个 v1 冻结文件在重建前后均通过原 SHA 校验。新输入完整哈希和数组差异见 `INPUT_CHANGE_SUMMARY.json`。本文件形成时尚未对修正数据进行任何模型拟合或性能评分。

## 复现命令

在工作区根目录使用原研究 Python 环境：

```bash
.venv_carbon_route_20261018/bin/python carbon_workspace/research/auction_clearing_innovation_20261018/calendar_correction_20261018/parse_calendar_vintages_corrected.py
.venv_carbon_route_20261018/bin/python carbon_workspace/research/auction_clearing_innovation_20261018/calendar_correction_20261018/rebuild_and_compare_inputs.py
```

第二步会拒绝在 `rerun/PROTOCOL_FREEZE.json` 已存在时覆盖输入。修正来源采用 EEX 官方历史日历档案：<https://www.eex.com/fileadmin/EEX/Downloads/Trading/Calendar/Auction_Calendar/20260217_History_EUA_Primary_Auction_Calendar_2011-2025.zip>。本次只读取已下载的 2017—2023 开发工作簿，没有打开 2024/2025 日历内容或封存价格。
