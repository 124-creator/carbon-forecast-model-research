# 封存确认资源用途状态：截至 PR19 的有界核查

核查日期：2026-10-18。仓库固定提交：`b05939a45252ec65ba3fcf3b454dfc1804850972`。本次只读取状态、来源清单、协议与报告声明，没有打开确认价格 CSV、确认逐点预测、权重或此前未读的确认成绩。

## 1. 可执行结论

**上海 2024–2025 已经完成确认拟合、评分、统计和公开交付，不能再次作为未见确认。EEX 2024–2025、全国作者 LNCEA 2024、ScienceDB 七市场预留尾段，在所核读的当前运行链条中没有用于项目模型评价，仍可保留其“未用于本项目选模/评分”的状态。** 后三者都曾下载、机械解析或忠实性核验，不可描述为原始文件从未被程序读取，或全世界无人见过。

其中只有 EEX 的产品定义目前较清楚，且与推荐单一拍卖任务直接相关。全国为处理后的作者对数目标；ScienceDB 为经过作者清洗的日成交均价，且不同来源下有同市场/时期重叠。这些限制不会因为未评分而消失，不能自动称为严格历史实时的独立外部验证。

| 资源 | 预留范围与条数 | 截至本轮可确认的用途状态 | 能否继续叫未用于本项目选模/评分的确认资源 | 证据 |
|---|---|---|---|---|
| 上海 Mendeley D4，2024 | 2024-01-15 起；163 条 | 已确认评分；按冻结年度更新规则，成熟的 2024 标签又进入 2025 拟合 | **不能** | BCM 主报告 §1、§4.1、§5、公开交付说明；后续 EEX 暴露声明再次确认 |
| 上海 Mendeley D4，2025 | 至 2025-09-25；98 条 | 已确认评分，确认预测与统计已公开 | **不能** | 同上 |
| EEX T3PA，2024 | 2024-01-15–2024-12-16；213 个日聚合观察 | 解析器为划分机械处理过原表；event 报告明确 0 次确认拟合或评价；后续 SMP、shape、transport 均未开启确认 | **可以，有用途范围限定** | event 主报告数据表、读取边界及执行账本；后续三个报告的开发文件范围与 STOP 声明 |
| EEX T3PA，2025 | 2025-01-07–2025-12-15；213 个日聚合观察 | 同上 | **可以，有用途范围限定** | 同上 |
| 全国作者 LNCEA，2024 | 2024-01-02–2024-08-30；162 条 | 获取、导出与逐条源对照完成；post-BCM 状态明确无项目模型评价；后续实验五序列不包含该文件 | **可以，有产品/预处理与跨源历史限制** | INDEPENDENT_DATA_MANIFEST；METADATA_SCOPE_AND_EXPOSURE；BCM/event/后续五序列范围 |
| ScienceDB 七市场尾段 | 七个预冻尾段，共 844 条 | 获取、结构核验与导出完成；post-BCM 状态明确无项目模型评价；后续五序列实验没有纳入 | **保留为单列敏感性队列，不直接认证为严格实时外部确认** | SCIENCEDB_DATA_MANIFEST；METADATA_SCOPE_AND_EXPOSURE；BCM 关于本轮未拟合的声明与后续数据范围 |

上海 261 条确认记录生成 221 个具有同年完整 H20 标签的评分起点，分别为 2024 年 143 个、2025 年 78 个。两年不是独立训练复制：2025 按事先规则使用成熟的 2024 标签。记录数、起点数与独立信息量必须区分。这里只引用已公开主报告的用途声明，没有重新打开确认预测或成绩。

## 2. ScienceDB 尾段逐市场清单

均依据 `SCIENCEDB_DATA_MANIFEST.json` 的 `confirmation_sealed` 元数据，未读取下列价格文件。原规则为各序列按时间排序，前 `floor(0.60*n)` 训练，接着 `floor(0.20*n)` 验证，其余封存；规则先于该轮下载固定。

| 市场 | 尾段首日 | 尾段末日 | 尾段条数 | 清单记载路径 |
|---|---|---|---:|---|
| 上海 | 2017-03-23 | 2017-12-29 | 119 | `local_only/sciencedb_v6/confirmation_sealed/SHANGHAI_MEAN.csv` |
| 北京 | 2017-01-20 | 2017-12-29 | 136 | `local_only/sciencedb_v6/confirmation_sealed/BEIJING_MEAN.csv` |
| 广东 | 2017-05-11 | 2017-12-29 | 147 | `local_only/sciencedb_v6/confirmation_sealed/GUANGDONG_MEAN.csv` |
| 深圳 | 2017-06-12 | 2017-12-29 | 120 | `local_only/sciencedb_v6/confirmation_sealed/SHENZHEN_MEAN.csv` |
| 湖北 | 2017-04-07 | 2017-12-29 | 182 | `local_only/sciencedb_v6/confirmation_sealed/HUBEI_MEAN.csv` |
| 天津 | 2015-12-03 | 2017-06-30 | 92 | `local_only/sciencedb_v6/confirmation_sealed/TIANJIN_MEAN.csv` |
| 重庆 | 2017-09-13 | 2017-12-29 | 48 | `local_only/sciencedb_v6/confirmation_sealed/CHONGQING_MEAN.csv` |
| 合计 | | | **844** | |

这些是作者成交均价，不是独立新增的同口径官方收盘。特别是上海尾段的经济时期已经包含在后来被 BCM 等模型使用的上海 D4 长历史范围内；更换来源和价格统计量不会自动恢复完全未见的市场时期身份。北京、深圳、天津、重庆曾相对于主池新增市场标签，但“新市场标签”仍不等于没有作者清洗、没有共同时期冲击或从未被原论文使用。

## 3. 时间顺序如何解决相互矛盾的 sealed 文字

1. **初次数据获取阶段。** `DATA_EXPANSION_ZH.md`、`EXPOSURE_LEDGER.json`、`INDEPENDENT_DATA_MANIFEST.json` 记录上海 2024/25、全国 2024、ScienceDB 尾段的预先封存。该阶段未训练市场模型。
2. **后续 BCM 阶段。** BCM 主报告明确上海确认已开启、实际使用 261 条、执行确认拟合与统计，并说明数据子目录保留早期封存文字作为历史记录。主报告还写明全国 LNCEA 与 ScienceDB 独立存档、本轮未拟合。
3. **后 BCM、EEX 准备阶段。** `METADATA_SCOPE_AND_EXPOSURE.json` 的 `prior_project_exposure` 明确：上海确认已完成并暴露；全国 LNCEA 2024 与 ScienceDB 尾段获取/审计过但无项目模型评价。
4. **RRC/SOLID 实际执行。** event 主报告明确 EEX 2024/25 共 426 条仍保留，确认 0 次模型拟合或评价；并披露解析器曾为分区机械解析这些原表。
5. **SMP → shape → transport。** 三轮只使用既有五份 2024 前开发文件；预设停止触发后未进入确认。transport 主报告明确未追加 2023、新深圳或封存区实验。

因此，不能以文件名含 `confirmation` 或旧清单含 `sealed` 判断当前状态。当前用途必须按最后一次实际执行声明更新；反过来，公开来源可下载也不自动等同已经用于项目选模。

## 4. 新候选的一次确认应如何表述

可继续隔离 EEX 的 426 条；新任务若使用单次拍卖而不是既有日聚合价，需要先在开发期冻结粒度与评分规则，再在确认执行时一致应用。不得为取得更好结果临时改变单次/日聚合目标。若加入同期二级市场变量，必须一并封存对应的确认期间输入构造与时点规则，不能在选模时通过外部来源间接查看这些未来目标。

全国与 ScienceDB 可继续保留未评分状态，但只有在目标身份、上游处理、跨源重复和主任务适配都交代清楚后，才决定是否纳入一个事先冻结的确认协议。不能将不同定义的尾段合并成额外独立重复，也不应同时开启多个确认集再挑最有利的一个报告。

本次结论是基于明确范围的报告/状态链核查，不是对所有人员、所有机器、整个公共互联网访问历史的证明。2022/2023 已被项目反复用于开发；所有新路线继续使用这些开发年份时都应披露已有暴露。

## 5. 精确证据文件

以下均属于固定提交 `b05939a45252ec65ba3fcf3b454dfc1804850972`；本地缓存路径位于 `/workspace/scratch/be2216688c09/research_planning/historical/sources/`，来源 Git blob 核验见 `SOURCE_READ_MANIFEST.json`。

- `research/cross_market_innovation_20261017/REPORT_ZH.md` → `BCM_REPORT_ZH.md`：上海使用量、年度更新、公开确认交付；全国/ScienceDB 未拟合。
- `research/event_feedback_innovation_20261017/REPORT_ZH.md` → `event_REPORT_ZH.md`：EEX 426 条保留、0 次确认拟合/评价、机械解析边界。
- `research/event_feedback_innovation_20261017/provenance/METADATA_SCOPE_AND_EXPOSURE.json` → 同名缓存：后 BCM 的三类既有资源用途状态。
- `research/cross_market_innovation_20261017/data_provenance/INDEPENDENT_DATA_MANIFEST.json` → 同名缓存：上海/全国的行数、日期边界与文件身份。
- `research/cross_market_innovation_20261017/data_provenance/SCIENCEDB_DATA_MANIFEST.json` → 同名缓存：七个尾段的日期、行数、哈希、目标定义与上游清洗限制。
- `research/cross_market_innovation_20261017/data_provenance/EXPOSURE_LEDGER.json` → 同名缓存：早期审计范围与封存阶段；它不是后 BCM 的最新用途状态。
- `research/structural_innovation_20261017b/REPORT_ZH.md` → `SMP_REPORT_ZH.md`；`research/distribution_shape_diagnostic_20261017c/REPORT_ZH.md` → `SHAPE_REPORT_ZH.md`：开发范围与未运行确认的声明。
- 当前本地发布副本 `/workspace/scratch/be2216688c09/carbon_workspace/research/prior_transport_learning_20261018/REPORT_ZH.md`：最新 transport STOP 与未开启封存区声明。
- 当前本地副本 `/workspace/scratch/be2216688c09/carbon_workspace/research/event_feedback_innovation_20261017/data_reproduction/EEX_DATA_METHODS_ZH.md`：EEX 两个确认年份的实际起止日期及行数。

本轮没有访问清单所列 `local_only/confirmation_sealed` 或 `local_only/sciencedb_v6/confirmation_sealed` 价格路径。旧私有文件所在工作区已不可用；本状态核查不证明它们当前仍可直接读取。
