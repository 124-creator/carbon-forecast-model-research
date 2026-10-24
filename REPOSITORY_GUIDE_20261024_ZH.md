# 碳预测研究仓库：总导航与科研状态（2026-10-24）

> **作用：让新研究员找到正确证据；不是新的方法、事前协议或模型评分报告。** 以 `main` 提交 `70452e37896a12bd29d8fd4cb029ed0161a13d29` 和当时 GitHub PR API 为整理快照；中国地方市场补采内容在尚未合并的 [Draft PR #32](https://github.com/124-creator/carbon-forecast-model-research/pull/32) 中，不可假装已在 main。新页面不更改任何历史冻结文本与失败原点。

## 1 十分钟阅读路线

**如果接手当前“ETF 波动率结构方法”问题，严格按原入口和冻结顺序：**

1. [2026-10-22 唯一权威接手页](START_HERE_PRO_20261022_ZH.md)。
2. [五路径结项](research/ROUND_CLOSURE_20261022_ZH.md)。
3. [结构实验事前协议](research/structure_v2_20261022/PROTOCOL_ZH.md) 与 [报告](research/structure_v2_20261022/REPORT_ZH.md)。
4. [全部判定原值 `pairwise_equal_asset.csv`](research/structure_v2_20261022/pairwise_equal_asset.csv) —— **数值以此为准，不用四舍五入摘要替代**。
5. [部分反馈终结结论](research/volatility_range_20261022/FEEDBACK_ROUND_CONCLUSION_ZH.md)。
6. [广州风险迁移](research/gdea_risk_feedback_20261022/REPORT_ZH.md)。
7. [EEX 拍卖审计](research/eua_auction_20261022/audit_report.md)。

**如果接手“扩大中国市场及政策数据”：** 先看 [数据资格汇总](docs/DATA_ACQUISITION_GATES_20261024_ZH.md) → [PR #32 数据说明卡](https://github.com/124-creator/carbon-forecast-model-research/blob/data/china-pilots-policy-20261023/research/china_market_expansion_20261023/OPEN_SOURCE_DATASET_CARD_20261023_ZH.md) → [停止条件](https://github.com/124-creator/carbon-forecast-model-research/blob/data/china-pilots-policy-20261023/research/china_market_expansion_20261023/DATA_READINESS_STOP_RULE_20261023_ZH.md)。

**如果查历史价格/分布模型、GRN/多市场方法：** 从 [历史方向状态](research/neural_model_restart_20261021/CURRENT_RESEARCH_STATUS_ZH.md)、[分布建模第六轮阶段报告](research/neural_model_restart_20261021/FINAL_REPORT_ZH.md)、[多市场 2020 年代面板开发记录](research/fullcycle_h20_20261020/REPORT_ZH.md) 进入，**不要拿该路线的价格 R²/MQS 与 ETF 方差 QLIKE 直接排榜**。后者存在已见数据、窗口扫描及政策口径等不同限制。

## 2 研究对象必须分开

| 研究线 | 资产/产品 | 任务与主要度量 | 截至快照的科研状态 |
|---|---|---|---|
| A. ETF 方差（目前冻结结项） | KRBN、KEUA、KCCA：**美股碳 ETF** | 未来 h=1/5/22 个本地观察日的平均收益平方；QLIKE，2025 h5/h22 按事前门槛 | **五条技术路径均未通过；无可发表主模型**。2026 已被看过，不构成新的确认 |
| B. 中国/欧洲价格预测 | 广州、湖北、上海、全国 CEA 与欧洲价格等**不同产品**；见每个研究的面板字段 | 价格水平/收益/预测分布；RW、MQS、CRPS、pinball 等（不同路线不共用评测合同） | 大量历史开发/诊断/停止记录，具体以各路线协议和后续更正为准；不能据个别旧 R² 冒称主模型胜出 |
| C. GRN 风险/价格区间研究 | GRN 为 **ETN**，非上述 ETF | Parkinson range 代理与区间覆盖、QLIKE/分位损失；零 range 有特殊含义 | 历史状态见 `neural_model_restart_20261021`；曾有未完成的第22轮计划记录，不能当本次实时运行证明 |
| D. EEX 拍卖 | 官方一级市场 **EUA spot auction**（T3PA）与 EUAA（EAA3）需分开 | 需求压力、日期与市场时区核对 | 来源审计已做；本轮需求预测路径**未建模即停止**，不是 ICE Endex 期货日结算价 |
| E. 中国地方试点及全国 CCER 政策 | BEA/SHEA/TJEA/GDEA/HBEA/SZEA/CQEA/FJEA、CEA、CCER 需要分表 | 原始报价/成交量/渠道/政策首次可得时间的获取与核验 | **数据-only、未完成准入**，资料在 Draft PR #32；0 新拟合/QLIKE |

## 3 唯一应被正确复述的 ETF 结论

2025 h=5、h=22 **各自**须同时击败 `EWMA97` 与 `GARCH11_t`，且逐个基准至少 **2/3 资产改善**；不是挑较好期限、事后平均或改资产构成的门槛。

| 2025 比较（候选减基准，QLIKE 越小越好） | h5 等资产 ΔQLIKE | h5 改善数 | h22 等资产 ΔQLIKE | h22 改善数 |
|---|---:|---:|---:|---:|
| A2 − K1（同信息直接 HAR） | −0.028421 | 3/3 | −0.070565 | 3/3 |
| A2 − EWMA97 | +0.021889 | 1/3 | +0.037075 | 1/3 |
| A2 − GARCH11_t | −0.102939 | 3/3 | −0.153654 | 3/3 |

- A2 的形式：`gap²/id²` 分别多期 HAR，用 `f=Ĝ+Î+2ρ̂√(ĜÎ)，ρ̂=tanh(θ·x)` 有界合成。**它对直接 HAR 一致改善，但没过 EWMA97 门槛；输入表示、函数类、拟合顺序与有界约束的单独贡献尚未识别。**
- A3 的独立反例：**18** 个原点违反二阶矩交叉项边界，其中 **10** 个产生非正预测。保留全部异常，非正 QLIKE 未定义，不删除、不断尾后算平均。
- 上表均来自 [固定 CSV 原值](research/structure_v2_20261022/pairwise_equal_asset.csv)；是历史样本描述，**无提供区间的统计推断**。不能写“显著优于”、不能对负损失水平报百分比变化，更不能把“未通过”写成“已证明无效”。
- 2024、2025、2026 都被既往模型研究查看过；**不是未触碰独立留出**。对未来新数据的新实验需独立先冻结，不能事后修改期限/资产/基准/区间。

## 4 当前文件与 PR 的实际规模

- 该 `main` 快照：共 **7,959 个已提交文件**（其中 `research/` **7,772 个**），`research/` 有 **64 个一级子目录**，根目录 **103 个文件**，含大量 `SCRIPTS_*.py` 历史可执行脚本。
- PR 总数 **32**；**2 merged（#14、#15），30 open（29 Draft、1 非 Draft）**。#16—#31 是连续以前一个研究分支为 base 的**链式 PR**，#32 则独立面向 `main` 的中国数据 Draft；**不进行批量合并/关闭**。
- 目录清单：[64 个实验目录逐项索引](docs/RESEARCH_DIRECTORY_INDEX_20261024_ZH.md)；按时间线与研究线的结果归类总表：[全部实验总整理](docs/EXPERIMENT_OVERVIEW_20261024_ZH.md)。PR 精确 base/head：[PR 状态图](docs/PR_BRANCH_MAP_20261024_ZH.md)。新提交规则：[维护与归档约束](docs/REPOSITORY_MAINTENANCE_20261024_ZH.md)。
- `README.md` 中较早入口、手稿构想、历史拟合介绍仅为**时间顺序史料**；后来的 `START_HERE_PRO_20261022_ZH.md` 与冻结判定优先。2026-10-22 结项不是 2026-10-21 之前所有不同目标研究的数值总表。

## 5 数据尚需什么，何时停止

**不是“下载完成”。** 见 [数据采集门槛](docs/DATA_ACQUISITION_GATES_20261024_ZH.md)：地方八市场 SCEEX 转载的 **25,712** 行只证明历史转载原表行数，不证明成交渠道或相同价格口径；湖北/广东固定历史段需要官方分渠道导出与历史规则；国内最新原始行情截至 2026-10-13 的缺口仍在；政策候选 **19 项 / 17 原文可获取 / 0 项已认证历史首次公开版本**；版权及再分发权利未解决。

既然继续抓同一镜像不能解决语义/授权，应把受阻列入数据获取停止清单，申请交易所合法导出。当前可整理/复核公开元数据，但**不启动额外模型拟合**。达到单个拟定任务的数据准入后，先交 **≤60 行事前冻结方案与计算预算** 给作者确认，再决定是否运行。

## 6 整理完成后的使用规则

1. 找到一个实验时，按 **协议 → manifest/失败原点 → 逐资产原值 → 报告/后续更正** 的顺序判断，不按 README 里的“最好数字”选结论。
2. 历史失败与错误保留、计数，资料可公开与否按权利单独认定；公有 GitHub **没有自动给第三方原文开放许可证**。
3. 此导航只添文档和根部入口；不移动 `research/`、`data/`、`metadata/` 的任何文件，不对 PR #32 宣告完工，不创建 LICENSE，不替作者发论文成功声明。
