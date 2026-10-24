# 全部实验总整理（2026-10-24，按结果归类）

> 快照基准：`main` 提交 `70452e37896a12bd29d8fd4cb029ed0161a13d29`，另含 Draft
> [PR #32](https://github.com/124-creator/carbon-forecast-model-research/pull/32)
> 分支 `data/china-pilots-policy-20261023` 的未合并内容。本页在
> [目录索引](RESEARCH_DIRECTORY_INDEX_20261024_ZH.md)（只列路径）之上补一层
> **结果归类**：每条研究线做了什么、结果如何、哪些数据段已被消耗。
> 本文只做汇总，不改变任何历史判定；数值以各目录冻结协议与原始评分文件为准。

## 一、一页总览

| 统计项 | 数值 |
|---|---|
| 实验目录总数 | 65（main 上 64 + PR #32 未合并 1） |
| 时间跨度 | 2026-10-07 → 2026-10-23（约 17 天） |
| `research/` 已提交文件 | 约 7,772 个 |
| 开放 PR | 32 个（2 已合并，30 开放，其中 29 为 Draft） |
| 通过发表门槛的主模型 | **0** |
| 一致正向的受控结果 | A2 分解结构（对同信息直接 HAR 3/3 改善，仍输 EWMA97） |
| 可独立引用的方法学证据 | A3 二阶矩反例；区间聚合效应 H1；GDEA h5 期限结构 |

**五条研究线终局判定**（沿用 [总导航](../REPOSITORY_GUIDE_20261024_ZH.md) §2 分类）：

| 研究线 | 对象 | 任务 | 终局状态 |
|---|---|---|---|
| A. 碳 ETF 方差 | KRBN / KEUA / KCCA（美股 ETF） | h=1/5/22 平均收益平方，QLIKE | **五路径全部未过门槛，已冻结结项**（2026-10-22） |
| B. 中/欧碳价预测 | GDEA、CEA、EUA 作者快照等 | 价格水平/收益，MSE 等 | 大量开发记录，无确认性胜出；广州 2024–2026 确认集已烧毁 |
| C. GRN（ETN）区间/波动 | GRN（Barclays ETN，非 ETF） | Parkinson range 代理、区间覆盖 | 第 13–22 轮全部无确立改进；观测算子线已按预注册关闭 |
| D. EEX 拍卖 | EUA/EUAA 一级市场拍卖 | 需求压力 → 波动 | 原始筛查无一致信号，**未建模即停止** |
| E. 中国试点 + CCER 政策数据 | 八试点 + CEA + CCER | 数据采集与核验 | **仅数据工程，0 拟合 0 打分**，准入门槛未过（PR #32） |

## 二、按时间线整理（65 个目录）

### 2026-10-07 ｜ 第三轮 tian 实验（1 个目录）

[`aprime_gate3_20261007`](../research/aprime_gate3_20261007/)：Aprime 第三轮资格门，
是后续 Stage C 独立审查的基础交付（本地另有未入库的发布归档包）。

### 2026-10-13 ｜ 早期局部模型搜索与探针（19 个目录）

主题：在碳价序列上做候选模型搜索、信息探针与阶段审查。代表目录：
`stageC_20261013_submitted`（原 ZIP 225 份文件的独立复核查验）、
[stageC_20261013_review](../research/stageC_20261013_review/INDEPENDENT_REVIEW_ZH.md)、
`stageF_20261013_review`、`local_candidate_search` / `local_candidate_next` /
`local_structure_search_next` / `local_mature_feedback` / `local_memory_diagnostic`、
`asof_information_probe`、`causal_frequency_baseline`、`conditional_distribution_probe`、
`horizon_consistency_probe`、`broader_direction_scan`、`deep_direction_review`、
`expanded_model_search`、`multi_angle_model_search`、`persistent_residual_search`、
`decomposition_direction_selection`、`pro_data_packet`。

**关键已核实结果**（Stage C 独立审查）：候选模型 relative MSE = 1.8197（比随机游走差 82%），
普通 MLP 1.5263、全局时间尺度 1.6535、局部漂移 0.9924（但 2022 年差 5.4%）；
**候选被数值支持地淘汰**。审计另确认：作者 GCP 表与官方快照 1,915 个共同日期中
1,881 个价格一致、34 个不同，不能直接混用。

### 2026-10-14 ｜ 学习路线与主干模型时代（18 个目录）

主题：NHITS 官方基线、分段价格主干、Torch 主干、循环响应、跨市场训练、外生信息路线、
观测时钟、覆盖接口审计等。代表目录：`official_nhits_baseline`、`segmented_price_backbone`、
`torch_segmented_backbone`、`recurrent_response`、`signed_bridge_forecaster`、
`causal_multimarket_forecaster`、`cross_market_training_route`、`direct_learning_route`、
`exogenous_information_route`、`observation_clock_model`、`local_coverage_model`、
`coverage_interface_audit`、`literature_code_model_search`、`shared_private_model_search`、
`structured_response_search`、`next_model_route`、`model_route_reset`、`learning_route_review`。

**关键裁决**（learning_route_review）：所审查的学习机制落入已有方法（Tangent Prop
非零方向导数、TempoPFN 类生成先验），**不保留为独立贡献候选，不训练**。

### 2026-10-17 ｜ 结构创新与分布形状（6 个目录）

`structural_innovation_20261017b`（775 文件，结构创新主轮）、`cross_market_innovation_20261017`、
`distribution_shape_diagnostic_20261017c`、`event_feedback_innovation_20261017`、
`web_pro_execution_20261017` / `web_pro_handoff_20261017`（Pro 执行交接）。
无通过门槛的结果。

### 2026-10-18 ｜ 拍卖清算与先验传输（2 个目录）

`auction_clearing_innovation_20261018`（含完整 docs/ 方法学预审材料）、
`prior_transport_learning_20261018`（480 文件）。无通过门槛的结果。

### 2026-10-19 ｜ 文献对齐与门控训练（8 个目录）

`literature_aligned_model_search_20261019d`（含 ScienceDB 七试点数据来源账本）、
`literature_tricks_primary_model_20261019f`（799 文件）、`anchored_gate_training_20261019g`、
`conditional_persistence_20261019b`、`distribution_update_learning_20261019`、
`forecast_geometry_filter_20261019c`、`ohlc_path_learning_20261019`、
`signed_response_search_20261019e`。无通过门槛的结果。

### 2026-10-20 ｜ 全周期数据扩容（4 个目录）

| 目录 | 主题 | 结果 |
|---|---|---|
| [fullcycle_h20_20261020](../research/fullcycle_h20_20261020/REPORT_ZH.md) | GDEA 全周期 2,839 条（2013–2026），h∈{1,5,20} 期限结构 | **h=20 不可预测；h=5 有增量**；但 **2024-01-02→2026-09-30 确认块在冻结前被遍历检视，已不再干净**（全程披露在报告 §0） |
| `data_expansion_20261020` | 数据扩充 | 记录性 |
| [results_recovery_20261020](../research/results_recovery_20261020/STATUS_ZH.md) | 结果恢复 | 恢复历史结果文件 |
| `complement_residual_20261020h` | 补充残差 | 单文件记录 |

### 2026-10-21 ｜ 神经模型重启：第 13–22 轮（1 个目录，713 文件）

[neural_model_restart_20261021](../research/neural_model_restart_20261021/CURRENT_RESEARCH_STATUS_ZH.md)：

| 轮次/实验 | 内容 | 结果 |
|---|---|---|
| 第 20 轮（R20 holdout） | joint vs continued 冻结确认 | 改善仅 0.0821%，区间跨 0，**未确立**；整体仍落后 MEM |
| 第 21 轮（R21） | Markov 状态 + Gamma 观测神经密度 | 开发期改善 0.4626% 但仍落后 raw/GRU；2026 诊断中神经密度增量未泛化 |
| 跨期稳定性审计 | 14 模型 × 2025/2026 各 3 块 | 跨期排序 Spearman 仅 0.170；**评测分辨率（0.0847）比已追效应量（0.0004–0.0025）大 34–212 倍**——单窗口胜负不构成有效推断 |
| 观测算子/测量误差线 | GRN 噪声梯度预注册检验 | C1 0/9、C2 2/6、绝对门槛 24 配置 0 通过，**按预注册规则关闭该线** |
| 区间（分布）实验 | iid vs 全局池 vs 条件池 | **H1 聚合效应强确认**（iid 使 h22 覆盖率掉到 0.553）；**H2 条件化被否证**（优势全部来自剔除 2020 的对照 B2）；最优简单法 B2 不构成方法创新 |
| 第 22 轮 | 风险强度 S 与非零区间 Z 联合路径学习 | 已启动训练后随 10-22 结项冻结，**无判定结果** |

撤回记录：早期 HAR-D 的 7–10% 改善及"显著"结论**已撤回**；第 13–19 轮保留为历史探索。

### 2026-10-22 ｜ 结项轮：五路径全部未过门槛（5 个目录）

事前冻结门槛：2025 年 h=5 与 h=22 各自须同时压过 EWMA97 与 GARCH11_t，
且每个基准 ≥2/3 资产改善。详见
[结项报告](../research/ROUND_CLOSURE_20261022_ZH.md)与
[接手指南](../START_HERE_PRO_20261022_ZH.md)。

| 路径 | 目录 | 结果 |
|---|---|---|
| 部分损失反馈在线组合（四/五专家、GDEA 迁移） | [volatility_range_20261022](../research/volatility_range_20261022/FEEDBACK_ROUND_CONCLUSION_ZH.md) + [gdea_risk_feedback_20261022](../research/gdea_risk_feedback_20261022/REPORT_ZH.md) | 未通过，方向不稳定；GDEA 迁移四单元全负（负迁移） |
| 门控 / 归一化尺度 | 同上 | 未通过，增量 1e-3 量级且跨年翻向 |
| **A2 隔夜-日内分解 + 有界合成** | [structure_v2_20261022](../research/structure_v2_20261022/REPORT_ZH.md)（1,968 次拟合，重建差 6e-16） | 未过门槛，**但为本轮唯一一致正向**：对同信息 K1 直接 HAR 改善 3/3（h5 −0.0284、h22 −0.0706），对 GARCH11_t 3/3 胜，对 EWMA97 仍差 +0.0219/+0.0371 |
| **A3 自由有符号交叉项** | 同上 | 反例：18 原点违反二阶矩边界、10 个非正方差预测（QLIKE 无定义，原样保留） |
| 双资产幅度混合 DM_pair | 同上 | 未通过，无可辨识增量 |
| EEX 拍卖需求压力 | [eua_auction_20261022](../research/eua_auction_20261022/audit_report.md)（EUA 1470 场 / EUAA 38 场审计） | 原始筛查无一致信号（Spearman +0.009/−0.031/+0.043，安慰剂最大且反向），**未建模即停止** |

另：[pilot_price_replication_20261022](../research/pilot_price_replication_20261022/README_ZH.md)
（试点价格复现核验，36 文件）。

### 2026-10-23 ｜ 中国市场数据扩充（PR #32，未合并，仅数据工程）

`research/china_market_expansion_20261023`：八试点 + CEA + CCER。
实测状态：转载历史 25,712 行可追溯；HBEA/GDEA 2018–2024 机械筛查通过但
**价格语义未认证**；政策候选 19 项 / 17 份原文 / **0 项 PIT 首见版本认证**；
最新 10 月行情未取得。**0 模型拟合、0 打分、0 ΔQLIKE**。已写好
《下一阶段资格门》（候选假设：面向已核验成交事件风险的两部模型）与
《数据停机准则》，均未冻结执行。

### 2026-10-24 ｜ 整理与导航（本分支 PR #33，纯文档）

`REPOSITORY_GUIDE_20261024_ZH.md` + `docs/` 四份导航（目录索引 / PR 图 /
数据门槛 / 维护约束）+ 本页。无新实验。

## 三、"干净确认集"消耗清单（战略资源现状）

| 数据段 | 状态 |
|---|---|
| ETF 2024 / 2025 / 2026（KRBN、KEUA、KCCA） | **全部已被查看，无未触碰留出集** |
| KRBN / KCCA 2026 | 第 20 轮已打开并完成一次冻结确认，此后只可作诊断 |
| KEUA | 已停止更新（数据止于 2026-03-20） |
| GDEA 2024-01→2026-09 | fullcycle_h20 轮在冻结前遍历检视，**已烧毁** |
| GRN 2026 | **唯一尚未打开的隔离段**（仅做过质量检查，未构造目标/评分） |
| GRN 开发期（至 2025 底） | 已用于训练 |

结论：当前所有已在手数据上，**不存在任何可支持"确认性优越"声明的留出集**；
未来确认价值只来自全新数据（ICE Endex 期货、官方授权导出的试点数据、
或 GRN 2026 的一次性冻结使用）。

## 四、可复用资产（不依赖"压过 EWMA97"即可引用）

1. **A2 受控对照**：同信息、同训练、同评分下，分解结构对直接 HAR 的 3/3 一致改善，
   并缩小对 EWMA97 差距 56%/65%——方法学短文的核心素材。
2. **A3 反例**：分量独立回归再相加在真实数据上破坏二阶矩一致性（18 越界 / 10 非正）
   ——约束必要性的直接证据。
3. **区间聚合效应 H1**：iid 假设使 h=22 覆盖率掉到 0.553，方差膨胀 3.84 倍（解析 3.80）
   ——可独立成立的描述性结论。
4. **评估分辨率审计**：效应量比评测噪声小 34–212 倍——对整个"单窗口胜负"范式的批评素材。
5. **数据审计方法学**：GDEA/官方快照逐日比对、EEX 拍卖审计、试点数据 PIT 认证流程、
   PR #32 数据集卡——数据论文雏形。
6. **GDEA 期限结构**：h=20 不可预测 / h=5 有增量（开发性结论）。

## 五、仍在进行 / 悬而未决的事项

| 事项 | 状态 | 依据 |
|---|---|---|
| PR #32（中国数据） | Draft，数据门未过 | `DATA_READINESS_STOP_RULE_20261023_ZH.md` |
| PR #33（仓库导航） | Draft，待合并 | docs 分支 |
| A2⊕EWMA97 补缺组合（+0.022/+0.037） | 已建议、未执行 | [接手指南](../START_HERE_PRO_20261022_ZH.md) §7 |
| 方法学短文（A2+A3） | 已建议、未动笔 | 同上 |
| R22 路径学习 | 训练已启动但随结项冻结，无判定 | `neural_model_restart_20261021` |
| ICE Endex 期货数据获取 | 未开始，需作者授权 | 接手指南建议第 1 条 |

## 六、一句话总结

17 天、65 个实验目录、五条研究线，**没有任何模型通过事前冻结的发表门槛**；
这不是搜索不足，而是当前数据的信息量上限（评测分辨率比效应量大 1.5–2 个数量级）
与强基准（EWMA97）共同决定的。手里真正的资产是 A2/A3/H1 三项可独立成立的方法学结论、
一套严格协议与审计记录、以及尚未打开的 GRN 2026 隔离段。
