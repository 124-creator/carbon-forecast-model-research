# GitHub PR、分支与合并关系清单（2026-10-24）

> 仅据 GitHub 当前 PR API（`main` 文档整理前的快照）；**这是状态索引，不是合并建议**。共有 32 个 PR：30 个仍 open（其中 29 draft）、2 个已 merged（#14、#15）。历史分支不等于尚无资料在 main：main 中可有同名文件，须逐 SHA/diff 判断实际差异。

## 一、已合并至 main

| PR | 内容 | 状态 |
|---|---|---|
| [#14](https://github.com/124-creator/carbon-forecast-model-research/pull/14) | 完成网页入口研究执行：42次正式拟合、PROCEED消融与独立审计 | merged 2026-10-20 |
| [#15](https://github.com/124-creator/carbon-forecast-model-research/pull/15) | research: 扩展四市场数据并完成 BCM-GRU 创新机制实验 | merged 2026-10-20 |

## 二、独立打开的历史候选（#1—#13，base=main）

此组是各阶段的历史实验归档请求；不要因为标题含“完成”就认为科学门槛通过。用户目前的主任务不要求清理/关闭。

| PR | 标题 | GitHub 状态 |
|---|---|---|
| [#1](https://github.com/124-creator/carbon-forecast-model-research/pull/1) | Stage D：低维稳定控制通过，联合投影无独立增量，调参阶段停止 | open |
| [#2](https://github.com/124-creator/carbon-forecast-model-research/pull/2) | Stage E：增益适配未通过合成机制门（源码上传受限，文档与结果已归档） | open / draft |
| [#3](https://github.com/124-creator/carbon-forecast-model-research/pull/3) | Stage F：直接符号交互网络有开发增量，但未超过低维强基线 | open / draft |
| [#4](https://github.com/124-creator/carbon-forecast-model-research/pull/4) | Stage G：有序稳定记忆未通过合成门；协议和结果归档（源码上传受限） | open / draft |
| [#5](https://github.com/124-creator/carbon-forecast-model-research/pull/5) | Stage H：跨时间条件分解完成288次拟合，未通过机制门；源码与记录完整归档 | open / draft |
| [#6](https://github.com/124-creator/carbon-forecast-model-research/pull/6) | Stage I：完成参考—循环交互实验与完整结果归档；机制失败且跨环境不逐值一致 | open / draft |
| [#7](https://github.com/124-creator/carbon-forecast-model-research/pull/7) | Stage J：后验积分循环单元完成双环境实验；机制未通过，完整数值材料归档 | open / draft |
| [#8](https://github.com/124-creator/carbon-forecast-model-research/pull/8) | Stage K：重叠依赖神经检索完成双环境实验；新增机制未胜简单读出 | open / draft |
| [#9](https://github.com/124-creator/carbon-forecast-model-research/pull/9) | Stage L：有向矩规则未胜普通闭式头；完整双环境实验与材料归档 | open / draft |
| [#10](https://github.com/124-creator/carbon-forecast-model-research/pull/10) | Stage M：持续时间响应未通过；低秩和固定基函数提供更强对照，完整材料归档 | open / draft |
| [#11](https://github.com/124-creator/carbon-forecast-model-research/pull/11) | Stage N：完成自适应期限节点实验与完整归档；新增自由度未胜简单变形 | open / draft |
| [#12](https://github.com/124-creator/carbon-forecast-model-research/pull/12) | Stage O：前推收益子空间学习未胜普通低秩；完整代码、双环境结果与审计归档 | open / draft |
| [#13](https://github.com/124-creator/carbon-forecast-model-research/pull/13) | Stage P：完成保持均值分布标签实验；未胜简单对照，完整材料归档 | open / draft |

## 三、链式 PR（#16—#31，非直指 main）

这些 PR 的 base **彼此接力**，不宜当作 16 个相互独立、可以任意顺序合入 main 的请求。若将来考虑合并，先从 #16 开始逐层核查祖先提交、实际 diff、数据/许可证与冻结结果；**本次不执行合并/改 base/删除分支**。

| PR | 标题 | head → base | 状态 |
|---|---|---|---|
| [#16](https://github.com/124-creator/carbon-forecast-model-research/pull/16) | Research: EEX data expansion and audited RRC / SOLID experiments | `research/revision-residual-20261017` → `research/cross-market-innovation-20261017` | open / draft |
| [#17](https://github.com/124-creator/carbon-forecast-model-research/pull/17) | research: 完成 SMP 结构候选实验，保留失败结果与独立审核 | `research/structural-smp-20261017` → `research/revision-residual-20261017` | open / draft |
| [#18](https://github.com/124-creator/carbon-forecast-model-research/pull/18) | research: 完成条件分布形状诊断、40次拟合与深圳数据来源核验 | `research/distribution-shape-20261017` → `research/structural-smp-20261017` | open / draft |
| [#19](https://github.com/124-creator/carbon-forecast-model-research/pull/19) | research: deliver prior-transport experiment and recovery audit (STOP) | `research/prior-transport-learning-20261018` → `research/distribution-shape-20261017` | open / draft |
| [#20](https://github.com/124-creator/carbon-forecast-model-research/pull/20) | Research: complete EUA stochastic-clearing experiments and publication route | `research/auction-clearing-innovation-20261018` → `research/prior-transport-learning-20261018` | open / draft |
| [#21](https://github.com/124-creator/carbon-forecast-model-research/pull/21) | Research: complete reported EUA price-range learning and independent audit | `research/ohlc-path-learning-20261019` → `research/auction-clearing-innovation-20261018` | open / draft |
| [#22](https://github.com/124-creator/carbon-forecast-model-research/pull/22) | Research: complete budgeted path-control distribution learning and independent audits | `research/model-innovation-expansion-20261019` → `research/ohlc-path-learning-20261019` | open / draft |
| [#23](https://github.com/124-creator/carbon-forecast-model-research/pull/23) | Research: conditional persistence experiments and GAS-Norm comparison (fixed candidate stopped) | `research/conditional-persistence-20261019b` → `research/model-innovation-expansion-20261019` | open / draft |
| [#24](https://github.com/124-creator/carbon-forecast-model-research/pull/24) | Complete forecast-geometry qualification: numerical repair, full audits, STOP | `research/forecast-geometry-filter-20261019c` → `research/conditional-persistence-20261019b` | open / draft |
| [#25](https://github.com/124-creator/carbon-forecast-model-research/pull/25) | 研究：文献对齐的碳价模型实验与天津官方后段检验 | `research/literature-aligned-model-search-20261019d` → `research/forecast-geometry-filter-20261019c` | open / draft |
| [#26](https://github.com/124-creator/carbon-forecast-model-research/pull/26) | research: complete signed-response model experiments and budget sensitivity | `research/signed-response-search-20261019e` → `research/literature-aligned-model-search-20261019d` | open / draft |
| [#27](https://github.com/124-creator/carbon-forecast-model-research/pull/27) | research: complete primary-model literature enhancements and audited results | `research/literature-tricks-primary-model-20261019f` → `research/signed-response-search-20261019e` | open / draft |
| [#28](https://github.com/124-creator/carbon-forecast-model-research/pull/28) | research: complete anchored-gate training enhancement and eight-cell development evaluation | `research/anchored-gate-training-20261019g` → `research/literature-tricks-primary-model-20261019f` | open / draft |
| [#29](https://github.com/124-creator/carbon-forecast-model-research/pull/29) | research: archive verified stage-g results and structural reassessment | `research/results-recovery-20261020` → `research/anchored-gate-training-20261019g` | open / draft |
| [#30](https://github.com/124-creator/carbon-forecast-model-research/pull/30) | research: expanded H1 carbon forecasting and EEX auction information-boundary experiments (i/j/k) | `research/expanded-single-step-20261020ijk` → `research/results-recovery-20261020` | open / draft |
| [#31](https://github.com/124-creator/carbon-forecast-model-research/pull/31) | research: audited probabilistic horizon reconciliation benchmark and journal readiness verdict | `research/vr_probabilistic_horizon_reconciliation_20261020` → `research/expanded-single-step-20261020ijk` | open / draft |

## 四、独立的中国市场数据采集 PR #32

- [#32](https://github.com/124-creator/carbon-forecast-model-research/pull/32)：`data/china-pilots-policy-20261023` → `main`；**open / Draft / 未合并**。它是数据来源、审计、脚本及政策 PIT 门槛的公开复现包，并非「八市场最新合法原始数据已下载完成」，更没有预测拟合。
- 详见 [数据获取状态](DATA_ACQUISITION_GATES_20261024_ZH.md) 与 [PR #32 数据卡](https://github.com/124-creator/carbon-forecast-model-research/blob/data/china-pilots-policy-20261023/research/china_market_expansion_20261023/OPEN_SOURCE_DATASET_CARD_20261023_ZH.md)。

## 维护纪律

1. 不批量关闭历史 PR、不强制 squash/rebase、不合并跨分支上的实验结果；已签收审计的哈希和运行配置保持不变。
2. 将来单独做 PR 归并前，逐 PR 给出：origin SHA、base SHA、新增/重复文件清单、许可证、冻结协议兼容性、会不会覆盖结果。
3. 本导航 PR 使用“open/merged”描述**GitHub 操作状态**，不代表研究上的“通过/未通过”。
