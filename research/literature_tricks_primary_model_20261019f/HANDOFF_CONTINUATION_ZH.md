# 阶段 f 接续记录：增强已执行，结果已可审阅

## 用户目标与持续授权

用户要求继续碳价创新模型实验，在已有主模型上借鉴发表文献认可的处理和训练技巧；具体方式要先查原文，再实际执行，而不能仅因将其称为“轻微泄露”就排除。用户也要求需要数据时下载，并在实验支持创新后推进论文。

用户已多次明确授权继续实验、提交代码和推送 GitHub，包括本轮最后的“继续完成之后推送 GitHub”。本阶段目标是完成并交付三类已冻结增强。允许创建结果分支和草稿 PR；原有阶段 d/e 草稿不合并。后续不应把这份工作记录误解为要求再次询问同一推送许可。

## 当前完成状态

本轮四个正式拟合源、两个冻结计划在执行前后 SHA 均不变，三块已全部结束，没有尚待收集的拟合进程。完整实验、统计、独立方程核验、公开状态重放和报告已经完成。

机器状态：ENHANCEMENTS_COMPLETED_PRIMARY_PERFORMANCE_IMPROVED_DURATION_INCREMENT_UNSUPPORTED。

- 598 个新离线拟合：EWT 响应 408、EWT Ridge 54、密集路径响应 136。
- 实际梯度更新 80,398 次；额外初始动力学读出诊断 24 次。
- 成熟标签读出 13,632 次小规模求解，无新增非线性训练。
- 752 份正式运行 JSON 已完整公开导出。
- 90 个预测路径角色，统计记录 98 个指标身份行（含 8 个验证所选策略别名），不是 98 个独立实验。
- 598 checkpoint 的独立 NumPy 方程核验通过，最终预测最大绝对差 3.0241×10⁻¹¹。
- 全部 90 个面板从公开数值状态重放通过，公开汇总指标最大差 0。
- 两市场、两块长各 50,000 次配对循环块 bootstrap 完成，无无效 RW 分母抽样。

阅读顺序：[REPORT_ZH.md](REPORT_ZH.md) → [FINAL_DECISION.json](FINAL_DECISION.json) → [REPRODUCE_ZH.md](REPRODUCE_ZH.md)。最终 GitHub 身份与远端回读记录见 provenance/GITHUB_DELIVERY.json。

## 本轮实际结果与归因

Q 为 H1/5/20 的 MSE/RW-MSE 均值，越低越好。

| 主模型设置 | 天津 Q | 全国 CEA Q |
| --- | --- | --- |
| 原始 max320 | 0.9222501584 | 0.8714230126 |
| window_past | 0.9208077523 | 0.8543273302 |
| train_joint_test_window | 0.9983075460 | 0.9980798018 |
| full_series_joint | 0.6614319047 | 0.6314614210 |
| dense_path | 0.9121839369 | 0.8910102355 |
| 验证所选成熟标签更新 | 0.8762288653 | 0.9094661381 |

天津在线主模型验证选择 rolling252；全国选择 rolling126。天津测试 rolling126 更优，但它不是验证所选，不能替换主结果。

窗口 EWT 对两个市场主模型均有点值改善，家族区间跨零。全序列 EWT 的改善区间在两市场两个块长都为正，对应完整序列可用的回顾性设置。训练联合／评价窗口在两市场均变差；密集监督和成熟读出改善均不能跨市场保持。不能把不同市场较优技巧按测试结果拼为一个已确认系统。

关键机制证据必须保留：

- 全序列 EWT 下，静态和无成交量门在两市场点值均比主模型好；同输入价格 Ridge 更好，Q 为天津 0.4365939333、全国 0.3402859225。
- 天津窗口 EWT 主模型显著差于同输入静态模型，机制家族区间在两个块长均负。
- 天津不同家族分别验证选择在线策略时，主模型点值胜出；但都用主模型验证选择的 rolling252 后，静态 Q=0.8582437835、无成交量门 Q=0.8518825749，优于主模型 Q=0.8762288653。该同策略表是后结果描述性归因检查，无新 fit/CI，不属于冻结十五项机制比较。
- CEA train_joint 主模型比无成交量门存在极小正区间，但主模型 epoch=0、门全零、与静态／幅度门预测相同，且整体较原主模型差。这不是已学到门机制优势。

当前决定是保留真实的主模型提效与完整实验材料，尚未把条件持续时间门的独立创新写成已证实。没有通用“提高百分之几即达发表标准”的门槛。若继续研究，应围绕门机制增量提出新的具体问题，再固定下一轮比较；原 f 全部结果与计划保留。

## 冻结身份

| 对象 | 身份 |
| --- | --- |
| 本地冻结提交 | 9803953b7927d232608710aef21dafe870cf6cbc |
| GitHub 冻结提交 | 137d58e78ddf28c0163ab8d1cadf50b74e79ef67 |
| 相同内容树 | 08e25bb2cd1ec3387e2c14c0ac8cd016dfbb89d3 |
| GitHub 冻结时间 | 2026-10-19T14:49:14Z |
| 实验计划 SHA256 | 78893370f6862b1295ba9cfcc5e71fc89776944aeae39346de788a4e09cc8880 |
| 统计计划 SHA256 | 9c8d3188b5e6235f90f47e54fdd9192e974d71165926936c272f793405eff9d3 |

分支为 research/literature-tricks-primary-model-20261019f，草稿 PR 目标为 research/signed-response-search-20261019e。阶段 e 父提交本地为 249ce3045726afd71ecfa5f76269ef56ead1c4f3，远端为 ed2d0edc73eacea558640a213ecbe9b3808da93e，树相同。阶段 e [PR #26](https://github.com/124-creator/carbon-forecast-model-research/pull/26) 与阶段 d [PR #25](https://github.com/124-creator/carbon-forecast-model-research/pull/25) 保持未合并。

公开结果提交与本地提交可能因提交元信息而具有不同 SHA；推送核验以远端分支头、完整内容树相同及关键 blob 为准。不能把本地 SHA 冒称成 GitHub commit SHA。

## 数据与本会话运行位置

完整 checkout：

/workspace/scratch/562d14c1693a/carbon-research

正式 f 私有运行目录：

/tmp/carbon-primary-model-tricks-runs-20261019f/{ewt,dense_path,matured_readout}

原始参考运行目录：

/tmp/carbon-signed-response-runs-20261019e/{core,budget320,neural_references}

公开参数重放输出：

/tmp/carbon-stage-f-public-state-replay-20261019f

同 SHA 数据：

- 天津：/tmp/carbon-next-market-data-20261019d/TIANJIN_OFFICIAL_MEAN.csv，SHA256 85ca07648855db00fbf67c4e1c083db715f62ab829f74c143c3b3f67da21252a。
- 全国：/tmp/carbon-signed-response-data-20261019e/CEA_OFFICIAL_MEAN.csv，SHA256 f9f0e09b7500b2bdb4ec8b16c61d5d5fa2b64de63a24c0cf7bd1d269aa2be75d。

两个评价尾段都已暴露。目标是官方每日总成交均价，不是收盘价、挂牌价或自行用成交额重算的价格。H1/5/20 为随后观测记录步长；天津存在长日历空档。该身份在下一轮也应继续保留。

Python 3.12.14 环境为 /tmp/carbon-research-env/bin/python；EWT 依赖为 /tmp/carbon-forecast-20261019d-deps。各线程数为 1，实际科学计算包版本见复现说明。公开复现入口不要求这些原临时路径；需要完整 checkout、相同官方数据和对应软件依赖。

## 交付文件与前置检查历史

四个正式拟合源与两个事前计划不因写报告而修改。导出器、公开重放器、独立统计和方程审查在交付阶段另行保存自身 SHA，其作用与正式拟合区分清楚。READOUT_METHOD_AUDIT_ZH.md 的交付修订只追加公开状态重放入口说明，不改变公式、信息边界或运行前方法审查。

正式前置检查是 EWT_PREFLIGHT_FINAL.json、DENSE_PATH_PREFLIGHT_FINAL.json 与 MATURED_READOUT_PREFLIGHT_FINAL2.json。早期 DENSE_PATH_PREFLIGHT.json、MATURED_READOUT_PREFLIGHT.json、MATURED_READOUT_PREFLIGHT_FINAL.json 是实施过程中的旧前置检查，保留为历史；它们不是最终冻结 source/manifest 的通过凭据。最终身份以冻结计划及三个 FINAL 检查为准。

公开树保留全部训练轨迹、数值参数和模型级汇总。逐日原价、逐原点预测、逐期在线头及 PT/NPZ 不入仓；公开状态重放已在相同数据上真实运行。完整从头 598 次重训尚未封装成摆脱旧物证绑定的单一命令，这个边界在复现文档中明确说明。
