# 阶段 e 续接记录

## 用户目标

用户要求继续碳价预测模型创新研究，参考当前文献实际实验做法，需要数据就下载，实验支持后再写论文。已授权研究代码、数值结果及草稿 PR 交付，不要反复请求既有权限。允许训练未来监督、过去窗口内双向处理、公开清洗基准与成熟标签更新；来源 QC 不自动等于模型评分暴露。不要把工作转成纯泄露审计，不把任意 1% 或全市场全比较显著当作统一发表标准。

## 已完成、不得重跑或改写的部分

阶段 d 已在 Draft PR25 交付并保持原字节；此前多个关闭方向及失败结果不重开。本轮阶段 e 原始 max80 的三份模型/运行源、数据身份与统计计划已冻结；另加 max320 包装器只扩预算一次，完整保留全部五个非线性模型与两个市场。原始 244 拟合＋补充 170 拟合全部完成，合计 44,568 个正梯度更新。没有正在后台继续运行的训练任务，也没有已通过证据门槛的正结果论文。

本轮状态：`SIGNED_RESPONSE_LEARNED_BUT_CONTEXT_DURATION_INCREMENT_UNSUPPORTED`。

## 最重要的结果

| 对象 | 天津 Q | CEA Q |
|---|---:|---:|
| 原始主模型 | .920781190 | .878516090 |
| 原始价格 Ridge | .902010973 | .851061816 |
| 原始静态持续度 | .903970526 | .876991713 |
| 原始条件振幅 | .915516580 | .876827283 |
| 原始 SWEWT–GRU | .990924388 | .808417960 |
| 补充主模型 | .922250158 | .871423013 |
| 补充静态持续度 | .903957462 | .870625084 |

Q 是三个期限 MSE/RW-MSE 的均值，越低越好。原始主模型相对 RW 的家族校正区间为正；条件时长对关键控制的点估计无一致优势。不要把其相对神经参考的点排名直接写成显著。原始两个主模型的两个模式系数均负；合成表达允许正负组合不等于实际识别了这样的组合。

独立 NumPy/KKT/选择复放及 50k 配对成块统计全部完成。预算审计对 160 配置的 9,736 个共同 epoch 前缀差为零。7/10 个补充模型仍选 320 上限，不能声称全局最优或排除一切优化不足；按补充 stop_rule 不再继续扩大这一候选的训练预算。结果后去门＋同 alpha 重解训练头是诊断，不能升格为新的确认模型。

## 数据与暴露

天津官方日总成交均价：1,077 行，196 个共同评价原点，全部开发。

CEA 官方转载主表：1,251 行，231 个共同评价原点。此前原值/QC 已见但项目证据中无 CEA 模型评分；仅原始阶段 e 具有该新增评价身份。本轮完成之后，CEA 所有后续模型修改/预算敏感性都是开发；不能再把它叫新的未见确认。两市场同为元/吨的公布总成交均价，不是 close。H1/H5/H20 是随后观测记录，天津最大记录间隔 319 天。逐日发布版本/完整交易日历未认证。

私有运行根：`/tmp/carbon-signed-response-runs-20261019e`，子目录 `core`、`neural_references`、`budget320`。实际 Python：`/tmp/carbon-research-env/bin/python`，EWT `PYTHONPATH=/tmp/carbon-forecast-20261019d-deps`，数值库线程 1。原始行情位于 `/tmp/carbon-next-market-data-20261019d/tianjin_body.js`。天津标准数据位于同目录；CEA 标准数据位于 `/tmp/carbon-signed-response-data-20261019e/CEA_OFFICIAL_MEAN.csv`。完整哈希/命令见复现说明。

逐日原值、NPZ 和二进制权重不在公开树；自行训练的数值 state_dict 已授权公开并完整导出，不能将其误报为必须删除的原始数据。不要重试此前被拒绝的原始档案上传，也不通过拆包规避既有审批结果。

## 创新边界与下一项实际问题

几何阻尼、输入依赖持续率、低秩分布滞后与变量投影都有直接文献先例。本轮只能检验条件时长是否超出静态、条件振幅、自由期限头及强基线，当前结果未支持它。原始 11 项近邻阅读范围和本轮未复现的文献方法均已明确记录。

下一项建议仅保留“真实交易渠道构成监督是否提供额外预测信息”：总均价包含挂牌协议、大宗协议与可能存在的单向竞价，未来成熟的渠道份额可作为训练监督，部署只用过去信息。先取得真正的官方分项面板，验证它能与原总量/总额一致对齐；不能把天津网站两张 CEA 总量表解释成两渠道，不能漏掉单向竞价，不能将挂牌价直接叫有效价。教师未来信息本身已有先例，不是创新结论。最小对照应区分额外数据、普通多任务、一般蒸馏与未来构成监督；详见 `literature/NEXT_SINGLE_CANDIDATE_ZH.md`。

这个候选还没有模型结果，也没有冻结新的训练选择。数据可行性当前状态为 `STOP_PUBLIC_ENTRY_HTTP403_NO_CHANNEL_PANEL`：官方查询入口实际返回 HTTP403，取得的逐日渠道行数为 0，覆盖与对齐未执行。详细收据见 `provenance/NEXT_CHANNEL_DATA_FEASIBILITY.json`，独立于本轮 414 个拟合，不混入原评分。该失败来自源站访问拦截，工具网络授权没有被自动审批拒绝。

## 文件与 Git 身份

- 全报告：`REPORT_ZH.md`。
- 复現：`REPRODUCE_ZH.md` 与 `source/reproduce_stage_e.py`。
- 结论：`FINAL_DECISION.json`。
- 原始独立审计：`audits/INDEPENDENT_STATISTICAL_AUDIT_ZH.{md,json}`。
- 补充审计：`audits/BUDGET_SENSITIVITY_AUDIT.{md,json}`。
- 原始本地冻结：`eff4540abfeb63e599859459d0d188d65fd637a3`，tree `b36bf3b5301e3b5728460ab9b8f4e0b2c2960693`。
- 补充本地冻结：`c12443a7adeb36f48683290287f85ac7760ee833`，tree `d29714d8568e01be42c6caabb6382d9beab2c27b`。
- 分支：`research/signed-response-search-20261019e`。
- 远端对应：`provenance/LOCAL_REMOTE_FREEZE_MAPPING.json`；最终交付回执另列实际 PR、head 和内容树。

已推送并创建 [Draft PR26](https://github.com/124-creator/carbon-forecast-model-research/pull/26)。完整结果的本地提交为 `015c9b3d891eb56ad3c6c5a078d43ef3ac4189da`，远端提交为 `d999f26db91d8b581b14aa42292d7db4782afaf4`，内容树均为 `f9b34244e027d3e5607b05284c8d1d0936e3aeee`；REPORT、FINAL_DECISION、REPRODUCE 的远端内容和 Git blob 身份也已读回核对。交付元数据及此链接随后另行提交，实际最新 head 应读取 PR/分支，不能把内容提交 SHA 直接当成永远不变的 head。详见 `provenance/GITHUB_DELIVERY_RECEIPT.json`。

远端镜像在拟合之后，本地 commit SHA 与远端 commit SHA 可以不同，必须通过同内容 tree 比对。不得把本地 SHA 当作 GitHub head，也不得称后续远端上传是拟合前外部预注册。仅创建草稿 PR，不合并旧 PR25 或本轮草稿。
