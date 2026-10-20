# 两个 runner 的补充比较接口

这份说明规定如何连接已有预测档案，不新增训练、参数搜索或确认数据。它是实现接口说明，不是独立审计结果。候选模型内部的机制比较与继续门槛以 [MARKET_PROTOCOL.json](../provenance/MARKET_PROTOCOL.json) 和 [aggregate_persistence_market.py](../source/aggregate_persistence_market.py) 为准；下面的跨模型表是另行标注的 **secondary contextual comparison**，不能替代内部匹配消融或修改其决策。字段与 family 定义分别依据 [候选 runner](../source/run_persistence_market.py) 与 [GAS runner](../source/run_gas_benchmark.py)。

## 1. 完整性和比较范围

候选实验须先完成 312 个神经拟合、13 组共享统计参考（39 个统计优化器），并核对相应 COMPLETE 清单、输入身份、checkpoint 和预测哈希。GAS 对照须完成 198 个神经拟合、22 个 GAS 统计拟合以及 154 份固定控制预测，核对其阶段回执、源码 fingerprint 和全部预测哈希。跨 runner 比较只在两套既定任务均完成后执行。

仅比较两个 runner 共有的 11 个开发 panel，期限为 5、20，共 22 个单元。panel 是 `EUA_LONG`、`GUANGZHOU`、`HUBEI`、`SHANGHAI` 与 2021–2023 的预定组合，排除 `SHANGHAI_2021`。深圳不进入此表：本轮没有为深圳训练 GAS 对照。

将候选私有根目录记为 `CANDIDATE_RUN`，GAS 私有根目录记为 `GAS_RUN`。实际路径由运行参数提供，不从报告内容猜测。

## 2. 档案路径与字段映射

主候选使用 `GRU16/conditional_joint005`，种子固定为 17、41、89。

| 内容 | 候选 NPZ | GAS/固定控制 NPZ |
|---|---|---|
| 相对路径 | `forecasts/{panel}__GRU16__conditional_joint005__s{seed}.npz` | `predictions/{panel}_h{h}/{family}/seed_{seed}.npz` |
| origin 索引 | `test_indices`，形状 N | `origin`，形状 N |
| origin 日期 | `test_dates` | `date` |
| 目标索引 | `test_target_indices[:, col]` | `target_index` |
| 目标日期 | `test_target_dates[:, col]` | `target_date` |
| 标准化目标 | `test_y[:, col]` | `truth` |
| 从标准化收益转为实际对数收益的单位 | `test_unit[:, col]` | `unit` |
| 起点价格（只用于私下核对） | `test_origin_price` | `origin_price` |
| 目标价格（只用于私下核对） | `test_target_price[:, col]` | `target_price` |
| 标准化 CRPS | `test_score_h{h}_normalized_CRPS` | `score_normalized_CRPS` |
| 价格 CRPS | `test_score_h{h}_price_CRPS` | `score_price_CRPS` |
| 预测均价平方误差 | `test_score_h{h}_mean_price_squared_error` | `score_mean_price_squared_error` |
| 随机游走均价平方误差 | `test_score_h{h}_rw_mean_price_squared_error` | `score_rw_mean_price_squared_error` |
| 其余评分 | `test_score_h{h}_{metric}` | `score_{metric}` |

`col` 必须由冻结的 `[1, 5, 20]` 找到：h5 对应列 1，h20 对应列 2。GAS 档案每份只含一个期限。两者的 `normalized_CRPS` 都已处于同一个 origin 的 `daily_sigma * sqrt(h)` 单位，不得再次除以单位，也不得拿 GAS 自己的条件尺度再标准化比较分数。

加载数值前先检查日期、panel 身份和既定来源边界。随后必须逐项核对 origin 和 maturity 索引、日期、目标与单位一致，且三个 seed 也一致；不能通过取交集或删除难例来修复不一致。GAS 的 `prepared/{panel}_h{h}.json` 中 `base_prepared_sha256` 应与候选该 panel 回执中的 `prepared_sha256` 相同。该检查把两套不同特征构造连接到同一组报价、切分和评价目标。

## 3. 已有对照 family

| family | seed 文件 | 含义 |
|---|---|---|
| `gas_gru_gmm` | 17、41、89 | GAS 标准化后训练的 GRU-GMM |
| `ewma_gru_gmm` | 17、41、89 | EWMA 标准化后训练的 GRU-GMM |
| `global_gru_gmm` | 17、41、89 | 训练期全局标准化后训练的 GRU-GMM |
| `daily_ewma_normal` | 0 | 共同 origin 日 EWMA 单位下的单个 N(0,1) |
| `gas_empirical101`、`ewma_empirical101`、`global_empirical101` | 0 | 训练期标准化残差的固定 101 个分位点原子，经对应 origin 的位置和尺度变换 |
| `gas_no_neural`、`ewma_no_neural`、`global_no_neural` | 0 | 对应标准化器加固定初始三分量 GMM；这些不是单个正态分布 |

所有 family 都已经在既定 runner 中生成。增加它们在表中的一行不会产生新训练预算，但应把主上下文比较与补充固定控制明确列名，不按结果选择展示对象。

经验原子的连续密度 NLL 未定义。原子档案省略该指标时，表中记为“不适用”；不得补 0、加任意密度平滑或据此删掉整个模型的 CRPS 结果。连续密度模型的 NLL 比较使用差值，不能把可能为负的 NLL 当作相对比值分母。

## 4. 评分聚合与配对区间

每个神经模型先在同一 origin 平均三个 seed 的**评分**，固定控制使用单个 seed 0 的评分：

\[
\bar s^{(m)}_{c,t}=\frac13\sum_{s\in\{17,41,89\}}s^{(m,s)}_{c,t}.
\]

这里不平均模型 CDF，也不先平均预测均价再计算平方误差。seed 平均评分评价的是重复训练模型的平均表现。浮点实现使用 `first + mean(values - first)`：三个完全相同 seed 的评分应保持与单份参考逐点完全一致，避免不必要的重复求和再相除误差。这个实现没有改变算术均值的定义，也没有引入判胜容差。

每个单元的 CRPS 相对变化为

\[
R_c=\frac{N_c^{-1}\sum_t\bar s^{(\mathrm{candidate})}_{c,t}}
{N_c^{-1}\sum_t\bar s^{(\mathrm{control})}_{c,t}}-1,
\qquad R=\frac1{22}\sum_cR_c.
\]

这不是逐 origin 比值的平均，也不是先合并 22 个单元的评分再取一个比值。严格胜出使用原始浮点值 `R_c < 0`；不按显示位数判胜负。若某个补充比值的分母确为 0，应明确报告该比值未定义，保留其单元与绝对分数，不悄悄缩小固定样本集合。

区间使用 `numpy.random.default_rng(20261019)`，11 个 panel 按字母序，每个 panel 生成 1,999 个长度为 N 的循环块样本，块长固定为 40。先抽取 `ceil(N/40)` 个起点，将连续块按模 N 连接后截取前 N 个 origin。该 panel 的所有期限、候选和对照共用同一组索引；每次重采样重新求单元均值、相对比值，最后等权平均 22 个单元。取线性插值的 2.5% 与 97.5% 分位点。

报告中须说明：这是固定年度样本内的名义配对区间，没有建模不同市场或不同年份之间的相关性，也不是对独立市场总体的外推保证；多条上下文比较不是同时置信区间。

## 5. 表格输出建议与解释边界

公共表每行至少给出 comparator、22-cell 平均相对标准化 CRPS、严格改善/持平/恶化单元数、名义区间、逐单元 CRPS 比值，以及可用的价格 CRPS、均价 MSE/RW、覆盖率与区间宽度。价格 CRPS 和原始价格宽度可逐单元报告；跨币种总结优先使用同单元相对价格 CRPS或标准化宽度，不能把欧元与人民币的原始分数直接平均成一个可比较的价格指标。公共文件不得写出逐 origin 价格、收益、预测、评分或参数。

此表提供相同评价目标和 origin 下的总体表现背景，**不是等信息、等参数或等训练目标的机制消融**：

- 候选 GRU 使用 20×3 的日收益、日波动率与日历间隔特征；GAS/EWMA/Global GRU 使用相应期限下已成熟的 20 个目标收益标准化值，形状为 20×1。
- 候选联合训练 h1/h5/h20，主臂还有 20 步路径 NLL 辅助目标；GAS 对照按 h5 和 h20 分别拟合、以对应端点 CRPS 训练。因此不能仅因 epoch 数相同便称总计算或监督信息相同。
- 候选受固定状态方差结构、零对数收益均值和共享未来转移核约束；GRU-GMM 可学习分量位置、尺度与权重。神经参数与标准化器参数的数量也应分别据执行回执报告。
- 候选 `conditional_endpoint` 与 `independent_horizon_endpoint` 才是相同 endpoint 目标下的共享核限制对照，后者仍多 17 个神经参数；`conditional_joint005` 与 `conditional_endpoint` 则评价预定路径辅助训练的变化。跨 runner 的胜负不能替代这些解释。

此接口不修改候选的内部继续门槛，不触发新训练，也不以跨模型表的结果决定是否展示已经完成的消融。
