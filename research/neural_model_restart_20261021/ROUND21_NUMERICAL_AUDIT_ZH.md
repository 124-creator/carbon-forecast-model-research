# ROUND21 数学与数值审计

本次只读取冻结源码与纯 synthetic Gamma 校准表。未读取行情、2026、开发训练结果或 checkpoint；未 import/执行 R21；未训练、重拟合、生成 Brownian 路径或构造大 grid 矩阵。唯一数值补充是少量16点 GH 单状态公式、标量网格误差界及2×2后验代数例子。全部输入哈希与未舍入数值见[审计 JSON](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/ROUND21_NUMERICAL_AUDIT.json>)。

审计对象 [R21 源码](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/round21_gamma_observation_filter.py>) 的 SHA256：`86941e4ddb0806ca5a77d1014c750da71a34cbbf6f663f68383c4fe6a930d5db`。

结论：没有发现完整 M 后验遗漏 prior(S)/Gamma 径向证据，或所核公式路径使用未来信息的错误。确认 conditional/global 矩不完全闭合、状态离散化/截断、未来量与风险创新独立、几何 tolerance 以及 prior-log floor 等近似；不能把这些近似写成精确生成模型性质。本审计不修改当前候选。

**一、条件校准积分矩与 global cM 的差异**

令 `c_I(M)=Σ_s p(s|M)c(s,M)`，`m2_I(M)=Σ_s p(s|M)[v(s,M)+c(s,M)²]`。第二式对应 κ=1。参考全局均值记为 `c_G(M)`；全局原始二阶矩使用训练数据的 `mean(P²)`。

| M | global cM | Σp·c_cell | 均值相对差 % |
|---:|---:|---:|---:|
| 1 | 0.361057640 | 0.361057640 | 0.000000 |
| 2 | 0.441727896 | 0.441770427 | +0.009628 |
| 4 | 0.538714377 | 0.538441356 | −0.050680 |
| 8 | 0.634823248 | 0.634156806 | −0.104981 |
| 16 | 0.721107698 | 0.720306541 | −0.111101 |
| 64 | 0.846377139 | 0.845674403 | −0.083029 |
| 512 | 0.941852605 | 0.941266690 | −0.062209 |

| M | global mean(P²) | Σp·(var+c²) | 二阶矩相对差 % |
|---:|---:|---:|---:|
| 1 | 0.392796322 | 0.392800696 | +0.001114 |
| 2 | 0.452616052 | 0.452673344 | +0.012658 |
| 4 | 0.560424729 | 0.560149297 | −0.049147 |
| 8 | 0.695268350 | 0.694107320 | −0.166990 |
| 16 | 0.836668537 | 0.834966603 | −0.203418 |
| 64 | 1.072724250 | 1.070820091 | −0.177507 |
| 512 | 1.276333531 | 1.274555438 | −0.139313 |

最大均值差为0.111101%，最大二阶矩差为0.203418%。未收缩的 cell 原始均值/二阶矩按经验频率 `n_s/N` 汇总时，全局矩恢复到浮点舍入误差（≤4.45e−16）。收缩、Dirichlet 概率和方差的 Bessel 口径共同打破这一有限样本分区恒等式。JSON分别列出经验频率下的收缩/Bessel项和 Dirichlet 权重变动项；没有结构可能但训练数为零的格。

M=1的二阶矩小差只来自统计口径：`global_var_ddof1+c_G²=0.3928006955025687`，与条件积分完全相同；它比 raw `mean(P²)` 多4.37396793e−6。这不应归咎于缺失的 shape 格。其余 M 还包含收缩和 Dirichlet 的影响。

条件 Gamma 联合密度仍是合法归一化概率模型：p(shape|M)行和为1，所有支持内 Gamma 密度均正规化。但它的积分均值是 `S*c_I(M)/c_ref`，不是严格的 `S*c_G(M)/c_ref`。[future factor](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/round21_gamma_observation_filter.py#L185>) 使用 global cM，因此 current likelihood 与 future mean factor 是近似校准耦合。

在同一 S、同一非负 prior q、同一 c_ref/hurdle 下，两种均值因子的相对差是 `ε_M=c_I(M)/c_G(M)−1` 的正权重平均，范围为 −0.111101% 到 +0.009628%。对未来 volume 的正权重积分也保持此范围。**这只是固定状态和因子系数的代数界，不是改变 likelihood 后重新过滤、重拟合或 forecast score 的总误差界。**

κ不改变上述均值差。任意κ下条件二阶矩系数应写成 `Σp·[v_cell/κ+c_cell²]`；表中二阶矩差仅对应κ=1，不代表已拟合κ下的实际后验方差误差。

**二、GH16、S坐标插值与边界指标**

[R21 transition](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/round21_gamma_observation_filter.py#L120>) 对应未截断工作过程：

```text
S_next = (1−phi)*mu + phi*S*exp(sqrt(q)*Z−q/2)
E[S_next|S] = mu + phi*(S−mu)
```

GH节点的移动值随后 clip 到 `[exp(−8),exp(8)]`，再按 S（非logS）线性分配概率。这个插值对**已经 clip 的 GH 条件均值**保真；当整行16个节点都在边界内时，才可把它视为未截断工作过程的均值保真，仍带有限 GH 与浮点误差。`interior_rows` 只表示16个节点未越界，不表示连续正态尾越界概率在数学上为零。

均值误差应分解为：

```text
实际离散均值 − 未截断解析均值
= 插值舍入误差 + clip后的GH均值−原GH均值 + GH均值−解析均值
```

GH对平滑指数的一阶矩可极精确，但越界指示函数不平滑。仅作公式演示：mu=1、phi=.98、q=.03、S=exp(8)时，单行 GH16 clip mass为0.5，连续 lognormal 越界概率约0.419488；clipped GH均值误差为−176.2829，连续截断均值误差为−175.2455。这是极端单行、非拟合参数的例子，**不是模型有50%截断，也不是实际 posterior 位于该行的证据。**

实际记录值应这样解释：

- `posterior_weighted_clip_mass[t] = filtered_posterior[t−1] @ row_clip_mass`，在观察t之前的推进时计算；t=0使用初始分布。
- `boundary_mass[t] = filtered_posterior[t][0]+filtered_posterior[t][-1]`，是当前后验落在端点的离散质量；它也受临近端点插值与当前 likelihood 重加权影响，不等于前一项。
- `posterior_weighted_mean_error` 是加权有符号 row error。下端clip正误差和上端clip负误差可能抵消；`π @ abs(row_mean_error)` 是更保守的非抵消指标。现有 `abs(π @ row_mean_error)` 不能代替它。

本次不读取实际 posterior，因此不能报告真实运行的加权截断率或宣称其可忽略。

均值保真不保证方差保真。对夹在L/H之间的移动值x，线性分配给两节点会额外增加二阶矩 `(x−L)*(H−x)`。若对数间距为h，该项除以x²的最大值为 `sinh²(h/2)`：

| grid数 | log间距 | 单节点额外方差/x²的上界 |
|---:|---:|---:|
| 256 | 0.062745098 | 0.000984560（0.098456%） |
| 512 | 0.031311155 | 0.000245117（0.024512%） |

这是局部数值扩散界，不是 posterior/forecast error 的界。源代码的256/512 synthetic比较只检查某个固定合成参数设置下的最大状态均值差，不能推广为真实拟合模型误差保证。若将来授权评估敏感度，应固定同一 checkpoint 参数、bank、c_ref、hurdle/AR 和观测序列，不重拟合，比较过滤均值、方差、M后验TV、同origin的预测及加权边界指标。增加grid数不扩大边界，也不增加GH阶数，不能单独排除截断或GH误差。

[forecast](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/round21_gamma_observation_filter.py#L248>) 使用解析 `mu+phi^k*(filtered_mean−mu)`，这是未截断工作过程的正确条件均值；存在边界clip时，它不必等于把当前后验重复乘有限grid矩阵所得的均值。这属于网格近似与理想过程之间的一致性误差，不应声称在截断链上也精确。

**三、完整后验、N0与future因子**

[正观测更新](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/round21_gamma_observation_filter.py#L192>) 先计算 `prior(S)=previous_posterior @ transition`。M的未归一化后验为：

```text
prior_M(v) * p(shape|M)
* Σ_S prior(S) * Gamma(y | S,M,shape,kappa)
```

源码在对S作logsumexp后再对M归一化，确实包含prior(S)和Gamma radial，既不是shape-only，也没有错误地使用已更新posterior(S)而重复计入radial。`logpositive(v)`与S/M无关，在M归一化中消去是正确的。global_dense控制有意不使用shape likelihood；其他模式的joint likelihood只加shape一次。

N0分支使用 `evidence=log p0(v)`、`posterior(S)=prior(S)`，只推进且不给S新增信息，未把S设为零。`Mposterior=zeros(7)`是undefined的哨兵，应结合valid=false排除，不是M=1分布。其正确性依赖 **unresolved-range hurdle概率只随volume、与S/M无关** 的工作假设；不表示无交易或真实波动为零。

[future factor](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/round21_gamma_observation_filter.py#L239>) 预测AR volume，再用GH积分 `(1−p0(V))*Σq(M|V)c_G(M)/c_ref`。与 `E[S_future]` 相乘，需要未来volume与S创新独立这一工作假设。若两者相关，乘积漏掉协方差项。该hurdle/volume预测机制不是由Brownian校准证明的真实N0/交易次数规律。

**四、另两项已确认的数值约定**

[R20 geometry编码](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/round20_observation_operator_gru.py#L42>) 由F17恢复a/z并使用1e−10边界tolerance；参考表使用raw wick精确零。因此两者并非完全同一分类。合法纯数学反例：q=1、a=1−2^−34、z=0，两侧wick均为2^−35>0。raw参考属于interior cell26，运行时tolerance会归为atom cell0。这只是极薄边界带上的确定性定义差异；本次未检查其市场发生频率或实际影响，也未修改源码。

`log(prior.clamp_min(1e−300))`是正则化近似，严格来说不是完全不改prior的取对数。总注入质量最多为grid数乘1e−300；但在极端likelihood重加权下，不能据此给posterior影响一个无条件的小误差保证。结构不可能M的−inf likelihood仍保留。

以上没有提供真实M、activity A或IV识别。Gamma、Bayes更新和总方差公式本身也不是新理论。本次审计不据这些近似调参、改主候选或增加真实检验；所有冻结输入保持原样。
