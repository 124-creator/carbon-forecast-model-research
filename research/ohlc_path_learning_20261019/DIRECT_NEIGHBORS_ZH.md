# 报告价格—范围联合路径学习：直接近邻与独立复核约定

日期：2026-10-19。此文件为实验开始前的独立审查记录；不包含实验结果，不表示方法已成立。后续评分代码须等预测文件 schema 到位后再编写。

## 1. 当前数据语义

按本轮实验定义，P 是数据源提供的报告价格，H/L 是分别报告的高低范围。允许 P 不在 [L,H] 内；不额外解释为测量噪声、不把 P 改称满足 OHLC 几何约束的收盘价。

只需 P>0、H>=L>0，坐标

\[
d=\log(\sqrt{HL}/P),\qquad q=\tfrac12\log(H/L)\ge0
\]

便精确给出 H=P exp(d+q)、L=P exp(d-q)。这是观测值的可逆重表达，不是新的随机过程理论。

从同一未来随机序列生成每个 h∈{1,5,20} 的目标：

\[
T_h=((P_{t+h}),\max_{1\le j\le h}H_{t+j},\min_{1\le j\le h}L_{t+j}).
\]

主目标始终是报告价格均值预测的样本外 MSE。累计范围和概率评分属于训练机制/辅助验证；不能用它们改善来代替主目标成功。

## 2. 必须明确引用的直接先行工作

### 2.1 共享随机转移与多步评分：Brolly (2026)

Martin T. Brolly, *Stochasticity and probabilistic trajectory scoring are essential for data-driven closures of chaotic systems*, PNAS 123(35), e2609143123，在线发表 2026-08-24。

- DOI：https://doi.org/10.1073/pnas.2609143123
- 已读官方全文：https://pmc.ncbi.nlm.nih.gov/articles/PMC13534829/
- 已用另一搜索引擎核验官方全文命中；DOI 直接跳转工具本轮返回 internal error，不据此声称 DOI 页读成功。

§2.1–2.3 明确提出参数化 Markov 随机转移，迭代形成预测轨迹，以跨 lead 的能量评分损失训练共享参数。Eq.10 是逐 lead 的 ES 相加；本轮拟议的摘要向量联合 ES 与之有所不同。本文领域是粗粒化混沌动力系统，其物理机制与长期定理不能直接移用到碳价。

**限制本轮声称：**共享状态、随机 rollout、多步 ES 本身已有直接先例。不要把逐条随机轨迹的 MSE 方差惩罚误读为“均值预测用 MSE 不合理”。

### 2.2 多期限联合分布与 ES 训练：Kan et al. (2022)

Kelvin Kan et al., *Multivariate Quantile Function Forecaster*, AISTATS, PMLR 151:10603–10621。

- 官方入口与软件链接：https://proceedings.mlr.press/v151/kan22a.html
- 已读正文：https://proceedings.mlr.press/v151/kan22a/kan22a.pdf

MQF2 直接输出多期限联合分布；§4.2.1 使用样本能量评分训练，目标向量可跨多个时间步或多个序列。实验还用未来各步之和的 CRPS 检验依赖结构。其生成结构采用凸函数梯度，没有本轮的低维共享 Markov 转移。

**限制本轮声称：**联合多期限样本、ES 学习、通过路径汇总量评估依赖，都不是新的通用构造。

### 2.3 时序生成网络的 proper-score 估计：Pacchiardi et al. (2024)

Lorenzo Pacchiardi, Rilwan A. Adewoyin, Peter Dueben, Ritabrata Dutta, *Probabilistic Forecasting with Generative Networks via Scoring Rule Minimization*, JMLR 25。

- 官方入口：https://jmlr.org/papers/v25/23-0038.html
- 已读正文 §2.2、§3.1、Appendix C：https://jmlr.org/papers/volume25/23-0038/23-0038.pdf
- 作者代码：https://github.com/LoryPack/GenerativeNetworksScoringRulesProbabilisticForecasting

论文以历史窗口条件化生成网络，用 prequential scoring rule 训练；在平稳性、混合及识别等条件下讨论一致性。Appendix C 讨论样本评分和梯度估计。其理论还依赖具体假设，不能只凭使用 ES 便声称本轮模型一致估计或参数可识别。

### 2.4 max/min 变换评分：Pic et al. (2025)

Romain Pic, Clément Dombry, Philippe Naveau, Maxime Taillardat, *Proper scoring rules for multivariate probabilistic forecasts based on aggregation and transformation*, ASCMO 11:23–58，2025-03-13。

- DOI：https://doi.org/10.5194/ascmo-11-23-2025
- 已读官方全文 §3–4：https://ascmo.copernicus.org/articles/11/23/2025/index.html

Proposition 1 与 Corollary 1 给出变换后评分及其非负组合的 propriety；§4.2 明确列出 patch minimum/maximum 变换。降维变换一般损失对原联合分布的严格识别能力。

**限制本轮声称：**max/min 摘要配 proper score 的数学原则已有文献。本轮九维摘要的 ES 对摘要分布具有严格适当性；由于 h=1/5/20 的摘要不能还原整条二十步路径，不能据此声称对完整路径 law 或转移参数严格识别。

### 2.5 一致样本与端到端评分：Olivares et al. (2024)

Kin G. Olivares et al., *CLOVER: Probabilistic Forecasting with Coherent Learning Objective Reparameterization*, TMLR，2024-12。

- 审稿/论文：https://openreview.net/forum?id=q7YXEbFOAt
- 已读作者机构正文：https://www.stat.berkeley.edu/~mmahoney/pubs/3148_clubsuit_CLOVER_clubsuit_.pdf

论文以可微样本、具有层级一致性的多变量因子模型，使用 CRPS 或 ES 端到端训练。核心约束为线性聚合；本文讨论的累计 max/min 属于非线性路径泛函。但“从共同样本保证一致、再反向传播评分”的总体思想应被引用。

## 3. 本轮尚可检验的有限主张

目前没有证据支持“全新通用学习规则”。可以如实把方案定义为：在报告价格与报告范围允许不包含的碳价数据语义下，检验低维共享随机动力模型能否借助累计范围的联合分布监督，改善价格均值估计。

这一主张要求消融区分：使用了额外历史 H/L 信息、使用了额外未来范围训练标签、使用了联合而非边际评分、使用了累计范围而非单期范围、共享转移造成的归纳约束。完整模型仅优于价格历史 RW 不足以把增益归因于所声称学习机制。是否值得扩展消融，须先服从预先冻结的精度筛选门槛。

精确坐标恒等式和累计 max/min 的合法性可以直接代数检验；它们不保证未来价格有可预测均值，不保证 ES 改善带来 MSE 改善，也不保证跨来源泛化。

## 4. 独立评分准备（等待 schema；此时不编写 scorer）

1. **数据对齐。** 独立从原序列重新构造起点、终点与未来范围。窗口必须恰为 t+1 到 t+h，排除 t；核对各模型共同可评分起点。允许 P 在 [L,H] 外，不做投影或删去此类行。
2. **时序约束。** 核对每次拟合时全部训练标签已经成熟；缩放、阈值、特征和选择只能使用相应训练/验证信息。Quantum1805 无日期，结论只按索引块陈述；不能伪造年度外推检验。
3. **主指标。** 重新计算原价格单位的 MSE、相对 RW 的 MSE，逐数据源/块/期限报告并保留失败单元。价格均值必须是 E[P] 的估计，不是 exp(E[log P])。
4. **Monte Carlo。** 记录生成样本数量、随机种子、是否独立与均值的 MC 误差。有限样本均值评估的误差应足够小，不能让模拟噪声主导微小改进。若不是逐次独立采样，不能直接把通常的 U 统计量称无偏。
5. **能量评分。** 核对维度、训练期尺度、摘要顺序、坐标变换、均值和两两距离项的归一化。用独立路径样本的无偏版本时，第二项为全部不等样本对距离之和除以 2S(S-1)。
6. **区间诊断。** 由样本直接验证 H>=L、随 h 的累计 high 非减/low 非增。预测分位数区间与实测 high/low 区间应分开命名；后者不是覆盖概率区间。
7. **统计比较。** 对共同起点的配对损失做按源的时间块抽样，保留同一起点不同 h 的依赖；重叠 h=20 标签不能当独立观测。明确选择期与未触碰检验期；报告 CI 含 1 时不能称显著优于 RW。

## 5. 有限样本损失的两个数学核查项

若 X_1,…,X_S 为独立的价格预测样本，μ=E[X]、σ²=Var(X)，则

\[
E[(\bar X-y)^2]=(\mu-y)^2+\sigma^2/S.
\]

因此用有限 MC 均值的 MSE 直接作为训练损失，会增加一个方差惩罚。无偏均值损失估计可写成 (barX-y)^2-s²/S，其中 s² 是通常无偏样本方差；或使用两组独立样本均值与 y 的误差交叉乘积。若本轮仅用 ES 训练，则此问题只需在均值评估的 MC 误差分析中处理。

标准 ES 的 U 估计对独立样本成立。反向对偶、Sobol 等路径之间通常有关联；它们可用于降低某些均值积分误差，但直接套用独立样本的 U 公式不自动保持其无偏解释。两组独立采样的交叉距离法须检查组间独立性及每组边际是否正确。

