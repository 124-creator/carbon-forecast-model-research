# R20 联合活动时钟—采样分辨率观测算子：小范围 prior-art 核对

日期：2026-10-22。对象仅为主线程描述的新结构；未读取行情、2026 数据、模型输入/权重或开发结果，也未运行模型。此检索不作为当前实现的前置审批，不预判尚未运行的市场表现。

**可辩护的最窄定位是：在固定的有限分辨率 Brownian 参考族上，以受限的 volume prior 联合约束观测均值、观测不确定性与未来 Parkinson 目标倍率的观测算子。不能将 time-change、volume 入方差、有限采样偏差、OHLC shape 分解、Bayes mixture 或 Kalman/GRU 本身写成创新。本轮证据不足以声称完整组合“首次出现”，也不足以断言它只是任何现有固定规模 GRU 的精确重参数化。**

## 1. 实际阅读边界：六个 primary works，全文一篇

同一论文的 DOI、作者机构库和出版社记录计为一个 work，没有扩展到第七篇。全文实际读到 Saichev 等的 arXiv 版本；Martens–van Dijk 的机构 PDF 两次超时，已改读出版社提供的摘要、Introduction 和节摘，**未将其计为全文**。其余仅达到下表所列层级。检索结果中的其他论文及下列论文的参考文献未作为新增已审读 primary sources。

| 编号与一手来源 | 实际核对层级 | 可以据此确定的已有内容 | 不能据此确定的内容 |
|---|---|---|---|
| S1. Clark (1973), *A Subordinated Stochastic Process Model with Finite Variance for Speculative Prices*, Econometrica 41(1). [DOI](https://doi.org/10.2307/1913889)；[注册元数据](https://api.crossref.org/works/10.2307/1913889) | 书目信息，未读全文 | subordinated stochastic process 是明确的经典先行方向；time-change 不能作 R20 的新发现。成交量代理信息时间的直接内容证据另见 S2。 | 不凭书名或元数据断言 Clark 所有建模细节，更不推断其已经或尚未包含 R20 的联合结构。 |
| S2. Lamoureux & Lastrapes (1990), *Heteroskedasticity in Stock Return Data: Volume versus GARCH Effects*, Journal of Finance 45(1), 221–229. [DOI](https://doi.org/10.1111/j.1540-6261.1990.tb05088.x)；[出版社注册摘要](https://api.crossref.org/works/10.1111/j.1540-6261.1990.tb05088.x) | 原作者摘要，由出版社存入 Crossref；Wiley 正文未取回 | 摘要明确把 daily trading volume 用作 information arrival time 的 proxy，并将 volume 放进 variance equation；其样本中 ARCH effects 倾向于消失。 | 不是“volume 等于交易次数”的定理，也不是所有资产中 volume 都消除 GARCH 的结论；摘要不能排除更复杂的先行结构。 |
| S3. Chou (2005), *Forecasting Financial Volatilities with Extreme Values: The Conditional Autoregressive Range (CARR) Model*, JMCB 37(3), 561–582. [DOI](https://doi.org/10.1353/mcb.2005.0027)；[作者机构记录](https://ir.sinica.edu.tw/handle/201000000A/57886?mode=simple)；[出版社摘要入口](https://muse.jhu.edu/article/183856) | DOI/机构元数据；检索可见原摘要首句，正文未取回 | 对固定时间区间 high/low range 建立 dynamic conditional autoregressive model 早已存在。range forecasting 及条件尺度动态不能单独成为贡献。 | 本轮未逐式审 CARR/CARRX 及其所有扩展，不据元数据断言这些模型缺少任何特定 volume 或 observation 机制。 |
| S4. Martens & van Dijk (2007), *Measuring Volatility with the Realized Range*, Journal of Econometrics 138(1), 181–207. [DOI](https://doi.org/10.1016/j.jeconom.2006.05.019)；[出版社可见正文预览](https://www.sciencedirect.com/science/article/abs/pii/S0304407606000923) | 摘要、完整 Introduction、节摘；非全文 | Introduction 明确说明非连续观察使观测最高价偏低、最低价偏高，infrequent trading 导致 range 向下偏；bid–ask bounce 则可向上偏。也明确给出按 daily range 与 realized range 平均水平之比进行 scaling 的思路。 | 其 scaling 不是 R20 所述可学习离散 M 后验；但不能仅凭预览断言全文没有其他相关方法。不能把该文的高频 realized-range 结论直接移植成 ETF 日 OHLC 的真实 IV 识别。 |
| S5. Saichev, Sornette & Filimonov (2009 preprint), *Most Efficient Homogeneous Volatility Estimators*, arXiv:0908.1677. [全文](https://ar5iv.labs.arxiv.org/html/0908.1677) | 全文，重点 §2–3、§5.1/Fig.10、Appendix A | 已有 OHLC 幅度/几何的 homogeneous 表示、联合 OHLC 分布和 shape diagrams；§5.1 直接讨论离散 N 对连续极值的逼近、G&K/RS 均值随 N 的变化，并提及 N 可作典型交易数的实践解释。有限分辨率影响及按单位过程校准不是新发现。 | 该文这些章节研究估计器与离散化收敛，不构成对真实 ETF 交易数可识别的保证，也不自动给出 R20 的可学习 volume-tilted 后验/未来目标算子。不能将全文中 N/M 的模拟记号直接当 R20 参数。 |
| S6. Kawakatsu (2023), *Simple Factor Realized Stochastic Volatility Models*, Journal of Time Series Econometrics 15(1), 79–110. [DOI](https://doi.org/10.1515/jtse-2021-0049)；[出版社注册摘要](https://api.crossref.org/works/10.1515/jtse-2021-0049) | 原作者摘要，未读全文 | 摘要明确以多个 noisy daily variance measures 提取 latent variance，写成 linear state space 并用 Kalman filter；实证使用 daily OHLC。因此“OHLC 测量＋潜在风险状态＋Kalman”已有直接先例。 | 摘要不能证明其完整方法没有 latent sampling mixture、神经扩展或其他联合设计。本轮不作此负面断言。 |

六篇预算内优先覆盖了全部指定的金融先行分支；没有再增加一个专门的 deep state-space primary。下文对“神经观测学习”的比较是**直接写出的函数/状态重参数化分析**，不是已经排除所有神经观测文献的检索结论。本轮不能据此发出“神经观测学习中从未存在同样结构”的保证。

## 2. 哪些组成必须视作已知

- **活动或业务时间与成交量代理。** S1 是经典 subordination 来源；S2 的摘要直接支持 volume 进入条件方差及信息到达时间代理的先行关系。R20 的 A(v) 可以有新的受限实现，但不能把“量驱动波动”或“活动时钟”本身作为发现。
- **range 的条件尺度预测。** S3 已明确存在 CARR 这一方向。positive 三段解码、让总幅度可变化，可以是具体参数化选择；不是 range forecasting 的起点。
- **有限观察低估极值、需均值/尺度修正。** S4 已明确解释方向及 scaling；S5 §5.1/Fig.10 更直接覆盖有限 N 的单位过程模拟。M∈{1,2,4,8,16,64,512} 是一个参考离散化设计，不形成新的有限采样现象。
- **幅度不变量和 OHLC 几何信息。** S5 的 homogeneous 分解、shape diagrams 和联合 OHLC 密度是关键先行。不能把按 shape 分离尺度或用联合几何律本身宣称为新方法。
- **多噪声测量进入状态空间/Kalman。** S6 摘要已足以确立该类框架存在。Gaussian moment matching、Bayes 归一化、正输出、GH 数值期望也都属于标准概率/计算工具；下面的公式不被列作原创定理。

## 3. 将新候选写清楚后，剩下的结构差异

令 S=shape31，v 为指定的 volume 特征，X 为潜在对数风险状态。为避免单位歧义，以下把 A(v)>0 定义为**方差/活动时钟倍率**，从而 HLC 的幅度倍率为 sqrt(A)。若实现把 A 定义为 HLC 幅度倍率，方差与 log-observation 中应相应使用 A² 和 2logA。

在单位、零漂移 BM 参考族内，可将所述观测近似写作

\[
P_t=\exp(X_t)A(v_t)U_{M_t},\qquad
\log P_t=X_t+\log A(v_t)+\log U_{M_t}.
\]

这里的 shape-invariant A 是**特定标量幅度模型的限制**；不能外推为任意内生 time-change、任意离散采样时点或真实 ETF 都保留同一 shape law 的定理。

若 q_M 是固定基准权重，volume prior 的受限形式是

\[
\pi_M(v)\propto q_M\exp\{\gamma(v)\log M\},\qquad
w_M(S,v)=\frac{p_M(S)q_M\exp\{\gamma(v)\log M\}}
{\sum_jp_j(S)q_j\exp\{\gamma(v)\log j\}}.
\]

这是固定 likelihood 与单参数 exponential tilt 的 Bayes/gating 结构。它比七个自由 gating logits 更受约束，但“后验＝likelihood×prior”本身不是创新。M 在这里是**工作参考族的有效分辨率标签**，不是已经观测或识别的成交笔数。

可提出的差异是让**同一 w** 同时约束观测偏差、混合不确定性与目标尺度，而非三套独立神经头：

\[
\bar b=\sum_M w_M b_M,\qquad
R_{\rm mix}=\sum_Mw_M\{R_M+(b_M-\bar b)^2\},\qquad
\bar c=\sum_Mw_Mc_M.
\]

其中 total variance 必须包含分量均值之间的项；仅使用 sum(w_M R_M) 会丢掉 M 后验带来的不确定性。这是全方差公式，不是新的滤波理论。

**条件矩边界。** 如果 b_M/R_M/c_M 只是在每个 M 上全局统计，而 w 又依赖 S，以上通常不能被称为真实的 E[logU|S,v]、Var(logU|S,v)、E[U|S,v]。幅度随机因子与 OHLC shape 没有由齐次性保证独立；S5 的联合分布/diagrams 正是相关先行。更准确的写法是用 b_M(S)、R_M(S)、c_M(S)，或者明确声明当前使用的是“按 M 校准的工作测量分布＋shape gating 的 Gaussian moment 近似”。这只是说明正在实现的模型含义，不要求暂停实现或改换方向。

## 4. 与已有模型是否等价：区分观测表达与完整动力学

**在只保留上述矩的观测层上，可以重写。** 令

\[
f_{\mu}(S,v)=\log A(v)+\bar b(S,v),\quad
f_R(S,v)=R_{\rm mix}(S,v).
\]

进入 Gaussian moment filter 的内容，就是确定性的条件均值映射与异方差映射；再加一个未来目标倍率映射。允许足够灵活的神经观测函数时，该计算可被包含在普通异方差观测模型中；固定 lookup 与 gamma 网络的组合也仍是一个显式 feature-to-moments 映射。不能因此声称出现了全新的通用表达类或新 Bayes 原理。

**但不是已经证明与原固定规模 GRU 完全等价。** 共享 BM lookup、volume-only 指数倾斜、A 只作用幅度、同一后验耦合 b/R/c、参考尺度一致性，都会限制函数族和训练路径。将 X 改名为 X*=X+logA(v)+barb(S,v)，虽可消去观测均值项，却一般改变原来约束的状态转移律：X* 不再自动保有同一个 AR1/独立创新结构。观测层可重写，不等于完整受约束状态空间模型只是无损换名；“足够灵活的网络可包含”也不等于当前同参数量网络精确实现。

因此最合理的贡献分类是：**模拟校准、尺度一致的结构约束/归纳偏置及其工程组合**。若实现随后把 b、R、c 各自自由学习，或让任意残差头完全解除它们的关系，这个窄差异会退化成一般的 learned observation/gating。若不解除这些关系，则不能仅凭其可重写为 f_mu/f_R 就判定为毫无结构差异。

cbar 的归一化可固定公共常数尺度约定，避免 A 与 M 的常数水平在观测/目标转换中互相挤占；它**不单独证明** A、latent risk 与 M 的动态或因果可识别性。即使在理想参考族内某组参数可统计估计，也不能转述为真实 ETF 的采样数已被识别。

## 5. 未来 target operator：可形成耦合约束，工具本身已知

pre25 volume AR1＋GH expectation 是预测 volume 分布再积分非线性函数的具体计算方案；AR1、Gaussian quadrature 和 reference normalization 本身不新。应写清楚，未来 shape 未观测时，使用的是未来 prior/转移分布，不是未经说明地拿当日 shape 后验替代所有未来 M。

若未来只通过 volume prior 决定 M，一种明确的目标因子为

\[
F_{t,k}=C_{\rm ref}^{-1}\mathbb E_t\!\left[
A(V_{t+k})\sum_M\pi_M(V_{t+k})c_M
\right].
\]

C_ref 的具体定义取决于公共状态框架；观测与输出必须采用同一约定。A 与 mixture weights 都由 V 驱动，通常不能把该期望擅自拆成 E[A]×E[cbar]；若潜在风险与未来 volume 相关，把风险期望再与 F 相乘也需要对应的模型假设。这里比较的是**同一参考族同时约束 measurement 与 future-target map 的结构**，不是提出新的期望算法。

positive 三段 decoder 不固定总量，避免把所有差异强压成期限形状。但自由总尺度也可能吸收 operator 的幅度效应，因此其存在本身不能证明时钟/分辨率的物理解读。old-GRU warm-start＋same-three-bucket、无 operator 控制，有助于保留当前架构问题的可比性；本报告不要求另行改评估任务或停止当前实现。

## 6. 可用的最窄表述与不能写的结论

可用表述：

> 我们提出并研究一个受约束的联合观测算子：以固定有限分辨率 Brownian 参考族的几何 likelihood 和校准矩为基础，用 volume-only 指数倾斜构造有效分辨率权重，将其与 shape-invariant 活动幅度因子结合，并使观测均值、不确定性及未来 Parkinson 目标倍率遵守共同的参考尺度与后验约束。该算子嵌入已有状态滤波与 GRU/正三段解码框架；这里的 M 是工作参考分辨率，未赋予真实成交次数或真实 IV 的识别结论。

这是一条**有边界的结构候选表述**，不是已经完成的新颖性证明。不写“首次发现有限采样”“首个 OHLC-Kalman/volume clock”“可识别真实采样 N”“Gaussian moment filter 新理论”，也不凭未读全文的摘要声称“既有工作均没有此联合结构”。

本轮六源中，已核实的先行内容覆盖了多数基础组件；尚不足以证实一个既有方法与 R20 的全部耦合约束完全相同，也不足以排除这种先行方法。神经观测学习分支的完整同构检索仍是证据缺口，应保留而非营销性填补。当前 R19/旧 GRU 的优劣不被当作 R20 成败或原创性的证据；新候选的市场表现尚未被本报告读取或判断。
