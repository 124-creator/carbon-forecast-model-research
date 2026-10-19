# 阶段 e：条件阻尼双趋势的直接近邻与可证伪边界

审查日期：2026-10-19。范围是这轮冻结候选的针对性原始来源检索，不是系统综述，也不据这些案例判断整个碳价预测文献的多数做法。近邻检索与机制设计在本轮拟合前开展，统计计划另行在拟合前冻结；这份长文稿随后整理并校订，不能据此声称全文已在全项目模型评分前完成。本审查本身未运行市场拟合。精确来源与阅读范围见同目录 `DIRECT_MECHANISM_NEIGHBORS.json`。

## 1. 结论先行

候选应称为 **发行原点条件化的、有符号双阻尼趋势回归**。指数状态、几何累积的期限形状、输入相关衰减率和正则化变量投影均有直接先例。现阶段能提出的具体问题是：在相同价格历史和交易上下文下，**条件响应时长是否提供超出静态时长、条件振幅与自由期限读出的可复查预测增量**。这是一项尚待数据支持的领域机制主张；不是新状态空间理论、因果冲击识别，或已获证明的普遍优越架构。

最接近的数学先验并非仅是泛称“AR”。Gardner–McKenzie（1985）的阻尼趋势预测直接包含 `trend × sum(phi**j, j=1..h)`；本轮 `S_h(lambda)` 就是这一几何和。Askham–Kutz（2018）明确以变量投影拟合时间指数基底及线性系数。Mamba 的原始方法把离散化时步设为输入的函数，已体现输入改变衰减与记忆时长的广义思想。碳收益率 MuSTANet（2026）又已覆盖多尺度趋势与随时间变化的变量权重。因此不能仅以“多尺度”“状态依赖”“可解释”“共享期限”“VP”这些名称主张原创。[S1–S4]

## 2. 候选的准确数学对应

以下是对本地 `source/signed_response_models.py` 和实现者给定公式的独立推导，不归为任何论文的原话。令 `r_t` 为按训练尺度归一化的算术价格变化，历史窗口为64：

\[
v_{k,t}=(1-\rho_k)\sum_{\ell=0}^{63}\rho_k^\ell r_{t-\ell},\quad
z_t=(\log(1+|r_t|),\;\mathrm{volume\_surprise}_t\,\mathrm{valid}_t),
\]
\[
q_t=2\tanh(Wz_t),\qquad
\lambda_t=\mathrm{ordered\_rates}(a+q_t),\qquad
m_t(h)=\mu h+\sum_{k=1}^{2}b_kv_{k,t}S_h(\lambda_{k,t}),
\]
\[
S_h(\lambda)=\sum_{j=1}^{h}\lambda^j,
\quad h\in\{1,5,20\},\quad 0<\rho_1<\rho_2<0.98,
\quad0<\lambda_{1,t}<\lambda_{2,t}<0.98.
\]

`b_k` 可正可负；历史率由全局参数 `a` 决定，未来率只由发行时已知的上下文决定，并在该次预测内冻结。

**固定参数时是受限分布滞后回归。** 静态臂可以准确展开为

\[
m_t(h)=\mu h+\sum_{\ell=0}^{63}C_{h\ell}r_{t-\ell},\qquad
C_{h\ell}=\sum_k b_k S_h(\rho_k)(1-\rho_k)\rho_k^\ell.
\]

跨期限—滞后系数矩阵秩至多为2；固定率后拟合读出，就是指定指数特征上的线性岭回归。64窗口有限截断时，单个期限是有限分布滞后预测器。只有在取消截断并采用相应递推约定时，指数滤波器才是通常的有理传递函数/有限维线性系统；不能不加条件地把当前整套多期限预测称为某一纯 AR 或 ARMA 的完全等价实现。

**共同形状不等于生成一致性。** 历史状态以全局 `rho` 过滤真实价格变化，未来使用 `lambda(z_t)`，并未定义未来未知上下文和观测的联合概率法则。即使共享 `S_h`，也不自动满足滚动条件期望的塔式性质。标准 Holt 的创新状态更新与本轮EWMA变化特征也不同。可说“共享跨期限限制”，不宜说“保证时间一致的生成状态空间”。

**延续、回撤与完整均值回归要分开。** 未来第 `j` 步的预测变化是 `mu + sum_k b_k v_k lambda_k**j`。有符号混合允许相反方向及随期限改变形状，但是否发生符号转变取决于拟合系数和状态，不能由模型名称保证。面对一个已发生正变化，负的未来累计预测表示回撤；若要称价格完全回到冲击前水平，还必须把当前变化加回总响应后核查。价格变化和总成交量本身不识别外生冲击及其因果效应。[S1、S11仅提供相邻先验；本段结论来自上述公式]

**单步不能独立识别“时长”。** `S_1(lambda)=lambda`，在单个期限上，改变时长可由足够灵活的振幅变化吸收。多期限提供的是 `S_5/S_1`、`S_20/S_1` 等形状约束；它们才使条件时长与仅调振幅具有可比较差异。两个分量叠加后仍可能抵消或弱识别：有序率只消除标签交换，不能消除相近率或很小读出系数造成的不确定性。

**VP 是估计方式。** 固定非线性参数后，将加权设计列按训练RMS标准化，精确求解

\[
\min_\gamma\; n^{-1}\|A(\theta)\gamma-y_w\|^2+\alpha\|\gamma\|^2,
\qquad\gamma=s(\theta)\beta.
\]

这属于正则化可分离非线性最小二乘。线性子问题有唯一岭解；外层非线性优化不因此获得全局最优保证。列尺度依赖非线性参数，当前实现将其保留在计算图内；原始系数坐标下的惩罚是 `alpha * ||s(theta) beta||²`，不能误写为与尺度无关的普通 `alpha * ||beta||²`。VPNet 的逐样本投影特征层与这里的训练集监督读出求解，是同一数学工具的不同用途。[S2、S5、S6、S7]

## 3. 已固定对照如何否证机制

| 对照 | 回答的问题 | 结果的科学含义 |
|---|---|---|
| `rw`、`ridge_price` | 是否超过无变化预测及已有较强价格线性预测？ | 弱基线获胜不能单独支持条件时长。 |
| `legacy_fixed_response` | 旧固定先验能否解释改善？ | 描述性参照；不是这轮受监督读出的替代品。 |
| `fixed_response` | 固定 `.5/.9` 时，只拟合3个线性系数已能做到多少？ | 与VP epoch0区分；epoch0已经学习线性读出。 |
| `static_response` | 学习全局时长是否已足够？ | 若与主模型相当，条件化时长证据有限。 |
| `signed_shock_only` | 成交量是否超出当前价格活动指标？ | 不能把仅价格活动产生的收益归于量能机制。 |
| `amplitude_context` | 同9个拟合标量、相同上下文，仅条件振幅能否解释收益？ | 最直接区分“时长”与一般上下文门。参数数量相同不等于优化几何完全相同，仍须披露验证选择和学习情况。 |
| `free_horizon_context` | 相同状态和条件时长特征，改为各期限独立9系数读出是否更好？ | 检验共享限制的有限样本价值，不证明塔式一致性。它有15个拟合标量，应如实披露。 |
| `ridge_interaction` | 最近20个变化、当前2上下文与40个交互项，是否已解释非线性收益？ | 比仅加两个加性变量的Ridge更能检验一般状态相关系数解释。其历史长度20与响应窗口64不是完全相同，不能声称所有线性对照信息窗口逐项相同。 |

所有核心对照的样本原点、目标成熟规则和评分定义相同。正则化参数的“数值相同”不自动代表惩罚强度相同：经典Ridge的平方和目标与VP的均值目标须标清。已登记的原始GRU与SWEWT-GRU是评分前的两项次要参考，使用同目标和价格/成交量信息；它们是固定预算下的参考，不是完整复现 MuSTANet 或全领域SOTA比较。统计家族以 `audits/STATISTICAL_PLAN_BEFORE_FIT.json` 为准。

## 4. 原始文献证据及阅读边界

### S1. 阻尼趋势：最直接的期限公式先例

Gardner, E. S. Jr. & McKenzie, E. (1985), **Forecasting Trends in Time Series**, *Management Science*31(10):1237–1246。出版社记录日期1985-10-01；DOI[10.1287/mnsc.31.10.1237](https://doi.org/10.1287/mnsc.31.10.1237)。已读[作者机构公开全文](https://www.bauer.uh.edu/egardner/3301H%20Operations%20Management/ESG%20Publications/1985%20Forecasting%20trends.pdf)§2.1、§2.3、§3.1。Eq6给出几何累积趋势，§2.3列出相应ARIMA关系。论文完整Holt更新与本轮历史特征不同，不能直接借用其全部概率对应。原文有计算附录；本审查未找到并执行作者程序。

### S2. 指数时间基底加VP：已发表并有代码

Askham, T. & Kutz, J. N. (2018), **Variable Projection Methods for an Optimized Dynamic Mode Decomposition**, *SIADS*17(1):380–416，2018-02-01；DOI[10.1137/M1124176](https://epubs.siam.org/doi/10.1137/M1124176)。已读[作者稿](https://arxiv.org/pdf/1704.02343)§2.2、§3.1及Remark7：时间指数矩阵与线性载荷分离，外层不保证全局最优。作者[optdmd代码](https://github.com/duqbo/optdmd)README及目录可读，`src/examples`声明MIT；本轮未运行、未把该代码直接改装成碳价基线。它支撑估计与指数基底的先例，不证明与本轮有完全相同的条件核。

### S3. 输入改变衰减：已发表通用序列先例

Gu, A. & Dao, T., **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**。首稿2023-12-01，所读作者v2为2024-05-31；[COLM2024官方记录](https://openreview.net/forum?id=tEYskw1VY2)发布2024-07-10。已读[原文](https://arxiv.org/html/2312.00752v2)§2、§3.2、§3.5、AppendixC：`Abar=exp(Delta A)`，`Delta`可依赖输入。其上下文沿序列更新；本轮只在发行时冻结未来时长，模型与用途不同。[作者代码入口](https://github.com/state-spaces/mamba)可得；未移植、未训练。

### S4. 2026碳收益率多尺度趋势：必须正面定位，全文边界保留

Liu, D.; Tang, M.; Tang, Z., **Interpretable forecasting of carbon allowance returns using multi-scale trend-aware network with dynamic variable selection and squeeze-excitation**, *Applied Energy*414,127810，刊期2026-07-01；DOI[10.1016/j.apenergy.2026.127810](https://www.sciencedirect.com/science/article/pii/S0306261926004629)。读到出版社缓存的摘要、引言和分节摘录；正文网页403，未核完整方法公式/补充材料。可确认MuSTANet的并行卷积趋势感知注意力、动态变量权重和通道重标定；**不能声称已核其预测头、时点切分和所有消融**。本轮未找到并读取作者代码；不写“代码不存在”。它足以否定“碳收益率首次同时多尺度+动态变量权重”的宽泛表述，不足以断言我们的精确公式已经被发表。

### S5. VPNet的正确年份与用途

Kovács, P.; Bognár, G.; Huber, C.; Huemer, M., **VPNET: Variable Projection Networks**, *International Journal of Neural Systems*32(1),2150054。在线2021-10-13，卷期2022；DOI[10.1142/S0129065721500544](https://doi.org/10.1142/S0129065721500544)。已读[作者机构VoR全文](https://real.mtak.hu/149201/2/s0129065721500544(1).pdf)首页和§3：以投影或投影系数作输入特征，再接神经网络。不是本轮跨训练样本求监督输出系数的同一个模块。Ref105给出[作者仓库](https://git.silicon-austria.com/pub/sparseestimation/vpnet)，页面可打开，但本审查未逐行审程序。

### S6. 正则化VP已有时间序列实验，但“指数”的含义不同

Guo, H.; Liu, G.; Wang, L. (2021), **An Improved Tikhonov-Regularized Variable Projection Algorithm for Separable Nonlinear Least Squares**, *Axioms*10(3),196，2021-08-22；DOI[10.3390/axioms10030196](https://www.mdpi.com/2075-1680/10/3/196)。MDPI直连429，改读[由MDPI提供的完整VoR文本](https://www.researchgate.net/publication/354071800_An_Improved_Tikhonov-Regularized_Variable_Projection_Algorithm_for_Separable_Nonlinear_Least_Squares)，§1–§3.1、Eq32及实验说明。Eq32是 `exp(-lambda*||x-z||²)` 的RBF输入空间基函数，**不是沿预测期限衰减的时间核**。本轮只用其支撑正则化VP及时间序列估计已有先例；未核作者代码。

### S7–S8. 指数AR与RBF估计近邻：仅到官方摘要/摘录

Chen, G.-Y.; Gan, M.; Chen, G.-L. (2018), **Generalized exponential autoregressive models for nonlinear time series: Stationarity, estimation and applications**, *Information Sciences*438:46–57；DOI[10.1016/j.ins.2018.01.029](https://www.sciencedirect.com/science/article/pii/S0020025518300409)。出版社摘要明确采用VP估计GExpAR；未取得完整公式，不能把名称中的“指数”自动等同本轮 `S_h`。

Zheng, S. & Feng, R. (2023), **A variable projection method for the general radial basis function neural network**, *Applied Mathematics and Computation*451,128009，刊期2023-08-15；[原出版社](https://www.sciencedirect.com/science/article/pii/S0096300323001789)。正确DOI是 **10.1016/j.amc.2023.128009**，不是先前线索中的128060。官方摘要/分节摘录支持用VP处理GRBFNN的可分离平方误差；本轮未核全部证明与作者代码。这两项说明“非线性特征+精确线性读出”不是新估计类别。

### S9. 2026碳价BVAR：发表实践与信息设置须分别说清

Bastianin, A.; Mirto, E.; Qin, Y.; Rossini, L., **Forecasting the Price of Carbon with Macroeconomic and Financial variables**。期刊DOI[10.1016/j.jedc.2026.105435](https://doi.org/10.1016/j.jedc.2026.105435)；本审查实际读取的是[作者v3全文](https://arxiv.org/html/2402.04828v3)，2026-10-03，§3.1–3.3、§4.2、§4.4–4.5及补充S.1/S.2。论文使用月频扩窗、多期限BVAR/因子和RW比较；DM未拒绝平均等精度，仍报告时变表现和用途。因子模型中期最高约3.9% RMSFE改善，正文一年期基准BVAR比率`.9372`约6.3%，两者不可混写成全篇最高3.9%。作者承认年度核证排放插值可能前视，并报告去除此变量后的稳健性；该敏感性支持其特定结论，不自动把插值数据变成真实历史可得版本。本文的训练窗口归一化、过去窗口处理与未来标签监督，应逐环节按声称的信息设置说明。未执行作者程序。

### S10. 当前碳价状态依赖预测：不能声称该领域尚无此类框架

Chen, Y.; Mamon, R.; Spagnolo, F.; Spagnolo, N.; Xiong, H. (2026), **Forecasting and Pricing in the Carbon Credits Market**，*Journal of Forecasting*，在线2026-05-25；DOI[10.1002/for.70168](https://onlinelibrary.wiley.com/doi/full/10.1002/for.70168)。实际只读到官方摘要、日期和数据声明；`full`链接重定向到`abs`。作者比较典型随机过程及HMM状态切换扩展，含滚动估计和多期限评价。数据声明为向作者索取、未公开；未核具体生成式、代码和全套评估。只能用于研究定位，不充作完整方法复核。

### S11. 价格响应先验与因果范围

Bouchaud, J.-P.; Gefen, Y.; Potters, M.; Wyart, M., **Fluctuations and response in financial markets: the subtle nature of “random” price changes**，*Quantitative Finance*4(2):176–190，2004-04；DOI[10.1088/1469-7688/4/2/007](https://doi.org/10.1088/1469-7688/4/2/007)。[作者稿记录](https://arxiv.org/abs/cond-mat/0307332)首稿2003-07-14。本轮只核作者摘要与出版对应：研究巴黎市场交易方向持久性、限价单反向作用和历史交易的propagator。本轮数据为日价格变化/总量，不具备相同的有符号逐笔订单流，不能借其叙述直接宣称已估计结构性市场冲击。其具体核方程未在本轮正文复核，未作逐式等价判断。

## 5. 如何写创新边界和证据结论

可在实验前写：**“我们把两种符号可变的阻尼趋势限制为跨期限共享的低维响应，并检验发行时已知的活动与量能是否通过持续时长、而不只是振幅，提供额外预测信息。线性读出用标准正则化变量投影精确求解，以分开结构先验、读出拟合与条件化动力学的贡献。”** 其中“提供额外信息”是待检验假设，不能提前写成发现。

强基线的平均表现、跨来源可复核性、机制对照与区间共同界定结论。区间跨零应保留，既不自动否定实用价值，也不能改用另一个块长或另一个主模型来掩盖。不能把每项消融显著、所有市场全胜或任意百分比当作期刊共同门槛；也不能把已有论文的报告方式视为任何数据处理都成立的证明。旧天津只提供开发证据，新CEA的确认范围由该轮冻结记录与来源暴露审查限定。
