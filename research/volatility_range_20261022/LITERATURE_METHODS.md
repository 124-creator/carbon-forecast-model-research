# 波动率/风险预测方法文献笔记

日期：2026-10-22。目的：为碳市场日 OHLC＋volume 数据构造可检验的方法改进；核心来源均非 MDPI。本文是方法研究笔记，不是评估/批判文章。下述“成功实验”均为原论文实证，本次没有在本地数据上训练或宣称候选模型成功。

## 1. 实际全文访问与阅读证据

### 主论文：HARQ

Bollerslev, T., Patton, A. J., Quaedvlieg, R. (2016). *Exploiting the errors: A simple approach for improved volatility forecasting*. **Journal of Econometrics 192(1), 1–18**, Elsevier。

- DOI：https://doi.org/10.1016/j.jeconom.2015.10.007
- 实际访问全文：https://public.econ.duke.edu/~get/browse/courses/pbc/Readings/BollerslevPattonQuaedvliegJoE2016.pdf
- 本次 Python urllib 实际响应 HTTP 200，573,468 bytes，PDF 共18页。SHA256：`2281a8d5d0a2d9a7ff8dca7ad4990cf6cea46d7fee0a0a23dd611d6194ee59f8`。
- web_fetch 对该PDF失败后，用现有 pypdf 在内存读取；未把失败抓取或摘要冒充全文。分批提取全部18页；工具部分长输出截断，另对 p3–4 方法公式、p9 Table4/样本外设计重新精读。阅读覆盖引言、方法、模拟、实证、稳健性、结论/附录及参考文献，重点证据来自直接可见原文。

### 更接近候选的已有方法：多测量 Realized EGARCH

Hansen, P. R., Huang, Z. (2016). *Exponential GARCH Modeling With Realized Measures of Volatility*. **Journal of Business & Economic Statistics 34(2)**，Taylor & Francis。

- DOI：https://doi.org/10.1080/07350015.2015.1038543
- 公开全文：https://cdr.lib.unc.edu/downloads/7m01bz46r?locale=en
- 本次 HTTP 200，622,072 bytes，19页；精读PDF p1–4、p16 Appendix C。不是声称整篇逐字精读。
- 原始 Realized GARCH：Hansen, Huang, Shek (2012), **Journal of Applied Econometrics 27(6), 877–906**，Wiley，https://doi.org/10.1002/jae.1234 。本次其 CREATES PDF `https://repec.econ.au.dk/repec/creates/rp/10/rp10_13.pdf` 经抓取超时、urllib证书失败；不声称已读该全文。标准公式由上述2016原文与下述2021可访问全文交叉核实。

### 碳市场已经使用 HARQ

Benschop, T., López-Cabrera, B. (2017). *Realized volatility of CO2 futures*. **SFB 649 Discussion Paper 2017-025**，非期刊论文。

- 机构页：https://edoc.hu-berlin.de/items/8c989ad1-38ea-4b32-8a4e-c69a32521665
- DOI：https://doi.org/10.18452/18745
- 实际全文：https://www.econstor.eu/bitstream/10419/191789/1/SFB649DP2017-025.pdf
- 本次 HTTP 200，3,654,354 bytes，26个PDF页；精读引言、PDF p9的HARQ公式、p13–18的估计/预测表、p20–21结论。
- PDF p9/正文p7 Eq(11)直接列HARQ，使用ECX 15分钟期货价格：2007-05-21—2012-04-13，49,279个日内观测、1,250交易日。故不能声称“首次把HARQ用于碳市场”。

## 2. HARQ核心公式与成功证据

以日内收益 r_(t,i)、每天M个观测、Δ=1/M记：

```text
RV_t = Σ(i=1..M) r_(t,i)^2                               [Eq3]
IQ_t = ∫(t−1..t) σ_s^4 ds
RV_t = IV_t + η_t,  η_t 近似 mixed-normal(0, 2Δ IQ_t)    [Eq4]
RQ_t = (M/3) Σ(i=1..M) r_(t,i)^4                        [Eq5]

HARQ:
RV_t = β0 + [β1 + β1Q sqrt(RQ_(t−1))] RV_(t−1)
       + β2 mean(RV_(t−1)...RV_(t−5))
       + β3 mean(RV_(t−1)...RV_(t−22)) + u_t             [Eq13]
```

原文实施中对 sqrt(RQ) 去均值，使 β1可解释为平均日系数。复现时去均值仅用各训练窗，不能全样本去均值。原文主要经济机制：测量误差大时降低近期RV的预测影响；测量精确时允许更高近期权重/更快响应。β1Q通常负，但原始OLS不是把权重硬约束到[0,1]。HARQ-F Eq12对周/月系数也加RQ调节，原文发现增加参数不保证改善。

原文p9 Table4，按日重估，滚动窗RW=1000天，另有扩展窗IW；损失为相对HAR比值，低于1表示改善：

| S&P500一步预测 | MSE RW | MSE IW | QLIKE RW | QLIKE IW |
|---|---:|---:|---:|---:|
| HAR | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| HARQ | 0.8266 | 0.8944 | 0.9464 | 0.8809 |

对应改善17.34%、10.56%、5.36%、11.91%。个股平均相对MSE为RW 0.9349、IW 0.9525；平均QLIKE为RW 0.9902、IW 0.9487。论文有真实样本外成功，但S&P500不同损失/窗口的最优者不同，HARQ不是每项第一；针对所有竞争者的修正Reality Check，S&P500 MSE p=0.063、QLIKE p=0.871，不应改写为全面显著胜出。论文还对全部模型统一使用训练窗范围外预测回退到训练均值的过滤（影响<0.1%），复现需记录。

碳期货已有成功实例：2017讨论稿PDF p17 Table5，测试区间2011-06-21—2012-04-11，一日RV预测RMSE：HAR 2.238、HARQ 2.187、SHAR 2.100、ARFIMA 3.007。HARQ对HAR的RMSE约改善2.28%，但MAE 0.973差于HAR 0.939，HAR/HARQ的DM统计量1.030未标显著；这支持可行性，不支持保证在中国碳市场或日OHLC中同样成功。

## 3. Realized GARCH 与更近的既有方法

经典log-linear Realized GARCH结构：

```text
r_t = μ_t + sqrt(h_t) z_t
log h_t = ω + β log h_(t−1) + γ log x_(t−1)
log x_t = ξ + φ log h_t + τ1 z_t + τ2(z_t²−1) + u_t
```

它联合描述收益、潜在条件方差和观测波动指标；测量方程把x与h联系起来。用日range当x可研究daily-range版本；把日range称为高频realized range或宣称复制原论文高频实证则不成立。

2016 Realized EGARCH原文PDF p3更直接给出：

```text
r_t = μ_t + sqrt(h_t) z_t
log h_t = ω + β log h_(t−1) + τ(z_(t−1)) + γ' u_(t−1)
log x_(k,t) = ξ_k + φ_k log h_t + δ_k(z_t) + u_(k,t)
```

其x明确允许 daily range、squared return、RV、bipower variation、realized kernel 等多个测量。p2同时使用realized kernel与daily squared range，协方差矩阵为[[0.130,0.139],[0.139,0.418]]，已把测量精度差异和相关性纳入联合建模。因此“融合range和return”或“用观测解释潜在风险”都不是可独占的新点。

原文Table1收益样本外log-likelihood（越高越好）：2008–2013，Realized EGARCH −2251.14、GARCH −2344.21；2010–2013，分别−1290.63、−1356.73。其模型用RK+daily range，故这些数值不能当成仅用OHLC即可复现的收益。Appendix C（PDF p16）还允许测量误差方差ARCH/GARCH；故“首次时变测量误差”同样不成立，该附录报告收益部分似然并无明显提升。

其他已核对的近邻：

- Gerlach & Wang (2016), *Forecasting risk via realized GARCH, incorporating the realized range*, **Quantitative Finance 16(4), 501–511**。已核对书目，未读其完整正文：https://econpapers.repec.org/article/tafquantf/v_3a16_3ay_3a2016_3ai_3a4_3ap_3a501-511.htm 。因此range入Realized GARCH并用于风险预测也已有先例。
- Wang & Gerlach (2021预印本), *A Bayesian realized threshold measurement GARCH framework for financial tail risk forecasting*：https://ar5iv.labs.arxiv.org/html/2106.00288 ，本次web_fetch HTTP200，读取公式/方法。它已有随阈值状态变化的测量方程并预测VaR/ES；泛称“状态门控测量方程”也不足以证明新颖。这里不将预印本冒称已核实的期刊版。

## 4. 只有日OHLC＋volume：可以复现与不能假冒

| 内容 | 日线能否做 | 边界 |
|---|---|---|
| close-to-close log return及平方 | 能 | 必须是真实可比收盘价；成交均价/结算价不能无说明冒充收盘价；调整需统一作用于OHLC |
| Parkinson `p_t=log(H/L)^2/(4 log 2)` | 能 | 在理想连续扩散/零漂移等条件下的日内方差代理；稀疏交易、价差、跳跃、开盘缺口会影响它 |
| GK/RS等OHLC日波动代理 | 能 | 它们是日OHLC估计，非高频RV/RQ；不能自动消除微观结构噪声 |
| HAR式1/5/22日均值、range/return的融合、成交量状态 | 能 | 应命名daily-proxy HAR/日线融合，别直接称为原HARQ复现 |
| daily-range Realized GARCH/EGARCH | 能估计 | 原模型可接受range，零range对数需明确处理；不能复现含RK版本数值 |
| 原HARQ的RV、RQ、BPV、日内半方差 | 不能 | 至少需要真实分钟/日内收益序列；日收益四次方和rolling fourth moment不是当天RQ |
| 高频realized range、realized kernel、subsampling | 不能 | 需要每个日内区间的高低价或高频成交序列；一天H/L无法恢复路径 |
| 真正买卖价差/订单簿流动性 | 不能直接观测 | volume、零收益率、零range仅是代理；volume不能等同成交笔数或价格观测次数 |

特别注意：日range主要反映交易时段，收盘到收盘收益还包含隔夜缺口。若有开盘价，定义 `g_t=log(O_t/C_(t−1))`，把g_t²作为单独协变量或分别预测隔夜/日内部分。简单把Parkinson和g_t²相加并不自动得到无偏的完整close-to-close方差，因为还涉及漂移、协方差及测量偏差。

## 5. 一个有依据且待验证的候选方向

建议表述：**日线观测可靠性调节range与收盘收益历史贡献的条件风险预测模型**。不是“首次可靠性加权”，不是“改名HARQ”。待验证假设是：日线稀疏交易/价格停滞条件下，两类历史信号的相对预测价值系统性变化；以可观察状态控制低维来源权重，是否优于固定融合及普通volume交互。

先统一单位并定义独立目标：

```text
r_t = log(C_t/C_(t−1))
c_t = (r_t − μhat_(t|t−1))²
p_t = log(H_t/L_t)² / (4 log 2)
Y_(t,H) = (1/H) Σ(j=1..H) c_(t+j),   H ∈ {1,5,20}
```

μhat必须由当时可用历史拟合；可预先指定零均值并同时报告均值调整敏感性。Y是未来平均收盘收益创新平方这一噪声风险代理，不是高频IV；也不等于H日累计收益方差，除非另作相关结构假设。所谓“独立”仅指目标在时间上属于未来、构造上不使用range/volume，不指其与输入统计独立。

一个控制参数量的实现：

```text
q_t = [past-only scaled log(1+V_t), 1(H_t=L_t),
       past zero-return fraction, past zero-volume fraction]
w_t = sigmoid(a0 + a' q_t)
z_t = w_t * p_t + (1−w_t) * c_t
F_(t,k) = mean(z_t,...,z_(t−k+1)), k ∈ {1,5,22}
fhat_(t,H) = epsilon + softplus(b0 + Σ_k b_k F_(t,k)
                                      + d g_t² + e' q_t)
```

也可以为range与return各建相同复杂度的HAR式历史头，再用w_t组合两者的未来风险预测；需预先二选一，不能依据测试集选择。z_t为模型特征而不是宣称无偏方差。q直接项允许流动性影响真正风险，避免把所有volume效应都硬解释为测量误差；若研究重点要更强，则需独立日内验证样本检验可靠性解释。价格停滞时close return也可能失真，不能假定return永远可靠。

与HARQ的差别：HARQ利用高频渐近方差的RQ调节同一RV序列的自回归系数；候选没有RQ理论保证，使用日线状态调节两个不同观测来源。共同点是观测质量与预测权重互动，不能掩盖。

与Realized GARCH/EGARCH的差别：上式是对未来收益平方风险的直接监督/条件均值预测，未声称联合生成r与x或识别潜在IV；门控由日线q显式决定。共同点是多源波动观测、收益风险目标、可变测量效应，且现有阈值和异方差测量扩展已相当接近。

数学动机只能条件化陈述：假设两测量已去偏、对应同一潜在量，误差方差v_p、v_c、协方差κ，则最小方差线性融合中range权重为 `(v_c−κ)/(v_p+v_c−2κ)`，若要求凸组合需截到[0,1]。忽略κ才退化为常见逆方差权重。日OHLC实际通常不满足这些假设，q到误差方差的对应也未知；因此sigmoid门控是可估计近似/经验假设，不能称为HARQ渐近理论推出的最优权重。

## 6. 怎样把候选推进为真实成功实验

1. 先验证数据是同交易日、同市场、同价格口径的真实OHLC；排查均价冒充close、拼接期货、缺失/停牌、涨跌限制；当天数据只能在收盘后预测t+1。
2. 用按时间训练/验证/冻结测试。H>1时训练样本的标签终点必须在当前训练截止前；不可让跨界未来平方收益进入训练。
3. 同一目标、同一信息集比较return-only HAR/EWMA/GARCH、range-only HAR/CARR适配、固定双源、普通q协变量双源、显式q×来源交互、门控候选；可行时加入daily-range Realized GARCH/EGARCH和阈值测量版本。没有日内数据不做伪HARQ基线。
4. 训练可用 `L(Y,f)=log(f)+Y/f`，f>0；当Y=0仍有定义。它与正Y时常见QLIKE仅差依赖目标的常数。评估同时报告MSE与该损失；这是针对收益平方风险的噪声代理评分，不把每个c_t当真实方差。
5. 必须做固定w、去掉q门控、保留q直接项等消融；在同参数预算下判断是不是仅靠非线性/参数数目赢。不能只看门控图就把它叫观测可靠性证据。
6. 成功必须来自冻结测试：平均损失确实下降、时间块置信区间/DM或bootstrap适当处理相关性，多市场或多个独立时段稳定，且不是range自预测目标带来的机械优势。未获得这些结果前只称“候选”，不保证成功。VaR/ES若作为终点，应另有分布或分位数估计与对应严格一致评分，不能直接把方差预测当VaR/ES。

结论：已有非MDPI方法论文提供了测量质量影响预测、range测量可补充收益信息的实证依据；也已存在碳市场HARQ、多测量Realized EGARCH、range风险预测和阈值测量扩展。最可守住的研究问题是日线可观察可靠性是否能在独立未来收益风险目标上带来额外样本外收益。当前证据支持做实验，不支持“没人做过”或“新模型已经成功”。
