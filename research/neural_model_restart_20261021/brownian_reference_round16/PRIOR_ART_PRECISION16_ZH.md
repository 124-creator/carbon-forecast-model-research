# Precision16 前例核查：组件已知与全链路同形的边界

核查日期：2026-10-21。范围只限下列 **6 篇直接来源**；不读取本项目代码、真实 CSV、2026 行情、模拟/市场模型结果，不运行模型，也不修改任何既有文件。本报告根据提出的方法描述进行前例比对，不是对当前实现的审计。

**结论：已有强组件前例和非常接近的金融状态空间前例；本轮没有核实到完整同形构造，也没有证据足以断言它不存在。** 特别是 Kawakatsu 2023 的出版商摘要可读、全文需要认证，因此不能声称已排除该文中的条件测量方差、具体变换或校准细节。现阶段不宜使用“首次 OHLC+Kalman”“首次 angular Brownian 校准”“首次 neural learned R”等表述。

## 1. 本次比对的构造

这里的 H、L、C 指相对开盘的**对数价格增量**。令

\[
q=H-L,\quad s=(\eta,\xi),\quad
\eta=C/q,\quad\xi=(H+L-C)/q,\quad
P=q^2/(4\log2),\quad y=\log P.
\]

待核查的完整构造是：在单位方差、零漂移 Brownian 参考中校准

\[
b_0(s)=E[y\mid s],\qquad R_0(s)=\operatorname{Var}(y\mid s),
\]

随后用形状及流动性特征对 **log 观测误差方差** 做神经修正，例如

\[
x_t=\log P_t-b_0(s_t),\quad
x_t=z_t+u_t,\quad
\log R_t=\log R_0(s_t)+g_\theta(s_t,\ell_t),
\]

再在 log-variance 状态空间中滤波，生成未来风险预测。这是用于前例对照的表达，不代表已读取或确认项目代码采用此方程。

## 2. 六条直接来源及实际核查深度

| 来源 | 可直接核实的内容及位置 | 已覆盖的组件 | 尚不能据此认定同形的部分 |
|---|---|---|---|
| **[1] Kawakatsu (2023), Simple Factor Realized Stochastic Volatility Models**，[出版商 DOI 页面](https://doi.org/10.1515/jtse-2021-0049)，Journal of Time Series Econometrics 15(1), 79–110 | 出版商摘要明确：多个有噪声的日方差指标、线性状态空间表示、Kalman 信号提取/预测与似然评价；实证使用 S&P 500 日 OHLC。发表时间 2023-02-07。**仅摘要和参考文献可读，正文需认证。** | “日 OHLC 噪声测度 → 潜在方差 → Kalman”已经是直接金融前例，不能只对照一般机器人滤波文献。 | 摘要不足判断 log 还是 level 测量、R 是否随时间/形状变化、是否使用 Brownian 条件矩。对这些细节不能写“该文没有”。这是本轮最重要的证据缺口。 |
| **[2] Bollerslev, Patton & Quaedvlieg (2016), Exploiting the Errors: A Simple Approach for Improved Volatility Forecasting**，[作者所在大学公开 PDF](https://public.econ.duke.edu/~get/browse/courses/pbc/Readings/BollerslevPattonQuaedvliegJoE2016.pdf)，Journal of Econometrics 192, 1–18；DOI 10.1016/j.jeconom.2015.10.007 | 同一 PDF 的公开索引可读到 HARQ、realized quarticity、时变测量误差导致参数衰减，以及不可靠 RV 时预测向无条件方差收缩的原文。**PDF 全文抓取工具失败，本轮证据限于直接 PDF 的索引段落。** | 将测量精度差异用于改进波动预测，并不是新思想。HARQ 用 RQ 调整预测系数/有效持续性。 | HARQ 不是 OHLC angular 条件矩，也不是 neural log-R Kalman。不能把 HARQ 简化为“训练损失做逆方差加权”；其主要构造是测量误差引起的动态系数调整。 |
| **[3] Saichev, Sornette & Filimonov, Most Efficient Homogeneous Volatility Estimators**，[作者预印本全文 arXiv:0908.1677](https://ar5iv.labs.arxiv.org/html/0908.1677) | §2.4 的 homogeneous/canonical estimators、式(24)–(28)的 spherical coordinates 与 diagrams；§3/Theorem 3.1、式(35)给出由角向矩函数 g₂/g₄ 构造的最有效方差估计；附录给出 Wiener OHLC 联合分布。**相关全文已读。** | OHLC 的尺度—无量纲形状分解、Brownian 联合分布、角向矩校准、效率最优估计均已有系统理论。 | 文中优化的目标主要是点态无偏 homogeneous 估计的方差，不等于已经做了 log-error 条件 R、流动性神经残差及序列风险预测。不能把相邻目标混为同一方法。 |
| **[4] Saichev, Sornette, Filimonov & Corsi, Homogeneous Volatility Bridge Estimators**，[作者预印本全文 arXiv:0912.1617](https://ar5iv.labs.arxiv.org/html/0912.1617) | §2 定义 incomplete bridge；§3.1 将估计写为 R^λψ(Θ,Φ)；§6 明确用有限步高斯随机游走、二维 angular bins、模拟矩统计及插值构造 diagram，并比较不同 tick resolution。**相关全文已读。** | “先在标准化 Brownian/离散参考中模拟，再按角度分箱校准”的技术路线已有非常接近的前例。有限分辨率敏感性也不是新的研究要求。 | 桥路径的高低价与原始路径 OHLC 不同；该文自己特别说明这一点。只有日 OHLC 不能重构真实桥极值。本文未呈现所提的 neural observation-R/log-variance forecast 全链路。 |
| **[5] Revach et al., KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics**，[作者预印本全文 arXiv:2107.10043](https://ar5iv.labs.arxiv.org/html/2107.10043) | Introduction 的主要贡献和 §III：将 Kalman gain 计算替换为 RNN，绕开未知噪声统计；并明确与显式估计缺失状态空间参数的路线区分。**相关全文已读。** | 在结构化 Kalman 流程中嵌入神经网络、从数据学习测量更新权重已有成熟前例。 | 标准 KalmanNet 的主要学习对象是 **gain**，不能无差别称为显式 learn-R。没有 Brownian OHLC 条件矩或金融 log-variance 特定构造。 |
| **[6] Kloss, Martius & Bohg (2021), How to Train Your Differentiable Filter**，[作者预印本全文 arXiv:2012.14313](https://ar5iv.labs.arxiv.org/html/2012.14313) | §IV-B sensor network 可以输出当前输入对应的 R_t；§IV-D 明确 constant 与 heteroscedastic Q/R、正定参数化；实验比较不同噪声学习方式。§IV-E 的 Gaussian NLL 含 logdet(Σ)+errorᵀΣ⁻¹error。**相关全文已读。** | 输入相关、神经学习的观测协方差 R_t、端到端可微滤波、常数/异方差强控制，均已有直接前例；“学 logR 来保证正值”本身也不足作为方法贡献。 | 该文是一般状态估计，不包含 OHLC angular Brownian 锚定或金融风险预测任务。它足以约束组件创新主张，但不证明完整构造相同。 |

上述恰为六篇文献。检索中出现的二手目录未另算作支持方法细节的直接证据，也未展开新的文献分支。

## 3. 为什么更换形状坐标不能单独构成新方法

由提出的坐标可反解

\[
H/q=(1+\eta+\xi)/2,\qquad
L/q=(-1+\eta+\xi)/2,\qquad C/q=\eta.
\]

所以 (η,ξ) 是 OHLC 方向/形状的重参数化，与 [3–4] 的角向 diagram 思路在信息层面非常接近。菱形约束、反射折叠或画出新的坐标图，并不创造新的信息。

而且

\[
\exp\{y-b_0(s)\}
=\frac{q^2}{4\log2}\exp\{-b_0(s)\}
\]

仍是一个**二阶 homogeneous OHLC 估计器**：同时将 H、L、C 乘以 c>0，形状不变，估计器乘以 c²。这说明形状偏差修正处在既有 homogeneous 家族内；它不自动获得该家族文献中“无偏”“最有效”的结论，因为目标损失与约束不同。

从 [3–4] 的联合分布或模拟矩出发，计算 log-error 条件均值/方差是自然的统计延伸。本轮没有在其相关全文中见到与本描述逐项一致的 log b/R 表与后续神经 Kalman 预测，但不能把“自然延伸”直接等同为“已有同一公式”，也不能将其包装为首次使用 Brownian/angular 几何。

## 4. 若真实预测成功，最窄可辩护的贡献

比较稳妥的条件式表述是：

> 在已有 homogeneous OHLC 与神经异方差滤波框架上，提出一个由标准化 Brownian log-measurement 条件矩锚定、再由观测形状和流动性做有限神经修正的测量精度模块；在固定偏差校正与潜在状态动态后，检验该模块对未来风险预测是否有增量价值。

其中可争取的贡献是**特定参考校准与有限数据神经迁移的结合、清楚的测量精度机制，以及严格控制下的实证增量**。不能据此主张：发明 OHLC angular geometry、首次 measurement-error-aware forecasting、首次 OHLC Kalman、首次 neural R、恢复了真实 IV，或发现了普遍成立的新市场规律。

如果收益只来自 b(shape)，贡献应归为 homogeneous 测量偏差修正；如果无 Brownian 锚定的普通 learned-R 同样好，不能将收益归因于 Brownian 先验；如果简单 log-linear R 同样好，不能将收益归因于神经表达能力。

## 5. 必要强基准：首先分离均值、精度、先验与神经能力

| 基准/控制 | 要回答的问题 |
|---|---|
| **同一个 b(shape) + 最佳训练常数 R**；并列全局 b/global R | 精度收益是否超越已有偏差修正？必须保持相同观测变换、状态转移与 Q，不能只打败未经形状校正的 Parkinson。 |
| 固定 Brownian R₀(shape)，不加神经修正；同一滤波骨架中的经典 GK/RS、适用时 YZ 或 homogeneous diagram 测量 | 是否仅需要已知 OHLC 估计器或静态参考精度，根本不需要神经部分？桥估计器仅在确有桥路径极值时适用。 |
| 同样 shape/liquidity 输入的非神经异方差 R，例如正则化 log-linear/spline；以及去掉 Brownian offset 的等容量 learned-R | 改进来自一般异方差权重，还是 Brownian 锚定与神经修正的特定结合？需匹配训练信息、搜索预算和正则化。 |
| shape-only、liquidity-only 的 R 消融；同样特征进入状态/预测均值而 R 固定的控制 | 流动性或形状是在提示测量可靠度，还是仅携带真实波动的预测信号？避免把特征预测能力直接解释为 measurement precision。 |
| Kawakatsu 风格的单/多 OHLC 测度因子状态空间，允许多个同日测度相关 | 与最接近的金融前例相比是否有增量？准确复刻需先获得其完整方程；不能仅凭摘要把自写 KF 标作该文复现。 |
| HAR；有真实 intraday RQ 时加入原 HARQ；风险分布任务加入适当的简单 return-volatility 模型 | 是否超过成熟的测量误差波动预测与低参数风险基准？仅日 OHLC 时不可把 r⁴ 或任意代理命名为原论文 realized quarticity；只能明确称 proxy-adapted analogue。 |

这些是用于界定未来贡献的比较要求，本轮没有运行或读取任何基准结果。

还需固定预测时点：当日完整 OHLC/成交量只能在其可得之后用于过滤和 t+1 及以后预测，不能用于开盘前或当日事前预测。测量分数好转与未来收益/风险预测好转必须分别报告，不能用模型自身修正后的同一 OHLC 量自证真实 IV。

R_t 的可解释性也需克制：Kalman innovation covariance 同时含预测状态不确定性与 R_t；灵活 R 网络可能补偿 Q、状态动态或分布设定错误。仅靠端到端似然改善，通常不能证明它识别了真实测量误差方差。固定/匹配动态、独立校准诊断以及上述强控制，比泛称“可解释精度”更重要。

## 6. 核查范围内能说与不能说的结论

- **能说**：六个直接来源明确覆盖了几乎所有单独组件；最接近的金融前例是 Kawakatsu，最接近的 angular 模拟校准前例是 [3–4]，显式 neural R 前例应引用 [6]，而不是误写标准 KalmanNet 已直接学习 R。
- **能说**：在本轮可访问证据中，尚未核实完整“Brownian b/R → shape+liquidity 神经 logR → log-variance Kalman → 未来风险预测”同形方法。
- **不能说**：检索已经证明该完整构造首次出现。尤其 Kawakatsu 全文未获访问，HARQ 本轮只核对了直接 PDF 索引段落；这些限制必须随新颖性结论一并保留。

本报告不评价当前代码、模拟或市场实验是否成功，不给期刊录用概率。
