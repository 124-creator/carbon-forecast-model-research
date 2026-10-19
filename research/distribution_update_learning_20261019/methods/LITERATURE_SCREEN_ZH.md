# 碳价格方法研究：新一轮原始文献核查与可执行数学子问题

本文件是筛选阶段记录；最终执行定义与判断以 METHOD_ZH.md 和 REPORT_ZH.md 为准，以下筛选时的历史判断予以保留。

日期：2026-10-19。作者角色：独立文献与方法审查。此稿是中间研究记录，不是实验结果或已完成的新方法。已复读旧实验的 REPORT_ZH.md 与 DIRECT_NEIGHBORS_ZH.md；未训练任何真实市场模型，未改动旧实验、预测、评分或 GitHub。

## 1. 核心判断

从价格均值扩展到条件分布预测，可以形成方法论文，但“换预测对象、改用 CRPS、再加几个网络模块”不能构成方法创新。旧实验没有通过价格 MSE 门槛，这个结论继续成立。新方案即使概率评分更好，也必须单独说明成功的是哪个预测对象；若用户仍坚持价格 MSE 为唯一主目标，概率路线不能替代它。

本次只保留一个值得进入合成开发的数学子问题：利用未来路径上可预测的控制变量，提高对分布更新收益的估计精度，并把弱对称性引入的偏差显式计入更新条件。其基础零均值控制、控制变量降方差、概率控制变量、CDF 混合都已存在；尚待形成的增量是“未知真实过程下、按时间拟合、偏差预算有效时才采用分布更新”的具体学习算子与可执行误差界。目前没有足够证据宣布它已达到论文新颖性要求。

角向 Rao–Blackwell 与全部连续区间 CRPS 两个候选均不建议按现有版本进入正式方法实验：前者与群平均论文的径向估计特例直接重合；后者的模型头、聚合监督与 CRPS 训练都已有直接近邻。它们可作比较工具，但不能靠碳价上的一次改善补足方法贡献。

## 2. 近期碳价格研究：已读全文与仅能核到摘要须分开

阅读深度 A：已读原始全文的方法、关键公式或实验设置。B：原始出版页只取得摘要/元数据，尚未核全流程。B 类不用于证明算法有效、切分合法或超越随机游走。

| 论文与确切来源 | 阅读深度及证据 | 对本研究的影响 |
|---|---|---|
| Berrisch & Ziel, CRPS Learning, Journal of Econometrics 237(2), 105221 (2023)，DOI 10.1016/j.jeconom.2021.11.008；[作者全文](https://arxiv.org/pdf/2102.00968)，[代码](https://github.com/BerriJ/profoc) | A：方法与第 7 节应用。2021 年预印本/接受稿，不能把方法起源写成 2023。应用是 2013–2020 年 2,092 个 EUA 日度月前期货记录，滚动训练 250 个观测、1,841 个样本外预测、99 个分位数。 | 已有真正的碳价概率学习方法：Bernstein 在线聚合，权重随时间和分位数变化，用样条控制复杂度并给出学习性质。泛泛的“动态尾部分位权重”不是新贡献。作者库有代码；数据代理本轮未在固定主分支找到原始 2,092 条碳价，不能宣称数据公开到手。 |
| Bastianin, Mirto, Qin & Rossini, Forecasting the Price of Carbon with Macroeconomic and Financial variables, JEDC (2026)，DOI 10.1016/j.jedc.2026.105435；[作者全文](https://arxiv.org/html/2402.04828v3) | A，承接上一轮全文审查：月度 BVAR，136 个样本、2012-06 至 2023-09；主要金融输入来自 LSEG。 | 短期随机游走很强，较长期点预测改善也不能跳过统计显著性审查。月度宏观预测不适合直接当日度神经模型的大样本证明。 |
| Han et al., Causal neural network for carbon prices probabilistic forecasting, Applied Energy (2025)，DOI 10.1016/j.apenergy.2025.126343；[出版入口](https://www.sciencedirect.com/science/article/pii/S0306261925010736) | B：官方索引摘要与作者机构出版清单；全文抓取失败。摘要描述 EPPT、monotonic composite quantile causal TCN、密度构造。 | 应列直接应用近邻；尚不能核实其分解是否仅用历史、实际基线强度或因果识别。标题的 causal 不能自动解释为干预因果模型。 |
| A novel probabilistic carbon price prediction model: Integrating the transformer framework with mixed-frequency modeling at different quartiles, Applied Energy (2025)，DOI 10.1016/j.apenergy.2025.125951；[出版入口](https://www.sciencedirect.com/science/article/pii/S0306261925006816) | B：官方索引摘要；QRTransformer–MIDAS、上海/湖北、分位数与区间。未取得方法正文。 | “分位数 Transformer + 混频外生变量”本身已有碳价先行工作；不据摘要采信其样本外改善或泄漏控制。 |
| European Union Allowance Price Forecasting With Multidimensional Uncertainties: A TCN-iTransformer Approach for Interval Estimation, Journal of Forecasting，DOI 10.1002/for.70024；[官方页](https://onlinelibrary.wiley.com/doi/full/10.1002/for.70024) | B：入口跳至摘要。首次在线日期 2025-09-21，不能只按 2026 期次写成 2026 年新提出。摘要含 QR、conformal 与 TCN-iTransformer。 | 概率/区间预测加 conformal 已有碳价近邻；未读正文就不能承诺有限样本覆盖保证适用于依赖、非平稳数据。 |
| Machine Learning Forecasts of Tail-Risk Spillovers in Carbon and Energy Markets, Journal of Forecasting (2026)，DOI 10.1002/for.70118；[官方页](https://onlinelibrary.wiley.com/doi/10.1002/for.70118) | B：首次在线 2026-02-16；摘要包括 MVMQ-CAViaR 与分位机器学习。 | 风险溢出预测与单市场价格条件分布并非同一任务；只列相关线索，不作为具体算法依据。 |
| An integrated prediction framework combining variational mode optimization decomposition, empirical quantile regression, and QLattice for dynamic point and interval prediction of carbon prices, Energy (2026)；[出版入口](https://www.sciencedirect.com/science/article/abs/pii/S0360544226004755) | B：仅题名/出版索引；作者仓库线索 StriveOrange/TGSINFO-VMD-NEQQLattice 尚未由本代理代码审查。 | 不把“优化分解 + 分位数 + 符号回归”的组合重新包装为本轮方向；目前不能引用它的具体提升数字。 |

## 3. 对方法贡献限制最直接的原始全文

下表不是泛泛综述；每篇只记录与本轮候选直接相交的部分。

| 原始文献 | 已读位置 | 不可重命名的既有机制 |
|---|---|---|
| Urettini et al., GAS-Norm: Score-Driven Adaptive Normalization for Non-Stationary Time Series Forecasting in Deep Learning, CIKM 2024，DOI 10.1145/3627673.3679822；[全文](https://arxiv.org/html/2410.03935v1) | §3.1–3.3 | GAS 位置/尺度过滤、神经网络前归一化和输出反归一化已存在；GAS + 神经预测器不够新。 |
| Urettini et al., Online Continual Learning for Time Series: a Natural Score-driven Approach，arXiv:2601.12931，2026-01-19；[全文](https://arxiv.org/html/2601.12931v1) | §4.1–4.5 与定理假设 | Student-t、自然梯度、动态尺度及持续更新已有具体方法。局部强假设下的界不能扩大为任意市场非平稳保证。 |
| van Brummelen, Gorgi & Koopman, Score-driven time-varying parameter models with spline-based densities, Statistics and Computing 36:94 (2026)，DOI 10.1007/s11222-026-10840-w；[正式全文](https://link.springer.com/article/10.1007/s11222-026-10840-w)，[作者稿](https://papers.tinbergen.nl/25011.pdf) | §2.2–2.5、§3.1 | 灵活样条密度与 GAS 动态位置/尺度已有方法。静态密度形状与动态状态须区分；不能说“首次放松正态误差”。 |
| Livieri & Palmari, Proper-Score Observation-Driven Filters: Local Geometry, Estimation, and Continuous-Time Limits，arXiv:2608.02828v3，2026-09-21；[全文](https://arxiv.org/html/2608.02828v3) | 核心构造与 Appendix G 的 Gaussian/t、CRPS/log/density-power/MMD 公式；未逐页读完全部长稿 | CRPS 驱动 observation-driven 状态更新已经有直接一般构造。准模型错设下，不同 proper score 可指向不同目标；稳健梯度不等于正确尾部概率。 |
| Zhao et al., From GARCH to Neural Network for Volatility Forecast, AAAI 2024，DOI 10.1609/AAAI.V38I15.29643；[全文](https://arxiv.org/html/2402.06642v1) | 模型映射与 GARCH/GJR/FIGARCH 的网络结构 | 用波动模型结构初始化/限制神经预测器已有方法。 |
| Buczynski & Chlebus, GARCHNet: Value-at-Risk Forecasting with GARCH Models Based on Neural Networks, Computational Economics 63:1949–1979 (2024)，DOI 10.1007/s10614-023-10390-7；[全文](https://link.springer.com/article/10.1007/s10614-023-10390-7) | §2.3–2.4 | 神经方差、动态自由度与偏度、似然估计已有先例。比较组不能只有 Gaussian 神经头。 |
| Chen, Dobriban & Lee, A Group-Theoretic Framework for Data Augmentation, JMLR 21(245):1–71 (2020)；[原刊全文](https://jmlr.org/papers/volume21/20-163/20-163.pdf) | §3–4、近似不变性讨论与 §8.2 | §8.2 直接用完整向量范数和随机旋转改善单坐标边际密度估计。群平均/Rao–Blackwell 降梯度方差与近似不变性的偏差收益权衡均已有框架。 |
| Tong et al., Rao-Blackwell Gradient Estimators for Equivariant Denoising Diffusion, NeurIPS 2025，arXiv:2502.09890；[作者全文](https://starai.cs.ucla.edu/papers/TongNeurIPS25.pdf) | §3.2、Theorem 1、Appendix B.1 | 以群轨道上的条件均值替换训练目标、降低梯度方差已有明确新方法；有限自归一重要性采样不能直接叫精确无偏 RB。 |
| Hesterberg & Nelson, Control Variates for Probability and Quantile Estimation, Management Science 44(9):1295–1312 (1998)，DOI 10.1287/mnsc.44.9.1295；[作者全文](https://users.iems.northwestern.edu/~nelsonb/Publications/HesterbergNelson.pdf) | §1–2 与概率/分位数控制构造 | 指示函数/CDF/分位数控制变量已经存在。与本任务的差别是他们的模拟控制信息，不是未知真实金融过程上的条件学习。 |
| Sabate-Vidales, Šiška & Szpruch, Unbiased Deep Solvers for Linear Parametric PDEs, Applied Mathematical Finance (2022)，DOI 10.1080/1350486X.2022.2030773；[作者全文](https://arxiv.org/pdf/1810.05094) | §2，式 7–9，算法讨论与 Appendix B | 神经网络学习可预测随机积分控制，在已知模拟 SDE/Brownian 驱动下无偏。不能把它的已知零均值性质直接赋给从市场拟合的残差。 |
| Pasupathy et al., Control-variate estimation using estimated control means, IIE Transactions 44:381–385 (2012)，DOI 10.1080/0740817X.2011.610430；[作者全文](https://web.ics.purdue.edu/~pasupath/PAPERS/2012pastaaschwan.pdf) | §1–3，式 1–4 | 控制均值估计误差、额外采样方差和最优权重收缩已明确处理；其前作 Biased control-variate estimation (2001)，DOI 10.1080/07408170108936824，已处理近似均值产生的偏差。不能把加入偏差惩罚的最优 CV 系数当创新。 |
| Pic et al., Proper scoring rules for multivariate probabilistic forecasts based on aggregation and transformation, ASCMO 11:23–58 (2025)，DOI 10.5194/ascmo-11-23-2025；[原刊全文](https://ascmo.copernicus.org/articles/11/23/2025/index.html) | §3、§4.1–4.2，式 16 | 固定变换后 proper score 再非负加权；明确包括 patch mean、sum、min、max，且适用于时间预测。全部连续区间 sum-CRPS 是其直接特例。 |
| Olivares et al., CLOVER: Probabilistic Forecasting with Coherent Learning Objective Reparameterization, TMLR (12/2024)；[作者全文](https://www.stat.berkeley.edu/~mmahoney/pubs/3148_clubsuit_CLOVER_clubsuit_.pdf) | §3.1–3.3，式 9–17 | 低秩因子加对角 Gaussian 头、固定线性聚合、对 aggregate 的 CRPS 端到端优化都已存在。其一般构造允许更换可微采样分布。 |
| Si et al., Semi-Autoregressive Energy Flows: Exploring Likelihood-Free Training of Normalizing Flows, ICML 2023，PMLR 202:31732–31753；[原刊全文](https://proceedings.mlr.press/v202/si23a/si23a.pdf) | §3 的式 4/Theorem 2 与 Appendix B | 固定方向分布的一维投影 CRPS/energy 训练多元生成模型已经存在。注意不能不加条件照搬其关于全部分布识别的结论到固定有限方向集合。 |
| Zheng & Sun, MVG-CRPS: A Robust Loss Function for Multivariate Probabilistic Forecasting，arXiv:2410.09133v2 (2025-01-31)；[全文](https://arxiv.org/html/2410.09133v2) | §3、Appendix A 与 C.5 | Gaussian 协方差特征分解、旋转后 Gaussian CRPS，作者给出了该 Gaussian 类内 strict propriety 证明。固定区间投影没有这个特征分解步骤，但这一实现差别本身不足以证明全新学习机制。 |

另已读 Kleen (2024), Scaling and measurement error sensitivity of scoring rules for distribution forecasts，DOI 10.1002/jae.3056，[原刊全文](https://onlinelibrary.wiley.com/doi/full/10.1002/jae.3056) 的尺度命题与测量误差部分；读 Patton 的 [Volatility forecast comparison using imperfect volatility proxies 作者稿](https://public.econ.duke.edu/~ap172/Patton_robust_JoE_forthcoming.pdf) 的代理稳健损失命题。普通 RV + QLIKE 不是新方向；不能从代理噪声稳健的方差损失推导任意分布伪标签稳健。

## 4. 保留的一个开发候选：有偏差预算的条件分布更新

### 4.1 研究问题与最小目标

问题不是“怎样再造一种 Gaussian 混合模型”，而是：在样本有限、路径重叠、部分条件对称可能失效时，能否更准确地判断一个分布更新是否值得采用？只有当控制变量减少的不确定性超过其潜在偏差，更新才应加大。

建议先研究有限状态或固定条件设计的 H 步收益分布。不要一开始引入 Transformer、可学习径向矩阵或多市场迁移。两个较早时期拟合的候选 CDF F0 与 F1 足够检验学习算子；F0 可是原有分布模型，F1 是其一轮新数据更新。最后输出仍是合法概率分布。

### 4.2 控制项：避免整条未来路径旋转的错误

记当前信息为 F_t，未来增量为 r_(t+j)，Y_h 为其累积和，Z_h(z)=1{Y_h≤z}。令 s_(j-1)>0 和 a_j(z) 只依赖起点信息与 j−1 之前已实现的前缀，并在所评分数据块之前冻结其拟合规则。定义

M_h(z)=Σ_(j=1)^h a_j(z) tanh(r_(t+j)/s_(j-1)).

若增量条件分布关于零对称，则每个 tanh 项相对当时信息均值为零，从迭代期望得到 E[M_h(z)|F_t]=0。GJR 式 leverage 可以改变下一期尺度，但不破坏这一逐步、可预测的性质。

这与“给定整条未来幅度路径，任意翻转各期符号”不同。后者在 leverage 下通常错误：后期幅度携带早期符号信息。

若只有弱对称性，假设

|E[tanh(r_(t+j)/s_(j-1)) | F_(t+j-1)]| ≤ δ_j，

则

|E[M_h(z)|F_t]| ≤ E[Σ_j |a_j(z)| δ_j | F_t]。

这只是一个可审查的假设与基本上界，不是已经从碳价证明的事实。使用估计的条件中心残差时，其中心误差必须计入 δ_j；“样本均值为零”不等于“条件均值为零”。

零偏差情形下可用 Z−M 作 CDF 伪标签。对固定、不随被优化预测器变化的控制项，积分 Brier 风险的差异只含预测器无关项，因此保留原风险最优点。伪标签可能在 [0,1] 外；不能把它硬裁剪后还宣称无偏。输出 CDF 本身仍须满足 [0,1]、单调性与正确尾端条件。

### 4.3 实际更新算子

令 Δ(z)=F1(z)−F0(z)，并定义 Fα=F0+αΔ，0≤α≤1。对于 CRPS 风险 R，有精确恒等式

R(Fα)−R(F0)=Aα²−2Bα，

A=E∫Δ(z)² dz，
B=E∫[Z_h(z)−F0(z)]Δ(z) dz。

利用控制项估计

B_cv=E∫[Z_h(z)−M_h(z)−F0(z)]Δ(z) dz。

如果可给出有效偏差界 |B−B_cv|≤b，校准样本给出 B_cv 的下界 hatB_cv−u，以及 A 的上界 A_upper，则令

B_lower=hatB_cv−u−b，
α=clip(B_lower/A_upper, 0, 1)。

若 A_upper=0，两个 CDF 几乎处处相同，定义 α=0。在事件 {B≥B_lower，A≤A_upper} 上，

R(Fα)−R(F0) ≤ A_upper α²−2B_lower α ≤ 0。

不能把随机 hatA 直接当真值。固定状态下，A 可以直接对两个 CDF 积分；若泛化到随机未来状态分布，需要其有效上界。有限阈值网格时，上述结论严格对应网格 Brier 目标；称完整 CRPS 需要处理积分/尾部误差。

这个算子的可检验机制是：M 减少 B 估计的方差，从而缩小 u；弱对称预算 b 会抵消收益。与普通 CRPS 在线聚合的区别应落在“用于决定更新的收益估计如何获得、何时拒绝采用有偏控制”上。单纯输出两个 CDF 的凸组合不是创新，置信界为负便不更新也是通用安全改进思想，均需在投稿前继续核直接近邻。

### 4.4 尚缺的真实方法贡献

目前缺的不是再加网络，而是一个不会掩盖未知条件均值的构造：如何在有限依赖历史中产生足够紧、真正对目标时期有效的 b 与 u。严格对称、已知 δ 的合成过程容易建立；从真实训练数据估计条件 δ，再用于非平稳未来，就需要新的假设、漂移预算或新的理论/校准算法。若做不到，只能叫经验更新规则，不能声称“有保证不恶化”。

Hesterberg–Nelson 已覆盖概率控制变量；神经 martingale CV 已覆盖可预测控制学习；2001/2012 CV 文献已覆盖有偏/估计控制均值及收缩；CRPS Learning 已覆盖动态分布聚合。真正可能保留的增量是上述顺序条件下可实现的收益界与更新方式及其统计效率，尚未证明新颖性或有效性。

尤其要防止研究目标自相矛盾：若对价格增量硬假设条件均值为零，则价格条件均值就是随机游走，不应同时把该假设当作战胜随机游走均值的理论根据。对数收益的分布预测也不能直接替代价格均值预测。

### 4.5 可立刻执行的 CPU 合成开发方案

该方案是开发实验，不是正式市场试验，也不应在失败后只保留好看的设置。

1. 使用 H∈{1,5,20}，两个至四个可观测状态；分布头为 Gaussian 或 2–3 个 Gaussian 成分，微型 MLP 或线性状态参数。固定所有方法相同容量和训练预算。
2. 三层环境：严格条件对称且带 GJR leverage；轻微位置/偏度扰动且 δ 已知；相同扰动但 δ 只能从较早历史估计。另加入状态漂移反例检验校准界失效。
3. 分开较早拟合块、控制系数块、收益校准块、未来评估块。a_j 不能用当前评估标签训练，所有 H 步标签成熟以后才能用于下一轮更新。
4. 对照固定 F0、直接 F1、普通原标签 CRPS 的混合更新、无偏差预算的 CV 更新、带预算但不用 CV 的更新、完整候选。完整算子必须超过“同样谨慎但无控制变量”的对照，才说明控制机制有价值。
5. 合成真分布已知时，核 b 是否真实覆盖、u 的覆盖率、风险是否按证书不恶化、控制方差比、非零更新率和未来真实 CRPS。独立 Monte Carlo 只用于计算合成真风险，标注其误差。
6. 第一轮否决条件：严格对称下都不能比原标签估计明显降低收益估计方差；只有 oracle δ 有效；估计 δ 的成本导致几乎全部 α=0；所谓安全更新仍系统性增加真风险；只在没有 leverage 的球对称理想情况有效。任一发生便记录原因，停止把它当成熟方向。

最有价值的数学交付是一个清楚的边界：在何种非平稳/条件偏差范围内，控制变量能够更快辨认一个真实有益的分布更新；在何种范围内，额外偏差与校准成本让方法应当退回原预测。即使最终否决，也能在小型合成开发内给出答案，而无需再消耗一次大规模碳价筛选。

## 5. 两个已否决候选及其精确原因

### 5.1 固定椭球形状的角向 Rao–Blackwell

原候选 r=A R U，U 球面均匀。固定 A 时，投影 c_h^T r 等于 a_h R U1，a_h=||A^T c_h||，U1² 服从 Beta(1/2,(H−1)/2)。这一表示本身成立，但 Chen–Dobriban–Lee §8.2 已直接讨论径向范数和随机旋转改善坐标边际分布；换成 CRPS 不是足够差别。

本审查另有一个直接代数发现：若所有期限共享同一固定 A 和条件径向分布，CRPS 的尺度齐次性使角向平均的各期目标只是同一标量损失乘 a_h。因此 Σ_h w_h score_h=(Σ_h w_h a_h) score_common。多期限在这里没有提供新的独立分布约束。若 A 随历史变化，仍在每个历史条件下退化为该形式，仅改变样本权重。

若 A 与预测器联合训练，R=||A^(-1)r|| 也随参数变化；不能省略其导数/条件测度变化后继续宣称相同风险的无偏梯度。训练期估计 A 能避免数据泄漏，不代表真实未来 law 符合椭球假设。

结论：当前版本否决，不再追加真实市场实验。

### 5.2 全部连续区间的解析 Gaussian-mixture CRPS

对 H 维未来增量，所有区间 I=[a,b] 的投影 c_I 为固定 0/1 向量。用预测分布的 sum_I r 的 CRPS 训练是 proper 的，但 Pic §4.2 的 sum-patch CRPS 已完整覆盖这个损失构造；CLOVER 已覆盖低秩+对角头、固定聚合矩阵与 aggregate CRPS 训练。Si et al. 已有投影 CRPS 学习；MVG-CRPS 已有 Gaussian 联合评分学习。解析计算及前缀和可以改善实现，却尚不足以支撑方法论文。

可以如实保留以下初等识别性质。若 V_ab=Var(Σ_(j=a)^b r_j)，则

Σ_aa=V_aa，
2Σ_ab=V_ab−V_(a+1,b)−V_(a,b−1)+V_(a+1,b−1)，a<b，

空区间的 V 定义为 0。故全部 H(H+1)/2 个区间方差可反演无结构对称协方差。均值由单步投影均值识别；有限二阶矩、共同径向 law 且尺度归一化的椭球类中，再配合标准化一维边际可识别共同生成元。

但必须附上限制：这不是任意非椭球 joint law 的识别；H(H+1)/2 的最小性仅指无结构对称矩阵的独立二次型测量，并不适用于实际低秩+对角子类；共同 Gaussian-scale-mixture 的尺度与径向分布需规范化，否则参数不唯一；210 个区间标签不等于 210 个独立样本。根代理提出的差分与识别逻辑正确，新颖性边界仍然很窄。

## 6. 如果发展条件分布方向，应设置的基线与共同协议

### 6.1 分布模型基线

| 组别 | 最少实现 | 比较目的 |
|---|---|---|
| 价格点预测 | 原价格单位 RW、drift、适当 ridge/小模型；独立报告价格 MSE | 保留原用户精度目标，不能用 CRPS 改善替代。 |
| 简单概率预测 | 固定历史波动、EWMA；Gaussian/t 创新；filtered historical simulation | 概率随机游走应有可校准创新分布，不要只用零宽度点质量作弱对手。 |
| 波动模型 | GARCH(1,1)、GJR-GARCH，t 与 skew-t；共同条件均值设置 | 检验厚尾、波动聚集、leverage 是否已解释提升。 |
| GAS | t location-scale GAS，必要时 skew-t；固定与在线更新方式明确 | 避免把自适应位置/尺度重新命名。 |
| 混合密度与神经波动 | 同容量 Gaussian mixture density network；微型 GRU 方差/t/skew-t 头；GARCHNet 思路的动态参数对照 | 隔离学习算子与分布族、容量的影响。原始 MDN：[Bishop 1994 作者报告](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-ncrg-94-004.pdf)。 |
| 直接分位数与风险 | 线性/正则化分位数、分位数树；若主张 VaR/ES，再加入 CAViaR 或 FZ 联合动态模型 | 尾部效果不能只和均值回归相比。[Patton–Ziegel–Chen 原文](https://arxiv.org/pdf/1707.05108)。 |
| 更新算子对照 | 普通 CRPS 更新、相同置信预算但无控制项、biased CV、CRPS Learning/BOA | 直接检验本轮究竟新增了什么。 |

基线与候选使用相同历史信息、成熟标签、再训练日程和选择预算。不能只对候选动态调参而让经典模型永久不更新。对所有基线做无限参数网格也没有必要；选择能够覆盖核心机制的有限、预先说明的设定。

### 6.2 价格均值存在性

若 Student-t/skew-t 放在对数收益上，再指数映射到价格，正向指数矩通常发散，价格均值可能不存在。此时 exp(预测对数均值) 是某种中心/中位数类量，不是价格条件均值；有限 Monte Carlo 样本均值也不能修复不存在的总体均值。

若必须同时输出价格 MSE 的分布均值，可使用具有有限指数矩的族，例如有限 Gaussian mixture 的对数收益模型，并准确计算指数矩；或对简单收益/价格增量建立支持和尾部清楚的模型。t 类基线可用于收益密度、分位数与尾部比较，同时单列其点预测规则。不得把两类预测含混为一列。

### 6.3 多步与评分

h=5/20 的递推分布须更新模拟路径中的状态/波动，不应默认按 sqrt(h) 缩放就等于真实 GARCH 多步预测。密度/分位数的积分或 Monte Carlo 误差应低于所声称提升。Sobol/反向对偶可降低积分误差，但不能把相关样本代入 iid 的无偏 U 统计公式后继续沿用该解释。

主要概率指标可用原目标上的 CRPS，补充 log score、PIT/条件分位覆盖；若主目标是 VaR/ES，则使用一致的 quantile/Fissler–Ziegel 联合损失与风险校准诊断，不能以平均区间宽度单独判优。少量年度切分对 1% 尾部的有效事件太少，5%/10% 通常更适合第一轮有识别力的开发检验。

所有阈值网格、尺度、波动代理和分位水平在训练期确定。样本外仅对共同起点比较真实标签的 proper score；不能在测试时用降噪伪标签评分。多期限同起点、重叠路径与多个模型的配对关系在 block bootstrap 中必须绑定。非平稳数据上的移动块区间本身不是无条件覆盖保证，方法论文中须明确推断范围。

## 7. 阅读与执行收据

A 类方法全文主要合法来源为 JMLR/PMLR 原刊、arXiv 作者稿、大学作者页面、Springer/Wiley 正式全文。未使用盗版论文库。OpenReview 本轮出现浏览器验证，已改用公开作者机构稿，未尝试绕过访问控制。部分 Elsevier/Wiley 出版页只能得到摘要，已明确标 B，不以猜测补正文。

本稿含若干本代理的代数推导与开发建议，已与论文作者的结果分开。没有可支持“尚无任何先行工作”的完备证据；检索没有命中完全相同标题不代表新颖性。当前推荐是针对第 4 节的一个窄数学问题做合成否决实验，完成偏差/置信界审查后才决定是否用真实碳价发展方法论文。

## 8. 后续补查：风险校准近邻与开发前必须加入的限制

### 8.1 三篇更直接的原始方法近邻

| 原始论文 | 实际阅读全文位置与已核机制 | 对创新边界的影响 |
|---|---|---|
| Almeida et al., High Probability Risk Control Under Covariate Shift, PMLR 266 (2025)，[作者机构全文](https://research.feedzai.com/oshuzuxa/2025/10/high_probability_risk_control_under_covariate_shift.pdf) | A：第 2 节风险校准框架、第 3.2 节、附录 D 全文。附录将控制变量加入重要性加权风险估计：以已知期望为 1 的密度比权重作为控制量；前半数据估计系数，后半数据做风险控制。作者报告其设置中方差下降未补偿分样与取值范围扩张，实际校准变得更保守。 | “控制变量使风险校准更有力”已有非常直接的尝试。不能把它作为本候选的首次贡献；其失败机制恰好是必须纳入的样本效率否决条件。本文未给出本稿的弱对称路径控制与 CRPS 二次收益更新。 |
| Einbinder, Ringel & Romano, Semi-Supervised Risk Control via Prediction-Powered Inference，TPAMI (2025)，DOI 10.1109/TPAMI.2025.3594263；[作者全文](https://arxiv.org/pdf/2412.11174) | A：第 IV 节定理 1、第 V 节及式 5–8。风险估计等于无标签集的插补损失平均，加上有标签集的真实损失减插补损失；再构造有效风险上界。论文处理预测器误差、估计方差及有限样本校准，差的插补也可能增加方差。 | “以辅助预测降低风险估计噪声，再用置信界决定预测器”已存在。它的校正来自额外无标签数据与标注残差；本候选拟用路径的条件矩限制。两者不能互相继承保证。 |
| Active, anytime-valid risk controlling prediction sets，arXiv:2406.10490；[作者全文](https://arxiv.org/pdf/2406.10490) | A：第 2.1–2.2 节、式 8、命题 3 及证明。可预测风险估计和已知随机标注概率进入残差修正，构造适用于持续取样的风险检验过程。风险预测误差影响效率；有效性依赖其条件无偏结构。 | 顺序控制变量、预测风险校准、持续更新均不是空白。该文的零均值来自已知标注机制；我们若只有不精确的弱对称性，就必须单独支付偏差预算，不能直接沿用其任意时刻保证。 |

因此，本稿建议仅保留一个尚待验证的增量：没有额外标注资源时，利用成熟未来路径的可预测控制与显式弱对称偏差上界，改进一个具体 CRPS 分布更新的收益辨识效率。是否存在完全相同的学习算子仍未被排除；以上差异只是待检验研究边界，尚不等于可发表贡献。

### 8.2 有限控制系数选择必须一并校准

以下是本轮与独立审计代理讨论后的算子细化，不归于上述作者。令每个校准观测上的收益为 B_i，积分控制量为 C_i，且 |E C_i| 的相应总体平均不超过 b。系数来自事先指定的有限集合 Λ，并包含 0。对每个 λ，构造同时有效的均值下界：

L_λ = mean(B_i − λ C_i) − u_λ − |λ| b，

L = max_(λ∈Λ) L_λ，

α = clip(L/A_upper, 0, 1)。

在所有候选下界同时成立、A≤A_upper 的事件上，第 4.3 节的风险不恶化推导仍成立。λ=0 给出同一候选集内无控制变量的退路，但在有限样本中增加 Λ 会付出多重选择代价；不能将它误解为一定优于独立校准的无控制方法。若 F0、F1 或控制学习器也在该校准块上选择，其选择必须纳入有效界或移到更早拟合块。

[Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control](https://arxiv.org/pdf/2110.01052) 已明确用有限候选的多重检验保护选择；已读第 2.1 节、定理 1 与第 2.2 节。初稿年份为 2021，本轮读取的 PDF 标注 2022-10-03，不在此推断最终出版年。因此“先得到联合保证，再自适应选择通过者”不是新原理。

[Maurer & Pontil, Empirical Bernstein Bounds and Sample Variance Penalization (2009)](https://arxiv.org/pdf/0907.3740) 已给出利用样本方差的有界变量均值界和有限类统一界；已读定理 4、推论 5、定理 11 与假设。其独立观测条件不能直接替换为重叠 H 步金融路径。即使隔 H 个起点取样，GARCH 的状态依赖仍在；独立但非同分布的平均期望界，也不自动保护另一个漂移后的未来时期。

### 8.3 无界真实价格不必使相对收益估计无界

对每个起点，令 Δ_i=F1_i−F0_i。第 4 节的观测收益满足

|B_i| = |∫(1{Y_i≤z}−F0_i(z)) Δ_i(z) dz|
       ≤ ∫|Δ_i(z)| dz = W1(F0_i,F1_i)。

因此若候选预测器之间的 W1 距离事先统一限制在 L_max，即使真实 Y 没有有界支持，B_i 也在 [−L_max,L_max]。若另有 |M_i(z)|≤M_max，则 |C_i|≤M_max L_max，进而 |B_i−λC_i|≤L_max(1+|λ|M_max)。这是使用有界相对收益界的一条可执行途径。它约束的是预测器差异，不是截断真实市场标签；控制取值范围扩大也会反过来放大置信半径。

该上界为本轮直接代数观察，并不消除时间依赖、未来漂移、估计 b 或数值积分误差。不能因为真实价格的支持问题被处理，就称整个金融过程获得分布无关保证。

### 8.4 必须加入一个理论上零收益的合成反例

取 h=1，真实收益条件对称，F0、F1 同中心且各自对称，仅尺度不同。此时 Δ 关于该中心为奇函数，而 B_i 作为真实收益 Y_i 的函数为偶函数。任何第一步仅依赖起点信息的系数乘 tanh(r_1/s_0) 都是奇控制量。因此 Cov(B_i,C_i)=0：控制变量没有总体方差缩减空间，非零控制系数只会增加方差（退化情形除外）。

这一点来自本轮代数审查，建议作为实现正确性与选择偏差的负对照。若该设置出现持续的样本外“大幅降方差”，应首先查控制系数拟合泄漏、收益符号、样本选择或评价代码。多步时系数可依赖已实现前缀，控制量不必全局为奇函数；例如两步收益乘积可以是条件零均值且全局为偶函数，故该反例不否决所有 h≥2 控制。它只排除了“一步奇对称控制自动改善纯尺度 CRPS 更新”的宽泛主张。

本补记只完善文献与数学开发边界，没有训练真实市场模型、修改旧实验或写入 GitHub。

## 9. 进入开发的唯一具体算子：条件反射控制与路径幅度预算

以下记录根代理在进入实现阶段明确的新定义，取代第 4 节中尚未具体化的 tanh 控制举例。它是本项目的待检验构造，尚未确证 original，也未因此推翻第 8 节的直接近邻限制。本轮没有再提出其他候选。

F0、F1 在此前拟合块冻结。当前实现拟用零中心 Gaussian scale mixture；F1 对 F0 的成分尺度做同一神经因子变换，因子属于 [0.5,1.5]，初始化为 1。主概率目标为按历史尺度归一化的对数价格增量。对数分布的有限 Gaussian mixture 映射到正价格后具有可计算且有限的价格均值；这不代表概率改进自动构成价格 MSE 改进。

对已经定义的观测 CRPS 收益函数 B(y)，用一个事先冻结的参考续接分布计算

G_j(s) = E_ref[B(s + 第 j 步之后的剩余噪声)]，

M_j = 0.5 × [G_j(prefix_(j−1)+r_j) − G_j(prefix_(j−1)−r_j)]。

参考分布只负责构造相关控制量；它不必等于真实续接分布，不能把参考期望当成真实过程的条件期望。对当前增量 r_j 反射时，过去前缀、G_j 及参考规则保持固定，所以 M_j 是当前 r_j 的奇函数；它没有同时翻转整条未来路径的符号，真实过程可以有 leverage。

给定事先选定的 K≥0，令剩余预算 K_j=K−Σ_(i<j)|Mclip_i|，再将当前 M_j 对称裁到 [−K_j,K_j]。预算仅依赖已实现前缀，故奇性保留，且逐条路径有 Σ_j|Mclip_j|≤K。最终控制为 C=Σ_j Mclip_j。

使用 |E C|≤δK 需要下列明确条件：对每步过去信息及当前非零幅度 R_j=|r_j|，正负号概率的差满足

|P(r_j>0 | past,R_j) − P(r_j<0 | past,R_j)| ≤ δ。

因为在固定过去和幅度时，Mclip_j 的两个符号取值互为相反数，可得 |E[Mclip_j | past,R_j]|≤δ|Mclip_j(R_j)|，再用路径预算推出 |E C|≤δK。零幅度的控制为 0。

这是较强的“给定幅度后的条件符号偏离”约束。它不等同于任意不对称性指标，不等同于未经定义的 TV 距离，也不能仅凭全样本偏度、无条件正负比例或估计残差均值来认证。若弱对称预算来自历史估计，仍须解释其对目标未来时期的适用性与估计误差；本轮未提供这种市场分布保证。

本构造仍须通过：严格条件对称下的控制效率、已知 δ 与估计 δ 的差距、预算截断的代价、漂移反例，以及第 8.4 节的一步纯尺度零收益负对照。只在 oracle 条件成立或只因保守混合变好的结果，不足以支持该学习机制的贡献。

## 10. CRPS Learning 强基线的本轮实现边界

本轮额外核读了 Berrisch–Ziel 接受稿的第 4–5 节与算法 1、Wintenberger 原始 BOA 图 1–2，以及 profoc 作者包固定版本 b5073ca70aea60ce209676faa99e04da725ffe04 的更新和损失代码。独立实现与成熟标签协议见同目录 ONLINE_BASELINES_ZH.md；代码位于 source/online_baselines.py，合成审计位于 audits/online_baselines_review.json。

区分三种对照：普通历史 CRPS 最小化的 CDF 混合；将 BOA 二阶梯度更新用于同样两个 CDF 的纵向混合；CRPS Learning 的无样条逐分位横向聚合。后者额外使用逐分位权重，并在本项目中适配为固定 19 个等权原子，不是完整 profoc 复现。采用 τ_j=(j−0.5)/19 时，其重排后的原子分布 CRPS 等于 2/19 倍相应 pinball 损失之和；价格变换后亦可直接计算该离散分布的精确 CRPS、均值与覆盖。19 原子会限制尾部表示，不能称为原连续专家分布的精确 CRPS。

冻结 BOA 与部署期继续接收成熟反馈的 BOA 应分别命名。后者使用更多目标时期信息，其优势不能全部归于更新算子；与主候选同一信息预算的冻结版本仍是必要对照。文献与代码代理仅核验合成数据与接口，本节不含真实市场实验结果。
