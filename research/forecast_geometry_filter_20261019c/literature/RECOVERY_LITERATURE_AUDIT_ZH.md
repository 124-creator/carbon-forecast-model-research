# 恢复执行后的独立文献补充审查

审查日期：2026-10-19。对象：`forecast_geometry_filter_20261019c`，冻结合成协议 v1.1。本文只审查方法近邻、文献真实性与可主张的创新边界；没有读取市场价格、保留确认标签或市场预测，没有更改协议、源码及主报告，也没有执行新的性能试验。

本次先完整阅读本目录的 `DIRECT_NEIGHBORS_ZH.md`、`PRIOR_MECHANISM_MAP_ZH.md`、`../methods/GEOMETRY_DERIVATION_ZH.md` 和 `../provenance/SYNTHETIC_PROTOCOL.json`。下文的文献范围和阅读程度均指本次独立核查，不继承旧文档“已读全文”的声明。

## 1. 独立结论

**目前只能把候选定位为既有 proper-score／变度量近端过滤中的一个未来目标度量选择，不能认证为已达到发表要求的模型创新。** 原清单中八篇文献的题名、作者、主要日期和版本均能由出版社、作者机构或 arXiv 原始页面核实；本次没有发现所列文献是虚构文献。

但原八篇清单并不构成充分的“最直接方法近邻”覆盖。至少还需加入三个方向：已正式发表的 **NGBoost 的 CRPS 诱导度量公式**、已正式发表的 **Robust CDF-Filtering**，以及 **Predictively Oriented（PrO）后验的先混合再评分目标**。其中 NGBoost 和 PrO 对候选的数学归类，比两篇碳价动态图论文更直接。[R9–R12]

本次没有找到与“固定四状态＋当前混合 CDF 的 CRPS＋未来 H5/H20 累计 CDF 距离＋固定 χ² 项＋受约束递归更新”逐项相同的原始算法。因此，不能声称已证明整个算法完全重复，也不能反向声称“此前无人提出”。**可保留的研究问题是当前观测损失与未来预测目标使用不同几何是否带来可重复增量；拉回 Gram、proper score、近端求解以及稳健过滤的基础原理均不属于本轮原创。**

这里不采用“只有全新数学理论才能发表”的门槛：成熟框架内的具体方法设计可以有论文价值，但需要清楚的非冗余设计、可解释的适用条件及匹配比较。合成门槛是本项目的资源分配规则，不能写成某个期刊的官方录用标准。本文不根据尚未审阅的本轮结果判定实验通过，也不允许文献补充改变已冻结的实验裁决。

## 2. 原八篇文献的真实性与阅读范围

| 原编号 | 本次核实的状态 | 本次实读程度与限制 |
|---|---|---|
| 1．TLCC-DyG-STFusion 碳价论文 | Springer 原始页面确认题名与四位作者；2026-10-13 在线发布。页面说明为已接受研究的早期版本，后续会被正式记录版本替换。[R1] | 读取出版社摘要、作者及版本说明。本次没有重新阅读其 29 页 PDF，不继承旧稿的全文核查结论。其时滞图和时空骨干属于应用近邻，不能证明本轮过滤几何的新颖性。 |
| 2．Lead–lag Aware 碳价论文 | Springer 页面确认题名、六位作者与 2026-06-30 日期。[R2] | 读取摘要、元数据、数据可得性与访问状态；页面明确是订阅内容预览。未取得完整正文，不能评价其全部实验、消融或防泄漏实现。 |
| 3．SDD | arXiv 确认 Biswas 与 El Karoui；2026-09-09 v1、2026-09-14 v2。[R3] | 读取版本记录及 v2 方法、梯度协方差命题与实验描述。其条件期望损失用于合成训练目标；不等于本轮限制当前权重更新。未复现其大模型实验。 |
| 4．Proper-score observation-driven filters | arXiv 确认 Livieri 与 Palmari；2026-08-03 v1、2026-09-21 v3。[R4] | 读取 Definition 2.4、§3 局部度量及 Appendix H 的向量局部跟踪条件。不能把标量主体定理或附录局部极限直接当作本轮全轨迹保证。 |
| 5．PRADA | Tinbergen 原始 PDF 确认 de Punder、TI 2026-022/III 与 2026-05-15 稿件日期。[R5] | 读取原稿摘要、§3.1 Equation (2)、相关学习率矩阵论述及参考文献。没有逐条复核 83 页的全部证明。 |
| 6．ISD | arXiv 确认 2025-12-02 v1、2026-04-20 v3；作者机构确认 Journal of Econometrics 255、106251、2026-05 发表及 DOI。[R6] | 读取作者 v3 的更新定义、加权二次项、参数约束与理论假设；期刊元数据通过机构记录核验。未重新取得出版社版全部正文。 |
| 7．PSEF | arXiv 确认五位作者、2026-06-25 v1、2026-10-05 v2；记录仍为 Submitted to JMLR。[R7] | 读取 v2 摘要、§3.1 Propositions 7–8 附近的总体目标与可实现性说明。没有把投稿状态写成录用，也没有复现有限粒子网络。 |
| 8．Newton-Flow Particle Filters | arXiv 确认 Hanebeck、2025-08-29 v1、2025-11-23 v2。[R8] | 读取 v2 版本说明及方法结构。v2 的 Note 仍说明实现细节、现有方法比较及评价将留待下一版本；其摘要效率表述不能作为已核实的对照实验成绩。 |

更新文献描述时应保留上述访问边界。“页面有全文链接”与“本次完整审完全文”是不同事实；预印本、投稿中稿件、早期已接受版本和正式期刊论文也应分别记录。

## 3. 最重要的补充：未来 CDF Gram 也是已有 CRPS 诱导度量的直接实例

NGBoost 的 ICML 2020 官方论文 §3.2、Equation (11) 已给出 CRPS 对参数化 CDF 诱导的度量：

\[
I_C(\theta)=2\int \nabla_\theta F_\theta(z)\nabla_\theta F_\theta(z)^\top dz.
\]

同节用该度量定义广义自然梯度。论文把公式来源追溯到 Dawid 的 proper-scoring-rule 几何；后者的原始论文也可以由期刊所属研究所取得。[R9,R10]

以下是对冻结候选的独立代数对应，不是从文献取得的新定理。令

\[
T=[e_1-e_K,\ldots,e_{K-1}-e_K],\quad q=q_\star+T\theta,
\quad \boldsymbol F_h(z)=(F_{h,1}(z),\ldots,F_{h,K}(z))^\top .
\]

因为工作模型固定，\(F_{h,q}=q^\top\boldsymbol F_h\)，所以

\[
\nabla_\theta F_{h,q}(z)=T^\top\boldsymbol F_h(z),\qquad
I_H=2\sum_{h\in\{5,20\}}\alpha_h
\int (T^\top\boldsymbol F_h)(T^\top\boldsymbol F_h)^\top dz
=2T^\top G_HT.
\]

这里的 \(G_H\) 是推导稿的全状态中心化矩阵；常数 2 不改变其方向。切空间中的 CDF 对比在两端相消，因此避免把各个未中心化 CDF 本身误当成平方可积函数。多期限加权和对应指定边际分数之和；它并不严格识别完整路径联合分布。

这一对应带来三个明确边界：

1. “首次用 CRPS 定义参数空间几何”“首次把 CDF 导数积分成 Gram”“首次使用分布距离预条件”均不可主张。
2. NGBoost 的实际算法是监督概率回归中的 boosting，并没有因此成为本候选的逐项等价递归实现。本次也没有将 NGBoost 外部论文分数拿来当本轮合成对照成绩。
3. 尚有辨别力的是**用未来累计预测边际的几何，约束当前观测分数的更新**。这与在同一目标上使用普通 natural gradient 不同；但这种目标选择要靠机制证据与匹配比较成立，不能靠一个新名称成立。

固定模型和固定期限时，\(G_H\) 不随本轮观测改变；真正随先验变化的部分主要来自 \(\operatorname{diag}(1/p)\) 的 χ² 项。不能将这一固定传播矩阵描述成由当前异常冲击在线学出的“自适应未来几何”。如果未来另行学习该矩阵，那是另一个需要重新定义、冻结和审查的方法。

## 4. PRADA 与 ISD 的涵盖范围：内部解的确切对应，不等于继承全部定理

PRADA §3.1 Equation (2) 明确允许在观测前可测的正定学习率矩阵乘 scoring-rule 梯度。[R5] 本轮当前混合 CRPS 在切空间是二次函数。令其梯度和 Hessian 分别为 \(g_y\)、\(H_0\)，当解处于单纯形内部时，候选可精确写成

\[
\theta^+-\theta_p
=-\left[H_0+\frac{2}{\eta}
\{T^\top\widetilde G_HT+\kappa T^\top\operatorname{diag}(1/p)T\}\right]^{-1}g_y.
\]

方括号中的矩阵在当前观测前已知。因此，在该内部解条件下，这不仅是“小学习率时近似 PRADA”，而是可预测矩阵预条件更新的精确实例。相应的常数 Hessian 来自固定混合族的二次分数，不来自新优化理论。

不过，正式实现允许单纯形边界解。此时活动约束由当前观测决定，不能把完整 QP 轨迹都写成一个不受约束、矩阵仍然可预测的相同梯度公式。PRADA 的小步局部改善结论、Proper-score filters 的向量局部极限，以及 ISD 在其假设下的收缩结论，都不能未经核验就转移给这条边界递归。[R4–R6]

ISD 的 §2 Equations (1)–(2) 已包含当前观测准则、预测参数、正定二次惩罚和约束参数域。[R6] 本轮把对数密度准则换成当前 CRPS，并指定未来目标矩阵。二次近端、隐式更新、严格凸 QP 的唯一解与活动集求解均是实现资格，不是独立方法贡献；在保留这些常规构件的前提下，未来目标矩阵是否有价值仍可实验检验。

## 5. 另外两条应补入的直接近邻

### 5.1 当前 CDF 的稳健过滤已有正式论文

Catania、Harvey、Luati 的 **Robust CDF-Filtering of a Location Parameter** 已由 Wiley 发表，出版社写明首次在线日期为 **2025-11-02**；当前页面的期刊年份标签为 2026，二者不能混为首发日期。[R11]

本次读取其摘要、§2 模型与讨论：它用当前残差 CDF 的变换构造位置参数更新，具有抑制极端残差的作用。本轮更新对象是有限状态混合权重，并用未来累计分布距离作惩罚，不能说二者算法相同。但是“CDF 过滤”“概率评分导出的稳健观测更新”“对异常值反应有界”不能脱离这类文献称新。

它不是要求现在新增一个不匹配的位置模型、改动冻结合成比较。现有 `current_cdf`、`crps_kl`、污染感知 Bayes 等控制继续按原协议使用；这里补的是文献归因和后续主张范围。

### 5.2 “先混合、再评分、再加散度”已有 PrO 框架

**Predictively Oriented Posteriors** 于 2025-10-02 首发，arXiv 当前 v3 为 2026-07-17。其 §2 直接在模型平均后的预测分布上施加 proper score，并用 KL 约束参数分布偏离参考分布；正文清楚区分“平均各分量分数”与“对平均分布评分”。[R12]

因此，本轮原生完整混合 CDF 的 `CRPS + KL(q||p)` 是必要基线。它与按单个分量 CRPS 产生 Gibbs 权重的目标确实不同，但这种区别本身已经有明确方法文献，不能作为本项目的新发现。令参数集合为有限状态、样本量为 1、参考分布为当前先验，可以得到相应的单次更新目标；将它递归应用并不自动继承该论文的批量统计结论。

Javeed 等的 **Duality and Error for Predictively Oriented Inference** 于 2026-08-11 首发，进一步研究对数分数配一般 φ 散度的对偶与误差分解。本次读取原始摘要、§1–2 与目标定义；没有把其对数分数理论直接应用到本轮 CRPS，也没有复核其全部证明。[R13]

该框架与本轮仍有差别：候选额外使用跨状态耦合的未来预测 Gram；一般的参数概率 φ 散度并没有自动给出这个矩阵。但是，“χ² 可以替换 KL”“混合权重不等于真实后验”“预测好并不保证潜在状态被识别”等基本区别也不能再包装为原创。

## 6. 2026 年新增过滤文献对主张的进一步限制

检索还确认两篇本轮原八篇表中未列的 2026 预印本。它们只作为范围核查，不作为已经复现的基线：

| 文献 | 已核范围 | 对本轮的意义 |
|---|---|---|
| Natural Gradient Bayesian Filtering: Geometry-Aware Filter for Dynamical Systems，2026-05-04 | arXiv 原始题名、日期与摘要。[R14] | 高斯族上的自然梯度、协方差正定和 Kalman 联系已有专门工作。仅把“几何感知”加到过滤器名称上不能确立贡献；本轮有限状态、CRPS 和未来目标与其仍不同。 |
| Robust ensemble Kalman filtering under observation noise misspecification via diffusion score matching，2026-05-26 | arXiv 原始题名、日期与摘要。[R15] | 异常观测稳健性与动态稳定性之间的取舍已是明确研究对象。该文用 diffusion score matching 与 Kalman/ensemble 方法，并非本轮完整混合 CRPS 或未来 CDF QP；没有依据从摘要认定算法等价。 |

本次通过两类检索引擎发现线索，再回到作者、出版社和正式会议论文来源验证。搜索结果中的聚合网站只用于定位，没有用于支撑方法结论。检索截止日为 2026-10-19；这是一轮有针对性的近邻补充，不是“穷尽全部研究并证明无人做过”的检索。

## 7. 在冻结实验中，什么证据真正能支持剩余差别

以下为对已有协议的解释，不新增门槛或参数：

- `current_cdf` 隔离了当前分数曲率或一般 CDF 距离的解释；若它已给出同样效果，则未来传播未提供需要的增量。
- `future_variance` 隔离了只改变未来二阶矩方向的解释；全 CDF 胜出才可能支持超出这些预列矩特征的作用。
- `rotated_future_cdf` 保持未来矩阵的谱与协议中的尺度，改变其方向。胜过它有助于证明方向相关，但一个固定反射不能代表所有可能方向，也不能证明原方向全局最优。
- `crps_kl` 与 χ²、欧氏控制隔离了稳健分数与普通收缩；不能只与未调节的 Gaussian Bayes 比较后就把全部改善归给未来几何。
- 干净正确 Bayes 是信息集一致时的总体分数最优参照；真实状态切换与转移错设诊断用于揭示过度保护未来预测可能造成的延迟，不能只保留有利污染场景。
- 当前发射相同的两个状态没有新的观测似然比。若候选改变它们的赔率，应解释为预测决策正则，而非发现了隐藏状态；比例保持消融用于辨别这项自由度的作用。

如果候选未通过原协议，应据此关闭这个固定未来几何候选。补到更多文献、把它改称 natural gradient、增加网络、修改期限或改变噪声强度，都不能把同一轮失败改写成通过。本文没有提出或启动另一轮试分。

后续若寻找新方案，现阶段的证据缺口是：**必须先说明现有当前分数、标准过滤与预测几何都不能解决的具体可观测问题，并展示新算子相对这些基线的不可约差别。** 已有失败地图提到的信息可得性或真实供体价格问题，仍应保持“未完成的资格检查”状态；本文未据此宣布发现模型创新，也未访问这些市场数据。

## 8. 一手来源与访问记录

所有链接访问日期均为 2026-10-19。下面仅保存书目信息和本人的审查说明，不复制或重新分发论文正文。

- **[R1]** Wang, Qiao, Liu, Wang. *Carbon price forecasting based on time-lagged cross-correlation dynamic graphs and a dual-stream spatiotemporal fusion network*. [Springer 原始页面](https://link.springer.com/article/10.1007/s40747-026-02550-7)。本次：摘要、元数据与早期版本说明。
- **[R2]** Ren et al. *Lead–lag Aware Carbon Price Forecasting with Dynamic Graphs and Uncertainty Intervals*. [Springer 原始页面](https://link.springer.com/article/10.1007/s10666-026-10146-1)。本次：摘要、元数据、数据可得性；订阅全文未取得。
- **[R3]** Biswas, El Karoui. *Distillation of Synthetic Data for Time Series Foundation Models*. [版本记录](https://arxiv.org/abs/2609.09586)；[v2 方法与命题](https://arxiv.org/html/2609.09586v2)。本次：相关正文；未复现。
- **[R4]** Livieri, Palmari. *Proper-score observation-driven filters: local geometry, estimation, and continuous-time limits*. [版本记录](https://arxiv.org/abs/2608.02828)；[v3 正文](https://arxiv.org/html/2608.02828v3)。本次：Definition 2.4、§3、Appendix H；非全稿证明复核。
- **[R5]** de Punder. *Proper and Robust Autoregressive Derivative Adaptive Models*. [Tinbergen 原始 PDF](https://papers.tinbergen.nl/26022.pdf)。本次：§3.1、Equation (2)、学习率矩阵段落与相关参考文献。
- **[R6]** Lange, van Os, van Dijk. *Implicit score-driven filters for time-varying parameter models*. [arXiv 版本记录](https://arxiv.org/abs/2512.02744)；[v3 §2](https://arxiv.org/html/2512.02744v3)；[作者机构期刊记录](https://research.vu.nl/en/publications/implicit-score-driven-filters-for-time-varying-parameter-models-2/)。DOI：`10.1016/j.jeconom.2026.106251`。
- **[R7]** Bach, Baptista, Bröcker, Chen, Stuart. *Learning Probabilistic Filters with Strictly Proper Scoring Rules*. [版本及投稿记录](https://arxiv.org/abs/2606.26497)；[v2 正文](https://arxiv.org/html/2606.26497v2)。本次：§3.1 总体结论；未复现。
- **[R8]** Hanebeck. *Newton-Flow Particle Filters based on Generalized Cramér Distance*. [版本记录](https://arxiv.org/abs/2509.00182)；[v2 正文与 Note](https://arxiv.org/html/2509.00182v2)。本次：版本限制和方法结构；无已核实比较成绩。
- **[R9]** Duan et al. *NGBoost: Natural Gradient Boosting for Probabilistic Prediction*. ICML 2020, PMLR 119:2690–2700。[正式会议记录](https://proceedings.mlr.press/v119/duan20a.html)；[官方 PDF，§3.2、Equations (5)–(11)](https://proceedings.mlr.press/v119/duan20a/duan20a.pdf)。本次：这些公式与上下文；未复现其监督回归实验。
- **[R10]** Dawid. *The geometry of proper scoring rules*. Annals of the Institute of Statistical Mathematics 59 (2007), 77–93；在线日期 2006-12-16，DOI `10.1007/s10463-006-0099-8`。[期刊所属研究所原始 PDF](https://www.ism.ac.jp/editsec/aism/pdf/059_1_0077.pdf)；[作者书目](https://www.statslab.cam.ac.uk/~apd25/research.html)。本次：来源与框架核实；CRPS 的具体张量公式直接以 [R9] 为证据。
- **[R11]** Catania, Harvey, Luati. *Robust CDF-Filtering of a Location Parameter*. [Wiley 正文](https://onlinelibrary.wiley.com/doi/10.1111/jtsa.70026)。本次：在线日期、摘要、§2 及讨论；未逐条复核其渐近证明。
- **[R12]** McLatchie, Chérief-Abdellatif, Frazier, Knoblauch. *Predictively Oriented Posteriors*. [版本记录](https://arxiv.org/abs/2510.01915)；[v3 正文](https://arxiv.org/html/2510.01915v3)。本次：§2、总体目标及 kernel-score 分解附近内容；非全稿证明复核。
- **[R13]** Javeed, Kouri, Portone, White. *Duality and Error for Predictively Oriented Inference*. [版本记录](https://arxiv.org/abs/2608.10592)；[v1 正文](https://arxiv.org/html/2608.10592v1)。本次：摘要、§1–2 与问题定义；未复现数值例子。
- **[R14]** Liu et al. [Natural Gradient Bayesian Filtering: Geometry-Aware Filter for Dynamical Systems](https://arxiv.org/abs/2605.02306)。本次仅原始摘要与元数据，不据此判断完整理论或比较实验。
- **[R15]** Reimann, Reich. [Robust ensemble Kalman filtering under observation noise misspecification via diffusion score matching](https://arxiv.org/abs/2605.26881)。本次仅原始摘要与元数据，不据此判断完整理论或比较实验。

**审查处置：**保留这份补充，维持原 v1.1 协议及停止条件。将“未来目标度量的剩余机制问题”与“已被文献覆盖的数学构件”分开表述；本次不授予原创性、市场有效性或投稿就绪认证。
