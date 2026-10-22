# Forest transport round5：聚焦最近邻与可辩护贡献边界

核查日期：2026-10-21。范围限定为四篇与当前算子最接近的真实文献，按结构相似度选择，并非按发表年份排序。本次只读文献，不训练、不修改旧源码或实验结果。

**判断：不能把指数重加权、矩匹配 affine 变换或“基线分布＋神经修正”分别声称为首创。最需要正面比较的架构邻居是 Distributional Refinement Network（DRN）；森林条件经验权重的邻居是 Distributional Random Forests（DRF）。当前组合仍有具体可检验的结构差异：显式零分支、带 uniform reservoir 的森林条件非零场景参考，以及由序列网络驱动、将输出矩与形状校正分开的经验支持输运。是否形成有说服力的方法贡献，取决于这些结构在匹配对照中的增量，而不是组件名称是否全新。**

## 1. 四个最近邻：已核原文内容与差异

| 文献 | 本次实际核到的内容 | 与当前模型的直接关系 | 当前不能据此声称的内容 |
|---|---|---|---|
| **Meucci (2008；2010 arXiv版), Fully Flexible Views: Theory and Practice** | 原文§4、式(26)定义离散相对熵，式(27)在观点约束下最小化相对先验的KL；保留情景、更新情景概率。 | `w ∝ u exp(β·φ)`属于既有指数倾斜／相对熵更新；非均匀参考u也不改变这一事实。 | 不能声称首创情景概率指数修正或KL更新；本模型也没有因此自动成为Bayesian posterior。 |
| **Gelbrich (1990), On a Formula for the L² Wasserstein Metric between Measures on Euclidean and Hilbert Spaces** | 核到原刊题名、作者、1990年147(1):185–203、DOI与摘要。原刊全文本次未取得；摘要讨论欧氏／Hilbert空间及椭圆族、Gaussian情形。下面单变量界与达界条件独立推导，不冒引未读的定理编号。 | 当前mark变换与经典均值—协方差Wasserstein几何相邻；一维给定均值/方差的最小W2投影可直接由Cauchy–Schwarz证明。 | 不能把该一维已知几何结果声称首创，更不能外推为含固定零atom的完整混合分布全局OT最优。 |
| **Ćevid, Michel, Näf, Bühlmann & Meinshausen (2022), Distributional Random Forests: Heterogeneity Adjustment and Multivariate Distributional Regression** | JMLR正式页面与开放全文均核到。摘要及方法强调forest诱导的kernel-type训练样本权重，并以MMD度量学习分布异质性，权重估计整个条件分布。 | `u_forest(x)`是已有的forest条件经验测度思想；历史标量响应加权构成预测分布也已有先例。 | 当前reference forest沿用均方误差split的QRF，不是DRF的MMD split复现；不能声称首创forest conditional distribution。 |
| **Avanzi, Eric Dong, Laub & Wong (2024), Distributional Refinement Network: Distributional Forecasting via Deep Learning** | 实际读到arXiv v1全文§4.1–4.2，尤其式(14)–(19)、KL正则讨论。式(17)把baseline区间质量与神经logits相加后做softmax。文中实例baseline为GLM，区间内为piecewise-constant密度，区间外恢复baseline。 | **最接近的神经分布修正架构。**其质量更新本质为 `b_k(x) exp(l_k(x)) / Σ_j b_j(x) exp(l_j(x))`，与当前指数修正高度接近。 | 不能以“先有条件基线、再用NN修正整个分布”作为首创点；仅把GLM换QRF、把MLP换LSTM，单独作为主创新说服力有限。 |

对应实际来源：

1. Attilio Meucci, **Fully Flexible Views: Theory and Practice**, *Risk* 21(10), 97–102, 2008。2010-12-13开放版本的首页脚注确认原刊信息；[原文](https://arxiv.org/html/1012.2848)。
2. Matthias Gelbrich, **On a Formula for the L² Wasserstein Metric between Measures on Euclidean and Hilbert Spaces**, *Mathematische Nachrichten* 147(1), 185–203, 1990，DOI **10.1002/mana.19901470121**。[原刊标识](https://doi.org/10.1002/mana.19901470121)；[实际取得的Crossref原刊元数据与摘要](https://api.crossref.org/works/10.1002/mana.19901470121)。1990为纸本发表年份，2010是在线归档日期，不应混写。
3. Domagoj Ćevid, Loris Michel, Jeffrey Näf, Peter Bühlmann, Nicolai Meinshausen, **Distributional Random Forests: Heterogeneity Adjustment and Multivariate Distributional Regression**, *JMLR* 23(333):1–79, 2022。[正式页面](https://jmlr.org/papers/v23/21-0585.html)；[开放全文](https://arxiv.org/html/2005.14458v3)。
4. Benjamin Avanzi, Eric Dong, Patrick J. Laub, Bernard Wong, **Distributional Refinement Network: Distributional Forecasting via Deep Learning**, arXiv:2406.00998v1，2024-06-03。[版本信息](https://arxiv.org/abs/2406.00998)；[实际核读全文](https://arxiv.org/html/2406.00998)。检索也返回2026年期刊条目，但本报告公式与结构判断以实际读到的v1为准，不根据未取得的后续全文推断改动。

## 2. 指数倾斜：可以怎样准确表述

对某个query，设参考权重u已归一、在当前成熟非零bank上严格为正；未成熟支持不在本式定义域内。令

\[
 w_i(\beta)=\frac{u_i\exp(\beta^\top\phi_i)}{\sum_j u_j\exp(\beta^\top\phi_j)}.
\]

它满足标准变分刻画

\[
 w(\beta)=\arg\min_{r\in\Delta}
 \left\{D_{\mathrm{KL}}(r\|u)-\beta^\top\sum_i r_i\phi_i\right\}.
\]

对r写Lagrangian并求导即可得到该指数族形式。这是已知熵正则更新，与Meucci的相对先验情景更新同属一类思想。若把约束矩取为 `Σ w_i φ_i`，也能得到相应KL投影解释。

**当前实际训练的区别**：β由序列网络和预测损失学习，不是在每个query上接收外部观点约束后求解entropy-pooling优化。因此可称“有界的、神经参数化的指数倾斜”，但不能把它写成已经求解了外部mean/variance观点约束。当前φ2为有符号位置／绝对幅度响应，其特征矩也不等同于后续目标mean/variance。模型若没有显式训练KL penalty，也不应声称损失中包含该penalty；上式是输出参数化的等价变分解释。

DRN的接近程度尤其高：其式(17)的区间质量是

\[
 r_k=C(x)\,\operatorname{softmax}_k\{\log b_j(x)+l_j(x)\},
\]

其中C为修正区间的总质量。本模型在离散NZ情景上把logit限制为 `β·φ2(z)`，再移动标量支持。**低维logit、经验情景、显式atom和支持移动是差异；乘baseline质量再指数归一这一运算不是差异。**

## 3. affine标记分支：确切的最小W2结论

令倾斜后的mark分布为 `P_w=Σ_i w_i δ_{z_i}`，其均值、标准差为m_w、s_w。假设二阶矩有限且 `s_w>0, s_t>0`。在所有均值m_t、标准差s_t的概率分布Q中，对任意耦合(Z,Y)，

\[
\begin{aligned}
\mathbb E(Y-Z)^2
 &= (m_t-m_w)^2+s_t^2+s_w^2-2\operatorname{Cov}(Z,Y)\\
 &\ge (m_t-m_w)^2+(s_t-s_w)^2.
\end{aligned}
\]

不等式只使用Cauchy–Schwarz。取

\[
 T(z)=m_t+\frac{s_t}{s_w}(z-m_w),\qquad Q^*=T_\#P_w,
\]

则均值和标准差满足要求、协方差达到s_ws_t，且a=s_t/s_w>0。因此

\[
 \min_{Q:\,\mathbb E Q=m_t,\,\operatorname{sd}(Q)=s_t}
 W_2^2(P_w,Q)
 =(m_t-m_w)^2+(s_t-s_w)^2.
\]

这对有限加权经验分布同样成立，不要求Gaussian。它可以作为当前mark算子的清楚几何解释；不应包装成新的普遍投影定理。若数值variance floor被触发，精确等式须以实际采用的分母／方差为准，不能把退化情形照写成精确矩匹配。

### 固定零atom的完整混合分布不能照搬此最优性

完整模型是

\[
 F=(1-p)\delta_0+p\sum_i w_i\delta_{T(z_i)}.
\]

固定zero分支、只运输mark的耦合，提供的是完整混合W2的一个**可行耦合／代价上界**，不是自动得到其最优耦合或全局约束投影。完整OT可以把原zero质量与目标mark配对，或把原mark与目标zero配对。

一个直接反例可把边界说清：令p=1/2、`P_w=(δ_{-2}+δ_{-1})/2`，目标mark均值1.5、标准差0.5。mark最小W2变换是 `T(z)=z+3`，得到 `Q*=(δ_1+δ_2)/2`。

- mark分支最小W2²为9；保持分支身份的完整耦合代价为4.5。
- 但含0.5零atom的两个完整混合分布，其一维最优W2²只有2.5。
- 进一步，另一个mark分布 `Q'=(1/17)δ_{-1/2}+(16/17)δ_{13/8}` 也有均值1.5、方差0.25，且无额外零atom。它与相同固定0.5零atom组成的完整目标分布，距原完整混合的W2²为 `83/34≈2.44118`，小于2.5。

所以mark分支的Q*甚至一般不能当成**固定零atom、给定mark均值方差的完整混合全局投影**。论文应明确写“mark-branch moment-constrained Wasserstein projection”。

另一个表述边界：p是分支混合质量。若输运后某些mark恰好等于0，则完整分布的零质量为

\[
 (1-p)+p\sum_{i:T(z_i)=0}w_i,
\]

实际非零概率不能直接等同于p。

## 4. forest条件参考＋神经refinement：具体还剩什么可主张

在本次实际核读的四篇中，DRF提供条件经验权重的已有基础，DRN提供baseline distribution神经残差修正的直接邻居。**本次没有核到与当前全部结构完全相同的已发表实现，但这不是穷尽查新，也不能据此声称“首个forest-conditioned neural refinement”。**

当前较可辩护的技术贡献应落在以下共同设计，而不是逐件列首创：

1. **针对有符号、含零收益的碳价分布设置参考测度**：森林给出query条件NZ情景参考，独立保留零分支；maturity约束只允许检索已成熟响应。森林/scaler/phi按允许fit区间估计，不把这写成训练query全参数vintage。
2. **有尾部保留约束的残差形状更新**：`u=(1−γ)u_forest+γu_uniform`，再以低维有界β修正。它保留森林局部性，同时避免稀疏leaf参考丢掉全部历史尾部。reservoir/污染混合本身已知，贡献应说明它为何对这一经验输运分母必要。
3. **以参考分布为尺度锚的序列残差矩控制**：`m_t=m_0+s_0δ`、`s_t=s_0 exp(ell)`，并在倾斜后用m_w、s_w重新标准化。固定u和位置／尺度头输出时，改变β不改变最终mark的目标均值／方差；这是算子层面的矩职责分离，不等于统计意义的参数正交或可识别性定理。
4. **保留原经验支持的信息而允许支持位置移动**：不同于已核DRN的固定区间piecewise-constant密度修正，本模型使用原始signed scalar场景及单调affine支持移动，配合完整atom＋经验mixture的精确CRPS。LSTM用过去顺序信息生成低维残差，u直接进入分布算子，不只是把QRF预测和LSTM预测作最终加权平均。

稳定性设计可以给出一个简短、非首创性质：在当前支持上 `u_i≥γ/N`，若两维φ均在[-3,3]、β各在[-.35,.35]，则指数能量在[-2.1,2.1]，有

\[
 w_i\ge e^{-4.2}u_i\ge\gamma e^{-4.2}/N,
 \qquad \operatorname{Var}_w(Z)\ge\gamma e^{-4.2}\operatorname{Var}_{uniform}(Z).
\]

条件是uniform支持方差正且不处于数值退化情形。这适合作为reservoir与bounded tilt的设计依据，不是新增通用不等式或全局OT定理。

## 5. 建议采用的方法贡献措辞与最小证据

可用如下表述作为方法段落的起点：

> 面向含零收益的碳价格多期限预测，构建一种条件森林经验参考上的序列残差输运模型。模型在成熟非零历史情景上执行有界指数校正，以保留uniform reservoir的森林参考为锚预测目标矩，并通过mark分支的矩匹配单调仿射变换分离形状与位置尺度修正，最终形成显式零分支与经验mark的完整预测分布。

紧随其后应正面说明：指数倾斜、forest加权经验分布、baseline神经refinement与单变量矩约束W2几何均有先例。具体新意是它们在该问题下的**参考约束、序列条件化和矩归一残差结构**，而非新的通用分布族。

需要直接支撑该方法主张的匹配证据，就是正在进行的 `forest_reference_only / forest_transport / forest_location_scale_only / forest_reweight_only`，外加相同条件森林参考下的强baseline；它们分别回答reservoir参考本身、affine矩控制和形状校正贡献。若进一步要区分于DRN式refinement，最有针对性的邻居对照是同一QRF参考上的分箱神经质量残差，不能只对照弱uniform经验分布。这里没有新增训练，也没有建议按测试市场选择组件。

已有通用组件并不否定具体方法组合；同样，组件组合本身也不保证达到可发表的新颖性。当前应把论证集中在这一可复现结构为何在碳价零收益、条件经验参考及时间序列残差之间解决了明确问题，并用实际增量限定主张强度。
