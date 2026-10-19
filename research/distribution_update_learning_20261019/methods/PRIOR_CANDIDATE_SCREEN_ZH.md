# 前期候选筛选：理论反例与原始文献审查

**性质：历史筛选记录。日期：2026-10-19。** 本文整理最终方法确定前的四个候选及其数学、识别与近邻审查，保留当时的第 1–8 节分析。文中的推进建议与最低对照要求属于该筛选阶段，不构成新增实验计划，也不是最终方法的执行协议。

最终方法定义与保证条件见 [METHOD_ZH.md](METHOD_ZH.md)，本轮已经完成的实验、结果及 STOP 决定见 [REPORT_ZH.md](../REPORT_ZH.md)。本文不替代这两份文档，也不将被否决候选的理论推导当作最终方法的实证证据。

原筛选阶段只查阅既有汇总报告和原始论文，没有拟合新的真实市场模型、读取新的逐点市场预测或打开封存确认集。以下反例用于限定方法主张，不表示所有相关模型或研究方向均不可能有效。

## 1. 当时的筛选裁决

当时连续审查的四个候选，结论如下。

| 候选 | 数学上可以成立的部分 | 无法据此认领的创新 / 主要问题 | 筛选裁决 |
|---|---|---|---|
| 用未来日增量平方和代替累计增量平方，训练 QLIKE | 条件鞅下有相同条件期望；某些模型下标签噪声明显较低 | 已知 realized-volatility proxy / robust loss；不是普遍降方差，MD 下也有反例 | 作为强基线，不能作为新方法核心 |
| 可预测神经鞅控制变量降低评分梯度噪声 | 真正 MD、正确可预测性与适当梯度处理下可保留期望 | Neural martingale CV 已有；与预测参数无关且 detach 的 loss CV 根本不改变训练梯度；cross-fit 不修复未知漂移偏差 | 现有表述不足，先否决创新主张 |
| 固定散布矩阵的角向 Rao–Blackwell CRPS | 正确条件椭球分布下，是同一评分/梯度的条件期望，协方差可降低 | Chen 等 2020 已给出群平均的风险/梯度理论，且专门讨论球面对称下用半径改善单坐标边际；可学习散布参与标签反传会有偏 | 普通版本否决；不能包装为新学习算子 |
| 全连续区间和的解析 Gaussian-mixture CRPS | 固定投影、有限正权重时 proper；共同尺度混合椭球族内，所有区间投影可严格识别联合分布 | Pic 等 2025 式(16)直接覆盖任意 patch 总和 CRPS 聚合；Gaussian-mixture CCRPS 神经学习已存在；最小投影和快速求和只是窄代数结果 | 可以作为可靠基线/研究起点，**尚不足以认证实质方法创新** |

最后一个候选比角向 RB 的数学风险小：它不人为改写训练目标分布，也不需要给市场路径假定未验证的旋转不变性来保证一般 propriety。但“评分正确”“族内识别”“可以解析求值”与“贡献新颖”“样本效率更高”是三个不同问题。筛选时的证据支持前者，没有解决后两者。

当时对这一支提出的待解决问题是：**在同一预测族、参数容量、训练数据和计算预算下，如何设计投影及其学习更新，使有限样本的累计价格风险预测比现有投影/条件评分更有效，而且能够证明这种改进针对的具体误差方向？** 不能先选定普通总和 CRPS，再把它的已知 proper 性作为创新答案。

## 2. 原实验失败没有留下一个空白的“概率预测方向”

下列事实来自仓库汇总文件，不是此次重算。

| 已完成研究 | 与新路线直接相关的证据 | 应有解释 |
|---|---|---|
| [conditional_distribution_probe_20261013/REPORT_ZH.md](../../conditional_distribution_probe_20261013/REPORT_ZH.md) | EMP、EWMA、条件尺度共 12 个概率配置；固定增益与条件覆盖推进条件未通过 | 简单区间/条件尺度预测已试，不可当成从未开始的新任务 |
| [distribution_shape_diagnostic_20261017c/REPORT_ZH.md](../../distribution_shape_diagnostic_20261017c/REPORT_ZH.md) | 30 个神经、10 个统计拟合；条件形状 CRPS 比值 0.994666769，全局形状 0.991848009；形状+尺度 1.017943136，尺度单独 0.994536684；五门槛仅一项通过 | 仅把均值换成分布头、固定均值后学尾形、GARCH/FHS/DeepNPTS 已有直接对照；失败只约束当时固定实现和数据 |
| [auction_clearing_innovation_20261018/REPORT_ZH.md](../../auction_clearing_innovation_20261018/REPORT_ZH.md) 与前史审计 | FDMC MSE/RW 1.004985，原推进门槛失败；缺少经认证的同一合约二级价格锚和数量约束 | 清算机制的命名不能增加观测识别；MSE 失败也不是所有波动率模型不可能的证明 |
| [ohlc_path_learning_20261019/REPORT_ZH.md](../../ohlc_path_learning_20261019/REPORT_ZH.md) | 完整模型 MSE 比值 1.020229，95% 配对块区间 [0.987905, 1.094544]；126 最终构建中 63 个按验证选择第 0 轮；全部预定任务完成后 STOP | 共享状态、报告范围辅助、路径聚合没有产生可信的均值增益；此结果不允许事后把辅助分数换成主目标宣布成功 |

报告依据均来自仓库 `research/` 下的已完成实验；前史总览见 [PRIOR_AUDIT_ZH.md](../../ohlc_path_learning_20261019/audits/PRIOR_AUDIT_ZH.md)。此前已做过时钟、价格过滤、永久/暂时状态、释放事件等方向，因此不能把“潜在有效价格”“不规则观测”“创新状态”三个名称本身当成新发现。

### 2.1 为什么均值不断失败：一个可核验的风险分解

设需要预测的未来累计价格增量为 \(Y\)，可用历史为 \(X\)，真实条件均值为 \(m(X)=E[Y\mid X]\)。预测为当前价格加增量估计 \(\hat m(X)\)，随机游走对应增量 0。则

\[
E[(Y-\hat m)^2]-E[Y^2]
=E[(\hat m-m)^2]-E[m^2].
\]

任何方法的最大可得改善，先受 \(E[m^2]\) 限制。若推进门槛为相对 RW 改善 2%，则即使条件均值完全已知，也必须满足

\[
\frac{E[m^2]}{E[Y^2]}\ge 0.02.
\]

真实模型还要支付估计误差，要求更高。前史中的验证选择第 0 轮、自由模型优于机制模型或机制消融无损，比较符合“现有信息与样本下净可学习信号很小”的解释；这些实验不能证明所有条件均值为零，也不能证明市场效率命题。

重叠的多期限标签不是独立新增样本；从同一未来路径制造更多摘要或更多模拟轨迹，同样不会增加独立观测数。它们可能改善归纳偏好或估计效率，必须靠严格匹配对照验证。仅增加形状、状态和潜在变量，可能继续提高 \(E[(\hat m-m)^2]\)。

概率预测仍可研究，但需要在新实验之前公开改换目标。例如主任务为未来累计增量分布的 CRPS、分位数或波动风险时，应明确这不再是原来的点均值 MSE 改进命题。未来日增量平方和也是有限观测网格上的观测量，不能在无 tick 或市场微结构认证时叫作真实潜在 integrated variance。原价格鞅与对数价格鞅也不是同一假设。

## 3. QV、鞅控制变量及去符号监督

### 3.1 相同期望不等于普遍更低的标签方差

令 \(r_j\) 为未来第 \(j\) 步增量，\(S_j=\sum_{i\le j}r_i\)，则

\[
D_h^2=Q_h+C_h,\qquad
Q_h=\sum_{j=1}^h r_j^2,\qquad
C_h=2\sum_{j=1}^h S_{j-1}r_j.
\]

若 \(E[r_j\mid\mathcal F_{t+j-1}]=0\)，则 \(E[C_h\mid\mathcal F_t]=0\)。例如 iid 正态增量方差为 \(\sigma^2\) 时，\(\operatorname{Var}(D_h^2)=2h^2\sigma^4\)，\(\operatorname{Var}(Q_h)=2h\sigma^4\)，确实相差因子 \(h\)。

但存在完全有界的鞅反例：先令 \(r_1=\pm1/2\) 等概率。终值 \(D=\pm1\) 满足 \(P(D=1\mid r_1=1/2)=3/4\)、\(P(D=1\mid r_1=-1/2)=1/4\)，再令 \(r_2=D-r_1\)。此时 \(E[r_2\mid r_1]=0\)，而

\[
D^2\equiv1,\qquad
Q_2=\begin{cases}1/2,&p=3/4,\\5/2,&p=1/4,\end{cases}
\qquad\operatorname{Var}(Q_2)=3/4>0.
\]

所以应将“QV 比累计平方低噪声”限定在具体过程类，不能拿鞅恒等式当一般 Rao–Blackwell 证明。\(\log v+Q_h/v\) 是已知 QLIKE 形式。Patton 2011 已系统研究使用不完美波动率代理时保持排序性质的损失条件。[1]

### 3.2 普通神经控制变量的关键边界

可以构造

\[
C_{\theta,\psi}=\sum_j a_{\theta,\psi,j}(\mathcal F_{t+j-1})r_j.
\]

在真实 MD 和可交换微分/积分条件下，其期望为零。但若 \(a\) 与预测参数 \(\theta\) 无关且 detach，则

\[
\nabla_\theta(L_\theta-C_\psi)=\nabla_\theta L_\theta
\]

逐样本完全相同，没有降低训练梯度噪声。要影响梯度，必须构造与 \(\theta\) 相关的控制变量，或直接修正梯度向量。学可预测 hedge、最小化估计方差、用神经网络近似鞅表示本身已有直接文献；相关理论通常依赖已知模拟器的布朗创新，不可直接移植成未知真实市场漂移下的无偏性。[2]

若把增量换成 \(r_j-\hat\mu_j\)，则控制变量残留条件偏差

\[
E\!\left[\sum_j a_j(\mu_j-\hat\mu_j)\right].
\]

cross-fitting 可以缓解对同一标签的过拟合，不能使 \(\hat\mu\) 自动等于真漂移。对原修正评分的 nuisance Gateaux 导数是 \(E\sum_j a_j\delta_j\)。若要对所有可预测 \(\delta_j\) 为零，就只剩 \(a_j=0\) 的平凡情形；需要额外识别条件和真正的正交修正，而不只是写上“交叉拟合”。

### 3.3 去符号会消除真实杠杆关系

令 \(r_1=\pm1\) 等概率，\(\epsilon_2=\pm1\) 独立等概率，并令

\[
r_2=\begin{cases}2\epsilon_2,&r_1=+1,\\\epsilon_2,&r_1=-1.\end{cases}
\]

每一期增量给定过去都对称且均值为零，但未来第二期的幅度暴露第一期的符号。终值 \(D=r_1+r_2\) 在 \(3,-1,0,-2\) 各取概率 \(1/4\)，所以 \(E[D^3]=4.5\)。在固定整条幅度路径后独立随机翻符号，会强制终值对称并把三阶矩变成零，改变了真分布。

“每期创新条件对称”不蕴含“给定整个未来幅度路径后所有符号独立均匀”。后者更强，通常会排除想保留的杠杆效应。

## 4. 角向 RB：正确定理、直接近邻和学习散布反例

### 4.1 固定正确散布时的精确算子

在维数 \(H\ge2\)、给定历史 \(X\) 下，假定 \(r=A(X)RU\)，其中 \(U\) 为单位球面 \(S^{H-1}\) 上均匀方向，且条件于 \(X\) 与 \(R\) 独立；\(A\) 是正确且相对于预测参数固定的非奇异矩阵。令

\[
q=\|A^{-1}r\|,\quad a_h=\|A^\top c_h\|,\quad b_h=q a_h.
\]

则给定 \((X,q)\) 的终点投影为 \(b_hV\)，其中 \(V^2\sim\mathrm{Beta}(1/2,(H-1)/2)\)，且符号对称。其 CDF 为 \(G_b\)：在 \((-b,b)\) 内，

\[
G_b(z)=\tfrac12+\tfrac12\operatorname{sign}(z)
I_{(z/b)^2}\!\left(\tfrac12,\tfrac{H-1}{2}\right),
\]

两侧为 0、1，\(b=0\) 时用原点原子。角向平均的 CRPS 为

\[
\bar S_A(F_\theta,r)=\int\{F_\theta(z)^2-2F_\theta(z)G_b(z)+G_b(z)\}\,dz.
\]

该量就是条件期望；其风险与原始 CRPS 相同，预测梯度的协方差不增。这不是新的理论原理。Chen、Dobriban、Lee 2020 的 Lemma 1、§4.2.2 已给出群平均的期望与梯度协方差分解，且 **§8.2 专门以球面对称数据的范数/旋转来改善单坐标边际估计**。[3]

固定但估计错误的 \(\hat A\) 会把真分布变成相应轨道平均分布。冻结矩阵仅消除标签反传问题，不保证 \(\hat A\) 正确或市场满足椭球性。把不同期限各自独立角度采样也只能得到边际，不能代表同一联合路径方向。

### 4.2 散布参与标签反传：真模型不是驻点的显式例子

令真实二维路径 \(r\sim N(0,I_2)\)，候选 \(A_\theta=\operatorname{diag}(e^\theta,1)\)，预测第一坐标为 \(F_\theta=N(0,e^{2\theta})\)。将对应径向 \(\chi_2\) 分布保持不变，但用参数化散布旋转训练标签，得到

\[
Y_\theta=e^\theta\sqrt{e^{-2\theta}r_1^2+r_2^2}\,V_1.
\]

在真参数 \(\theta=0\)，写 \(r=RU\)，则 \(Y_0=RV_1\sim N(0,1)\)，并有

\[
\partial_\theta Y_\theta\big|_0=(1-U_1^2)Y_0.
\]

预测侧 CRPS 期望导数在真分布为零，但标签侧多出

\[
E\!\left[(2\Phi(Y_0)-1)\partial_\theta Y_\theta\big|_0\right]
=\tfrac12 E[(2\Phi(Z)-1)Z]
=\frac1{2\sqrt\pi}>0.
\]

因此梯度会从真尺度向错误缩小的方向移动。仅凭“这是 RB”不能合法联合学习 \(A_\theta\)。全部停止标签梯度后可定义另一种自洽 estimating equation，但不再自动等于所声称风险的梯度，识别和局部稳定性要单独证明。

未知散布的 influence-function 修正可能是进一步理论问题，例如

\[
\psi_{\theta,\eta}=g_{\theta,\eta}
-B_{\theta\eta}J_\eta^{-1}s_\eta,
\]

其中 \(s_\eta\) 识别标准化散布，\(B=\partial_\eta E[g]\)、\(J=\partial_\eta E[s]\)。正确设定和非奇异切空间上，它能令 nuisance 一阶导数消失。但这仍是既有 Neyman orthogonal score / two-step influence-function 原理的实例，Chernozhukov 等 §2.2 已给出一般构造；不应因加上此式立即宣称新方法。[4] 它也不消除非椭球真分布的建模偏差。

## 5. 候选四：全部连续区间投影的 Gaussian-mixture CRPS

### 5.1 模型和评分的一个无歧义定义

令 \(r\in\mathbb R^H\) 为未来增量向量；在历史 \(X\) 下预测

\[
P_\theta(r\mid X)=\sum_{k=1}^K\pi_k(X)
N_H\!\left(\mu(X),\beta_k(X)\Sigma(X)\right),
\]

其中 \(\pi_k>0\)、\(\sum\pi_k=1\)，固定原始尺度格点 \(s_k>0\)，定义 \(\beta_k=s_k/(\sum_\ell\pi_\ell s_\ell)\)，于是 \(\operatorname{Cov}(r\mid X)=\Sigma\)。若格点定义的是标准差，应先平方再进行上述方差归一化，二者不能混写。采用 \(\Sigma=D+BB^\top\)，\(D\) 为严格正的对角矩阵。

对所有 \(1\le a\le b\le H\)，令 \(c_{ab}\) 为区间指示向量，\(Y_{ab}=c_{ab}^\top r\)。对应预测是共同均值 \(m_{ab}=c_{ab}^\top\mu\)、共同基础方差 \(V_{ab}=c_{ab}^\top\Sigma c_{ab}\) 的一维尺度混合。取预先固定、严格正的权重 \(w_{ab}\)，定义

\[
S_{\rm int}(P,r)=\sum_{a\le b}w_{ab}\,
\operatorname{CRPS}(P\circ c_{ab}^{-1},Y_{ab}).
\]

权重可以依据区间长度或训练期固定尺度归一化，但不能因当前被评分的模型方差或已实现终值而任意调整。若权重/投影在训练期外层估计后冻结，则 propriety 是给定该冻结对象成立；不要把预测自身改变评分尺子的做法当作同一个定理。

解析公式是已知 Gaussian-mixture CRPS。设

\[
A(z,s)=2s\phi(z/s)+z\{2\Phi(z/s)-1\},
\]

则共享均值的单个区间分数为

\[
\sum_k\pi_k A(Y_{ab}-m_{ab},\sqrt{\beta_kV_{ab}})
-\frac12\sum_{k,\ell}\pi_k\pi_\ell
A(0,\sqrt{(\beta_k+\beta_\ell)V_{ab}}).
\]

这些闭式及其实际实现已有 Grimit 等 2006 及官方 `scoringrules.crps_mixnorm` 文档；不是新积分公式。[5]

### 5.2 严格识别证明：哪些表述成立

**命题。** 在有有限二阶矩、非退化、共同尺度混合的上述椭球分布族上，所有连续区间的边际分布决定联合分布；因此严格正加权的区间 CRPS 在该族上严格 proper。

证明可拆成三步。

1. 两个预测期望分数相同且为最小，严格 proper 的单变量 CRPS 使每个区间的边际分布相同。Singleton 边际给出 \(\mu_a\) 和对角 \(\Sigma_{aa}\)，所有区间边际给出 \(V[a,b]\)。
2. 对 \(a<b\)，

\[
\Sigma_{ab}=\tfrac12\{V[a,b]-V[a+1,b]-V[a,b-1]+V[a+1,b-1]\},
\]

空区间取 0，**对角另用 \(\Sigma_{aa}=V[a,a]\)**。不能把上式直接用于 \(a=b\)，否则会少一半。因此所有协方差相同。
3. 任意一个标准化非退化坐标的特征函数为 \(E[e^{-t^2W/2}]\)，其中 \(W\) 是归一化的正混合尺度。它在 \(t\in\mathbb R\) 上决定 \(W\) 的 Laplace transform，故决定混合法则。给定相同 \(\mu,\Sigma,W\)，全部联合特征函数相同。

此结论识别的是**分布**。低秩因子 \(B\) 的正交旋转、重复混合格点、边界零权重等可能使参数表述不唯一；这不违背分布层面的 strict propriety。固定相异正格点、所有权重严格正可减少参数别名，但应优先陈述分布结论。

对一般多元分布，有限个线性投影不能使用 Cramér–Wold 的“所有方向”结论，因此此分数仅 proper，不是普遍 strictly proper；即使这些投影覆盖每个均值/协方差，也不识别任意高阶联合依赖。

### 5.3 最小性与复杂度的精确范围

对于任意不受结构限制的对称协方差矩阵，固定标量线性投影数量至少为 \(H(H+1)/2\)：每个已知 Gaussian 投影方差只提供线性测量 \(c_i^\top\Sigma c_i\)；若数量不足，则测量映射有非零对称核 \(M\)，在内部正定矩阵附近 \(\Sigma\pm\epsilon M\) 都有效且无法区分。连续区间集合恰好达到此维数下界。

这不能证明它是低秩+对角子族的最小集合，也不是唯一达到下界的设计。\(\{e_i\}\cup\{e_i+e_j:i<j\}\) 同样达到下界，并以更稀疏的投影实现协方差恢复。

对秩 \(d\) 的 \(B\)，前缀和可直接得到

\[
V[a,b]=\sum_{j=a}^b D_{jj}
+\left\|\sum_{j=a}^b B_{j,:}\right\|^2.
\]

因此无需先构造完整 \(\Sigma\)，所有区间方差成本为 \(O(H^2d)\)，所有 K-component CRPS 成本为 \(O(H^2K^2)\)。只有把 \(d,K\) 看作固定常数时才能简写 \(O(H^2)\)。一般完整协方差的二维前缀和也能 \(O(H^2)\) 求全部 \(V\)，但构造/输出矩阵本身已有相应成本。

### 5.4 直接近邻：评分算子已有，窄识别结果尚不足以构成主创新

**Pic 等 2025，§4.2 式(16)** 明确给出对任意 patch 集合的均值变换做 CRPS 后求和；紧接着将均值变换改为总和 \(\sum_{i\in P}X_i\)。取 patch 为全部连续区间，按区间长度吸收权重，便是上面的评分。其 Proposition 1–2 给出 transformation 与 aggregation 的 proper 性。[6] 因而不能把“固定区间投影 CRPS 联合训练”本身称为首次提出的学习算子。

更直观地，令 \(Z_j=\sum_{i=1}^j r_i\)，\(Z_0=0\)，则区间总和等于 \(Z_b-Z_{a-1}\)。该集合是累计路径坐标所有两两差分的 CRPS，含与固定原点的差分。其协方差恢复也是平方距离/variogram 类恒等式的对应形式。

**Roordink 和 Hess，ECML-PKDD 2023 / arXiv 2024** 已用 Conditional CRPS 训练神经网络输出多元 Gaussian-mixture 的均值、协方差和权重，并给出解析分数与条件混合公式；覆盖条件链时具有一般的严格 proper 保证。[7] 它应是强制的同预测族评分对照，不能只比 MC-ES、NLL 和单变量 CRPS。

在本候选的共同尺度、低秩+对角结构中，可利用顺序秩更新计算 conditional-chain CRPS：每步基础条件均值和方差对各尺度成分相同，只按 \(\beta_k\) 放缩方差并更新后验权重。基础线性代数约 \(O(Hd^2)\)，混合 CRPS 约 \(O(HK^2)\)。这是可实现复杂度推导，**此审计未写代码计时**。因此区间法的 \(O(H^2)\) 不自动优于现有条件评分。

**MVG-CRPS** 使用预测协方差的特征基底。其原文 Appendix A 有依赖 Schur–Horn 的 Gaussian strict propriety 证明；重复特征值导致逐样本分数随任选基底变化，是定义约定、旋转不变性和反向梯度的实质问题，但不能仅凭这一点宣布它不 proper。[8] 固定方向避开此数值问题是可说明的实现性质，不能把未经推翻的原理论当成错误靶子。

### 5.5 与碳增量任务的关键冲突

若 \(r_j\) 真是对自然未来信息流的 MD，则对 \(i<j\)，

\[
E[r_ir_j\mid\mathcal F_t]
=E[r_iE(r_j\mid\mathcal F_{t+j-1})\mid\mathcal F_t]=0.
\]

所以 \(\Sigma\) 的条件非对角元素本来为零。波动聚集可以表现为平方增量相依，却不要求增量线性相关；GARCH/杠杆过程通常也不产生这里假定的联合椭球路径。若主要方法贡献押在自由非对角协方差学习上，可能又把有限样本预算花在难以学习的线性均值/相关结构上。

共同 Gaussian scale mixture 能表达共同风险强度造成的非线性相依，但会限制路径形状，不能叫作一般波动动态或杠杆建模。必须保留同族 diagonal、历史条件单一尺度、固定混合权重和零均值消融。学 \(\mu\) 时不应因为分布分数改善就宣称点预测 MSE 改善。

## 6. 当时提出的最低限度可证伪设计

### 6.1 拟进入市场实验前的数学门槛

1. **解析数值一致性。** 对合成正定矩阵检验区间恢复，包括 H=1、对角、空区间、近奇异有效情形；解析混合 CRPS 对照高精度一维积分；梯度做有限差分。不能只验证同一实现的重写式。
2. **最小识别正/负控制。** 在 H=3 或 4 构造只有 off-diagonal 不同的两个 Gaussian，singleton score 无法区分但全部区间可区分；少于协方差维数的固定投影构造核方向反例。
3. **目标正确性。** 非椭球/杠杆/非零漂移合成过程保留原始标签评分，不把真实分布变换后作为“更干净的真值”。
4. **统计而非积分效率。** 多个独立训练样本重复，比较总体预测风险和方差。更快把同一个积分算准、不同随机种子中挑最好结果，都不能作为样本效率证明。

### 6.2 同一预测族必须纳入的评分/结构对照

| 对照 | 需要排除的替代解释 |
|---|---|
| 同一 encoder、同一 \(\mu,D,B,\pi\) 的 NLL | 只是换了分布族而不是学习评分贡献 |
| 精度/预算匹配的 joint ES | 只是降低 Monte Carlo 数值噪声 |
| Singleton CRPS 与原点累计 h=1…H CRPS | 更丰富区间是否提供实质依赖识别与泛化收益 |
| 相同数量固定随机投影 CRPS | 区间的特定方向是否优于普通 sliced score |
| \(e_i\) 与 \(e_i+e_j\) 的最小完整集合 | “最小完整识别”并不是区间集合独有 |
| Gaussian-mixture CCRPS，至少一种可核验的完整 chain 版本 | 避免忽略现成解析、联合、严格 proper 的学习规则 |
| 同评分的 diagonal / zero mean / 固定 mix / 仅动态 scalar scale | 排除容量与普通条件风险尺度已解释全部增益 |
| 已有成熟经验分布、EWMA/FHS、适当 GARCH 和 QLIKE 风险基线 | 保证不是只战胜极弱均值基线 |

不能把这些所有配置无限展开成没有停止条件的大网格。先用纯理论和合成实验筛出一个有明确差异的候选，再为真实实验固定最小对照、主任务、一次性模型预算与推进门槛。

### 6.3 训练标签可使用未来，预测输入和 nuisance 不可以

一个起点的未来路径可以用于构建它已成熟的监督评分；这是合法监督。训练时使用未来前缀做 control variate 或 conditional score，也必须与预测输入分开：原点输出联合分布只依赖历史，不能把真实未来前缀喂进原点 encoder 再报告成事前预测。

所有用于拟合、选择、scaler、投影尺度、CV nuisance 的标签必须在对应截止时刻完全成熟；若最长 H=20，训练/验证与 nuisance 区块需按完整路径端点处理边界，重叠依赖要用适当 purge 和块推断。随机拆分交叉拟合不提供时序安全保证。已多轮暴露的旧开发区只能称开发复用，新候选不能拿它重新叫独立确认。旧 EEX 2024–2025 不应在这些理论问题仍未解决时消耗。

## 7. 当时的方法创新筛选建议

**当时的建议是不直接将普通区间 CRPS 作为“新方法”进入大规模真实训练。** 其定义可靠，但当时核读的最直接论文已经写出了同样的 transformation-and-aggregation 算子。应先决定是否存在超出该式的可证伪机制，例如：在明确的协方差误差与下游累计分布风险之间，推导方向设计或更新规则的可量化优势，并与稀疏最小投影集合和 CCRPS 匹配比较。这项筛选要求是在新一轮大实验前确认需要检验的方法命题。

候选窄命题至少必须写成类似：在给定过程类、观测量和容量限制下，方法 A 的目标风险误差界/局部敏感度/有效信息优于一个具体且充分匹配的 B；超出该过程类时明确给出反例。仅证明 proper、写一个矩阵正定约束、降低 MC 误差或添加 auxiliary summary loss，都没有解决这个差别。

当时另列的理论备选问题是未知散布条件下角向平均的估计误差正交化，但筛选时只有通用 DML 模板，不应冒称已得到新算法。若该修正最终只重现两步半参数推断，亦应停止这一支。筛选重点是确认可检验的方法差异，避免以命名替代贡献。

## 8. 原始文献与来源范围

原筛选阶段已核读以下相关正文或官方文档；检索记录截至 2026-10-19，本次公开整理未追加新一轮文献检索。没有检索到某个标题不等于证明全球无先例。

1. Andrew J. Patton (2011), *Volatility forecast comparison using imperfect volatility proxies*, Journal of Econometrics 160(1):246–256。作者 PDF：<https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf>。相关：Propositions 1–4，QLIKE 与代理鲁棒性。
2. Marc Sabate-Vidales, David Šiška, Łukasz Szpruch，*Martingale Functional Control Variates via Deep Learning*，更新版 *Unbiased deep solvers for linear parametric PDEs*：<https://arxiv.org/abs/1810.05094>，<https://arxiv.org/pdf/1810.05094>。相关：§2，基于已知随机过程创新的神经鞅控制变量。
3. Shuxiao Chen, Edgar Dobriban, Jane H. Lee (2020), *A Group-Theoretic Framework for Data Augmentation*, JMLR 21(245):1–71：<https://jmlr.csail.mit.edu/papers/v21/20-163.html>，<https://jmlr.csail.mit.edu/papers/volume21/20-163/20-163.pdf>。相关：Lemma 1、§4.2.2、§8.2。本文不宣称该文专门命名了 endpoint CRPS；其一般损失/梯度理论和球面边际实例直接覆盖此构造的原理。
4. Chernozhukov et al., *Double/Debiased Machine Learning for Treatment and Causal Parameters*：<https://arxiv.org/abs/1608.00060>，<https://arxiv.org/pdf/1608.00060>。相关：§2.2.1–2.2.5，score 正交化、GMM 和 influence-function 修正。时序重叠情形不可不加证明套用其 iid 主定理。
5. 官方 `scoringrules.crps_mixnorm`：<https://scoringrules.readthedocs.io/en/latest/generated/scoringrules.crps_mixnorm.html>；其引用 Grimit et al. (2006), DOI <https://doi.org/10.1256/qj.05.235>。闭式 Gaussian-mixture CRPS 已有标准实现。
6. Romain Pic, Clément Dombry, Philippe Naveau, Maxime Taillardat (2025), *Proper scoring rules for multivariate probabilistic forecasts based on aggregation and transformation*, ASCMO 11:23–58：<https://ascmo.copernicus.org/articles/11/23/2025/index.html>。相关：Propositions 1–2、§4.2 式(15)–(16)与紧接的 total 变换。它是“全部区间和 CRPS”最直接的已核验邻居。
7. Daan Roordink, Sibylle Hess，*Scoring Rule Nets: Beyond Mean Target Prediction in Multivariate Regression*，ECML-PKDD 2023；作者预印本 <https://arxiv.org/abs/2409.14456>，<https://arxiv.org/pdf/2409.14456>。相关：§2.3、Theorem 3、Appendix B，解析 Gaussian-mixture CCRPS 与神经网络训练；chain strict propriety 条件见其定理与附录。实现时应自行检查条件正态公式，PDF 中的部分下标有排版/转写问题。
8. Vincent Zhihao Zheng, Lijun Sun，*MVG-CRPS: A Robust Loss Function for Multivariate Probabilistic Forecasting*：<https://arxiv.org/abs/2410.09133>，<https://arxiv.org/html/2410.09133v2>。相关：Appendix A Gaussian strict propriety 证明、SVD 梯度讨论。
9. Andrew J. Patton, Kevin Zhang (2026), *Bespoke realized volatility: Tailored measures of risk for volatility prediction*, Journal of Econometrics 254,106122：<https://public.econ.duke.edu/~ap172/Patton_Zhang_BespokeRV_2026_JoE.pdf>。相关：§1–2，专为预测学习 RV 构成权重。其脚注 1 明确只调整 RHS 输入，保留标准 RV 作为目标；所以它不是本文 label-CV/RB 的完全同法先例，但“任务定制风险统计量”的一般口号也已经有近邻。

本文中的风险分解、两个 MD 反例、可学习散布的 CRPS 偏梯度、区间识别证明及共同尺度低秩 chain 复杂度推导是此次独立分析；它们不是市场精度结果，也不构成新颖性保证。
