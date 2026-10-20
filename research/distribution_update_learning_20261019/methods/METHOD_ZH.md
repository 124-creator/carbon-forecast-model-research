# 基于有界反射控制的条件分布更新：拟议研究方法

**English title: A Provisional Research Method for Conditional Distribution Updates with Bounded Reflection Controls**

方法版本：2026-10-19，依据原始协议 1.0 与运行前补充协议 1.1。本文是可供论文改写的方法定义、命题与实现对应说明，不是结果报告，也不认证新颖性或投稿成熟度。公式中的一般评分、控制变量、同时推断和矩阵上界工具均有先行工作；本项目待检验的是它们在一个具体路径控制和有限校准更新中的作用。

## 1. 预测任务、信息与符号

设严格递增的报价索引为 o，正价格为 P_o，历史信息为 H_o。期限 h 表示后续 h 个实际报价观测，市场实现取 h∈{5,20}，不将缺少报价的自然日补成交易观测。令 s_o>0 为只使用已知历史计算的日对数收益尺度，u_o=s_o√h，定义

\[
Y_{o,h}=\frac{\log P_{o+h}-\log P_o}{u_o},\qquad
r_{o,j}=\frac{\log P_{o+j}-\log P_{o+j-1}}{u_o},\quad
Y_{o,h}=\sum_{j=1}^h r_{o,j}.
\]

原点预测的输入 X_o 必须对 H_o 可测。未来路径 r_(o,1:h) 仅在目标成熟后用于监督和控制变量，不能作为原点预测输入。

主风险是归一化累计对数增量上的 CRPS。价格 CRPS、价格条件均值的 MSE、90% 区间覆盖与宽度另外报告。主风险的改善不蕴含这些其他目标也改善，亦不替代此前价格均值预测的研究命题。对数增量除以起点尺度，是对预测任务的明确定义，不是在评分后按结果选择样本权重。

下文对一个固定市场、期限和训练划分叙述。F0(·|X)、F1(·|X) 为较早训练后冻结的两个预测 CDF；π 表示 Gaussian mixture 成分概率，α 表示两个 CDF 的混合权重，w 表示历史门控的三个系数，K(X) 表示控制量幅度预算。δ 是条件符号偏离预算，ε 是置信界失效概率，二者不能混用。

## 2. 两个普通概率专家与价格变换

### 2.1 Gaussian scale mixture

基准专家为三个零中心 Gaussian 成分的尺度混合：

\[
F_0(y\mid X)=\sum_{m=1}^3\pi_m\Phi(y/\sigma_m),\qquad
\sigma_m=s_*\rho_m,\quad \rho=(0.5,1,2).
\]

π_m>0、Σπ_m=1，公共尺度 s_* 与概率由训练集原始 CRPS 拟合。当前实现用两个相对第三成分的 logit 参数，限制在 [−6,6]，s_*∈[0.25,2.5]。这是普通的有限混合预测族，不作为新模型结构认领。

候选专家只学习一个历史条件尺度因子：

\[
f_\theta(X)=0.5+\operatorname{sigmoid}(g_\theta(X)),\qquad
F_1(y\mid X)=\sum_{m=1}^3\pi_m
\Phi\!\left(\frac{y}{f_\theta(X)\sigma_m}\right).
\]

闭包内 f∈[0.5,1.5]。输出层零初始化给出 f=1，因此第 0 轮精确等于 F0。F1 使用普通 CRPS 训练，验证选择包括第 0 轮；控制变量不参与该神经网络训练。本方法真正要检验的学习规则位于后续校准阶段：从两个固定专家之间学习条件分布更新。

当前两个编码器为 MLP(32,16) 与 GRU(16)。它们共享过去 20 个报价的三个历史特征：当时对数收益/历史尺度、log 历史尺度、log(1+自然日间隔)。特征均值与标准差仅由训练起点拟合。神经训练固定为 AdamW，学习率 0.002，weight decay 0.001，batch size 64，梯度范数上限 1，60 个训练轮次；验证 CRPS 选择检查点后在校准前冻结。具体结构与数值约定以 source/market_models.py 为准。

### 2.2 精确 CRPS 与有限价格矩

对任意具有一阶矩的单变量预测分布，采用损失越小越好的约定：

\[
S(F,y)=\int_{\mathbb R}(F(z)-\mathbf1\{y\le z\})^2dz
=E|Z-y|-\tfrac12E|Z-Z'|,
\]

其中 Z,Z' 独立同分布于 F。该评分及其表示是已有结果 [R1]。令

\[
\Psi(z,s)=2s\varphi(z/s)+z\{2\Phi(z/s)-1\}.
\]

则零中心 Gaussian mixture 的评分直接由有限求和得到：

\[
S(F,y)=\sum_m\pi_m\Psi(y,\sigma_m)
-\frac12\sum_{m,l}\pi_m\pi_l
\Psi(0,\sqrt{\sigma_m^2+\sigma_l^2}).
\]

这里使用标准的正态绝对距离期望，没有提出新评分。方法训练、相对风险系数和连续混合的评价均可解析求值，不依赖模拟样本均值。

部署时若 Y~Fα，价格预测为 P=P_o exp(u_oY)。它是六成分以内的 lognormal mixture，支持严格为正。对任意固定起点与有限 q，有

\[
E[P^q\mid H_o]=P_o^q\sum_m\pi_m\left[
(1-\alpha)e^{q^2u_o^2\sigma_m^2/2}
+\alpha e^{q^2u_o^2f_\theta(X_o)^2\sigma_m^2/2}\right]<\infty.
\]

价格点预测采用 q=1 的精确条件均值；P_o exp(E[Y]u_o)=P_o 在这里不是该价格均值。以上是给定起点的矩存在性，不是对随机历史尺度的无条件矩存在性声明。价格 CRPS 同样由 lognormal 绝对距离期望解析计算。非线性价格变换不会保留不同预测器的 CRPS 风险排序，所以后文关于 Y 的风险命题不自动成为价格 CRPS 或价格 MSE 的保证。

## 3. CDF 混合的精确二次风险

定义 d_x(z)=F1(z|x)−F0(z|x)，并令

\[
F_\alpha(z\mid x)=F_0(z\mid x)+\alpha d_x(z),\qquad0\le\alpha\le1.
\]

这是有效 CDF 的纵向凸混合。定义两个观测级量

\[
a(x)=\int d_x(z)^2dz,\qquad
b(x,y)=\int d_x(z)\{\mathbf1(y\le z)-F_0(z\mid x)\}dz.
\]

平方展开给出精确等式

\[
S(F_\alpha,y)-S(F_0,y)=a(x)\alpha^2-2b(x,y)\alpha.\tag{1}
\]

这不是局部 Taylor 近似。实现还利用

\[
b(x,y)=\tfrac12\{a(x)+S(F_0,y)-S(F_1,y)\}.
\]

对于两个同权重混合，令 D_rs=E|Z_r−Z_s'|，r,s∈{0,1}，则 a=D_01−(D_00+D_11)/2，各 D 均由成分正态绝对距离求和计算。代码仅为消除数值舍入将计算所得 a 的微小负值投到零；数学上 a≥0。

设风险 R(F)=E[S(F(·|X),Y)] 有限，令 A=E[a(X)]、B=E[b(X,Y)]，则

\[
R(F_\alpha)-R(F_0)=A\alpha^2-2B\alpha.\tag{2}
\]

普通经验混合解为 clip(Σb_i/Σa_i,0,1)。本候选保留同一有效预测集合，改变的是有限校准样本如何形成风险改善的可信证据。分布聚合和 CRPS 聚合本身已有直接工作 [R2,R3]。

## 4. 参考续接的逐步反射控制

固定起点 x 与专家后，记 S_(j−1)=Σ_(l<j)r_l。选取事先规定的参考续接噪声

\[
Z_j^{\rm ref}\sim N(0,v_j),\qquad v_j=(h-j)/h,
\]

定义

\[
G_j(s;x)=E_{\rm ref}[b(x,s+Z_j^{\rm ref})],
\qquad
M_j=\tfrac12\{G_j(S_{j-1}+r_j;x)-G_j(S_{j-1}-r_j;x)\}.\tag{3}
\]

参考过程是每步方差 1/h 的独立 Gaussian 续接，其终端参考方差为 1；它不是估计出的真实未来过程。给定当前步以前的信息，函数 r↦M_j(r) 为奇函数，只要反射的两边使用相同已固定的 G_j 和过去前缀。参考分布错误可以改变控制量与 b 的相关性及效率，但不改变这个代数奇性。它不能使不对称真实市场自动满足零均值控制。

对于当前 Gaussian mixture，卷积只需把成分方差加 v_j。与 y 无关的 a 和分布自身距离常数在两侧相减时消失，因此

\[
\begin{aligned}
M_j=\tfrac14\sum_m\pi_m\{&\Psi(S_{j-1}+r_j,\sqrt{\sigma_m^2+v_j})
-\Psi(S_{j-1}+r_j,\sqrt{f(x)^2\sigma_m^2+v_j})\\
&-\Psi(S_{j-1}-r_j,\sqrt{\sigma_m^2+v_j})
+\Psi(S_{j-1}-r_j,\sqrt{f(x)^2\sigma_m^2+v_j})\}.
\end{aligned}\tag{4}
\]

只改变当前增量的符号、保持已实现前缀，因此没有固定整条未来幅度路径后重抽全部符号。后续幅度可以依赖前面符号，真实过程可以具有 leverage。可预测控制变量、其神经构造以及概率/分位数控制均已有文献 [R4,R5]；本式尚未被确认为一个新颖的独立方法贡献。

## 5. 范围证明、主预算 K=W1/4 与版本记录

### 5.1 单步反射的确定性范围

有限一阶矩下，W(x)=W1(F0(·|x),F1(·|x))=∫|d_x(z)|dz。由 b 对 y 的导数或积分增量恒等式，

\[
\frac{\partial b(x,y)}{\partial y}=F_0(y\mid x)-F_1(y\mid x),\qquad
\operatorname{osc}_y b(x,y)\le W(x),
\]

其中 osc 表示完整支持上的上确界减下确界。两个专家同中心对称时，b 关于该中心为偶函数，进一步有

\[
\operatorname{osc}_y b(x,y)\le W(x)/2.\tag{5}
\]

证明只需在一侧半轴积分 |d_x|，其积分等于 W/2。当前 F1 是 F0 的正尺度倍数，d_x 在正半轴不换号，因此该上界达到；范围可以从 y=0 与 |y|→∞ 的两个极限精确得到。

卷积平均不会增加函数的 oscillation，于是

\[
|M_j|\le\tfrac12\operatorname{osc}G_j\le W(x)/4.\tag{6}
\]

共同正尺度耦合给出当前预测族的精确 Wasserstein 距离：

\[
W(x)=|f(x)-1|E|Z_0|
=|f(x)-1|\sqrt{2/\pi}\sum_m\pi_m\sigma_m.\tag{7}
\]

理由是 Q1(p)=fQ0(p)，一维单调分位耦合正好达到 W1。式 (5) 是固定 x 时关于 y 的范围，不能把它直接视为 x 也随机时 b(X,Y) 的全局支持长度。后者须另证联合范围或使用合法的统一包络。

### 5.2 逐路径绝对幅度预算

主设定取 K(x)=W(x)/4。令剩余预算与截断控制为

\[
K_j=\max\{K(x)-\sum_{l<j}|\widetilde M_l|,0\},\qquad
\widetilde M_j=\operatorname{clip}(M_j,-K_j,K_j),\qquad
C=\sum_{j=1}^h\widetilde M_j.\tag{8}
\]

K_j 在第 j 步发生前可知，对称截断保持当前增量的奇性。归纳立即得到逐路径界

\[
\sum_j|\widetilde M_j|\le K(x),\qquad |C|\le K(x).\tag{9}
\]

该界控制的是控制变量的绝对幅度总额；它不是概率分布的 total variation 距离。任何事先指定的非负 K 都可以作同类预算；W/4 并没有被证明是统计效率最优选择。

原协议 1.0 使用 K=W。保留该文件与其 full-cap 对照。补充协议 1.1 在独立四路径审计之后、一般反射控制合成与市场拟合开始之前，将主设定定为 W/4，理由仅为式 (5)–(6) 的确定性范围证明。同时预先保留 K=W、无截断及无前缀对照。时间与原因记录于 provenance/DESIGN_ADDENDUM_PRE_RUN.json；这是本地时间记录，不称外部预注册，也不将其说成在所有早期合成审查之前作出的决定。

## 6. 条件符号偏离与偏差预算

记第 j 步真实过去信息为 H_(j−1)，包含原点信息与已经实现的前缀。关键假设为

\[
\left|E[\operatorname{sgn}(r_j)\mid |r_j|,H_{j-1}]\right|\le\delta
\quad\text{a.s.},\qquad j=1,\ldots,h.\tag{10}
\]

对零幅度定义 sign(0)=0。这是给定当前幅度后的条件符号概率差约束；无条件正负比例平衡、零均值、条件鞅差或一次未拒绝的对称检验，均不能推出此约束。δ=0 且可积时，此约束蕴含条件鞅差；δ>0 则允许非零漂移，不能笼统称它在逻辑上强于零均值或鞅差假设。

固定过去与当前幅度时，奇控制的两种符号取值互为相反数。因此

\[
|E[\widetilde M_j\mid |r_j|,H_{j-1}]|
\le\delta|\widetilde M_j|,
\]

其中右侧绝对幅度对该条件信息可测。塔式期望与式 (9) 推出

\[
|E[C\mid X]|\le\delta E[\sum_j|\widetilde M_j|\mid X]
\le\delta K(X).\tag{11}
\]

若 v_k(X)≥0 为原点可知的门控基函数，还得到

\[
|E[v_k(X)C]|\le\delta\,E[v_k(X)K(X)].\tag{12}
\]

严格条件对称 δ=0 时控制均值为零；δ>0 时支付明确的偏差成本。当前主市场敏感性预算为 δ=0.05，预定敏感性为 0、0.025、0.10。这些是研究假设/敏感性设定，不是从市场数据认证的上界。

不能将式 (10) 换成普通的逐期反射 TV≤δ 后仍保留不含 h 的 δK 界。一般奇控制类的反例是每步以概率 δ 取 +1，否则为 0，第一次非零事件用尽全部 K。其逐期反射 TV 为 δ，但 EC=K[1−(1−δ)^h]，可能远大于 δK；给定非零幅度的符号偏离实际为 1。原始审计明确保留这一反例。

δ 预算错误或部署分布变化均可能使后面的风险保证失效。控制变量均值接近零的校准结果，不提供未来时期的同一条件保证。偏差控制与估计控制均值已有直接文献 [R6]，本研究不能据此声称一般意义上的首次有偏控制变量方法。

## 7. 有限控制系数选择与标量更新

预先固定 Λ={0,0.5,1}。对每个 λ，控制后的校准变量为 b_i−λC_i，样本均值为 μhat_λ。设 u_λ 给出同时有效的一侧统计误差界，κ_upper 是 E[K(X)] 的有效上界。则

\[
L_\lambda=\widehat\mu_\lambda-u_\lambda-|\lambda|\delta\kappa_{\rm upper},
\qquad L=\max_{\lambda\in\Lambda}L_\lambda.\tag{13}
\]

只要这些统计界同时成立，式 (11) 保证 B≥L。若另外有 A≤U_A，取

\[
\widehat\alpha=\operatorname{clip}(L/U_A,0,1),\tag{14}
\]

在 U_A=0 的无信息情形返回 α=0。由式 (2)，在联合事件上

\[
R(F_{\widehat\alpha})-R(F_0)
\le U_A\widehat\alpha^2-2L\widehat\alpha\le0.\tag{15}
\]

选择 λ 所需的同时保护不可省略。把多个独立 95% 界逐个计算后择优，不构成共同 95% 保证。λ=0 提供不使用控制的候选，但增加 Λ 的选择成本仍可能使整套方法比独立校准的 raw 方法保守。因此对照同时保留 raw 与相同联合预算的 raw 版本。有限族统一推断与 Learn then Test 已有相应基本原理 [R7,R11]。

控制量的潜在收益来自 Var(b−λC)=Var(b)−2λCov(b,C)+λ²Var(C) 的下降，但更新时同时要支付范围项、均值偏差与多重选择成本。标签方差下降本身不等于预测风险改善。

## 8. 三维历史条件门控与凸二次规划

### 8.1 固定基函数与学习参数

令 z(X) 为用训练集均值和标准差标准化后的 log 历史 EWMA 尺度，取中心 c=(−1,0,1)、宽度 τ=0.75，定义

\[
v_k(X)=\frac{\exp[-(z(X)-c_k)^2/(2\tau^2)]}
{\sum_{l=1}^3\exp[-(z(X)-c_l)^2/(2\tau^2)]}.
\]

v_k≥0、Σv_k=1。门控形式为

\[
\alpha_w(X)=w^\top v(X),\qquad w\in[0,1]^3.
\]

因此每个历史状态上均有 0≤α_w≤1，输出保持为有效 CDF。校准阶段仅学习三个系数；基函数的中心、宽度和标准化不在该阶段调优。

设

\[
A=E[a(X)v(X)v(X)^\top],\qquad
B=E[b(X,Y)v(X)].
\]

则由式 (1) 得到精确总体风险差

\[
R(F_w)-R(F_0)=w^\top Aw-2B^\top w.\tag{16}
\]

### 8.2 线性项的同时下界

对每个 λ,k，估计 μ_(λ,k)=E[(b−λC)v_k]；对 κ_k=E[Kv_k] 构造上界 κ_upper,k。若 μ_(λ,k)≥μhat_(λ,k)−u_(λ,k) 同时成立，则取

\[
L_k=\max_{\lambda\in\Lambda}
\{\widehat\mu_{\lambda,k}-u_{\lambda,k}
-|\lambda|\delta\kappa_{{\rm upper},k}\},\tag{17}
\]

得到 B_k≥L_k。不同坐标可以选择不同 λ；共同保护覆盖整个 Λ×{1,2,3}，而不是事后只保护被选中的三个值。

### 8.3 Gershgorin 上界与求解

令 Ahat=n^−1Σ_i a_i v_iv_i^T，它本身半正定。假设有对称逐元素误差界 |A_jk−Ahat_jk|≤e_jk=e_kj，定义

\[
D=\operatorname{diag}\!\left(\sum_k e_{1k},\sum_k e_{2k},\sum_k e_{3k}\right),
\qquad U=\widehat A+D.\tag{18}
\]

对任意向量 q，利用 2|q_jq_k|≤q_j²+q_k²，

\[
q^\top(A-\widehat A)q
\le\sum_{j,k}e_{jk}|q_jq_k|
\le q^\top Dq.
\]

因此 A≼U，且 U≽0。这是标准对称矩阵/Gershgorin 型上界，不作为新矩阵定理认领。最终解为

\[
\widehat w\in\arg\min_{0\le w\le1}
Q(w),\qquad Q(w)=w^\top Uw-2L^\top w.\tag{19}
\]

由于 w=0 可行、w≥0 且 B≥L，在联合事件上

\[
R(F_{\widehat w})-R(F_0)
\le Q(\widehat w)\le Q(0)=0.\tag{20}
\]

这个上界对整个连续盒约束同时成立，不必再把所有 w 离散成网格做多重检验。它保证相应目标总体的平均风险，不保证每个 X、每个年份或每个个体起点都改善。

实现枚举三个坐标“自由、固定 0、固定 1”的最多 27 个面，解自由坐标的线性方程并核可行性，在所有可行候选中取最小值；零向量始终保留。仅数值相同专家时使用预先规定的极小 a 阈值直接回退 F0。没有在此引入新的网络优化器。

## 9. 独立样本命题及明确的理论范围

**命题。** 给定独立的较早拟合资料，F0、F1、参考续接、K 与 v 均已冻结。设校准路径 (X_i,r_i,1:h) 独立同分布于 P，未来评价路径也来自 P；路径内部允许依赖。假定评分风险有限、式 (10) 成立、所用校准统计量具有事先已知的有界支持，并构造失效概率合计不超过 ε 的所有界。如果按式 (17)–(19) 更新，则校准样本概率至少 1−ε 下，有 \(R_P(F_{\widehat w})\le R_P(F_0)\)。

**证明。** 条件符号预算给出式 (12)；所有线性项与幅度预算的共同统计事件给出 B≥L；所有矩阵元素的共同事件与式 (18) 给出 A≼U。各事件的联合概率至少为 1−ε。其上式 (20) 对数据选出的 \(\widehat w\) 成立。证毕。

一个可实现的一侧均值半径由 Maurer–Pontil 的 empirical Bernstein 界提供 [R11]。若 Z_i 独立同分布、完整支持长度为 R_Z，n≥2，样本方差用 n−1 分母，则

\[
u(Z;\eta)=\sqrt{\frac{2s_n^2\log(2/\eta)}{n}}
+\frac{7R_Z\log(2/\eta)}{3(n-1)}.\tag{21}
\]

对需要下界的变量使用该界的相反方向；对 A 的两侧分别分配误差预算。d=3、|Λ|=3 时，9 个 B 下界、3 个 K 上界、6 个矩阵元素的两侧，合计 24 个一侧事件；η=ε/24 是一个有效而可能保守的简单分配。一般有限族统一界也可使用，但不能由测试效果决定保护哪些分量。

若 W(X)≤M、K(X)≤Kmax，则 a∈[0,M]，|b|≤M，|C|≤Kmax，v_k∈[0,1]，故校正线性变量的全局支持长度最多为 2(M+|λ|Kmax)。这些边界不需要截断真实 Y。更紧范围必须从预测族与完整支持证明，不能用样本最小值和最大值替代。

当前一般合成的标量验证固定 F0、F1，a 与 K 为已知常数，因此只需对 Λ 的 B 界分配误差，使用由 y=0 和无穷极限证明的 b 范围。source/learning.py 中的 bernstein_scalar 只实现这一已知 a、已知 K 的独立样本特殊情形；它不是对一般随机历史状态矩阵的完整有限样本实现。

本命题不因时间切分或路径不重叠而自动适用于金融时间序列。依赖路径、训练到校准的状态依赖、未知偏差预算、校准到部署的分布漂移，均需要额外条件。对不同市场、期限和随机种子同时声称保证，还要另分配跨问题误差预算。本文没有作这些额外保证。

## 10. 市场实现：固定的近似块校准

市场实现使用 circular moving-block bootstrap，沿成熟校准起点的原顺序抽取循环连续块，块长 2h，拼接后截取到 n=252，重复 399 次。同一重复对所有统计坐标使用相同起点索引。循环块方法属于已有依赖样本重抽样工作 [R12]；这里的特定使用方式是经验适配。

将统计向量写为

\[
T_i=\big((b_i-\lambda C_i)v_{ik}\ \forall\lambda,k;
\ a_iv_{ij}v_{ik}\ \forall j\le k;
\ K_iv_{ik}\ \forall k\big).
\]

主设定共有 18 个统计坐标。记 mhat 为样本均值，mhat* 为块重复均值，se 为其重复标准差，z*=(mhat*−mhat)/se。每个重复计算共同最大值

\[
T^*=\max\{z_B^*,\ |z_A^*|,\ -z_K^*\},
\]

取非负的名义 95% 分位 q*。构造 B 的未扣偏差下界 mhat_B−q*se_B、A 的双侧半径 q*se_A、K 的上界 max(0,mhat_K+q*se_K)，随后代入式 (17)–(19)。se≤1e−14 的坐标将标准化偏差置零，构界仍使用保存的 se；代码没有把该标准差本身清零。

这是一个共同 max-statistic 的**近似块自助法校准**。它不等于式 (21)，没有本文已经证明的有限样本覆盖，也不保证任意非平稳市场下一年的风险不增加。文件与运行收据明确标记 nominal_only。bootstrap 半径恰好很小或为零也不能被当成总体变量无随机性的证明。

raw_block_gate3_same_joint_budget 使用与主规则相同的 Λ 统计和共同最大值，只允许最终选 λ=0，因而保留相同多重选择预算；独立的 raw_block_gate3 则只构造无控制统计。δ 敏感性、full-cap、uncapped、scalar gate 和 no-prefix 均在运行前列出，不能在测试后择优替换主规则。

无截断版本使用每步 |M_j|≤W/4，给出 |ΣM_j|≤hW/4，支付相应 h 倍预算。no-prefix 将式 (3) 的过去前缀置零；由于当前零中心专家与参考续接使 G_j 为偶函数，该控制逐项恒为零。h=1 时也必有 C=0，作为机制与实现的负对照。

## 11. 时间划分、成熟标签与比较预算

对于测试年份的首个报价索引 T，当前实现使用以下闭区间起点：

\[
\begin{aligned}
c_{end}&=T-h-1,&c_{start}&=c_{end}-251,\\
v_{end}&=c_{start}-h-1,&v_{start}&=v_{end}-125,\\
t_{end}&=v_{start}-h-1.&
\end{aligned}
\]

训练为 [39,t_end]，验证 126 个起点，校准 252 个起点；校准不是固定的一整自然年。每个角色的最后 h 步标签端点严格早于下个角色的首个起点。测试只保留起点和 h 步端点都位于指定测试年份的观测。

历史 EWMA 方差按 v_o=0.94v_(o−1)+0.06r_o,daily² 递推；尺度下限为训练标签前缀的 0.1 倍日对数收益绝对值中位数与 1e−8 的较大者。前 20 个收益提供初始化，最早训练起点为 39，故部署历史特征使用的信息已经可得。训练得到的标准化参数、F0、F1 及门控基函数在校准前固定；w 在校准后冻结。测试时 α_w(X_o) 会随已知历史变化，但 w 不用测试标签再拟合。

必须采用相同两个专家比较经验标量混合、经验三维 gate、raw 校准与路径控制校准，以隔离更新规则。额外提供 BOA-CDF frozen 与 BOA-CDF online；后者在部署时先接收 target_index≤当前起点的已发出预测反馈，再发新预测，梯度使用历史实际 alpha。逐分位 BOAG 同样保存实际发出的分位数。由延迟回放产生的更新不直接承接无延迟 BOA 的 regret 定理；它是明确命名的实用延迟适配 [R3]。

逐分位强对照采用 19 个中点水平 τ_j=(j−0.5)/19，输出重排后作为等权 19 原子分布，计算该离散分布的精确 CRPS。其有限尾部表示与作者完整 profoc 不同；先重排再按分位更新也按作者实现作适配，不把未重排线性池的梯度证明直接搬用。细节见 ONLINE_BASELINES_ZH.md。

另有 FHS、EWMA_Normal 与 GARCH_Normal 等基础分布对照。当前 GARCH_Normal 拟合零均值 Gaussian GARCH(1,1)，h>1 时用累计条件方差期望形成单一 Gaussian，是方差匹配近似，不是对未来随机波动递推积分得到的完整多步 GARCH 密度。当前固定实验不等于穷尽 GAS、skew-t 或全部神经波动模型；可据实际对照得出结论的范围应保持一致。

## 12. 直接近邻与可认领贡献的边界

| 原始工作 | 已有部分 | 本项目具体差别及尚需检验之处 |
|---|---|---|
| Berrisch–Ziel CRPS Learning [R2]；Wintenberger BOA [R3] | 按 CRPS/分位损失学习组合、二阶在线聚合 | 本项目用成熟未来路径改善校准线性项，支付偏差与联合置信成本，再解三参数条件更新。CDF 混合、动态权重、二阶项均不是首次提出。 |
| Hesterberg–Nelson 概率与分位数控制 [R4] | 控制变量降低概率/分位估计误差 | 此处校正相对 CRPS 线性标签，并利用条件反射路径。单纯控制指示函数或分布评分不构成区别。 |
| Sabate-Vidales 等神经鞅控制 [R5] | 已知随机系统中的可预测控制与方差缩减 | 此处参考续接可错误；零均值或偏差界来自另外写明的真实条件符号假设，不能继承已知 Brownian 驱动的无偏保证。 |
| Pasupathy 等估计控制均值 [R6] | 控制均值误差、额外样本成本、系数收缩 | 本项目按条件符号与路径绝对幅度给偏差包络；显式处理有偏控制本身不是新思想。 |
| Learn then Test [R7] | 用同时风险推断保护参数选择 | 本项目的连续 w 不需要网格，是精确 CRPS 二次式与矩阵包络的结果。根据置信界拒绝更新不是首次风险控制。 |
| Almeida 等 2025 [R8] | Appendix D 已将控制变量用于风险校准，已关注分样与范围成本 | 此处使用弱对称路径条件而非期望已知的重要性权重。能否在支付成本后改善更新效率，需要实际证据，不能只展示样本方差下降。 |
| 半监督 PPI 风险控制 [R9]、主动 anytime 风险控制 [R10] | 辅助风险预测、残差校正、顺序或有限样本风险校准 | 此处不依赖额外标注/无标签样本或已知随机标注概率；代价是需要明确且未获市场认证的条件符号预算。 |
| 群平均数据增强 [R13] | 对称变换、风险与梯度的平均化及方差性质 | 当前方法是过去条件下的单步奇控制，并带绝对幅度预算，不是对整个路径作旋转或符号轨道平均。不能借对称操作一词认领一般新理论。 |

以上差别界定了一个可证伪的研究问题，尚未证明该组合在方法层面足够新颖。可能的增量仅限：解析反射续接、保持可预测性的路径预算、弱条件符号偏差成本，以及由此改变三维校准更新的有限样本效率。需要与充分匹配的 raw、同联合预算 raw、普通经验混合、BOA 及结构消融比较，才能判断控制机制是否有净价值。本文没有任何实验成功、优于所有基线或达到论文标准的结论。

## 原始参考文献

- **[R1]** Gneiting, T. & Raftery, A. E. (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation*. DOI: [10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437).
- **[R2]** Berrisch, J. & Ziel, F. *CRPS Learning*. [作者接受稿](https://arxiv.org/pdf/2102.00968v3). DOI: [10.1016/j.jeconom.2021.11.008](https://doi.org/10.1016/j.jeconom.2021.11.008).
- **[R3]** Wintenberger, O. (2017). *Optimal learning with Bernstein Online Aggregation*. [作者全文](https://arxiv.org/pdf/1404.1356v5). DOI: [10.1007/s10994-016-5592-6](https://doi.org/10.1007/s10994-016-5592-6).
- **[R4]** Hesterberg, T. C. & Nelson, B. L. (1998). *Control Variates for Probability and Quantile Estimation*. [作者机构全文](https://users.iems.northwestern.edu/~nelsonb/Publications/HesterbergNelson.pdf). DOI: [10.1287/mnsc.44.9.1295](https://doi.org/10.1287/mnsc.44.9.1295).
- **[R5]** Sabate-Vidales, M., Šiška, D. & Szpruch, Ł. (2022). *Unbiased Deep Solvers for Linear Parametric PDEs*. [作者全文](https://arxiv.org/pdf/1810.05094). DOI: [10.1080/1350486X.2022.2030773](https://doi.org/10.1080/1350486X.2022.2030773).
- **[R6]** Pasupathy, R., Schmeiser, B. W., Taaffe, M. R. & Wang, J. (2012). *Control-variate estimation using estimated control means*. [作者机构全文](https://web.ics.purdue.edu/~pasupath/PAPERS/2012pastaaschwan.pdf). DOI: [10.1080/0740817X.2011.610430](https://doi.org/10.1080/0740817X.2011.610430).
- **[R7]** Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I. & Lei, L. *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control*. [作者全文](https://arxiv.org/pdf/2110.01052).
- **[R8]** Almeida, D. C., Bravo, J., Bono, J., Bizarro, P. & Figueiredo, M. A. T. (2025). *High Probability Risk Control Under Covariate Shift*. [作者机构全文，Appendix D](https://research.feedzai.com/oshuzuxa/2025/10/high_probability_risk_control_under_covariate_shift.pdf).
- **[R9]** Einbinder, B.-S., Ringel, L. & Romano, Y. (2025). *Semi-Supervised Risk Control via Prediction-Powered Inference*. [作者全文](https://arxiv.org/pdf/2412.11174). DOI: [10.1109/TPAMI.2025.3594263](https://doi.org/10.1109/TPAMI.2025.3594263).
- **[R10]** *Active, anytime-valid risk controlling prediction sets*. [作者全文](https://arxiv.org/pdf/2406.10490).
- **[R11]** Maurer, A. & Pontil, M. (2009). *Empirical Bernstein Bounds and Sample Variance Penalization*. [作者全文，Theorem 4 / Corollary 5](https://arxiv.org/pdf/0907.3740).
- **[R12]** Politis, D. N. & Romano, J. P. *A Circular Block-Resampling Procedure for Stationary Data*. [Stanford 原始技术报告入口，1991](https://statistics.stanford.edu/technical-reports/circular-block-resampling-procedure-stationary-data). 后发表于 *Exploring the Limits of Bootstrap* (1992)。本处引用方法来源，不声称本文具体市场 bootstrap 已被该文定理覆盖。
- **[R13]** Chen, S., Dobriban, E. & Lee, J. H. (2020). *A Group-Theoretic Framework for Data Augmentation*. [JMLR 原文](https://jmlr.org/papers/volume21/20-163/20-163.pdf).

实现核对范围：source/scores.py、learning.py、market_models.py、run_market.py、run_synthetic.py、online_baselines.py 与两份运行前协议。本文只撰写方法；没有为写作改动这些源码或重新拟合市场模型。
