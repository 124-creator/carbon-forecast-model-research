# OHLC部分路径摘要路线：独立理论裁决（一页）

**裁决：可以登记一次低容量结构性试验，当前不能登记“已找到模型创新”。** `联合生成路径→累积max/min→联合评分`有可实现性，也有真正区别于三个独立边际头的约束；但生成模型的确定性推前、辅助监督和时序参数共享均是既有原则。Lin–Sun（2019）已只观测OHLC、将日内路径视为潜变量并推导H/L/C联合密度；Biswas等（2026）已研究任意非线性约束下的概率调和。因此，不应以“局部摘要约束路径”这一概括作为原创贡献。[1–3] 本审查没有执行市场拟合。

## 1. 两条必须先成立的边界

对 `X_s=μs+σW_s`、固定σ，整个路径相对零漂移的似然比为 `exp{μ(X_T−X_0)/σ²−μ²T/(2σ²)}`，只依赖终点。因此 `pμ(H,L|C,O,σ)=p0(H,L|C,O,σ)`：**给定C后，H/L没有额外漂移信息**。未知σ可由区间更精确估计，但这不自动改善恒定漂移的估计，更不保证原价MSE改善。若所有训练close路径已经可见，窗口max/min只是已有OHLC标签的确定性摘要；窗口/期限数不是新增独立样本。

必须认证同一合约、同一session的 `L≤min(O,C)≤max(O,C)≤H`。若“Price”可能是session外settlement，不能用夹紧、改H/L、删除越界行或任意观测噪声挽救路径假设。保留全部原始close目标；数据定义未解决时，只能单独预测报告H/L统计，不能声称它们约束该close的成交路径。日内H/L也不能替换为离散daily-close路径的max/min。

## 2. 最简可证伪实现（结构探针，非原创认证）

采用**日级OHLC元组的单潜状态递推**，避免假装有tick。所有值以当前close为零点、历史尺度s标准化；小型GRU16仅从过去20个OHLC元组初始化z₀及起点漂移a，过程其余系数全局共享：

\[
z_j=\rho z_{j-1}+\sigma_z\epsilon_j,\quad
o_j=c_{j-1}+a_g+b_gz_{j-1}+\sigma_g\eta_j,\quad
c_j=o_j+a_c+b_cz_{j-1}+\sigma_c\xi_j,
\]
\[
u_j=\max(o_j,c_j)+\operatorname{softplus}(q_u+d_uz_{j-1}+\tau_u\nu_j),\quad
\ell_j=\min(o_j,c_j)-\operatorname{softplus}(q_\ell+d_\ell z_{j-1}+\tau_\ell\omega_j).
\]

取 `|ρ|<1`、正尺度；每个样本一次产生20天，`U_h=max_{j≤h}u_j`、`L_h=min_{j≤h}ℓ_j`，再还原原价。它只表示日级元组的工作分布，不识别日内高低出现顺序，也不声称Gaussian局部价格模型在所有情景保持价格正支持。close均值可解析计算：`E[c_h]=h(a_g+a_c)+(b_g+b_c)z₀∑_{j=0}^{h−1}ρ^j`；h=1/5/20主MSE无需蒙特卡洛。高低与联合摘要用固定128个扰乱Sobol节点和一套独立验证节点，形成可微有限样本energy score；原价主损失使用训练期固定分母。评分向量为 `(C₁,C₅,C₂₀,U₁,U₅,U₂₀,L₁,L₅,L₂₀)`，需用**联合**energy score，不能三个边际CRPS加起来后声称识别联合依赖。只将其称为proper-score型复合目标，不称已认证观测似然。max/min几乎处处可导，离散节点积分误差应在训练前检查。

**这仍是已有状态空间＋受约束元组解码，均值族甚至接近项目旧永久/暂态预测族。** 它的唯一受检验主张应是：范围标签对同一低维动态的约束能否在相同信息与标签下带来有限样本close精度，而不是“表达能力更强”“识别真实日内路径”。

## 3. 最强匹配对照及一次停止条件

| 必要对照 | 排除的解释 |
|---|---|
| 同信息直接GRU、线性/树模型，均接收历史OHLC | 收益是否仅来自新增H/L输入 |
| 同容量自由联合未来OHLC解码器，同Sobol维数、同9维联合评分，仍用cummax/cummin | 收益是否真的来自共享动态；只比独立三个头太弱 |
| 同路径模型去摘要损失；同路径模型阻断摘要到close/漂移部分的梯度（编码/波动通道也要按声明处理） | 收益是否来自摘要监督和其对close估计的实际耦合；仅共享编码器的常规多任务效应不能冒充路径新约束 |
| 经典OHLC/Brownian区间模型与普通多任务模型 | 是否只是已知区间波动估计或辅助正则 |

先做两类合成检查：恒定漂移Brownian负控制中不应凭H/L“额外恢复漂移”；另设明确非Brownian、持续潜状态同时影响区间不对称与未来**有符号**均值的正控制，与相同OHLC信息的直接模型比较。再进入一个冻结开发实验。若只改善range/CRPS、只胜RW而未胜同信息自由联合模型，或去摘要/断开耦合不损伤close MSE，**停止本候选**，不追加结构救分。若新颖性是先于训练的硬门，本方案目前不足以通过，应继续寻找具体估计规则，不能用“非线性max/min”替代创新论证。

## 一手证据（已实际打开，非作者实验复现）

[1] Lin L-C, Sun L-H. *Modeling financial interval time series*. PLOS ONE, 2019, DOI10.1371/journal.pone.0211709。核读Main results、Theorem3/4及stochastic volatility段：[全文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0211709)。

[2] Deshpande P, Sarawagi S. *Coherent Probabilistic Aggregate Queries on Long-horizon Forecasts*, 2022版。§2限定线性aggregate（均值/趋势）；它不足以单独覆盖max/min，但排除“层级摘要提升底层预测”为新问题：[作者全文](https://arxiv.org/pdf/2111.03394)。

[3] Biswas A, Zambon L, Nespoli L, Corani G. *Nonlinear Probabilistic Forecast Reconciliation*, arXiv:2604.26668v3, 2026-09-17。§3.2 Eq.(7–10)给出非线性summary `f` 下的条件化密度及UKF近似；这是预印本，不夸成已正式发表：[全文](https://arxiv.org/pdf/2604.26668)。
