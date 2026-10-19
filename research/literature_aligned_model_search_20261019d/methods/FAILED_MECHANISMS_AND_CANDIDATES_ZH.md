# 旧失败机制、两项新候选与有限原创性定位

日期：2026-10-19。范围：只读旧协议／方法／报告，加本轮新候选设计与直接近邻阅读。旧分数来自其原报告，不跨不同样本、损失或时间切分直接排名；没有重新拟合旧候选，也没有把旧开发区变成新确认区。本轮模型开发者自行核查近邻，属于内部查新，不是外部独立盲审或完备的新颖性证明。

## 1. 仓库已经失败的具体机制

| 旧工作与来源 | 真正试过的机制 | 失败的范围与仍可保留的事实 |
|---|---|---|
| [cross_market](../../cross_market_innovation_20261017/REPORT_ZH.md) | 共享 GRU 上加入条件矩一致性，配合历史粗化／增强 | BCM 确认 MSE/RW≈0.970975，但同粗化、无条件矩控制≈0.970964 更好；不能把跨市场信息或数据增强的改善归给条件矩。也不能说所有收益均因泄漏。 |
| [structural SMP](../../structural_innovation_20261017b/README.md) | 最新非零涨跌符号＋零变化年龄的 9 状态 hurdle Markov-reward 多期预测 | 开发主分数≈1.323787，差于简单状态／一步训练控制，停止。重复报价不等于没有成交；不能把年龄／符号链换名字再提。 |
| [event_feedback RRC](../../event_feedback_innovation_20261017/README.md) | 利用已成熟前缀误差与当前状态的交互，预测剩余路径教师残差 | RRC≈0.991829，略好 DSOF≈0.993004，但 no-prefix≈0.991627、additive≈0.991680 更好。SOLID 相似检索也未优于近期样本控制。说明特定前缀交互／检索机制未成立。 |
| [auction](../../auction_clearing_innovation_20261018/README.md) | 随机需求清算；修订后的尺度分离版本 | 原 SDMC/RW≈1.225753；FDMC≈1.004985，与 GRU 的配对区间跨 1。尺度分离有局部帮助，随机清算机制未成立。单清算价＋供给量不足以识别完整需求曲线。 |
| [OHLC path](../../ohlc_path_learning_20261019/README.md) | 价格、幅度及路径摘要联合状态学习 | MSE/RW≈1.020229，区间跨 1，停止。OHLC 辅助状态并未提供可归因的均值增量；两个 EUA 作者来源不等于两个独立市场。 |
| [distribution shape](../../distribution_shape_diagnostic_20261017c/README.md) | 条件分布形状、形状＋尺度 | 条件形状 CRPS 比≈0.994667，全局形状≈0.991848；形状＋尺度≈1.017943，尺度控制≈0.994537。不能把波动预测优势说成条件形状有效。 |
| [distribution update](../../distribution_update_learning_20261019/REPORT_ZH.md) | 未来路径控制变量＋偏差预算选择 CDF 更新 | 相同预算控制的平均 CRPS 改善 0.045856%，全来自广州 2021 H20，其余 21/22 单元预测相同。净风险、单一成功单元与推定对称性边界共同限制主张。 |
| [prior transport](../../prior_transport_learning_20261018/README.md) | 共享表征变化时搬运目标 ridge 先验、元学习来源权重 | transport/RW≈1.116561，优于固定先验但差于简单二阶矩控制≈1.070761；学来源权重增量约 0.00000733。来源权重／先验搬运未获独立支持。 |
| [conditional persistence](../../conditional_persistence_20261019b/REPORT_ZH.md) | 零中心收益的三态波动保持／重抽，条件持续性改变尾形，路径似然辅助 CRPS | 条件持续性对固定控制 CRPS +0.183892%，对同容量尺度 +0.258964%，路径辅助也变差。该机制主要改变依赖与尾形，不能再用它替代预测均值。 |
| [forecast geometry](../../forecast_geometry_filter_20261019c/REPORT_ZH.md) | 固定 HMM 中用未来预测几何改变鲁棒观测更新 | 合成相对当前 CDF 等有局部改善，但对验证选择强对照预期分数更差；未来方差／tempered 控制更好。1% 原门存在 oracle 空间不足，必须披露，但删去该门仍没有机制成功。合成不证明真实碳价有效。 |

上述结论只针对相应数据、实现与预算；它们不证明所有状态空间、分解、跨变量信息或碳价均值预测都不可学习。后续可以使用新机制和正当的信息边界，但必须诚实处理已有失败和已经暴露的评价数据。

### 与当前候选尤其接近的旧工作

| 旧机制 | 原结果／边界 | 当前应避开的重复及本轮区别 |
|---|---|---|
| [端点修订筛选](../../decomposition_direction_selection_20261013/PARENT_ACCEPTANCE_ZH.md) | 固定移动平均＋ridge 的分解／修订未胜原价控制；仅有限线性试验 | 不能把固定滤波的重参数化当新特征。当前 A 不预测端点修订标签。 |
| [revision_distribution_candidate](../../recurrent_response_20261014/revision_distribution_candidate/DESIGN_REVIEW_ZH.md) | 成熟分解端点作辅助目标的分布条件化候选；均值资格试验失败，完整 GRU 尚未训练 | 直接覆盖“未来／双边分解教师＋历史学生”的近亲。单纯再做蒸馏不构成新机制，也不能把一次 ridge 资格失败推为所有神经教师无效。 |
| [recurrent_response 主轮](../../recurrent_response_20261014/PROTOCOL.md) | GRU 输出 \(bh+a(1-e^{-h/\tau})\)，固定／全局／条件 \(\tau\)；adaptive≈1.008303，free GRU≈0.975373 | 单纯端点产生指数响应已试过。A 的门作用于每个历史创新，直接改变之后库存；不只是最后预测头选择 \(\tau\)。但指数响应本身仍不是新。 |
| [predictive_decomposition](../../recurrent_response_20261014/predictive_decomposition_model/PROTOCOL.md)／[分离分量](../../recurrent_response_20261014/separated_predictive_decomposition/REPORT_ZH.md) | 状态控制的多速率 EMA，守恒分量后接自由／分离 GRU；候选未胜普通 GRU | A 以价格预测创新为分配对象，守恒后直接用衰减读出均值；不把“多速率＋神经门”独立当贡献。 |
| [prequential_state](../../recurrent_response_20261014/prequential_state_learning/PROTOCOL.md) | 已有 RW＋30 个永久／暂时统计模型参照；GRU 表征＋解析头＋成熟标签预序列学习失败 | 永久／暂时统计模型早在仓库存在。A 必须胜可学习常数分配和同容量终端读出，才有条件分配的独立证据。 |
| [predictive_observer](../../recurrent_response_20261014/predictive_observer_market/REPORT_ZH.md) | RKN 条件观测方差相对常数方差未提供稳定支持；已认识重建捷径 | A 不声称发明 learned gain，不用真实潜状态标签或当期重建训练；仅换监督损失仍不够，需结构位置控制与样本外增量。 |

## 2. 两项候选的选择

最多保留两项，当前只实现 A；B 不自动作为 A 失败后的追加搜索。

### A：历史创新的条件永久／暂时分配——当前实施

完整公式、损失、时点、复杂度、限制与三个对照见 [独立设计文稿](SHOCK_ALLOCATION_DESIGN_ZH.md)，可执行实现见 [innovation_models.py](../source/innovation_models.py)。核心为

\[
e_t=r_t-(\rho-1)^\top c_{t-1}-\beta^\top u_t,\quad
c_t=\rho\odot c_{t-1}+w_\theta(r_{t-1},c_{t-1},u_t,|e_t|)e_t,
\quad \widehat m_h=(\rho^h-1)^\top c_o.
\]

优势是学习对象直接影响未来均值、只需数百参数、有明确的同容量终端门控制、没有未取得的成交指示依赖。代价是限制单个价格创新的趋势延续；它适合检验有限样本中的暂时偏离假说，不是通用最优预测器。真实梯度与数学实现已检查，市场表现和创新成立尚未由本文认证。

### B：已观察外生冲击的剩余延迟响应——仅设计备选

这个方向假设外生变化到达后，部分影响尚未体现在当前目标价格，而未来均值主要是已有外生信号剩余影响的释放。它与 A 的“目标价格偏离以后消退”不同：这里的主要非零均值可以沿既有外生变化继续同向展开，不需要假称重复价格是没有成交。

设已观察、按目标报价时点对齐的外生变化为 \(u_{t,j}\)。低维历史上下文 \(z_{t-1}\) 只用该外生变化到来以前已知信息，网络产生总响应幅度 \(a_j(z_{t-1})\) 和分配到两个已排序时间尺度的 simplex 权重 \(\pi_{j,k}(z_{t-1})\)。幅度可带符号，两个衰减系数为 \(0<\lambda_1<\lambda_2<1\)。令尚未释放的响应库存为

\[
q_{t,j,k}=\lambda_k q_{t-1,j,k}
 +a_j(z_{t-1})\pi_{j,k}(z_{t-1})u_{t,j}.
\]

以当前真实目标价为锚，输出

\[
\widehat m_{o,h}=\sum_{j,k}(1-\lambda_k^h)q_{o,j,k}.
\]

同一旧外生变化对不同未来期限的作用来自**同一累计响应函数的尾部差**，而不是每期限独立头。将 \(q_{o,j,k}\) 展开，可见它等于过去外生变化的几何分布滞后和；所有门为常数时就是标准几何 distributed-lag 模型，不能给这个常数特例改名字。

如果未来外生变化在已知过去下条件均值为零，且幅度／滞后分配只由变化到来前的上下文决定，则未来新进入库存的期望为 0，上式给出既有库存的均值释放。若允许门读取当期外生变化的符号，就必须额外处理条件均值或给出对称性等假设；不能默默沿用零均值结论。外生变量有趋势时，该模型的零新外生假设也可能错设。

训练用与 A 相同的成熟原价多期 MSE。状态数 \(DK\)，网络只用一层宽 16 的历史上下文映射，总运算量随 \(L\{DMK+DK\}+HDK\) 线性增长；D=3、K=2 时状态仅 6 维，适合现有千条级报价。有限样本下先固定两个全局时间尺度并估计小门，比每条变量／每个期限任意滞后曲面节省参数；该节省能否抵消错设需实验验证。

强消融必须同时有：①常数 \(a,\pi\) 的嵌套 distributed-lag；②同参数门只在预测末端对静态库存重新加权的位置对照；③同参数把门作用于总幅度、保持滞后份额不变的控制，用来区分“条件传导强弱”与“条件传导时长”；④相同外生历史的 ARX／Ridge、GRU 与自由期限响应。较小常数臂与其他臂不能冒称完全同容量。

可识别机制检查应使用已知条件滞后但**总长期幅度相同**的两个状态，外生变化具有充分的时间变化且不完全共线；检验恢复的响应质量在快／慢时段怎样分配，以及未来条件均值。负控制应固定滞后仅改变幅度；还需一个没有外生预测信息的过程。幅度缩放与 simplex 约束消除一部分任意分解，已排序不同时间尺度消除模态互换，但高度相关的 Brent／WTI、有限窗口及神经冗余仍阻止无条件识别保证。真实碳价中的估计关联也不能被解释为因果传导。

直接近邻已经包括传统 DLNM、近年的神经 distributed-lag、条件滞后门、连续时间卷积和跨市场响应核。因此 B 的有限候选差异只能是**对已有信号未释放响应的共同多期限尾部读出和位置／幅度控制**；不能主张首次条件滞后、首次外生预测或首次神经传导。鉴于当前 A 已有充分的实际运行条件，本轮不同时加入 B 来扩大挑选空间。

## 3. 最近邻阅读与原创性包含范围

以下区分原文方法、作者正文和出版页摘要；无法取得正文的来源只用来确认其已公布的主题。检索范围不是全数据库完备查新。

| 原始来源 | 本次实际阅读范围与包含关系 | 当前可保留的有限区别 |
|---|---|---|
| Beveridge–Nelson，1981，[出版页](https://www.sciencedirect.com/science/article/pii/0304393281900404)，DOI 10.1016/0304-3932(81)90040-4 | 官方检索摘要确认随机永久／暂时分解；直开正文未取得。不从摘要推断具体训练算法。 | 永久／暂时分解概念不是新；A 只提出条件创新分配与预测约束的受限实现。 |
| Hasbrouck，1993，[RFS 出版页](https://academic.oup.com/rfs/article-abstract/6/1/191/1575874) | 出版页摘要确认证券成交价的随机游走／平稳偏离分解。未取得方法全文。 | 不能把模型暂时库存直接命名为真实定价误差或流动性噪声；本数据没有等价的交易级识别条件。 |
| [DeepState，NeurIPS 2018 原文](https://papers.neurips.cc/paper/8004-deep-state-space-models-for-time-series-forecasting.pdf) | 已读 §3–4 状态与观测式、神经参数映射、似然训练与未来传播。原文使用 RNN 参数化线性高斯 SSM，给出解析／采样预测，且其协变量设定包含预测期可用值。 | 神经状态参数与结构化预测都已有。A 用价格创新的 simplex 分配，当前绝对创新进入门，不需未来能源输入，直接学习原价多期 MSE；这是特定限制，不是比 DeepState 更一般的新框架。 |
| [KalmanNet，IEEE TSP 2022 作者全文](https://www.weizmann.ac.il/math/yonina/sites/math.yonina/files/KalmanNet_Neural_Network_Aided_Kalman_Filtering_for_Partially_Known_Dynamics_0.pdf)，DOI 10.1109/TSP.2022.3158588 | 已读模型、预测／新息更新、RNN 学习 Kalman gain 的动机与方法输入。该版监督训练设定有真实状态标签。 | A 的 \(w_te_t\) 明确属于 learned innovation gain 的宽泛家族。可区分之处是小维度保守分配、固定共享衰减结构和无潜状态标签的未来原价损失；不能说首次学增益，也不能把某一版有状态监督推广为所有 KalmanNet 必须如此。 |
| [RKN，ICML 2019 官方页](https://proceedings.mlr.press/v97/becker19a.html)及仓库旧作者代码审计 | 当前打开官方摘要；旧项目已读取并运行固定 MIT 作者单元，含观测方差、局部线性转移和因子化 Kalman 更新。 | “循环＋不确定性＋增益”并非空白。A 没有学习观测方差、协方差或潜空间编码；结构区别能否带来性能增量由匹配控制决定。 |
| [Smyl，IJF 2020 出版页](https://www.sciencedirect.com/science/article/pii/S0169207019301153)，DOI 10.1016/j.ijforecast.2019.03.017；[Leddam，ICML 2024 官方页](https://proceedings.mlr.press/v235/yu24s.html) | 已核官方摘要；旧仓库另有固定 ES-dRNN／Leddam 代码审计。指数平滑与 RNN 联训、可学习分解均已存在。 | A 必须靠逐次价格创新分配的具体作用成立，不能只靠“可学习分解”或“端到端”作为名称创新。 |
| [Using time-series privileged information，AISTATS 2022](https://proceedings.mlr.press/v151/k-a-karlsson22a.html) | 已核官方摘要：训练使用预测时点与结局之间的时序信息，测试只用基线；线性高斯固定期限下有样本效率结果。全文下载返回错误。 | 训练期未来信息不是自动非法；这一学习设定本身也不是新。其理论不能不加条件套给碳价、双边分解或 A。 |
| [TimeKD，ICDE 2025 作者正文](https://arxiv.org/html/2505.02138v2) | 已读问题设定、§IV teacher／student／相关与特征蒸馏。训练 ground-truth prompts 提供未来表征，学生部署用观察历史。 | 仅把未来分解改为 privileged teacher 并不清除新颖性重叠；应保留合法的信息边界，同时寻找可归因的具体算子差别。 |
| [VMDNet，EUSIPCO 2026 accepted author manuscript](https://arxiv.org/abs/2509.15394) | 已核作者摘要与版本信息：sample-wise VMD、频率嵌入与并行 TCN、双层参数选择；未把摘要当完整复现。 | 逐样本／滚动分解本身已有。用户所说按已发表流程开展实验可以接受，但不能把协议合法性直接当模型原创性。 |
| [Distributed Lag Linear and Non-Linear Models，作者原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3191524/)，以及 [DLNAM 2026 作者预印本](https://arxiv.org/abs/2609.07381)、[AC-GATE 2026 作者预印本](https://arxiv.org/abs/2605.21542) | DLNM 原文入口与摘要已核。后两项本次只有作者 arXiv 检索摘要，直开失败：已分别描述神经 exposure–lag 曲面、条件滞后权重及合成滞后恢复；不声称精读其公式。 | B 不能主张首次神经分布滞后或首次条件滞后门；需进一步全文核查共同期限尾部预测是否已有完全直接实现。 |

## 4. 当前有限结论

A 是一个具体、可训练、与仓库旧固定响应头／平滑后自由读出不相同的候选；其思想仍被永久／暂时分解、innovations state-space 和 learned-gain 文献包含。仅凭公式不同或非线性检验通过，不能宣布已足够创新。一个诚实而可审查的成功表述应是：**受限的条件创新分配在所列碳价开发／确认任务中，相对同容量终端调幅和强同信息均值控制带来稳定增量，并明确指出何种冲击条件下有效。**

现在可以开始按已固定的统一运行器实际检验，而不必等待出现一个全新的数学领域。与此同时，若市场不支持逐次分配机制，保留实现与分析记录即可；不换指标、不隐藏更好的控制、不把旧失败或本轮信息诊断改名为创新成功。
