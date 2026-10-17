# 新轮研究边界与条件形状诊断建议

2026-10-17。**建议继续一个有界的必要性实验，不预先命名新算法。** 最有科学价值的问题是：固定预测均值和可见动态尺度后，历史信息是否还能预测未来变化分布的形状。若答案为否，结束这一需求；若为是，再研究超出直接概率头的学习机制。

## 1. 失败机制与禁止复活清单

| 已有研究 | 必须继承的限制 |
|---|---|
| HSC、RUM | 期限竞争与残差效用记忆的固定候选已失败；不可改成概率专家权重、检索或效用门控后重命名。 |
| HD／强GRU | 后验冠军不等于可部署选型；初态／收缩有影响不等于没有训练或全部无信号，不同刷新协议不可直接排名。 |
| NLinear／DLinear／PROCEED | 已有真实作者核心执行；完整漂移适配未胜普通在线与no_drift。固定GLS、常量适配、稳定极点、RLS／Kalman不因换名而创新。[E1] |
| BCM | 相同收缩下，总体最优本来满足塔条件；机制不能识别任意共同偏差或跨期反号。小机制损失不等于独立梯度已被测量；不再靠shrink或λ救旧机制。[E2] |
| RRC／SOLID | 配参去前缀、去交互未支持RRC核心；SOLID距离选择未优于近期成熟样本。作者执行锚与共同GRU适配不是同一benchmark。[E3] |
| SMP | full=1.323787，one_step=1.035092，Markov=1.023790；时钟和状态幅值的必要性未建立。不可只改NLL／特征函数／事后缩放复活概率SMP。[E4] |

普通加权、分位数头、分解和重采样可以作为强基线。不要求每个组件首创，但不能把这些名称本身当作贡献；反例或审核通过也不证明新模型有效。

## 2. 数据和选择暴露不会因换指标清零

现有五条开发序列共 **8,119行**：广州1434、湖北1408、上海1507、EU_REPORTED1515、EEX2255。2022／2023已经多轮用于选型与机制判断。上海原2024／2025确认段已评分；换成CRPS或另一CSV来源，仍不能称未见测试。EEX2024／2025的426条原始记录继续隔离，机械来源解析不等于模型评分。[E5]

五序列不是五个独立市场复制；两个EU ETS序列相关，H20标签重叠，种子、原子和模拟路径不是新增独立观察。旧暴露台账还列出`conditional_distribution_probe_20261013`；本次未重开该旧报告，不声称项目首次研究概率预测。按根任务要求，不扩扫远端pro-*，不读旧C数值、CSV、NPZ或权重，不编辑旧冻结研究。

## 3. 新问题成立，模型新颖性仍须单独证明

令预测测度为

\[
\widehat P_{t,h}=p_t+\sigma_{t,h}Z_{t,h},\quad
\mathbb E_{\widehat F}Z_{t,h}=0,\quad
\operatorname{Var}_{\widehat F}(Z_{t,h})=1.
\]

所有形状模型的解析预测均值都是RW，**MSE应相同**；形状仍可在偏斜、尾部和中心质量上变化。CRPS是适用于有限一阶矩分布的严格适当评分，同均值同方差并不意味着同CRPS。[L1] 因此事前以概率预测为主目标仍符合预测模型研究，MSE作为不同目标的secondary即可，不应把概率收益包装成点预测收益。

先用两个合成世界证伪：只有尺度变化而标准化形状不变时，不应保证形状机制获益；可见状态对应同均值、同方差但不同分布时，正确条件形状应胜过最优静态混合。这个区别证明问题可识别，不证明市场存在所需信号。

这里的标准化变化尚未被证明是真实、独立的“创新过程”。GARCH／FHS已估计动态尺度并抽样标准化残差；DeepNPTS已学习经验分布采样权重；直接单调分位数头属于已有建模。[L2、L3] 本轮只能称**条件形状学习的必要性诊断**。若收益仅来自可学尺度、静态分布拟合或修正已有评分错误，应停止形状创新主张。

## 4. 当前有限实验边界

五个既有开发CSV，2022同年完整H20支持，末80个成熟验证起点与H20 purge；中心固定RW，过去60步初始化EWMA(.94)，基尺度为\(\sigma_t\sqrt h\)。四主臂：无context的可学单调shape、GRU条件shape、固定shape的GRU scale-only、GRU shape+scale。31等权原子逐origin／h精确中心化与单位方差，尺度倍率初始化为1。

固定／滚动过滤经验分位池、normal、t5、Gaussian／t GARCH为对照。经典GARCH-FHS应从一步标准化残差递推路径；根任务已增加这条预测控制并复用Gaussian GARCH参数，不新增fit。最终2048路径压成31原子再以精确CRPS评分，是有限测度近似，不能写成连续GARCH真分布的解析评分。

正式冻结前另加两条作者DeepNPTS损失对照后，当前预算为 **30 NN＋10统计＝40次拟合、65条预测流**；计时另计，不与最初20NN建议混称。作者RPS对照与exact-CRPS校正属于既有代码正确性研究，不是新算法。

主比较必须同时说明：`shape-only`对无context可学shape／滚动经验池，`shape+scale`对同GRU的`scale-only`。若采用1%总体增益，它只是资源推进门槛，不是期刊标准；不能因单一市场、期限、seed或副指标而救分。两组配对未支持时不进入D2，D1通过也只获得继续研究资格。

## 5. 实现与报告必须守住的审计点

机器可读要求见 [AUDIT_REQUIREMENTS.json](AUDIT_REQUIREMENTS.json)，独立合成检查另交付。

- 每个滚动池成员满足`s+20<=t`；目标使用其**原origin的尺度**，不能用目标时点、当前query或整年度波动回算。验证按origin构造已知context。
- 实际测度按其概率计算均值／方差；DeepNPTS不等权时不得调用等权归一化冒充正确。退化须按冻结规则报错，不用variance clamp冒充exact1。31等权原子有\(|z_k|\le\sqrt{30}\)的可检验界。
- 原经验分位点应在gap floor之前检查退化。最小间距会改变重复原子，须称正则化经验近似，不声称保留精确零质量。
- 原价CRPS为\(\sum p_i|a_i-y|-\tfrac12\sum p_ip_j|a_i-a_j|\)。有限测度用V式；训练要乘回原价尺度，再除固定training-only常数，不能只训练标准化CRPS。
- 评分器只接受`(B,H,K)`原子／概率与`(B,H)`标签，概率非负且每格和为1，维度非空、数值有限。覆盖率／宽度不能代替proper score，不混比不同测度下的密度NLL。
- 配对共享输入、初始化、优化和早停；记录实际steps、epoch0／选中epoch、矩误差及全部期限分数。单独统计真实fits与不拟合预测流。
- 边际分布不自动给出一致联合路径，不能据此主张路径风险、经济收益、漂移识别或发表资格。EEX逐点报价、预测和标度不公开。

## 来源和实际访问

[E1] [web_pro报告](../../../carbon_workspace/research/web_pro_execution_20261017/REPORT_ZH.md)。[E2] [BCM公开开发与数学审查](../../../carbon_workspace/research/event_feedback_innovation_20261017/audits/NEXT_DESIGN_AUDIT_ZH.md)。[E3] [RRC／SOLID判断](../../../carbon_workspace/research/event_feedback_innovation_20261017/PUBLISHABILITY_ASSESSMENT_ZH.md)。[E4] [SMP报告](../../../carbon_workspace/research/structural_innovation_20261017b/REPORT_ZH.md)。[E5] [数量台账](../../../carbon_workspace/research/event_feedback_innovation_20261017/provenance/DATA_QUANTITY_SUMMARY.json)、[暴露更新](../../../carbon_workspace/research/event_feedback_innovation_20261017/provenance/METADATA_SCOPE_AND_EXPOSURE.json)。另读RESEARCH_STATE、原prompt及旧暴露台账；BCM只读取非确认方法／开发诊断段。

[L1] Gneiting & Raftery (2007), JASA，[作者全文](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)，§4.2已读；DOI10.1198/016214506000001437。本文使用最小化方向的CRPS。
[L2] [arch官方预测文档](https://arch.readthedocs.io/en/stable/univariate/forecasting.html)，已读递推和bootstrap信息边界，未执行作者软件。
[L3] [DeepNPTS作者全文](https://arxiv.org/html/2312.14657v1)已打开；本审查者未执行作者模型。另见arXiv2306.03782创新表示摘要，未读全文，不以此认证新颖性。
