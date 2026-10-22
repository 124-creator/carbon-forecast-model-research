# 部分反馈向扩展窗学习器迁移：事前固定计划

在保存动态窗口实验的25模型完整结果和中文报告后，独立追加两个模型：PartialFeedback_expanding与LeadBalanced_expanding。已有动态实验仅呈现小幅QLIKE机制信号且有MSE代价、2026强基线失败，不视为模型成功。本轮检验相同模块是否能迁移至已保存的Online_expanding；不改旧实验输出。

只使用原共同评分集中s<t的全部已发布原点；不截252、不衰减。固定原四专家、几何组合、SLSQP解析梯度、.01 ridge。同一全部历史窗口内完整反馈至少44条才启动，两新候选之前均等权。

partial取m=min(H,t-s)已兑现日，响应knownsum/m、质量m/H。lead均衡对每j取Z_j=已兑现该lead的原点数，mass_sj=1/(H*Z_j)，a_s=sum_j mass_sj，q_s=sum_j mass_sj*r²[s+j]，响应q_s/a_s，质量a_s。仅从原始日收益读取s+j<=t；不由完整目标截取或推算。每lead质量=1/H、sum a=1；全m=H时目标/梯度退化full；h1独立优化后与保存Online_expanding逐预测/权重紧容差验证，报告实际最大差。

完整控制直接复用保存的Online_expanding，不重训也不覆盖它。新候选与原25模型合并为27模型共同原点对照，QLIKE/MSE、逐资产2025/2026 h5/h22、强基准及权重均保留。2024仅预热，2026不完整且各资产末端不齐，年度此前已见，不宣称独立确认性效果或新颖性。只运行这两个迁移候选，不试新窗口、衰减或专家集。
