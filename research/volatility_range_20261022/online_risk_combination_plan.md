# 在线风险组合机制探测计划

固定复用 normalized_comparison_predictions_common.csv 的18模型共同原点，仅选择四位专家 EWMA97、HAR_return、QHAR_joint、QHAR_liquidity_plus_scale。原脚本、原预测和原始数据保持只读。

在每资产、期限、原点t，只允许原scored_common=True且s+h<=t的历史原点反馈进入权重拟合。目标为凸组合log forecast对应的加权平均QLIKE，加0.01/2乘权重偏离四专家等权的平方和，约束非负且和为1；每天SLSQP及解析梯度，失败记录并停止，不删除原点或隐式替代。

预先固定动态版与PartialFeedback_dynamic共享同一个s<t且原scored_common=True的最近252个已发布预测原点窗口，按原点交易日年龄半衰期22。full只用其中s+h<=t的完整反馈，不向更早原点补满；partial使用m=min(h,t-s)>=1个已发生收益分量，目标贡献为decay*[(m/h)*log f+knownsum/(h*f)]，除以sum(decay*m/h)，再加同一ridge。knownsum直接从原始数据构造的r²读取已发生日，禁止由未来聚合目标反推。两者只有当共享窗口完整反馈>=44才优化，否则均等权。h=1逐行断言启动、权重、预测完全相同；h=5/22以partial vs full为主要机制比较。扩展版全部成熟反馈等权。不足44条时用等权。另设固定等权几何、固定等权算术，以及2025首个原点使用2024全部已成熟共同反馈拟合、之后固定的Frozen2024。Frozen2024在2024仅为等权预热，不使用2024未来反馈。

所有原共同原点均保留评分。2024明确为warmup/exploration，仅2025和2026用于本次机制判断；2026为现有部分年度。比较动态版与扩展、等权、冻结和四专家的QLIKE差、MSE、权重；记录训练反馈数、最新标签结束索引、半衰期、失败和无未来断言。

生成独立online_risk_combination.py与online_前缀预测、权重、诊断、年度/全期比较、manifest和中文复盘。保存输入与脚本hash。此为已知online aggregation家族内的机制探测，不宣称新颖；结果不佳也不调整窗口、半衰期、罚项或专家名单。

事前追加：在首次输入检查发现最大3.39e-21的目标末位序列化差异、尚未发生任何权重拟合时，按父任务新指示加入LeadBalanced_partial，不依结果选择报告。共享窗口、启动、四专家、衰减和ridge不变。对j=1..H单独计算Z_j=sum_{s:s+j<=t}decay_s，a_s=(1/H)sum_{j<=m_s}decay_s/Z_j，q_s=(1/H)sum_{j<=m_s}(decay_s/Z_j)*r²[s+j]。拟合响应q_s/a_s、importance=a_s。断言sum a=1、每个lead总质量1/H、h1三者预测权重完全相同；全部m=H时退化full。此仅分离提前反馈与短lead偏重，非平稳下不宣称无偏。新组合共7个，机制表11模型，追加全部原18模型形成25模型强基线表；原机制表保留。
