# GDEA观察步数风险反馈迁移：固定方案

冻结于任何本轮GDEA风险预测或损失结果生成前。仅机制迁移，所有年份均已见，不构成独立确认测试；不据结果调参。

数据：data/local_only/gdea_close_candidate.csv，SHA256 61b9afaa0784496ff183681bff1c1b49a40bb5b869b91d881756503635476e32。取2018-01-01起原记录，升序独立观察索引；不补日、不填0、不删零量、量冲突或沿价记录。endPrice是交易所模板标示收盘价(元)，可能按挂牌加权均价及低量沿价规则形成，不声称最后成交价或后端全历史算法认证。来源审计称没有重构、平滑或删除；不等于源交易所无沿价处理。

连续性门槛审计：2018–2026记录数依次214,245,244,247,246,247,245,245,171（2026到9月30日），最大日历间隔14,10,18,8,10,11,11,9,11天。唯一>14天是2020-01-23至02-10，18天，位于春节/疫情延长假期区间；没有取得逐日官方休市认证，因此不把它全部解释为已证实休市。2024–26没有异常长缺段，按观察步数定义继续；2018低覆盖保留，不推定连续交易日。

目标：r_t=log(P_t/P_(t-1))，Y_(t,H)=mean(r_(t+1)^2,...,r_(t+H)^2)，H=1,5,22观察步；保留真实零目标。训练与评分排除跨年目标及样本尾部未完成目标；基模型仍每个2024起原点发布预测。2024预热，2025与2026已见诊断。训练从2018起expanding，每22原点重训，严格s+H<=t；GARCH中间逐步过滤状态。

固定五专家EWMA94,EWMA97,HAR_return(smearing),QHAR_return,GARCH11_t；另MA22仅基准，不参加组合。完整复用corrected_risk_experiment.py的回归/GARCH/重训/评分公式；close-only特征是log(mean过去1/5/22步r²+eps)，eps=2018–2023样本r²均值×1e-4；HAR OLS log目标加eps，Duan smearing再减eps；QHAR loglink QLIKE，非截距ridge .01/2；GARCH11零均值Student t，百分收益训练。HAR非正预测处理沿原已审计实现并完整记录；优化失败不得静默跳过。

组合支持任意K（本轮K=5），几何exp(Xw)，simplex，ridge .01/2||w-uniform||²；固定均匀初值SLSQP，maxiter1000,ftol1e-11，失败/可行性/一阶gap>1e-5即停止。

动态：共同最近252个已发布scored_common原点s<t，decay=2^(-(t-s)/22)；完整反馈仅共同池中成熟原点，不回补旧点。Expanding：共同全部历史scored_common原点，decay=1。每种池中至少44个完整成熟反馈后，full/partial/LB同步启动，之前等权。

Partial：m=min(H,t-s)，alpha=m/H，q=sum已知r²/H；目标sum d(alpha*eta+q*exp(-eta))/sum(d*alpha)+ridge，等价response=knownsum/m, importance=d*m/H，归一一次。

LeadBalanced：各j的Z_j=sum已观测该lead原点的d；mass_sj=d_s*1{s+j<=t}/(H*Z_j)，a_s=sum_j mass_sj，q_s=sum_j mass_sj*r²_(s+j)；response=q/a，importance=a，所有Z_j>0，每lead质量1/H。无bandit/无偏性声称。

同时报告五专家等权几何/算术；共6基模型+6在线方法+2固定组合=14模型。H=1所有在线分支均独立求解后验证同池full/partial/LB紧容差一致；随机及实际全成熟窗口校验LB退化为full的objective/gradient；所有反馈组件结束索引<=t；输入哈希不变。输出共同预测、权重、诊断、按年及2025_2026的QLIKE/MSE和NW lag=max(5,H-1)探索性配对差、manifest、短报告。QLIKE只给差值，无百分比；MSE给比值。结果只描述跨实际碳市场观察风险机制，不作为新方法或未见确认。
