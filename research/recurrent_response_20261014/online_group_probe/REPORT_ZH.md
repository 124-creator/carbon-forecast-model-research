# 在线真实到达反馈固定先导结果

父助手亲自执行，无子agent。3算法×2seed=6在线rollout，各470真实update/9210同到达标签贡献，451主origins；initial预训练预测完整首块逐位复现。3代表真实未来扰动在线重放（已包含协方差机制启用时段）参数/签发预测一致，normal/O评分/日志核验通过。

主score ordinary .9808062841132927；diagonal .9753440032004521；correlated .9777848644473727。候选非对角相关未胜对角调整，且未胜历史MLE .949793或state .942705（历史参考未重训）。不得称相关信息创新成立，不扩该固定收缩结构，不追加lambda/lr救结果。

correlated两个seed各391次真正非对角更新，条件数最大约63.22/61.90；ordinary梯度clip442/447次，diagonal436/438，correlated313/320。更新幅度及梯度截断行为不同；sameoptimizersteps和labels不是同FLOPs，更不是证明改进来自梯度去相关。下一若研究需同幅度对照，当前无超过simplediagonal理由，不必扩DSOF大型复现救候选。

算法是收缩GLS样式在线加权损失，未完成独立新颖性证明或论文条件。协方差来自实际历史成熟发行误差，不认为它是真实测量误差或可识别潜在信号。C每次仅来自T前的完整误差group，当前group之后入池；未来数据只用于最终评分。

边界实际在线观察p_T并更新过去o<T真实标签后签发T预测，前提报价当日可观察，observedstep不是认证交易日。价格当前快照offline开发，2020/2021已经重复使用，非holdout。没有未来标签伪造、回填oldforecast或2022price读取。未commitpush。只加载本轮自生成numericalNPZ，不pickle。
