# 年龄/证据路由原型：实际实施与预算截断报告

**结论：design §2 的30参数机制已经实际实现并训练，不是此前阻尼幅度pilot；本次完整合成机制门没有完成，也没有通过。停止当前实现的继续推进，不停止研究，不开放市场。年龄与证据各自的独立增量仍未判定，不能把未运行写成无增量。**

## 实际模型与对照

core.py严格采用5logit、L=12、单调有限核；25参数4→4→1共享路由，四输入为冲击绝对值、年龄、两个已观察延续证据。每个origin重新固定q，然后对角1的三角innovations解与共享多h响应。不增加阻尼、时钟或新的架构。q固定后是已知线性innovations/FIR/shift-register SSM，永久暂态分解、线性观测器均不算创新；新主张仅年龄/证据的估计增量且待证。

写入并hash运行前PROTOCOL_ZH.md：两seed=(7,17)、总训练预算1200秒、Adam500epochs/.001、每h独立成熟、validation checkpoint选择；MLE100epochs/.03。实际fit墙钟合计1199.937秒。**预算预估不足：48个预定训练配置中执行26个，25个用完预定epoch，G2/seed7/constant保留181epoch的截断训练；22个未运行。**全部已执行配置的末状态、选定状态、每epoch loss/gradient和validation记录保留。G0与G1两个seed全部六对照完成；G2只seed7/full及截断constant；G5仅额外边界Gaussian-core诊断，尚未训练。

对照包括同核constant、age-only、evidence-only、35参数小网络、8参数未知有限核/永久暂态测量方差的exact joint Gaussian covariance MLE、RW。MLE没有拿到true核/参数，不把overlap多步target当似然；线性oracle仅diagnostic。小网络虽拿到同一特征，固定汇聚仅年龄<12，未充分利用完整60lag，不能当完全满足同信息强小网络要求；此限制保留，不事后修改网络。

## 已确认结果

- G0 full/RW逐seed均值MSE比率：seed7=0.987471556，seed17=1.027213406。后者恶化2.721%，超过预注册1%无信号门，整体不能通过。
- G1 full/正确已知线性Gaussian oracle比率：0.988822192、1.018847132，均在≤1.05门内；样本测试误差低于oracle不意味着优于Bayes条件期望，更不是创新证据。
- 所有已执行full的validation末5次稳定与训练下降门通过；末梯度分别G0:0.06706/0.35825，G1:0.05417/0.20863，G2seed7:0.08212。**这不是充分优化或全局最优的证明。**
- 已执行full所有origin的inverse谱范数认证上界范围1.52959–1.63470，未超过10；重构仅代数检查。
- 30坐标gradient有限差分最大误差3.17e-11，方向误差6.52e-12；未来suffix扰动预测差和未成熟label扰动训练gradient差均严格0。constant叠加误差5.55e-16；非线性路由实际非叠加；h12/h24平台差0。普通与python -O测试通过。
- 合法近瞬时有限核+q≈0逆放大控制norm=25.3968，证明结构约束并不自动保证inverse稳定。前缀外未消退冲击的zero-state重构仍可精确但latent innovation误差norm=3.8913，不能宣称识别状态。
- G5边界60/120窗Gaussian-core已知参数诊断预测差max约3.81e-6/4.51e-6；因G5额外相关反向冲击不属于独立Gaussian core，它**不是完整G5生成器的精确oracle**，不用于学习门通过。

## 保存与独立核查

REPLAY_VERIFICATION.json逐checkpoint重放、重算35个保存结果行；独立重新生成observables并验证一致，重算测试指标与末gradient；额外REPLAY_predictions恢复预算截断时尚未写出的seed7/G2预测，没有补训练。PROFILE_DIAGNOSTICS.json仅做未选取的kernel/gate局部profile与last/selected预测分歧，不包装为跨初始化辨识。

主要入口：core.py / run.py / audit.py / replay.py；PROTOCOL_ZH.md、INPUT_AUDIT.json、RESULTS.json、ENGINEERING.json、REPLAY_VERIFICATION.json、INCOMPLETE.json、MANIFEST.json；*.pt、*_trace.json、synthetic及predictions npz与全部执行日志。

没有读取市场价格、2022或禁用边界，没有commit/push。所有未完成项在INCOMPLETE/REPLAY_VERIFICATION逐项列出。下一次研究应先做不查看分数的整网格计算成本检查并在训练前确定预算，不能用本轮G0/G1替代两个机制的G2消融结论。
