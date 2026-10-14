# 小样本自适应正则：实际代码近邻审查

本轮开场提前声称取得AdaRNN/可微遗忘论文内容，实际没有检索工具输出支持，撤回。实际web_extract三目标均安全拒绝，没有摘要/方法/全文，编号1/2/4只为失败URL登记，不引用。不据失败页面推断条件分布假设，也不绕门禁。

真实有效路径：gh读取AdaReg作者README、完整src/model.py与src/train_eval.py训练相关上下文，固定commit86d70f12ebdef57b4665c75dc5d22147825a7366，APIlicenseMIT。README说明小样本数据依赖正则。[3] 核心实现对参数矩阵两侧用非梯度协方差，通过先feature再task的顺序SVD逆谱截断更新，训练加入对应二次权重惩罚；bias不在此矩阵项内。[5][6] README示意与core更新顺序不完全相同，应以实际代码为准，不能说复制README就是作者demo复现。

对20×16输出权重，已有算法适用的参数几何与我们GRU20hhead相吻合，这是潜在已有强基线，不是新模型。只给head加可学习row/column covariance不能称独立贡献，也不能将以前correlated residual loss失败泛化为这种参数正则必然失败。前者作用在当前残差，后者在权重几何，两者不等于同一训练配置。

本轮独立NumPy转写顺序更新与冻结协方差惩罚进行数值核验：谱界、三个梯度有限差分、零权重正控均通过normal/O，非原作者网络运行，不是市场预测。最大梯度差1.6680727910056703e-9。未来若用该基线需要冻结lambda/协方差刷新周期、内部时间验证、成熟支持和完整训练预算，不能根据开发最优反复搜索。没有在碳价上训练AdaReg。

贡献边界：小样本时怎样确定多期限共享参数的可估计方向，是研究问题；但自适应协方差/经验Bayes/矩阵二次惩罚已有。针对时变预测关系的新规则需在等价性检查中显式区别这种既有正则，也需区别普通fine-tuning/replay，不能仅用成熟标签替换更新输入就宣布创新。当前没有具体独立数学构造，更没有正向预测证据，不给新名字。

本轮停止再运行随意新增的固定模块，取得真实方法近邻和数值审查是有效进展，但不等于找到创新。无新数据/新forecast/未见测试/发表资格，无Gitpush。

## Sources
[3] https://github.com/yaohungt/Adaptive-Regularization-Neural-Network
[5] https://github.com/yaohungt/Adaptive-Regularization-Neural-Network/blob/86d70f12ebdef57b4665c75dc5d22147825a7366/src/model.py
[6] https://github.com/yaohungt/Adaptive-Regularization-Neural-Network/blob/86d70f12ebdef57b4665c75dc5d22147825a7366/src/train_eval.py
