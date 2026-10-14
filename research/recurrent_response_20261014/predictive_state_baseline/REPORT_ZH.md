# 完整GRU历史一步辅助基线结果

本人实际48fit，两模型×两seed×12blocks，451开发起点。正式fit214.3273424秒不含代表额外causal refit。lambda0同参数主GRU预测逐点复现既有GRUfree所有451点，两seed均通过；normal/O评分、成熟目标/purge和计数检查通过。lambda.1 seed17初始from459未来扰动真实复训参数和尺度一致、截至458预测一致。没有子agent/2022price。

主baseline .9753725253647837；fixedaux .9785570034610523。2020h1略改善而h5/h20恶化，2021三个h略改善；整体不胜baseline。输入窗口内历史一步监督目标没有新未来信息，是否导致长短目标梯度冲突未诊断，不据单元变化归因。相同最大训练预算但选epoch不同，不能称实际更新次数或FLOPs匹配。训练多一个辅助head且计算二次encoder，额外计算披露。

该配置不支持继续扩展普通一步辅助监督，不事后搜索权重救成绩，不用元学习权重换名。没有任何本项目新算法训练/发表资格。基线成立不等同原文PRNN或PSRNN复现；本次是原完整GRU上的固定辅助对照。

来源边界：开场声称PRNN出版社摘要已核读，实际获取安全拒绝，已在聊天及契约撤回；编号1至3都是失败获取，不支持任何题名/方法判断。后来真实读取AuxiLearn作者README关于隐式微分辅助优化。[4] 真实读取PSRNN作者README表明其预测状态循环网络代码身份，未读核心或原论文。[5] 辅助学习/预测状态不是我们首创，不能将失败基础aux调成可学习权重后直接声称创新。

## Sources
[4] https://github.com/AvivNavon/AuxiLearn
[5] https://github.com/cmdowney/psrnn
