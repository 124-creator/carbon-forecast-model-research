# GRU历史一步辅助监督基线

真实核read作者AuxiLearn与PSRNN README之后本契约。辅助学习、预测状态已有，非本项目首创；本轮不实现作者方法，也不称PRNN全文已读（原开场提前声称，已撤回，实际出版社/两个arXiv页面均安全阻塞）。普通固定辅助损失只是资格对照，不能将其称新算法。

既有完整GRU1×16隐藏、20h主head保持；新增共享隐藏序列的一维线性辅助head，预测历史窗口内下一差分x[j+1]，j0..58，每训练样本59项。辅助目标完全为当时已观察历史，没有新增未来标签。loss=main20hMSE+lambda*aux59MSE，lambda固定0或.1，2seed17/29，12刷新块，48fit；不选lambda，不比较更多范围。lambda0包含同样参数但auxhead不参与梯度，以保证骨干初始化和main相同；须逐点复现之前GRUfree全部预测，若不一致停止诊断，不放宽容差。

base.arrays同prefix950截至2021、60diff、训练内sd、purge20与最后60内验证；原AdamW.003/.001、clip1、最大200epoch、patience20，选择只由主任务验证MSE决定，不用辅助验证选择。相同最大预算不等于相同实际更新/算力，全部记录epoch。lambda.1的梯度是辅助正则，非额外独立信息量；主评分六year×h1/5/20等权两seed平均loss。完整451起点。

lambda.1 seed17初始future459扰动真实fit、参数/尺度及截至458预测一致。normal/O评分和预算时间验证。若lambda.1不胜GRU不继续元学习救结果；若有效仍只保留已有辅助基线，下一贡献必须区别AuxiLearn等直接近邻，且胜简单快慢强组合。所有目标/预测本地NPZallow_pickleFalse；无2022，无子agent，正式fit预算600秒，逐块保存未完成不冒成功。
