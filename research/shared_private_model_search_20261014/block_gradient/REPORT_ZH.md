# 时间块共享梯度机制落实

已读取同目录上级 design/DESIGN_ZH.md 与 pilot/run_pilot.py。唯一候选严格采用设计的欧氏最近共同下降方向约束；在原4任务单输出pilot移植为16个任务-时间块约束，而不是新设计。骨干保持原bank averaged tanh MLP，不声称实现GRU版本。340个存储参数，其中168共享；bank averaging具有冗余，不声称相等有效容量。无协方差惩罚。私有更新始终来自自己的任务，未投影，无跨任务梯度输入。

## 数学诊断（不声称首创）
min_{u in conv(G)} u^T d>=0 当且仅当对每行g_i有g_i^T d>=0。因此连续时间块仅定义伪任务集合；候选与这些伪任务上的共同下降QP完全相同，无需重复训练完全相同算法。归一化行不改变可行域。双对偶非负最小二乘求lambda，使d=gmean+A^T lambda；保存可行性与互补性。该形式与GEM投影形式相同，但时间块数据选择/参考梯度不同，不声称整个GEM算法相同。PCGrad逐任务随机次序两两投影后平均，一般不等于一次欧氏最近点QP。这里不把对训练梯度凸包的最坏方向约束称为未来分布鲁棒性；其只等价于当前有限梯度集合的局部鲁棒约束，不等价于任意DRO全目标。
主源检索成功：Yu et al., Gradient Surgery for Multi-Task Learning, arXiv:2001.06782；Lopez-Paz & Ranzato, Gradient Episodic Memory for Continual Learning, NeurIPS 2017, paper f87522788a2be2d171666752f97ddebb。来源入口检索与摘要/公式片段支持上述近邻，不冒充完整全文复现或作者代码认证。

## 固定运行与结果
protocol.json先于训练写入，start.json记录协议/源码SHA256。种子17/29、800 SGD步骤、学习率0.03、连续4块、64训练/64验证/384测试。6方法共12fits完整，87.579秒，900秒预算内；不需要耗尽预算。各任务自己的val每20步选checkpoint；共享方法每任务保存对应完整快照，独立方法也各任务选。共享checkpoint不构成一个统一部署实例，应按任务快照理解。

四任务平均test MSE：
|seed|SP|PCGrad|independent|block|random block|no block|
|17|1.00391750|0.93115753|1.04880283|1.00735054|0.97223509|1.00022707|
|29|0.93322262|0.91581781|0.93876187|0.93381230|0.93352301|0.93323811|

本候选没有超过普通SP或PCGrad的双种子平均成绩；随机块也未证明时序优越性。任务4无共同分量，block test MSE分别1.06917978/1.04881725，独立1.05877309/1.12743975；不可由整体均值宣称私人无预测伤害。私人隔离只保证梯度来源，不保证共享表征无伤害。

verify.py实际执行：12fits、144任务级split MSE、原生成器数据精确重建和预测重建，0错误；私人梯度他任务标签扰动不变；QP反向梯度见证通过。4800约束方法步骤最小归一化内积-2.60e-16，最大互补残差1.73e-16。保存所有块真实before/after_shared_only/after_joint；3034步至少一块共享单独更新风险上升，1543步联合更新至少一块风险上升（1e-10阈值），说明一阶可行性不保证有限步长无害。

## 交付
run.py/verify.py/protocol.json/start.json/receipt.json/results.json/verification.json；2 data NPZ；12 fit NPZ、12 checkpoint NPZ、12 trace JSON；actual_block_losses.json完整4800步。verification.json列出实际可见文件名/大小。全部写指定Gitrepo绝对路径；未读市场、未git提交或push。

限制：统一SGD是机制比较，不是对独立方法调参后的最强上界；未完成全设计CAGrad/伪任务PCGrad扩展，用户此次指定六组已完成。计算成本不同，不声称等FLOPs。SciPy NNLS atol弃用警告未影响结果；验证JSON曾遇numpy int64序列化，修正后重跑通过。