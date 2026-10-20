# 审计说明（不改变训练配置）

- 默认Hermes解释器没有torch。实际发现并使用 research/torch_segmented_backbone_20261014/.venv/Scripts/python.exe 的torch；该环境没有scipy，本实现不依赖scipy。
- 首次核端点测试发现softmax累计的roundoff使k(12)=1.11e-16。训练前改为严格零常量，保留首次失败与通过日志。没有改变核参数化或生成器。
- 所有origin的inverse谱范数用 sqrt(||inverse||_1*||inverse||_infinity) 上界认证，而不错误地单用infinity norm；另按batch worst infinity位置实测谱范数与condition。q=0、最早下降近1的合法核作为逆放大阳性控制。测试60lag危险控制，不给学习模型追加clipping。
- 本实现为了准确梯度检查使用float64和dense59x59 batched triangular solve。墙钟预算与500epochs是否相容必须由实际运行判断；若预算耗尽，已训练方法全部保留、剩余配置标明未运行，不能凭前两个无信号/线性场景宣布年龄证据机制无增量。
- MLE在1199个训练差分上采用未知核与未知方差的exact joint stationary Gaussian likelihood。它是已知的Gaussian线性state-space/有限MA等价部件，不是新机制。其似然下降与validation稳定也不能证明充分优化。
- 小网络逐冲击四特征与原始signed变化，共享4隐单元输出3h，总35参数；与30参数候选接近但不是完全参数/FLOPs匹配。不额外提供隐藏状态或生成器真参数。必须说明一个限制：网络汇聚将年龄≥12的项固定为零，而候选三角逆解仍受更老的历史影响；因此它并非充分利用完整60lag的强直接小网络。本次不能凭击败它声称机制增量，也不事后改网络救结果。
- 边界oracle120仅诊断；零初态回构可逆，不代表恢复正确latent。G5生成器本身有未观察先前状态。工程审计额外展示未消退前缀外冲击造成的latent偏差。
