# 隐藏适配直接近邻与等价性审查

真实gh代码核读范围：作者README、adapt_z.py中adapter构造/forward、val/online、checkpoint加载上下文。commitf5b70e1127daeb0e9e07b72bf6f4c8393776bb3b。[1][2] adapter用当前feature和历史gradient生成z，再传回base模型。在线部分显示delayed batch选取；这不是完整loader、全骨干数据成熟性认证。发现torch.load调用但没有执行源码或下载预训练。LICENSE API404，未确认再分发，不推作者代码。作者paper正文未本轮实际核读，不声称发表身份/原论文结果。

本轮排除本项目一个常见假创新：最终GRUhidden16维加适配z而head仍为线性20h，则W(h+z)+b与Wh+b+Wz严格相同。z可依赖原历史和成熟反馈，恒等式仍成立。梯度也相同，说明不仅推理函数相同，若目标及初始化参数一致训练局部导数相同。输出修正处于W列空间，有低秩约束，不是一般全20维残差，但匹配同低秩条件残差模型可完全复现。

实际double tensor非线性adapter例检查：最大前向差4.440892098500626e-15；两个adapter参数梯度最大差6.661338147750939e-16；rank16。normal/O通过。随机张量只是代数验证，不是碳价训练或性能证据，也不是ADAPT-Z作者网络复现。不能将这个final-head特例泛化为所有深层/内部适配等价。

研究决策：不实现“在GRU最后加隐藏适配”作为我们的核心新模型，不将普通参数微调改名认定创新。真正的新贡献若在学习动力学上，需要改变如何估计状态转移/共享状态，而非只最终输出校正；改变早期状态位置本身也不是首创，需直接邻近方法与原理差异、同预算强对照、可学习正控和真实市场证据。

现在尚无可放行的新数学机制。直接近邻已更具体，减少盲目训练，但不是完成目标。与之前单纯小窗口/aux固定试验不同，本轮先真实源码核验再数值审查，没有新增市场训练/数据、没有子agent、没有创新或发表声明、未push。

## Sources
[1] https://github.com/xiannanhuang/ADAPT-Z/blob/f5b70e1127daeb0e9e07b72bf6f4c8393776bb3b/adapt_z.py
[2] https://github.com/xiannanhuang/ADAPT-Z/blob/f5b70e1127daeb0e9e07b72bf6f4c8393776bb3b/README.md
