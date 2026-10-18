# R2D2：可微精确读出的直接近邻与执行范围

完整阅读方案：Luca Bertinetto, João F. Henriques, Philip H. S. Torr, Andrea Vedaldi. **Meta-learning with differentiable closed-form solvers**, ICLR 2019。

- [作者论文 arXiv v3 全文](https://arxiv.org/html/1805.08136v3)
- [ICLR记录](https://openreview.net/forum?id=HyxnZh0ct7)
- [固定作者代码](https://github.com/bertinetto/r2d2/tree/fc0c13ec991bb9f84395cb12de57cb150ce76f8d)

根任务通过arXiv HTML读完 §§1–5、实验表和附录A–C。Oxford作者PDF请求实际超时；OpenReview的PDF请求返回浏览器验证页。这两条访问不计作成功全文读取，也没有绕过它们。arXiv的独立公开正文读取成功。本方案占本轮第三个完整近邻，前两项是Deep Factors和Auto-Lambda；其他方法的局部摘要或作者主页核查不被表述为已完成全文复核。

## 已有内容与本轮边界

作者的完整方案以神经网络提供表示，在每个少样本任务支持集内解岭回归，再用该任务查询损失训练表示及读出超参数。正文给出primal/dual等价解，以及用Woodbury减少低样本、高表示维数时的求解开销；还研究了基于Newton/IRLS的逻辑回归内层。实验是图像少样本分类。附录讨论对角岭先验、输出校准及简单预训练后读出与适配感知训练的区别。

因此，精确重新拟合目标头、对岭解反传、使用对角可学习正则以及支持/查询式训练，均不是本轮新机制。把查询损失换成碳价平方损失也只是任务适配。与Auto-Lambda的元权重组合必须是强对照，而不是本轮创新声明。

本轮运输规则的差异与局限由自己的矩阵推导给出，详见[方法定义](../METHOD_ZH.md)和[独立理论审查](../TRANSPORTED_PRIOR_THEORY_ZH.md)。其中历史二次型先验仍属于已知二次正则原理；其是否带来必要且无法被已有控制解释的收益，必须由固定实验决定。

## 实际执行到哪一层

固定commit为`fc0c13ec991bb9f84395cb12de57cb150ce76f8d`。原始MIT许可和源码子集保存在[作者目录](../../vendor/r2d2/)。`rr_standard`和`rr_woodbury`原始函数节点按原样加载；为适配现代CPU Torch，在外部兼容层提供等价的旧`gesv`返回接口，避免修改作者函数。原函数对X的原地缩放通过传入clone隔离，Y仍采用作者相同的平方根样本数归一化。

[兼容层](../../source/author_kernel.py)的独立公式比较实际执行了primal、dual及反向梯度，最大误差低于7×10⁻¹⁶。正式学习器每个源片段的岭求解调用这个原始数值核心，并记录逐拟合调用计数。

这不等于重现原图像基准。没有运行原CNN、图像分类标签加载器、CUDA训练封装或原校准层；当前源片段和20维连续回归目标由本项目实现。源代码取得、数值核心执行、任务适配训练、原论文基准重现是不同的证据层级，报告分别标明。
