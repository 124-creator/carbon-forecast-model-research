# 联合适配：已执行结果与限制

## 结论
这一项候选没有带来共享学习：两个 seed 各 1200 次固定共享试探全部被拒绝，候选与只做私有适配、只按 shared 风险判断的消融完全相同。不能把“相对私有适配未增加训练块风险”当作泛化收益；这里的可行性只是通过不更新 shared 获得的。此实现不升级市场实验，不称创新，但不据此停止整体模型探索。

10 次训练完整，36.297 秒，未超过 600 秒上限。每模型 340 参数，数据与 pilot 完全一致。候选/消融/SP_budget 每 seed 各 6000 次梯度调用；PCGrad 4800；private_only 1200。相同梯度调用数不等于相同 FLOPs：候选还需要有限步前向评估。没有测试集挑 step：每任务使用 64:128 validation 每 20 步选择自己的完整模型快照；训练 0:64，测试 128:512。

| 方法 | seed17 测试 MSE | seed29 测试 MSE | 两 seed 平均 |
|---|---:|---:|---:|
| PCGrad | 0.744210 | 0.697681 | 0.720945 |
| SP 同额外梯度预算 | 0.861794 | 0.800206 | 0.831000 |
| 同私有适配，不更新共享 | 0.981727 | 0.881825 | 0.931776 |
| 只 shared 风险判断消融 | 0.981727 | 0.881825 | 0.931776 |
| 联合适配候选 | 0.981727 | 0.881825 | 0.931776 |

本表平均每个 seed 的四任务损失，不平均预测。不能直接与上一轮 block_gradient 的数字作机制差异归因：上一轮 .03/800 steps，这轮预先固定 .04/1200 steps；本轮对照内部同源、同初始化。

## 一个具体机制及等价性
令 p=(s,u)，共享 mask 为 s，私有 mask 为 u（包括私有隐层与任务偏置）。A(p)=p-alpha P_u grad L(p)，alpha=.04。共享提议使用 F(p)=L(A(p)) 的超梯度：

    grad_s F = grad_s L(A(p)) - alpha H_su L(p) grad_u L(A(p)).

代码以两次对称梯度差计算 mixed-Hessian product；导数有限差分单测已通过。给定 z=p-.04 P_s grad F，重新计算 A(z)，将 4任务×4连续训练块的真实 MSE 与 A(p) 比较。全部不增加才接受 A(z)，否则执行 A(p)。不是拿当前 p 作基准，也不是只验证一阶内积。只试一次固定步长，没有回溯/普通 line search，没有 GEM 二次规划伪任务。

这正是一个单步展开的普通 bilevel 目标加有限步可行性门控。保留持久私有参数、私有隐层而非仅最后 head、用相同训练样本作内外目标，所以并不等于完整 episodic MAML 的实验协议；但这些区别没有构成新的学习原理。ANIL 的公开主源摘要已明确只适配任务 head 再进行外循环，不能把“私有适配后评价共享”单独称作首创。[2] PCGrad、GEM 与普通 line search 均为已存在的方法；本实验没有重新命名它们为创新（该常识的原论文正文此次未取得，不声称完成精读）。

## 最近邻与访问边界
ConsMTL（CVPR2025）的搜索索引主源摘录明确将共享和 task-specific 参数放入 bilevel 优化，算法交替处理 shared gradient aggregation 与 private gradient alignment。它是直接最近邻，但索引不能支持本文已逐式证明与 ConsMTL 等价的说法。[1]
Scalable Bilevel Loss Balancing for Multi-Task Learning（2025）把模型参数与 loss router 参数放在内外层，不是本实现的私有响应；它排除了“MTL + bilevel”本身的创新主张。[3]
Progressive Task-Specific Adaptation 的主源摘要说明早期共享 adapter、后期任务特定 adapter，并按梯度相似度分配共享；检索返回 v2 于 2026-04-25 修订及 AISTATS2026 注释，仅作为检索记录，不声称这是所有最新邻居。[4]
Google Research 对 TAG 的作者介绍说明先试一次任务梯度更新，再评价它对其他任务损失的作用。有限更新后评价收益也并非这里首创。[5]

直接主源抓取全部被安全边界拒绝（private/internal address），已保留 PRIMARY_ACTUAL.json、PRIMARY_ADDITIONAL_ACTUAL.json、OPENREVIEW_ACCESS.json，未用 curl/browser 绕过。实际可读证据只是主源搜索索引/摘要摘录，不是取得论文 PDF 全文。最初取得主源失败后，本地 pilot 已执行，后续才完成成功的邻居检索；因此“先完成主源最近邻再实现”的先后要求没有完全满足，不能把本轮当作研究前置条件已通过。最近邻检索仍需父代理在获准且可用的读取路径下补足。

## 实际有限步与验证
actual_finite_step.json 保存全部 4800 条提议/消融记录，每条包括 private-only before、trial_after、actual_after、gate_before、gate_after。两个 seed 的联合候选各 1200 提议至少有一块风险高于 private-only；全部拒绝后 actual_after 等于 private-only before。该数量是“至少一块增加的步数”，不是均值上升次数，也不是原始 p 到最终 p 的安全保证。
verify.py 重新生成相同合成数据、逐步执行候选与消融、读取全部已保存预测及标签重新评分。正常及 python -O：10 fits、4800 actual records，最大重放误差 0.0。单测在实现前缺失 run.py 时失败，后续正常和 -O 均通过。无市场读取、无 commit、无 push，未更改旧实验。

父代理源码验证入口：本目录 run.py、test_joint.py、verify.py；结果入口 results.json、receipt.json、VERIFY.json、VERIFY_OPTIMIZED.json；精确 data 与 fit 均为不含 pickle 的 npz。日志和 source receipts 是本次实际输出，不是模拟回复。

## Sources

[1] https://openaccess.thecvf.com/content/CVPR2025/papers/Qin_Towards_Consistent_Multi-Task_Learning_Unlocking_the_Potential_of_Task-Specific_Parameters_CVPR_2025_paper.pdf — Towards Consistent Multi-Task Learning: Unlocking the Potential of Task-Specific Parameters
[2] https://openreview.net/forum?id=rkgMkCEtPB — Rapid Learning or Feature Reuse? Towards ...
[3] https://arxiv.org/html/2502.08585v1 — Scalable Bilevel Loss Balancing for Multi-Task Learning
[4] https://arxiv.org/abs/2509.19602 — [2509.19602] Parameter-Efficient Multi-Task Learning via Progressive Task-Specific Adaptation
[5] https://blog.research.google/2021/10/deciding-which-tasks-should-train.html — Deciding Which Tasks Should Train Together in Multi-Task Neural Networks
