# 本轮文献核读范围：检索神经预测的直接近邻

本轮没有执行作者模型、安装其仓库、下载其训练集或声称系统性查新完成。下面“方法段落”意味着取得原始HTML并阅读列明章节，不等于通读全部证明/实验。没有重新获取此前受限PDF，也没有把检索空结果当作不存在近邻。

|文献|实际证据与时点|与本候选的关系|
|---|---|---|
|Han等，Retrieval Augmented Time Series Forecasting (RAFT)，ICML 2025|协议前取得arXiv:2505.04163v1 HTML，核读3.1—3.3，重点3.2；官方PMLR摘要身份|历史输入片段及其后续值作为键值对，相关性检索、Top-k和Softmax加权已有。本轮不是RAFT复现，单变量/输入归一化/GRU均不同。|
|RATL: Learning from Retrieved Residuals for Robust Multivariate Time-Series Forecasting，arXiv:2609.03937v1|协议前读概览、冻结base及train-only residual memory；运行中进一步核对检索与Direct规则。预印本，不认证同行评审状态|先冻结预测器，再检索其历史残差并以set-aware block-variable路由组合。残差库、标签可用时点、零修正选项不是我们的首次。本轮共享GRU联合训练和指定重叠矩阵是待检验差异，不是新颖性证明。|
|Retrieval Is Not Enough: Refreshing Memory for Frozen Time-Series Forecasters (FreshCast)，arXiv:2610.07834v1|协议前核读HTML3.1—3.4；2026年10月预印本。未通读理论证明|冻结主预测器、随已观测数据扩充记忆、季节/形状候选和关系核回归已存在。本轮不使用未经认证的未来交易日或季节标签；记忆随观察更新本身也不构成新贡献。|
|Stationarity-Aware Retrieval-Augmented Time Series Forecasting (SARAF)，arXiv:2606.04135v1|协议前只读身份/引言；协议后、正式运行期间进一步阅读3.2—3.4及复杂度说明。不得倒写成协议前完整核读|已有平稳性指导的相关性—多样性取舍和MMR选择。THIN只是简单时间间隔对照，不是完整SARAF。多样性或去冗余本身不能算新。|

## 可说与不可说

目标区间重叠的Gram矩阵是直接代数事实；它用于GLS层并不使GLS成为新算法。真实误差包含信号、局部尺度和参考误差，不能把这个工作矩阵说成已估计的碳价误差协方差。若其效果只能由独立权重、普通Softmax或简单去重解释，就不能放行方法贡献。本轮实际结果未通过这项门槛。

我们没有找到已核全文中与整个本候选逐式相同的实现，但查找有限，而且通用检索、核平滑、已知协方差最小方差权重都已有。不能据缺少完全同名条目宣称首创。论文中的作者改善比例没有用来替代本项目相同目标、时间和预算下的成绩。

原始来源：
- https://proceedings.mlr.press/v267/han25d.html
- https://arxiv.org/html/2505.04163v1
- https://arxiv.org/html/2609.03937v1
- https://arxiv.org/html/2610.07834v1
- https://arxiv.org/html/2606.04135v1

这些地址为已取得的来源登记，不是再分发论文全文。检索期间出现的其他碳价混合网络书目只作为背景线索，未升级为全文证据或本轮竞争基线。
