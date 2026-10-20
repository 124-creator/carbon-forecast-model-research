# 最小发布范围（静态清单，未批准发布）

## 定位与统计
实际 Git 仓库为独立数据归档下的 carbon-forecast-model-research；原实验工作目录不是 Git 仓库。HEAD `6f39562b81b46dfa6be63504b216b980abaf6682`，分支 `main`。仅清点指定五个范围，不混入其他新实验。完整逐文件列表、哈希、Git 状态和依赖见 PUBLISH_CANDIDATES.json。

Python 统计：候选 57 项，其中未跟踪 57 项；源码 17 项；排除 262 项（目录通配项不等于内部文件总数）；accepted 为 0。这里只生成候选，不执行 git add、commit 或 push。

## 父级审核边界
用户说明已有父核实验，但当前未取得可逐文件对应的父级接纳记录，因此不将任何子交付标为 accepted。报告自己的 passed/通过、receipt.complete 和保存指标一致不能替代父核或独立重训。子实验列为 static_inventory_only_parent_acceptance_unverified；market_parent 相关文件仅列为 parent_execution_receipt_only，本次没有运行模型或重放预测。

pilot 双种子正向门失败且存在负迁移；block_gradient 的一阶下降约束不保证有限步无害，且没有优于普通 SP/PCGrad；joint_adaptation 全部共享提议拒绝，属于 private-only 坍缩，不是成功共享。报告中的“主源检索成功”指访问/索引证据，不是模型成功或论文全文精读；联合适配仍有先实现后补检索的先后限制。

joint_history 原比较同时改变信息集与表示；matched_information 仅在相同 ridge 预算和指定模拟器下比较同步其他市场信息，不能推断实际市场因果收益、显著性或创新。no_shared 结果并未支持跨市场收益。

PyTorch REPORT_ZH.md 是较早合成阶段文档，其“没有执行市场/市场未执行”不能覆盖后续父级 market_parent 执行。父市场 receipt 显示 950 条截止 2021-12-31、451 发行点、rolling40、60 fits；这是开发段结果而非未见测试。SegRNN 两 seed 平均 score=1.046754127601（按已保存 score 做 Python 平均，不重新评分）；score 为相对 RW 的等年/步长损失，超过 1，不得称市场成功。没有读取2022市场数据。

## 依赖与旧 cwd
block_gradient/joint_adaptation 使用同仓 pilot/run_pilot.py；本地模块关系已逐源码解析。NumPy、SciPy、PyTorch 是外部运行依赖，不上传 .venv。torch backbone/market_parent/prefix_loader 本地源码齐全。

joint_history/run.py 明确硬编码旧实验 cwd 的 baseline/benchmark.py；旧 cwd 文件存在=True，仓库同路径存在=False。仓库虽有 original_baseline_source.py 快照，但当前导入并不使用它，因此 matched_information 也间接依赖旧 cwd。快照所有权/许可未明确，先排除，不能声称纯仓库可独立复现。没有为发布改写历史源码。

所有含本机绝对路径的候选先保留为 hold；provenance、环境/资源记录可能暴露本机路径，排除。旧 cwd 同名文件只核对字节，不从那里复制实验结果。

## 许可与最小纳入原则
已实际读取 torch 子目录完整 Apache-2.0 LICENSE，backbone.py 有上游固定 commit 归属和修改说明；只证明这里存在许可文本和说明，不把它扩展为整个仓库许可。其他指定实验没有发现对应本地许可，需父级确认自有源码授权；不根据公开可见推断再发布权。

候选限源码、冻结协议、紧凑聚合表和带限制的结果摘要；带绝对路径、版权未确认或父核未对齐的文件只列 hold/pending。不上传 NPZ、checkpoint、参数、预测、观测、全训练轨迹、日志、论文原件、网页全文、primary/source/search receipts 或 vendor 源码；source 快照也先排除。聚合 CSV/JSON不是训练重放认证；不承诺无原始数据的归档可完整复现。

旧结果、旧摘要和协议未修改。本目录仅提供本次清单及范围说明，由父代理逐路径审核后决定最小暂存集合。
