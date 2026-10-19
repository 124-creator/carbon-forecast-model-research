# 拟公开文件范围核查

**PASS_CURRENT_PROPOSED_PUBLIC_SCOPE**。完整回执为 [PUBLIC_SCOPE_PREFLIGHT.json](PUBLIC_SCOPE_PREFLIGHT.json)，对应 HEAD `ba048cb24189b3e0a869a2e9eaffff2b2a6b8cfc`。本次审查读取 132 个 Git 可见的本研究文件，其中包含 69 个 JSON／压缩 JSON 和 21 个 CSV；核查期间文件哈希未变化。没有执行外部发布、市场拟合或预测评分。

## 本轮明确的范围

| 内容 | 处理与核查结果 |
|---|---|
| 新下载、未声明许可的作者工作簿／CSV／notebook | 留在私有 `/tmp` 数据目录；拟公开目录中未发现对应文件或改名后的同哈希副本。 |
| 广州／湖北逐日期辅助量表 | 私有保存；拟公开部分只有解析／对齐代码、来源哈希、数量和有效性收据。 |
| EWT 逐点预测和观测价格变化 | 已核验私有副本后移除仓库文件；公开保留聚合和迁移收据。 |
| 各实验完整 torch checkpoint 文件与逐点 NPZ | 当前拟公开目录中未发现；运行目录保存完整原件。 |
| 自研候选及结构对照的数值参数 JSON | **主代理已明确授权公开**，供机制与 forward 复核；216 处 `state_dict` 或等价表示被记录为允许内容。不能再笼统声称所有完整数值权重都只在 `/tmp`。 |
| ScienceDB 819 原 XLS、三个市场与九分区 CSV | 依据当前官方 CC BY 4.0 元数据和主代理授权保留公开；这 12 个 CSV 是明确许可的日成交均价基准。 |
| 聚合指标、验证选择、epoch 训练记录、来源文献、哈希与审查收据 | 属于本轮拟公开范围。 |

扫描核对了 10 个已知私有原件／辅助量／逐点文件哈希，并检查文件扩展名、全部 CSV schema 和 JSON 中可能的逐点数值字段。未发现新无许可证作者原价量表或逐点预测路径误入。来源哈希、文件 URL 和汇总统计不是原表副本。

## EWT 迁移的可核实结果

当前私有文件：

`/tmp/carbon-literature-aligned-runs-20261019d/information_diagnostic/PREDICTIONS.csv`

该文件有 3,510 条记录、281,843 bytes。复制前后 SHA-256 均为：

`d21795622bbb037772ff39f63c79b689ada5ad21721740219ca38efa7c660715`

逐点文件先通过 [独立复算](INDEPENDENT_EWT_DIAGNOSTIC_ZH.md)，随后复制并逐字节核验，最后移除本研究结果目录中的副本。[迁移回执](../results/information_diagnostic/PRIVATE_PREDICTIONS_MANIFEST.json) 绑定原审计 JSON 哈希。原源代码与原 JSON 审计回执没有修改；复算命令用 `--predictions` 指向迁移后的私有位置，结果目录 README 和审计说明已更新。

## 旧研究保护与范围时点

本研究的首个 commit 为 `20dee5191a75338812f43e16820e2dfe3a15acbe`，比较基线为其父 commit `230ccf574cc3046c382f18e091e3d64e62b6af29`。从该基线至当前工作区，其他 `research/` 研究目录的跟踪文件改动数为 **0**，其他旧研究目录新增未跟踪文件数也为 0。本次迁移没有触碰任何旧仓库原始数据。

本回执记录上述时点的拟公开文件快照。之后新增的 V3 结果或最终交付文件应在最终打包时增量核查；原快照保留，不覆盖。可用 `audit_public_scope.py --repo ... --study ... --out 新回执路径` 执行同一检查。Git 忽略的运行缓存不属于该拟公开候选列表。
