# D1 独立选择与指标审核

审核结论：**指标复算通过；固定候选的晋级判定为不通过。** 本次审核没有增加训练、改变候选网格、读取 EEX 2024/2025 确认数据，或把开发集上的结果当作论文发表保证。

## 选择程序及结果

独立选择程序为 `source/independent_selection.py`，SHA256 为 `66562c83a3187740c9d700996bfd843a5ba149ee9215a5b7c1b7c76c10e96855`。它只依据固定的 `provenance/D1_PROTOCOL.json` 编写，在本审核者读取任何 D1 数值、根研究者选择程序或选择输出前，于 2026-10-17 10:20:34 UTC 冻结。协议本身已在实验前冻结；这里记录的是独立实现的冻结，不是另一次外部预注册。

程序对每个配置先算系列×种子×期限的等权分数：5 条价格序列、2 个种子、3 个期限，共 30 格。2022 年分别选择每个模型族，再从指定基线族中选择 2022 年分数最低者。所有这些配置身份原样用于 2023 年，没有以 2023 年成绩重新选择。程序对 CSV 已输出的有限十进制分数采用精确有理数求均值、判定精确同分和阈值，因此没有在晋级边界上追加浮点容差。

| 用途 | 2022 年选定且用于 2023 年的配置 |
|---|---|
| RRC | `full_a0p01_lr0p0003` |
| 总体基线 | `rw` |
| 无前缀消融 | `no_prefix_a0p1_lr0p0003` |
| 仅输出修正消融 | `output_intercept_a0p1` |
| DSOF 族 | `dsof_lr0p0003` |

| 原定晋级要求 | 独立计算的 2023 年结果 | 判定 |
|---|---|---|
| 比选定总体基线至少改善 1%，且分数 < 1 | RRC 0.9918290154，RW 1；改善 0.8170984574% | 不通过 |
| 至少 3/5 条序列优于该基线 | 湖北、上海、EU_REPORTED，共 3 条 | 通过 |
| 比 2022 年独立选出的无前缀、仅输出修正分别至少好 0.0001 | 两个绝对分差分别为 0.0015486445、0.0366560497 | 通过 |
| EEX 分数 < 1 且优于 2022 年选出的 DSOF | EEX RRC 1.0559489721，DSOF 1.0577500247；前者仍 > 1 | 不通过 |
| 所有候选结果有限，完整报告固定网格 | 38 个配置、2,280 格，均通过结构与有限性检查 | 通过 |

总体 `advance=false`。结果保存在 `results/D1/independent_selection/`，包括全部配置的年度和系列分数、各族 2022 年选择，以及 alpha/学习率对应的消融比较。错误旧目标的结果只作为合理性诊断，不计为晋级机制证据。

## 从预测文件独立复算指标

`source/independent_metric_audit.py` 的 SHA256 为 `22134f7e74467f7f1af8191351ca6a0044d4d52c8d9400693e6347d3362ce261`。原实现于 10:16:37 UTC 完成已知信号合成检查，此后未因任何 D1 数值而修改；复制到研究目录时保留了相同字节。

该程序不导入 `rrc_core.py`、训练器或 PyTorch。它从协议批准的开发 CSV 重建价格和日期，读取全部 **380 份预测 NPZ**，逐份检查起点顺序、H20 形状、浮点类型和预测摘要哈希。所有期限共同采用完整 H20 且目标仍在同一年内的起点，再独立复算原价 MSE、RW MSE、两者之比、MAE、方向准确率。

全部 **2,280 个指标格** 与各任务局部表、`ALL_METRICS.csv`、每个任务的配置分数一致，共 **32,300 次已报告值比较通过**。回执为 `results/D1/INDEPENDENT_METRIC_AUDIT.json`。其中附带的跨 2022+2023 的 60 格总体均值只用于核验，不作为新的候选选择或晋级依据。

审核数量中的 380 是保存的配置预测序列数量，不代表 380 个独立市场、独立随机样本或 380 次重新训练。训练执行次数仍应按主协议分别报告联合离线拟合、教师流、学生流和校准流。

## 推断边界

EEX 拍卖价与 EU_REPORTED 连接到同一个 EU ETS 市场体系；5 条价格序列对应 4 个市场体系，不能把它们当作 5 个独立市场样本进行独立同分布的显著性推断。种子、期限和重叠预测标签也不构成额外的独立市场观察。本审核没有计算这类置信区间或 p 值。

本判定执行了预先设定的实际研究筛选门槛。虽然若干消融差异满足门槛，该候选仍未通过总体改善和 EEX 相关性条件，因此不能据这些开发结果打开新确认区、把差异称为已确认创新，或宣称达到发表标准。

## 原样复核命令

在研究目录下，以新的输出路径运行；既有回执或选择目录不会被覆盖：

```bash
python3 -B source/independent_selection.py check --protocol provenance/D1_PROTOCOL.json --output /path/to/new_selection_synthetic.json
python3 -B source/independent_selection.py run --protocol provenance/D1_PROTOCOL.json --output /path/to/new_independent_selection
python3 -B source/independent_metric_audit.py check --output /path/to/new_metric_synthetic.json
python3 -B source/independent_metric_audit.py run --protocol provenance/D1_PROTOCOL.json --output /path/to/new_metric_audit.json
```

选择程序只需完成的聚合表和协议。逐预测审核还需要本地保留的开发 CSV、预测 NPZ 与任务完成回执；这些报价和逐点预测不在公开包中。
