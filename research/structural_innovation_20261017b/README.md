# SMP：有符号 Markov 传播的碳价预测开发实验

**状态：2026-10-17 D1 全部完成，独立指标与选择器核验 PASS；固定候选失败并停止。**

真实执行 150 次市场拟合、160 个预测任务，涉及五份开发序列、8,119 条开发报价记录。SMP 的 2022 开发集主评分为 **1.323787 MSE/RW_MSE**，高于随机游走 1、平滑 Markov 1.023790 和同条件单步训练消融 1.035092。关键门槛未通过，因此 D2、实测 bootstrap、确认预测均为 0；本候选尚未建立可发表论文的方法贡献。

## 先读这些文件

- [完整中文实验报告](REPORT_ZH.md)：创新假设、数据量、模型、比较、实际计数、误差定位与发表判断。
- [方法定义及最近邻归属](methods/METHOD_AND_ATTRIBUTION_ZH.md)：公式、结构闭合、标准组件与精确消融边界。
- [可移植复现说明](REPRODUCE_ZH.md)：数据身份、准备工具、准确命令和后续阶段条件。
- [原字节冻结协议](provenance/PROTOCOL.json)、[执行账本](results/EXPERIMENT_LEDGER.json)、[停止决定](results/STOP_DECISION.json)。
- [全部配置](results/D1_ALL_CONFIGS.json)、[选择结果](results/D1_SELECTION.json)、[每个真实任务](results/D1/jobs/)、[表格](results/tables/)、[图表说明](figures/FIGURES_ZH.md)。
- [独立指标复算](audits/D1_independent/METRIC_AUDIT.json) 与 [选择器一致性](audits/D1_independent/SELECTOR_AGREEMENT.json)。

## 代码与复现入口

训练协议 SHA256：

```text
b10409de072efaba5fce1d570c15e6e0b4128f5234f037c25afeecfd90efc503
```

先验证源码，不读取市场数据：

```bash
python -B source/prepare_reproduction.py check-sources
```

使用五份字节身份一致的开发 CSV 写入数据映射后，准备一个不存在的新工作目录：

```bash
python -B source/prepare_reproduction.py prepare --work-root /absolute/path/NEW_SMP_WORK --data-map /absolute/path/data_map.json
```

准备程序会打印准确的训练、选择和独立审核命令；按 [复现说明](REPRODUCE_ZH.md) 逐条执行。原运行使用 Python 3.12.14、NumPy 2.5.4、PyTorch 2.3.1+cpu、CPU float64。原结果不复制到新工作目录，新的运行不能继承历史审核 PASS。

## 公开边界

这里公开模型和对照源码、完整配置与训练日志、聚合误差和数据指纹。EEX 原价逐行数据、逐起点预测及权重不随本目录分发；数据复原路径和限制均见复现说明。PatchTST 四个作者核心文件与 Apache 许可证保留，公开任务适配不等于作者论文 benchmark 重现。

SMP 是本轮条件性候选简称，不表示 Markov 递推或神经模块已经取得原创性证明。2022/2023 是已暴露开发年份；本轮失败记录不能包装为成功的方法论文。
