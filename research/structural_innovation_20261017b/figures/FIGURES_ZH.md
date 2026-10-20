# D1 公开聚合结果图表说明

本交付只读取本轮公开的 results/ JSON，保留已生成的全部配置、选择结果与任务记录，不运行模型、选择器或新的统计检验。全部图表对应 **2022 年 development、神经模型 seed 17、无置信区间**；统计参考中的 seed=0 是任务标识，不是额外随机重复。D1 固定门槛未通过，图表不改变该停止决定。

## 图 1：主结果与匹配消融

文件：figures/D1_main_and_ablations.png 与同名 SVG。

左图显示八项主要基线、RW 和 SMP-full。八项基线是原生风险 GRU、旧归一化损失 GRU、带相同辅助似然的 GRU、LSTM、PatchTST、全局已选 ridge、局部验证已选 ridge 和 Markov MLE。它们读取 D1_SELECTION.json 中的既定选择；图中使用固定模型顺序，不按分数重新排序。项目原有参考模型完整保留在表中，由于其历史支持不同，不混入这八项主要基线图。

右图显示 full 加六个 SMP 消融，均取 matched_to_smp 的同一输出收缩系数 alpha=0.1。每点是五个序列、h=1/5/20 等权的相对价格 MSE，RW=1，数值越低越好。SMP-full=1.323786771085，单步训练消融=1.035092047339。此处没有把消融优于 full 包装成完整递推机制的实证支持。

## 图 2：EEX 各期限风险及精确误差分解

文件：figures/D1_EEX_horizons_and_decomposition.png 与同名 SVG。

左图显示 EEX 同一组 193 个合法预测起点下的 h=1..20 相对 MSE，横轴是观察步，不是日历天。SMP one-step 使用与 full 匹配的 alpha；其他曲线使用既定已选配置。每条曲线来自已公开的全期限聚合结果，不读取或绘制逐日报价或逐起点预测。

右图来自 D1_FAILURE_DECOMPOSITION.json 中已计算的 SMP-full 聚合量。令 f 为已发行的累计价格变化预测，y 为相同起点和期限的实际累计变化。对同一批起点取平均，有

$$
\frac{\mathbb{E}[(f-y)^2]}{\mathbb{E}[y^2]}
=1+
\underbrace{\frac{\mathbb{E}[f^2]}{\mathbb{E}[y^2]}}_{\text{prediction energy}}
-
\underbrace{\frac{2\mathbb{E}[fy]}{\mathbb{E}[y^2]}}_{\text{twice alignment}}.
$$

预测能量不是预测误差的方差，图中的 twice alignment 项也不是相关系数；两者在图中均除以同期限 RW MSE。较大的正 twice alignment 会抵消部分预测能量。虚线显示 energy−twice alignment，即相对 MSE 超出 RW 的部分；若要还原相对 MSE，需要加上 RW 基准 1。图中只有单位化聚合数值，没有原始价格、尺度或逐点预测。对全部 85 个已选模型×序列记录、1700 个期限单元核对恒等式，最大绝对偏差为 7.55e-15。

这一分解是选择完成后的代数描述，不是碳价变化的因果解释，也不能单独证明某个网络组件造成失败。

## 图 3：五个序列的小型热图

文件：figures/D1_selected_series_heatmap.png 与同名 SVG。

各单元是该序列 h=1/5/20 的等权相对 MSE。显示 full、one-step、原生损失 GRU、Markov MLE 与 PatchTST 的既定已选结果。颜色使用以 RW=1 为中心、上下界互为倒数的对数色标，并标注实际数值；接近 1 的小幅差异保持浅色。EEX 与 EU_REPORTED 都涉及 EU ETS，不能当作统计独立市场；本图没有把市场或期限当成独立重复进行显著性检验。

## 完整导出表

所有表均为 UTF-8 CSV，保留 Python float 的可往返精度，未按图中的四位小数截断。空字段代表不适用或原记录缺少该项，不替换为数值 0。每个表保留来源路径和 SHA256，输出清单记录完整输入身份。

| results/tables 文件 | 数据行数（不含表头） |
| --- | ---: |
| `D1_all_configs.csv` | 33 |
| `D1_all_configs_market_horizons.csv` | 3300 |
| `D1_selected_models.csv` | 17 |
| `D1_selected_models_market_horizons.csv` | 1700 |
| `D1_matched_models.csv` | 12 |
| `D1_matched_models_market_horizons.csv` | 1200 |
| `D1_job_execution.csv` | 160 |
| `D1_job_horizons.csv` | 3200 |
| `D1_selected_decomposition_horizons.csv` | 1700 |

33 个全局配置中，ridge_val 是从已有 ridge 任务借用其各序列局部验证选择后的汇总视图，不是第 161 次拟合。17 个 selected 与 12 个 matched 均原样继承公开选择 JSON。三份 market_horizons 表分别完整保留 all-config、selected 和 matched 的五序列×20期限支持。job_execution 保留原任务计划顺序、160 项完成状态、拟合轮数、选中轮数、梯度步数、参数量、支持和时间；job_horizons 保留全部 160×20 的单位化误差。statistical reference 和 RW 没有神经训练 FIT 记录，相关字段保留为空，梯度步数为 0。

## 重算导出与范围

从研究目录执行：

```bash
python -B source/export_results.py
```

默认写入 results/tables、figures 和 figures/FIGURES_ZH.md，并生成 results/TABLE_FIGURE_EXPORT.json。已存在的交付文件会被拒绝覆盖。若要再次验证导出，在新的输出根目录运行：

```bash
python -B source/export_results.py --output-root /absolute/path/new_export
```

输入仍只来自 --study-root 指向的公开 JSON。脚本不导入训练、数据处理或选择模块，不打开 CSV 输入、local_only、检查点或确认文件，不重新训练、不补实验、不计算新选择或置信区间。图形依赖 NumPy 和 Matplotlib。公开数据 JSON 的指标恒等式检查属于导出一致性验证，不替代已完成的逐起点独立结果审核。
