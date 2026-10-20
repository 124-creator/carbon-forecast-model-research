# 独立概率指标复算与停止结论

独立复算 **PASS**；固定条件形状必要性实验的晋级门槛 **不通过，停止**。两者含义不同：前者说明保存预测与公布指标、验证轮次和门槛判断一致，后者说明本次预定研究需求没有获得足够支持。本文不把实现正确解释为预测机制成功。

## 冻结时序与核验范围

独立程序 `source/independent_results.py` 于 `2026-10-17T15:47:41.197496+00:00` 冻结，早于本代理第一次读取正式成绩。正式训练此前已经开始；这不是训练开始前冻结审核器的声明。原运行结果完成时间为 `15:47:53.625815 UTC`，独立核验实际开始于 `15:49:26.554222 UTC`、结束于 `15:49:32.231768 UTC`。

程序只依赖 NumPy 与标准库，未导入或调用生产 `run.py`、`scores.py`、`data.py`、模型和统计实现。它实际读取本轮 65 份保存预测 NPZ 及相应公共元数据，没有读取源价格 CSV、模型 checkpoint 或确认分区，也没有重新拟合模型。

| 核验项目 | 实际数量/结果 |
|---|---:|
| family × series 单元 | 13 × 5 = 65 |
| 评价预测起点行，跨全部 family 计 | 13,845 |
| 验证预测起点行，跨全部 family 计 | 5,200 |
| 去重后的 series × 评价起点组合，未重复计 family | 1,065 |
| 评价边际分布，含全部 20 期 | 276,900 |
| 验证边际分布，含全部 20 期 | 104,000 |
| 检查数 | 21,061 |
| 数值比较数 | 1,432,423 |
| 精确比较数 | 747,433 |
| 最大绝对数值差 | 8.526512829121202e-14 |
| NN 拟合 / 统计参数拟合 | 30 / 10 |
| NN 优化步 / 完成训练轮次 | 5,761 / 688 |
| 选择第零轮 checkpoint 的 NN | 2 |

65 条预测流不是 65 次拟合。`garch_fhs` 重用 Gaussian GARCH 的拟合；其余无参数参考也不冒称新拟合。

## 独立计算了什么

对每个评价与验证分布，程序用显式原子双重求和计算精确有限分布 CRPS；用排序后的 lower inverse CDF 计算中位数及 50%、80%、90% 闭区间覆盖和宽度；用概率加权均值计算 MSE、RW MSE 比和最大均值偏差。每个 family 的全部 20 期指标都与公共记录比较，而非只抽查主评分的 1、5、20 期。

共同起点、日期、真值增量和参考尺度逐元素核对；累计标签的重叠恒等式、全 H20 验证成熟条件、公开支持声明、固定尺度臂的分布矩与作者臂的概率下界也被检查。源文件身份只与冻结的 SHA256 声明核对；本核验没有重新读取 CSV，因此不会将内部标签恒等式描述成独立的源价格真值审计。

每个 NN 的公开训练 trace 按 `epoch0` 与固定 `min_delta` 重新回溯，检查选中轮次，并用保存的验证分布重新计算该轮验证分数。程序先从 NPZ 得到自己的聚合和五项门槛，再打开主运行器 `RESULTS.json` 比较。全部选择、计数和门槛一致。

## 主评分与五项门槛

分数为五个序列、三个期限的等权平均：每一格 CRPS 除以该格固定 FHS 的 CRPS。低者较好；固定 FHS 为 1。

| family | 独立复算的主分数 |
|---|---:|
| `fhs_fixed` | 1.000000000 |
| `fhs_rolling` | 1.010456876 |
| `ewma_normal` | 1.024652291 |
| `ewma_t5` | 1.009690044 |
| `garch_normal` | 1.029440292 |
| `garch_t5` | 1.002488724 |
| `garch_fhs` | 1.005485868 |
| `global_shape` | 0.991848009 |
| `conditional_shape` | 0.994666769 |
| `scale_only` | 0.994536684 |
| `shape_scale` | 1.017943136 |
| `deepnpts_rps` | 1.004651946 |
| `deepnpts_crps` | 0.993165695 |

| 冻结门槛 | 实际比较 | 通过 |
|---|---|---|
| 条件形状比固定 FHS、滚动 FHS、全局形状中的最强者改善至少 1% | `0.994666769 > 0.99 × 0.991848009` | 否 |
| 形状加尺度比仅尺度改善至少 1% | `1.017943136 > 0.99 × 0.994536684` | 否 |
| 形状加尺度优于最强统计参考 | `1.017943136 > 1.000000000` | 否 |
| 条件形状在至少 3/5 序列优于全局形状 | 3/5 | 是 |
| 形状加尺度在至少 3/5 序列优于仅尺度 | 2/5 | 否 |

条件形状略优于固定 FHS，但没有超过更强的全局形状对照；联合形状/尺度臂也没有通过所需控制。作者精确 CRPS 臂优于原评分臂，是本次适配任务中的配对诊断结果，不能称作原作者基准的独立重训，更不是新算法或显著性结论。

均值固定为 RW，所有预测均值 MSE 与 RW 相同至舍入误差。该相等是设计约束，不能被当作条件均值预测获益。2022 已在历史开发中使用，EEX/EU 相关且起点重叠；本协议没有 bootstrap、D2 或 C，也没有取得发表或原创认证。

## 复核命令与身份

以下在研究目录、已经准备好依赖的 Python 环境中运行，并接在 `REPRODUCE_ZH.md` 的新运行命令之后。新运行的 `--output fresh_run/results` 和 `--private-output fresh_run/local_only` 必须对应下面相同的两个输入根目录；审核输出文件必须尚不存在。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B source/independent_results.py audit \
  --protocol provenance/PROTOCOL.json \
  --results fresh_run/results \
  --private-output fresh_run/local_only \
  --freeze provenance/INDEPENDENT_RESULTS_FREEZE.json \
  --output fresh_run/results/INDEPENDENT_D1_AUDIT.json
```

`--results` 接收目录，不接收 `RESULTS.json` 文件；`--private-output` 接收直接包含 `<market>/<family>.npz` 的预测根目录。原代码不会自动追加 `D1`。原始实验的公开目录是研究内 `results/D1`，私有目录是工作区的 `scratch/new_round_20261017c/local_only/D1`；这里的 `D1` 来自原运行实参。不要把原布局的子层添加到上面新运行命令生成的 `fresh_run` 布局中。

本研究不公开逐起点 EEX 分布、价格或预测。新的完整重放需要自行获取获准使用的来源并产生新的 NPZ。单独运行 `check --output NEW.json` 只执行解析合成检查；`freeze --synthetic-receipt NEW.json --output NEW_FREEZE.json` 可以冻结同一字节审核器，但不会继承这次实际结果的 PASS。使用现有原 freeze 也只验证审核源身份；上面命令必须对新运行实际执行并产生新的收据。

| 文件 | SHA256 |
|---|---|
| `source/independent_results.py` | `0c62112f357f0cbd8eeb7dfb2884d47d85e4cfe534929a98389dac10c5248b9a` |
| `provenance/INDEPENDENT_RESULTS_SYNTHETIC.json` | `ec58bb5b9956289045f9096ff11680c86fc0f3eeb88f6b1586adff75667f10b6` |
| `provenance/INDEPENDENT_RESULTS_FREEZE.json` | `7b9a399b595b0f55b3bdbe8013024bc0c241a2627b7fadeca9699b0a04c7b97e` |
| `results/INDEPENDENT_D1_AUDIT.json` | `c52cca0f1cbee84d221b90ed5e6b29749b88b463436014bbfb4f3a684f853d13` |

完整机器可读收据保存全部聚合指标、逐文件 SHA256、比较次数和独立 gate。原预测与原聚合没有修改；停止结论按预定门槛保留。
