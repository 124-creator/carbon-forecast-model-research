# 固定刷新试验：没有增量，不给候选背书

同一官方 NeuralForecast 1.7.7 NHITS / DLinear，seed 17/29。没有换基础模型，没有改变网络深度、正则、学习率、300步预算或完整20步MSE与原生mask；仅变更刷新策略和最近窗口支持。每40个观察从零fit，当期价格已经可用。保留最近最多480个至少有一个成熟标签的输入窗口，输入60；不足480时使用全部可用窗口。首发行459索引纳入当前可用价格，所以首fit400窗口，冻结初始fit399窗口。后期480窗口的h1/h5/h20支持480/476/461，不能称为480个完整20标签窗口。原生scaler仅依据每个窗口的历史60值。

只读取既审计950条2021及以前前缀，没有读取2022边界。原价score同451起点（2020年224、2021年227），先平均seed损失，再平均year×h1/5/20 RW相对MSE。

|模型|冻结|刷新|
|---|---:|---:|
|NHITS|1.372253201|1.526508427|
|DLinear|1.088400996|1.122618794|
|RW|1.000000000|1.000000000|

总体均比冻结更差，也输RW；部分year/h局部改善不足以背书。完整year×20步原价MSE/RMSE/MAE及冻结对照在verify_normal.json、scores.json。不是滚动创新，不推出整个模型族无效，也不追加调参救援。MLE及strong历史线性只是历史比较结果，本次没有重新训练、没有加载其模型，不把刷新结果写成这些强基线的复现。

完成12刷新块×2模型×2seed=48次正式fit，每次300更新。额外8次完整未来后缀扰动从零fit（首窗口与成熟480窗口，各模型各seed），参数哈希与相同发行时预测完全相同。全部48个正式fit的原生目标mask及训练内scale未来标签扰动检查通过。normal与-O独立从950前缀重建truth/451起点、复核保存预测与源码哈希，通过。没有加载pickle/checkpoint，没有修改硬链接包源码，没有提交或push。

全部实际fit累计150.516秒，包含初次诊断7.156秒，低于600秒。预算回调每batch检查；异常时保留逐batch日志、已完成块和NaN未完成预测，不补造；本次没有触发timeout。预算机制未人为超时演练。首诊断因为比较40序列批量预测与单序列预测，参数相同但预测未逐位相同，已保存diagnostic_batch_shape_attempt；固定相同推理批量形状后正式从零重跑，未来扰动检查通过。不是删除失败证据或放宽容差。

## 父核验绝对命令（现有文件可直接重放，不重新训练）
```bash
"E:/碳价格/独立数据归档/carbon-forecast-model-research/research/official_nhits_baseline_20261014/.venv/Scripts/python.exe" "E:/碳价格/独立数据归档/carbon-forecast-model-research/research/official_nhits_baseline_20261014/rolling/verify.py"
"E:/碳价格/独立数据归档/carbon-forecast-model-research/research/official_nhits_baseline_20261014/.venv/Scripts/python.exe" -O "E:/碳价格/独立数据归档/carbon-forecast-model-research/research/official_nhits_baseline_20261014/rolling/verify.py"
```
run.py是从零主训练入口，拒绝覆盖既有预测；diagnose_mature480.py是额外480窗口因果复训入口。不要在当前结果上重复执行诊断，否则属于新增训练预算。
