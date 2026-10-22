# GRN独立Gamma runner：仅代码、purechecks与初始化诊断

本轮交付新 [grn_gamma_development_runner.py](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/grn_gamma_development_runner.py>)。**没有运行正式train入口，没有创建优化器、更新权重或选择epoch，没有打开GRN2026数据。**

源码SHA256：`e8d46b01d5e7e777c5f451688ed4a27173e8a625f81cbb3a08a23ecb1623f70e`。

只执行：

```powershell
python -B grn_gamma_development_runner.py --check-only
python -B grn_gamma_development_runner.py --actual-init-sanity
```

[纯检查结果](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/grn_gamma_development_purechecks.json>) SHA256 `fe8ac448aaf2419d5325f4e4290a0a4ddca3ce75771eff73290206aab711ee83`。

[真实初始化诊断](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/grn_gamma_development_actual_init_sanity.json>) SHA256 `8c86f57d65f9a797446fcc22bc7b2190f8fcf936d16aa7886510e721ca2391cc`。

## 实际初始化检查

只有一个seed11初始化case：fixed-mixture base与同θ冻结的NN0。它们均使用原grid256、log bounds[-8,8]、GH16及真实GRN pre2026数据；没有用validation选择参数。

| 检查 | 结果 |
|---|---|
| GRN F3 | float64，1,581×3；2*a-1、2*z-1、GRN pre25 volumeZ |
| train/validation/te | 1,158 / 228 / 空 |
| validation date/Y | 与既存GRN NPZ逐值exact |
| validation零targets h1/5/22 | 33 / 0 / 0，全部保留 |
| base θ | μ=1，φ=.98，q=.03，κ=1，均有限 |
| base自由参数 / NN自由参数 | 4 / 43，NN θ为buffers冻结 |
| train objective | 1.6171753684416257，有限 |
| NN head weight梯度范数 | 0.011398607359892347 |
| NN head bias梯度范数 | 0.008931296445013562 |
| NN0 vs fixed-mixture | 预测逐值exact，最大误差0 |
| optimizer构造 / steps | false / 0 |
| backward | 1次，仅检查梯度；权重/buffer指纹未变 |

这些值只验证初始化、数据与梯度通路，不是训练成绩、选择结果或独立样本确认。sanity报告的validation指标均标注INIT_ONLY_NOT_SELECTED。

## 私有复用与绑定

- 硬绑定已等价验证的 [Fast源码](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/round21_fast_filter.py>) SHA `99df4c242379eac9c695f660fd770d02433cc8d4f039d202f16d61a2dd824e7e`。
- 使用原R21.train_fit/train_epoch的未修改bytecode，通过FunctionType私有globals替换make_tensors=G.make_tensors、GammaVarianceFilter=Fast及带资产检查的objective/evaluate。没有调用R21.main，也没有改旧ASSETS/global/source。
- 静态检查确认原objective和evaluate都循环model.assets；运行时再次要求model.assets/datasets/T/processes均为真实GRN键。所有模型和train_fit调用显式传入`assets=('GRN',)`，不使用旧默认两ETF tuple。
- 直接加载、哈希绑定既存GRN开发NPZ与audit；`fixed=audit['fixed_preprocessing']`，不重拟hurdle/AR/scale，不加新超参数或N先验。F17沿用standalone GRN pre25标准化，未执行ETF权重transfer。
- SHA同时绑定原GRN builder/source/calibration及新runner；前后核验。所有数据date及label_end严格早于2026，te为空。没有GRN2026读取入口。

## 零target指标处理

实际查到的S17.metrics本来就是raw `log(v)+y/v`和MSE，无log(y)；原R21训练/evaluate使用B.qlike减去已知正EWMA的log常量，也不对y取log。这两条训练/epoch选择路径可以保留y0。

新runner额外提供safe_metrics：

- primary raw QLIKE、knownscale QLIKE及MSE都在每h的全部228行上计算，包括33个h1零target；不floor或删除任何raw y。
- normalized QLIKE的任何组只要存在y0，其`value=null`、`status=undefined_contains_zero_targets`。因此真实GRN h1及总体aggregate为null。
- 可另列positive-only subset，明确n、被排除零数及descriptive_only用途；它不参与primary、epoch选择或百分比改善结论。h5/h22全正的normalized值可以定义，但不能替换含零的整体primary。
- 所有JSON及checkpoint metadata通过allow_nan=False校验；不会把非有限计算悄悄转成一个有利分数。内部结构性masked logits可保留原模型-inf，未写成JSON指标。

## 后续正式训练入口：已准备，本轮未调用

```powershell
python -B grn_gamma_development_runner.py --run-development
```

此入口留给主线程审读后手工执行。HP没有CLI调参开口：max80、patience15、min20、Adam.01、clip2，原sensor NLL .01、κ penalty .005、epoch0纳入选择、改进阈值1e-5。

运行设计是三base（各4参数、seed11）在GRN独立训练，再以selected fixed-mixture θ冻结运行NN三seed11/22/33（各43参数）。每个NN训练前必须与selected fixed-mixture通过逐值exact的NN0 hardgate；不放宽误差或换模型。

输出全部使用独立`grn_gamma_development...` prefix：3个base checkpoint、3个NN checkpoint、validation predictions NPZ、results JSON；已有文件即拒绝。metadata保留sourcehash、HP、fixed preprocessing、selected epoch与全history。NPZ的GRN validation日期、origin、label_end及原始Y必须与既存builder NPZ exact，未来y不floor。预测写入后才构造安全scores。

本轮没有以上任何正式训练checkpoint/results/predictions产物，也没有训练后的GRN2026授权。后续必须先完成2025开发选择，再另行冻结external GRN2026确认协议，之后才能开启该隔离数据。
