# GRN standalone 开发数据 builder 交付

已完成独立 [GRN builder](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/grn_range_development_builder.py>)、[开发NPZ](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/grn_range_development.npz>)、[开发audit](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/grn_range_development_audit.json>)和[纯检查记录](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/grn_range_development_purechecks.json>)。

本轮只构造数据及固定预处理，没有实例化/训练Gamma或NN，没有生成预测或模型评分，没有打开GRN隔离2026 raw、2026 labels或scores。数据读取严格限定为已批准的1,581行pre2026快照；另只读取Brownian模拟校准JSON及GRN身份/历史split元数据。旧源码/模块ASSETS/SOURCES、原CSV、模型和报告均未改动。

## API

```python
import grn_range_development_builder as G

datasets, T, audit = G.build_pair(development_only=True)
D = datasets['GRN']
fixed = audit['fixed_preprocessing']
# 以上没有训练或预测。真实key为GRN，不是KRBN别名。
```

该API故意不开放test或任意CSV参数。`development_only=False`会拒绝；不承接原HB的`check_npz=True`（GRN尚无旧R16预测归档可比），未来私有runner应明确调用上述新builder，再执行其独立建模门。

其他接口为 `make_tensors(datasets)`、`fixed_preprocessing(datasets)`、纯内存 `construct(frame,bundle)`。后者供独立检查使用，拒绝所有2026行和缺失/非正OHLC，不做日期填补或删行。

实际执行命令：

```powershell
python -B grn_range_development_builder.py --check-only
python -B grn_range_development_builder.py --build-development
```

输出exclusive；再次运行须使用新的`--tag grn_range_development_...`，不会覆盖旧产物。

## 数据范围与成熟期

| 项目 | 结果 |
|---|---|
| 原始观测 | 2019-09-18—2025-12-31，共1,581行 |
| cut / pre2025行数 | 1,331 |
| train origin | 1,158；2020-02-11—2024-11-27 |
| train h22末 | 最晚2024-12-31，严格早于2025 |
| validation origin | 228；2025-01-02—2025-11-28 |
| validation h22末 | 最晚2025-12-31，严格早于2026 |
| te | 空数组，test_cut=1,581 |
| bank | 1,022行；row>=21、pre2025、resolvable-range |

保留全部1,581条观测。origins按旧HB规则：从index100开始、h22共同成熟、12个context全部有限，跨2024/2025边界22个origin不入train/val。eligible区域另有51个origin因context非有限而不入监督样本；这是原HB origin规则，不是删除原始bar，其proxy仍参与后续合法targets及历史状态。

目标严格为后续1/5/22条观测的**原始Parkinson proxy均值**，包括真实零range，不是只数正range日，也不称真实QV。训练y在h1/5/22分别有167/65/17个零；validation分别33/0/0。原始proxy的342个零逐值保留，没有floor、填补或用log placeholder替换。

HB/S17兼容字段包括date、origin/split/tr/va/te、label_dates、y、observed/zero_range、proxy_raw、log_proxy、even/F16/F17、cell/cell_b/cell_R、context12、baseline_x、ewma及ewma22_all、bank_index、initial_mu/global_b/global_R等。T为GRN真实键，具有F17、proxy_raw及原B.tensors所需字段。

## 零range、volume和缺失的区分

- `quote_H_equals_L`共342行，pre2025为302、2025为40。其中**240行volume>0**；零quote range不能称为没交易或零真实波动。
- `raw_volume_zero`为102行，全部在pre2025；原始volume缺失为0。
- `first21`单独标记21行；volume logratio无效/特征填补为117行，各mask可以重叠，不能简单相加。
- `volume_logratio_raw`保留原始 log(V/rolling22mean)，无效值保留为非有限；仅F17中的该特征按固定GRN pre2025 mean填入后标准化为0。missing volume、zero volume和first21有各自标志。
- 原始OHLC/adj_close缺失、零或负值均直接拒绝，绝不补0或删除后冒充可用行。当前数据没有此类缺失。
- raw P0/proxy不floor；log/context/EWMA只按既有HB EPS约定处理，`observed=false`行的log_proxy=0是既有占位约定，必须和raw proxy及mask一起读。

## Standalone标准化和固定preprocessing

**这不是原两ETF的混合normalizer。**F17使用GRN自身pre2025中“正resolvable-range且volume logratio有限”的1,022行统计：mean `-0.26638669173617463`，std `0.9076886165532762`。元数据保留旧接口`pooled_mean/std`键名以兼容HB，但明确 `scope=STANDALONE_GRN_PRE2025`、`asset_weight=1.0`、`original_two_ETF_pooled_stats_used=false`。

F16按原B16的mean/std+1e-5约定保留，F17几何按固定2*a-1、2*z-1。bank仅取GRN pre2025、row>=21且resolvable-range行。

R21相同的hurdle和volume AR算法只在GRN pre2025做固定预处理，不是Gamma/NN训练：scale均值包括302个原始零proxy；hurdle使用1,331行current零range标志，zeros=302，intercept `-1.3029315587044947`、slope `-0.3712702584930811`；AR也只用这1,331个pre25 F17 volumeZ。完整参数、行范围和Newton预处理记录见audit。

验证集高低价及volume大幅扰动后，normalizer/F16统计、hurdle/AR/scale、bank、pre25 F17及训练targets逐值不变。没有validation选择或2026读取。

未来**独立GRN拟合**可以明确使用这套standalone统计。未来**原两ETF checkpoint transfer**若要求原encoder输入尺度，则必须另行声明并构造冻结原normalizer的映射；不能把本套F17直接称成原两ETF pooled输入或静默替换normalizer。本轮没有执行任何transfer。

## R21后续私有wrapper接口缺口

本builder通过私有函数globals复用S17.install_features与R21.preprocessing/R20 volume AR；没有改原模块全局。后续建模方仍需处理：

1. R21.make_tensors→S17.make_tensors的ASSETS硬编码，应使用本builder的generic make_tensors或私有作用域版本。
2. R21.preprocessing→R20.fit_volume_ar的ASSETS硬编码，应使用本builder.fixed_preprocessing或audit中已固定参数。
3. R21.run_development/ensemble/evaluate/main相关循环及source_hashes仍绑定原两ETF；私有runner必须绑定GRN builder的source allowlist，不能继承原两ETF CSV来源清单。
4. GammaVarianceFilter及train_fit定义时的默认assets已经绑定原两ETF；即使克隆函数globals，也不能依赖旧默认值。后续所有实例化/训练调用应明确传入`assets=('GRN',)`，并在私有wrapper中调整调用；本轮不修改这些模型源码。
5. 原HB `check_npz=True`针对原ETF R16归档，并无GRN历史模型归档。不能创建假归档过门，应以本builder的data invariants作为GRN新开发数据验收。

## 来源及产品边界

唯一市场数据源是 [GRN pre2026 CSV](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/data_acquisition_round13/GRN_ETN_USD_available_history_20190918_20251231_round17.csv>)，SHA256 `52cba5fd28beb71995ce4b0092cd7b0b41ece97aa1b28c0d369d90cd04c57e9b`。起始数据晚于2019-09-09首售/预期09-10挂牌，缺口没有填。

身份为Barclays Bank PLC的iPath Series B Carbon ETN，CUSIP06747C322、ISINUS06747C3227、USD，含发行人信用/bail-in风险。2019条款主要敞口为EU碳期货，与已有产品底层风险重叠，不能称独立宏观市场。

[供应商split记录](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/data_acquisition_round13/GRN_corporate_actions_round17_vendor_unmodified.json>)的2021-06-04 numerator1/denominator5原样进入audit；不人工重写OHLC或volume。供应商历史复权/量单位与实际交易口径尚未独立认证。

源码SHA256：`78e55425a0e6a9a04476dbb4dbaa2cb0a1f210e268c3d8a6cfb6ee18f085eb12`。

开发NPZ SHA256：`acd4a33c31a49c62563a7eae2dcaa236e27a1a0432c3dab42f015d8969d47a3f`。

Audit SHA256：`4d12814c03a0202b63ba1045005806c3d231f952e2427a15acc5e831cf603e10`。
