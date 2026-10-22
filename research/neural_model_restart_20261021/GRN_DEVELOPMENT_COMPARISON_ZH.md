# GRN 2025开发：Gamma与全部强均值基准联合审计

**没有相对最强均值预测的实质增量。** Gamma NN ensemble优于同框架fixedmix，但输给固定EWMA22，也输给stationary/integrated MEM；MSE同样弱于这些控制。这里保持全部模型、零目标和原选点，不以差的HAR或positive-only子集包装成功。

## 对齐、零值和评分

全部模型同228个2025-01-02—11-28起点、684个target cells，label最晚2025-12-31。date/Y/origin/label_end/EWMA与冻结G逐元素一致，训练标签全部pre2025成熟。验证y0=[33,0,0]，训练y0=[167,65,17]，全部保留。

primary是所有raw y上的log(v)+y/v与MSE；knownscale只减同一logEWMA。整体normalized QLIKE和h1因零目标为None，未用positive-only分数/百分比挽救；h5/h22的合法normalized值仅在JSON按原组保留。NN先均3seed预测再算loss，不平均seedloss或挑最佳seed。

| 模型 | knownscale QLIKE | raw QLIKE | MSE | NNensemble减该模型QLIKE |
|---|---:|---:|---:|---:|
| global_dense | 1.240380705376 | -8.929059726420 | 1.29130990e-08 | -0.083301471197 |
| shape_dense | 1.232789483252 | -8.936650948543 | 1.38394845e-08 | -0.075710249074 |
| fixed_mixture | 1.164493445397 | -9.004946986399 | 1.39863974e-08 | -0.007414211218 |
| neural_density_seed11 | 1.160999492502 | -9.008440939294 | 1.44661983e-08 | -0.003920258323 |
| neural_density_seed22 | 1.151728209298 | -9.017712222498 | 1.45078479e-08 | +0.005351024881 |
| neural_density_seed33 | 1.160041055256 | -9.009399376540 | 1.42631966e-08 | -0.002961821077 |
| neural_density_ensemble | 1.157079234179 | -9.012361197617 | 1.44087803e-08 | +0.000000000000 |
| EWMA22 | 1.127771300722 | -9.041669131074 | 1.32110664e-08 | +0.029307933457 |
| HAR_geometry | 4.792425769335 | -5.377014662461 | 6.06801514e-03 | -3.635346535156 |
| HGB_geometry | 1.181028378830 | -8.988412052965 | 1.26059359e-08 | -0.023949144651 |
| MEM_stationary | 1.149080347422 | -9.020360084374 | 1.27330912e-08 | +0.007998886757 |
| MEM_integrated | 1.149083927412 | -9.020356504384 | 1.27331540e-08 | +0.007995306767 |
| HGB_Poisson | 1.185907329134 | -8.983533102662 | 1.26945393e-08 | -0.028828094955 |
| HAR_QLIKE | 1.246457650291 | -8.922982781505 | 1.35352664e-08 | -0.089378416112 |

NN−fixedmix=-0.007414211218，但NN−EWMA=+0.029307933457、NN−MEMstationary=+0.007998886757。最优单seed也没有超过这两个均值基准；没有整体normalized改善百分比可报告。

## 选点与训练预算

| Gamma模型 | best epoch / run | 参数 | selected knownscale |
|---|---|---:|---:|
| global_dense | 0/20 | 4 | 1.240380705376 |
| shape_dense | 1/20 | 4 | 1.232789483252 |
| fixed_mixture | 80/80 | 4 | 1.164493445397 |
| neural_density_seed11 | 38/53 | 43 | 1.160999492502 |
| neural_density_seed22 | 59/74 | 43 | 1.151728209298 |
| neural_density_seed33 | 33/48 | 43 | 1.160041055256 |

**fixedmix选中80/80，处于预算cap，不能说已充分收敛。** NN选38/59/33、run53/74/48，三seed theta均逐项等于该fixedmix选中theta。这里不延长训练、不换epoch、不重选seed。三seed均改善fixedmix，但同用2025不是三个独立样本。

## 各期限：同一228行

| h | NN−fixedmix | NN−EWMA22 | NN−MEMstationary | NN MSE | best mean EWMA MSE |
|---|---:|---:|---:|---:|---:|
| 1 | -0.0107446699 | +0.0329326524 | +0.0268934783 | 2.84303056e-08 | 2.72348857e-08 |
| 5 | -0.0158005746 | +0.0372561061 | +0.0310109733 | 1.08481428e-08 | 9.32202495e-09 |
| 22 | +0.0043026108 | +0.0177350419 | -0.0339077914 | 3.94789263e-09 | 3.07628852e-09 |

## 强基准与限制

两个MEM均按训练目标选原两starts，selected优化状态收敛；stationary的rho参数贴近上界，与integrated几乎相同，不能当作独立重复支持。HAR_QLIKE三h优化都收敛且无active beta/intercept边界；HGB两模型保留原超参。原HGB已是Poisson，额外版本只更换为共同pre25 rawP scale。

HAR_geometry原logEPS回归+smear的有限预测表现非常差，MSE达约.006，仍原样列出；不能以击败这个失效控制代替击败EWMA/MEM。零目标只在原HAR内部log变换按既定EPS处理，primary rawY未floor、未丢行。strong结果中的schema.model_fit_performed=False仅描述前置schema验证，不表示随后正式基准没有拟合；fitted_parameters与预测产物已核对。

Gamma固定scale/hurdle和数值方案、稀疏未解析range、ETN报价/风险及碳市场重叠限制仍保留；不把quote-range0称为无交易/真实IV0，不把M当真实交易次数。开发集已重复使用，未作bootstrap/显著性筛选，未打开GRN2026或任何新test。
完整逐组指标、输入前后SHA和优化记录见[比较JSON](<research/neural_model_restart_20261021/GRN_DEVELOPMENT_COMPARISON.json>)。
