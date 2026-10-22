# R22 v1预测路径开发：独立只读审计

**没有相对最强均值预测的实质增量，也没有看到coupling本身的开发优势。** coupled与independent几乎相同且略差，equal137参数direct control更好；旧R21 NN、raw KF和强GRU仍更强。v1按原选点/原center约束报告，不用未训练v2辩护。

## 数据、评分与全部强基准

两asset各228共同origin，2025-01-02—11-28，label至2025-12-31；date/Y/origin/label_end与R21、旧2025 replay逐元素相同。所有rawY>0，本报告未打开任何2026、CSV或GRN数据；未fit、模型前向、bootstrap或重选。每asset×h等权QLIKE，ensemble先均三seed预测后评分。

| 模型 | knownscale QLIKE | normalized QLIKE | MSE | coupled减该模型QLIKE |
|---|---:|---:|---:|---:|
| coupled_path_ensemble | 1.081353933964 | 0.367461354275 | 2.53407356e-08 | +0.000000000000 |
| independent_path_ensemble | 1.081352946921 | 0.367460367232 | 2.53406447e-08 | +0.000000987043 |
| direct_marked_control_ensemble | 1.080811309399 | 0.366918729710 | 2.51869511e-08 | +0.000542624565 |
| R21_global | 1.082293275204 | 0.368400695515 | 2.55732343e-08 | -0.000939341240 |
| R21_shape | 1.081230687343 | 0.367338107653 | 2.57411556e-08 | +0.000123246621 |
| R21_fixedmix | 1.082262726828 | 0.368370147138 | 2.59413781e-08 | -0.000908792864 |
| R21_NN_ensemble | 1.080558645999 | 0.366666066310 | 2.57524222e-08 | +0.000795287965 |
| EWMA22 | 1.141317770970 | 0.427425191281 | 2.61191560e-08 | -0.059963837006 |
| MEM_stationary | 1.090458714432 | 0.376566134743 | 2.49413474e-08 | -0.009104780468 |
| MEM_integrated | 1.101562383060 | 0.387669803371 | 2.71975142e-08 | -0.020208449096 |
| R16_raw | 1.068306284657 | 0.354413704968 | 2.37204897e-08 | +0.013047649307 |
| R16_frozen | 1.070153583725 | 0.356261004035 | 2.37848191e-08 | +0.011200350239 |
| HAR_geometry | 1.079094685930 | 0.365202106241 | 2.86899944e-08 | +0.002259248034 |
| HGB_geometry | 1.136205293396 | 0.422312713706 | 2.55009733e-08 | -0.054851359432 |
| GK_global_KF | 1.076767865908 | 0.362875286218 | 2.38916219e-08 | +0.004586068056 |
| RS_global_KF | 1.081253958889 | 0.367361379200 | 2.40029250e-08 | +0.000099975075 |
| C18_GRU_ensemble | 1.057260925049 | 0.343368345359 | 2.29524638e-08 | +0.024093008915 |
| R20_continued_ensemble | 1.055157524077 | 0.341264944387 | 2.27589111e-08 | +0.026196409887 |
| R20_clock_only_ensemble | 1.054758366396 | 0.340865786707 | 2.27342095e-08 | +0.026595567568 |
| R20_density_only_ensemble | 1.055160965833 | 0.341268386143 | 2.27594822e-08 | +0.026192968131 |
| R20_joint_ensemble | 1.054757767641 | 0.340865187952 | 2.27343873e-08 | +0.026596166323 |

coupled−independent=+0.000000987043；coupled−direct=+0.000542624565；仅相对globalGamma gain=0.000939341240（normalized改善0.254978%）。它没有超过已有Gamma NN，更没有超过更强raw/GRU。

## 三seed、参数和冻结backbone

| seed | coupled | independent | direct | coupled−independent | coupled−direct |
|---|---:|---:|---:|---:|---:|
| 11 | 1.081912684642 | 1.081912425995 | 1.082237755357 | +0.000000258648 | -0.000325070714 |
| 22 | 1.081510829705 | 1.081508914017 | 1.080931477471 | +0.000001915688 | +0.000579352235 |
| 33 | 1.080649286033 | 1.080648489433 | 1.079661301580 | +0.000000796600 | +0.000987984453 |

coupled对global三seed点方向均好，但对independent三seed均微差，对direct只有seed11好。重复使用的2025不是独立确认，seed也不是独立市场样本。

| mode/seed | best/run | trainable |
|---|---|---:|
| coupled_path/11 | 13/28 | 137 |
| coupled_path/22 | 13/28 | 137 |
| coupled_path/33 | 17/32 | 137 |
| independent_path/11 | 13/28 | 110 |
| independent_path/22 | 13/28 | 110 |
| independent_path/33 | 17/32 | 110 |
| direct_marked_control/11 | 3/20 | 137 |
| direct_marked_control/22 | 7/22 | 137 |
| direct_marked_control/33 | 12/27 | 137 |

参数实际是coupled137、independent110、direct137：6→8→9或6→8→6输出。每asset(mu,phi,q,kappa)等于原R21_global，原checkpoint字节SHA匹配；theta不参与head优化。原训练源码还逐buffer检查cache未变，本审计没有重新加载权重/重放模型。

## curve与EqSZ/covariance核对

每日正曲线的前1/5/22日均值复原保存预测；ensemble daily/moments/prediction均为component均值。以存档R21 global mean/active/curve重构zero-anchor比例，最大相对误差<1e-10；这只是存档数组恒等式，未执行F-K/滤波。

逐seed核EqSZ=EqS*EqZ+Cov；independent/direct组件Cov为0，coupled有小的非零模型Cov。ensemble总Cov=平均component Cov+Cov_seed(component means)：independent/direct混合ensemble也会有Cov，不能据此称其拥有path coupling。

| mode/asset ensemble | totalCov mean | within mean | between-seed mean |
|---|---:|---:|---:|
| coupled_path/KRBN | -1.1632329434e-08 | -4.2737118582e-09 | -7.3586175785e-09 |
| coupled_path/KCCA | -7.9857787998e-09 | -3.3717498290e-09 | -4.6140289709e-09 |
| independent_path/KRBN | -7.2353201153e-09 | +0.0000000000e+00 | -7.2353201156e-09 |
| independent_path/KCCA | -4.6119333725e-09 | +0.0000000000e+00 | -4.6119333742e-09 |
| direct_marked_control/KRBN | -1.5544118827e-07 | +0.0000000000e+00 | -1.5544118827e-07 |
| direct_marked_control/KCCA | -8.3969352724e-08 | +0.0000000000e+00 | -8.3969352724e-08 |

joint expectation及非零Cov并不自动转成主损失优势，更不是市场真实IV/交易数关联的识别。

## quote BCE、KL和bank penalty分别核

quote指rawP>0，不是no-trade。保存的future观测mask在本次两asset全部228×22均为正，合计10032个重叠日标签中负例0；小BCE只反映这个窗口的概率校准，不能证明quote分类能力。EqZ的BCE从存档概率和事件标签独立复算，逐component及ensemble均复现JSON。

| mode | ensemble quote BCE | mean component KL/day |
|---|---:|---|
| coupled_path | 0.002936383296 | 4.815445494038e-05 |
| independent_path | 0.002936378743 | 4.816838760230e-05 |
| direct_marked_control | 0.002935654499 | not applicable |

训练objective=Q+.05*BCE+.05*KL/day+已加权bank penalty，其中bank项=.005*等asset mean_bank SUM(active6/9输出平方)。所有记录的asset均值与拆分复核；selected bank数组项与下一epoch的preupdate训练记录匹配。同epoch的train packet发生在update前，不能冒充选中theta的训练损失。

direct无path分布，KL不适用，NPZ零向量只是占位。ensemble KL是component KL均值，不是混合分布KL。direct与coupled同137参，但direct缺path KL，regularizer不完全匹配；independent又少27参。这些限制列出，不改写已经没有coupled优势的点结果。

## v1全中心化是实际概率族限制

保存bank_reference逐通道核实：每asset/bucket的a/b/c均强制bank均值0；daily a/c按路径bucket长度1/4/17缩放，direct不除长度，b均不除，均与源码一致。

此约束不只是数值centering：b是quote logit shift、c是risk×quote interaction potential，常数项被强制0，会排除合法的常数概率校准/关联项；a常数risk tilt也被排除。不能把受限v1成绩当作所有概率路径模型的结论，也不能借此宣告v1成功。主线程提出的v2属于另一个待训练方案，本报告未读其源码或任何结果，效果未知，不能抵消这里的v1结果。

## 各asset/h主指标差

| asset/h | coupled−globalGamma | coupled−independent | coupled−direct |
|---|---:|---:|---:|
| KRBN/1 | +0.0009513119 | +0.0000019561 | -0.0014223646 |
| KRBN/5 | -0.0001449773 | +0.0000004718 | +0.0008534710 |
| KRBN/22 | -0.0011557676 | -0.0000001028 | +0.0005538836 |
| KCCA/1 | -0.0019583602 | +0.0000019507 | +0.0034316313 |
| KCCA/5 | -0.0007910452 | +0.0000005884 | +0.0004310667 |
| KCCA/22 | -0.0025372089 | +0.0000010581 | -0.0005919407 |

本轮只支持相对冻结globalGamma的局部开发点改善。对independent/direct及更强mean forecast没有实质增量，不作显著性、整体成功或新独立确认声明；原ETF2026已使用，本次完全未读。
完整逐格评分、训练拆分、矩/曲线误差和输入前后SHA见[比较JSON](<research/neural_model_restart_20261021/ROUND22_DEVELOPMENT_COMPARISON.json>)。
