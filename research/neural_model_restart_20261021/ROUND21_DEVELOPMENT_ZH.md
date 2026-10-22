# R21 Gamma开发：独立重评分与数值边界

**NN在冻结fixedmix参数下有真实的局部开发点增量；整体竞争优势未成立。** ensemble先平均三seed预测再评分，known-scale QLIKE=1.080558646，优于fixed mixture1.082262727；但旧R16 raw和GRU控制更强，不能只展示较弱MEM。这里只分析已用2025，不做模型修改、拟合、滤波推理或2026确认。

## 目标与评分核对

- 两asset各228共同origin，2025-01-02—11-28，h1/5/22，最晚label2025-12-31。date/Y/origin/label_end与旧2025 replay逐元素一致，F17日期/Y一致，目标仍未来Parkinson均值，未因scale/c_ref改写标签。
- baseline joint由各asset分数等权平均；六个baseline fit独立选asset epoch，不把其best_joint_epoch字段误读为两asset共享选点。NN ensemble是预测均值后算损失，不能平均三seed loss或挑最优seed替代。
- 主指标为等asset×h QLIKE，known=raw−logEWMA22；normalized=y/v−log(y/v)−1，仅对其正均值计算百分比。sensorNLL权重.01与正则用于训练，未混入预测主损失。

## 全部R21模型与强旧控制

| 模型 | known-scale QLIKE | normalized QLIKE | MSE | NNensemble减该模型QLIKE |
|---|---:|---:|---:|---:|
| global_dense | 1.082293275204 | 0.368400695515 | 2.55732343e-08 | -0.001734629205 |
| shape_dense | 1.081230687343 | 0.367338107653 | 2.57411556e-08 | -0.000672041344 |
| fixed_mixture | 1.082262726828 | 0.368370147138 | 2.59413781e-08 | -0.001704080829 |
| neural_density_ensemble | 1.080558645999 | 0.366666066310 | 2.57524222e-08 | +0.000000000000 |
| MEM_stationary | 1.090458714432 | 0.376566134743 | 2.49413474e-08 | -0.009900068433 |
| MEM_integrated | 1.101562383060 | 0.387669803371 | 2.71975142e-08 | -0.021003737061 |
| R16_raw | 1.068306284657 | 0.354413704968 | 2.37204897e-08 | +0.012252361341 |
| R16_frozen | 1.070153583725 | 0.356261004035 | 2.37848191e-08 | +0.010405062274 |
| C18_GRU_ensemble | 1.057260925049 | 0.343368345359 | 2.29524638e-08 | +0.023297720950 |
| R20_continued_ensemble | 1.055157524077 | 0.341264944387 | 2.27589111e-08 | +0.025401121922 |
| R20_joint_ensemble | 1.054757767641 | 0.340865187952 | 2.27343873e-08 | +0.025800878358 |

NN相对fixedmix ΔQLIKE=-0.001704080829，normalized改善0.462600%。对MEM stationary改善2.629038%，但对R16 raw退化3.457079%，对R20 continued退化7.443226%。这些是反复选择的开发点，不能作独立确认或显著性声明。

## epoch、冻结theta和三seed方向

baseline selected epoch：global KRBN/KCCA5/5、shape4/7、fixedmix2/5；每asset4个参数(mu,phi,q,kappa)，六fit分别run20/20、20/22、20/20。NN43个参数共享volume-only prior网络；8个asset theta值作为buffer冻结，三seed记录逐项等于选中fixedmix。未更换fixedmix theta来追NN得分。

| NN seed | best epoch/run | knownQLIKE | jointΔvsfixedmix | KRBNΔ | KCCAΔ |
|---|---|---:|---:|---:|---:|
| 11 | 56/71 | 1.078952547937 | -0.003310178891 | -0.003668879018 | -0.002951478764 |
| 22 | 38/53 | 1.081562761944 | -0.000699964884 | -0.001429311819 | +0.000029382052 |
| 33 | 0/20 | 1.082262726828 | +0.000000000000 | +0.000000000000 | +0.000000000000 |

这是2个joint改善＋1个精确baseline回退。seed22的KCCA微差，不能称三seed/双asset全面改善；seed33 best epoch0的预测及Mposterior与fixedmix逐元素完全相同。NN所有run未撞80上限，记录的gradient norm均<.006，未触发clip2。

| asset/h | NN−fixedmix | NN−R16raw | NN−R20continued |
|---|---:|---:|---:|
| KRBN/1 | -0.001310384155 | +0.024226601364 | +0.039665885013 |
| KRBN/5 | -0.002891998617 | +0.014898664507 | +0.025071106666 |
| KRBN/22 | -0.002027401788 | +0.026123967500 | +0.026455128758 |
| KCCA/1 | +0.002438059560 | -0.027924205378 | -0.007523380656 |
| KCCA/5 | -0.003617770649 | +0.006563299777 | +0.031484717034 |
| KCCA/22 | -0.002814989325 | +0.029625840279 | +0.037253274717 |

## prior与M后验：有效变化，但不是物理N识别

固定prior∝M，M={1,2,4,8,16,64,512}。静态源码显示NN为volume-only1→4→7网络，pi=softmax(logM+net(clamp(v,-5,5)))，形状只进joint Gamma sensor一次。NPZ没有pi/log_prior_M，未加载权重或前向NN，所以不能声称直接复算了NN prior概率曲线。可直接核验的是以下Mposterior、c_ref和预测变化。

| asset/seed | meanTV vsfixed | posterior E[M] | fixed E[M] | c_ref | fixed c_ref | futurefactor min/max |
|---|---:|---:|---:|---:|---:|---|
| KRBN/11 | 0.397110 | 225.810725 | 396.990496 | 0.809206812 | 0.913009571 | 0.877861/1.097882 |
| KRBN/22 | 0.357433 | 236.648624 | 396.990496 | 0.821404849 | 0.913009571 | 0.940394/1.073357 |
| KRBN/33 | 0.000000 | 396.990496 | 396.990496 | 0.913009571 | 0.913009571 | 0.998011/1.001407 |
| KCCA/11 | 0.451527 | 195.783471 | 390.713698 | 0.770693321 | 0.914974110 | 0.904606/1.117331 |
| KCCA/22 | 0.393040 | 212.607366 | 390.713698 | 0.807878122 | 0.914974110 | 0.953272/1.073074 |
| KCCA/33 | 0.000000 | 390.713698 | 390.713698 | 0.914974110 | 0.914974110 | 0.999570/1.000412 |

11/22的posterior变化明显，33精确不变。这是工作参考族中有效混合变化，不沿用R20的alpha通道未启用解释；也不证明真实交易次数。c_ref随prior变化，NN的增量包括prior、sensor参考与futurefactor的耦合，不能写成固定全部尺度下单独改变N的纯识别实验。

## hurdle、scale与c_ref

scale为pre2025全部rawP均值，包含zero：KRBN0.00011277869412054874(n1112)、KCCA0.0001824213189756938(n815)，与旧MEM尺度记录相同到浮点容差。scope/cut与首个2025origin一致；本次未从CSV重新计算均值。

hurdle用pre2025未解析range数4/1112、1/815，ridge.01、两类各.5伪计数；p0=logistic(intercept+slope*volumeZ)，logit截到±20。它表示无可解析range，不表示真实volatility0或无交易。NN不重拟合hurdle/scale/AR1。

| asset | hurdle intercept/slope | saved2025 p0 mean[min,max] | volume abs>5 count | logit abs>20 count |
|---|---|---|---:|---:|
| KRBN | -5.496397880/-0.483331829 | 0.004287[0.001108,0.011677] | 0 | 0 |
| KCCA | -6.372863045/-0.220162085 | 0.001805[0.000810,0.003490] | 0 | 0 |

c_ref=pre2025 observed-bank平均[(1−p0(v))*sum piM(v)c_globalM]，不是固定dense c512。futurefactor为同一ceff/c_ref的GH16条件volume AR1期望，依赖未来volume与latent-risk innovations独立的工作假设。由已存posterior mean、固定theta、scale和futurefactor复核daily恒等式，再以前h日均值复核预测；均在数值容差内，未重跑滤波/NN。

## clip/boundary与不可省略的数值误差

grid256，logS∈[-8,8]，GH16、S上非负线性概率scatter。存档max_clip_mass及max_abs_transition_mean_error是后验加权的已访问路径统计，不是所有条件状态的最坏row误差。NN完整prefix的maxboundary约1e-15–1e-13，加权clip max约1.5e-11–2.0e-11；2025 forecast-origin的加权clip约1e-58或更小，加权mean误差约1e-16。详细每mode/asset/seed值在JSON。

这些小路径诊断不能推出exact continuous model或零数值误差。主线程先前纯数值审计报告mean closure约0.111%、second moment约0.203%、pure256vs512 posterior difference约0.00254。本次未重新运行该审计，也未定位独立保存的audit文件；保留给定指标名称，不擅自改称某个相对/绝对范数，更不当作市场预测误差界。closure百分比与QLIKE效应不是同一量纲，不能直接相互抵消或比较。

Gamma sensor是校准的工作分布，离散grid、GH、interpolation、边界与校准误差都保留。可表述为在该固定数值方案下的开发增量，不声称精确Brownian joint law、真实M/IV识别或无近似误差。

## 结论与使用边界

R21的NN prior在冻结theta下有实际posterior/输出变化及小开发增量；它胜过旧MEM的点成绩不应掩盖更强raw/GRU。反复使用2025及其epoch选择不构成新确认，paired seeds也不是独立datasets。本报告没有追加bootstrap/显著性筛选或任何拟合。

2026已被R20用过；R21随后若评价同一旧2026，只能明确标为diagnostic，不能重新包装为全新独立确认。本次没有打开任何2026产物、GRN或CSV，没有改模型。
完整数值、输入hash与逐格对照见[审计JSON](<research/neural_model_restart_20261021/ROUND21_DEVELOPMENT_COMPARISON.json>)。仅新增这两份报告，所有输入hash分析前后不变。
