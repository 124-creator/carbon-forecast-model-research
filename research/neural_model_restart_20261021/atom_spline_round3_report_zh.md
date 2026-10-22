# 第三轮：零质量加直接条件mark样条

已完成5variants×5market×seed11/22，共50次inner训练和对应full-precut refit，最大48epoch，正式运行68.1秒。主候选预设为neural_atom_spline；不按test把控制改称主模型。仅新增本轮前缀文件。

连续mark的分位函数由固定33个对称u-knots（含0/1和15个评分taus）之间的线性段定义，网络学习位置和rank4正间距残差。atom组以当前拟合区间非零标准化收益经验quantiles初始化，continuous以全部收益初始化；严格正间距下界产生的小改动逐fit保存。概率采用静态、0.8滚动+0.2静态规则或神经融合。没有使用上一轮moment/variance budget。

训练CRPS是模型完整CDF的精确分段积分：合并全部knots与0，处理零跳跃的左右极限，逐线性CDF区间积分并加支持外尾部。两seed混合后重新合并双方knots，计算同一个混合F的完整CRPS，而不是平均seed CRPS。单样条quantile解析插值，混合quantile才使用CDF二分。均匀分布、纯atom、atom+uniform、不同支持的两seed混合独立CDF积分和finite-difference/autograd梯度全部通过。knots跨零处排序为分段可微。

输入新增原点log price、相对120/250历史均价的log位置、250价格range位置，以及原点年内sin/cos和二阶harmonics；早期不足250历史时只用已存在前缀。所有主组接收同一增强输入。no-price-calendar控制保留相同输入维数和网络布局，末8维标准化特征置0，仍有既有收益/波动/非零率状态；不伪称完全无量纲输入。日期仅用origin，target_date只作输出元数据。

每fit只用当前训练分区拟合scaler/经验knots/静态概率。统一normalized验证MQS选epoch，允许epoch0；final重新估计全precut统计并重拟合固定轮数。所有4307个原点×期限键与前轮完全一致，预测仅反变换一次。

## 预设主候选的真实增量

几何改善按20个市场×h单元等权，正值越好；括号为胜出单元。

| 参照 | MQS改善 | 完整CRPS改善 |
|---|---:|---:|
| continuous_spline | +0.2905% (11/20) | +0.3213% (12/20) |
| static_atom_spline | +0.2424% (10/20) | +0.2534% (11/20) |
| rule_atom_spline | -0.0359% (5/20) | -0.0598% (7/20) |
| neural_atom_no_price_calendar | -0.5576% (8/20) | -0.5274% (8/20) |
| double_normal | +0.7761% (11/20) | +1.2355% (11/20) |
| empirical_constant | +0.6801% (13/20) | +0.3044% (13/20) |
| gaussian_constant | +3.6658% (13/20) | +3.7109% (13/20) |

预设主候选对经验常数有0.68% MQS和0.30%完整CRPS改善，但神经概率头没有胜过规则，增强价格日历信息反而整体有害。旧doubleNormal只含原收益状态，且epoch0政策不同；对它的改善不能单独归因于分布结构。

## 各样条组对经验常数

| 模型 | MQS改善 | 完整CRPS改善 | epoch0次数/10 |
|---|---:|---:|---:|
| continuous_spline | +0.3907% (10/20) | -0.0169% (10/20) | 3 |
| static_atom_spline | +0.4387% (11/20) | +0.0512% (9/20) | 3 |
| rule_atom_spline | +0.7157% (13/20) | +0.3640% (13/20) | 3 |
| neural_atom_spline | +0.6801% (13/20) | +0.3044% (13/20) | 3 |
| neural_atom_no_price_calendar | +1.2308% (16/20) | +0.8275% (16/20) | 6 |

normalized-state-only控制（no-price-calendar）相对经验常数的改善更大，但这是本轮完整比较后观察到的控制结果；保留其身份并列出两个seed，不能按test选择最好seed或声称已经泛化成功。

## 逐seed：自身预测分布对照

| seed | 候选/参照 | MQS改善 | 完整CRPS改善 |
|---|---|---:|---:|
| 11 | neural_atom_spline_vs_continuous_spline | +0.2411% (11/20) | +0.2884% (12/20) |
| 11 | neural_atom_spline_vs_rule_atom_spline | -0.0298% (7/20) | -0.0576% (3/20) |
| 11 | neural_atom_spline_vs_neural_atom_no_price_calendar | -1.5128% (5/20) | -1.3680% (7/20) |
| 11 | neural_atom_spline_vs_empirical_constant | -0.4959% (12/20) | -0.7549% (12/20) |
| 11 | neural_atom_no_price_calendar_vs_empirical_constant | +1.0018% (16/20) | +0.6048% (16/20) |
| 11 | neural_atom_no_price_calendar_vs_double_normal | +0.6265% (10/20) | +1.0006% (11/20) |
| 22 | neural_atom_spline_vs_continuous_spline | +0.2636% (10/20) | +0.2747% (10/20) |
| 22 | neural_atom_spline_vs_rule_atom_spline | -0.0410% (6/20) | -0.0627% (5/20) |
| 22 | neural_atom_spline_vs_neural_atom_no_price_calendar | -0.1502% (6/20) | -0.2057% (7/20) |
| 22 | neural_atom_spline_vs_empirical_constant | +1.1429% (14/20) | +0.6946% (13/20) |
| 22 | neural_atom_no_price_calendar_vs_empirical_constant | +1.2911% (17/20) | +0.8985% (16/20) |
| 22 | neural_atom_no_price_calendar_vs_double_normal | +2.1880% (13/20) | +2.6217% (13/20) |

## 分市场主候选MQS改善

| 市场 | 对continuous | 对rule | 对no-context | 对经验 | 主候选epoch11/22 |
|---|---:|---:|---:|---:|---:|
| CEA | +0.9195% | -0.0000% | +0.0000% | +0.9195% | 0/0 |
| GUANGZHOU | +0.1863% | -0.1408% | -3.1844% | -2.3867% | 10/8 |
| SHANGHAI | +0.3061% | -0.0409% | +0.4290% | +4.8680% | 8/6 |
| HUBEI | +0.0512% | +0.0039% | +1.1187% | +1.2356% | 2/4 |
| EU | -0.0132% | -0.0017% | -1.2076% | -1.3965% | 4/0 |

## 主候选逐市场/h覆盖与概率

| market | h | MQS | 完整CRPS | coverage80 | coverage95 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| CEA | 1 | 0.00728484 | 0.00819345 | 0.8223 | 0.9307 | 0.08461941 |
| CEA | 5 | 0.01783179 | 0.02015494 | 0.6697 | 0.8545 | 0.00738537 |
| CEA | 10 | 0.02776775 | 0.03117647 | 0.5719 | 0.8226 | 0.01600125 |
| CEA | 20 | 0.04730523 | 0.05332721 | 0.4720 | 0.7888 | 0.00395899 |
| GUANGZHOU | 1 | 0.01759732 | 0.02014933 | 0.7970 | 0.9152 | 0.15322255 |
| GUANGZHOU | 5 | 0.02619835 | 0.03010119 | 0.8659 | 0.9451 | 0.01209493 |
| GUANGZHOU | 10 | 0.02923491 | 0.03262199 | 0.9110 | 0.9571 | 0.00610586 |
| GUANGZHOU | 20 | 0.03901138 | 0.04305539 | 0.9097 | 0.9502 | 0.00926204 |
| SHANGHAI | 1 | 0.01296872 | 0.01463437 | 0.7923 | 0.9000 | 0.11798004 |
| SHANGHAI | 5 | 0.02102824 | 0.02373747 | 0.7656 | 0.9219 | 0.06602043 |
| SHANGHAI | 10 | 0.02595367 | 0.02901417 | 0.7600 | 0.8960 | 0.00819382 |
| SHANGHAI | 20 | 0.03379181 | 0.03822827 | 0.7833 | 0.8750 | 0.00853757 |
| HUBEI | 1 | 0.01064548 | 0.01209629 | 0.7329 | 0.9521 | 0.05143131 |
| HUBEI | 5 | 0.01445264 | 0.01652748 | 0.7986 | 0.9722 | 0.00697617 |
| HUBEI | 10 | 0.01660710 | 0.01907739 | 0.8085 | 1.0000 | 0.01410691 |
| HUBEI | 20 | 0.02038970 | 0.02294643 | 0.8824 | 0.9926 | 0.00733881 |
| EU | 1 | 0.01095340 | 0.01288856 | 0.7233 | 0.9686 | 0.00627468 |
| EU | 5 | 0.02586664 | 0.03057043 | 0.6752 | 0.9299 | 0.00009033 |
| EU | 10 | 0.03754139 | 0.04407684 | 0.6516 | 0.9032 | 0.00004644 |
| EU | 20 | 0.04729626 | 0.05552673 | 0.7267 | 0.9067 | 0.00011601 |

## Brier汇总

| 模型 | pooled Brier | 等市场×h macro Brier |
|---|---:|---:|
| continuous_spline | 0.03459485 | 0.03171707 |
| double_normal | 0.03133303 | 0.02916668 |
| empirical_constant | 0.04255763 | 0.03619106 |
| gaussian_constant | 0.03459485 | 0.03171707 |
| neural_atom_no_price_calendar | 0.03118664 | 0.02888997 |
| neural_atom_spline | 0.03132253 | 0.02898815 |
| rule_atom_spline | 0.03110535 | 0.02876832 |
| static_atom_spline | 0.04255763 | 0.03619106 |

某些高h单元完全没有零收益，continuous的Brier恰为0，几何相对比会失真；以绝对pooled/macro分数与逐单元胜数一起解释，不把其极端几何比作为优越性依据。

全部160行市场/h/模型评分另存cells CSV；逐原点34456行=4307键×8模型，CSV含概率、完整CRPS及15分位数，NPZ另存每seed knots和p。seed_cells保存两seed全部所列对照。已核对有限、不交叉、相同原点/期限/目标。

限制：历史测试已经开发复用，所有结果都是探索；样条有限支持、33个固定概率knots、数据量及两seed均有限。尚无显著性、稳定滚动外推或新颖性证明。正结果不等于可发表成功，也不因负消融结果关闭整个ML/DL方向。经验基准CRPS为全部precut训练样本ECDF闭式，MQS为linear样本分位数、Brier为Beta平滑概率，三项对应不同经验表示，未把15点分数冒充完整CRPS。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python research/neural_model_restart_20261021/atom_spline_round3.py --seeds 11 22 --epochs 48 --tag atom_spline_round3
```
