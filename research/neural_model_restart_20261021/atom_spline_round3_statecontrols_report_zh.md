# State-only同信息结构控制与固定BCE探索

完成40fit，运行46.73秒。原神经state模型直接复用，没有重训。所有模型相同36维布局，价格日历8维置0、同正间距rank4样条、同CRPS/epoch0选择与full-precut refit。BCE组仅训练目标增加预先固定0.05*BCE(nonzero)，不调lambda；评估CRPS始终为原完整精确CDF积分。

| 对照 | MQS改善 | 完整CRPS改善 |
|---|---:|---:|
| state_neural_original_vs_state_continuous | +0.3400% (14/20) | +0.3460% (17/20) |
| state_neural_original_vs_state_static_atom | +0.2669% (12/20) | +0.2541% (14/20) |
| state_neural_original_vs_state_rule_atom | -0.0071% (5/20) | -0.0220% (9/20) |
| state_neural_bce005_vs_state_neural_original | -0.0027% (3/20) | -0.0031% (2/20) |
| state_neural_bce005_vs_state_rule_atom | -0.0099% (4/20) | -0.0252% (8/20) |
| state_neural_original_vs_empirical_constant | +1.2308% (16/20) | +0.8275% (16/20) |
| state_continuous_vs_empirical_constant | +0.8938% (13/20) | +0.4831% (10/20) |
| state_static_atom_vs_empirical_constant | +0.9665% (13/20) | +0.5749% (11/20) |
| state_rule_atom_vs_empirical_constant | +1.2379% (16/20) | +0.8493% (16/20) |
| state_neural_bce005_vs_empirical_constant | +1.2281% (16/20) | +0.8244% (16/20) |

## 两seed自身方向（均保留）

| seed | 对照 | MQS改善 | 完整CRPS改善 |
|---|---|---:|---:|
| 11 | state_neural_original_vs_state_continuous | +0.3800% (18/20) | +0.3843% (18/20) |
| 11 | state_neural_original_vs_state_static_atom | +0.2878% (16/20) | +0.2652% (15/20) |
| 11 | state_neural_original_vs_state_rule_atom | +0.0003% (9/20) | -0.0180% (10/20) |
| 11 | state_neural_bce005_vs_state_neural_original | -0.0030% (2/20) | -0.0047% (0/20) |
| 11 | state_neural_bce005_vs_state_rule_atom | -0.0027% (5/20) | -0.0227% (9/20) |
| 11 | state_neural_original_vs_empirical_constant | +1.0018% (16/20) | +0.6048% (16/20) |
| 11 | state_continuous_vs_empirical_constant | +0.6242% (12/20) | +0.2213% (7/20) |
| 11 | state_static_atom_vs_empirical_constant | +0.7161% (13/20) | +0.3405% (8/20) |
| 11 | state_rule_atom_vs_empirical_constant | +1.0015% (16/20) | +0.6227% (16/20) |
| 11 | state_neural_bce005_vs_empirical_constant | +0.9988% (16/20) | +0.6001% (16/20) |
| 22 | state_neural_original_vs_state_continuous | +0.2991% (12/20) | +0.3131% (14/20) |
| 22 | state_neural_original_vs_state_static_atom | +0.2616% (10/20) | +0.2556% (10/20) |
| 22 | state_neural_original_vs_state_rule_atom | -0.0077% (5/20) | -0.0244% (8/20) |
| 22 | state_neural_bce005_vs_state_neural_original | -0.0018% (2/20) | -0.0009% (2/20) |
| 22 | state_neural_bce005_vs_state_rule_atom | -0.0095% (5/20) | -0.0253% (8/20) |
| 22 | state_neural_original_vs_empirical_constant | +1.2911% (17/20) | +0.8985% (16/20) |
| 22 | state_continuous_vs_empirical_constant | +0.9950% (16/20) | +0.5872% (10/20) |
| 22 | state_static_atom_vs_empirical_constant | +1.0322% (13/20) | +0.6445% (11/20) |
| 22 | state_rule_atom_vs_empirical_constant | +1.2987% (17/20) | +0.9226% (16/20) |
| 22 | state_neural_bce005_vs_empirical_constant | +1.2894% (17/20) | +0.8976% (16/20) |

本轮同信息对照支持一个小的atom/滚动频率增量：原神经state模型比continuous的MQS改善0.3400%、完整CRPS改善0.3460%，两个seed方向均正；但规则atom模型略优于神经atom，因此没有证据把收益归功于学习到的概率门控。固定0.05辅助BCE未改善主评分，只产生很小的概率评分变化，不能据此断言其他辅助权重或更强编码器都无效。

| 模型 | 全部h pooled Brier |
|---|---:|
| state_continuous | 0.03459485 |
| state_static_atom | 0.04255763 |
| state_rule_atom | 0.03110535 |
| state_neural_original | 0.03118664 |
| state_neural_bce005 | 0.03117081 |

Brier存在高h近零基准，几何相对比会失真；应结合这里的绝对pooled分数和JSON逐单元分数，不把极端几何比作为优越性依据。

完整逐market/h的MQS、完整CRPS、WIS、coverage80/95、Brier及验证历史均保存在JSON；逐原点另存CSV/NPZ。原输出SHA均未变。历史测试复用，仅探索，不按本轮test重选最终模型或宣告创新成功。
