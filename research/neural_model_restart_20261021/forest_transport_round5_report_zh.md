# 第五轮：条件森林参考上的LSTM矩归一输运

完成75次内层选择/全precut重拟合组合及各市场固定reference，运行389.70秒。没有按test选择模型/seed。

完整神经输运相对固定平滑森林参考的MQS退步0.4992%、完整CRPS退步0.4985%；相对规则QRF也退步0.3960%/0.4255%。本轮没有建立神经残差的额外收益。固定forest_reference_only本身相对规则QRF有0.1027% MQS和0.0727% CRPS的小幅改善（12/20、14/20胜），这是预设5%平滑参考的表现，不能归功于LSTM。完整模型虽然比旧uniform seq_transport好0.1275% MQS，仍不足以称为成功超过强基线。

75个inner encoder均有非零梯度，31个非零epoch最终refit均更新encoder，44个最终epoch0由原normalized validation选择。额外raw-relative validation轨迹完整记录75条，其中21条best_raw_epoch不同于normalized best（主候选为8/25）；它们只作诊断，未用于选轮或新增refit，且只在原训练访问过的epoch中取诊断best。76个旧文件及5个prior cache SHA保持不变；checkpoint每模型固定12个query重建a/b和真实5seed mixture Q/CRPS误差均0，floor与额外零碰撞均0。

唯一主要变化为参考u：0.95*成熟QRF conditional NZ+0.05*成熟uniform；head输出位置/尺度/shape2，其余骨干与训练设置复用冻结第四轮。inner只读inner forest与Dtr，final只读final forest与Drf；每phase fit indices/scaler/bank D行顺序/maturity/sid全部严格核对。模型参数2468、64past+36ctx、所有h首轮联合。

w正比u exp(beta·phi2)，参考条件矩来自u，倾斜矩来自w；位置尺度目标归一后shape不改变指定条件矩。fixed reference_only无NN更新，隔离gamma平滑作用。保持旧beta²正则，没有新增delta/ell penalty。

所有支持随query/seed变化，最终五seed先合并其概率/affine支持及零atom、排序再算同一F的quantile和完整CRPS。Brier使用完整F非零质量，branch p固定rule且另记碰撞。variance floors与实际minvar/稳定性下界比均另存。

| 主候选 / 参照 | MQS改善 | 完整CRPS改善 |
|---|---:|---:|
| forest_transport_vs_forest_reference_only | -0.4992% (4/20) | -0.4985% (11/20) |
| forest_transport_vs_rule_qrf | -0.3960% (9/20) | -0.4255% (10/20) |
| forest_transport_vs_qrf_original | -0.2210% (10/20) | -0.2694% (11/20) |
| forest_transport_vs_lstm_qr | +0.2640% (13/20) | 15q不可识别 |
| forest_transport_vs_uniform_seq_transport | +0.1275% (13/20) | +0.0133% (12/20) |
| forest_transport_vs_uniform_rule | +2.1040% (13/20) | +1.7679% (12/20) |
| forest_transport_vs_forest_location_scale_only | +0.0529% (8/20) | +0.0327% (7/20) |
| forest_transport_vs_forest_reweight_only | -0.1409% (8/20) | -0.1180% (8/20) |

正值表示左侧forest_transport更好。JSON含全部三学习组及reference对强基线评分；cells有逐seed/market/h，NPZ保存单seed与真正mixture Q/CRPS/actualp/branchp/a/b。checkpoint包含15模型state、final bank/u/times/sid、scaler及inner/final cache审计与SHA，可重建权重。

森林/scaler/phi是整个允许fit区间参数；成熟mask限制响应检索，不称每个训练query都使用当时冻结的全部参数。历史测试已开发复用，结果不证明新颖性、显著性或可发表成功。
