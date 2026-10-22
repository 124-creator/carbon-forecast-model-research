# 第五轮：条件森林参考上的LSTM矩归一输运

完成3次内层选择/全precut重拟合组合及各市场固定reference，运行2.65秒。没有按test选择模型/seed。

唯一主要变化为参考u：0.95*成熟QRF conditional NZ+0.05*成熟uniform；head输出位置/尺度/shape2，其余骨干与训练设置复用冻结第四轮。inner只读inner forest与Dtr，final只读final forest与Drf；每phase fit indices/scaler/bank D行顺序/maturity/sid全部严格核对。模型参数2468、64past+36ctx、所有h首轮联合。

w正比u exp(beta·phi2)，参考条件矩来自u，倾斜矩来自w；位置尺度目标归一后shape不改变指定条件矩。fixed reference_only无NN更新，隔离gamma平滑作用。保持旧beta²正则，没有新增delta/ell penalty。

所有支持随query/seed变化，最终五seed先合并其概率/affine支持及零atom、排序再算同一F的quantile和完整CRPS。Brier使用完整F非零质量，branch p固定rule且另记碰撞。variance floors与实际minvar/稳定性下界比均另存。

| 主候选 / 参照 | MQS改善 | 完整CRPS改善 |
|---|---:|---:|
| forest_transport_vs_forest_reference_only | +0.0000% (0/20) | -0.0000% (1/20) |
| forest_transport_vs_rule_qrf | +0.2379% (3/20) | +0.1208% (3/20) |
| forest_transport_vs_qrf_original | +0.8310% (3/20) | +0.5586% (3/20) |
| forest_transport_vs_lstm_qr | +0.1652% (3/20) | 15q不可识别 |
| forest_transport_vs_uniform_seq_transport | -0.7419% (2/20) | -0.6435% (1/20) |
| forest_transport_vs_uniform_rule | -0.7419% (2/20) | -0.6435% (1/20) |
| forest_transport_vs_forest_location_scale_only | +0.0000% (0/20) | +0.0000% (0/20) |
| forest_transport_vs_forest_reweight_only | -0.0077% (2/20) | +0.0020% (3/20) |

正值表示左侧forest_transport更好。JSON含全部三学习组及reference对强基线评分；cells有逐seed/market/h，NPZ保存单seed与真正mixture Q/CRPS/actualp/branchp/a/b。checkpoint包含15模型state、final bank/u/times/sid、scaler及inner/final cache审计与SHA，可重建权重。

森林/scaler/phi是整个允许fit区间参数；成熟mask限制响应检索，不称每个训练query都使用当时冻结的全部参数。历史测试已开发复用，结果不证明新颖性、显著性或可发表成功。
