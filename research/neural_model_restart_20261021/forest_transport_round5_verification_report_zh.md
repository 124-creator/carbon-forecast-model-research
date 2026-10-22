# 森林参考输运：75fit与phase/mixture核对

75配置、4307共同原点×期限键、76旧文件与5 prior cache SHA保持一致。每phase fit indices、boundary、scaler、bank D行、maturity和query sid严格核对。

checkpoint重建每模型12固定query，a/b最大误差0，五seed真实mixture Q误差0、CRPS误差0。完整精确概率混合满足CRPS凸性，不是seed分数/quantile平均。

全部75条raw-relative validation诊断轨迹已留存，21条best_raw_epoch不同于原normalized best；没有用它们改本轮selector或额外refit。仅在既有visited epoch内取诊断best，不宣称新的完整搜索。

最终epoch0 44/75；inner非零encoder梯度75/75。reference/weighted floor访问0/0，额外零碰撞0；实际非零概率与branch差最大2.22e-16。

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

固定forest_reference_only将gamma平滑贡献与NN残差区分；所有三学习组的其他强基线对比在JSON/cells中全保留。所有历史测试已开发复用，不以正结果证明发表成功，也不以负结果否定整个方向。
