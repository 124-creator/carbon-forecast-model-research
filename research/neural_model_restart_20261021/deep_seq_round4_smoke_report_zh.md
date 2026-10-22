# 第四轮任务A：正确训练的四个深度序列分位基线

完成4次inner训练与固定所选epoch的全precut重拟合，运行2.16秒。所有模型/seed保留，不按test挑选。

历史64步r/原点已知sd及不变价indicator，加入与冻结候选完全相同36context。上下文scaler与经验15tau底座只fit当前训练集；final重新估计全部precut统计。原cut/h/128lookback/step2与目标保持完全一致。每tau使用正确不对称pinball，第一epoch即联合训练四个h。zerohead恢复经验底座，sort保持单调并允许零ties。

LSTM/GRU为单层hidden16；TCN为d1/2/4因果卷积，全64输出平均与末输出拼接；Transformer使用patch4保留64步内顺序，16tokens显式sin/cos位置编码、1层2头。四模型共享16维context与24维head。记录了inner encoder非零梯度/参数变化，以及每个非零epoch final encoder的更新；epoch0 final不训练是允许的验证选择。

统一AdamW lr=.001 wd=.003，max80/min30/patience16（smoke参数另存），clip1，torch1thread。QR五seed采用quantile平均，这与冻结候选的五seed完整概率混合不同；同时给候选五seed quantile平均的无重训对比。只有15q，未构造或声称完整CRPS。

| 深度模型 / 参照 | MQS几何改善 | WIS几何改善 |
|---|---:|---:|
| lstm_qr_vs_candidate_probability_mixture | -2.5318% (1/4) | -3.1209% |
| lstm_qr_vs_candidate_quantile_average | -2.4649% (1/4) | -2.9386% |
| lstm_qr_vs_legacy_empirical | +0.0000% (4/4) | +0.0000% |
| lstm_qr_vs_uniform_static | -0.1734% (1/4) | -0.2708% |
| lstm_qr_vs_gaussian | +0.4469% (1/4) | +0.0652% |
| gru_qr_vs_candidate_probability_mixture | -2.5003% (1/4) | -3.0937% |
| gru_qr_vs_candidate_quantile_average | -2.4335% (1/4) | -2.9115% |
| gru_qr_vs_legacy_empirical | +0.0307% (4/4) | +0.0264% |
| gru_qr_vs_uniform_static | -0.1426% (1/4) | -0.2444% |
| gru_qr_vs_gaussian | +0.4774% (1/4) | +0.0916% |
| tcn_qr_vs_candidate_probability_mixture | -2.5318% (1/4) | -3.1209% |
| tcn_qr_vs_candidate_quantile_average | -2.4649% (1/4) | -2.9386% |
| tcn_qr_vs_legacy_empirical | +0.0000% (4/4) | +0.0000% |
| tcn_qr_vs_uniform_static | -0.1734% (1/4) | -0.2708% |
| tcn_qr_vs_gaussian | +0.4469% (1/4) | +0.0652% |
| transformer_qr_vs_candidate_probability_mixture | -2.5318% (1/4) | -3.1209% |
| transformer_qr_vs_candidate_quantile_average | -2.4649% (1/4) | -2.9386% |
| transformer_qr_vs_legacy_empirical | +0.0000% (4/4) | +0.0000% |
| transformer_qr_vs_uniform_static | -0.1734% (1/4) | -0.2708% |
| transformer_qr_vs_gaussian | +0.4469% (1/4) | +0.0652% |

完整逐seed/市场/h的MQS、WIS、中位数MAE及80/95覆盖在cells CSV与JSON，逐原点各seed与ensemble15q在NPZ；原点日期与target供合并。所有冻结源/结果SHA保持不变。历史测试已开发复用，本实验不证明新颖性或发表成功。
