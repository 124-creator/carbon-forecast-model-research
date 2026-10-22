# 第四轮任务A：正确训练的四个深度序列分位基线

完成100次inner训练与固定所选epoch的全precut重拟合，运行610.01秒。所有模型/seed保留，不按test挑选。

整体MQS上，LSTM-QR相对冻结候选概率混合改善0.8704%，GRU-QR改善0.4305%，Transformer-QR改善0.1410%；TCN-QR退步0.0724%。因此冻结候选尚未建立超过这些正确训练深度序列基线的整体优势。改为候选自身的五seed分位平均后，方向保持，不能归因于集成表示差异。表中正值始终表示左侧深度模型更好。

分市场，冻结候选在CEA/广州优于全部四模型；深度QR在上海/湖北/EU整体更强。100个内层编码器均有非零梯度；74个非零epoch最终refit均有编码器更新，26个最终epoch0由验证选择，未伪称它们的最终编码器经过训练。该结果针对这里固定的小宽度设置，不泛化为全部架构/超参的结论。

历史64步r/原点已知sd及不变价indicator，加入与冻结候选完全相同36context。上下文scaler与经验15tau底座只fit当前训练集；final重新估计全部precut统计。原cut/h/128lookback/step2与目标保持完全一致。每tau使用正确不对称pinball，第一epoch即联合训练四个h。zerohead恢复经验底座，sort保持单调并允许零ties。

LSTM/GRU为单层hidden16；TCN为d1/2/4因果卷积，全64输出平均与末输出拼接；Transformer使用patch4保留64步内顺序，16tokens显式sin/cos位置编码、1层2头。四模型共享16维context与24维head。记录了inner encoder非零梯度/参数变化，以及每个非零epoch final encoder的更新；epoch0 final不训练是允许的验证选择。

统一AdamW lr=.001 wd=.003，max80/min30/patience16（smoke参数另存），clip1，torch1thread。QR五seed采用quantile平均，这与冻结候选的五seed完整概率混合不同；同时给候选五seed quantile平均的无重训对比。只有15q，未构造或声称完整CRPS。

| 深度模型 / 参照 | MQS几何改善 | WIS几何改善 |
|---|---:|---:|
| lstm_qr_vs_candidate_probability_mixture | +0.8704% (12/20) | +0.5463% |
| lstm_qr_vs_candidate_quantile_average | +0.9221% (12/20) | +0.5707% |
| lstm_qr_vs_legacy_empirical | +2.0948% (18/20) | +2.3620% |
| lstm_qr_vs_uniform_static | +2.0888% (15/20) | +2.3543% |
| lstm_qr_vs_gaussian | +5.0380% (13/20) | +4.7360% |
| gru_qr_vs_candidate_probability_mixture | +0.4305% (12/20) | +0.1111% |
| gru_qr_vs_candidate_quantile_average | +0.4825% (12/20) | +0.1356% |
| gru_qr_vs_legacy_empirical | +1.6604% (18/20) | +1.9347% |
| gru_qr_vs_uniform_static | +1.6544% (14/20) | +1.9270% |
| gru_qr_vs_gaussian | +4.6166% (14/20) | +4.3191% |
| tcn_qr_vs_candidate_probability_mixture | -0.0724% (11/20) | -0.2800% |
| tcn_qr_vs_candidate_quantile_average | -0.0201% (11/20) | -0.2554% |
| tcn_qr_vs_legacy_empirical | +1.1637% (18/20) | +1.5508% |
| tcn_qr_vs_uniform_static | +1.1576% (15/20) | +1.5430% |
| tcn_qr_vs_gaussian | +4.1349% (13/20) | +3.9445% |
| transformer_qr_vs_candidate_probability_mixture | +0.1410% (10/20) | -0.0948% |
| transformer_qr_vs_candidate_quantile_average | +0.1931% (10/20) | -0.0703% |
| transformer_qr_vs_legacy_empirical | +1.3744% (17/20) | +1.7326% |
| transformer_qr_vs_uniform_static | +1.3684% (15/20) | +1.7248% |
| transformer_qr_vs_gaussian | +4.3393% (13/20) | +4.1219% |

完整逐seed/市场/h的MQS、WIS、中位数MAE及80/95覆盖在cells CSV与JSON，逐原点各seed与ensemble15q在NPZ；原点日期与target供合并。所有冻结源/结果SHA保持不变。历史测试已开发复用，本实验不证明新颖性或发表成功。
