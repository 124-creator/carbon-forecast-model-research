# 条件经验输运：检查点、精确混合与分市场核对

100fit、4307个共同原点×期限键核对完成；58个旧文件SHA未改。每模型从checkpoint重建固定12个query（每h首/中/末），合并五seed不同支持再算同一分布。

最大a/b复现误差0，mixture Q原始收益单位误差0，mixture CRPS误差0。所有mixture CRPS满足凸性，不是seedCRPS平均。

全部训练/验证/refit/test几何query访问中，实际最小均匀方差0.0595555、最小倾斜方差0.0582295；uniform floor访问0、weighted floor访问0（总访问10537707，含重复epoch，不是唯一query数）。最终选epoch0共42/100。

| 市场 | 对frozen tilt | 对raw QRF | 对rule QRF | 对LSTM-QR | 对no-seq | 对loc-scale-only | 对reweight-only |
|---|---:|---:|---:|---:|---:|---:|---:|
| CEA | -1.5946% | +1.5613% | +0.9727% | +0.9004% | +0.0000% | -0.0331% | +0.0000% |
| GUANGZHOU | -2.6453% | -6.8926% | -7.0793% | -0.3995% | +1.7104% | +1.6970% | -3.6125% |
| SHANGHAI | +2.3148% | +2.8815% | +2.8509% | -0.8709% | +0.1277% | -0.0494% | +1.2504% |
| HUBEI | +5.1690% | +1.9889% | +1.9375% | +1.1874% | +0.3417% | +0.1082% | +5.1577% |
| EU | +1.5853% | -1.5952% | -1.6134% | -0.1497% | +0.0694% | +0.0152% | +0.1518% |

## 同seed自身对照（全部保留）

| seed | 参照 | MQS改善 | 胜出/20 |
|---|---|---:|---:|
| 11 | no_sequence_transport | +0.0746% | 6/20 |
| 11 | seq_location_scale_only | -0.0154% | 4/20 |
| 11 | seq_reweight_only | +0.3834% | 10/20 |
| 11 | frozen_tilt_same_seed | -0.6384% | 11/20 |
| 11 | lstm_qr_same_seed | +0.6979% | 16/20 |
| 22 | no_sequence_transport | -0.0739% | 2/20 |
| 22 | seq_location_scale_only | -0.0291% | 5/20 |
| 22 | seq_reweight_only | +0.0686% | 10/20 |
| 22 | frozen_tilt_same_seed | -0.2292% | 8/20 |
| 22 | lstm_qr_same_seed | +0.1190% | 14/20 |
| 33 | no_sequence_transport | +0.1290% | 7/20 |
| 33 | seq_location_scale_only | -0.1892% | 5/20 |
| 33 | seq_reweight_only | -0.0136% | 6/20 |
| 33 | frozen_tilt_same_seed | +2.4996% | 11/20 |
| 33 | lstm_qr_same_seed | -0.4053% | 11/20 |
| 44 | no_sequence_transport | -0.1462% | 8/20 |
| 44 | seq_location_scale_only | +0.0324% | 6/20 |
| 44 | seq_reweight_only | +0.6513% | 8/20 |
| 44 | frozen_tilt_same_seed | +0.4194% | 10/20 |
| 44 | lstm_qr_same_seed | -0.6477% | 12/20 |
| 55 | no_sequence_transport | +0.7001% | 12/20 |
| 55 | seq_location_scale_only | +0.5609% | 9/20 |
| 55 | seq_reweight_only | +0.2396% | 5/20 |
| 55 | frozen_tilt_same_seed | +2.1964% | 15/20 |
| 55 | lstm_qr_same_seed | +1.1595% | 16/20 |

正值表示seq_transport更好。固定branch p与最终F非零质量分开保存；Brier使用最终F，零碰撞计数另存。没有宣称退化subset单位方差、总体方差不随频率或新的通用分布族。所有历史测试复用，结论仅为候选模型实验。
