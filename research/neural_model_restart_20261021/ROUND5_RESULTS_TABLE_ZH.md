# 第5轮统一测试集结果表

基线：`qrf_rule_atom`（规则零原子 + QRF 条件响应库）。所有模型共用同一origin/horizon/target；MQS由存盘的15个分位重算，CRPS只取有完整分布存盘者。

**口径**：只有 5 种子集成行（下表第一部分）算“一个模型”；`*_seedNN` 是诊断行，单独列在第二部分，**不得**用其中某个种子的最好值冒充模型结果。

## 集成级配置（35 个）

| 模型 | 相对基线 MQS | 胜格 | 相对基线 CRPS | 胜格 |
|---|---:|---:|---:|---:|
| `honest_no_readout` | +0.2662% | 14/20 | +0.2031% | 12/20 |
| `rbf_prior` | +0.2636% | 10/20 | +0.1570% | 11/20 |
| `forest_reference_only` | +0.1027% | 12/20 | +0.0727% | 14/20 |
| `reference` | +0.1027% | 12/20 | +0.0727% | 14/20 |
| `rawselect` | +0.0003% | 15/20 | -0.0990% | 15/20 |
| `rule_qrf` | +0.0000% | 0/20 | +0.0000% | 0/20 |
| `matched_honest_no_readout_reweight_only` | -0.1038% | 13/20 | -0.4003% | 12/20 |
| `qrf_original` | -0.1747% | 7/20 | -0.1557% | 8/20 |
| `forest_reweight_only` | -0.2547% | 10/20 | -0.3070% | 11/20 |
| `honest_readout_reweight_only` | -0.3699% | 9/20 | -0.5630% | 8/20 |
| `forest_transport` | -0.3960% | 9/20 | -0.4255% | 10/20 |
| `normalized` | -0.3960% | 9/20 | -0.4255% | 10/20 |
| `old_forest_transport` | -0.3960% | 9/20 | -0.4255% | 10/20 |
| `matched_insample_no_readout` | -0.4166% | 10/20 | -0.4539% | 10/20 |
| `forest_location_scale_only` | -0.4492% | 9/20 | -0.4583% | 10/20 |
| `uniform_seq_transport` | -0.5242% | 7/20 | -0.4388% | 8/20 |
| `insample_readout` | -0.5618% | 11/20 | -0.6501% | 12/20 |
| `honest_readout` | -0.6412% | 9/20 | -0.7166% | 9/20 |
| `lstm_qr` | -0.6618% | 7/20 | — | — |
| `horizon_kernel_rule` | -0.8331% | 8/20 | -0.5649% | 8/20 |
| `pooled_original` | -0.9441% | 8/20 | -0.7600% | 8/20 |
| `horizon_kernel` | -1.0199% | 8/20 | -0.7992% | 8/20 |
| `linear_prior` | -1.0770% | 8/20 | -1.2735% | 8/20 |
| `gru_qr` | -1.1084% | 7/20 | — | — |
| `transformer_qr` | -1.4024% | 6/20 | — | — |
| `tilt_full_5seed` | -1.5456% | 7/20 | -1.6874% | 7/20 |
| `candidate_probability_mixture` | -1.5456% | 7/20 | — | — |
| `candidate_quantile_average` | -1.5986% | 7/20 | — | — |
| `tcn_qr` | -1.6191% | 7/20 | — | — |
| `hgb_augmented` | -2.1321% | 3/20 | — | — |
| `uniform_rule` | -2.5537% | 7/20 | -2.2328% | 8/20 |
| `uniform_static` | -2.8092% | 7/20 | -2.5189% | 6/20 |
| `legacy_empirical` | -2.8155% | 7/20 | -2.5189% | 6/20 |
| `hgb_state_only` | -4.1286% | 4/20 | — | — |
| `gaussian` | -6.0021% | 5/20 | -6.1457% | 6/20 |

## 逐种子诊断（95 行）

| 模型 | 相对基线 MQS | 胜格 | 相对基线 CRPS | 胜格 |
|---|---:|---:|---:|---:|
| `rbf_prior_seed33` | +0.5200% | 11/20 | +0.3177% | 12/20 |
| `rawselect_seed33` | +0.3379% | 11/20 | +0.2710% | 12/20 |
| `normalized_honest_no_readout` | +0.2662% | 14/20 | +0.2031% | 12/20 |
| `frozen_honest_no_readout` | +0.2662% | 14/20 | +0.2031% | 12/20 |
| `rbf_prior_seed44` | +0.2243% | 11/20 | +0.2048% | 12/20 |
| `honest_no_readout_seed11` | +0.1868% | 11/20 | +0.1270% | 12/20 |
| `frozen_honest_no_readout_seed11` | +0.1868% | 11/20 | +0.1270% | 12/20 |
| `frozen_honest_no_readout_seed33` | +0.1660% | 14/20 | +0.1007% | 14/20 |
| `honest_no_readout_seed33` | +0.1660% | 14/20 | +0.1007% | 14/20 |
| `rbf_prior_seed55` | +0.1535% | 11/20 | +0.0934% | 12/20 |
| `matched_honest_no_readout_reweight_only_seed11` | +0.1000% | 15/20 | -0.2082% | 12/20 |
| `rbf_prior_seed22` | +0.0457% | 11/20 | -0.0757% | 12/20 |
| `honest_no_readout_seed55` | +0.0338% | 8/20 | -0.0207% | 8/20 |
| `frozen_honest_no_readout_seed55` | +0.0338% | 8/20 | -0.0207% | 8/20 |
| `honest_readout_reweight_only_seed33` | -0.0370% | 12/20 | -0.2490% | 9/20 |
| `rawselect_seed44` | -0.0382% | 11/20 | -0.0647% | 12/20 |
| `matched_honest_no_readout_reweight_only_seed44` | -0.1060% | 14/20 | -0.3991% | 13/20 |
| `matched_honest_no_readout_reweight_only_seed33` | -0.1095% | 13/20 | -0.3982% | 10/20 |
| `forest_location_scale_only_seed33` | -0.1159% | 9/20 | -0.1036% | 10/20 |
| `forest_transport_seed33` | -0.1177% | 9/20 | -0.1091% | 10/20 |
| `forest_reweight_only_seed33` | -0.1294% | 11/20 | -0.1850% | 11/20 |
| `qrf_22` | -0.1407% | 7/20 | -0.1378% | 6/20 |
| `forest_reweight_only_seed44` | -0.1415% | 12/20 | -0.1974% | 13/20 |
| `qrf_11` | -0.1741% | 9/20 | -0.2439% | 8/20 |
| `honest_no_readout_seed22` | -0.1768% | 11/20 | -0.2601% | 12/20 |
| `frozen_honest_no_readout_seed22` | -0.1768% | 11/20 | -0.2601% | 12/20 |
| `lstm_qr_seed33` | -0.2152% | 7/20 | — | — |
| `qrf_55` | -0.2191% | 5/20 | -0.2914% | 6/20 |
| `rawselect_seed11` | -0.2356% | 11/20 | -0.3613% | 11/20 |
| `matched_insample_no_readout_seed33` | -0.2438% | 9/20 | -0.2737% | 10/20 |
| `forest_transport_seed44` | -0.2464% | 10/20 | -0.2419% | 11/20 |
| `qrf_33` | -0.2555% | 6/20 | -0.2295% | 5/20 |
| `qrf_44` | -0.2893% | 8/20 | -0.2280% | 7/20 |
| `forest_reweight_only_seed55` | -0.2951% | 12/20 | -0.3492% | 12/20 |
| `matched_honest_no_readout_reweight_only_seed22` | -0.3206% | 14/20 | -0.6393% | 13/20 |
| `rbf_prior_seed11` | -0.3228% | 9/20 | -0.5215% | 10/20 |
| `forest_reweight_only_seed22` | -0.3543% | 7/20 | -0.4079% | 9/20 |
| `matched_insample_no_readout_seed11` | -0.3581% | 9/20 | -0.3798% | 10/20 |
| `forest_reweight_only_seed11` | -0.3626% | 6/20 | -0.4348% | 6/20 |
| `normalized_honest_readout_reweight_only` | -0.3699% | 9/20 | -0.5630% | 8/20 |
| `forest_transport_seed11` | -0.3746% | 10/20 | -0.4176% | 11/20 |
| `forest_location_scale_only_seed44` | -0.4093% | 9/20 | -0.3812% | 10/20 |
| `matched_honest_no_readout_reweight_only_seed55` | -0.4259% | 11/20 | -0.6719% | 11/20 |
| `honest_readout_reweight_only_seed55` | -0.4352% | 11/20 | -0.6219% | 9/20 |
| `forest_location_scale_only_seed11` | -0.4375% | 9/20 | -0.4538% | 10/20 |
| `insample_readout_seed11` | -0.4552% | 10/20 | -0.5553% | 11/20 |
| `matched_insample_no_readout_seed44` | -0.4864% | 9/20 | -0.4745% | 10/20 |
| `insample_readout_seed22` | -0.5154% | 10/20 | -0.5316% | 11/20 |
| `honest_readout_reweight_only_seed44` | -0.5319% | 12/20 | -0.6945% | 10/20 |
| `normalized_insample_readout` | -0.5618% | 11/20 | -0.6501% | 12/20 |
| `insample_readout_seed33` | -0.5640% | 11/20 | -0.7212% | 10/20 |
| `honest_readout_reweight_only_seed11` | -0.5666% | 11/20 | -0.7398% | 11/20 |
| `rawselect_seed22` | -0.5685% | 9/20 | -0.5808% | 10/20 |
| `matched_insample_no_readout_seed22` | -0.5751% | 9/20 | -0.5524% | 10/20 |
| `rawselect_seed55` | -0.5794% | 12/20 | -0.8566% | 10/20 |
| `matched_insample_no_readout_seed55` | -0.6354% | 11/20 | -0.7633% | 11/20 |
| `honest_readout_seed33` | -0.6389% | 10/20 | -0.7801% | 8/20 |
| `normalized_honest_readout` | -0.6412% | 9/20 | -0.7166% | 9/20 |
| `forest_transport_seed22` | -0.6614% | 9/20 | -0.6783% | 10/20 |
| `honest_readout_reweight_only_seed22` | -0.6678% | 11/20 | -0.8730% | 10/20 |
| `forest_location_scale_only_seed22` | -0.6731% | 9/20 | -0.6709% | 10/20 |
| `insample_readout_seed44` | -0.6810% | 9/20 | -0.6965% | 10/20 |
| `insample_readout_seed55` | -0.8031% | 10/20 | -0.9268% | 11/20 |
| `forest_transport_seed55` | -0.8085% | 9/20 | -0.8807% | 10/20 |
| `forest_location_scale_only_seed55` | -0.8251% | 9/20 | -0.8733% | 10/20 |
| `linear_prior_seed55` | -0.8310% | 8/20 | -0.9767% | 10/20 |
| `linear_prior_seed11` | -0.8616% | 7/20 | -1.0493% | 8/20 |
| `honest_no_readout_seed44` | -0.9406% | 12/20 | -1.0812% | 13/20 |
| `frozen_honest_no_readout_seed44` | -0.9406% | 12/20 | -1.0812% | 13/20 |
| `transformer_qr_seed22` | -1.0853% | 6/20 | — | — |
| `linear_prior_seed33` | -1.0866% | 7/20 | -1.2313% | 7/20 |
| `gru_qr_seed33` | -1.1071% | 8/20 | — | — |
| `gru_qr_seed11` | -1.1295% | 6/20 | — | — |
| `transformer_qr_seed44` | -1.1954% | 5/20 | — | — |
| `honest_readout_seed55` | -1.2007% | 9/20 | -1.3459% | 8/20 |
| `honest_readout_seed22` | -1.2120% | 12/20 | -1.3128% | 12/20 |
| `lstm_qr_seed44` | -1.2188% | 8/20 | — | — |
| `honest_readout_seed11` | -1.2341% | 14/20 | -1.3314% | 13/20 |
| `gru_qr_seed44` | -1.2407% | 8/20 | — | — |
| `lstm_qr_seed22` | -1.2692% | 8/20 | — | — |
| `linear_prior_seed44` | -1.3474% | 8/20 | -1.4973% | 9/20 |
| `gru_qr_seed22` | -1.4447% | 8/20 | — | — |
| `linear_prior_seed22` | -1.5394% | 8/20 | -1.8109% | 9/20 |
| `transformer_qr_seed33` | -1.6846% | 6/20 | — | — |
| `honest_readout_seed44` | -1.6864% | 13/20 | -1.4855% | 14/20 |
| `tcn_qr_seed33` | -1.7042% | 6/20 | — | — |
| `tcn_qr_seed11` | -1.8183% | 8/20 | — | — |
| `tcn_qr_seed44` | -1.8358% | 8/20 | — | — |
| `lstm_qr_seed11` | -1.8887% | 6/20 | — | — |
| `lstm_qr_seed55` | -1.9130% | 8/20 | — | — |
| `tcn_qr_seed55` | -1.9585% | 6/20 | — | — |
| `transformer_qr_seed55` | -2.0259% | 7/20 | — | — |
| `transformer_qr_seed11` | -2.0543% | 6/20 | — | — |
| `gru_qr_seed55` | -2.1639% | 5/20 | — | — |
| `tcn_qr_seed22` | -2.3051% | 8/20 | — | — |

## 结论

在 35 个集成级配置中，`honest_no_readout` 的 MQS（+0.2662%）与 CRPS（+0.2031%）同时排名第一；逐种子里有更高的单点值（例如 `rbf_prior_seed33`），但那是种子选择偏差，不计入模型比较。
第5轮所有神经增量与参考结构调整方向都没有超过参考本身；增益主要来自参考构造，且经自助法只在上海中长周期有区间支持。历史测试已复用，本表不证明显著性。
