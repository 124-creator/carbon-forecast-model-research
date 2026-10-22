# 参考读出与前缀森林参考的机制实验

完成100configs，运行623.33秒，全部市场/seed/四组保留。

预设主候选honest_readout没有取得整体增量：MQS相对固定森林参考退步0.7446%、相对rule-QRF退步0.6412%、相对同容量honest_no_readout退步0.9098%；完整CRPS方向也为负。相对同新Net的insample_readout也小幅退步0.0789% MQS。因此本设定没有证明prior-readout或prefix+readout的预测优势，不能将模型按test重新指定。

作为预设的次要控制，honest_no_readout相对固定参考改善MQS0.1637%/CRPS0.1305%（14/20、13/20胜），相对rule-QRF改善0.2662%/0.2031%（14/20、12/20胜）。这是可保留的控制结果，不是事后替换主模型，也不能仅凭它将收益全部归因于prefix honesty：本矩阵没有同容量insample_no_readout对照。

100个inner encoder均有非零梯度，57个非零epoch最终refit均更新encoder；75个正常readout权重有非零梯度，25个no-readout的prior权重梯度全部为0，符合标准化后零输入控制。40个prefix block的源目标时间严格早于block起点，20套prior scaler按phase eligible行重算一致；边界后u逐位同旧。110个旧文件及十份参考缓存SHA保持一致，checkpoint每模型固定12个query重建a/b、真实五seedmixture Q/CRPS误差均0，floor/额外零碰撞均0。

100条visited raw-relative诊断轨迹完整保留，33条best_raw_epoch不同于normalized best（主组9、no-readout6、insample9、reweight9），未用于本轮主选轮或额外拟合。NN、context scaler和phi仍按整个允许phase正常拟合，prefix只改变参考训练误差机制，不使旧实验失效。

新网络2636分配参数：LSTM16+context16+prior4经Linear4->8/Tanh后concat40->shared16->4zerohead。prior为clip(m0/s0)、clip(log s0)、log((ESS+1)/(count+1))、clip(skew_u)，仅当前成熟参考；phase fit eligible count>=32行估计mean/std，clip[-8,8]，无效sid置0且不入fit统计。no-readout在标准化后四维置0、同容量/同初始化。

honest三组显式读取forest_prior_honest_round5缓存，insample组显式读取旧forest_prior_round5缓存；没有替换旧文件。每phase fitindices/scaler/bank-D序/maturity/sid严格核对，t>=boundary的u与旧fullphase逐位相同。prefix是reference-training-bias机制对照，不宣称旧结果无效。

保持旧conditional transport/shape2/O(M)exactCRPS/samebranchp/64-min30-pat16/AdamW与正则。raw-relative validation轨迹仅作诊断，主normalized selector和refit不改。每seed支持a_s z+b_s合并0atom后排序，完整同F算五seed mixture Q/CRPS与实际非零概率；LSTMQR只有15q无完整CRPS。

| honest_readout / 参照 | MQS改善 | 完整CRPS改善 |
|---|---:|---:|
| honest_readout_vs_forest_reference_only | -0.7446% (11/20) | -0.7899% (11/20) |
| honest_readout_vs_rule_qrf | -0.6412% (9/20) | -0.7166% (9/20) |
| honest_readout_vs_qrf_original | -0.4657% (8/20) | -0.5601% (8/20) |
| honest_readout_vs_lstm_qr | +0.0205% (14/20) | 15q不可识别 |
| honest_readout_vs_old_forest_transport | -0.2441% (11/20) | -0.2900% (11/20) |
| honest_readout_vs_uniform_seq_transport | -0.1163% (12/20) | -0.2766% (10/20) |
| honest_readout_vs_honest_no_readout | -0.9098% (11/20) | -0.9216% (11/20) |
| honest_readout_vs_insample_readout | -0.0789% (10/20) | -0.0661% (10/20) |
| honest_readout_vs_honest_readout_reweight_only | -0.2703% (10/20) | -0.1528% (12/20) |

正值表示主候选更好。JSON/cells保留各组全部基线、逐seed与raw轨迹；NPZ存每seed及真实mixture Q/CRPS/actualp/branchp/a/b。各市场checkpoint有20模型state、honest/insample各自final bank/u、prior scaler eligible行映射与cache路径SHA。

历史测试已开发复用，结果不证明显著性/新颖性或发表成功；没有用测试挑模型/seed，也不把负结果推广为整个方向无效。
