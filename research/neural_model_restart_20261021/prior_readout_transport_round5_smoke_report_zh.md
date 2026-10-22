# 参考读出与前缀森林参考的机制实验

完成4configs，运行3.72秒，全部市场/seed/四组保留。

新网络2636分配参数：LSTM16+context16+prior4经Linear4->8/Tanh后concat40->shared16->4zerohead。prior为clip(m0/s0)、clip(log s0)、log((ESS+1)/(count+1))、clip(skew_u)，仅当前成熟参考；phase fit eligible count>=32行估计mean/std，clip[-8,8]，无效sid置0且不入fit统计。no-readout在标准化后四维置0、同容量/同初始化。

honest三组显式读取forest_prior_honest_round5缓存，insample组显式读取旧forest_prior_round5缓存；没有替换旧文件。每phase fitindices/scaler/bank-D序/maturity/sid严格核对，t>=boundary的u与旧fullphase逐位相同。prefix是reference-training-bias机制对照，不宣称旧结果无效。

保持旧conditional transport/shape2/O(M)exactCRPS/samebranchp/64-min30-pat16/AdamW与正则。raw-relative validation轨迹仅作诊断，主normalized selector和refit不改。每seed支持a_s z+b_s合并0atom后排序，完整同F算五seed mixture Q/CRPS与实际非零概率；LSTMQR只有15q无完整CRPS。

| honest_readout / 参照 | MQS改善 | 完整CRPS改善 |
|---|---:|---:|
| honest_readout_vs_forest_reference_only | +0.2758% (4/20) | +0.1039% (4/20) |
| honest_readout_vs_rule_qrf | +0.5131% (4/20) | +0.2246% (3/20) |
| honest_readout_vs_qrf_original | +1.1045% (3/20) | +0.6619% (3/20) |
| honest_readout_vs_lstm_qr | +0.4406% (3/20) | 15q不可识别 |
| honest_readout_vs_old_forest_transport | +0.2758% (4/20) | +0.1039% (4/20) |
| honest_readout_vs_uniform_seq_transport | -0.4641% (2/20) | -0.5389% (2/20) |
| honest_readout_vs_honest_no_readout | +0.2758% (4/20) | +0.1039% (4/20) |
| honest_readout_vs_insample_readout | +0.2758% (4/20) | +0.1039% (4/20) |
| honest_readout_vs_honest_readout_reweight_only | +0.2582% (4/20) | +0.1026% (4/20) |

正值表示主候选更好。JSON/cells保留各组全部基线、逐seed与raw轨迹；NPZ存每seed及真实mixture Q/CRPS/actualp/branchp/a/b。各市场checkpoint有20模型state、honest/insample各自final bank/u、prior scaler eligible行映射与cache路径SHA。

历史测试已开发复用，结果不证明显著性/新颖性或发表成功；没有用测试挑模型/seed，也不把负结果推广为整个方向无效。
