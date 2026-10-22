# Prefix与moment transport的同初始化匹配机制对照

完成50新configs及原25模型冻结参照，运行3.31秒。旧参照未重训，两个新control各5市场×5seed，全保留。

两control均新Net2636参数、same-seed初始state与honest_no_readout逐值一致，prior标准化后四维强制0。matched_insample_no_readout只换旧insample训练参考；matched_honest_no_readout_reweight_only仍honest参考，且严格a1/b0，隔离affine。显式缓存路径、原phase与成熟mask/32NZ/64past/36ctx/selector/refit协议全部不变。

全部visited raw-relative轨迹仅作诊断，主选择仍normalized；没有按test选择seed/epoch。五seed合并query-specific支持与零atom，精确同F计算Q/CRPS/实际非零质量。LSTMQR仅15q无完整CRPS。

| 冻结honest_no_readout / 对照 | MQS改善 | 完整CRPS改善 |
|---|---:|---:|
| frozen_honest_no_readout_vs_matched_insample_no_readout | +0.0856% (4/20) | +0.0301% (4/20) |
| frozen_honest_no_readout_vs_matched_honest_no_readout_reweight_only | +0.0689% (4/20) | +0.0297% (4/20) |
| frozen_honest_no_readout_vs_forest_reference_only | +0.0856% (4/20) | +0.0301% (4/20) |
| frozen_honest_no_readout_vs_rule_qrf | +0.3234% (4/20) | +0.1509% (3/20) |
| frozen_honest_no_readout_vs_qrf_original | +0.9159% (3/20) | +0.5886% (3/20) |
| frozen_honest_no_readout_vs_lstm_qr | +0.2507% (3/20) | 15q不可识别 |

正值表示冻结honest_no_readout更好。参数分配与随机初始化精确匹配后才解释机制，不把原B不同初始化/常量分支差异称为prefix贡献。新control与固定ref/ruleQRF/rawQRF/LSTM的全部分数另存JSON/cells。

no-readout推断时prior_encoder(0)=tanh(prior_bias)是常量8向量，shared可将W_prior*tanh(prior_bias)并入shared bias；只说明函数等价，未删模块、未改训练/初始化。人工fold误差已记录。

checkpoint可按显式cache/path/scaler/mapping重建新10state/market；旧25state继续引用原checkpoint。全部旧文件SHA保持，历史测试已复用，正增量仅是开发候选，尚需冻结滚动复核。
