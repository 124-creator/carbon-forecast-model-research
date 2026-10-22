# Prefix与moment transport的同初始化匹配机制对照

完成50新configs及原25模型冻结参照，运行322.71秒。旧参照未重训，两个新control各5市场×5seed，全保留。

匹配后，冻结honest_no_readout相对insample控制改善MQS0.6800%/CRPS0.6541%，相对honest纯重加权控制改善0.3696%/0.6009%。这给出了本开发设置下prefix参考和affine输运的正向整体增量，容量与初始化已匹配。它仍是原实验的次要候选，不把失败的prior-readout主假设改名为成功。

单seed方向不是全部稳定：两项机制对照均4/5个seed为正，seed44的prefix对照MQS/CRPS为-0.4520%/-0.6037%，affine对照为-0.8338%/-0.6794%。市场增量主要来自上海：prefix MQS+3.3019%、affine+3.1404%；后者在广州为-1.0693%、湖北-0.3519%。不能把整体均值解释为所有市场/seed都改善。

核对通过：原25模型逐seed与mixture预测直接拷贝一致，未重训；4307共同键、144旧文件及全部参考缓存SHA一致。50个inner encoder均有真梯度，26个非零epoch最终refit均更新encoder；prior权重梯度均0，reweight全部a精确1/b精确0。checkpoint每模型12固定query重建a/b与真实五seedmixture Q/CRPS误差均0，floor/额外零碰撞均0。50条raw诊断轨迹均保留，20条best_raw不同于原norm选择，未用于本轮选轮或额外拟合。

两control均新Net2636参数、same-seed初始state与honest_no_readout逐值一致，prior标准化后四维强制0。matched_insample_no_readout只换旧insample训练参考；matched_honest_no_readout_reweight_only仍honest参考，且严格a1/b0，隔离affine。显式缓存路径、原phase与成熟mask/32NZ/64past/36ctx/selector/refit协议全部不变。

全部visited raw-relative轨迹仅作诊断，主选择仍normalized；没有按test选择seed/epoch。五seed合并query-specific支持与零atom，精确同F计算Q/CRPS/实际非零质量。LSTMQR仅15q无完整CRPS。

| 冻结honest_no_readout / 对照 | MQS改善 | 完整CRPS改善 |
|---|---:|---:|
| frozen_honest_no_readout_vs_matched_insample_no_readout | +0.6800% (12/20) | +0.6541% (12/20) |
| frozen_honest_no_readout_vs_matched_honest_no_readout_reweight_only | +0.3696% (10/20) | +0.6009% (9/20) |
| frozen_honest_no_readout_vs_forest_reference_only | +0.1637% (14/20) | +0.1305% (13/20) |
| frozen_honest_no_readout_vs_rule_qrf | +0.2662% (14/20) | +0.2031% (12/20) |
| frozen_honest_no_readout_vs_qrf_original | +0.4401% (14/20) | +0.3582% (12/20) |
| frozen_honest_no_readout_vs_lstm_qr | +0.9219% (14/20) | 15q不可识别 |

正值表示冻结honest_no_readout更好。参数分配与随机初始化精确匹配后才解释机制，不把原B不同初始化/常量分支差异称为prefix贡献。新control与固定ref/ruleQRF/rawQRF/LSTM的全部分数另存JSON/cells。

no-readout推断时prior_encoder(0)=tanh(prior_bias)是常量8向量，shared可将W_prior*tanh(prior_bias)并入shared bias；只说明函数等价，未删模块、未改训练/初始化。人工fold误差已记录。

checkpoint可按显式cache/path/scaler/mapping重建新10state/market；旧25state继续引用原checkpoint。全部旧文件SHA保持，历史测试已复用，正增量仅是开发候选，尚需冻结滚动复核。
