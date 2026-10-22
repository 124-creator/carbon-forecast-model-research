# LSTM条件位置尺度与矩归一经验形状输运

完成4fit，运行2.71秒。四消融/所有市场/seed均保留，不按test选择。

每query仅用t_key+h<=query_origin的至少32个非零响应。均匀成熟子集给m0/s0；有界两系数倾斜给w，再减mw除sw并施加目标m0+s0*delta、s0*exp(ell)。LSTM16、ctx16、shared16及4zerohead同分配参数；no-seq将64步输入置0，loc-scale-only将beta置0，reweight-only不施加affine。

训练使用给定O(M)完整CRPS，max64/min30/pat16，AdamW.001/wd.003及beta²正则.003，same inner equal-horizon normalized pinball选epoch0和full-precut refit。方差floor1e-8仅数值保护，实际最小方差/floor query访问次数逐fit保存，不声称退化subset有单位方差。

p_rule是固定branch mixing概率；输运mark碰到0时，最终非零概率按完整F重算用于Brier，碰撞count/mass另存。五seed支持按query变化，合并全部a_s*z+b_s及零atom后排序，精确计算同一mixture的Q/CRPS，没有平均参数/seed分数/quantiles冒充mixture。

| seq_transport / 参照 | MQS改善 | 完整CRPS改善 |
|---|---:|---:|
| seq_transport_vs_no_sequence_transport | +0.0000% (0/20) | +0.0000% (0/20) |
| seq_transport_vs_seq_location_scale_only | +0.0000% (0/20) | +0.0000% (0/20) |
| seq_transport_vs_seq_reweight_only | +0.0000% (0/20) | +0.0000% (0/20) |
| seq_transport_vs_frozen_tilt | -1.5946% (1/20) | -1.2398% (1/20) |
| seq_transport_vs_qrf_original | +1.5613% (4/20) | +1.1944% (4/20) |
| seq_transport_vs_rule_qrf | +0.9727% (2/20) | +0.7594% (3/20) |
| seq_transport_vs_lstm_qr | +0.9004% (4/20) | 不可由15q识别 |
| seq_transport_vs_uniform_rule | +0.0000% (0/20) | +0.0000% (2/20) |

正值表示seq_transport更好。所有模型单seed与mixture Q/CRPS/最终p、branch p及a/b另存NPZ；每市场checkpoint包含全部seed state/refitbank/scaler，可重建w。canonical origin/h/target/scale严格对齐四类冻结参考，所有读取旧文件SHA保持不变。

这是历史复用测试上的模型探索，不证明新颖性、显著性或可发表成功；负结果不否定整个ML/DL方向。
