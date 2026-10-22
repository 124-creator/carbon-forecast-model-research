# LSTM条件位置尺度与矩归一经验形状输运

完成100fit，运行663.25秒。四消融/所有市场/seed均保留，不按test选择。

完整模型相对无序列消融MQS改善0.4519%，相对仅位置尺度改善0.3499%，相对旧冻结倾斜网络改善1.0058%，也比LSTM-QR改善0.1366%。但它仍落后原始QRF约0.3489%、规则概率QRF约0.5242%；完整CRPS对这两个强基线也为负。因此是有小内部增量的候选，尚未达到全面胜过强分布基线的目标。单seed内部方向波动，全部seed均保留，不能将ensemble正收益描述为每个seed都有效。

100个内层编码器都有非零梯度，58个非零epoch最终重拟合均有编码器更新，42个最终epoch0由验证选出。四组均分配2468参数。共10537707次几何query访问，最小均匀方差0.0595555、最小倾斜方差0.0582295，两个floor触发次数均0。最终支持未观察到额外零碰撞，真实非零质量与固定branch概率最大差4.44e-16。58个旧文件SHA保持不变；从checkpoint每模型重建12个固定query的a/b、五seed真实mixture Q及CRPS误差全部为0。

每query仅用t_key+h<=query_origin的至少32个非零响应。均匀成熟子集给m0/s0；有界两系数倾斜给w，再减mw除sw并施加目标m0+s0*delta、s0*exp(ell)。LSTM16、ctx16、shared16及4zerohead同分配参数；no-seq将64步输入置0，loc-scale-only将beta置0，reweight-only不施加affine。

训练使用给定O(M)完整CRPS，max64/min30/pat16，AdamW.001/wd.003及beta²正则.003，same inner equal-horizon normalized pinball选epoch0和full-precut refit。方差floor1e-8仅数值保护，实际最小方差/floor query访问次数逐fit保存，不声称退化subset有单位方差。

p_rule是固定branch mixing概率；输运mark碰到0时，最终非零概率按完整F重算用于Brier，碰撞count/mass另存。五seed支持按query变化，合并全部a_s*z+b_s及零atom后排序，精确计算同一mixture的Q/CRPS，没有平均参数/seed分数/quantiles冒充mixture。

| seq_transport / 参照 | MQS改善 | 完整CRPS改善 |
|---|---:|---:|
| seq_transport_vs_no_sequence_transport | +0.4519% (14/20) | +0.4088% (12/20) |
| seq_transport_vs_seq_location_scale_only | +0.3499% (12/20) | +0.3241% (13/20) |
| seq_transport_vs_seq_reweight_only | +0.6294% (9/20) | +0.7335% (8/20) |
| seq_transport_vs_frozen_tilt | +1.0058% (12/20) | +1.2279% (12/20) |
| seq_transport_vs_qrf_original | -0.3489% (8/20) | -0.2827% (9/20) |
| seq_transport_vs_rule_qrf | -0.5242% (7/20) | -0.4388% (8/20) |
| seq_transport_vs_lstm_qr | +0.1366% (12/20) | 不可由15q识别 |
| seq_transport_vs_uniform_rule | +1.9790% (13/20) | +1.7548% (15/20) |

正值表示seq_transport更好。所有模型单seed与mixture Q/CRPS/最终p、branch p及a/b另存NPZ；每市场checkpoint包含全部seed state/refitbank/scaler，可重建w。canonical origin/h/target/scale严格对齐四类冻结参考，所有读取旧文件SHA保持不变。

这是历史复用测试上的模型探索，不证明新颖性、显著性或可发表成功；负结果不否定整个ML/DL方向。
