# 匹配机制控制：冻结参照、零输入与true mixture核对

完成50新configs，原25 frozen honest_no_readout未重训，逐seed与mixture Q/CRPS/p/a/b直接拷贝核对一致。4307共同原点×期限键和全部旧SHA一致。

两控制2636参数/同5seed初始化与原架构精确相同；standardized prior输入和prior weight梯度全0。reweight控制所有a精确1、b精确0。checkpoint每模型12固定query重建a/b误差0、true5mixture Q误差0、CRPS误差0。

50条visited raw诊断全留，20条best_raw不同于norm；没有改本轮selector/refit。reference/weighted floor访问0/0，额外零碰撞0。

| 冻结honest_no_readout / 对照 | MQS改善 | 完整CRPS改善 |
|---|---:|---:|
| frozen_honest_no_readout_vs_matched_insample_no_readout | +0.6800% (12/20) | +0.6541% (12/20) |
| frozen_honest_no_readout_vs_matched_honest_no_readout_reweight_only | +0.3696% (10/20) | +0.6009% (9/20) |
| frozen_honest_no_readout_vs_forest_reference_only | +0.1637% (14/20) | +0.1305% (13/20) |
| frozen_honest_no_readout_vs_rule_qrf | +0.2662% (14/20) | +0.2031% (12/20) |
| frozen_honest_no_readout_vs_qrf_original | +0.4401% (14/20) | +0.3582% (12/20) |
| frozen_honest_no_readout_vs_lstm_qr | +0.9219% (14/20) | 15q不可识别 |

## 同seed机制方向，全5保留

| seed | 对照 | MQS改善 | CRPS改善 |
|---|---|---:|---:|
| 11 | matched_insample_no_readout | +0.5429% | +0.5049% |
| 11 | matched_honest_no_readout_reweight_only | +0.0869% | +0.3346% |
| 22 | matched_insample_no_readout | +0.3961% | +0.2908% |
| 22 | matched_honest_no_readout_reweight_only | +0.1433% | +0.3768% |
| 33 | matched_insample_no_readout | +0.4089% | +0.3733% |
| 33 | matched_honest_no_readout_reweight_only | +0.2752% | +0.4969% |
| 44 | matched_insample_no_readout | -0.4520% | -0.6037% |
| 44 | matched_honest_no_readout_reweight_only | -0.8338% | -0.6794% |
| 55 | matched_insample_no_readout | +0.6649% | +0.7370% |
| 55 | matched_honest_no_readout_reweight_only | +0.4577% | +0.6469% |

noReadout常量支路仅在推断函数层面可折叠，训练参数与优化路径未改。历史test已开发复用，正方向仅作为开发候选，需要冻结后滚动复核，不将先前prior-readout primary改名为成功。
