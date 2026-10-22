# 仅验证集选择context的独立组合

每market/seed比较既有best validation normalized MQS，更低者入选；差<=1e-12选state。零次重训，没有用test选context。选出seed分布后重新合并所有knots+0，精确积分同一F的CRPS，CDF二分同一F求quantiles。

| market | seed | aug best val | state best val | chosen |
|---|---:|---:|---:|---|
| CEA | 11 | 0.556249877 | 0.556249877 | neural_atom_no_price_calendar |
| CEA | 22 | 0.556249877 | 0.556249877 | neural_atom_no_price_calendar |
| GUANGZHOU | 11 | 0.428865392 | 0.435427188 | neural_atom_spline |
| GUANGZHOU | 22 | 0.429907021 | 0.434797275 | neural_atom_spline |
| SHANGHAI | 11 | 0.272397664 | 0.269145473 | neural_atom_no_price_calendar |
| SHANGHAI | 22 | 0.278777989 | 0.275479814 | neural_atom_no_price_calendar |
| HUBEI | 11 | 0.463605765 | 0.463978339 | neural_atom_spline |
| HUBEI | 22 | 0.463625994 | 0.463978339 | neural_atom_spline |
| EU | 11 | 0.472200733 | 0.477023226 | neural_atom_spline |
| EU | 22 | 0.477233555 | 0.477233555 | neural_atom_no_price_calendar |

| 对照 | MQS改善 | 完整CRPS改善 |
|---|---:|---:|
| contextselected_vs_all_aug | -0.0860% (1/20) | -0.1699% (1/20) |
| contextselected_vs_all_state | -0.6441% (5/20) | -0.6982% (5/20) |
| contextselected_vs_empirical_constant | +0.5947% (13/20) | +0.1351% (13/20) |
| contextselected_vs_double_normal | +0.6907% (10/20) | +1.0678% (11/20) |

验证支持aug/state各5个设置；这与test更喜欢全state并不相同，不能以test反向取代这个验证规则。全部原点/目标/h与原实验一致，旧SHA全部保留。
