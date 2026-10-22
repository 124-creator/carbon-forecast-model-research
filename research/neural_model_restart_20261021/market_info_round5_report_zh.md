# 跨市场信息对条件参考本身的作用（无训练）

127.1秒。仅改变条件信息列，参考构造（成熟掩码、γ=.05均匀库、规则零原子、5种子、叶子大小）全部不变。

base必须逐位复现冻结forest_reference_only；cross为4个其它市场×{5日收益,20日波动}按as-of对齐到本市场日期；perm为同样列但按本市场日期打乱（控制“只是多了列”）。

- CEA: 复现误差 Q=2.5e-03, CRPS=4.2e-05；cross MQS相对base +0.040%，perm +0.085%
- GUANGZHOU: 复现误差 Q=5.9e-03, CRPS=9.5e-05；cross MQS相对base -1.553%，perm -0.120%
- SHANGHAI: 复现误差 Q=5.1e-03, CRPS=3.0e-05；cross MQS相对base +0.484%，perm -0.103%
- HUBEI: 复现误差 Q=1.8e-02, CRPS=1.2e-04；cross MQS相对base -0.614%，perm -0.068%
- EU: 复现误差 Q=3.4e-03, CRPS=1.7e-04；cross MQS相对base +0.007%，perm +0.093%

| 对比 | MQS | 完整CRPS |
|---|---:|---:|
| base vs rule_qrf_MQS | +0.1028% (12/20) |
| base vs rule_qrf_CRPS | +nan% (0/0) |
| cross vs rule_qrf_MQS | -0.2215% (11/20) |
| cross vs rule_qrf_CRPS | +nan% (0/0) |
| cross vs base_MQS | -0.3247% (10/20) |
| cross vs base_CRPS | -0.2570% (9/20) |
| perm vs rule_qrf_MQS | +0.0804% (10/20) |
| perm vs rule_qrf_CRPS | +nan% (0/0) |
| perm vs base_MQS | -0.0225% (9/20) |
| perm vs base_CRPS | -0.0452% (7/20) |

若cross未显著超过perm，则跨市场列没有提供可用信息；若接近0，说明条件参考已饱和于本市场状态。历史测试复用，不证明显著性。
