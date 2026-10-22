# 样本量自适应参考收缩（无训练）

运行10.66秒。无任何参数训练；每市场仅按inner验证行从预先固定网格选一个配置，final只用该配置与固定0.05锚。

u=(1-lam)u_forest+lam*u_uniform，lam=lam0*kappa/(kappa+ESS)，ESS=1/sum_i w_i^2取自已成熟森林条件权重；fixed_0.05锚必须复现冻结forest_reference_only。

- CEA: 选中 `fixed_0.50`，inner log比值-0.0029；锚复现误差Q=0.00e+00，CRPS=1.00e-08
- GUANGZHOU: 选中 `fixed_0.50`，inner log比值-0.0262；锚复现误差Q=0.00e+00，CRPS=1.04e-08
- SHANGHAI: 选中 `ess_l0.75_k16`，inner log比值-0.0002；锚复现误差Q=0.00e+00，CRPS=1.22e-08
- HUBEI: 选中 `fixed_0.20`，inner log比值-0.0030；锚复现误差Q=0.00e+00，CRPS=3.62e-09
- EU: 选中 `fixed_0.50`，inner log比值-0.0147；锚复现误差Q=0.00e+00，CRPS=7.12e-09

| 对照 | MQS | 完整CRPS |
|---|---:|---:|
| chosen_vs_rule_qrf | -0.2271% (13/20) | -0.2662% (10/20) |
| chosen_vs_qrf_original | -0.0523% (13/20) | -0.1103% (12/20) |
| chosen_vs_lstm_qr | +0.4318% (15/20) | 不可识别 |
| chosen_vs_forest_reference_only | -0.3301% (12/20) | -0.3391% (8/20) |
| chosen_vs_forest_transport | +0.1683% (13/20) | +0.1586% (10/20) |
| chosen_vs_honest_no_readout | -0.4947% (10/20) | -0.4702% (8/20) |

正值表示收缩参考更好。这是参考构造实验，不涉及神经残差；历史测试已复用，不证明显著性或发表成功。
