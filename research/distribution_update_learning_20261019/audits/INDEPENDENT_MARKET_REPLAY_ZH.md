# 市场实验独立重放与结果核验

日期：2026-10-19。状态：总体通过。未重新拟合，未读取源 CSV 或保留期文件。

## 结论

132 个 checkpoint 的全部 calibration/test factor 共 63,186 个数值完成重放；独立定义 MLP/GRU 架构并加载保存权重，没有 import 实验源码或 analyze_results.py。

全部 21 个方法、22 个 cell、6 个 learner 的 normalized CRPS、price CRPS 与价格均值 MSE逐点外部公式核验通过，累计 628,362 条方法评分。原子分布使用等权中点 pinball 恒等式；Gaussian/lognormal 使用已通过外部积分审计的数学公式。

主候选相对同等 joint budget 的 raw 对照，22 个 cell 等权相对差为 **-0.00045856048351**，即 **-0.04585605%**。固定 cell 内配对环块区间为 **[-0.0005899451103215872, -0.0003498176661173693]**。

只有 **GUANGZHOU_2021_h20** 改善，cell 内相对差为 **-1.00883306%**；其余 21 个 cell 按 1e−10 容差持平。改进出现于该 cell 的六个 learner，不能把六个重复结构当作六个独立市场证据。

该 CI 固定了 22 个市场×年份×期限 cell，只在各 cell 内以 2h 环块重采样起点。它不重采样市场、不捕捉跨 cell 相关性，不能证明跨市场可迁移增益或未来分布变化下的改善。60% cell 改善条件失败，本轮维持 STOP。

计数应区分：63 份校准结果至少一坐标选择非零控制系数，不等于 63 个实际预测更新。主方法只有 14 个非零 gate，只有 6 个作业的已发出 gate 路径不同于 same-joint raw，且全部位于同一个 cell。

## 训练与 epoch 0 的解释

46/132 选择了 epoch 0 初始化权重，这是验证集择优后的回退。全部 132 个作业均有 epoch 0–60 的 61 条完整验证历史，60 个训练 epoch，共计 7,920 epoch、116,640 个 optimizer steps，与每作业 60×ceil(ntrain/64) 完全一致。每份历史均存在实际数值变化。

本次重放没有 optimizer、backward 或 fitting 调用，新增训练数与梯度步数均为零。训练完成数量来自已存完整收据、历史及源码 hash 对账；不把重放包装成重新观察了全部历史梯度执行。

## 数值差异

```json
{
  "path_sum_to_endpoint": 1.7763568394002505e-15,
  "log_price_target_to_normalized_label": 5.051514762044462e-14,
  "basis_partition": 3.3306690738754696e-16,
  "standardized_train_feature_mean": 2.3561024492769793e-12,
  "standardized_train_feature_sd": 5.649924972317422e-13,
  "checkpoint_calibration_factor": 0,
  "checkpoint_test_factor": 0,
  "independent_a_calibration": 6.661338147750939e-16,
  "independent_b_calibration": 8.881784197001252e-16,
  "independent_a_test": 7.771561172376096e-16,
  "independent_b_test": 7.771561172376096e-16,
  "a_b_quadratic_all_mixture_scores": 2.6645352591003757e-15,
  "all_normalized_CRPS_external_formula": 4.440892098500626e-15,
  "all_price_CRPS_external_formula": 4.529709940470639e-14,
  "all_price_mean_squared_error": 3.296918293926865e-12,
  "saved_gate_weights_to_issued_alpha": 0,
  "atomic_coverage": 0,
  "atomic_width": 0,
  "random_walk_squared_error": 0,
  "recorded_control_cap": 2.7755575615628914e-17,
  "private_receipt_metric_means": 0,
  "public_cell_metrics": 7.105427357601002e-15,
  "summary_raw_mean": 3.0357660829594124e-17,
  "summary_raw_interval": 3.885780586188048e-16,
  "summary_F0_mean": 3.0357660829594124e-17,
  "summary_F0_interval": 2.8796409701215e-16,
  "summary_secondary_ratios": 2.220446049250313e-16,
  "primary_mean": 0,
  "primary_interval": 1.6642503347652493e-17,
  "primary_no_prefix_mean": 0,
  "primary_no_prefix_interval": 8.55435514091063e-17
}
```

## 方法比较

以下均为先平均 cell 内六个 learner 的 loss，再计算每 cell 比值并对 22 个 cell 等权平均；不是先混合六个 CDF 再评分。

| 方法 | CRPS / F0 | 相对 same-joint raw | 改善 / 持平 / 变差 cell | price CRPS / F0 |
|---|---:|---:|---:|---:|
| F0 | 1.0000000000 | -0.0000196250 | 2 / 19 / 1 | 1.0000000000 |
| F1 | 1.0043921746 | 0.0043714603 | 9 / 6 / 7 | 1.0053710958 |
| equal_mixture | 1.0007141095 | 0.0006940880 | 10 / 6 / 6 | 1.0010855477 |
| empirical_scalar | 1.0015288886 | 0.0015085159 | 8 / 9 / 5 | 1.0014988416 |
| empirical_gate3 | 1.0011223257 | 0.0011024973 | 10 / 7 / 5 | 1.0009686133 |
| raw_block_gate3 | 1.0000220156 | 0.0000023755 | 1 / 19 / 2 | 1.0000266032 |
| raw_block_gate3_same_joint_budget | 1.0000196388 | 0.0000000000 | 0 / 22 / 0 | 1.0000228453 |
| path_block_gate3_delta005 | 0.9995610784 | -0.0004585605 | 1 / 21 / 0 | 0.9995765399 |
| path_block_gate3_delta0 | 0.9988567301 | -0.0011629088 | 5 / 16 / 1 | 0.9989343269 |
| path_block_gate3_delta0025 | 0.9993526726 | -0.0006669662 | 2 / 20 / 0 | 0.9993722782 |
| path_block_gate3_delta010 | 0.9999431717 | -0.0000764671 | 1 / 21 / 0 | 0.9999479517 |
| path_block_scalar_delta005 | 0.9994822498 | -0.0005374464 | 3 / 17 / 2 | 0.9995585590 |
| path_fullcap_gate3_delta005 | 1.0000182832 | -0.0000013546 | 2 / 19 / 1 | 1.0000207428 |
| path_uncapped_gate3_delta005 | 1.0000182832 | -0.0000013546 | 2 / 19 / 1 | 1.0000207428 |
| path_no_prefix_gate3_delta005 | 1.0000220156 | 0.0000023755 | 1 / 19 / 2 | 1.0000266032 |
| BOA_CDF_frozen | 1.0004385502 | 0.0004180846 | 11 / 6 / 5 | 1.0006535013 |
| BOA_CDF_online | 0.9989559968 | -0.0010642268 | 11 / 6 / 5 | 0.9990919249 |
| quantile_BOAG_online | 1.0041863200 | 0.0041658522 | 6 / 0 / 16 | 1.0049808780 |
| FHS | 1.0053104046 | 0.0052906668 | 9 / 0 / 13 | 1.0030008410 |
| EWMA_Normal | 1.0365242665 | 0.0365031703 | 7 / 0 / 15 | 1.0422589653 |
| GARCH_Normal | 1.2077344710 | 0.2077017546 | 4 / 0 / 18 | 1.1908282750 |

## 边界与身份

核验 checkpoint 内的 weights、sigma0、feature_mean、feature_sd 与 prepared 数组完全一致，train X 的中心与标准差符合标准化定义；各角色标签成熟边界与 ≤2023 日期限制通过。没有重读原始价格，所以不把这次复核表述为重新认证数据来源。

Gaussian 预测的 coverage/width 在聚合层对账；其 normalized CRPS、price CRPS、均值 MSE 是全量独立重算。两个原子预测方法另全量重算了 coverage/width。

JSON 收据列出全部 checkpoint/prediction/prepared hashes、冻结源码身份、bootstrap 约定与计数；不写逐点价格、预测因子或损失。

冻结后的变化已显式保留：run_market.py 在 22 个 prepared panel 完成后加强了日期防护，改为先仅读日期并拒绝 2024+ 输入，再读价格。审计将该精确代码块逆换回旧顺序，整份文件 SHA256 严格恢复 MARKET_PREFIT 的 deca05dea6…，证明没有混入其他变化；其余冻结模块逐字节未变。修复和零重训记录来自 provenance/POST_START_AUDIT_FIXES.json。首次核验遇到该 hash 差异时先停止，第二轮数值检查通过但保留待解释状态，最后完成精确逆 patch 证明后才报告整体通过。

复现命令：

```bash
python -B research/distribution_update_learning_20261019/audits/independent_market_replay.py
```
