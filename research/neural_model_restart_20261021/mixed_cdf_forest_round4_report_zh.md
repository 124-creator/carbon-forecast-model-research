# 第四轮：CDF分裂、零质量与期限共享森林模型实验

四个固定候选均实际训练，五市场×五seed(11/22/33/44/55)，每forest80树、leaf25、n_jobs2。
per_h_CDF为每h单独depth6；pooled_MSE、pooled_CDF、pooled_zero_CDF为全部h共享depth8，保留context中的h列26。
175个最终forest fits。所有参数、fit-only网格、scaler和分割在首个fit前保存manifest；没有inner超参搜索或测试后设置调整。

本轮四组均未超过原QRF、rule-atom QRF或LSTMQR的整体MQS。最接近原QRF的是pooled_MSE，MQS/完整CRPS分别变差0.768%/0.603%；matched pooled_CDF又比pooled_MSE变差2.002%/1.835%。额外zero坐标相对pooled_CDF改善0.182%/0.209%，但不足以抵消主要差距，且seed22/33的两项组件收益反向。

pooled模型实际跨期限借样很强：按market/h汇总，其他期限概率质量均值约64%–75%。因此共享已经发生，当前结果需要检验共享程度和期限条件化，而不能归结为仅拼接了数据却没有用到其他h。当前设置按预先manifest完整报告，不据测试结果调整或选择模型。

CDF训练响应为33个fit-only经验分位去重所得区间上的sqrt(Δ)*I(z<=中点)，zero组追加sqrt(.5)*I(y==0)。
Multi-output squared-error只学习分区，是有限grid近似的integrated-CRPS impurity；最终分布仍为leaf内历史真实scalar z的概率，不使用CDF坐标均值拼概率。
最终完整weighted-ECDF CRPS为精确离散分布评分。所有ensemble先平均五seed完整概率权重，再算Q/CRPS。

## 对强基线的五seed ensemble结果

改善=1−20格score比值的几何均值，正值表示候选优于基线。

| 候选 | 基线 | MQS改善% | 胜格 | 完整CRPS改善% | 胜格 | WIS改善% |
|---|---|---:|---:|---:|---:|---:|
| per_h_CDF | qrf_original | -1.010 | 10/20 | -1.014 | 10/20 | -0.952 |
| per_h_CDF | qrf_rule_atom | -1.186 | 9/20 | -1.171 | 9/20 | -1.161 |
| per_h_CDF | lstm_qr | -0.521 | 15/20 | — | — | -0.391 |
| per_h_CDF | tilt_full_5seed | +0.354 | 12/20 | +0.508 | 14/20 | +0.158 |
| pooled_MSE | qrf_original | -0.768 | 8/20 | -0.603 | 8/20 | -1.013 |
| pooled_MSE | qrf_rule_atom | -0.944 | 8/20 | -0.760 | 8/20 | -1.222 |
| pooled_MSE | lstm_qr | -0.280 | 7/20 | — | — | -0.451 |
| pooled_MSE | tilt_full_5seed | +0.592 | 9/20 | +0.912 | 11/20 | +0.098 |
| pooled_CDF | qrf_original | -2.785 | 6/20 | -2.450 | 5/20 | -3.316 |
| pooled_CDF | qrf_rule_atom | -2.965 | 6/20 | -2.609 | 5/20 | -3.531 |
| pooled_CDF | lstm_qr | -2.288 | 5/20 | — | — | -2.742 |
| pooled_CDF | tilt_full_5seed | -1.397 | 7/20 | -0.907 | 8/20 | -2.181 |
| pooled_zero_CDF | qrf_original | -2.598 | 6/20 | -2.236 | 5/20 | -3.104 |
| pooled_zero_CDF | qrf_rule_atom | -2.778 | 6/20 | -2.395 | 5/20 | -3.318 |
| pooled_zero_CDF | lstm_qr | -2.102 | 5/20 | — | — | -2.531 |
| pooled_zero_CDF | tilt_full_5seed | -1.213 | 7/20 | -0.696 | 8/20 | -1.971 |

## 组件对照

pooled_CDF vs pooled_MSE隔离CDF分裂；pooled_zero_CDF vs pooled_CDF隔离额外zero-Brier坐标。
pooled_CDF vs per_h_CDF包含期限借样/深度分配改变，不能把共享收益统称zero模块。

| 候选 | 对照 | MQS改善% | 胜格 | 完整CRPS改善% | 胜格 |
|---|---|---:|---:|---:|---:|
| pooled_CDF | pooled_MSE | -2.002 | 4/20 | -1.835 | 4/20 |
| pooled_zero_CDF | pooled_CDF | +0.182 | 14/20 | +0.209 | 17/20 |
| pooled_zero_CDF | pooled_MSE | -1.816 | 4/20 | -1.623 | 4/20 |
| pooled_CDF | per_h_CDF | -1.757 | 8/20 | -1.422 | 7/20 |

## 逐seed对rule-atom QRF强基线

| 候选 | seed | MQS改善% | 完整CRPS改善% |
|---|---:|---:|---:|
| per_h_CDF | 11 | -1.335 | -1.304 |
| per_h_CDF | 22 | -1.336 | -1.302 |
| per_h_CDF | 33 | -1.088 | -1.099 |
| per_h_CDF | 44 | -1.529 | -1.464 |
| per_h_CDF | 55 | -1.214 | -1.204 |
| pooled_MSE | 11 | -0.968 | -0.727 |
| pooled_MSE | 22 | -0.831 | -0.600 |
| pooled_MSE | 33 | -1.285 | -1.152 |
| pooled_MSE | 44 | -1.070 | -0.816 |
| pooled_MSE | 55 | -1.537 | -1.277 |
| pooled_CDF | 11 | -3.138 | -2.767 |
| pooled_CDF | 22 | -2.642 | -2.400 |
| pooled_CDF | 33 | -3.222 | -2.866 |
| pooled_CDF | 44 | -3.441 | -3.024 |
| pooled_CDF | 55 | -3.136 | -2.832 |
| pooled_zero_CDF | 11 | -2.645 | -2.320 |
| pooled_zero_CDF | 22 | -2.946 | -2.648 |
| pooled_zero_CDF | 33 | -3.224 | -2.929 |
| pooled_zero_CDF | 44 | -3.076 | -2.630 |
| pooled_zero_CDF | 55 | -2.711 | -2.323 |

## 各市场/期限的ensemble MQS

| 市场 | h | per_h_CDF | pooled_MSE | pooled_CDF | pooled_zero_CDF | QRF | rule-QRF | LSTMQR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CEA | 1 | 0.0072885 | 0.0080556 | 0.0080283 | 0.0079018 | 0.0074369 | 0.0075033 | 0.0073242 |
| CEA | 5 | 0.0179697 | 0.0187627 | 0.0186710 | 0.0185703 | 0.0179154 | 0.0177823 | 0.0180024 |
| CEA | 10 | 0.0274830 | 0.0284476 | 0.0285765 | 0.0284957 | 0.0281508 | 0.0277788 | 0.0281015 |
| CEA | 20 | 0.0475072 | 0.0458268 | 0.0465038 | 0.0465342 | 0.0484604 | 0.0478840 | 0.0477587 |
| GUANGZHOU | 1 | 0.0154505 | 0.0159543 | 0.0160755 | 0.0160760 | 0.0155010 | 0.0154341 | 0.0164478 |
| GUANGZHOU | 5 | 0.0247048 | 0.0239395 | 0.0244518 | 0.0244386 | 0.0237894 | 0.0237762 | 0.0247986 |
| GUANGZHOU | 10 | 0.0267244 | 0.0279269 | 0.0287912 | 0.0287756 | 0.0263354 | 0.0263157 | 0.0277992 |
| GUANGZHOU | 20 | 0.0366532 | 0.0375320 | 0.0391291 | 0.0390957 | 0.0346114 | 0.0345648 | 0.0380890 |
| SHANGHAI | 1 | 0.0126110 | 0.0137367 | 0.0136596 | 0.0136188 | 0.0128439 | 0.0128234 | 0.0134287 |
| SHANGHAI | 5 | 0.0205922 | 0.0214126 | 0.0216210 | 0.0215552 | 0.0207627 | 0.0207528 | 0.0208712 |
| SHANGHAI | 10 | 0.0277400 | 0.0270252 | 0.0276266 | 0.0275286 | 0.0279147 | 0.0279364 | 0.0256083 |
| SHANGHAI | 20 | 0.0379624 | 0.0361984 | 0.0376115 | 0.0375139 | 0.0373471 | 0.0373484 | 0.0332852 |
| HUBEI | 1 | 0.0101812 | 0.0106714 | 0.0106560 | 0.0106633 | 0.0102824 | 0.0102804 | 0.0106376 |
| HUBEI | 5 | 0.0140089 | 0.0142819 | 0.0142979 | 0.0143109 | 0.0135503 | 0.0135547 | 0.0141871 |
| HUBEI | 10 | 0.0170487 | 0.0163701 | 0.0165011 | 0.0165000 | 0.0165243 | 0.0165314 | 0.0161868 |
| HUBEI | 20 | 0.0231507 | 0.0205980 | 0.0208731 | 0.0208882 | 0.0217878 | 0.0217299 | 0.0198762 |
| EU | 1 | 0.0106710 | 0.0113527 | 0.0116928 | 0.0116882 | 0.0107525 | 0.0107558 | 0.0107915 |
| EU | 5 | 0.0253146 | 0.0247421 | 0.0258068 | 0.0257961 | 0.0255575 | 0.0255776 | 0.0255491 |
| EU | 10 | 0.0364639 | 0.0337460 | 0.0354912 | 0.0354948 | 0.0360860 | 0.0360773 | 0.0373033 |
| EU | 20 | 0.0483068 | 0.0403766 | 0.0436178 | 0.0436036 | 0.0453396 | 0.0452681 | 0.0462948 |

## 实际跨期限借样质量

对每query计算sum_i w_i I(source_h_i != query_h)，下表为每market/h的均值与中位数。逐原点质量另存CSV。

| 市场 | 模型 | h | 跨h质量均值 | 跨h质量中位数 |
|---|---|---:|---:|---:|
| CEA | pooled_MSE | 1 | 0.7312 | 0.7347 |
| CEA | pooled_MSE | 5 | 0.7358 | 0.7375 |
| CEA | pooled_MSE | 10 | 0.7378 | 0.7346 |
| CEA | pooled_MSE | 20 | 0.7536 | 0.7478 |
| CEA | pooled_CDF | 1 | 0.7232 | 0.7227 |
| CEA | pooled_CDF | 5 | 0.7276 | 0.7235 |
| CEA | pooled_CDF | 10 | 0.7288 | 0.7226 |
| CEA | pooled_CDF | 20 | 0.7513 | 0.7506 |
| CEA | pooled_zero_CDF | 1 | 0.7007 | 0.7018 |
| CEA | pooled_zero_CDF | 5 | 0.7178 | 0.7129 |
| CEA | pooled_zero_CDF | 10 | 0.7162 | 0.7118 |
| CEA | pooled_zero_CDF | 20 | 0.7407 | 0.7433 |
| GUANGZHOU | pooled_MSE | 1 | 0.7208 | 0.7223 |
| GUANGZHOU | pooled_MSE | 5 | 0.7433 | 0.7443 |
| GUANGZHOU | pooled_MSE | 10 | 0.7435 | 0.7456 |
| GUANGZHOU | pooled_MSE | 20 | 0.7435 | 0.7454 |
| GUANGZHOU | pooled_CDF | 1 | 0.7106 | 0.7167 |
| GUANGZHOU | pooled_CDF | 5 | 0.7394 | 0.7424 |
| GUANGZHOU | pooled_CDF | 10 | 0.7384 | 0.7420 |
| GUANGZHOU | pooled_CDF | 20 | 0.7356 | 0.7402 |
| GUANGZHOU | pooled_zero_CDF | 1 | 0.7092 | 0.7153 |
| GUANGZHOU | pooled_zero_CDF | 5 | 0.7393 | 0.7422 |
| GUANGZHOU | pooled_zero_CDF | 10 | 0.7382 | 0.7419 |
| GUANGZHOU | pooled_zero_CDF | 20 | 0.7354 | 0.7401 |
| SHANGHAI | pooled_MSE | 1 | 0.7175 | 0.7294 |
| SHANGHAI | pooled_MSE | 5 | 0.7449 | 0.7469 |
| SHANGHAI | pooled_MSE | 10 | 0.7431 | 0.7456 |
| SHANGHAI | pooled_MSE | 20 | 0.7447 | 0.7466 |
| SHANGHAI | pooled_CDF | 1 | 0.6617 | 0.6849 |
| SHANGHAI | pooled_CDF | 5 | 0.7290 | 0.7345 |
| SHANGHAI | pooled_CDF | 10 | 0.7237 | 0.7286 |
| SHANGHAI | pooled_CDF | 20 | 0.7272 | 0.7320 |
| SHANGHAI | pooled_zero_CDF | 1 | 0.6442 | 0.6665 |
| SHANGHAI | pooled_zero_CDF | 5 | 0.7251 | 0.7292 |
| SHANGHAI | pooled_zero_CDF | 10 | 0.7196 | 0.7232 |
| SHANGHAI | pooled_zero_CDF | 20 | 0.7231 | 0.7270 |
| HUBEI | pooled_MSE | 1 | 0.7240 | 0.7248 |
| HUBEI | pooled_MSE | 5 | 0.7383 | 0.7395 |
| HUBEI | pooled_MSE | 10 | 0.7423 | 0.7430 |
| HUBEI | pooled_MSE | 20 | 0.7430 | 0.7428 |
| HUBEI | pooled_CDF | 1 | 0.7154 | 0.7169 |
| HUBEI | pooled_CDF | 5 | 0.7335 | 0.7335 |
| HUBEI | pooled_CDF | 10 | 0.7376 | 0.7377 |
| HUBEI | pooled_CDF | 20 | 0.7373 | 0.7382 |
| HUBEI | pooled_zero_CDF | 1 | 0.7148 | 0.7160 |
| HUBEI | pooled_zero_CDF | 5 | 0.7337 | 0.7340 |
| HUBEI | pooled_zero_CDF | 10 | 0.7380 | 0.7387 |
| HUBEI | pooled_zero_CDF | 20 | 0.7378 | 0.7386 |
| EU | pooled_MSE | 1 | 0.7382 | 0.7387 |
| EU | pooled_MSE | 5 | 0.7438 | 0.7449 |
| EU | pooled_MSE | 10 | 0.7445 | 0.7455 |
| EU | pooled_MSE | 20 | 0.7526 | 0.7488 |
| EU | pooled_CDF | 1 | 0.7299 | 0.7298 |
| EU | pooled_CDF | 5 | 0.7397 | 0.7393 |
| EU | pooled_CDF | 10 | 0.7416 | 0.7409 |
| EU | pooled_CDF | 20 | 0.7491 | 0.7478 |
| EU | pooled_zero_CDF | 1 | 0.7299 | 0.7298 |
| EU | pooled_zero_CDF | 5 | 0.7397 | 0.7393 |
| EU | pooled_zero_CDF | 10 | 0.7416 | 0.7409 |
| EU | pooled_zero_CDF | 20 | 0.7491 | 0.7478 |

## 解释与保存

- pooled支持中的每个样本均按其自身t_i+h_i检查成熟时间，并严格早于fitcut；source-h只提供已知标准化响应，query用自身sigma_t*sqrt(h)反变换。
- 共享不是未来标签输入，也不是测试期追加样本；保存了source horizon、mature索引及每query-h的来源h概率矩阵。
- QRF强基线曾在inner选leaf10/25/50，本组固定25；per_h_CDF与旧QRF对比不是完全匹配的分裂损失消融。三组pooled是匹配设置。
- LSTMQR为缓存五seed分位数平均，仅比较MQS/WIS/覆盖率，不从15tau编造完整CRPS。
- 每seed和ensemble Q/CRPS/p全保留；每市场/模型独立NPZ保留ensemble完整weights、support及训练来源；seed权重哈希已保存。
- 原点/h/y/scale与四套缓存逐值完全相等；所有读取冻结源/结果SHA前后一致。
- 方向只落在碳价zero-aware proper-score分裂与多期限借样结构；不称新forest分布族、首创CDF特征或复现DRF/MMD。

训练/评分/落盘耗时 226.8 秒。

运行时sanity：
```json
{
  "max_leaf_bruteforce_error": 9.310050369526479e-10,
  "max_independent_CDF_CRPS_error": 5.8281015152816806e-08,
  "max_probability_mass_error": 2.9516741051338613e-08,
  "max_CRPS_convexity_error": 0.0
}
```
