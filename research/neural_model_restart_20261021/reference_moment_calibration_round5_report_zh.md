# 第五轮：全局reference mean/scale校准控制

这是已知的低成本校准控制，不是新模型；没有训练森林或NN，不改主NN协议。每市场只在inner D.va从预冻结36组选择一个共享全部h的参数对，final只应用该pair与identity对照。
选择准则是四个h的raw MQS相对于各自固定reference(1,1)的mean log ratio，未按test选择；全部180个val候选分数保留。
conditional mark权重u和p_rule保持固定，仅M=lambda_s*z+(lambda_mu-lambda_s)*m0。真正排序NZ mark+零atom后统一计算Q/完整CRPS/actual p；目标mean=0不强制median=0。

| 市场 | lambda_mu | lambda_s | inner几何MQS改善 |
|---|---:|---:|---:|
| CEA | 0 | 2 | +11.6020% |
| GUANGZHOU | 0 | 1 | +2.6982% |
| SHANGHAI | 0.75 | 0.75 | +1.4095% |
| HUBEI | 1.25 | 1 | +0.0585% |
| EU | 0.5 | 1 | +1.3010% |

| 校准reference 对参照 | 测试MQS改善 | 测试完整CRPS改善 |
|---|---:|---:|
| forest_reference_only | -1.6269% (7/20) | -1.6647% (7/20) |
| rule_qrf | -1.5225% (8/20) | -1.5908% (8/20) |
| forest_transport_normalized | -1.1220% (10/20) | -1.1604% (10/20) |
| forest_transport_rawselected | -1.5228% (7/20) | -1.4903% (9/20) |
| lstm_qr | -0.8551% (13/20) | 15q不可识别 |

正值表示校准reference更好；20个market×h格等权几何汇总。LSTMQR没有可用完整CRPS，不由15个分位点构造替代CRPS。
固定(1,1) reference的Q逐值复现冻结forest_reference_only，CRPS/p误差及独立CDF积分核对保存在JSON。
m0用归一后的同一F权重以float64计算，避免缓存float32均值舍入影响目标矩；固定reference权重首先复用原算子的float32归一以严格匹配anchor。
JSON保留36val scores/market、目标矩误差、实际零碰撞、全部20格和源SHA；NPZ保留selected/fixed以及只读基线Q/CRPS/p，另存selected/anchor validation预测。
运行9.210秒；forest fits=0，NN fits=0，所有读取旧源/基线/cache SHA一致。

该控制用于判断prior-readout神经模型是否超出简单全局mean/scale校准；不改变已有NN训练或选择协议。
