# 第15轮后续对照：真实影线、正确份额目标与总量敏感性

更新：2026-10-21。结论：当前运输结构未达到预测成功标准，不打开2026保留期；继续其他模型结构。

## 修正后的辨识协议

固定22步总风险并不足以排除伪增益。若只优化1/5步QLIKE，头可从远期借预算补偿总量低估。纯合成反例真实Y=2*(.05,.15,.8)、冻结T=1，改份额为(.1,.3,.6)能让短期QLIKE改善−0.30685，却令正确份额损失恶化+0.18303、远段QLIKE恶化+0.37898。

新目标为`-sum_b(Y_b/c_t)*log(w_b)`，c_t=22×当期已知过去22日range均值。其逐X最优是条件桶均值之比，共同幅度乘子抵消；不使用未来实现总量归一化。有限样本/网络仍需控制，不能单靠目标就声称完全识别。

四个同为362参数的头：保持body的有限运输、同边势平稳分配、body-only、严格过去匹配库的wick安慰剂。比较均包括epoch0；2025选epoch。全部模型同float64 T且22步预测逐行严格一样。

## 两标的结果

以下为相对固定背景的损失差，负值较好，均是开发验证，n=228个共同origin。

| 标的/模型 | 风险份额主损失差 | 1步QLIKE差 | 5步QLIKE差 |
|---|---:|---:|---:|
| KRBN：body-matched运输 | 0 | 0 | 0 |
| KRBN：平稳分配 | −0.0000553 | −0.0022120 | −0.0003934 |
| KCCA：body-matched运输 | −0.0047297 | +0.0367036 | +0.0173396 |
| KCCA：平稳分配 | −0.0040543 | +0.0587632 | +0.0212129 |
| KCCA：body-only | −0.0007510 | +0.0451679 | +0.0032025 |
| KCCA：条件wick安慰剂 | −0.0017180 | +0.0028883 | +0.0121589 |

KRBN运输三个种子均选epoch0，恢复背景。KCCA份额目标有改善且好于body/安慰剂，但标准预测QLIKE恶化，不能把份额分数改善当预测成功。

## 已冻结份额的总量敏感性

为了检查是否只是旧总量不够好，保持所有份额/epoch不变，用同信息HGB的完整总风险重新缩放；所有模型依然共用同T，不重训或重新挑选头。KCCA新总量的22步QLIKE改善−0.01077，但运输对固定份额在1/5步仍更差；KRBN运输仍等于背景。未得到可挽救主结构的证据。

## 实现与数据检查

成熟标签、逐年OOF、过去donor与前缀不变、合法几何、零range中性、同容量、同边势平稳分布、零头嵌套、double预算均通过。首次真实执行发现body平方从已舍入eta重算有约6e-8差异；已改为直接复制原body列并重新检查。失败发生在任何候选拟合或结果保存前，旧日志保留；未放松精确保留断言。

2026在特征/目标构造前删除，始终未评分。结果用于内部候选筛选，不转阴性投稿。

入口：

- [条件对照脚本](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/wick_counterfactual_round15.py>)
- [KRBN完整结果](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/wick_counterfactual_round15_KRBN_allocation_development_results.json>)
- [KCCA完整结果](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/wick_counterfactual_round15_KCCA_allocation_development_results.json>)
- [总量敏感性结果](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/amplitude_sensitivity_round15_results.json>)
