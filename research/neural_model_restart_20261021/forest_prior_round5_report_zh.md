# 第五轮条件森林参考缓存交付

五市场inner/final两phase已完成，使用冻结QRF实际leaf、seed11/22/33/44/55、每seed80树、CPU2。
inner只拟合D.tr，final只拟合D.rf；所有支持标签均按各自t_i+h_i严格早于phase boundary。
全体unique origin sid与seq_transport.prepare一致；每bank稳定非零顺序严格按原D索引映射，重复z不按值匹配。
成熟mask t_key+h<=query_t；mature NZ leaf质量0则成熟uniform fallback；u=.95forest+.05uniform。
缺失该h目标的sid用成熟uniform并标记无效；无成熟NZ的sid全0，沿用count>=32过滤。

200个必要forest重建和缓存生成耗时 72.543 秒。

| 市场 | cache MiB | 全局Nsid | inner各h Nbank | final各h Nbank |
|---|---:|---:|---|---|
| CEA | 9.86 | 566 | 84/117/122/125 | 151/192/195/199 |
| GUANGZHOU | 121.36 | 1355 | 721/746/743/741 | 990/1019/1015/1013 |
| SHANGHAI | 48.15 | 819 | 442/481/485/487 | 607/658/669/673 |
| HUBEI | 30.95 | 668 | 359/368/367/365 | 500/515/513/512 |
| EU | 36.60 | 723 | 405/405/402/396 | 561/562/560/554 |

## final test重现检查

- 原QRF非零leaf概率按严格D索引重排后最大差：0。
- gamma0条件mark最大差：1.86e-09。
- gamma0+同p_rule分位数最大差：0；各格MQS完全相同。
- gamma0完整CRPS最大差：7.84e-08（float32运算误差）。
- 所有冻结旧源/结果SHA前后一致；落盘读回重新核对sid、scaler、T.bank、maturity mask、概率归一及5%uniform保留。

## 消费与边界

每市场独立PT已原子落盘，phases.inner/final均可加载。接口文档说明所有字段；load_cache(market,phase,D)只读并断言D/sid一致。
val只能消费inner forest；测试和最终重拟合消费final forest。
森林/scaler/phi按整个允许fit区间估计，成熟mask只限制检索响应，不宣称训练query全参数vintage。
本任务仅构建新模型需要的参考，没有重调leaf或重新选择测试模型。
