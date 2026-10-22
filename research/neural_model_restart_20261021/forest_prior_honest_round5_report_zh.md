# 第五轮：prefix-forest训练查询参考cache交付

目标是改变训练残差看到的参考误差机制；不宣称旧评估无效，不改验证/测试参考。
inner/final仍对应D.tr/D.rf，四个block与RF设置均在fit前固定。每block森林只使用label t_i+h_i<block_start的源行，prefix scaler按所有h的该prefix估计。
首block与prefix单h<=1条时直接使用当前成熟NZ uniform；其余按原已选leaf、五seed、各80树重建；不丢合法NN query。
参考仍为.95prefixforest+.05当前成熟NZ uniform；uniform可以利用block开始后但不晚于query已成熟的病例。

实际forest fits=600，生成耗时114.171秒，CPU2。

| 市场 | MiB | inner block边界 | final block边界 |
|---|---:|---|---|
| CEA | 10.09 | [128, 207, 287, 367, 447] | [128, 245, 362, 479, 597] |
| GUANGZHOU | 122.14 | [128, 504, 880, 1256, 1633] | [128, 640, 1153, 1665, 2178] |
| SHANGHAI | 48.67 | [128, 378, 629, 879, 1130] | [128, 472, 817, 1162, 1507] |
| HUBEI | 31.37 | [128, 315, 503, 691, 879] | [128, 389, 650, 911, 1173] |
| EU | 37.05 | [128, 331, 535, 738, 942] | [128, 410, 692, 974, 1257] |

## 不变项与新元数据

- 所有q>=phase boundary的u/forest/u_forest/m0/v0/fallback逐值不变，inner D.va和final D.te的u位级相同。
- 全部原D keys、unique sid、完整NZ bank支持/phi/mature/allowed/count/uniform、fitindices、NN x/scaler均保持旧cache值；合法训练query eligibility不变。
- honest_blocks保存每block全prefix源索引/target成熟索引/scaler、每h森林源D索引及NZ到完整bank列的映射、query索引。
- query_block_sid指明训练sid所属block，边界后为-1；source_cache_sha256记录原cache。
- 所有读取旧源/结果/cache SHA前后一致；缓存只使用新前缀，已原子落盘。

## 训练参考实际变化

下表按所有count>=32的合法训练query统计新旧u的L1距离，阶段性分组；完整market/h明细另存数值表。

| 市场 | phase | 训练query数 | L1均值 | 改变query数 |
|---|---|---:|---:|---:|
| CEA | inner | 435 | 0.5180 | 435 |
| CEA | final | 735 | 0.7038 | 735 |
| GUANGZHOU | inner | 2849 | 0.8154 | 2849 |
| GUANGZHOU | final | 3939 | 0.8572 | 3939 |
| SHANGHAI | inner | 1826 | 0.8851 | 1826 |
| SHANGHAI | final | 2580 | 0.9160 | 2580 |
| HUBEI | inner | 1340 | 0.7258 | 1340 |
| HUBEI | final | 1928 | 0.7939 | 1928 |
| EU | inner | 1468 | 0.7770 | 1468 |
| EU | final | 2098 | 0.8049 | 2098 |

## 范围

这只让森林训练query参考以固定时间前缀构造。NN参数、NN scaler、phi和完整phase bank仍正常使用允许的整个fit partition；不称所有参数historical vintage。
旧参考训练机制与新机制的效果由后续NN训练比较；本cache任务没有选择新超参、重训NN或更改forecast参考。
消费接口与原cache一致，切换forest_prior_honest_round5前缀即可。
