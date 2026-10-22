# Prefix-forest训练查询参考cache接口

新cache命名为 `forest_prior_honest_round5_{MARKET}.pt`，原 `forest_prior_round5` 的全部文件只读且冻结。接口保持原schema，消费端只需切换路径前缀；新模块也提供同签名 `load_cache(market, phase, D=None)`。

- `phases.inner`沿用D.tr的NN context/scaler/phi和完整支持；`phases.final`沿用D.rf对应对象。
- 对phase边界前的sid，森林参考改为固定四个时间block的prefix预测；首block直接用当前成熟NZ uniform。
- block边界是start+floor(f*(boundary-start))，f=[0,.25,.5,.75,1]，每block query为左闭右开区间。
- RF训练源是phase fitids中各自`t_i+h_i<block_start`的行；prefix scaler按所有h的prefixfit估计，RF仍逐h使用冻结leaf/五seed/80树。
- prefix leaf权重按原D行索引映射到旧完整phase NZ bank；重复z不按值匹配。
- reference=`.95*prefix_forest+.05*当前成熟uniform`。uniform可以包含block开始后、但不晚于当前query才成熟的病例；它们没有参与该block森林的监督拟合。
- 没有prefix样本或单h prefix<=1条时，仍对合法NN query给成熟uniform，不新增训练query筛选。
- `z/phi/phi2/support/mature/allowed/count/uniform/fitindices/p`及phase NN `x/scaler`保持原值。`u/forest/u_forest/m0/v0`只改t<boundary的sid，训练sid的fallback标记同步。
- 所有t>=boundary的参考字段逐值保持不变；特别检查inner D.va与final D.te的u位级相同。

新增metadata：

- 顶层`source_cache_file/source_cache_sha256`与`honest_reference_policy`。
- 顶层`honest_blocks`及phase的`honest_blocks`：四个block的prefix全D索引/目标成熟索引/scaler、query sid，以及各h的RF源D索引、NZ源D索引到旧bank列的映射、query D索引。
- phase `query_block_sid[Nsid]`：训练sid取0..3，边界后为-1。
- bank `honest_query_block_sid`同上；`old_fullphase_fallback`保留原fallback标记。

该控制只改变训练query的森林参考误差机制。NN参数、NN scaler和响应phi仍用整个允许phase正常拟合，不称所有参数都按每个query历史冻结，也不据此断言旧评估无效。
