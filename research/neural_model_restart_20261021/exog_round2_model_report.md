# 外生分任务滞后注意力候选小实验

候选机制：碳价历史状态与logh生成query，energy/equity-risk/fx各组20个历史槽位，以位置与正分位间距分别选择滞后；logit减可学习age惩罚。不是首次注意力主张。

只在输入审计确认有覆盖的HUBEI/GUANGZHOU运行；EU九列全NA。相同canonical目标、原点和h1/5/10/20，seed11/22均值；epoch0可选，最多64轮，验证早停，选轮后从头全precut refit；asinh/diff/scaler分别在inner train和最终refit重估。

结论：本轮未显示稳定外生增量。dual_query 相对纯内生仅 3/8 个市场×步长单元胜出，MQS 中位比 1.00161；HUBEI 在 h5/10/20 有小幅改善，GUANGZHOU 四个步长均不及纯内生。相对 mean_pool/shared_query 的中位比分别为 0.99988/0.99996，差距极小。全部 HUBEI 模型与两 seed 验证选中 3 轮，GUANGZHOU 选中 2 轮；有限训练下不能认定模型机制已形成可靠正增量。

mean_pool 与 shared/dual_query 接收相同的变化、mask、age；dual_no_age 同时移除 token age 和 logit age 惩罚，因此该消融识别整体 age 信息贡献，不单独识别 logit 惩罚。20 个槽位指过去20个canonical行，缺失保留mask，不跨缺失搜集更早20条记录。WIS沿用共同基准的5组中心区间（95%、90%、80%、70%、60%）加中位数误差定义。

|dual_query相对基线|MQS中位比|平均log比|胜出cell|
|---|---:|---:|---:|
|dual_query_vs_empirical_anchor|0.99039|-0.00905|6/8|
|dual_query_vs_gaussian_rw|0.90402|-0.10743|8/8|
|dual_query_vs_ridge_exog_summary|0.90828|-0.12637|8/8|
|dual_query_vs_endogenous_only|1.00161|-0.00056|3/8|
|dual_query_vs_mean_pool|0.99988|-0.00011|7/8|
|dual_query_vs_shared_query|0.99996|-0.00002|6/8|
|dual_query_vs_dual_no_age|0.99972|-0.00080|5/8|

|市场|h|模型|MQS|WIS|80%覆盖|95%覆盖|
|---|---:|---|---:|---:|---:|---:|
|HUBEI|1|empirical_anchor|0.011018|0.009554|0.6986|0.9178|
|HUBEI|1|gaussian_rw|0.010930|0.009376|0.7808|0.8836|
|HUBEI|1|ridge_exog_summary|0.010946|0.009431|0.6781|0.9041|
|HUBEI|1|endogenous_only|0.010911|0.009444|0.6849|0.9110|
|HUBEI|1|mean_pool|0.010918|0.009449|0.6918|0.9110|
|HUBEI|1|shared_query|0.010917|0.009449|0.6918|0.9110|
|HUBEI|1|dual_query|0.010917|0.009449|0.6918|0.9110|
|HUBEI|1|dual_no_age|0.010911|0.009443|0.6849|0.9110|
|HUBEI|5|empirical_anchor|0.014289|0.012037|0.7431|0.9514|
|HUBEI|5|gaussian_rw|0.015027|0.012661|0.8889|0.9722|
|HUBEI|5|ridge_exog_summary|0.015353|0.012618|0.7222|0.9375|
|HUBEI|5|endogenous_only|0.014235|0.011973|0.7639|0.9514|
|HUBEI|5|mean_pool|0.014208|0.011955|0.7569|0.9514|
|HUBEI|5|shared_query|0.014207|0.011955|0.7569|0.9514|
|HUBEI|5|dual_query|0.014207|0.011954|0.7569|0.9514|
|HUBEI|5|dual_no_age|0.014217|0.011961|0.7639|0.9514|
|HUBEI|10|empirical_anchor|0.015978|0.013172|0.8440|0.9929|
|HUBEI|10|gaussian_rw|0.017669|0.015059|0.9716|1.0000|
|HUBEI|10|ridge_exog_summary|0.019081|0.015389|0.7589|0.9291|
|HUBEI|10|endogenous_only|0.016211|0.013407|0.8298|0.9929|
|HUBEI|10|mean_pool|0.016103|0.013310|0.8298|0.9929|
|HUBEI|10|shared_query|0.016102|0.013310|0.8298|0.9929|
|HUBEI|10|dual_query|0.016101|0.013308|0.8298|0.9929|
|HUBEI|10|dual_no_age|0.016136|0.013338|0.8298|0.9929|
|HUBEI|20|empirical_anchor|0.019354|0.016446|0.8824|0.9779|
|HUBEI|20|gaussian_rw|0.022885|0.020144|0.9559|1.0000|
|HUBEI|20|ridge_exog_summary|0.030264|0.024000|0.6324|0.9265|
|HUBEI|20|endogenous_only|0.019953|0.016921|0.8603|0.9632|
|HUBEI|20|mean_pool|0.019696|0.016727|0.8676|0.9632|
|HUBEI|20|shared_query|0.019695|0.016725|0.8676|0.9632|
|HUBEI|20|dual_query|0.019694|0.016724|0.8676|0.9632|
|HUBEI|20|dual_no_age|0.019774|0.016781|0.8676|0.9632|
|GUANGZHOU|1|empirical_anchor|0.016722|0.014453|0.7727|0.9030|
|GUANGZHOU|1|gaussian_rw|0.017250|0.014681|0.8061|0.8909|
|GUANGZHOU|1|ridge_exog_summary|0.016946|0.014548|0.7727|0.8970|
|GUANGZHOU|1|endogenous_only|0.016510|0.014218|0.7818|0.8970|
|GUANGZHOU|1|mean_pool|0.016557|0.014266|0.7758|0.9030|
|GUANGZHOU|1|shared_query|0.016552|0.014261|0.7788|0.9030|
|GUANGZHOU|1|dual_query|0.016554|0.014264|0.7788|0.9030|
|GUANGZHOU|1|dual_no_age|0.016553|0.014268|0.7758|0.9030|
|GUANGZHOU|5|empirical_anchor|0.025605|0.021880|0.8110|0.9482|
|GUANGZHOU|5|gaussian_rw|0.028006|0.024089|0.8902|0.9756|
|GUANGZHOU|5|ridge_exog_summary|0.026728|0.022381|0.7927|0.9390|
|GUANGZHOU|5|endogenous_only|0.024997|0.021310|0.8079|0.9543|
|GUANGZHOU|5|mean_pool|0.025121|0.021427|0.8049|0.9543|
|GUANGZHOU|5|shared_query|0.025116|0.021423|0.8049|0.9543|
|GUANGZHOU|5|dual_query|0.025117|0.021424|0.8049|0.9543|
|GUANGZHOU|5|dual_no_age|0.025102|0.021422|0.8079|0.9543|
|GUANGZHOU|10|empirical_anchor|0.028395|0.024805|0.8528|0.9632|
|GUANGZHOU|10|gaussian_rw|0.034146|0.030441|0.9571|0.9755|
|GUANGZHOU|10|ridge_exog_summary|0.031575|0.026088|0.8436|0.9632|
|GUANGZHOU|10|endogenous_only|0.027599|0.024109|0.8681|0.9632|
|GUANGZHOU|10|mean_pool|0.027717|0.024224|0.8650|0.9632|
|GUANGZHOU|10|shared_query|0.027714|0.024220|0.8650|0.9632|
|GUANGZHOU|10|dual_query|0.027713|0.024219|0.8650|0.9632|
|GUANGZHOU|10|dual_no_age|0.027725|0.024229|0.8650|0.9632|
|GUANGZHOU|20|empirical_anchor|0.039258|0.034390|0.8816|0.9533|
|GUANGZHOU|20|gaussian_rw|0.046595|0.041797|0.9533|0.9751|
|GUANGZHOU|20|ridge_exog_summary|0.042798|0.035062|0.8474|0.9533|
|GUANGZHOU|20|endogenous_only|0.037940|0.033237|0.8972|0.9533|
|GUANGZHOU|20|mean_pool|0.038142|0.033420|0.8972|0.9533|
|GUANGZHOU|20|shared_query|0.038145|0.033420|0.8941|0.9533|
|GUANGZHOU|20|dual_query|0.038142|0.033418|0.8941|0.9533|
|GUANGZHOU|20|dual_no_age|0.038146|0.033410|0.8972|0.9533|

限制：外生真实发布日/上游填补未核实；age只是归档来源日的保守年龄；GZ尾部陈旧达260天，DAX全缺失；EU不具备候选覆盖。测试历史已用于本项目探索，不能作为新盲测。单次有界比较不支持创新优先权或确定正增量；详细seed选轮、缩放、逐原点评分与15分位预测均已保存。

运行秒数：51.5。
