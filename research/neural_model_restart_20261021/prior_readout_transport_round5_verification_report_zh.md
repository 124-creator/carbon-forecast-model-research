# Prior读出、prefix源边界和真实混合核对

100configs、4307共同原点×期限键、110旧文件与10缓存SHA核对通过；全部模型2636分配参数。

逐字段核对40个prefix block：森林源目标均严格早于block开始，与query行无交集；NZ映射按原D行而非z值。inner/final以各自fiteligible行拟合4feature scaler，20组mean/std/fitindices重算一致；无效特征行全0；边界后参考u逐位同旧。

checkpoint每模型12固定query重建，a/b最大误差0，真实五seed mixture Q误差0、CRPS误差0。no-readout prior weight非零梯度0，其四维标准化输入全0；正常readout的内容观测单元测试与零head恢复均通过。

100条visited raw诊断均保留，33条best_raw不同于原normalized选择；未用于本轮选轮或额外拟合。reference/weighted floor访问0/0，额外零碰撞0。

| honest_readout / 参照 | MQS改善 | 完整CRPS改善 |
|---|---:|---:|
| honest_readout_vs_forest_reference_only | -0.7446% (11/20) | -0.7899% (11/20) |
| honest_readout_vs_rule_qrf | -0.6412% (9/20) | -0.7166% (9/20) |
| honest_readout_vs_qrf_original | -0.4657% (8/20) | -0.5601% (8/20) |
| honest_readout_vs_lstm_qr | +0.0205% (14/20) | 15q不可识别 |
| honest_readout_vs_old_forest_transport | -0.2441% (11/20) | -0.2900% (11/20) |
| honest_readout_vs_uniform_seq_transport | -0.1163% (12/20) | -0.2766% (10/20) |
| honest_readout_vs_honest_no_readout | -0.9098% (11/20) | -0.9216% (11/20) |
| honest_readout_vs_insample_readout | -0.0789% (10/20) | -0.0661% (10/20) |
| honest_readout_vs_honest_readout_reweight_only | -0.2703% (10/20) | -0.1528% (12/20) |

prefix是reference训练误差机制实验，不使旧结果失效。所有历史test复用，全部模型/seed保留，不通过test决定结构、epoch或发布成功。
