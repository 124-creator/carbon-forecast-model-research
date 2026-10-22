# 形状分裂响应：四臂冻结机制实验

完成24森林拟合，运行11.21秒；没有NN训练。A旧leaf固定，B/C/D仅inner选各market-arm共同leaf，四臂全部保留test。

A虽然叫mse_y，实际沿用冻结QRF的标准化float32 z响应；B=abs(z)、C=fit-only平均秩正态分数、D=z²。叶内仍用原有符号z的经验分布，未替换mark。仅更改分裂响应，不把单一变换响应MSE称为完整形状/CDF impurity。

每seed原始leaf概率包含零质量，canonical先等权均值raw概率再条件NZ、套rule p和γ=.05；等价最终条件seed混合权重与其原始NZ质量成正比，不是等权conditional-F平均。每seed独立结果和等权conditional-F诊断另外全保存，不把异质性藏进均值。

A每h逐seed/mean raw权重与冻结QRF逐位一致；γ0通过旧完整zero支持evaluator逐位复现ruleQRF，γ.05通过旧identity evaluator逐位复现forest_reference_only。失败会终止，未静默容忍锚改变。

| 模型 | 基线 | MQS改善 | CRPS改善 |
|---|---|---:|---:|
| mse_y | rule_qrf | +0.2379% (3/20) | +0.1208% (3/20) |
| mse_y | honest_no_readout | -0.0857% (0/20) | -0.0302% (0/20) |
| mse_y | rbf_additive | +0.0000% (0/20) | +0.0000% (3/20) |
| mse_y | lstm_qr | +0.1652% (3/20) | 15q不可识别 |

硬性约束逐项核对：
1. 满足：仅inner Dtr拟合，Dva为原cut前最后25%（保留原lookback下限），seed11的一组80树森林；按四h rawMQS相对A的mean log比，共同leaf固定10/25/50；精确并列取更小leaf并在selection.tied_best_leaves留痕。
2. 满足：A不搜索旧四h leaf；完整多零支持γ0 evaluator与ruleQRF逐位，γ.05 identity evaluator与fixed reference逐位，任何失败均终止。
3. 满足：四臂四h五seed全final；逐seed、canonical和equal-conditional-final三套完整保留。
4. 满足：composite标为reference-only，留在JSON/cells，未进入本主表和机制结论。
5. 满足：inner mean log比>0的B/C/D仍全部跑final，没有改判据/换leaf/中途剔除。

正值左侧更好；composite仅按inner预注册规则选臂，是二次选择参考而非主结果。逐market/h与每seed对各基线gains完整保存在JSON summary/cells，NPZ含全部Q/CRPS/p及每h/seed/canonical u、rawNZ质量和条件seed混合权重，可重建F。

历史test已开发复用；若正向只来自上海或个别市场，应解释为分裂目标未改变当前结论，不用composite掩盖负方向。全部旧文件SHA前后一致，不宣称模型新颖性/显著性或发表成功。
