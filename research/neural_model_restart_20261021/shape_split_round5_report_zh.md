# 形状分裂响应：四臂冻结机制实验

完成600森林拟合，运行695.56秒；没有NN训练。A旧leaf固定，B/C/D仅inner选各market-arm共同leaf，四臂全部保留test。

在本预注册协议下，分裂目标不改变当前结论：三个变换响应臂均未得到更强的跨市场参考。相对A，abs(z)的整体MQS/CRPS退步1.0929%/1.0956%，秩正态分数退步1.0060%/0.9369%，z²退步1.3485%/1.3774%。三臂在所有5个匹配seed上的整体MQS与CRPS也全部为负，不是某个seed主导的均值现象。

市场方向并不一致：abs(z)与z²在CEA和上海为正，但广州、湖北、EU为负；秩正态只有CEA为正。abs(z)的MQS相对A按CEA/GZ/SH/HB/EU为+1.0158%/-4.1911%/+0.7003%/-1.3098%/-1.7670%；秩正态为+0.4741%/-0.0934%/-0.9631%/-3.8081%/-0.6932%；z²为+0.8953%/-5.3690%/+0.2133%/-0.7593%/-1.8409%。因此不以局部改善或二次选臂composite掩盖整体负结果。

核对通过：20个A锚的raw逐seed/mean权重、γ0 Q/CRPS/p、γ.05 Q/CRPS/p全部逐位一致；4307共同键与177旧文件SHA一致。保存u重建canonical、各seed与equal-conditional诊断的Q/CRPS/p误差均0，独立CDF积分最大误差6.88e-15。canonical与按rawNZ质量加权seed F的最大L1浮点差6.63e-8；A的最终条件seed权重范围0.1824–0.2142，不能将它误称为最终条件F的等权平均。

独立逐单元gain表保留4800行market/h/seed/基线比较，并明确baseline_seed；同seed A比较和逐市场方向见核对报告。A保留旧每h leaf、B/C/D采用共同leaf是预设协议，因此结论限于此设置，不推广为所有分布森林或ML/DL方向无效。

A虽然叫mse_y，实际沿用冻结QRF的标准化float32 z响应；B=abs(z)、C=fit-only平均秩正态分数、D=z²。叶内仍用原有符号z的经验分布，未替换mark。仅更改分裂响应，不把单一变换响应MSE称为完整形状/CDF impurity。

每seed原始leaf概率包含零质量，canonical先等权均值raw概率再条件NZ、套rule p和γ=.05；等价最终条件seed混合权重与其原始NZ质量成正比，不是等权conditional-F平均。每seed独立结果和等权conditional-F诊断另外全保存，不把异质性藏进均值。

A每h逐seed/mean raw权重与冻结QRF逐位一致；γ0通过旧完整zero支持evaluator逐位复现ruleQRF，γ.05通过旧identity evaluator逐位复现forest_reference_only。失败会终止，未静默容忍锚改变。

| 模型 | 基线 | MQS改善 | CRPS改善 |
|---|---|---:|---:|
| mse_y | rule_qrf | +0.1027% (12/20) | +0.0727% (14/20) |
| mse_y | honest_no_readout | -0.1640% (6/20) | -0.1307% (7/20) |
| mse_y | rbf_additive | -0.1614% (5/20) | -0.0845% (10/20) |
| mse_y | lstm_qr | +0.7594% (14/20) | 15q不可识别 |
| mse_abs | mse_y | -1.0929% (9/20) | -1.0956% (7/20) |
| mse_abs | rule_qrf | -0.9891% (9/20) | -1.0221% (7/20) |
| mse_abs | honest_no_readout | -1.2587% (9/20) | -1.2277% (6/20) |
| mse_abs | rbf_additive | -1.2560% (10/20) | -1.1810% (8/20) |
| mse_abs | lstm_qr | -0.3252% (12/20) | 15q不可识别 |
| rank_score | mse_y | -1.0060% (8/20) | -0.9369% (8/20) |
| rank_score | rule_qrf | -0.9023% (7/20) | -0.8635% (7/20) |
| rank_score | honest_no_readout | -1.1716% (7/20) | -1.0687% (7/20) |
| rank_score | rbf_additive | -1.1690% (8/20) | -1.0221% (9/20) |
| rank_score | lstm_qr | -0.2389% (12/20) | 15q不可识别 |
| mse_z2 | mse_y | -1.3485% (9/20) | -1.3774% (7/20) |
| mse_z2 | rule_qrf | -1.2445% (9/20) | -1.3037% (7/20) |
| mse_z2 | honest_no_readout | -1.5147% (9/20) | -1.5099% (8/20) |
| mse_z2 | rbf_additive | -1.5121% (9/20) | -1.4630% (8/20) |
| mse_z2 | lstm_qr | -0.5789% (12/20) | 15q不可识别 |

硬性约束逐项核对：
1. 满足：仅inner Dtr拟合，Dva为原cut前最后25%（保留原lookback下限），seed11的一组80树森林；按四h rawMQS相对A的mean log比，共同leaf固定10/25/50；精确并列取更小leaf并在selection.tied_best_leaves留痕。
2. 满足：A不搜索旧四h leaf；完整多零支持γ0 evaluator与ruleQRF逐位，γ.05 identity evaluator与fixed reference逐位，任何失败均终止。
3. 满足：四臂四h五seed全final；逐seed、canonical和equal-conditional-final三套完整保留。
4. 满足：composite标为reference-only，留在JSON/cells，未进入本主表和机制结论。
5. 满足：inner mean log比>0的B/C/D仍全部跑final，没有改判据/换leaf/中途剔除。

正值左侧更好；composite仅按inner预注册规则选臂，是二次选择参考而非主结果。逐market/h与每seed对各基线gains完整保存在JSON summary/cells，NPZ含全部Q/CRPS/p及每h/seed/canonical u、rawNZ质量和条件seed混合权重，可重建F。

历史test已开发复用；若正向只来自上海或个别市场，应解释为分裂目标未改变当前结论，不用composite掩盖负方向。全部旧文件SHA前后一致，不宣称模型新颖性/显著性或发表成功。
