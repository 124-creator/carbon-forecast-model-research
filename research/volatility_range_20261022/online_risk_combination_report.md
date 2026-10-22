# 在线风险反馈组合：固定协议结果

结论：部分兑现反馈存在小幅、方向一致的等资产QLIKE机制信号，但不足以认定模型成功。h=5/22的四个“年份×期限”组合中，普通partial和lead均衡partial相对完整动态反馈均改善，每格仅2/3资产改善；逐资产Newey-West近似检验没有一个达到5%水平。h=22的MSE反而恶化。更根本的是动态full在2026明显落后于等权、冻结权重与GARCH，partial没有扭转这一结论。不能把相对较弱动态基线的小幅改善说成稳定强基线胜利。

2024仅作预热与探索；判断表只用2025和不完整2026。所有年度此前已见，不是未触碰的外部检验集。KRBN/KCCA原始数据到2026-06-08，KEUA到2026-03-20，各期限评分原点还须保留完整且不跨年目标；等资产平均给予每个资产相同质量，不能解释为等日数汇总。

## 部分兑现的机制比较

QLIKE差为候选减基准，负数更好。MSE比是先逐资产取候选/基准再等资产平均，小于1更好；资产改善数分母为3。

| period | horizon | model | delta_QLIKE_equal_asset | MSE_ratio_equal_asset | assets_QLIKE_better |
| --- | --- | --- | --- | --- | --- |
| 2025 | 5 | LeadBalanced_partial | -0.000501 | 1.005183 | 2 |
| 2025 | 5 | PartialFeedback_dynamic | -0.000539 | 1.006196 | 2 |
| 2025 | 22 | LeadBalanced_partial | -0.004281 | 1.084623 | 2 |
| 2025 | 22 | PartialFeedback_dynamic | -0.005163 | 1.071573 | 2 |
| 2026 | 5 | LeadBalanced_partial | -0.013741 | 0.994654 | 2 |
| 2026 | 5 | PartialFeedback_dynamic | -0.013661 | 0.994535 | 2 |
| 2026 | 22 | LeadBalanced_partial | -0.016187 | 1.167616 | 2 |
| 2026 | 22 | PartialFeedback_dynamic | -0.014107 | 1.206238 | 2 |

h=1 full与普通partial独立求解得到逐位相同预测/权重。lead均衡也独立求解，再以权重rtol/atol=1e-9、预测rtol=1e-10/atol=1e-15验证数学等价；没有复制full权重充作验收。h=22普通partial在2025/2026的平均MSE比分别为1.071573/1.206238，lead均衡为1.084623/1.167616。短lead均衡后QLIKE方向仍在，说明观察到的均值改善不完全由短lead质量偏重解释；样本内的小幅差异仍不足以确立迁移性或非平稳无偏性。

## 动态full相对基本组合控制

下表均为动态full减列中基准的等资产QLIKE差：

| period | horizon | Equal_arithmetic | Equal_geometric | Frozen2024 | Online_expanding |
| --- | --- | --- | --- | --- | --- |
| 2025 | 1 | -0.039941 | -0.030303 | -0.013644 | -0.005063 |
| 2025 | 5 | -0.057708 | -0.050597 | -0.012854 | -0.004715 |
| 2025 | 22 | -0.046978 | -0.033997 | -0.017467 | -0.003714 |
| 2026 | 1 | -0.006597 | 0.018718 | -0.027461 | -0.057237 |
| 2026 | 5 | 0.149536 | 0.150074 | 0.024843 | -0.015504 |
| 2026 | 22 | 0.500746 | 0.461255 | 0.413593 | 0.228455 |

## 动态full相对全部模型中的强基准

| period | horizon | EWMA94 | GARCH11_t | MA22 | N_gate_dynamic | QHAR_interaction |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | 1 | -0.043591 | -0.131569 | -0.089961 | 0.005843 | -0.033975 |
| 2025 | 5 | -0.054326 | -0.161862 | -0.096542 | -0.047105 | -0.122607 |
| 2025 | 22 | -0.097327 | -0.203227 | -0.161926 | -0.009887 | -0.091673 |
| 2026 | 1 | -0.337761 | 0.069012 | -0.516520 | -0.035048 | 0.160705 |
| 2026 | 5 | -0.272175 | 0.256615 | -0.469874 | 0.007647 | 0.260981 |
| 2026 | 22 | -0.233346 | 0.730968 | -0.501597 | -0.053311 | 0.568211 |

2025相对GARCH各期限三资产均改善；2026 h=5对GARCH三资产均恶化，h=22只有1/3资产改善。2026 h=22普通partial/lead均衡相对GARCH的等资产QLIKE差仍分别为+0.716861/+0.714781。2025对EWMA94虽有负的等资产均值，各期限却只有1/3资产改善，不能只用均值描述普适胜利。完整25模型表同时包括原18模型和7个新组合，保留所有对照及MSE。

## 分资产机制结果与不确定性

Newey-West使用lag=max(5,h−1)，表中p为未调整的双侧正态近似；不把重叠目标、多个比较和短2026样本下的近似值当严格确认性证据。

| asset | period | horizon | model | delta_QLIKE_negative_better | MSE_ratio | NW_normal_p_two_sided |
| --- | --- | --- | --- | --- | --- | --- |
| KCCA | 2025 | 5 | PartialFeedback_dynamic | -0.000711 | 1.014295 | 0.838994 |
| KCCA | 2025 | 5 | LeadBalanced_partial | -0.000670 | 1.012159 | 0.837958 |
| KCCA | 2026 | 5 | PartialFeedback_dynamic | -0.005118 | 0.975763 | 0.624083 |
| KCCA | 2026 | 5 | LeadBalanced_partial | -0.004493 | 0.976185 | 0.664021 |
| KCCA | 2025 | 22 | PartialFeedback_dynamic | -0.013600 | 1.046356 | 0.348030 |
| KCCA | 2025 | 22 | LeadBalanced_partial | -0.013867 | 1.040019 | 0.274584 |
| KCCA | 2026 | 22 | PartialFeedback_dynamic | -0.019921 | 1.289728 | 0.598152 |
| KCCA | 2026 | 22 | LeadBalanced_partial | -0.035409 | 1.233911 | 0.389813 |
| KEUA | 2025 | 5 | PartialFeedback_dynamic | 0.000946 | 1.003723 | 0.772381 |
| KEUA | 2025 | 5 | LeadBalanced_partial | 0.001138 | 1.002970 | 0.715151 |
| KEUA | 2026 | 5 | PartialFeedback_dynamic | -0.036969 | 1.000827 | 0.185789 |
| KEUA | 2026 | 5 | LeadBalanced_partial | -0.038014 | 1.000656 | 0.184236 |
| KEUA | 2025 | 22 | PartialFeedback_dynamic | -0.003803 | 0.989010 | 0.466143 |
| KEUA | 2025 | 22 | LeadBalanced_partial | -0.000929 | 1.035493 | 0.854954 |
| KEUA | 2026 | 22 | PartialFeedback_dynamic | 0.041166 | 1.465940 | 0.715121 |
| KEUA | 2026 | 22 | LeadBalanced_partial | 0.050685 | 1.406592 | 0.622013 |
| KRBN | 2025 | 5 | PartialFeedback_dynamic | -0.001853 | 1.000571 | 0.564248 |
| KRBN | 2025 | 5 | LeadBalanced_partial | -0.001970 | 1.000420 | 0.537782 |
| KRBN | 2026 | 5 | PartialFeedback_dynamic | 0.001104 | 1.007015 | 0.572915 |
| KRBN | 2026 | 5 | LeadBalanced_partial | 0.001285 | 1.007120 | 0.519809 |
| KRBN | 2025 | 22 | PartialFeedback_dynamic | 0.001915 | 1.179351 | 0.759647 |
| KRBN | 2025 | 22 | LeadBalanced_partial | 0.001954 | 1.178357 | 0.762824 |
| KRBN | 2026 | 22 | PartialFeedback_dynamic | -0.063567 | 0.863046 | 0.197400 |
| KRBN | 2026 | 22 | LeadBalanced_partial | -0.063837 | 0.862346 | 0.194958 |

## 权重的实际行为

以下为动态full的h=22年度平均权重，各行四权重之和为1。完整日级权重同时保留各候选、有效反馈量、共享窗口数、标签截止、拟合时间与状态，可核查变化是否由当时已兑现信息触发。

| asset | year | n | mean_w_EWMA97 | mean_w_HAR_return | mean_w_QHAR_joint | mean_w_QHAR_liquidity_plus_scale |
| --- | --- | --- | --- | --- | --- | --- |
| KCCA | 2025 | 228 | 0.028926 | 0.424461 | 0.235489 | 0.311124 |
| KCCA | 2026 | 86 | 0.504971 | 0.011924 | 0.187344 | 0.295761 |
| KEUA | 2025 | 227 | 0.738078 | 0.028672 | 0.154792 | 0.078458 |
| KEUA | 2026 | 32 | 0.718750 | 0.020625 | 0.113513 | 0.147112 |
| KRBN | 2025 | 228 | 0.770323 | 0.114668 | 0.039875 | 0.075134 |
| KRBN | 2026 | 86 | 0.271321 | 0.728679 | 0.000000 | 0.000000 |

## 固定协议与审计

四专家固定为EWMA97、HAR_return、QHAR_joint、QHAR_liquidity_plus_scale；预测为log forecast的单纯形组合。full、partial与lead均衡共用最近252个s<t且原scored_common为真的已发布原点；full只取共享窗中的完整标签，不向更老原点补满。三者均以共享窗完整标签至少44个启动，之前等权，半衰期22原生交易日，ridge=.01。原始日收益平方来自既有feature_data，partial只读取s+j<=t的已发生分量，不由未来目标反推。

partial的每原点观测数m=min(h,t−s)，响应knownsum/m、质量decay*m/h，等价于规定的部分QLIKE贡献除以总兑现质量。lead均衡对每j将已兑现原点的衰减质量归一为1/h，再构造a与q/a。各lead质量、sum(a)=1、全m=h时随机及实际窗口的objective/gradient退化full均已断言。

扩展控制用全部成熟标签等权。Frozen2024在首个2025共同原点只用已成熟2024反馈拟合，h1/5/22分别251/247/230条，此后冻结到2026；2024仅等权预热，未回填拟合后的历史权重。

首次尝试在任何权重拟合前被输入目标逐位唯一断言拦下：34个原点的CSV往返表示末位不同，最大3.3881317890172014e−21。核对后统一同原点MA22记录的目标，不改真实目标、原文件或评分样本；训练的完整原始r²均值也与保存目标紧容差核对。随后完成固定协议运行；又按事前新增的数值验收要求将lead均衡h1改为独立求解并验证，参数和h5/h22预测不变。没有依据结果调整专家、窗口、衰减或罚项。

最终成功优化18417次，失败0次；5073个共同原点，7模型35511条新预测，25模型126825条完整对照。最大单纯形一阶最优性间隙2.87e-06；输入hash核验未变。

这是已有online aggregation家族中的机制实验；部分风险反馈和lead均衡仅称候选，仍需独立查重，不作新颖性声明。

## 输出

- [固定计划](online_risk_combination_plan.md)、[在线脚本](online_risk_combination.py)、[主manifest](online_manifest.json)。
- [25模型共同预测](online_full25_predictions_common.csv)、[年度和合并期间摘要](online_full25_summary.csv)、[全部配对比较](online_full25_pairwise.csv)、[等资产配对比较](online_full25_equal_asset.csv)、[25模型manifest](online_full25_manifest.json)。
- [原11模型机制表](online_comparison_predictions_common.csv)、[日级权重](online_weights.csv)、[权重年度摘要](online_weight_summary.csv)、[优化诊断](online_fit_diagnostics.csv)。
