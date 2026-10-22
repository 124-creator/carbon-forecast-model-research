# 部分兑现反馈向扩展窗迁移：固定两候选结果

结论：弱机制信号可以迁移到扩展窗，且这次没有动态版本的平均MSE代价；仍不构成模型整体成功。两个候选在2025/2026的h=5/22四格等资产平均QLIKE与MSE都略优于保存的Online_expanding。各候选在12个“资产×年×期限”格中有8格实质改善、2格恶化、2格数值上持平；QLIKE/MSE的改善范围一致。2026相对GARCH和等权几何仍明显落后，不能把相对既有组合的小幅改善说成稳定优于强基准。

协议仅新增PartialFeedback_expanding与LeadBalanced_expanding：使用全部原共同评分集的s<t已发布原点，不衰减、不截窗；四专家和几何组合、44条完整反馈启动、.01 ridge均沿用，完整比较器直接复用保存的Online_expanding。设m=min(H,t−s)，普通partial用响应knownsum/m、质量m/H；balanced对每lead用Z_j=已兑现该lead的原点数，mass_sj=1/(H Z_j)，a_s=Σ_j mass_sj，响应Σ_j mass_sj r²[s+j]/a_s。两个目标均加.01/2‖w−uniform‖²，只读取s+j<=t的原始日收益分量。

QLIKE差为候选减基准，负数更好；MSE比先逐资产相除再平均，小于1更好；改善资产数分母3，绝对QLIKE差≤1e−12记作数值持平。

| period | horizon | model | delta_QLIKE_equal_asset | MSE_ratio_equal_asset | assets_QLIKE_better |
| --- | --- | --- | --- | --- | --- |
| 2025 | 5 | LeadBalanced_expanding | -0.000069 | 0.999734 | 1 |
| 2025 | 5 | PartialFeedback_expanding | -0.000063 | 0.999792 | 1 |
| 2025 | 22 | LeadBalanced_expanding | -0.001305 | 0.991935 | 3 |
| 2025 | 22 | PartialFeedback_expanding | -0.001419 | 0.989638 | 3 |
| 2026 | 5 | LeadBalanced_expanding | -0.004045 | 0.999025 | 2 |
| 2026 | 5 | PartialFeedback_expanding | -0.004058 | 0.999020 | 2 |
| 2026 | 22 | LeadBalanced_expanding | -0.004917 | 0.989978 | 2 |
| 2026 | 22 | PartialFeedback_expanding | -0.004677 | 0.990008 | 2 |

分资产结果如下。KCCA h5两个年份的差仅约1e−16，实质上完全持平；KRBN 2025 h5与KEUA 2026 h22恶化，不能用等资产均值掩盖这些反例。

| asset | period | horizon | delta_QLIKE_negative_better_Balanced | delta_QLIKE_negative_better_Partial | MSE_ratio_Balanced | MSE_ratio_Partial |
| --- | --- | --- | --- | --- | --- | --- |
| KCCA | 2025 | 5 | -0.000000 | -0.000000 | 1.000000 | 1.000000 |
| KCCA | 2025 | 22 | -0.000091 | -0.000108 | 0.999908 | 0.999892 |
| KCCA | 2026 | 5 | -0.000000 | 0.000000 | 1.000000 | 1.000000 |
| KCCA | 2026 | 22 | -0.001369 | -0.001216 | 0.996369 | 0.996429 |
| KEUA | 2025 | 5 | -0.000322 | -0.000317 | 0.997499 | 0.997615 |
| KEUA | 2025 | 22 | -0.000698 | -0.001090 | 0.982677 | 0.975081 |
| KEUA | 2026 | 5 | -0.011656 | -0.011672 | 0.997533 | 0.997525 |
| KEUA | 2026 | 22 | 0.005081 | 0.005784 | 1.005993 | 1.006253 |
| KRBN | 2025 | 5 | 0.000115 | 0.000128 | 1.001703 | 1.001763 |
| KRBN | 2025 | 22 | -0.003125 | -0.003059 | 0.993220 | 0.993940 |
| KRBN | 2026 | 5 | -0.000480 | -0.000501 | 0.999543 | 0.999534 |
| KRBN | 2026 | 22 | -0.018463 | -0.018599 | 0.967572 | 0.967341 |

不确定性：两个候选只有KEUA 2026 h5的未调整Newey-West双侧正态近似p约0.0066，普通partial的ΔQLIKE=−0.011672、balanced=−0.011656；其余实质差异的p均>0.05。若每候选的12格比较做Bonferroni调整，该最小p约0.079，仍不足以作为多重检验后的确认性发现。这里使用lag=max(5,H−1)的同一近似诊断，没有按结果选择显著格作为最终模型结论。

下面为两候选相对主要强基准的等资产QLIKE差；全部27模型的逐资产QLIKE、MSE和配对比较留在CSV。2026 h5对GARCH约+0.268、h22约+0.498；相对等权几何分别约+0.162、+0.228。2025虽较好，跨年稳定成功未成立。

| period | horizon | model | EWMA94 | Equal_geometric | GARCH11_t | MA22 | N_gate_dynamic | QHAR_interaction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | 5 | LeadBalanced_expanding | -0.049679 | -0.045951 | -0.157216 | -0.091896 | -0.042459 | -0.117961 |
| 2025 | 5 | PartialFeedback_expanding | -0.049674 | -0.045945 | -0.157210 | -0.091890 | -0.042453 | -0.117955 |
| 2025 | 22 | LeadBalanced_expanding | -0.094918 | -0.031589 | -0.200818 | -0.159517 | -0.007478 | -0.089264 |
| 2025 | 22 | PartialFeedback_expanding | -0.095032 | -0.031703 | -0.200932 | -0.159631 | -0.007592 | -0.089378 |
| 2026 | 5 | LeadBalanced_expanding | -0.260716 | 0.161533 | 0.268074 | -0.458415 | 0.019106 | 0.272440 |
| 2026 | 5 | PartialFeedback_expanding | -0.260729 | 0.161521 | 0.268062 | -0.458427 | 0.019093 | 0.272427 |
| 2026 | 22 | LeadBalanced_expanding | -0.466718 | 0.227883 | 0.497596 | -0.734968 | -0.286682 | 0.334839 |
| 2026 | 22 | PartialFeedback_expanding | -0.466478 | 0.228123 | 0.497836 | -0.734729 | -0.286443 | 0.335079 |

验收：5073共同原点，2候选10146条新预测，27模型136971行；9204次优化、0失败。h1两个候选均独立优化，与保存扩展控制的最大权重绝对差4.23e-14、最大预测相对差1.95e-14；h1恒等比较不把浮点残差用于显著性检验。已通过共享启动、无未来分量、每lead总质量1/H、sum(a)=1、全成熟窗口目标/梯度退化full以及旧预测/权重/25模型报告hash不变断言。

2024仅预热。2025和不完整2026此前已见，且资产末端不同；这是一轮固定两候选的适用性实验，未尝试其他专家、半衰期或窗口，也不宣称新颖性、非平稳无偏性或确认性优势。动态实验的负结果及完整报告保留原样。

- [事前计划](online_expanding_transfer_plan.md)、[脚本](online_expanding_transfer.py)、[manifest](online_expanding_transfer_manifest.json)。
- [27模型共同预测](online_expanding_transfer_full27_predictions.csv)、[27模型摘要](online_expanding_transfer_summary.csv)、[全部逐资产配对比较](online_expanding_transfer_pairwise.csv)、[等资产配对比较](online_expanding_transfer_equal_asset.csv)。
- [日级权重](online_expanding_transfer_weights.csv)、[权重摘要](online_expanding_transfer_weight_summary.csv)、[拟合诊断](online_expanding_transfer_fit_diagnostics.csv)。
- [此前25模型完整报告](online_risk_combination_report.md)。
