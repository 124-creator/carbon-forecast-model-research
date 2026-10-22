# 尺度归一化追加实验：2026-10-22

本报告补充[前一轮完整报告](ROUND_REPORT_ZH.md)。在不覆盖九模型与门控结果的前提下，新增四个归一化QHAR、两种归一化门控、一个自由尺度系数公平对照；没有查看或下载新日期的数据。

## 当前判断

归一化缓解了2025 h22的部分MSE失配，但未形成稳定收益：原liquidity→N-liquidity的MSE%MA22，KRBN从539.02降到345.01，KEUA从791.86降到147.37，KCCA却从55.80升到79.59。自由尺度系数对照在同三资产分别为327.40、168.41、54.60，表明收益不能只归因于齐次约束。N-liquidity对plus_scale的等资产QLIKE差在9个阶段期限格中8格为正；2026 h22差+0.26324，约束缺乏稳定优势。

归一化动态门对同归一化固定门9格仅2格QLIKE差为负。2025 h22动态门对EWMA97等资产差−0.00261，资产内却是KRBN +0.04267（NW t=2.814）、KEUA +0.01776（t=1.125）、KCCA −0.06826（t=−1.056）；三个资产MSE均比EWMA97高，分别高64.37%、12.49%、27.60%。2026 h22动态门对EWMA97等资产差+0.07461，对原动态门差+0.57682，跨阶段代价明显。

归一化后2025 KRBN三个期限w>.99的比例均100%；KEUA h22 w<.01比例100%、h5为77.05%；KCCA h22 w>.99为90.35%。这主要是极端选择代理，不能宣传动态可靠性融合。固定门消融不支持动态增益，**应撤下本轮流动性gate的核心机制主张**，保留失败候选和诊断以追溯，不宣告所有风险模型研究结束。极小的1e−7量级静动态差可能仍给出较大NW t，属于几乎相同饱和预测，不解释为经济上有意义的机制证据。

## 参数化与公平性

每个历史原点s使用当时可得的EWMA97风险尺度S_s，训练响应Y_s/S_s；own/range/gap的原log(mean+eps)减log(S_s)，rel/flat不变，interaction中range−own差不变；最终f_t=S_t exp(g_t)。目标CSV始终保留原始Y，零目标没有替换，尺度也没有floor。

QLIKE log(f)+Y/f 与归一化目标的QLIKE仅相差已知log(S)，但这不意味着和原模型同一假设类。按未标准化系数展开，liquidity候选是 `log f=b0+Σ_{全部7个log代理} b_k log(V_k)+(1−Σ_{全部7个log代理} b_k)log(S)+q直接项`，求和包括gap，归一化带来了offset/齐次系数约束。标准化后ridge几何也改变，不能把全部差异归于单一机制。共享斜率动态门控在含gap系数时同样遵循这种展开。

公平对照 QHAR_liquidity_plus_scale 使用原9个liquidity特征再加log(S)，在原始Y上QLIKE训练，无固定offset、尺度系数自由。NQHAR_liquidity与它的比较有助区分新增EWMA尺度信息和约束的作用。N_gate_dynamic须与同归一化N_gate_static比较，不能只比较旧模型。

沿用同一expanding窗口、每22个原点refit、训练标签s+h≤refit_t、跨年训练与计分剔除、预2024 epsilon、训练内标准化、ridge=.01与三组门权初始化。2024 validation、2025 development、2026 diagnostic均已看过，没有干净测试。

## 三资产等权QLIKE差

Δ=模型−MA22，负数更好，不报QLIKE百分比。每格先算各资产共同原点内差，再三资产等权平均。

| 年 | h | EWMA97 | plus_scale | N-return | N-joint | N-liquidity | N-interaction | N-static | N-dynamic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024 | 1 | -0.00094 | 0.02301 | -0.00837 | 0.00727 | 0.03558 | 0.04914 | 0.01023 | 0.03590 |
| 2024 | 5 | -0.02166 | -0.01993 | 0.00108 | -0.00330 | -0.00222 | 0.00205 | -0.00421 | -0.00307 |
| 2024 | 22 | -0.07998 | -0.11428 | -0.05504 | -0.09237 | -0.09123 | -0.09177 | -0.09295 | -0.09257 |
| 2025 | 1 | -0.02966 | -0.07500 | -0.05548 | -0.04177 | -0.07855 | -0.08218 | -0.08860 | -0.09580 |
| 2025 | 5 | -0.05951 | -0.03486 | -0.03929 | -0.03034 | -0.03096 | -0.01175 | -0.05443 | -0.04944 |
| 2025 | 22 | -0.14943 | -0.12979 | -0.11435 | -0.11052 | -0.11169 | -0.10285 | -0.15219 | -0.15204 |
| 2026 | 1 | -0.16825 | -0.56068 | -0.46683 | -0.52747 | -0.53206 | -0.54051 | -0.48072 | -0.48147 |
| 2026 | 5 | -0.25604 | -0.55078 | -0.45888 | -0.47501 | -0.50181 | -0.50265 | -0.47955 | -0.47752 |
| 2026 | 22 | -0.52289 | -0.77693 | -0.32616 | -0.49006 | -0.51369 | -0.52088 | -0.44829 | -0.44829 |

机制对照差如下，仍是负数更好：

| 年 | h | N-liq−EWMA97 | N-liq−原liq | N-liq−plus_scale | N-dynamic−N-static | N-dynamic−原dynamic |
| --- | --- | --- | --- | --- | --- | --- |
| 2024 | 1 | 0.03652 | 0.02098 | 0.01257 | 0.02567 | 0.00703 |
| 2024 | 5 | 0.01945 | 0.02684 | 0.01772 | 0.00113 | 0.03696 |
| 2024 | 22 | -0.01125 | 0.07425 | 0.02305 | 0.00038 | 0.08475 |
| 2025 | 1 | -0.04890 | -0.02529 | -0.00355 | -0.00721 | -0.04394 |
| 2025 | 5 | 0.02855 | -0.03216 | 0.00390 | 0.00499 | -0.04667 |
| 2025 | 22 | 0.03774 | -0.03112 | 0.01810 | 0.00015 | -0.05401 |
| 2026 | 1 | -0.36381 | 0.13191 | 0.02862 | -0.00075 | 0.13015 |
| 2026 | 5 | -0.24578 | 0.22069 | 0.04897 | 0.00203 | 0.18931 |
| 2026 | 22 | 0.00920 | 0.54297 | 0.26324 | 0.00000 | 0.57682 |

## 2025 h22：逐资产检查尺度失配

MSE%MA22为100×比率；应减100才是相对变化百分比。不同资产必须分别解释，不能用均值代替资产结论。

| 资产 | 模型 | n | ΔMA22 | MSE%MA22 |
| --- | --- | --- | --- | --- |
| KCCA | EWMA97 | 228 | -0.48344 | 65.30651 |
| KCCA | QHAR_liquidity | 228 | -0.61908 | 55.79796 |
| KCCA | QHAR_gate_dynamic | 228 | -0.61888 | 55.34383 |
| KCCA | NQHAR_return | 228 | -0.41014 | 107.29190 |
| KCCA | NQHAR_joint | 228 | -0.53573 | 78.30928 |
| KCCA | NQHAR_liquidity | 228 | -0.53448 | 79.58731 |
| KCCA | NQHAR_interaction | 228 | -0.50798 | 82.57914 |
| KCCA | N_gate_dynamic | 228 | -0.55170 | 83.33169 |
| KCCA | N_gate_static | 228 | -0.55216 | 83.22943 |
| KCCA | QHAR_liquidity_plus_scale | 228 | -0.59170 | 54.59863 |
| KEUA | EWMA97 | 227 | 0.02180 | 103.90224 |
| KEUA | QHAR_liquidity | 227 | 0.15815 | 791.85976 |
| KEUA | QHAR_gate_dynamic | 227 | 0.12249 | 569.34354 |
| KEUA | NQHAR_return | 227 | 0.03321 | 110.78271 |
| KEUA | NQHAR_joint | 227 | 0.05781 | 143.14577 |
| KEUA | NQHAR_liquidity | 227 | 0.05565 | 147.37439 |
| KEUA | NQHAR_interaction | 227 | 0.05739 | 150.80608 |
| KEUA | N_gate_dynamic | 227 | 0.03956 | 116.87635 |
| KEUA | N_gate_static | 227 | 0.03956 | 116.87659 |
| KEUA | QHAR_liquidity_plus_scale | 227 | 0.06499 | 168.41418 |
| KRBN | EWMA97 | 228 | 0.01335 | 98.47602 |
| KRBN | QHAR_liquidity | 228 | 0.21924 | 539.01867 |
| KRBN | QHAR_gate_dynamic | 228 | 0.20230 | 445.12566 |
| KRBN | NQHAR_return | 228 | 0.03387 | 122.33799 |
| KRBN | NQHAR_joint | 228 | 0.14636 | 352.14749 |
| KRBN | NQHAR_liquidity | 228 | 0.14377 | 345.00916 |
| KRBN | NQHAR_interaction | 228 | 0.14205 | 336.30724 |
| KRBN | N_gate_dynamic | 228 | 0.05602 | 161.86192 |
| KRBN | N_gate_static | 228 | 0.05602 | 161.86180 |
| KRBN | QHAR_liquidity_plus_scale | 228 | 0.13734 | 327.40072 |

## 门控饱和诊断

以下列出2025原动态门与归一化动态门的分位数和极端比例；全部阶段、两种静态门也保存在[完整饱和诊断](all_gate_saturation.csv)。门权接近1意味着几乎只选range、接近0意味着几乎只选own；不自动支持融合或状态自适应。原KRBN2024三个h、KEUA2024/25 h5/h22、KCCA2026 h22有明显饱和。没有事后给a0加惩罚来制造曲线；参数图形不识别测量误差。

| 资产 | h | 模型 | P05 | 中位 | P95 | w<.01比例 | w>.99比例 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| KCCA | 1 | N_gate_dynamic | 0.20200 | 0.43147 | 0.89599 | 0.00000 | 0.00000 |
| KCCA | 1 | QHAR_gate_dynamic | 0.13468 | 0.20390 | 0.83017 | 0.00000 | 0.00000 |
| KCCA | 5 | N_gate_dynamic | 0.25004 | 0.41807 | 0.76300 | 0.00000 | 0.00000 |
| KCCA | 5 | QHAR_gate_dynamic | 0.14717 | 0.36346 | 0.58410 | 0.00000 | 0.00000 |
| KCCA | 22 | N_gate_dynamic | 0.93306 | 1.00000 | 1.00000 | 0.00000 | 0.90351 |
| KCCA | 22 | QHAR_gate_dynamic | 0.66810 | 0.82583 | 1.00000 | 0.00000 | 0.17544 |
| KEUA | 1 | N_gate_dynamic | 0.32005 | 0.40273 | 0.58499 | 0.00000 | 0.00000 |
| KEUA | 1 | QHAR_gate_dynamic | 0.37159 | 0.45832 | 0.54520 | 0.00000 | 0.00000 |
| KEUA | 5 | N_gate_dynamic | 0.00000 | 0.00000 | 0.18901 | 0.77049 | 0.00000 |
| KEUA | 5 | QHAR_gate_dynamic | 1.00000 | 1.00000 | 1.00000 | 0.00000 | 1.00000 |
| KEUA | 22 | N_gate_dynamic | 0.00000 | 0.00000 | 0.00000 | 1.00000 | 0.00000 |
| KEUA | 22 | QHAR_gate_dynamic | 1.00000 | 1.00000 | 1.00000 | 0.00000 | 1.00000 |
| KRBN | 1 | N_gate_dynamic | 1.00000 | 1.00000 | 1.00000 | 0.00000 | 1.00000 |
| KRBN | 1 | QHAR_gate_dynamic | 0.61491 | 1.00000 | 1.00000 | 0.00000 | 0.66667 |
| KRBN | 5 | N_gate_dynamic | 1.00000 | 1.00000 | 1.00000 | 0.00000 | 1.00000 |
| KRBN | 5 | QHAR_gate_dynamic | 0.90379 | 1.00000 | 1.00000 | 0.00000 | 0.76735 |
| KRBN | 22 | N_gate_dynamic | 1.00000 | 1.00000 | 1.00000 | 0.00000 | 1.00000 |
| KRBN | 22 | QHAR_gate_dynamic | 0.68829 | 1.00000 | 1.00000 | 0.00000 | 0.72807 |

## 实现与核验

第一次执行完成KRBN后，被对“全部早期日期尺度>0”的过严断言中断。KEUA最早索引1/2/3的EWMA为0，但没有完整22日特征，从未进入原训练/评分样本。修正为在原共同有效特征日期断言尺度>0，其他早期logscale保持无效NaN；没有新增删样本、改目标或补尺度。修复后重跑，原计分标志和零目标标志完全一致；目标CSV往返最大绝对差3.39e-21，为浮点表示误差，没有实质变动。

新增7模型37,275条总预测；18模型91,314条计分记录=5,073个资产×期限×日期共同原点×18，不是9万独立样本。1,230次线性QHAR/自由尺度对照拟合、1,476次门控初值拟合、492次最终门控拟合。优化flag计数见[manifest](normalized_manifest.json)；全部37,275条新预测由保存参数独立重建通过，QLIKE/MSE重算通过，归一化QLIKE恒等式通过。NW lag=max(5,h−1)，未做多重比较调整，2026样本尤其有限。

## 研究解释与下一步边界

尺度×修正的构造已有component multiplicative-error、volatility component、GARCH-MIDAS等相关原理，不能称新family，也不能仅靠relative-risk改名声称创新。当前epsilon来自pre2024 mean r²，整体伸缩时应随风险单位变动；不要把固定epsilon版本宣称严格等变。归一化后是否保留流动性gate，应依据同归一化静态消融和饱和诊断，而不是模型名字。

KRBN/KCCA档案截至2026-06-08，而当前是2026-10-22。若之后真实数据可取得，可在冻结候选和比较规则之后构成新增时间外验证；目前没有下载或查看这些日期，也不保证可取得。KEUA档案止3月，不能假设能补齐。应先完成当前诊断和候选冻结，再决定拉取新数据。

## 复现与产物

先保留上一轮两脚本的结果，再在同一目录与同一Python环境运行：

```powershell
& 'C:\Users\Dreamboat\AppData\Local\hermes\hermes-agent-v0.21.5\venv\Scripts\python.exe' normalized_risk_experiment.py
& 'C:\Users\Dreamboat\AppData\Local\hermes\hermes-agent-v0.21.5\venv\Scripts\python.exe' build_normalized_report.py
```

- [新实验脚本](normalized_risk_experiment.py)、[manifest](normalized_manifest.json)、[新模型全部预测](normalized_predictions_all.csv)、[18模型共同计分预测](normalized_comparison_predictions_common.csv)。
- [全部18模型资产/年/期限汇总](normalized_summary.csv)、[逐基准差与NW/MSE比率](normalized_pairwise.csv)、[等资产描述汇总](normalized_pairwise_equal_asset.csv)、[关键资产消融](normalized_focused_asset_results.csv)。
- [线性拟合诊断](normalized_regression_fit_diagnostics.csv)、[线性参数](normalized_regression_parameters.csv)、[门控拟合诊断](normalized_gate_fit_diagnostics.csv)、[门控参数](normalized_gate_parameters.csv)、[逐日门权](normalized_gate_weights.csv)、[饱和诊断](all_gate_saturation.csv)、[独立核验](normalized_validation.json)。
