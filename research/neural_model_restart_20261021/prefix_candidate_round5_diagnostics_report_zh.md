# Prefix候选只读证据提取

对象为预设次要控制honest_no_readout；不改主候选honest_readout，不训练、不选seed，不重算full-mixture评分或重复checkpoint-mixture验证。
评分直接读取冻结cells/NPZ；真实head通过N.load_explicit(kind=honest)的final D.rf context与每个checkpoint在全部te上net(X,te)读取。

| seed | 参照 | MQS几何改善 | MQS胜格 | CRPS几何改善 | CRPS胜格 |
|---|---|---:|---:|---:|---:|
| 11 | rule_qrf | +0.186775% | 11/20 | +0.127026% | 12/20 |
| 11 | forest_reference_only | +0.084195% | 2/20 | +0.054385% | 2/20 |
| 22 | rule_qrf | -0.176763% | 11/20 | -0.260050% | 12/20 |
| 22 | forest_reference_only | -0.279716% | 5/20 | -0.332973% | 5/20 |
| 33 | rule_qrf | +0.166047% | 14/20 | +0.100673% | 14/20 |
| 33 | forest_reference_only | +0.063445% | 8/20 | +0.028013% | 5/20 |
| 44 | rule_qrf | -0.940618% | 12/20 | -1.081151% | 13/20 |
| 44 | forest_reference_only | -1.044357% | 0/20 | -1.154671% | 0/20 |
| 55 | rule_qrf | +0.033766% | 8/20 | -0.020689% | 8/20 |
| 55 | forest_reference_only | -0.068971% | 1/20 | -0.093437% | 1/20 |
| ensemble | rule_qrf | +0.266216% | 14/20 | +0.203084% | 12/20 |
| ensemble | forest_reference_only | +0.163718% | 14/20 | +0.130499% | 13/20 |

所有seed按预定顺序完整呈现。严格胜格按ratio<1计算，另存1e-6相对误差内的近相等格数，避免把epoch0附近数值差异当机制证据。

## 最终epoch

共25个market×seed重拟合，epoch0=17，非零epoch=8。

| 市场 | seed11/22/33/44/55最终epoch | epoch0 / 非零 |
|---|---|---|
| CEA | 0/0/4/0/0 | 4 / 1 |
| GUANGZHOU | 0/0/0/23/0 | 4 / 1 |
| SHANGHAI | 16/18/1/0/2 | 1 / 4 |
| HUBEI | 0/1/0/0/0 | 4 / 1 |
| EU | 0/0/0/0/2 | 4 / 1 |

## 真实head描述

delta=1.5tanh(raw0)，ell=1.5tanh(raw1)，尺度倍数exp(ell)，beta=.35tanh(raw2:4)。beta RMS定义为对query和两个坐标的平方平均开根号。
下列整体数值先在每个market/h内等权汇总五个seed组件，再对20格等权平均；逐seed/market/h完整表另存。

- mean(delta)=-0.0006420；mean|delta|=0.0271659 个参考标准差。
- mean exp(ell)=1.0014109；mean|exp(ell)-1|=0.0206764。
- beta RMS的20格均值=0.0203122。
- 非零epoch的活跃fit-cell中，mean|delta|=0.0848934，mean|exp(ell)-1|=0.0646139。

幅度上是位置与尺度的联合小残差，位置相对参考标准差的平均调整略大，两者同阶；不能解释为单纯尺度校准。全局有符号均值接近0或尺度均值接近1，都可能掩盖不同query间正负方向的抵消。由目标头推得的条件mark mixture标准差倍数20格均值为1.00581，mean绝对偏离1为0.01834。

只有seed11、33在固定reference上的MQS/CRPS同时为正；25个最终fit里17个epoch0。除上海含一个零head组件外，其余市场各含四个零head组件，ensemble具有很强的reference保留，不能把它的微小正增量描述为五个单模型都稳定改善。

delta直接控制目标mark均值m0+s0*delta，exp(ell)直接控制目标mark标准差s0*exp(ell)；a/b受倾斜后的mw/sw共同影响，不能单独解释成mean/variance变化。
真实五seedmixture没有单个神经head；ensemble_components行只描述组件等权统计。另存每query seed均值delta的绝对均值，以及共同参考下由目标矩推得的mixture std ratio=sqrt(mean(exp(2ell))+Var(delta))；它包含组件均值差异。
头幅度只能说明参数行为，不能分解MQS/CRPS收益，也不能替代matched位置/尺度/shape机制对照。现有source验证显示floor和额外零碰撞均为0，本次沿用该既有结论。

完整cells表有20格×全部五seed及ensemble，epochs表25行，heads表120行，head_queries存所有query raw及实际变换后的heads和参考矩。
只读诊断耗时7.045秒；源文件、checkpoint、honest cache与panel SHA前后一致。
