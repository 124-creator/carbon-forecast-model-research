# R20冻结2026确认：独立只读审计

**按冻结定义，模块方向性确认通过；整体竞争模型目标未达。** 主候选仍为R20_joint_ensemble。相对matched continued仅有小幅点改善，3/3 seed同向，但两种95%区间跨零；MEM stationary、R16 raw及多个经典bench的主QLIKE更强。不能将方向性通过写成整体研究成功，也不以MSE救主指标或改选clock。

**2026已使用。** 本次已完成一次冻结授权确认。任何后续模型、seed、epoch或metric修改，都不得把这些2026观测重新称为全新独立确认。本审计不追加模型测试。

## 数据、checkpoint与normalizer

- 每asset179个共同origin，2026-01-02—2026-09-18，h1/5/22；标签最晚2026-10-20，全部成熟。这里不是完整全年2026，也没有使用未成熟后续origin。29个保存预测（含seed）都对齐同一date/Y元数据，目标仍为未来Parkinson均值，不是真实IV。
- 14个预列ensemble/经典模型以及29组含seed预测全部重评分，asset/h和joint的raw/known/normalized QLIKE、MSE复现；5组NN ensemble均是三seed预测均值。标签成熟按保存label_end及builder共同h22规则检查，没有从CSV重新构造目标。
- 15份.pt只读字节SHA核验，未反序列化；selected epoch与冻结selection及原开发记录一致。12个R20 reconstruction记录旧prefix精确、theta/bank/decoder/AR不变；本次不重放推理。
- canonical preprocessing JSON独立重算SHA为`85bd63bb036677e8cd6d2c2941bd1ddbc34d86df15f9a8441f5b09fabd15a513`，与selection和test report一致。原pre2025 bank日期及F17的摘要由保存数组复现；R20开发NPZ只有bank日期，F17复核使用原R18保存bank，并列入本次输入hash。
- selection SHA：`d39a08d0db8f99eefb37e0fbac75c0821eee9578719d8d6264c785a387785617`；test NPZ SHA：`3d93f8534ad6575544e5ccf9b2c1f51b3a411e80559f7e4d03b57e3418f67ca1`。所有非CSV冻结依赖实际hash匹配；两份CSV仅核selection/report声明相同，未重开原始行情。
- 测试记录无neural/MEM/KF训练、无epoch选择、无selection新建；HAR/HGB按原pre2025/F16流程，旧2025重放gate保留。

## 原criteria逐项判断

主指标固定为equal-asset/equal-horizon QLIKE，primary减R20_continued_ensemble。directional要求ensemble差<0且至少2/3 pairedseed差<0，**没有要求区间全负**。secondary要求完整报告各asset/h、normalizedQLIKE、MSE及所有预列比较。
ensemble差为-0.000432716386，3/3 seed为负，因此方向性PASS。按原claims，CI跨0只允许点改善声明，不能作显著性或真实A/M/IV识别。整体竞争力是原value条款中的另一个研究目标，不是事后新增的directional门槛。

## 全部预列模型与经典bench

known-scale=raw−log(EWMA22)，共同项不改变paired差。normalized=y/v−log(y/v)−1，按正目标计算；本次每asset/h正目标数详见JSON。百分比只对正normalized均值计算，不对负raw均值取%。

| 冻结模型 | known-scale QLIKE | normalized QLIKE | MSE | primary减该模型QLIKE |
|---|---:|---:|---:|---:|
| R20_joint_ensemble | 1.193289470518 | 0.526397062733 | 2.85001407e-08 | +0.000000000000 |
| R20_continued_ensemble | 1.193722186904 | 0.526829779118 | 2.85011718e-08 | -0.000432716386 |
| R20_clock_only_ensemble | 1.193282689347 | 0.526390281561 | 2.85001120e-08 | +0.000006781171 |
| R20_density_only_ensemble | 1.193720832897 | 0.526828425111 | 2.85010952e-08 | -0.000431362378 |
| C18_GRU_ensemble | 1.193636482395 | 0.526744074609 | 2.84996075e-08 | -0.000347011876 |
| R16_frozen | 1.190719222012 | 0.523826814226 | 2.84761478e-08 | +0.002570248506 |
| R16_raw | 1.166675567022 | 0.499783159236 | 2.82198646e-08 | +0.026613903496 |
| MEM_stationary | 1.159432708300 | 0.492540300514 | 2.94363875e-08 | +0.033856762219 |
| MEM_integrated | 1.173776800881 | 0.506884393095 | 3.04387907e-08 | +0.019512669637 |
| HAR_geometry | 1.195471487406 | 0.528579079620 | 2.89490816e-08 | -0.002182016888 |
| HGB_geometry | 1.189436132752 | 0.522543724966 | 3.12272396e-08 | +0.003853337767 |
| GK_global_KF | 1.189933113645 | 0.523040705859 | 2.84052885e-08 | +0.003356356873 |
| RS_global_KF | 1.184582116136 | 0.517689708350 | 2.85316744e-08 | +0.008707354383 |
| EWMA22 | 1.199879705034 | 0.532987297248 | 3.02952187e-08 | -0.006590234515 |

primary相对continued的normalized损失改善仅0.082136%；相对最强MEM stationary反而退化6.873907%，相对R16 raw退化5.325090%。这些经典bench的点优势远大于matched模块增量。

primary的MSE虽略优MEM stationary，但不能挽救QLIKE主目标；R16 raw的MSE本身也更好。continued本次还略差旧C18 GRU，因此不能把2025中continuation/bucket改善自动当作2026收益。clock在本次测试略好joint，但只是按冻结列表报告ablation，不替换主候选。

## 配对seed与asset/h方向

| seed | joint knownQLIKE | continued knownQLIKE | joint−continued |
|---|---:|---:|---:|
| 11 | 1.199425329547 | 1.200425285730 | -0.000999956184 |
| 22 | 1.195538904734 | 1.195948549883 | -0.000409645149 |
| 33 | 1.189417994814 | 1.189545157595 | -0.000127162782 |

seed不构成独立dataset。三个期限按asset平均均改善，但6格并非全好：KCCA22更差，KCCA5几乎为零。

| asset/h | joint−continued | joint−MEM stationary | joint−R16 raw |
|---|---:|---:|---:|
| KRBN/1 | -0.000825930359 | +0.111059181830 | +0.058351623031 |
| KRBN/5 | -0.000514483506 | +0.178370974577 | +0.072341673480 |
| KRBN/22 | -0.000970402652 | -0.006525776913 | +0.043051842861 |
| KCCA/1 | -0.000547636220 | -0.070392766728 | -0.004470512901 |
| KCCA/5 | -0.000000417069 | -0.017037904534 | +0.006744266138 |
| KCCA/22 | +0.000262571491 | +0.007666865079 | -0.016335471630 |

| equal-asset h | joint−continued |
|---|---:|
| 1 | -0.000686783289 |
| 5 | -0.000257450288 |
| 22 | -0.000353915580 |

## 严格重现冻结区间

保留2000次noncircular moving origin-date block方法，block22/44，seed=20261022+block；同一日期索引同步两个asset和全部h。没有改成开发时的10000次circular。全部13组原区间都从预测数组独立复现。

| primary减比较模型 | QLIKE点差 | block22 CI95 | block44 CI95 |
|---|---:|---|---|
| R20_continued_ensemble | -0.000432716 | [-0.001046032, +0.000067809] | [-0.000852169, +0.000093656] |
| R20_clock_only_ensemble | +0.000006781 | [-0.000005000, +0.000021225] | [-0.000001256, +0.000020829] |
| R20_density_only_ensemble | -0.000431362 | [-0.001047861, +0.000070731] | [-0.000849301, +0.000098208] |
| C18_GRU_ensemble | -0.000347012 | [-0.004792569, +0.003070446] | [-0.003647267, +0.003544456] |
| R16_frozen | +0.002570249 | [-0.008636750, +0.011062363] | [-0.008972173, +0.011592223] |
| R16_raw | +0.026613903 | [-0.000624174, +0.066586280] | [-0.003701648, +0.071249292] |
| MEM_stationary | +0.033856762 | [-0.054823253, +0.166551639] | [-0.045252818, +0.166914758] |
| MEM_integrated | +0.019512670 | [-0.079793839, +0.160553409] | [-0.067643902, +0.168678531] |
| HAR_geometry | -0.002182017 | [-0.114826968, +0.138378805] | [-0.090476759, +0.165090976] |
| HGB_geometry | +0.003853338 | [-0.108122247, +0.153300001] | [-0.089255874, +0.164098916] |
| GK_global_KF | +0.003356357 | [-0.013642508, +0.023534308] | [-0.005943755, +0.024792021] |
| RS_global_KF | +0.008707354 | [-0.007328872, +0.033074783] | [-0.011631466, +0.034847484] |
| EWMA22 | -0.006590235 | [-0.119879862, +0.140278008] | [-0.099265797, +0.117826455] |

## N通道及最终分层

冻结开发记录中所有effective alpha_N均为0，density/joint beta保持初始化。当前.pt字节hash逐一绑定原checkpoint，runner记录仅扩展观测buffer而theta/bank/decoder/AR不变。本次测试NPZ没有额外operator场数组，审计也未加载权重或重放前向，所以该判断依据是冻结参数记录＋checkpoint一致性＋runner参数保持检查，不能伪称又独立计算了测试Delta_b。

joint-clock的微小差不证明M贡献，当前主要是小clock项；不声称真实交易次数或IV识别。可保留的新性只是受约束结构、prior与观测/未来目标耦合设计，不能将结构能力写成已确认的M机制。

结论分层：①冻结方向性module criterion PASS；②只支持小幅点改善，不支持显著改善声明；③主QLIKE未胜最强经典模型，overall competitive model goal未达；④2026确认窗口已用，后续模型不得重新包装为全新独立确认。未用2026换primary/seed/epoch/asset/h/metric。

仅新增本说明及ROUND20_HOLDOUT_AUDIT.json，测试报告/预测、selection和预测源码未改。所有hash、14模型/29预测的单格分数、配对seed及原criteria均存JSON。
