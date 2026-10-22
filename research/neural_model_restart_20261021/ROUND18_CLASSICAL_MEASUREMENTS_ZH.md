# Round18传统GK/RS测量+全局噪声滤波基准

仅新文件，四组asset×measurement拟合，每组仅seed11一次确定性拟合，不作独立重复。目标始终是原future mean Parkinson proxy，不是GK/RS目标、真实IV或return-GARCH。2025仍用于每asset选epoch，2026价格完全未读。

## 隔离与校准

原B16.prepare和R14.build先整表读取再截2026，因此本实验使用新pre2026前缀副本：源文件无缓冲逐行只先读10字节日期，遇首2026日期立即停止、不读该行其余字段。B16.SOURCES只在内存临时指向副本并恢复，源代码/冻结文件未改。原完整CSV不重新hash，既有归档hash作为声明保留；实际prefix及全部代码/模拟/结果源hash已记录。

GK=.5 log(H/L)^2−(2ln2−1)log(C/O)^2；RS=log(H/O)[log(H/O)−log(C/O)]+log(L/O)[log(L/O)−log(C/O)]。同Brownian路径的P/eta/xi代数式与直接HLC计算一致。只打开grid512 TRAIN六个数组，没有使用heldout或2048。

cP=0.942773538125847，状态框架log(cP*sigma²)。观测bias严格为b_estimator−log(cP)，不使用cG/cRS。R是positive log估计量残差样本方差，ddof=1。estimator<=EPS(1e-12)时skip，不把zero RS替换成logEPS；不按theta阈值删除行，原target/origin/tr/va/ewma保持不变。

| 测量 | synthetic正值数 | skip数 | b(log估计量) | bias=b−logcP | R(ddof1) |
|---|---:|---:|---:|---:|---:|
| GK | 60000 | 0 | -0.219427746 | -0.160498570 | 0.279824095 |
| RS | 59880 | 120 | -0.273878114 | -0.214948939 | 0.449143910 |

## 四组拟合与skip

| asset/测量 | epoch/run | 4全局参数 | 更新/总行 | 2025skip | 原共同起点 |
|---|---|---:|---|---:|---:|
| KRBN_GK | 11/26 | 4 | 1358/1362 | 0 | 228 |
| KRBN_RS | 0/20 | 4 | 1325/1362 | 7 | 228 |
| KCCA_GK | 30/45 | 4 | 1064/1065 | 0 | 228 |
| KCCA_RS | 25/40 | 4 | 1025/1065 | 3 | 228 |

## joint六格等权

| 模型 | raw QLIKE | normalized QLIKE | MSE | R18mean减该模型QLIKE |
|---|---:|---:|---:|---:|
| R18_shared_mean | -8.311893725 | 0.349706056 | 2.341369e-08 | +0.000000000 |
| Parkinson_raw_KF | -8.307186076 | 0.354413705 | 2.372049e-08 | -0.004707649 |
| Parkinson_fixed_KF | -8.305338777 | 0.356261004 | 2.378482e-08 | -0.006554948 |
| GK_global_KF | -8.298724494 | 0.362875286 | 2.389162e-08 | -0.013169230 |
| RS_global_KF | -8.294238402 | 0.367361379 | 2.400293e-08 | -0.017655323 |

## 每asset/h直接比较

| asset/h | GK QLIKE | RS QLIKE | R18mean QLIKE | mean−GK | mean−RS |
|---|---:|---:|---:|---:|---:|
| KRBN/1 | -8.575633 | -8.568005 | -8.592744 | -0.017111 | -0.024739 |
| KRBN/5 | -8.537003 | -8.533545 | -8.548431 | -0.011428 | -0.014886 |
| KRBN/22 | -8.562019 | -8.566418 | -8.567960 | -0.005941 | -0.001542 |
| KCCA/1 | -8.058267 | -8.041840 | -8.078418 | -0.020151 | -0.036578 |
| KCCA/5 | -7.997734 | -7.995646 | -8.017987 | -0.020253 | -0.022341 |
| KCCA/22 | -8.061691 | -8.059976 | -8.065822 | -0.004130 | -0.005846 |

所有date/Y与R18归档逐元素一致。模型逐asset在2025选epoch，未调整超参追求输赢，未作真实IV识别或因果观测噪声结论。R18mean在6/6单格的QLIKE点估计均优于本协议下的GK/RS全局噪声滤波，因此这两条具体传统基准没有解释掉均值头的开发增量；这不能排除更丰富的homogeneous估计器或替代测量组合，也不替代普通NN控制和独立封存期确认。最强的旧Parkinson raw仍比GK/RS更难击败，原R18相对raw/fixed的bootstrap区间跨零结论不变。完整MSE/拟合历史/参数/校准/源hash在round18_classical_measurements_results.json，预测在同名前缀validation_predictions.npz。
