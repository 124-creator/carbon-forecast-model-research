# GRN 强基准：代码与验证交付，尚未正式拟合

本脚本只从冻结的 GRN development NPZ/audit 读取数据，保留原始零 range、全部训练成熟标签和同一批 228 个验证日期。CSV、其他市场 NPZ/JSON、GRN2026 数据/结果均有独立 IO 拒绝检查。原 B/G/MEM 的 ASSETS/SOURCES 不修改，不把 GRN 冒充其他资产。

脚本默认只做纯检查。此次交付会运行 `--review-validation`，它进行纯检查和真实 schema 检查并写验证清单，**不执行 optimizer 或 estimator.fit**。需要主线程 review 代码 SHA 后，才执行后面的独立正式开发命令。

## 预先固定的基准

| 模型 | 方法与信息 |
|---|---|
| EWMA22 | 原 G 冻结因果 EWMA22，不重新选 span |
| MEM_stationary | 原 decode/forecasts；训练目标选两个 starts；L-BFGS-B 300；原 bounds/options |
| MEM_integrated | 同原实现，rho=1；训练目标选两个 starts，不用验证集选 start |
| HAR_geometry | 原 B.direct_baselines 字节码，same baseline_x_F16 全 15 个输入；原内部 log(max(y,EPS)) 与 smear，不改 raw target |
| HGB_geometry | 原 B.direct_baselines 字节码和参数；原实现已经是 Poisson loss，保留真实 y=0 |
| HGB_Poisson | 相同树参数；统一 y/pre25 rawP scale；Poisson 仅作正均值 quasi loss，不声称价格/range 是计数 |
| HAR_QLIKE | 同 15 输入；只用成熟 pre25 X[tr] 标准化、截断 ±5；log link；包含零目标的凸 QLIKE + .01 sum(beta²) |

树参数固定：max_iter150、max_leaf_nodes7、learning_rate.04、L2=1、min_samples_leaf20、early_stopping=False、random_state11。HGB_Poisson 相对原 HGB 的区别是全 horizon 使用相同的 G pre25 raw-P scale；原 HGB 使用各 horizon 的成熟训练目标均值。

HAR_QLIKE 的 intercept bounds 为 [-8,8]，beta bounds 为 [-3,3]，标准差加 1e-5；15 输入保证 |f|≤233，不截断 f 或 y 来改变凸目标。L-BFGS-B options 与 MEM 固定一致。初值只用训练目标均值，无验证超参选择。

scale 是全部 pre25 rawP 的均值（含零），与 G fixed 一致。primary QLIKE=mean(log(v)+y/v)，MSE 使用所有 raw y。某个评分组含零时，normalized QLIKE 返回 None，不删行、不换成 positive-only rescue。

## 运行入口

使用既有 Hermes 训练环境，工作目录为研究目录：

```powershell
& 'C:\Users\Dreamboat\AppData\Local\hermes\hermes-agent-v0.21.5\venv\Scripts\python.exe' .\grn_strong_baselines.py --check-only
& 'C:\Users\Dreamboat\AppData\Local\hermes\hermes-agent-v0.21.5\venv\Scripts\python.exe' .\grn_strong_baselines.py --review-validation
```

`--check-only` 不写市场预测/报告，也不打开市场数据。`--review-validation` 只读冻结 NPZ/audit，成功后独占创建 purechecks/schema/delivery manifest 三个 JSON。它不会正式拟合。重复写相同 tag 会拒绝；需要另一个明确的本地开发 tag 才可创建新版本，不能用新 tag 绕过数据/代码哈希检查。

**以下仅为主线程 review 后的运行入口，本次不执行：**

```powershell
& 'C:\Users\Dreamboat\AppData\Local\hermes\hermes-agent-v0.21.5\venv\Scripts\python.exe' .\grn_strong_baselines.py --run-development
```

正式分支先核对当前代码/依赖/库版本与 validation manifest 完全一致，再独占创建 run reservation。保存 MEM 两 starts/训练选择、HAR/QLIKE 系数和 normalizer、树超参与 fitted trees pickle、所有预测/原 target/date/tr/va/ewma/F16 矩阵，以及带 SHA 的完成 manifest。未完成 manifest 不能当作一次成功拟合；任何已有输出/预约均拒绝覆盖。

当前验证来源是已经使用的 2025 development。不能把它或后续已用的 GRN2026 诊断称为全新独立确认。原始 ETN 报价、零 range 与碳市场重叠等数据来源限制仍保留。
