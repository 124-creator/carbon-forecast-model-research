# ZIST-v1 模型卡（零膨胀分位数参考 + 矩归一化运输）

状态：**冻结候选**（`honest_no_readout`，5 seed）。本卡只描述可复现的实现与已测到的证据，不含未验证的主张。

---

## 1. 任务与协议（固定，不可改）

- 面板：`research/fullcycle_h20_20261020/PANEL_5MARKET.csv`，SHA256 `d4702383…b56c`，8912 行，5 市场（CEA 1262、GUANGZHOU 2839、SHANGHAI 1768、HUBEI 1466、EU 1577）。
- 目标：`y = log p[t+h] − log p[t]`，`h ∈ {1,5,10,20}` 为**观测面板步**；起点 `t` 只观测到 `p[t]`。
- 切分：CEA/GZ/SH 截断 `2024-01-01`，HUBEI/EU `2023-01-01`；内层验证取截断前最后 25%；训练 `t+h<val_start`；验证 `t≥val_start ∧ t+h<cut`；最终重训用全部 `t+h<cut`；测试 `t≥cut`。W=128、STEP=2。
- 归一化：`scale_t = 滚动日标准差 × sqrt(h)`，`z=y/scale`；**只在解码时乘一次 `scale`**。
- 度量：15 个 τ `[.025,.05,.10,.15,.20,.30,.40,.50,.60,.70,.80,.85,.90,.95,.975]`；pinball `max(τe,(τ−1)e)`；`MQS=2×mean pinball`；聚合为 market×horizon 等权 20 格的 `100*(1−exp(mean(log(MQS_model/MQS_ref))))`。

## 2. 模型（四件套，全部确定性）

**(a) 参考库（reference bank）**
对每个市场、每个 `h`：取最终重训分区内所有 `z≠0` 的已成熟样本（`t+h < cut`），按 `T.make_bank` 的稳定非零选择与 float64 `z` 的稳定排序建库。查询时用**成熟掩码**：只有 `mature_t = t_bank + h ≤ 查询起点` 的库元素允许有质量；未成熟元素质量恒为 0。叶子权重由 5 个种子的 `RandomForestRegressor`（叶子大小取 QRF 内层选定的 `chosen_minleaf`）在**原始叶子概率**上等权平均得到，再条件化到非零质量上。

**(b) 均匀储层（uniform reservoir）**
`u = (1−γ)·forest_only + γ·uniform_allowed`，`γ=.05`。它保证 `Var_w ≥ γ·exp(−4.2)·Var_uniform > 0`，即预测律永不退化；这也是“参考平滑”的来源。

**(c) 零原子规则（rule zero atom）**
`p = .8·local_zero_freq + .2·p_static`，`local_zero_freq` 为起点邻域内 `y=0` 的频率，`p_static` 为库内零比例。零原子按 `p` 放在 0 处，非零部分按 `u` 加权。

**(d) 矩归一化仿射运输（mark branch）**
对标记 `z` 施加 `v = a·z + b`，其中 `a = s_t/√Var_w`、`b = m_t − a·E_w[z]`，`m_t = m0 + √v0·1.5·tanh(θ₀)`、`s_t = √v0·exp(1.5·tanh(θ₁))`。在**给定加权的标记分支**上，这是把预测均值/方差匹配到目标 `(m_t, s_t)` 的唯一仿射映射，且是 W2 意义下的最小范数解（见 `forest_transport_round5_PRIOR_ART_ZH.md` 的推导与反例说明：全局混合测度的最优运输**不是**仿射，故只主张标记分支）。

冻结候选使用**恒等运输**（`a=1,b=0`，即 `honest_no_readout`）：训练仍然进行，但输出被映射到参考本身。实测该映射在 4/5 市场上与 `forest_reference_only` 逐位相同（maxabs 0.0），只有 SH/HB 因训练产生非零 `θ`。

**(e) 前缀诚实参考（prefix-honest reference）**
训练期的参考库由“只用到该起点为止的前缀数据”重算（600 个前缀森林，`forest_prior_honest_round5`），避免训练查询用到未来标记；测试期参考与 insample 版本**逐位相同**，因此“诚实 vs insample”的测试差异只来自学习到的头，不来自参考。

**(f) 5 seed 概率混合**
5 个种子的原始叶子概率（含零原子）先平均，再条件化非零质量，再施加固定规则 `p` 与 γ 储层。等权“条件化后再平均”只作诊断，不作锚。

**(g) 闭式 CRPS**
混合分布的 CRPS 由 CDF 积分闭式得到（`R.cdf_integral`），与 `fast_crps` 的梯度检查一致（误差 <1e-11），不做蒙特卡洛。

## 3. 实测证据（全部相对 `rule_qrf`，20 格等权）

| 项目 | MQS | CRPS | 说明 |
|---|---:|---:|---|
| `honest_no_readout` | **+0.2662%**（14/20） | **+0.2031%**（12/20） | 37 个配置中两项同时第一 |
| 自助法区间（块长10） | [+0.0327, +0.5063] | 跨 0 | P(>0)=0.9855 |
| 自助法区间（块长5） | [−0.0489, +0.5811] | 跨 0 | P(>0)=0.9535 |
| SHANGHAI | +3.3019% | +3.14%（matched 对照） | 唯一有区间支持的市场 |
| ex-SHANGHAI | +0.0135%（P=0.535） | +0.0109%（跨 0） | 与基线不可区分 |
| 相对 `qrf_original` | +0.4400% | +0.3585% | 由“候选/规则”与“规则/QRF”两段几何链式合成；其中规则/QRF 一段两档 CI 均为正，候选/QRF 的区间未直接计算 |

机制消融（同协议、同种子）：

| 消融 | 结果 | 解释 |
|---|---|---|
| 去掉零原子与参考（`uniform_static`） | −2.8092% | 零膨胀与条件参考合计贡献约 2.8% |
| 用普通 QRF 取代规则参考（`qrf_original`） | −0.1747% | 规则零原子 + γ 储层 = +0.17% |
| 参考本身（`forest_reference_only`） | +0.1027% | γ=.05 平滑的净贡献 |
| 前缀诚实化（vs `matched_insample`） | +0.6800% | CI 全正，但 SH +3.30%、ex-SH +0.013% |
| 仿射运输（vs `matched_reweight_only`） | +0.3696% | CI 跨 0；SH +3.14%、ex-SH −0.335% |
| 跨市场状态列（base 复现冻结参考 +0.1028%） | 合计 −0.325%（10/20） | GZ −1.553、HB −0.614、SH +0.484、CEA +0.040、EU +0.007；打乱同列 −0.023% |
| LSTM/TCN/Transformer-QR | −0.66%/−1.62%/−1.40% | 深度分位回归全部劣于参考 |
| 内层选 λ 的 ESS 收缩 | −0.2271% | 内层验证与截断后测试不同分布 |
| 森林分裂目标 B/C/D（相对 A） | −1.09%/−1.01%/−1.35% | 换分裂准则在 inner 已更差，维度饱和 |
| 跨市场分位数池化（测试期扫描 λ=.4） | +2.01%（13/20） | 见 §3b：截断后重标定伪影，不可估 |

## 3b. 最大假阳性：跨市场离散度重标定

把本市场参考分位数按“其它市场同日分位数的离散度 / 本市场离散度”缩放，测试集可达 +2.01%（λ=.4，13/20；h=20 为 +6.0%），四类拆解显示它主要是逐市场离散度缩放（CEA 需 ×1.7、GZ 需 ×0.8），同日信息仅占 0.3–0.4 个百分点。但把缩放系数改为**只用截断前验证窗估计**后为 −0.505%（相对 rule，9/20），上海 −5.48%：截断前估计 1.10/1.22、截断后实际 0.27/0.23，方向翻转。故该现象是截断处结构断点导致的假性提升，与 ESS 失败同源，已从所有结论中剔除，仅作为限制写入。

## 4. 明确不能宣称的内容

- 不能宣称全局显著提升：ex-SH 的区间跨 0。
- 不能宣称神经结构创新：所有神经变体都未超过参考本身；`rbf_prior` 的增益在 4/5 市场上等于参考。
- 不能宣称跨市场信息有效：5/5 市场为负。
- 不能宣称“指数倾斜/KL 更新”类新颖性（Meucci 2008）、 “基线+NN 指数校正”新颖性（DRN Avanzi 2024）、森林条件分布新颖性（DRF Ćevid 2022）、全局混合测度上的最优运输（仅标记分支成立）。
- 不能把历史留出集当独立验证，也不能把 `P_boot` 当 p 值。

## 5. 复现入口

```
python research/neural_model_restart_20261021/forest_prior_honest_round5.py
python research/neural_model_restart_20261021/prior_readout_transport_round5.py
python research/neural_model_restart_20261021/round5_results_table.py
python research/neural_model_restart_20261021/bootstrap_round5.py
```

全部输入 SHA256 记录在各 `*_manifest.json` 的 `frozen_sha256_before/after` 中，且运行前后一致。
