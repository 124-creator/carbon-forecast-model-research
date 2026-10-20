# 文献对比与发表方向判定

日期：2026-10-20。配套 `RESULT_VR_PROBABILISTIC_ZH.md`（我们的结果）。

## 0. 检索条件与效力声明

**必须先把这一点说清楚：本节的文献结论效力有限。**

- 本环境的 `web_fetch` 对所有外网域名返回 "resolves to a non-public IP address"，
  包括 sciencedirect.com、wiley.com、nature.com、mdpi.com、baidu.com。
  即**无法读取任何论文全文或摘要正文**。
- 百炼联网通道返回 HTTP 401（API-key 失效），且 Token Plan 身份本就不授权 MCP 搜索。
- 唯一可用的是内置检索，它只返回**标题、来源域名、以及检索索引里的零星正文片段**。

因此下文对文献的判断建立在标题与片段之上，**属于方向性判断，不是已核实的全文比对**。
凡标注「待核实」的，投稿前必须由能够访问全文的人复核。

## 1. 文献版图

检索到的 2024–2026 年相关文献，可归为四条线。

### 线一：混合深度学习 + 分位数回归（最拥挤）

- *A novel probabilistic carbon price prediction model: Integrating the transformer
  framework with mixed-frequency modeling at different quartiles* — Applied Energy, 2025
- *A hybrid model for carbon prices: Integrating investor attention, mixed-frequency data,
  quantile regression and deep learning* — Expert Systems with Applications, 2026
- *A hybrid carbon price prediction model based-combinational estimation strategies of
  quantile regression and long short-term memory* — J. Cleaner Production, 2023
- *An optimized extreme learning machine composite framework for point, probabilistic,
  and quantile regression forecasting of carbon price* — 2025

共同范式：更大的网络 + 更多外部变量（投资者关注、混频宏观）+ 分位数损失。
**共同点是把不确定性交给网络逐期限学出来**，不讨论区间宽度随 h 应如何增长。

### 线二：区间预测的组合方法

- *Shapley value-driven superior subset selection algorithm for carbon price interval
  forecast combination* — Scientific Reports, 2025

用 Shapley 值挑选区间预测的子集做组合，评价用 PICP/PINAW 一类指标。
这是与我们**最接近的工作**：同样关注区间质量，但路径是「组合多个区间」，
而非「校正单个区间的期限缩放」。**待核实其是否处理多期限。**

### 线三：可预测性 / 市场有效性

- *Is China's National Carbon-Allowance Price Predictable? An Interpretable Machine
  Learning and Volatility Analysis Around the 2025 Market Expansion* — Sustainability, 2026
- *UK Carbon Price Dynamics: Long-Memory Effects and AI-Based Forecasting* —
  Fractal and Fractional, 2025
- *Fractal properties with both long-term dependence and anti-persistence* —
  Emerging Markets Finance and Trade, 2024
- *Is EU-ETS Future Carbon Price Mean-reversion?* — 较早文献

这一线直接触及均值回归与长记忆，与我们 VR 的证据同源。
**待核实**：Sustainability 2026 那篇是否已给出「可预测性有限」的结论——
若已给出，我们的定位必须从「发现可预测性低」转为「如何在此条件下正确表达不确定性」。

### 线四：综述与可解释性

- *Forecasting Carbon Prices: A Literature Review* — Journal of Forecasting, 2025
- *Explainability in carbon price forecasting systems: A systematic literature review* —
  Knowledge-Based Systems, 2025
- *Toward Explainable and Policy-Aware AI for Carbon Credit Price Prediction* — arXiv, 2026

**Journal of Forecasting 在 2025 年刊出了碳价预测综述**，说明该刊对这一主题有持续编委兴趣。

### 线五（方法论竞争者）：共形预测

- *Enhancing reliability in prediction intervals using point forecasters: Heteroscedastic
  Quantile Regression and Width-Adaptive Conformal Prediction* — arXiv 2406.14904

**这是我们必须正面回应的竞争者。** 共形预测用校准集残差直接给出每期限的分位数，
天然能学到正确的区间宽度，不需要任何 VR 理论。

## 2. 空白判定

综合以上，我们的主张落在一个**具体且未被占据**的位置：

> 现有文献要么把区间宽度交给网络逐期限学习（线一），要么组合多个区间（线二），
> 要么用基于波动率模型 + √h 缩放的参数化区间（GARCH 类，散见于各文）。
> **没有文献论证 √h 这一缩放在碳市场上系统性错误，也没有给出正确的缩放形式。**

我们的证据恰好补这个洞：VR(h) 从 h=1 的 1.000 降到 h=40 的 0.10–0.17，
√h 假设使区间宽 2.2–3.1 倍，而实测宽度比与 1/√VR(h) 的相关达 0.9904。

**但必须承认一个削弱因素**：线一的分位数网络和共形预测都不会犯 √h 的错误，
因为它们不假设缩放形式。所以我们的贡献**不是「所有方法都错了」，
而是「参数化区间这条路径错了，且错得可预测、可修正」**。
这个范围窄，但它是真的，而且可以精确量化。

## 3. 差异化论证：为什么不是共形预测

这是投稿时最可能被拒的理由，必须提前想清楚。三条可站住的理由：

1. **数据效率。** 共形预测需要在每个期限上有足够校准残差才能给出稳定分位数。
   我们的确认窗长度是：广州 601–658、上海 221–259、湖北 38–53、EU 42–57。
   在湖北和 EU 这种样本量下，逐期限共形分位数不可靠。
   VR 用一个跨期限的平滑函数借力，参数极少（一条 VR 曲线），小样本下更稳。
   **这一点可以直接实验验证**（见 §5 实验 2），验证后就是硬论据。
2. **可解释与可外推。** VR 是市场微观结构的可解释量（暂态噪声占比），
   不是黑箱校准常数；它还能外推到未观测的期限。
3. **诊断价值。** 即使最终用共形，VR 仍能告诉研究者「这个市场的区间该比 √h 窄多少」，
   这是共形给不出的结构性信息。

## 4. 目标期刊

按「SCI 二/三区、非 OA、方法学 + 实证」筛选。**分区与收录状态我无法在此核实，
投稿前必须查当年 JCR / 中科院分区表。**

| 候选 | 理由 | 风险 |
|---|---|---|
| Journal of Forecasting | 2025 年刚刊碳价预测综述，编委兴趣明确；方法与实证并重 | 偏 SSCI，需核实是否满足「SCI」要求 |
| Energy | 非 OA，能源经济与预测方法都收 | 竞争激烈，需更强跨市场证据 |
| International Journal of Forecasting | 方法学最对口 | 门槛高，对理论新意要求严 |
| Resources Policy | 碳市场政策相关，非 OA | 更偏政策，方法学篇幅受限 |
| Emerging Markets Finance and Trade | 中国试点市场题材契合 | 影响因子偏低 |

**推荐首选 Journal of Forecasting**，理由是该刊 2025 年刚发碳价预测综述，
我们的论文正好是对该综述所指文献的方法学补充，编辑能直接看到定位。

## 5. 投稿前必须补的实验（按优先级）

1. **扩市场（最高优先）。** 现在只有广州、上海有统计效力；湖北 53 点、EU 57 点。
   EU 序列止于 2024-03-27。需要补 EU 至 2026、补长湖北，最好加全国 CEA。
   没有这一步，「碳市场」只能降级为「中国两个试点市场」。
2. **对共形预测的对比实验。** 固定覆盖率目标，比较 VR 参数化区间与 split-conformal
   在不同校准样本量（50/100/200/400）下的 CRPS 与覆盖率。
   若 VR 在小样本下占优，§3 的第 1 条论据成立；若共形全面占优，方向需重估。
3. **分位数回归基线。** 线一是领域主流。需实现线性分位数回归（各期限独立），
   与 VR 校正区间比 CRPS。若 VR 校正加在分位数回归之上仍有增益，则「即插即用」成立。
4. **位置模型无关性。** 已部分完成（RW / SCTD / GARCH / 滚动）。
   补一个线性岭回归位置，证明 VR 校正对任意位置模型都有效。
5. **补 PICP95 校准。** 当前广州 0.884–0.926，低于 0.95。
   需给出一个尾部校准方案（如对 VR 尺度做单调变换），否则这是明确的拒稿理由。

## 6. 结论（2026-10-20 修订）

方向定为：**碳价不确定性的两参数结构刻画**，见 `RESULT_TWO_PARAMETER_VR_ZH.md`。

```
VR(h) = ρ + (1-ρ)·(1-φ^h) / (h·(1-φ))
```

论文的一句话主张：

> 碳价约 90% 的长期限方差来自半衰期约 1 天的暂态成分（广州 ρ=0.11、φ=0.46；
> 上海 ρ=0.08、φ=0.51），使方差随期限的增长远慢于随机游走假设；
> 据此计算的预测区间在 20–40 日期限上使 CRPS 下降 11%–20%，
> 效果与非参数估计和滚动共形预测持平，但只需两个跨期限的参数。

### 相对 §5 定稿的变化及原因

原定"VR-MHPF 校正 √h 缩放"，本轮实验后调整为"两参数结构刻画"，理由三条：

1. **参数化完全替代了非参数查表**（§4 表，差距 0.0001 量级），
   且两个市场拟合出几乎相同的参数。两参数叙事比"我们查了个表"强得多。
2. **与滚动共形基本打平**（§5.3 表），"打赢共形"不成立，
   必须把卖点从"更准"移到"同样准但只用一个参数对"。
3. **单参数（局部水平模型）被数据否证**（`RESULT_TWO_PARAMETER_VR_ZH.md` §2），
   这反而是一个正面发现：短期限与长期限的不确定性确有不同结构。

### 仍然缺的东西（按优先级）

1. **扩市场。** 只有广州、上海有统计效力；湖北 53 点、EU 57 点且止于 2024-03。
   没有这一步，结论只能限定为"中国两个试点市场"。
2. **95% 区间欠覆盖**（广州 h=20 的 PICP95 仅 0.874）。正态尾部太薄，
   t 分布在 CRPS 最优下不被选中。这是最可能的拒稿理由，必须先解决。
3. **√h 的覆盖率其实更好**（0.928–0.967）。论文必须同时报 CRPS 与区间宽度，
   否则读者会认为我们的方法覆盖率更差。措辞已按此校正。
4. **分位数回归基线。** 领域主流的概率预测做法，未做对比。

### 期刊

首选 **Journal of Forecasting**（2025 年刚刊碳价预测综述，编委兴趣明确）。
备选 Energy、International Journal of Forecasting、Resources Policy。
**分区与收录状态无法在此核实，投稿前必须查当年 JCR / 中科院分区表。**

