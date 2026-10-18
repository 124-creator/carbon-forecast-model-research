# 碳价格预测近期文献独立核查与方向边界

> 阅读顺序说明：这是数据认证完成前的研究备忘录，后文OHLC包含性候选已被实际字段审计否决。最终采用允许P落在H/L之外的报告价格语义，正式方法与协议以 `METHOD_ZH.md`、`PROTOCOL_ZH.md` 为准；直接数学近邻补充见 `DIRECT_NEIGHBORS_ZH.md`。保留原备忘录是为了呈现方向为何改变。

核查日期：2026-10-19（北京时间）。范围：2022—2026 年原始论文、出版社正文、作者机构全文与作者代码。此文件是研究判断材料，不是新模型已成立或实验已完成的声明。

## 一、当前判断

**尚未找到一个可以负责任地直接宣布为原创主模型、并预期超过随机游走的现成路线。** 仓库既往已覆盖时钟／事件反馈、残差、多期限、跨市场和分布形状；文献又已覆盖实时分解、一般收缩、DILATE、OHLC 结构变换及逆变换偏差修正。把这些技术重新组合或换名称，不能解决当前创新不足。

本轮最值得进入低成本实测的对象是：**日频 EUA 的 OHLC 新信息是否提供稳定的收盘价增量预测能力**。它首先是信号必要性检查。只有检查通过，才建议开发下文的“跨期限 OHLC 聚合一致路径学习”候选。任何仅改善最高／最低价或概率区间的结果，都必须明确标注预测对象，不能写成已改善原来的收盘价点预测。

## 二、十篇高相关论文及可用边界

证据等级：A＝实际阅读到方法与实验正文；B＝出版社／作者机构元数据与摘要或章节摘录核实；C＝另核作者仓库实际文件。B 级不能据摘要认定无泄漏或已复现。以下每篇的摘要和分析均控制为短评；未获取的细节直接留空。

| 论文与一手入口 | 核到的内容 | 对本项目的用途与限制 |
|---|---|---|
| **Ren et al. (2022), Carbon prices forecasting in quantiles**, Energy Economics 108, 105862. [DOI](https://doi.org/10.1016/j.eneco.2022.105862)；[Essex 接收稿](https://repository.essex.ac.uk/32112/1/Carbon_Prices_Forecasting_in_Quantiles.pdf) | A。Quantile Group LASSO / Group SCAD；月度 EUA 收益，2009-03—2020-12；18 技术指标＋44 宏观变量；训练到 2019-12。比较 LASSO、adaptive/group LASSO、ARMA、GARCH。 | 分位数分组收缩已有明确先例。正文的数据段不能支持大样本结论；须区分后续 6／12 个月评价段与独立长期滚动外测。读取的基线清单未见 RW。 |
| **Tan et al. (2022), Forecasting European carbon returns using dimension reduction techniques: Commodity versus financial fundamentals**, International Journal of Forecasting 38(3), 944–969. [DOI](https://doi.org/10.1016/j.ijforecast.2021.07.005)；[出版社](https://www.sciencedirect.com/science/article/pii/S0169207021001163) | B。周度 EUA 收益，53 个相关、商品与金融预测量；比较变量收缩与因子。作者报告样本依赖，模型与基准的平均能改善稳定性。 | “RW 锚定／模型与无预测基准平均”本身不新。未取得完整数据代码，不以摘要增益作本项目达标线。2021 为 online/DOI 年，2022 为卷期年。 |
| **Li et al. (2023), Forecasting carbon prices based on real-time decomposition and causal temporal convolutional networks**, Applied Energy 331, 120452. [DOI](https://doi.org/10.1016/j.apenergy.2022.120452)；[CityU 作者机构](https://scholars.cityu.edu.hk/en/publications/forecasting-carbon-prices-based-on-real-time-decomposition-and-ca/) | B。MEPT–ICEEMDAN 实时分解＋CTCN；输入层 adaptive group LASSO；EUA 的 1／5／10 步及分位数。机构核实 online 2022-12-08、卷期 2023-02-01。 | 避免一次性分解泄漏、因果 TCN、group penalty 均非空白。“Granger”也不能替换为干预因果结论。未读到足以核实全套复现的正文／代码。 |
| **Zhang et al. (2024), Carbon Price Forecasting Using Optimized Sliding Window Empirical Wavelet Transform and Gated Recurrent Unit Network to Mitigate Data Leakage**, Energies 17(17), 4358. [DOI](https://doi.org/10.3390/en17174358)；[出版社](https://www.mdpi.com/1996-1073/17/17/4358) | B。SWEWT–GRU，以广东收盘价检验滑动窗口分解，TPE 调参；摘要明确把数据泄漏作为问题。 | 证明“实时分解避免泄漏”已被直接研究。官网本轮多次 429，未声称完整阅读，未核清 RW、完整切分与公开训练代码。 |
| **Zhao et al. (2024), Carbon futures price forecasting based on feature selection**, Engineering Applications of Artificial Intelligence 135, 108646. [DOI](https://doi.org/10.1016/j.engappai.2024.108646)；[作者开放全文](https://freepose.github.io/paper/journal/eaai2024-Zhao%20%28Carbon%20futures%20price%29.pdf) | A。噪声扰动后预测误差差异用于筛选；2013-01-04—2022-08-31 日频，主要数据 Bloomberg/Wind；9:1、17:3、8:2 切分；五折筛选。 | PFS-LR/RR 在其表中明显优于树与 SVR，支持先测低维。正文按日对齐＋线性插补，发布时间与折内因果性仍需另核；全文未检到 RW。 |
| **Huang, Zhao & Wang (2024), Model-driven multimodal LSTM-CNN for unbiased structural forecasting of European Union allowances open-high-low-close price**, Energy Economics 132, 107459. [DOI](https://doi.org/10.1016/j.eneco.2024.107459)；[作者机构介绍](https://cem.cau.edu.cn/art/2024/5/20/art_34744_1025518.html)；[机构 PDF](https://cem.cau.edu.cn/module/download/downfile.jsp?classid=0&filename=f31489e677164e789c9c4183bab67495.pdf) | B。数值 OHLC＋蜡烛图特征，模型驱动 LSTM-CNN，目标是结构有效及无偏逆变换。正式元数据已核；机构全文本轮下载超时。 | 是 OHLC 新方向必须正面比较的最近邻，不能把排序约束或逆变换偏差修正说成新贡献。其 2022 CNN 前作亦已使用 unconstrained transform。 |
| **Niu et al. (2025), Paying attention to distortion: improving the accuracy of multistep-ahead carbon price forecasting with shape and temporal criteria**, Humanities and Social Sciences Communications. [DOI](https://doi.org/10.1057/s41599-025-05110-5)；[出版社全文](https://www.nature.com/articles/s41599-025-05110-5)；[作者数据](https://github.com/yuanchen286/EU-ETS) | A。SVMD＋LSTM＋既有 DILATE；两 EUA 段 2016–2019、2017–2020，各 1026 点，80/20，h=3/4/5。 | 两段大量重叠；主比较 MLP/CNN/LSTM。大幅 MSE 增益按正文明确对最差 CNN；“train–test splitting”一句不能证明逐起点分解。不能照搬其百分比作目标。 |
| **Liu et al. (2025), A Novel Multi-Task Learning Framework for Interval-Valued Carbon Price Forecasting Using Online News and Search Engine Data**, Mathematics 13(3), 455. [DOI](https://doi.org/10.3390/math13030455)；[作者仓库](https://github.com/dinggaoliu/Interval-valued-carbon-prices-forecasting) | B＋C。TFT 多任务，日内最高／最低价，新闻、Google Trends、市场数据；h=1/3/5。仓库实际列 README 与两个 CSV。 | 高低价区间不是具有某置信覆盖率的概率区间。公开数据可作新增字段候选，但“提供完整实现”的 README 不能替代实际不存在的训练脚本；数据规模/许可/字段时间须实查。 |
| **Ewald & Li (2026), A Novel Text-Based Framework for Forecasting Carbon Prices**, Journal of Forecasting. [DOI 与出版社全文](https://onlinelibrary.wiley.com/doi/full/10.1002/for.70202) | A。官方标 first published 2026-08-02；260 周，2020–2024；156 周滚动训练，h=12，共 54 预测；PCA 初始训练拟合固定，深度模型输入仅过去。 | 含 RW，且 ARIMAX 明确用未来已实现外生变量作条件基线；两者不可当同一可部署任务。数据合理请求提供。54 个重叠预测不是 54 个独立季度。 |
| **Bastianin et al. (2026), Forecasting the Price of Carbon with Macroeconomic and Financial variables**, JEDC 接收稿. [DOI](https://doi.org/10.1016/j.jedc.2026.105435)；[arXiv 元数据](https://arxiv.org/abs/2402.04828)；[v3 全文](https://arxiv.org/html/2402.04828v3) | A。2026-10-03 v3，官方 arXiv 标 accepted JEDC；136 月，2012-06—2023-09；BVAR/因子；6 月约 0.99、9 月约 0.961、12 月约 0.937 的 RW 相对 RMSFE，DM 未拒绝。 | 大 BVAR 无增益。年排放插值的前视风险已讨论并做去排放敏感性；LSEG 原数据明确不能公开分发。这是严肃低维参照，不是现成大规模公开数据主线。 |

## 三、补充核查：两个不能遗漏的近邻

1. **OHLC 变换不是新方法。** Huang et al., *A structural VAR and VECM modeling method for open-high-low-close data contained in candlestick chart*, Financial Innovation (2024), [DOI／完整正文](https://doi.org/10.1186/s40854-024-00622-6)。已读方法。其变换为 log L、log(H−L)、开盘和收盘相对区间位置的 logit；再用 VAR/VECM/SVR 并逆变换。故正价格、正范围、开收盘在高低之间以及这些位置特征不能独立支撑原创声明。

2. **公开预印本仍要核实现细节。** Pang et al., *Carbon Price Forecasting with Quantile Regression and Feature Selection* (2023), [官方 PDF](https://arxiv.org/pdf/2305.03224)。已读第 4–7 页。实现段明确以测试集均值处理 NaN/Inf；EUA 样本数量与其日期写法也不自洽。其分位数稀疏分组思想可辨识，但不得据其指标宣称严格样本外增益。上述问题限定于这份实际阅读的版本，不推断作者其他版本。

## 四、公开代码与数据核查

| 入口 | 当前核到的内容 | 本项目使用前还需检查 |
|---|---|---|
| [TravisCao/Quantum-LSTM](https://github.com/TravisCao/Quantum-LSTM) | 2023 Quantum Machine Intelligence 论文作者代码；保留 src 与 data；README 标 EUA 2014-01-01—2020-12-31，含 Price/High/Low/Open/Vol 及日历字段。 | 合约与价格定义；原始价格与预拼接 x_3d/x_5d 特征的区别；实际时间排序、训练切分、许可；不能将重叠衍生文件算新增样本。 |
| [dinggaoliu/Interval-valued-carbon-prices-forecasting](https://github.com/dinggaoliu/Interval-valued-carbon-prices-forecasting) | 实际目录为 README、original_data_matrix.csv、sentiment_compound_b.csv。 | 原始数据字段/日期/源许可；Google Trends 的回溯归一化与新闻当时可得性。未见训练源码，不能称端到端作者复现。 |
| [yuanchen286/EU-ETS](https://github.com/yuanchen286/EU-ETS) | Niu et al. 官方正文明确给此仓库作数据来源。 | 仓库实际可访问内容、两时间段重复、来自 Investing 的许可与连续合约定义。 |
| [edwinzhang98/JEM](https://github.com/edwinzhang98/JEM) | 2024 JEM 二次分解论文代码目录，含 IVMD、ICEEMDAN、LSSVM、TCN、SOA-GRU；MIT 软件许可。 | 软件许可不自动覆盖价格再分发；各分解调用边界是否逐起点；目录存在不等于一个可执行复现实验。 |

本轮仅核网页列出的仓库文件与 README，没有把这些公开仓库全部克隆并复跑。BMambaXer 等较新搜索命中未取得足够一手核验，未纳入核心结论。

## 五、唯一建议进入实测的方向

### 5.1 先做 OHLC 增量信号必要性检查

保持原目标：预测未来 h 个交易日后的收盘价，h 固定为 1、5、20。新增数据不改变这个主目标。利用获准开发区中的过去 OHLC，比较共同起点的 RW、带漂移 RW、仅收盘信息的正则线性预测器，以及相同训练法加 OHLC 信息的预测器。

在 t 日收盘后，定义：

\[
g_t=\log(O_t/C_{t-1}),\quad b_t=\log(C_t/O_t),\quad
u_t=\log(H_t/\max(O_t,C_t)),\quad
d_t=\log(\min(O_t,C_t)/L_t).
\]

g 是隔夜间隙，b 是日内涨跌，u/d 是上／下影线。所有量在 t 收盘时可得。H=L 时保留合法退化蜡烛，不除以零、不为了美化结果删除。先用少量滞后与固定长度过去汇总，避免大量技术指标搜索。

这是一个需要被数据支持的假设：同样的收盘收益路径下，隔夜与日内变化的分配、上／下影线是否包含下一时段条件均值的信息。它不是已确立的金融规律。若 OHLC 仅使波动率或范围更可预测，不能据此说原点预测目标已成功。

应固定三项比较：收盘信息→加 range；收盘信息→加 gap/body/wicks；全部 OHLC→以打乱时间的附加变量作负对照。所有模型使用相同滚动／扩展窗口、同样的调参预算、训练区拟合尺度、已成熟标签。负对照仅检验实现／信号，不可替代公平强基线。

继续开发的必要条件应由根任务预注册，并在看到该轮分数前冻结。建议至少要求开发期跨年份的方向一致性；同信息集低维竞争者不能已解释全部收益；外测误差及置信区间必须按共同起点计算。保留区是否可开由当前授权和仓库协议决定，此文不授权提前读取任何封存标签。

### 5.2 若信号存在，候选学习机制是什么

候选是**跨期限 OHLC 聚合一致的路径学习**，目前仅为待排重、待实验的研究假设，不能称为已选定原创模型。

让一个低维条件转移核生成未来日烛线创新 (g,b,u,d)，按定义还原每日 OHLC；所有期限来自同一组路径。h 日窗口目标通过固定的真实聚合算子得到：

\[
A_h(X_{1:h})=(O_1,\max_{1\le j\le h}H_j,\min_{1\le j\le h}L_j,C_h).
\]

训练直接评分经过 A_h 推送后的联合预测分布，而不是先预测变换空间均值再逆变换。训练损失还应保留终点收盘价 MSE，使用户原来的精度目标明确参与学习：

\[
\mathcal L(\theta)=\sum_{t,h:\,t+h\le T_{\rm train}} w_h\left[
S((A_h)_\#P_\theta(\cdot\mid\mathcal F_t),Y_{t,h})
+\lambda\frac{(\mathbb E_\theta[C_{t+h}\mid\mathcal F_t]-C_{t+h})^2}{s_t^2}
\right].
\]

其中 S 应选已有的 proper score；s_t 仅用过去信息固定；已成熟标签掩码不能省略。这里创新若成立，应来自“同一个条件转移机制如何利用不同期限的 max/min 聚合监督学习”，而非 proper score 或排序本身。尤其不可把 max E[H] 写成 E[max H]，二者一般不相同。

本稿尚未冻结转移核的具体参数族、优化与识别条件，因此这不是一条已经数学定义完整的最终模型。要在根任务开发时补全，并检查是否只是已知随机路径模型的应用。该谨慎限制是科学判断，不是要求用户另行确认。

### 5.3 必须能否证候选机制

最低对照应包括：同一生成器仅用一步监督；相同模型容量但各期限独立学习；OHLC 直接结构回归；同信息集 Ridge／AR／既有最强模型；RW 收盘预测。若 full loss 的收益只来自额外监督数量、既有尺度预测或加参数，则不能归因于聚合结构。若终点收盘 MSE 不改善，应保留负面结果并明确是否转为另一项 OHLC／范围预测研究，不能重写初始目标。

## 六、未解决事项

- 尚无可以独立支撑“原创方法＋严格价格点预测增益”的已完成证据链。
- 本轮未替根任务选择／验分用户的新实验，也未写入 GitHub。
- 尚未复跑所列公开作者代码；公开价格许可、合约滚动、字段时点与原始 OHLC 质量需由实际数据核查解决。
- 新的 2026 论文必须以正式 DOI、出版社或作者 arXiv 实体复核，不能只用一次搜索摘录；本表 Ewald 与 Bastianin 已有直接正文核验。
- 此文件不对任何期刊的分区、费用或录用作保证。目标期刊费用需另查当前官方作者指南。

