# 已发表碳价预测研究的实际做法审查

审查日期：2026-10-19。研究目录：`literature_aligned_model_search_20261019d`。

## 1. 本轮完成了什么

**本轮读到了 7 篇论文的实质方法/实验原文，并另行核对了 1 篇 Energy 2026 论文所链接的作者公开实现。** 这 8 项不能合称“8 篇全文”。P1 是 2023 卷期、2022-12-16 首次在线；若严格按首次在线发生于 2023–2026 年计算，仍有 6 篇符合范围的方法原文。RF–LSTM 读取的是出版社 accepted manuscript，版本差异已保留。SWEWT 的 MDPI 官网访问受限，但其 ResearchGate 全文明示由 MDPI 提供；本轮读取的是该原文，不是他人的摘要。

选样目的是找能实际复现、且能解释模型增量的发表做法，覆盖多变量分解、二次分解、残差纠正、神经网络、树模型、符号回归与区间校准。**这是有目的的案例审查，不能据此说全领域“多数论文都泄露”或“多数论文都采用某协议”。** “未说明”也不表示作者实际上做错了。

本报告只报告文献/源码证据，没有执行这些作者模型、复现其成绩或打开旧研究再试分。配套 JSON 给出逐项字段、来源、版本、阅读范围和代码哈希。

## 2. 最值得落实的三个对照

1. **用 SWEWT–GRU 的四种分解范围搭建文献复现实验。** P3 把整段分解、训练验证合并、只训练整段、全部按起点滑窗明确区分，并在同一研究中比较。这是本轮最直接的“全样本分解也可以做，但必须标明其信息条件”的已发表先例。
2. **把“先预测、再分解残差”作为一个独立模型增量。** P5 的 WOA–XGBoost 先给价格预测，再对残差做 CEEMDAN/VMD 和分组预测，最后加回。它与直接分解价格并非同一结构；下一实验可以检验残差纠正是否留下可重复收益。原文未说明的残差生成/筛选范围必须公开设定，不能替作者补成已经严格执行。
3. **利用 TFR 原文直接给出的公开多变量数据，做 AE、分解、TCN 的模块对照。** P4 有结构和消融，作者数据目录可读，落地成本低于从商业数据平台重新拼数据。数据是否与我们旧样本重叠，交由本轮数据审查决定，公开可取得不等于全新未暴露。

Energy 2026 作者实现还提供一个很具体的旁证：预分解分量、QLattice 符号回归、验证残差经验分位数区间可以组成完整发表框架。它适合做代码流程对照；目前不能当作已取得该论文全部方法、基线和消融。

## 3. 从实际原文能学到什么

### 模型创新的组织方式

本样本中的论文通常把增量落在某个操作：同步分解外生变量并重构、改善高频分量选择、换成残差建模、改变不同频率的预测器，或把点预测扩成区间。论文随后使用模块比较、多市场、误差指标或统计检验解释该操作的收益。由此可以学习“明确任务困难 → 指定一个能检验的模型变化 → 给出相应对照”的组织方式；仅增加几个算法名字仍不足以把收益归因于所说机制。这里的观察限于下表，来源见各项。[M1–M7]

### 评价指标与独立比较

已读主文献主要用 MAE、RMSE、MAPE、R²，部分加入方向准确率与耗时；P8 代码又报告区间覆盖率和宽度。已读主比较表/代码中没有发现明确的随机游走基线，但这不是全领域结论，也不保证未读附录中没有。ARIMA、Holt–Winters 或深度模型清单不自动等同于已经计算了 $\hat p_{t+1}=p_t$ 的基线。[M1–M7, C1R]

有消融、显著性检验和“已发表”是三个不同事实。P5 的部分已发表模型对照没有使用百度指数，输入差异会妨碍把全部优势归因于模型。P6 明确取 paired t 与 Wilcoxon 中较小的 p 值；该操作本身不证明错误率被控制。P1 把显著性翻译成模型胜出的概率，本报告不继承这种解释。这些是可核查的局部方法描述，不能据此概括论文的全部质量。[M1, M5, M6]

### 信息范围要逐环节说清楚

“前 80% 训练、后 20% 测试”不能单独说明分解、插值、缩放、PACF、特征筛选和残差生成分别用了哪些日期。P3 明确给训练期极值和四种分解范围；P7 明确滚动重分解。另几篇正文没有把这些边界交代到可直接复现程度，应保留为未知。TCN 的因果卷积只约束网络卷积方向，BiLSTM 在历史窗口内双向处理也不自动读取真实未来；关键仍是它们实际收到的输入。[M1–M7]

## 4. 逐篇方法与复现字段

| ID | 发表/阅读级别 | 实际增量 | 可得材料 |
|---|---|---|---|
| P1 | Energy Science & Engineering 11(3):1148–1179，2023卷期；首次在线2022-12-16，单列此日期边界；方法原文 | MFIF同时分解价格和市场类外生序列；按价格IMF样本熵重构并同步重构外生IMF，TFT做分量预测与变量/时滞解释。 | [原始来源](https://scijournals.onlinelibrary.wiley.com/doi/10.1002/ese3.1380) |
| P2 | Energy Science & Engineering 12(6):2568–2590，2024-05-13在线；方法原文 | CEEMDAN→样本熵分高/低/趋势→高频再VMD→IPSO优化BiLSTM分量预测→另一BiLSTM做非线性集成；IPSO改惯性权重/学习因子。 | [原始来源](https://scijournals.onlinelibrary.wiley.com/doi/10.1002/ese3.1769) |
| P3 | Energies 17(17):4358，2024-08-31发表；方法原文 | 每起点只分解最近w点EWT，取末l点分量并与原始值共同输入GRU；TPE优化；明确比较四种分解信息范围。 | [原始来源](https://www.mdpi.com/1996-1073/17/17/4358) |
| P4 | Energy Science & Engineering 2025卷期；首次在线2024-12-17；方法原文 | CEEMDAN价格特征与多源外生数据拼接，经autoencoder压缩，再以3个残差块TCN预测。 | [原始来源](https://scijournals.onlinelibrary.wiley.com/doi/10.1002/ese3.2015) |
| P5 | Carbon Balance and Management 20:55，2025-11-14发表；方法原文 | 先WOA-XGBoost预测价格，再CEEMDAN分解残差、VMD细分IMF1；模糊熵重构4组、重做特征筛选与XGBoost；加回原预测。 | [原始来源](https://link.springer.com/article/10.1186/s13021-025-00348-7) |
| P6 | Scientific Reports 16:1803，2026卷期；2025-12-18首次在线；方法原文 | CEEMDAN→AFETM频率/能量双阈值筛高频→GWO优化VMD二次分解→GWO调学习率/dropout的多头注意力BiLSTM。 | [原始来源](https://www.nature.com/articles/s41598-025-31325-2) |
| P7 | Scientific Reports，2026-01-14发表；本轮方法读取为出版社accepted manuscript；方法原文 | 每滚动窗CEEMDAN→多尺度熵分组；RF预测最高频组，LSTM预测其余组，合成预测；BO调参并评价准确性/耗时权衡。 | [原始来源](https://www.nature.com/articles/s41598-026-35085-5) |
| P8 | Energy，2026；官方搜索确认DOI/题名/代码链接，出版社方法全文未取得；代码深读，论文未全文 | 从公开实现可核：TGSINFO优化VMD→各分量QLattice符号回归→求和点预测→验证绝对残差经验分位数校准对称区间。 | [原始来源](https://www.sciencedirect.com/science/article/pii/S0360544226004755) |

### P1. A novel interpretable model ensemble multivariate fast iterative filtering and temporal fusion transform for carbon price forecasting

[官方入口](https://scijournals.onlinelibrary.wiley.com/doi/10.1002/ese3.1380) · DOI：`10.1002/ese3.1380`。Energy Science & Engineering 11(3):1148–1179，2023卷期；首次在线2022-12-16，单列此日期边界。证据：M1。

- **市场与样本：**广东、北京、上海、湖北、深圳，从各市场设立至2022-06-02；删除零成交日；样本分别1755/1250/1235/1909/1921。
- **训练/验证/测试：**时序70/10/20；例如广东1229/175/351。输入过去10步预测下一步。
- **实际模型增量：**MFIF同时分解价格和市场类外生序列；按价格IMF样本熵重构并同步重构外生IMF，TFT做分量预测与变量/时滞解释。
- **分解与切分：**正文写原多变量序列分解/对称端点延拓；未交代是否每训练/验证/测试起点重新估计，不据图判定全样本测试泄露。
- **缩放/清洗：**进入模型前min–max，拟合极值的日期范围未明确。
- **特征筛选：**TFT内置变量选择；PACF选10步、熵分组和旁支RF/因子分析的拟合样本边界未明确。
- **外生变量时点：**Wind、Yahoo Finance，同日期对齐及插值；涵盖国际能源/EUA、汇率、经济、气象。具体收盘时区、发布时间、历史版本和可用滞后未说明。
- **调参与随机性：**广东验证集比较batch 16/32/64/128与units 4–128；验证集自适应降学习率与早停；Adam，MAE损失，dropout .2。 未说明具体随机种子或逐seed结果。
- **基线与随机游走：**BP、TCN、RNN、LSTM、GRU、TFT、MEMD-SE-TFT及外生变量处理对照。 已读主比较表未见随机游走/naive；不是对全领域的判断。
- **消融/显著性与指标：**有分解/外生变量处理比较、DM及灰关联；正文把95%显著性表述为模型以>95%概率更好，本审查不继承此概率解释；HAC细节未说明。 指标：MAE、RMSE、MAPE、R²、方向准确率DA。
- **代码/数据可取得程度：**原始来源写明，未见可下载作者代码或归档数据链接。
- **复现判断：**中：多变量分解/重构与TFT是可对照设计；数据时点与分解边界须自行明确。

### P2. A carbon price ensemble prediction model based on secondary decomposition strategies and bidirectional long short-term memory neural network by an improved particle swarm optimization

[官方入口](https://scijournals.onlinelibrary.wiley.com/doi/10.1002/ese3.1769) · DOI：`10.1002/ese3.1769`。Energy Science & Engineering 12(6):2568–2590，2024-05-13在线。证据：M2。

- **市场与样本：**广东1245日2017-07-04–2022-09-30；北京1247日2014-03-14–2022-09-30；湖北1265日2017-04-28–2022-09-30；EUA1250日2017-11-21–2022-09-30。依Table1；正文广东终点另误写2020。
- **训练/验证/测试：**前80%训练、后20%测试；训练中10%用于所谓cross-validation，具体fold时序未说明。
- **实际模型增量：**CEEMDAN→样本熵分高/低/趋势→高频再VMD→IPSO优化BiLSTM分量预测→另一BiLSTM做非线性集成；IPSO改惯性权重/学习因子。
- **分解与切分：**价格先CEEMDAN、高频再VMD，但相对样本划分的估计范围未说明。
- **缩放/清洗：**已读方法中未找到数据归一化拟合范围。
- **特征筛选：**样本熵分组、历史价格特征；拟合边界未说明；无外生特征筛选主模型。
- **外生变量时点：**主模型使用历史价格；宏观/能源等外生因素是未来工作。
- **调参与随机性：**IPSO人口20×100轮；VMD层数10；优化BiLSTM层数、units、batch、学习率、dropout；1日超前。 每模型平均10次运行；未给具体seed列表。
- **基线与随机游走：**17项比较含ARIMA、BP、DNN、GRU、LSTM、BiLSTM、PSO/IPSO及分解变体。 已读主表未见RW/naive；ARIMA不等同于已落实RW基线。
- **消融/显著性与指标：**有优化器、分解、网络结构比较；未找到正式DM/配对显著性检验说明。 指标：R²、RMSE、MAE、MAPE、SIGN方向指标。
- **代码/数据可取得程度：**数据来自碳交易网站/Investing；Data availability声明真实性与原创性，未提供可直接下载的数据/代码链接。
- **复现判断：**中：有明确组合层级及10次重复；第二层集成的训练预测是否样本外未说明。

### P3. Carbon Price Forecasting Using Optimized Sliding Window Empirical Wavelet Transform and Gated Recurrent Unit Network to Mitigate Data Leakage

[官方入口](https://www.mdpi.com/1996-1073/17/17/4358) · DOI：`10.3390/en17174358`。Energies 17(17):4358，2024-08-31发表。证据：M3。

- **市场与样本：**主广州/广东GDEA1972交易日，2014-03-11–2023-03-10；扩展湖北/北京/天津。扩展样本分别1365/1122/672，完整日期此轮未逐项读出。
- **训练/验证/测试：**时序70/10/20，1380/197/395；l=17。
- **实际模型增量：**每起点只分解最近w点EWT，取末l点分量并与原始值共同输入GRU；TPE优化；明确比较四种分解信息范围。
- **分解与切分：**Case1训练+验证+测试整段；Case2训练+验证合并、测试滑窗；Case3训练整段、验证及测试滑窗；Case4全部按起点滑窗。作者用leakage统称，但本审查保留三个不同信息边界。
- **缩放/清洗：**Eq28明确极值来自训练集。
- **特征筛选：**历史价格单变量；l=17的正文解释来自ADF检验程序自动选lag17后的输入长度讨论；PACF原文是二阶截断。该统计估计范围此轮未确认。
- **外生变量时点：**主实验无外生变量，因此无跨时区/统计发布期问题。
- **调参与随机性：**验证集TPE；窗口24–64、分量2–4，及dropout/batch/学习率；Table4明确TPE runs=10、最大epochs=1000、patience=100。 未给具体seed列表；TPE runs=10不能直接解释为10个独立随机seed重复。
- **基线与随机游走：**GRU、LSTM、RF、SVR、SWVMD-GRU、SWCWT-GRU、SWEWT-LSTM等。 已读比较表未见RW。
- **消融/显著性与指标：**四信息范围对照、分解和骨干对照、多市场；未见正式显著性检验说明。 指标：MAE、RMSE、MAPE、R²、误差方差VAE、时间。
- **代码/数据可取得程度：**MDPI提供全文可读；数据声明指向文章及联系作者，本轮未取得作者代码/完整数据包。
- **复现判断：**高：最明确的已发表全样本分解/滑窗分解对照协议，可直接作为复现实验结构。

### P4. TFR: A Temporal Feature-Refined Multi-Stage Carbon Price Forecasting

[官方入口](https://scijournals.onlinelibrary.wiley.com/doi/10.1002/ese3.2015) · DOI：`10.1002/ese3.2015`。Energy Science & Engineering 2025卷期；首次在线2024-12-17。证据：M4、D1。

- **市场与样本：**上海Wind日价格，2013-12–2021-03；公开库另含湖北、广东与外生因素。
- **训练/验证/测试：**Table1：训练2013-12-22–2019-08-02；验证2019-08-03–2020-06-03；测试2020-06-04–2021-03-23。文字给2049/303/263且称80/10/10，比例/交易日称谓需对照数据，不私自修正。
- **实际模型增量：**CEEMDAN价格特征与多源外生数据拼接，经autoencoder压缩，再以3个残差块TCN预测。
- **分解与切分：**正文展示上海序列CEEMDAN结果，未明确按每起点重分解/只训练期估计。
- **缩放/清洗：**未说明数据scaler拟合区间；TCN的weight normalization是网络参数操作，不能当作训练期数据缩放证据。
- **特征筛选：**AE特征压缩，具体拟合样本边界未明确。
- **外生变量时点：**能源、全球股指/EUA、AQI、百度搜索与政策不确定性；具体发布时间、统计版本和可用滞后未说明。
- **调参与随机性：**固定表：lr .01、batch64、1000 epochs、Adam；AE8/64、TCN[64,32,16]，kernel3，dilation[1,2,4]，dropout .3；未说明搜索预算/seed。 未说明。
- **基线与随机游走：**SVR、LSTM、GRU、TCN，EMD/EEMD/CEEMDAN组合。 已读Table5–6未见RW。
- **消融/显著性与指标：**有去AE、去CEEMDAN等消融；未找到显著性检验。 指标：MAE、RMSE、MAPE、R²。
- **代码/数据可取得程度：**正文直接链接Zzysty/carbon_dataset，目录和README已核；该库提供数据，未含模型源码。
- **复现判断：**高：公开多变量数据与可落地模型结构；实际复现需锁定日期对齐/缩放/AE/分解口径。

### P5. Method for predicting the price of carbon based on quadratic decomposition and multiscale prediction

[官方入口](https://link.springer.com/article/10.1186/s13021-025-00348-7) · DOI：`10.1186/s13021-025-00348-7`。Carbon Balance and Management 20:55，2025-11-14发表。证据：M5。

- **市场与样本：**湖北1789日2016-09-12–2024-05-02；EU1577日2018-01-02–2024-03-27；Wind与百度指数等15外因。
- **训练/验证/测试：**主实验前70%开发/后30%评价；另为对比已发表模型用湖北2018-01-09–2021-06-30的80/10/10、EU2021-01-04–2021-11-22的80/20。
- **实际模型增量：**先WOA-XGBoost预测价格，再CEEMDAN分解残差、VMD细分IMF1；模糊熵重构4组、重做特征筛选与XGBoost；加回原预测。
- **分解与切分：**残差分解明确；残差来自样本内还是样本外预测、分解是否含测试残差未说明。不能因残差纠正本身判定泄露。
- **缩放/清洗：**min–max公式给出，极值拟合范围未明确。
- **特征筛选：**PACF选择历史滞后，RF对15外因取8；分解后再筛选。筛选估计范围未明确。
- **外生变量时点：**互为外因的EU/湖北价格、AQI、能源、股指、百度搜索；历史价格有滞后，跨时区与百度指数可用时点未说明。
- **调参与随机性：**WOA优化XGBoost；正文指Table7，其独立表体未取得；验证目标/调参预算/seed未说明到可复现程度。 未说明。
- **基线与随机游走：**GBDT、TFT、CEEMDAN-BiGRU、SVR、XGBoost、COA/WOA及分解变体；另有已发表模型匹配样本对照。 已读主表与比较叙述未见RW。
- **消融/显著性与指标：**有优化器/一二次分解对照、平方误差DM；匹配论文对照明确有些基线未使用百度指数，因此优势不能仅归因于模型结构。 指标：MSE、MAE、R²、RMSPE/RMSE、MAPE；指标命名与部分数值应复核。
- **代码/数据可取得程度：**数据声明仅表示同意在适当情况下放入公开仓库，没有具体数据URL；未见作者源码。
- **复现判断：**高（机制对照）：先预测再分解残差与直接分解价格是实质不同的增量；应在复现实验明确残差生成时间。

### P6. Intelligent carbon price prediction system based on improved secondary decomposition and multi-head attention bidirectional long short-term memory (BiLSTM) model

[官方入口](https://www.nature.com/articles/s41598-025-31325-2) · DOI：`10.1038/s41598-025-31325-2`。Scientific Reports 16:1803，2026卷期；2025-12-18首次在线。证据：M6。

- **市场与样本：**湖北2386、广州2384日：2015年初–2024-10-31；深圳2724日：2013-06-18–2024-10-31；交易平台来源。
- **训练/验证/测试：**已读正文未说明train/validation/test比例、日期或滚动方法；表格独立页未成功取得。
- **实际模型增量：**CEEMDAN→AFETM频率/能量双阈值筛高频→GWO优化VMD二次分解→GWO调学习率/dropout的多头注意力BiLSTM。
- **分解与切分：**分解各市场价格序列；相对切分的估计范围未说明。
- **缩放/清洗：**min–max、插值、3σ异常处理；其拟合范围未说明。
- **特征筛选：**AFETM按IMF频率95分位/5%能量阈值筛选；平均价、成交量额、累计量额作为特征，筛选及变量时间边界未说明。
- **外生变量时点：**主要是同市场日交易/累计特征；用于预测哪一天、是否在起点已发布未说明。
- **调参与随机性：**CEEMDAN100次、noise .2；GWO调VMD alpha10–4000/K1–15；预测GWO以RMSE寻优50轮；hidden128/batch64/epochs100，Adam、早停。 未说明具体随机种子或重复次数。
- **基线与随机游走：**11项含LSTM/BiLSTM、多头注意力、GWO/分解变体及DA-FEDformer、多尺度Transformer。 已读比较段未见RW。
- **消融/显著性与指标：**有AFETM/GWO/MAM消融；paired t与Wilcoxon，正文以两者较小p值作综合判据；不能把取较小p本身当作控制错误率的证明。 指标：MSE、RMSE、MAE、MAPE。
- **代码/数据可取得程度：**数据可向通讯作者合理请求；未见公开模型代码。
- **复现判断：**中低：组合与超参数较具体，切分/预测起点缺失影响可核对复现。

### P7. RF-LSTM carbon price prediction based on CEEMDAN decomposition and multiscale entropy reconstruction

[官方入口](https://www.nature.com/articles/s41598-026-35085-5) · DOI：`10.1038/s41598-026-35085-5`。Scientific Reports，2026-01-14发表；本轮方法读取为出版社accepted manuscript。证据：M7。

- **市场与样本：**湖北2015-03-26–2025-03-27，2408日；EU2014-04-01–2024-04-08，文称2587日。
- **训练/验证/测试：**明确先时序80/20：湖北1927/481；EU文列2096/518，与2587不相加，原文算术不一致。§2.5一步滚动，§3实际step=5，其具体预测/更新解释仍待代码确认。
- **实际模型增量：**每滚动窗CEEMDAN→多尺度熵分组；RF预测最高频组，LSTM预测其余组，合成预测；BO调参并评价准确性/耗时权衡。
- **分解与切分：**初训练窗分解；得到下一实际值后添入、删最早值，重复分解/预测；初窗形成的IMF分组合并规则在后续窗口沿用。
- **缩放/清洗：**accepted正文未说明数据归一化拟合范围。
- **特征筛选：**初训练窗多尺度熵m3、r=.15std、尺度1–20，分组规则用于后续滚动；无外生特征主模型。
- **外生变量时点：**历史碳价单变量；无跨市场外生时点。
- **调参与随机性：**BO：RF树数50–500/depth3–15等；LSTM3层units30–100、dropout.1–.5、lr.001–.01；验证目标、BO轮数未明确。 未说明。
- **基线与随机游走：**Transformer、CNN-LSTM-Attention、CNN、XGBoost、Holt-Winters、RF/LSTM及分解/熵重构变体。 已读比较/框架未见RW。
- **消融/显著性与指标：**有去重构/RF/LSTM的消融、DM、误差与耗时综合指标；accepted版DM公式排版及误差差/损失差定义需复核。 指标：MSE、RMSE、MAE、MAPE、Dstat、runtime及综合Score。
- **代码/数据可取得程度：**声明数据在正文或补充材料；本轮未取得实际补充数据文件/作者代码。
- **复现判断：**高：给出完整滚动分解思想与混合频率预测，但不能原样补猜样本计数/step含义。

### P8. An integrated prediction framework combining variational mode optimization decomposition, empirical quantile regression, and QLattice for dynamic point and interval prediction of carbon prices

[官方入口](https://www.sciencedirect.com/science/article/pii/S0360544226004755) · DOI：`10.1016/j.energy.2026.140372`。Energy，2026；官方搜索确认DOI/题名/代码链接，出版社方法全文未取得。证据：C1P、C1R、A1、D2。

- **市场与样本：**作者广州示例：按脚本清洗后取1522个价格观测，2016-11-08–2023-05-31；原表有1603个带日期行、80缺价，不能把原始行数当模型样本。论文完整样本/其他市场未取得。
- **训练/验证/测试：**分量构造5滞后后，时序90/10；前90%以train_test_split(test_size=.4, random_state=42)随机分成训练与验证/校准，约54/36/10；默认shuffle=True经官方文档确认。
- **实际模型增量：**从公开实现可核：TGSINFO优化VMD→各分量QLattice符号回归→求和点预测→验证绝对残差经验分位数校准对称区间。
- **分解与切分：**MATLAB对1522点整段优化VMD后分解；notebook先载预分解CSV再生成滞后/切分。独立来源审查确认16分量求和与同一清洗后价格序列最大差4.30e-8，支持这条流程的数值桥接（未执行MATLAB）。
- **缩放/清洗：**已读脚本与notebook未显式scaler；不能推断feyn内部完全不做变换。
- **特征筛选：**每分量过去5步，QLattice auto_run(train_t)后取models[0]；没有把验证集传入auto_run。
- **外生变量时点：**可见代码为单变量分解分量历史滞后。
- **调参与随机性：**可见VMD优化5个体×5轮，alpha100–1.5×序列长，K2–15；QLattice自动搜索的预算/版本未锁定。 内层划分42；未见MATLAB/QLattice随机种子。
- **基线与随机游走：**所读notebook只实现目标模型；论文完整基线不能由此确定。 可见notebook无RW；论文是否有RW未知。
- **消融/显著性与指标：**可见notebook无模型消融或显著性检验；论文全文未知。 指标：代码有R²/MSE/RMSE/MAE/MAPE、PICP/PINAW/CWC；alpha .05，区间用一个固定q加减点预测，不是代码中另训条件分位数模型。
- **代码/数据可取得程度：**作者仓库公开可读；原xlsx和notebook已按Git blob校验；需feyn.connect_qlattice外部服务，数据路径是待替换占位；仓库未声明许可证。未执行源码、未引用输出成绩。
- **复现判断：**高（文献式流程对照）：可精确重建操作顺序，但论文全文层面的全部声称尚不能复核。

## 5. 全样本分解的已发表对照：具体怎样使用

P3 §3.4.2 / Table 6 给出如下四个协议。下表 RMSE 都是**原论文报告值**，不是本项目复现值；它们只用来说明文献确实做过这些对照。[M3]

| 论文协议 | 训练/验证/测试表示怎样构造 | 原文 RMSE | 本轮解释 |
|---|---|---:|---|
| Case 1 | 训练、验证、测试一起整段分解 | 1.594 | 表示构造包含测试期价格；可作为明示此条件的事后文献对照，不能直接当逐日起点可用预测结果。 |
| Case 2 | 训练+验证合并分解；测试按起点滑窗 | 7.106 | 验证集与训练表示构造发生共享；论文说明测试在模型构建中不可见。 |
| Case 3 | 训练整段分解；验证/测试滑窗 | 5.998 | 未由此证明使用测试目标；需检验训练表示与应用时表示的差异。 |
| Case 4 | 各阶段均按起点滑窗分解 | 2.317 | 表示构造的可用信息规则一致；这是该论文主方法。 |

作者把这些情形放在“data leakage”标题下，本报告按实际信息范围描述。尤其不能把 Case 3 自动写成“测试集被泄露”，也不能只看 Case 1 最小误差就判断其真实预测最好。做文献复现时，可以把四种协议都保留，并在每个协议内部保持相同骨干、样本日期、预测起点、指标与调参预算。这样能回答“哪个模型在什么信息条件下改善”，也能检验训练表示与应用表示不一致是否解释差异。

## 6. Energy 2026 作者代码的可复核操作顺序

这部分来自作者仓库的可见实现，并非未取得的论文全文。固定 commit：`3e8a155f034f38cb20430e601c4edb13c22ed7fc`。[C1P, C1R]

1. MATLAB 脚本 `TGSINFO-VMD/TGSINFO_VMD_GZ.m` 读取广州工作簿，按数值矩阵逐行删 NaN，再逆序、转置、取 1522 点；对这一段做 TGSINFO 参数优化和 VMD 分解。该脚本没有 train/test 切分。

来源字节说明：该 MATLAB 上游 blob 为 `103d67df7f2d6bcda51b282b750b4aa67dc338ff`、2669 bytes；本地保存文本为 2670 bytes，可能规范化了末尾换行。本地文本 SHA256 仅用于本地审查追溯，不能声称是上游 raw 的原样字节 SHA256。
2. Notebook 先读取预分解的 `TGSINFO-VMD-GZ.csv`，对每个 IMF/残差列构造过去 5 步特征，删除滞后产生的空行，再以前 90%/后 10% 做时序分割。
3. 前 90% 用 `train_test_split(test_size=0.4, random_state=42)` 再分训练与验证。官方 API 的 shuffle 默认值是 True，所以这一步是随机划分，而不是尾段时序验证；总体约 54%/36%/10%，取整会改变精确数量。[A1]
4. 仅 `train_t` 进入 QLattice 的自动搜索，取第一模型；各分量预测相加。可见调用没有把验证集传给自动搜索。
5. 用验证集总残差的绝对值经验分位数计算一个 q，再给所有测试点预测加减同一个 q。代码设 alpha=.05；此实现没有另训随输入变化的条件分位数模型。

独立数据审查已完成桥接：1522行、16个分量之和，与按脚本清洗得到的2016-11-08至2023-05-31价格序列最大差为4.30×10⁻⁸，达到舍入误差量级；没有执行MATLAB重新分解。[D2] 因此可见实现的顺序确为整段预分解后再切分预测样本。随机校准本身不单独证明测试目标进入训练，前面的整段分解才是此实现需标明的信息条件。这个实现适合作为论文式对照，但不能把这套含后段价格的表示直接称为逐日起点可用的预测输入，也不能据可见notebook推断未取得论文的基线、消融或运行效果。

该 notebook 已按 Git blob 验证并只读取 code cells：SHA-256 `1e36a5c1f19e271adef449b6232602be8b31b504016776e6436a83ea212feb9e`，21 个代码单元。没有执行其外部 QLattice 调用，也没有把作者已保存的输出成绩当成我们的实测。

## 7. 可据此形成的发表证据

可以接受并保留文献常用设置，同时把它的条件写清楚。研究创新的实证部分至少应使读者判断：改动是什么、同信息集下是否改善、改善由哪个模块贡献、在哪些市场/时期失效。主文献对照可用论文式设置；若要声称可用于实际未来预测，则还需要相应起点可用信息下的结果。二者是不同科学问题，都有可报告的结果。

对当前新实验，最有信息量的安排是把一个模型增量与一个信息范围增量分开比较。例如先固定简单预测器，比较价格分解与残差分解；再固定该结构，比较整段与滑窗表示。调参与最终评估的日期、每次模型试验、具体 seeds、简单 RW 与较强基线、配对误差不确定性都应留可追溯记录。此建议是本轮证据综合后的实验判断，不是任何一篇文献已证明的定理，也不能保证录用。

## 8. 来源与实际阅读范围

- **[M1]** [原始来源](https://scijournals.onlinelibrary.wiley.com/doi/10.1002/ese3.1380)。出版社开放正文；已读方法、数据、实验设置及比较段。 范围：§2–3；§4.1–4.2；§5.1–5.2；Tables 3–5, 8。
- **[M2]** [原始来源](https://scijournals.onlinelibrary.wiley.com/doi/10.1002/ese3.1769)。出版社开放正文；已读结构、数据划分、调参、实验、数据声明。 范围：§3；§4.1–4.4；§5；Table 1, 4；Data availability。
- **[M3]** [原始来源](https://www.mdpi.com/1996-1073/17/17/4358)。官方HTML 429；改读明确标注由MDPI提供的全文。 [出版社提供的全文](https://www.researchgate.net/publication/383685619_Carbon_Price_Forecasting_Using_Optimized_Sliding_Window_Empirical_Wavelet_Transform_and_Gated_Recurrent_Unit_Network_to_Mitigate_Data_Leakage)。 范围：§2–3（根代理）；§3.2；§3.4.2；Tables 3, 4, 6；Data availability；出版页/全文来源声明。
- **[M4]** [原始来源](https://scijournals.onlinelibrary.wiley.com/doi/10.1002/ese3.2015)。出版社开放正文；已读结构、数据、超参数、比较、消融。 范围：§2.5；§3.1；§4.1–4.3；Tables 1, 4–6。
- **[M5]** [原始来源](https://link.springer.com/article/10.1186/s13021-025-00348-7)。出版社开放正文；已读方法、数据/预处理、实验、讨论；独立表7页未取得。 范围：Methods: WOA-XGBoost-CEEMDAN-VMD；Data and evaluating indicators；Parameter configuration；Comparative experiments；DM test；Data availability。
- **[M6]** [原始来源](https://www.nature.com/articles/s41598-025-31325-2)。出版社开放正文；已读模型构建、数据/预处理、调参、比较、消融、检验；网页表格部分仅标题。 范围：Proposed model；Empirical study；Normalization；Secondary decomposition；Prediction model construction；Ablation and statistical significance；Data availability。
- **[M7]** [原始来源](https://www.nature.com/articles/s41598-026-35085-5)。已核正式发表记录；正式HTML/PDF抓取不稳定，实质方法审查读取出版社43页accepted manuscript，不能保证与最终排版版逐字一致。 [实际阅读版本](https://www.nature.com/articles/s41598-026-35085-5_reference.pdf)。 范围：§2.5–2.6；§3；§4.1–4.3；Tables 1–4；Data availability。
- **[C1P]** [原始来源](https://www.sciencedirect.com/science/article/pii/S0360544226004755)。官方搜索结果确认题名、DOI及作者代码链接；正文403，未取得论文方法全文。
- **[C1R]** [原始来源](https://github.com/StriveOrange/TGSINFO-VMD-NEQQLattice)。GitHub公开树/脚本直接读取；固定commit notebook下载并只读21个code cells，未执行notebook，未读其输出成绩。 固定 commit：`3e8a155f034f38cb20430e601c4edb13c22ed7fc`。
- **[D1]** [原始来源](https://github.com/Zzysty/carbon_dataset)。TFR正文直接链接；已读取公开树和README；本审查未直接执行价格数据实验。
- **[A1]** [原始来源](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)。官方API文档确认shuffle默认True及random_state用途；作者未固定sklearn版本。

- **[D2]** [独立数据桥接收据](../provenance/PUBLIC_AUTHOR_WORKBOOK_AUDIT.json)。已核1522行、16分量、对应日期及4.30×10⁻⁸重构差；没有执行MATLAB。数据CSV SHA-256：`3d2deb6d8a3d9a6407a9f7d1b5b6ed0eceb11ffdf36b9a1dcad68f63a066ba17`。

### 未作为全文计数的线索

- [入口](https://www.mdpi.com/2071-1050/18/17/8967)：根代理仅官方搜索缓存；本代理HTML反复429；方法、数据声明与补充文件未取得，不用于凑全文篇数。
- [入口](https://link.springer.com/article/10.1007/s40747-026-02550-7)：已知2026-10-13早期发表的TLCC-DyG-STFusion；本轮PDF未取得，未继承旧轮摘要审查为新全文审查。
- [入口](https://github.com/merwanroudane/peltwtcn)：第三方实现，不能作为原作者代码或已发表原文证据；不计主表。

**局限：**这是发表做法与复现可用性审查，没有执行模型训练、确认论文全部数值、核对全部参考文献、完成统计效力审计或保证发表。正文版本差异、折叠表格缺失、未知时间戳均逐项保留。

