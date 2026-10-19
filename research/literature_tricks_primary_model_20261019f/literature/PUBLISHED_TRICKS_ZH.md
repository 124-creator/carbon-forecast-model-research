# 已发表做法与阶段 f 主模型增强

整理日期：2026-10-19。本文依据已实际读取的一手正文、作者版本和公开作者代码，记录三类可以落地的增强：EWT 分量输入、成熟标签后的读出更新、密集多步监督。检索跨阶段 e/f 开展，本文在阶段 f 实现期间整理；方案的评分前固定时间以冻结协议与运行日志为准。

目标是在已有 signed_context_response 上检验具体增强收益。EWT 方案已调整为原始序列加全部三个分析分量，不再实施删除最高频分量。它将主模型从 9 个拟合标量扩为 15 个，必须如实计算容量；单独的 raw 密集监督仍为 9 参数。两个市场的评价尾段此前均已评分，本阶段全部结果属于开发证据。[项目决定与来源元数据：provenance/RUN_DATASETS.json；精确执行以冻结源码为准。]

## 1. EWT 分量输入：原文做法与四种信息设置

### 1.1 直接原始来源

[S1] Zhang 等，Carbon Price Forecasting Using Optimized Sliding Window Empirical Wavelet Transform and Gated Recurrent Unit Network to Mitigate Data Leakage，Energies 2024，17(17)，4358，2024-08-31，DOI 10.3390/en17174358。已读 MDPI 提供的全文镜像，重点为方法部分、§3.2.2、Table 4、§3.4.1–3.4.2、Tables 5–6。原文对历史窗口进行 EWT，把原始序列和分量的窗口尾部交给 GRU；l=17，w 在 24–64、分解尺度在 2–4 中搜索。Table 4 列 TPE runs=10、epochs=1000、patience=100；不能把 runs=10 解释为十个独立随机种子。[S1]

原文 §3.4.2 / Table 6 明确比较以下设置：

| 原文设置 | 分解时取得的数据 | 原文 MAE | 原文 RMSE |
|---|---|---:|---:|
| Case 1 | train、validation、test 全部联合分解 | 1.121 | 1.594 |
| Case 2 | train+validation 联合分解，test 逐窗口 | 6.492 | 7.106 |
| Case 3 | train 整段分解，validation/test 逐窗口 | 5.115 | 5.998 |
| Case 4 | 所有样本都逐历史窗口分解 | 1.569 | 2.317 |

这些数值是原论文的实验，不是本项目成绩。原文把 Case 1 用作展示全序列信息影响的负面对照，而非认可它代表实时可发行预测；同时写明，除 Case 1 外，构建模型时未取得 test。Case 2、Case 3 实际比 Case 4 差，训练联合分解不应被预设为必然带来提升。Table 5 中单 GRU 的 RMSE 为 4.772。[S1]

### 1.2 本项目采用的可复现改编

保留既有条件响应机制，将价格变化历史扩展为四路：原始变化及三个 EWT 分析分量的变化。全部分量保留；各路共享两种历史响应速率和原有上下文门。用原始价格冲击与原始 volume context 决定门，用原始官方价格变化作为预测目标。分析分量不是额外市场数据，也不自动具有“真实基础价格”的解释。

对应的共享读出为

\[
m_{t,h}=\mu h+\sum_{c=0}^{3}\sum_{k=1}^{2}
b_{c,k}v_{c,k,t}\sum_{j=1}^{h}\lambda_{k,t}^{j}.
\]

这是本项目的四路条件响应改编：原文使用 GRU，我们使用共享速率/门与精确 Ridge 读出；不是作者 GRU 的数值复现。64 个历史变化需要 65 个价格水平，窗口与差分端点须在实现中固定，不能无意缩短历史。EWT 的分析输出、边界方法和软件版本也属于可复现协议。

| 阶段 f 设置 | 表征构造 | 可诚实报告的信息范围 |
|---|---|---|
| raw | 原始输入 | 原始历史预测，作为增强前参考 |
| window_past | 每个原点只处理截至该原点的历史窗口 | 逐原点历史窗口表征 |
| train_joint_test_window | selection 用训练前缀整体构造训练表征，validation 用窗口；final refit 用 pretest 前缀整体构造，test 用窗口 | 拟合前缀联合表征、发行阶段窗口表征；训练样本内允许使用该前缀中较晚价格 |
| full_series_joint | 全序列共同决定各历史表征 | 包含评价期信息的全样本联合分解实验；不标作实时预测 |

我们的第三行按拟合阶段切换前缀，不是原文 Case 2/3 中任一个设置的逐字复现。所有设置都可以实际报告，并把成绩与取得的信息范围对应起来。完整联合设置的改善不能自动转写成 window_past 下的改善。

### 1.3 参数容量与匹配控制

| 模型 | 原始输入版本：非线性+线性=合计 | 四路 EWT 版本：非线性+线性=合计 |
|---|---:|---:|
| signed_context_response | 6+3=9 | 6+9=15 |
| static_response | 2+3=5 | 2+9=11 |
| amplitude_context | 6+3=9 | 6+9=15 |
| signed_shock_only / no-volume | 4+3=7 | 4+9=13 |

EWT 主模型没有继续保持 9 参数。四路状态带来的额外六个线性系数和 EWT 表征本身都应披露。匹配 Ridge 使用四路末 20 个变化，共 80 个输入特征；80 是特征维数，其总拟合系数还取决于截距和多期限读出实现，不能直接与主模型的 15 个拟合标量等同。

主模型、static、amplitude、no-volume 和 Ridge 应获得相同的 EWT 信息设置。原始 9 参数主模型回答“增强整体是否有效”；同信息的四路 static/amplitude/no-volume 回答“条件持续度门是否仍有独立增量”。这些是本项目归因设计，不是文献承诺的结果。

## 2. 成熟标签后更新读出：expanding、rolling 与近期权重

### 2.1 原文证据

[S2] Bacher 等，Onlineforecast: An R Package for Adaptive and Recursive Forecasting，The R Journal 15(1):173–194，DOI 10.32614/RJ-2023-031。卷期标注 March 2023；网页发表日期为 2023-09-07。正式 PDF 的 Regression 部分和附录 pp.192–193、Eq.40–44，用刚观测到的 y_t 配对此前 t-k 原点的输入更新回归系数；R_t=lambda R_(t-1)+xx'。lambda=1 对过去等权，lambda<1 提高近期权重。它允许非线性输入变换加可变线性系数，但原文的系数按预测期限分别估计。[S2]

[S2a] 作者 CRAN vignette 的 Setting up a model / Tune the parameters 明确使用 RLS forgetting factor，示例范围下界 0.9、初值 0.99、上界 0.9999，示例优化结果 0.985。这是热负荷演示值，不是碳价最优参数。[S2a]

[S3] Bastianin 等的 2026 作者全文 §3.2，使用 expanding window，每发行一次预测便增加一个新观察；不是把第一段训练参数永久冻结。该碳价实例支持域内顺序更新，但没有提供本项目的“三个共享系数、H20 统一成熟、半衰期 126 行”这套具体配置。[S3]

### 2.2 本项目改编与限制

本阶段读出更新作为独立 raw block：固定阶段 e 已拟合的非线性参数，只更新三个线性系数，不将其自动与 EWT 或 dense block 合并。validation 使用原训练阶段的选择 checkpoint，test 使用原 final-refit checkpoint，不能以 final 参数回看 validation 选择更新策略。

在观察当前原点 t 的官方行情后发行预测，可加入的完整标签 origin 为 o+20<=t。三个期限同时成熟，最新 H1 标签不能冒充已经成熟的 H20 标签。以下策略的差异应在同一原点集内评价：

- frozen：保留原读出；
- expanding_equal：所有已成熟历史等权；
- expanding_half_life126：已成熟样本按原点年龄使用 2^(-age/126) 权重；
- rolling：仅保留预先声明的最近成熟原点。

126 的单位为目标市场观测记录，不是日历日；它是项目预先指定的实验配置，不是从 [S2] 直接复刻的值。Ridge 沿用原 alpha 与初始训练的目标尺度，对权重归一化后求平均损失，并按同一成熟样本权重计算列 RMS。这样“更新样本”与“改变正则的计量单位”不会被隐含混合。新增线性 solve 数须报告，但不伪造独立 seed 重复或新增非线性训练。

## 3. 密集未来路径监督：同一 9 参数预测式，多训练期限

[S4] Zeng 等，Are Transformers Effective for Time Series Forecasting?，AAAI 2023，37(9):11121–11128，DOI 10.1609/aaai.v37i9.26317，官方发表日期 2023-06-26。已读作者 LTSF-Linear 的 exp/exp_main.py：_select_criterion 使用 MSELoss；Linear/DLinear 分支只以历史 batch_x 调用模型，对完整 pred_len 的输出与标签计算 MSE。它直接支持“整个未来预测区间作为训练监督”这一标准做法。[S4]

本项目 raw dense block 保持主模型 6 个非线性参数和 3 个线性系数。训练坐标从 H={1,5,20} 扩为 H=1,...,20，validation 与最终主要评价仍为 H={1,5,20}。所有未来值只进入已成熟训练标签，不进入发行输入；仍用原 H20 成熟边界。

训练损失按 origin 和 20 个期限取均值，各期限使用初始训练集二阶矩尺度；原 H1/5/20 的尺度逐值沿用既有值。精确线性读出与非线性优化继续采用原有 VP 目标。该改动可能更充分地约束同一响应曲线，但并不增加独立 origin 数，也不保证三个评价期限都改善。

准确名称为“密集多步路径监督”。这里没有教师网络或蒸馏损失，不称 TimeKD 复现，不声称提出新的未来监督理论。原始三期限监督与二十期限监督应采用相同最大训练预算，并保留原数据、尺度和评价口径；观察到的差值才是这项训练增强的证据。

## 4. 一个允许具体信息折衷并检验其影响的碳价实例

[S3] 的 §3.1 使用 Chow–Lin 方法，将年度核证排放结合工业生产指标分解为月度序列。作者明确讨论了由此带来的信息时点问题；§4.5 / Supplement S.4.6 删除插值排放变量进行敏感性检验，随后仍保留它用于情景分析。Table S.9 的 BVAR(1) 相对 RW 的 RMSFE，含/不含排放分别为 h1：1.0505/1.0498，h12：0.9372/0.9389。[S3]

这是具体、可复查的文献实践：保留一个有用途但非逐时点 vintage 的信息设置，同时展示删除变量后的成绩。它没有证明本项目的完整测试序列 EWT 与该插值设置等价，也不能概括为“多数论文认可前视信息”。阶段 f 应采用同样务实的表达：把方法、信息范围、匹配对照和实测增量共同交代清楚，而不是只用一个“泄露/不泄露”标签代替模型实验。

## 5. 引用、访问范围与成果边界

本文的一手证据包括完整方法正文、作者版本和作者代码的明确片段；没有把仅可见摘要的其他论文补成完整协议，也没有以少量定向样本估计文献中某做法的占比。论文中的数值仅作协议例证，不进入本项目的模型选择。

- [S1] 官方入口：https://www.mdpi.com/1996-1073/17/17/4358
  全文阅读入口：https://www.researchgate.net/publication/383685619_Carbon_Price_Forecasting_Using_Optimized_Sliding_Window_Empirical_Wavelet_Transform_and_Gated_Recurrent_Unit_Network_to_Mitigate_Data_Leakage
- [S2] 正式论文：https://journal.r-project.org/articles/RJ-2023-031/
  PDF：https://journal.r-project.org/articles/RJ-2023-031/RJ-2023-031.pdf
- [S2a] 作者演示代码：https://www.stat.ethz.ch/CRAN/web/packages/onlineforecast/vignettes/setup-and-use-model.html
- [S3] 已读取作者版本：https://arxiv.org/html/2402.04828v3
  对应期刊 DOI：https://doi.org/10.1016/j.jedc.2026.105435 。本次技术核验依据 2026-10-03 的 v3，不把尚未来到的期刊封面月份当成在线公开日期。
- [S4] 官方出版：https://ojs.aaai.org/index.php/AAAI/article/view/26317
  作者代码：https://github.com/cure-lab/LTSF-Linear/blob/main/exp/exp_main.py

机器可读的标题、日期、阅读范围、原文协议与改编边界见同目录 SOURCES.json。本文不修改已冻结的阶段 d/e 方法、实验或结论。

