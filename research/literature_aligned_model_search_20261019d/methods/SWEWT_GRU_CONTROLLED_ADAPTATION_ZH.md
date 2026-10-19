# SWEWT–GRU 文献结构的多步受控改编

日期：2026-10-19。此处新增两项文献结构强对照，**不是提出新的创新候选，也不是原论文的数值复现**。该决定发生在已经读取至少北京 v3 部分成绩之后，不属于 v3 原先注册的基线集合。新对照单独冻结、输出和披露，不回写已有 v1／v2／v3 模型、协议或结果，不新增事后 p 值。

独立运行器：[run_swewt_gru_comparison.py](../source/run_swewt_gru_comparison.py)。实现资格检查：[SWEWT_GRU_SYNTHETIC_CHECK.json](../audits/SWEWT_GRU_SYNTHETIC_CHECK.json)；脚本：[check_swewt_gru_comparison.py](../audits/check_swewt_gru_comparison.py)。截至本说明完成，只有合成前向／时点检查；本实现者没有执行市场拟合。

## 1. 文献对应与改编范围

参照 Energies 2024 论文 *Carbon Price Forecasting Using Optimized Sliding Window Empirical Wavelet Transform and Gated Recurrent Unit Network to Mitigate Data Leakage*，DOI [10.3390/en17174358](https://doi.org/10.3390/en17174358)。[官方入口](https://www.mdpi.com/1996-1073/17/17/4358)；[明确由 MDPI 提供的可读全文](https://www.researchgate.net/publication/383685619_Carbon_Price_Forecasting_Using_Optimized_Sliding_Window_Empirical_Wavelet_Transform_and_Gated_Recurrent_Unit_Network_to_Mitigate_Data_Leakage)。本轮文献审查已读其方法及实验原文，阅读范围详见 [发表流程审查](../literature/PUBLISHED_PRACTICE_AUDIT_ZH.md)。

原文的直接对应结构是：每个起点只分解最近 \(w\) 个已观察价格，保留末 \(l=17\) 个原价与 EWT 分量序列，交给 GRU 做预测。原文按验证作 TPE 调参，窗口搜索 24–64、分量数 2–4；采用单步任务与时序 70/10/20 划分。原文最大 epoch1000、patience100、TPE runs10，不把 runs10 解释成已知的十个独立训练种子。GRU32 是本次共同预算的明确指定，未声称它等于原论文最终层宽。

| 项目 | 本次受控改编 |
|---|---|
| 市场与目标 | 既定北京、天津、重庆 ScienceDB 日交易均价；沿用 v3 数据资格和原 60/20/20 行边界 |
| EWT 窗口／分量 | 固定 \(w=64,N=3\)，不再用已暴露结果搜索 TPE 配置 |
| GRU | 一层、32 隐单元，保留末 17 个序列 token |
| 输出与损失 | 20 行零初始化线性头，只有 H1／5／20 参与相同原价风险损失 |
| 特征尺度 | 当前起点中心化的原价／低频分量，加原训练前缀固定 RMS；不沿用论文 train-minmax 方案 |
| 训练预算 | 原共同 lr0.003／0.01、max80、min15、patience12、batch128、同 AdamW／clip5 |
| 选择与 refit | 同既往验证选择 lr／epoch（含 epoch0），种子 17／29／43 按选定 epoch 重训全部已成熟 pretest 标签 |
| 外生版本 | 一臂只用原价与分量；另一臂增加已允许能源、量异常度与有效掩码，属于额外同来源信息控制 |

因此结果应称“文献结构的多步受控改编”，不能写成“成功复现论文报出的误差”或把统一预算下的较差成绩归为原作者方法无效。

## 2. 完整特征时点和模型接口

对起点 \(o\)，先仅将

\[
(P_{o-63},\ldots,P_o)
\]

这 64 个价格水平传给 EWT。沿用已记录的 `ewtpy 0.2` 与纯函数 `literature_diagnostic.decompose`：\(N=3\)、`log=0`、`detect="locmax"`、`completion=1`、`reg="average"`、`lengthFilter=10`、`sigmaFilter=5`。这里不需要训练 EWT 参数，也不向 EWT 提供整个序列。特征从这次起点自己的分量中截取末 17 点，不使用发行起点 \(o\) 之后的观察。窗内较早位置的分量可以依赖同窗后部已经观察到的价格；这里的信息边界是最终发行起点 \(o\)，不能将这种合法的窗内表示构造一概称为未来泄露。

对于这 17 个历史位置，四个价格通道为

\[
\left(
\frac{P_t-P_o}{s_P},
\frac{E^{(o)}_{t,1}-P_o}{s_P},
\frac{E^{(o)}_{t,2}}{s_P},
\frac{E^{(o)}_{t,3}}{s_P}
\right),
\qquad t=o-16,\ldots,o.
\]

\(s_P\) 是 v3 原训练前缀的固定目标 RMS；上标 \(o\) 表示分量来自该起点的独立过去窗口。原价和低频分量减去的是发行时已观察的 \(P_o\)，不是任何未来价格；其余频段保留分量值。这里没有以各分量的单独验证表现调整尺度，也没有把 EWT 分量直接相加当作必须精确重建原价的条件。

两个预定家族为：

| 家族 | 实际进入 GRU 的序列 | 名义参数数 | H1／5／20 损失理论可更新的参数数 |
|---|---|---:|---:|
| `swewt_gru_price` | 17×4：原价＋三个 EWT 分量 | 4,308 | 3,747 |
| `swewt_gru_full` | 17×9：上述四个＋三能源＋量异常度＋有效掩码 | 4,788 | 4,227 |

为复用共同训练器，数据准备统一返回 17×9；price 模型在无参数输入视图中只取前四列，GRU 本身只有四列输入权重，不保留零填外生的无效权重。full 的五个额外通道直接取原 v3 同起点最后 17 个 token，保持原能源尺度、14 日滞后对齐、成交量异常度定义和 mask。

两臂的接口都是 `forward(x) -> (B,20)`。头部零初始化，初始预测变化为 0，等价于 RW。其余 17 个未受 H1／5／20 损失监督的输出行共有 561 个名义参数，初始为 0；不把这些行描述为已经训练的完整多步预测。表中的理论可更新数量也不是对每个市场实测所有梯度非零的断言。

full 与原 v3 强控制使用相同信息来源和时点规则，但辅助输入序列从 64 点变成 17 点；量异常度仍可借助其此前 60 条记录作为基准，底层量历史最多覆盖 77 条目标记录。价格 EWT 使用 64 个水平。这些长度差是文献结构改编的一部分，不能把它说成与原 64×6 GRU 逐 token 完全相同的输入。现有两臂都含 EWT，也不单独识别 EWT 相对 raw17 GRU 的因果贡献。

## 3. 数据与选择继承

`prepare_swewt` 直接调用冻结的 `run_new_market_replication.prepare_replication`，保留其所有 `origins`、`y`、`masks`、`scale` 和 `variance`，只改特征张量并追加方法说明。正式新运行冻结时，将完整原 manifest 与原 v3 文件逐项比较，并核对五份 v1–v3 可执行源码 SHA。新源文件与 EWT helper 另外加入本次源码冻结集合。

训练直接复用原 `common.train_fit` 和 `common.neural_job`。因此标签成熟、损失单位、early stopping、lr／epoch 选择、三种子最终训练、原预测文件格式均保持同一个实现。两臂×三市场×（两次 lr 选择＋三次 refit）计划 **30 个神经任务／构造**；若选择 epoch0，应从收据另算真正更新的 refit 数，不能称 30 次全部完成了梯度学习。

北京、天津沿用 v3 的主市场角色，重庆保持小样本压力测试角色。作者均价序列已有按月参考值进行异常筛查／替换的来源限制，真实历史 vintage 和处理标记不全；本次不会因为改用滚动 EWT 就消除原数据自身的限制。

## 4. 已完成的最小检查

合成检查无任何 optimizer 更新、市场准备或性能评分。它包含有曲率的确定合成价格、独立能源／量序列，以及恒定／全零窗口：

| 检查 | 结果 |
|---|---|
| 4 个起点的特征／预测形状 | 4×17×9；两臂均 4×20 |
| 只扰动 cutoff 后价格和辅助特征 | cutoff 以前所有起点的特征差为 0；后续起点产生非零变化 |
| 原价当前起点中心化 | 末端原价通道精确为 0；浮点缩放最大误差 5.37×10⁻⁸ |
| 外生通道 | 与原输入最后 17 个能源／量 token 逐值相同 |
| 恒定／全零窗口 | EWT 特征有限；最大中心化绝对值分别约 8.36×10⁻¹⁵／0 |
| 初始模型 | 两臂所有输出精确为 0 |
| 有意义的输入隔离检查 | 给头部指定固定非零合成权重后，price 不受额外通道扰动，full 受其影响；未据标签拟合 |
| 数据 adapter | 合成 fixture 的起点、标签、划分、尺度与风险分母逐项不变 |
| 旧冻结文件 | v1–v3 五个源码 SHA 全部保持原值 |

当前新 runner SHA256：`0b3395f547fad792c6880a9d25819c6e326ab48ef733240ad690e4833b63218d`。`ewtpy` 的实际源文件 SHA256 为 `35cb4532ae35a41867ee1411cb61a33165f7eebab3ebf70a58b47c74c6e6a289`；新运行协议会同时记录实际 Python、NumPy、torch 与 EWT 包版本。资格检查只说明时点和接口按设计工作，不产生模型有效性结论。

另有未参与此源文件实现的文献审查者完成只读接口／信息时点快审，未发现阻塞点；该审查者未重复拟合或进行市场 preparation。其核对包括过去窗口切片、原点辅助序列对齐、共同标签／尺度／划分继承和 4／9 通道参数数。原价／低频减去 \(P_o\) 是本次明示的数值尺度处理，不被表述为自适应 EWT 边界在任意平移下严格不变的理论保证。

## 5. 主执行者冻结后的启动方式

以下为启动接口说明，本实现者未执行该命令。输出目录必须为空，既有结果不可覆盖。

```bash
PYTHONPATH=/tmp/carbon-forecast-20261019d-deps \
/tmp/carbon-research-env/bin/python \
research/literature_aligned_model_search_20261019d/source/run_swewt_gru_comparison.py \
  --out /tmp/carbon-literature-aligned-runs-20261019d/swewt_gru_control \
  --reference-v3-run /tmp/carbon-literature-aligned-runs-20261019d/new_market_v3 \
  --workers 4
```

工作目录为仓库根目录。默认处理三个既定市场；也可显式传 `--markets` 选择既定市场子集，或用 `--data-root` 显式指定原 v3 manifest 记录的数据目录；来源目录与 SHA 一同参与一致性核对。旧 v3 控制不在此脚本中重新拟合。完成后独立结果目录包含事前新协议、输入 manifest、所有选择／refit 记录、checkpoint、两臂预测、汇总和源码／EWT 身份核验；后续只按已经指定的两臂分别报告，不以测试结果挑选其中一臂并隐去另一臂。
