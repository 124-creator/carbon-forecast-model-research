# SOLID 备选：来源、实际执行接口和机制边界

2026-10-17。本目录记录既定备用方案 SOLID 的作者源码审查与因果适配，不增加第四个完整方案。RRC 已按原开发门槛停止；本目录的工作不改变该决定，也不打开 EEX 2024/2025 确认数据。

**本轮决定是完成已发表 SOLID 的有限适配性参照，不为尚未建立的差异再命名一个新模型。** 输入相似不保证未来条件均值映射相同、残差分布变化不保证均值值得修正、重叠窗口不等于独立信息，这些是需要说明的限制；普通残差均值检测、ridge、距离重加权或历史效用记忆不能据此成为新的学习原理。

## 1. 直接来源与核查程度

论文为 Chen、Shen、Fu、Li、Sun、Liu 的 *Calibration of Time-Series Forecasting: Detecting and Adapting Context-Driven Distribution Shift*，KDD 2024，DOI [10.1145/3637528.3671926](https://doi.org/10.1145/3637528.3671926)。本轮阅读[作者全文](https://arxiv.org/html/2310.14838v2) §5.1–5.3、Algorithm 1–2、§6 的假设和主要结论、B.3–B.4；不能把前一阶段仅摘要核查的状态和本轮已读方法混为一谈。

论文直接链接的[作者仓库](https://github.com/HALF111/calibration_CDS)固定为：

~~~text
repository: HALF111/calibration_CDS
commit: 3c74e5f8b3c921a2e7cda30f36117c295bf8c112
commit date: 2025-06-19
tree: 5861b99937ea1b26be2c44ad845a6f9c9624161f
~~~

完整树未截断。仓库声明 MIT；已读取完整 LICENSE，其原始版权行为 “Copyright (c) 2021 DAMO Academy @ Alibaba”。分发所用代码时保留该原始许可证和版权文字，不自行改写为本研究所有。

本轮下载 11 个小型来源文件，共 174494 字节，逐个验证 UTF-8 字节的 Git blob SHA。详细路径、下载 URL、Git blob SHA 和 SHA-256 见 SOURCE_MANIFEST.json。SOURCE_PAYLOAD.json 保存来源响应的必要文件内容和树，用于取得源码的取证；它不是模型结果。

### 实际运行必需的作者文件只有两个

| 文件 | Git blob SHA | SHA-256 |
|---|---|---|
| author_solid/exp/exp_main.py | e021dfbe79885e379018168b475915f17cba2883 | d0087fac4a2dd4b9d3b5650b7d9330715be12d371f203b8935f6440eb8777d56 |
| author_solid/LICENSE | 974cdf19be0c4c70cb33715c0c9dfa5ed80c2609 | 74323c2d2119ab99a00c4d0bda4f72c4ff3e3aca49af17eaf62aec593f2af1b2 |

其余已下载的 README、配置、数据加载器和模型文件用于核查。新的适配器不依赖作者全部 benchmark 导入链，也未安装 requirements.txt 中的全部历史环境。实际调用环境在合成收据中记录为 PyTorch 2.3.1+cpu、NumPy 2.5.4、float64、单线程。

## 2. 必须保留的作者实现语义

实际调用的是固定文件中的完整 Exp_Main.select_with_distance 函数，而非根据论文自行重写一个近似函数。该函数原始 AST 的 SHA-256 为：

~~~text
4857f32830d2475b654f0e3b50fbfffd52c03b407fded33a3b8563f3ab75f0a3
~~~

适配器从已验证文件中提取这个未修改的函数 AST，提供运行所需的全局接口。没有执行作者 Exp_Main 构造器、完整 benchmark 驱动器、原数据加载器、原缓存分支、原离线训练流程或其他模型。主比较使用本研究已经选定并保存的离线 GRU32；因此应称“作者原始 SOLID 核心对固定研究骨干的实际应用适配”，不称完整作者 benchmark 复现。

已读非 AMP 路径有以下具体语义：

1. 每个 query 都从同一基础模型重新复制，局部头部状态不向下一个 query 传播。
2. 先选时间范围和周期满足条件的完整成熟历史窗口，再按输入欧氏距离选近邻；作者已有 remove_distance=true 的按时间倒序控制。
3. 对每个近邻，在同一个初始头上计算一次 MSE 梯度，分别清零梯度。最后把各近邻梯度**相加**，再手动更新一次权重与偏置。
4. 参数更新没有除以近邻数。日志中的 mean_loss/selected_data_num 不参与更新。
5. 在这个非 AMP 路径把 test_train_epochs 增大，会重复采集初始点梯度；不能把它记录为重新计算参数后的顺序多步 SGD。适配器固定为 1。
6. SHOW_LOSS_DETAILS 的作者分支会使用 query 真值计算诊断梯度。本方固定关闭该分支，并且根本不向接口提供 query 的未来真值。
7. 作者构造器按数据文件名设置周期；未知数据及金融默认路径为 1。本方固定 period=1 且 remove_cycle=false，保留该默认行为，不声称重新执行论文的 FFT 周期发现。
8. 作者 head 映射需按骨干指定。普通 DLinear 没有列在已读映射中，不能把直接实例化 DLinear 称为已验证可用路径。本方明确映射到研究 GRU 的单一 head。

## 3. 因果输入适配：完整窗口打包，不改变有效历史

本研究 CausalData.at(s) 产生原模型实际使用的 20 个增量 token（来自 21 个价格顶点），每个 token 包含价格增量、实际日期间隔，以及该系列协议允许的历史协变量。所有输入均保持原值和原顺序。

为了调用作者连续窗口接口，将一个完整输入 $x_s\in\mathbb R^{20\times C}$ 展平成一个 packet 行。作者接口层的 seq_len 设置为 1，表示一次选中一个完整 packet；它不表示把研究 GRU 的历史缩短到一个价格点。原模型 callback 将 packet 解包为原来的 $20\times C$ 输入。

作者欧氏距离因而恰好作用于 $\operatorname{vec}(x_t)$ 和 $\operatorname{vec}(x_s)$。原 PyTorch pairwise_distance 默认 epsilon 也保留。起点索引单独放在 x_mark 中，只用于查找已知输入和合法标签；索引不进入距离或模型。

对当前 query $t$、最大池容量 $N_{\max}$ 和 $H=20$，实际容量为

$$
N_t=\min\{N_{\max},\,t-H-60+1\}.
$$

候选起点为

$$
\mathcal C_t=
\{t-H-N_t+1,\ldots,t-H\}.
$$

因此每个候选都满足 $s\ge60$、$s+H\le t$。早期历史不足时缩短实际池，且同步作者的 test_train_num；若连固定邻居数都不足则明确停止，不补未来、不用负下标。receipt 逐 query 记录实际池首尾、容量、选中起点和实际标签最大索引。

标签仍是研究协议的归一化累计变化：

$$
y_{s,h}=\frac{P_{s+h}-P_s}{s_s\sqrt h}.
$$

其中 $s_s$ 为 CausalData 的起点尺度。作者的目标函数和 head 更新使用这些同口径标签；callback 的输入与输出单位属于本方应用适配，不能冒称复现作者原数据缩放。返回给调用者的预测恢复为原价水平：

$$
\widehat P_{t,h}=P_t+s_t\sqrt h\,\widehat y_{t,h}.
$$

当前 query 的未来标签使用零占位，但不会用于损失、选择或评分。原 metric 接口被关闭；真正的共同支持评分在主 runner 中执行。

## 4. 固定编码器缓存的等价范围

在每个 query 重置到同一离线模型、仅调整单一线性 head 的前提下，编码器对历史输入的输出不变。适配器可以缓存

$$
z_s=\operatorname{encoder}_{\mathrm{offline}}(x_s)_{\mathrm{last}}.
$$

callback 仍调用当前 query 的实际 head。缓存不存未来标签；每个 $z_s$ 只在该输入已知时生成。第一次缓存还核对基础模型 forward 与 head(last encoder state) 的等价性，不支持任意隐藏额外运算的自定义模型。

已完成独立缓存与不缓存对照：distance 和 recency 两种路径的逐 query 预测差和梯度差均为 0。缓存是减少重复编码器计算的等价工程，不是新算法、额外预训练或额外独立实验证据。

## 5. 可调用接口与离线状态身份

本方文件 source/solid_author_adapter.py，冻结初版 SHA-256：

~~~text
2e38d06e39928e771f8e09ed50add4f8ffb291ec2dd6918e2b7c6dc4469648ac
~~~

接口为：

~~~python
predictions, receipt = run_solid_author(
    data,
    pretrained_teacher,
    query_origins,
    author_root=None,
    context_samples=128,
    neighbors=10,
    learning_rate=0.001,
    remove_distance=False,
    cache_latents=True,
    deadline=None,
    artifact_sink=None,
)
~~~

predictions 为 n×20 的原价预测。data 实现现有 CausalData 的 at 和 batch 接口。pretrained_teacher 必须是**已选离线 teacher**，其 encoder 为单个 GRU/LSTM，head 为带偏置的 nn.Linear，权重使用 CPU float64；预期 state keys 为 encoder.*、head.weight、head.bias。

仅凭一个模型对象无法判断它来自离线检查点还是最终在线检查点，适配器不会虚构这个能力。主 runner 必须按原 D1 清单校验所加载的离线文件和 teacher 状态哈希；适配器另记录输入模型完整状态哈希，并验证每个 query 后 encoder 和基础模型不变。

artifact_sink 可接收每个 query 的选中起点、归一化与原价预测、梯度总和及实际适应头，供私有 NPZ 保存。receipt 记录实际 head 更新次数、逐邻居反传次数、已成熟标签边界、缓存计数、源码身份和状态哈希。到达 deadline 抛出 TimeoutError；不能将中途退出流作为完整候选。

## 6. 可移植验证与取得来源

默认作者目录相对于适配器文件位置解析，不依赖本工作区绝对路径。最小分发包保留：

~~~text
source/solid_author_adapter.py
source/check_solid_author_adapter.py
backup_review/author_solid/exp/exp_main.py
backup_review/author_solid/LICENSE
backup_review/SOURCE_MANIFEST.json
~~~

从研究目录执行以下命令只验证作者来源，不进行市场拟合：

~~~sh
python source/solid_author_adapter.py --verify-author
python source/check_solid_author_adapter.py
~~~

若采用已有作者 checkout，可以显式指定：

~~~sh
python source/solid_author_adapter.py --verify-author --author-root /path/to/calibration_CDS
~~~

若需要自行取得固定版本，可在一个新目录正常执行以下命令。失败就保留错误并停止，不换未经核实的来源、不切换版本、不自动重试：

~~~sh
git init solid-reference
git -C solid-reference remote add origin https://github.com/HALF111/calibration_CDS.git
git -C solid-reference fetch --depth 1 origin 3c74e5f8b3c921a2e7cda30f36117c295bf8c112
git -C solid-reference checkout --detach FETCH_HEAD
python source/solid_author_adapter.py --verify-author --author-root solid-reference
~~~

主实验重放还需要原开发数据、原 D1 离线状态及其清单。现已提供独立入口 source/reproduce_solid.py；完整参数、路径模式、双重初始化身份检查与准备/执行区分见 REPRODUCE_SOLID_ZH.md。新入口生成新协议，复制五个原运行文件的原始字节，不读取历史协议中的旧绝对数据路径。上述来源验证或纯合成运行不等于重新执行了市场适配实验。

## 7. 已完成的纯合成检查

source/check_solid_author_adapter.py 的 15 项检查均通过，收据为 SOLID_ADAPTER_SYNTHETIC.json。检查覆盖：

- distance 与 recency 的缓存/不缓存预测、梯度一致；
- 原作者求和一步更新与独立 NumPy 公式一致；
- 实际邻居反传及头部更新次数；
- 完整标签成熟边界、早期候选池裁剪；
- 每个 query 重置、未来后缀扰动不改变当前预测及梯度；
- C=5 的 GRU 和 C=2 的 LSTM 形状；
- deadline 在更新前停止、调用者离线模型保持不变。

最大差分别为：缓存/不缓存预测与梯度 0；独立公式更新不超过 $8.33\times10^{-17}$。这是合成正确性证据，市场读取数和市场拟合数均为 0。它不证明 SOLID 在真实价格上有效。

另有独立审查代理的 SOLID_THEORETICAL_REVIEW_ZH.md 与 SOLID_THEORETICAL_SYNTHETIC.json。实际适配器的独立审查也已完成：SOLID_EXECUTION_INDEPENDENT_AUDIT.json 记录 32 个纯合成 query、320 次邻居反传、513 项检查，比较 12580 个数值。独立 NumPy GRU 门方程、原 pairwise_distance epsilon、目标尺度和求和头更新的最大绝对差为 $4.45\times10^{-16}$ 以下；缓存/不缓存、query 重置与未来后缀扰动的差均为 0。独立审查的市场读取、市场拟合和确认访问均为 0。它证明该适配器在所列合成条件下与原语义一致，不是另一次市场训练或预测效果证据。

## 8. 原作者机制的具体边界

### 输入相似与未来增量映射

SOLID 的论文已明确把输入相似当作不可见 context 的代理。若过去和现在的输入分布相同，但条件均值映射发生反转，即便找到输入距离为零的历史窗口也未必有用。检索距离不能单独证明未来映射一致。已读理论使用已知分组的固定设计线性关系和独立同方差噪声；不能把这个假设直接推广到有噪价格的检索窗口。

### 完整 H20 标签延迟

完整多步标签约束使最近一批起点不能进入完整窗口池，但不能据此说这段时间完全没有新监督：每次新报价仍会使一个较旧起点的完整标签成熟，其尾端含有最新值。利用这些跨状态标签改善当前映射需要额外结构假设。简单启用部分成熟标签不是未开发空白：作者代码已有 use_nearest_data 相关分支，本次没有启用或验证该路径，也不把它改名作为创新。

### 重复窗口与实际更新幅度

固定特征 $\bar z=[z;1]$、基础预测 $f_0$ 下，作者求和一步在 query 的变化精确为

$$
f_1(q)-f_0(q)=
\frac{2\eta}{H}
\sum_{i\in\mathcal I_q}
(\bar z_q^\top\bar z_i)\{y_i-f_0(i)\}.
$$

这是核残差修正；内积可以为负，因此通常不是近邻均值或凸组合。对近邻的复制会增加此更新幅度，却没有同幅增加独立信息。固定 N 后比较距离检索与按时间倒序，可以检验输入距离是否比单纯局部更新带来增量；不能把任意步长变化解释成更好地识别了 context。

### Reconditionor 分布变化不等于均值可改进

即使两个 context 的残差条件均值均为零，只要方差不同，残差与 context 的互信息也可为正；当前 MSE 最优点预测却无需修正。Gaussian 平均 KL 还包含边际混合分布被单 Gaussian 近似的误差，不总等于严格互信息。独立报告给出了准确推导与合成反例。

这些限制支持谨慎设计同口径参照，不足以支持本轮提出新模型。把检索换成任务效用记忆、把检测换成条件均值 gate、把一步更新换成局部 ridge，均未建立超出已有学习原则的差异；本轮不推进这些方向。

## 9. 市场执行记录的边界

来源审查、适配器开发和上述合成阶段均无市场拟合。主任务随后另行冻结唯一真实计时协议，不能倒写成来源审查阶段已有的执行：

~~~text
provenance/SOLID_PILOT_PROTOCOL.json
SHA-256 a8f629ed09bbe62144b8f7b803f82ab07b24d2b9023bd697888a309f797ba05b

source/archive/run_solid_reference_pilot.py
SHA-256 4ddf705c4d8b686d32dc7b99f8e9eed3c6f24884af449d0e483314aeb40b716f
~~~

该计时使用 EEX seed 17 的 D1 已选离线 teacher、首 20 个 2022 年完整 H20 query，context=128、k=10、lr=.001；预算为 20 次真实 query 头适配、200 次邻居反传，0 次新预训练、0 次评分。计时所用原 runner 字节已保存在上述 archive 路径；当前 source/run_solid_reference.py 是正式 B1 的冻结版本，不应再用计时版本的哈希描述当前文件。

正式 B1 的协议 SHA-256 为 b44357f30ce5c7f50f2023ebb0fd3fcc17e1598038724738b0b64b56c7ca76fd，runner SHA-256 为 856f2fd24f8d5863dae533ca1bc1eeb4b1c9e981ea9675ed893b25679ec7bfea。正式执行前的等价补充说明固定了两组 recency 容量配置的复用：period=1、同一离线头、固定最近 10 个成熟起点下，候选池容量 128 与 512 给出完全相同的有序邻居和一步更新。因此 20 个等价预测系列从已执行系列精确派生，不计为额外作者流或额外适配。

results/SOLID_B1/COMPLETE.json 记录正式执行于 2026-10-17 10:49:54 UTC 完成：10 个“系列×种子”作业，**60 个实际作者适配流＋20 个精确派生系列，共 80 个预测系列、480 个评分格**；实际逐 query 头更新为 **25332 次**，新骨干预训练为 **0 次**，确认数据读取为 **false**。五个价格系列涉及四个底层市场体系，不能把 EEX 拍卖价和 EU_REPORTED 当作独立市场复制。这一段只从完成汇总读取执行计数，读取范围登记于 SOLID_COMPLETION_COUNTER_READ.json；评分、选择与原始预测的独立审核由主报告及各自结果文件说明。已有方法的适配结果不构成本研究的新模型创新证据。
