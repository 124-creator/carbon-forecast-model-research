# 小样本分段 GRU 多步输出：数学缺陷、一个接口设计及否证门

日期：2026-10-14，中国标准时间。范围：独立数学分析与约十分钟主源查新；没有读取市场数据，没有修改另一子任务的骨干或实验文件，没有提交。

## 结论先行

只提出一个可接入机制：**共同已知信息上的同目标预测修订矩约束**。它改变训练目标，不添加辅助头、注意力、步长权重或低秩参数；推理仍然是普通 SegRNN。但本轮**不把它保留为已成立的模型创新**：数学主体是已有预测合理性条件与矩估计；分段边界只提供诊断分层，未提供 SegRNN 专属的新识别定理。因此当前状态是 `REJECT_AS_MODEL_NOVELTY / DESIGN_FOR_FALSIFICATION_ONLY`，不是“候选成功”。

明确的近邻不只是 SegRNN / NHITS：Wen 等的 forking-sequences 已经跨预测起点训练；Potosnak 等的 TMLR 论文进一步研究修订波动；Ma 等已有兼顾准确性和跨起点一致性的可微目标。本方案与它们的区别是“不惩罚所有修订，只惩罚可由双方共同旧信息预测的修订”，但这还不足以自动产生模型贡献。[8][9][10]

## 1. 先纠正“不同 h 信息成熟”的含义

固定起点 t 时，全部 f(t,h) 使用同一观测窗口 I_t，不能说 h 大就真的看到了更多未来信息。不同 h 的目标不同，误差方差也不必随 h 单调。小样本或非平稳过程下，更不能把总体误差单调性直接当作逐样本约束。

可以识别的信息成熟是**同一目标 τ 在不同起点的更新**：f(t,h) 与 f(t-1,h+1) 都预测 y_(t+h)，新起点实际增加了 y_t，也可能丢弃了窗口最老的 y_(t-L)。预测应允许依据新增信息变化，不能直接强迫两者相等。

普通 SegRNN 将历史分段投影后送入 GRU；PMF 将最终编码状态复制给各输出段，配合各段位置嵌入进行独立的一步解码，最后共享线性段输出。它不含前一预测段到下一预测段的递归边，而 RMF 才有这一边。[1][2]

这意味着“PMF 未显式施加相邻预测段的联合结构”，**不意味着“PMF 无法建模跨段依赖”**：共享编码状态、解码参数和真实多步标签仍然可以学习联合规律。不能只看计算图就宣布实际缺陷。

## 2. 分段边界到底可能哪里出了问题

设段长 s，段号 k(h)=floor((h-1)/s)，段内位置 r(h)=(h-1) mod s。旧起点 h+1 与新起点 h 的同目标配对中，h mod s=0 时会跨越输出段嵌入；其他位置只移动段内坐标。输入也会因起点前移而重新分段：这使“新增真实观察”和“分段坐标变化”混在一起。

因此可检验缺陷是：在共同旧信息、预测步长和目标尺度得到控制后，**跨段配对的可预测修订**是否比段内配对更大；不是“边界跳跃越小越好”。价格真实跳变、目标变化或新增观察都不能算模型错误。

最低诊断：
- 对齐同一 τ，计算 Δ_(t,h)=f(t,h)-f(t-1,h+1)。
- 共同信息 C_t 来自两窗口交集 [t-L+1,t-1]；不能使用新观测 y_t。
- 预先固定 φ(C_t)，检测 E[φ Δ]，并报告跨段 / 段内结果。
- 跨段标签与 h 的周期位置完全绑定；单一 s 下的“边界效应”可能只是步长效应，无法独立识别。需要预设多个 s，或同容量逐点 GRU / 非分段输出作为控制；只作为诊断，不额外添加第二个机制。
- 未发现可重复、超出步长效应的可预测修订，就没有分段专属缺陷，立即停止。

## 3. 唯一设计：共同信息修订矩约束

### 数学目标

I_old=σ(y_(t-L),...,y_(t-1))，I_new=σ(y_(t-L+1),...,y_t)。两者并不嵌套，**不能直接套用完整历史滤过的塔性质**。

令 C=σ(y_(t-L+1),...,y_(t-1))，两者都包含 C。若最优输出分别是 E[Y|I_new] 和 E[Y|I_old]，对于平方可积、C 可测的固定 φ，有

E[φ(C)(E[Y|I_new]-E[Y|I_old])]
= E[φ(C)Y]-E[φ(C)Y]=0。

这个推导无需滑动窗口嵌套。但它只是一个必要条件：有限个 φ 不能识别任意神经条件均值，任意两输出共有的常数偏差都会在差分里消失。监督多步 MSE 必须保留。多步预测最优性与预测修订的统计限制已有直接文献；这里的交集窗口论证是本设计的自行推导，不应冒称论文中的专属定理。[5]

预设 φ=(1, z_last_common, z_common_mean, z_common_slope)，其中三个统计量只在训练集确定尺度，以双方共同观察计算。φ 不是因果工具变量，不作 IV 识别宣称。若 L 太短导致统计量不存在，直接停用，不临时补特征。

对训练起点集合 B 和 h=1,...,H-1：

m_(h,j)(θ)=mean_(t in B)[φ_j(C_t) (fθ(t,h)-fθ(t-1,h+1)) / S]。

J(θ)=mean_supervised[((fθ-y)/S)^2] + λ mean_(h,j)[m_(h,j)(θ)^2]。

S 是仅训练集确定的全局价格尺度；如果骨干输出经过按窗口归一化，先分别反归一化到真实价格，再使用同一 S。否则不同窗口基准会伪造“修订”。λ 只用时间验证选择，候选包含 0；不设置边界额外权重，不设置单步偏好。

这是预测输出矩约束，不是参数 ridge。它跨起点耦合输出，具有非零混合二阶项，不能写成仅逐 h 改权的可分 MSE。但数学上仍然属于传统矩条件训练，不能靠换名称升级成模型创新。[6][7]

### 实際骨干接法

训练数据提供相邻两个起点的历史窗口；两次普通 backbone.forward 输出形状均为 [B,H,C]。旧输出取 `[:,1:H,:]`，新输出取 `[:,:H-1,:]`，使目标时间完全相同。

```text
old = denorm(backbone(x_old))
new = denorm(backbone(x_new))
revision = (new[:, :H-1, :] - old[:, 1:H, :]) / train_scale
moments = mean_over_origins(revision[..., None] * fixed_common_phi)
loss = supervised_multistep_mse + lambda * mean(moments ** 2)
loss.backward()  # 两侧都传播，不把旧预测当作真值，不增加教师/学生支线
```

训练有效起点必须让 old 和 new 的全部监督目标都位于训练区间；验证只能在验证起点生成配对，不能把跨切分配对混入训练。该损失只在训练使用未来标签，推理时没有未来观察，仍然单窗口调用骨干。

同一批次必须同步两个起点，φ 不参与梯度；骨干的 dropout 应关闭或确保诊断采用确定性评估，避免把随机丢弃造成的修订当作结构现象。推理计算量与普通 SegRNN 相同；训练约增加一次窗口前向及对应反传，不能称“免费”。

**重要小样本局限**：`(batch mean)^2` 的期望包含批内矩估计方差，不等于纯总体矩平方；小 batch 下会间接压缩合法修订。建议采用一个连续训练块形成矩，跨起点聚合后统一反传，并报告块长敏感性；如果使用跨两个独立块的无偏交叉积估计，将改变实现且可能产生负目标，本轮不再添加该第二变体。重叠窗口高度相关，不能把配对数量当独立样本量。

## 4. 与近邻的实际区别和创新审计

| 主源 | 已有内容 | 本设计差异 | 判定 |
|---|---|---|---|
| SegRNN / 作者 `models/SegRNN.py` | 分段编码、PMF、位置嵌入、共享段输出 | 完全不改结构，新增跨起点矩损失 | 骨干没有新意 [1][2] |
| NHITS / 作者 `nhits.py` | 多速率输入池化、多层级插值输出、残差叠加 | 不插值、不压缩输出自由度 | 不能把平滑/多尺度叫本设计贡献 [3][4] |
| Patton & Timmermann | 多步最优预测限制、修订合理性检验 | 将共同窗口矩用作训练约束 | 已知条件的训练用途 [5] |
| DeepGMM | 用神经网络估计满足矩条件的函数 | 无对抗网络、不作 IV、固定少量测试函数 | 一般矩训练已存在 [6][7] |
| Wen 等 MQ-RNN | forking-sequences，跨多个起点共享训练 | 普通配对前向，不复用完整序列编码 | 多起点训练不是新意 [8] |
| Potosnak 等 TMLR 7/2026 | 多步预测修订、波动指标、forking-sequences/集成 | 不集成，只检查共同旧信息可预测部分 | 更强的直接近邻，必须比较 [9] |
| Ma 等 2026 v3 | 准确性与跨起点稳定性的可微联合目标 | 不最小化所有修订的大小，只约束可预测投影 | 有概念区别，尚无分段专属贡献 [10] |

**新意不是“SegRNN + 一致性损失”**。当前也没有证明固定四个 φ 的矩约束为什么优于已有预测合理性训练、为什么专属于分段小样本 GRU，或为何改 h / s 就会产生可识别的新结构。它目前只是受任务启发的组合，不达到用户要求的模型创新；不要先训练出一个指标提升再反过来包装理论。

## 5. 否证门：按顺序执行，失败不绕

G0 数学：同目标索引、交集可测、反归一化正确；已提供有限状态代数检查。若任一不满足，停止。

G1 缺陷存在：只在训练/验证诊断普通 SegRNN。固定仪器和 h，采用保留时间相关性的块重采样并控制多重比较。没有可靠共同信息可预测修订，或边界信号在匹配 h 后消失，停止，不换市场找显著性。

G2 创新：与预测合理性/GMM、AC 稳定损失、forking-sequences 分别对照；需要一个无法由普通矩正则与配对采样解释的分段特有结论。**本轮未通过，因而停止正式候选晋级。** 下列 G3-G5 是接口设计的预注册否证标准，不是推荐继续做市场实验。

G3 必要公平对照：同骨干 λ=0；相同双起点采样但没有矩项；同容量非分段输出；NHITS；原始修订平方惩罚；forking-sequences 可复现版本。总优化步数、数据起点和容量可比。只比 λ=0 不能证明机制。

G4 预先固定模拟：a) 新信息可改变同目标均值，合法非零修订应保留；b) 人为输出段位置偏差，机制应减少其可预测部分；c) 真实跳变，不得因为追求稳定而延迟反应；d) 共同偏差，明确矩无法检测；e) 改变 s，边界诊断应随人为段结构移动而不是固定 h。a/c 失败则停止，即使平均 MSE 下降。

G5 小样本价值：须降低可预测修订且改善或至少不损害预注册主要预测指标，不能只降低原始修订方差。报告所有 h、种子、连续块、λ=0 和块长敏感性；若 λ=0 被验证选中、改采样已解释收益、或结果只存在于某单个 h，撤销机制，不扩大超参搜索。

## 6. 收据、实际验证及访问失败

`source_receipts.json` 保留真实 URL、HTTP 成功/失败、获取 UTC 时间、文件字节和 SHA256；下载 PDF、PDF 文本、作者代码都在 `sources/`。SegRNN 固定提交 `8e869ecfdf1daab3a0ba14d1d620796c1a5d2c4f`；NHITS 作者仓库固定提交 `dd2c92d42ea40237743cc58e92603bb8b3838fd2`；DeepGMM 固定提交 `c2a62ed885792145849de98d6cf8fe81cd4185ee`。[2][4][7]

AC 论文指向作者仓库，首次假定 main 返回 404；查询实际默认分支为 master 后得到真实树收据，树 SHA 为 `1d0663d9b2522fbdbc78553ce2e71519c1b25f30`。不能把论文中的“code available”单独当作代码已复现。[10][11]

作者代码确认级别：SegRNN 与 NHITS 核心实现完整读取；DeepGMM 作者 README 和目录身份确认，并未运行原实验；AC 已读取固定树版本的 `forecast_metric_utility.py`：它实际对齐 `[:-1,1:]` 和 `[1:,:-1]`，并将 accuracy 与 mult_stability*stability 相加。注意该文件计算了 sharpness，却在返回 score/distance 前注释掉相应减项；因此当前代码与论文完整 energy 定义不能直接视为一致，原实验尚未复现。[12]；forking-sequences 新论文作者博客明确参考实现非论文使用、非 Amazon 官方实现，因此本轮不把参考代码误列为实验作者代码收据。Patton/Timmermann 论文全文已读相应预测修订条件，未找到并验证作者实验代码，明确缺失。

OpenReview PDF 首次 403，改取真实 arXiv v5 成功；若干 raw GitHub 请求 SSL EOF，改用固定提交 GitHub Contents API 成功，所有初次失败均保留，不删除。不把仅正文可读等同于实证协议已审核。

`math_probe.py` 已实际运行，`math_probe_results.json` 为 7/7：同目标索引、非零新信息修订允许、旧信息可预测修订检测、共有偏差不可识别、原始修订惩罚误伤、梯度有限差分、跨起点混合二阶项。这些只是有限状态代数验证，不是市场表现，不是骨干集成训练，也不证明创新。

## Sources

[1] https://arxiv.org/pdf/2308.11200 — segrnn_paper
[2] https://api.github.com/repos/lss-1138/SegRNN/contents/models/SegRNN.py?ref=8e869ecfdf1daab3a0ba14d1d620796c1a5d2c4f — segrnn_model_pinned
[3] https://ojs.aaai.org/index.php/AAAI/article/download/25854/25626 — nhits_paper
[4] https://api.github.com/repos/Nixtla/neuralforecast/contents/neuralforecast/models/nhits.py?ref=dd2c92d42ea40237743cc58e92603bb8b3838fd2 — nhits_author_model_pinned
[5] https://public.econ.duke.edu/~ap172/Patton_Timmermann_bounds_JBES_2012.pdf — revision_paper
[6] https://arxiv.org/pdf/1905.12495 — deepgmm_paper
[7] https://api.github.com/repos/CausalML/DeepGMM/contents/README.md?ref=c2a62ed885792145849de98d6cf8fe81cd4185ee — deepgmm_readme_pinned
[8] https://arxiv.org/pdf/1711.11053v2 — forking_original_paper
[9] https://arxiv.org/pdf/2510.04487v5 — forking_current_paper
[10] https://arxiv.org/pdf/2601.10863v3 — stability_ac_paper
[11] https://api.github.com/repos/causify-ai/beyond_accuracy/git/trees/master?recursive=1 — stability_ac_author_code_master
[12] https://api.github.com/repos/causify-ai/beyond_accuracy/contents/forecast_metric_utility.py?ref=1d0663d9b2522fbdbc78553ce2e71519c1b25f30 — stability_ac_metric_pinned
