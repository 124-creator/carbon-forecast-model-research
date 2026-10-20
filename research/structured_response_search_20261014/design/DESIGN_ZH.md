# 60lag 低信号估计：有限暂态响应 + 已观察年龄证据路由

日期：2026-10-14。状态：**一个可实施的研究假设，不是已经发现的创新，也不是通过合成门或市场门的模型。**本任务只查主源、固定作者代码并设计模型；没有读取任何价格、2022 年记录或禁用边界，没有提交。

## 1. 主源和代码实际核查

下载正文、提取文本并静态核查代码，不等于安装作者项目或复现实验。以下代码均由 GitHub contents API 在明确 commit 获取，按 Git blob 规则重新计算 SHA-1，四个文件全部相符；commit 与 blob 见 receipts/acquisition.json 和 additional_acquisition.json。

- N-BEATS：正文的 basic block 分别输出 backcast/forecast 系数，generic 版本学习无约束波形；解释版本约束趋势/季节 basis。作者 models/nbeats.py 实现相关块。因此“学一个多步响应字典”本身没有新意，不能靠改名字避开这个邻居。[1][5]
- DLinear：作者代码第 15–22 行使用窗口端点扩展的移动平均，第 33–36 行分离残差/趋势，第 65–66、83–86 行分别用自由 Linear(seq_len,pred_len) 映射后相加。当前候选不使用这个分解，也不配置逐 horizon 的自由线性头。[2][6]
- Koopa：正文和作者代码同时核查。FourierFilter 第 7–22 行区分时变/时不变部分；KPLayer 第 74–100 行在 latent snapshots 上用 lstsq 估计 K 后滚动预测。局部 segment 编码 + 稳定/变化状态分开并不新；本候选没有 FFT、DMD 或自由 latent 转移矩阵。[3][7]
- S4：正文以状态空间/卷积计算为核心；作者 ssm.py 的 SSMKernel 家族及 step 路径体现状态更新。因此有限响应也能写成 shift-register 状态空间，不能声称摆脱了 SSM 先例或首次结构化状态分解。[4][8]

访问问题保留原状：S4 初始猜测路径 src/models/sequence/ss/ssm.py 返回 404，换成同仓实际 kernels/ssm.py 后获得 200；这只是路径纠正。Hyndman 的 BNpaperver4.pdf 返回 **403，停止该来源**，没有换域名、代理或正文别名继续获取。因此 Beveridge–Nelson 神经扩展、非线性 innovations observer / switching state-space 的全面先例排查仍未完成。正文与代码存在差别不代表首次发表证明。

### 更近的响应/观测器先例

- Deep Impulse Responses 正文已经提出以神经网络估计/参数化 IR，并联合估计未知谱噪声，针对低 SNR；但它有已知 source signal、空间坐标及声学响应。本任务没有已知冲击输入，所以不能照搬其可识别结论，亦不能把“神经响应核适合低信号”当首创。[9]
- Neural Luenberger state observer 下载版本为 2026-04-17 的 v2：离线训练使用历史/模拟 state measurements，学习 input-affine 项及 observer→state 非线性映射。本候选既没有真实 latent state 标签，也没有已知受控输入，不具备该文的误差保证。[10]
- **Correct then Forecast / OSSM** 下载版本为 2026-09-27 的 v1，已明确分开 measurement correction 与 autonomous propagation，而且上下文与预测共享 transition。这比仅比较 S4 更接近，故“先纠正再预测”绝不能作为本任务的新贡献。候选仅额外约束永久/有限暂态两路并采用已观察年龄证据；是否超过该文实例仍待核。[11]
- DS3M 下载版本为 2025-01-10 的 v3，使用离散 regime、连续 driving factors、RNN 与非线性 switching SSM。候选不是 Markov discrete regime 的变分模型，但存在条件响应分配的概念重合，后续必须加入小型 switching 线性状态模型强对照。[12]
- GitHub 的 APINN-Obs README 已固定 commit/blob，可确认公开工程采用 PINN observer/state/gain 学习；**不能确认它与上述论文作者关联，故不算作者代码或算法复现**。其 README 的 citation 还是占位 author，可靠性有限。[13]

针对 OSSM / DS3M / Deep Impulse 的 GitHub 专项检索，没有确认作者算法实现。OSSM 正文提到 TFB 和 SpaceTime 基线仓库，不能据此谎称找到 OSSM 作者实现。DS3M 搜索请求失败，不等于代码不存在。先例范围因此仍不完整。

## 2. 只选一个候选

暂称“有限暂态响应的年龄证据路由”，这个名字仅用于本项目定位。选择依据是：把难估计的 60lag→多步自由映射缩成一个全局短响应核和共享小门控，而不是堆叠另一套分段大网络。

输入是发行时点已知的 60 个观测点，记为 y_0,…,y_59；d_j=y_j-y_(j-1)，j=1,…,59。观测点步长不是日历间隔。不添加时钟、不接入外生变量。s 是训练前缀拟合并冻结的变化尺度，所有切分独立拟合，不能在完整文件上拟合。

### 2.1 可学习响应核：5 个自由参数

固定暂态最长年龄 L=12，节点 a=(0,2,4,6,8,10,12)。设 v=(v_1,…,v_5,0)，w=softmax(v)。

k(0)=1；k(2r)=1-Σ_(i≤r)w_i，r=1,…,6；节点间线性插值；k(a)=0（a≥12）。

所以 k 非负、单调且有限支撑，端点及幅度固定。它不是自由指数叠加，也没有可学习阻尼率；可以形成不同位置的肩部，但不允许振荡、overshoot 或第二个暂态核。固定末项 logit 消除 softmax 平移不辨识。

持久响应恒为 1，暂态响应为 k(a)。这里“持久”只指水平冲击长期保留，不表示有可预测漂移。

### 2.2 额外任务定制机制：随已观察年龄更新分配，而不是发行时点硬判一次

每个历史变化 j 在当前发行 t=59 形成四个特征。m=min(12,t-j)，S=Σ_(r=1)^m d_(j+r)，A=Σ_(r=1)^m |d_(j+r)|。

z_(j,t)=(|d_j|/s, m/12, d_j S/(d_j²+s²), d_j S/(|d_j| A+s²))。

对特征进行固定范围裁剪，范围只通过训练前缀确定，作为协议超参数公开。所有后续变化都不晚于当前发行时点；这里可以使用“某个过去冲击之后、但当前已知”的变化，不能使用当前发行之后的变化。

q_(j,t)=sigmoid(b_2+W_2 tanh(b_1+W_1 z_(j,t)))，W_1 为 4×4，W_2 为 1×4。

q 是持久份额，不先称为经过校准的永久状态概率。25 个神经参数在所有冲击、六个 10lag 段和 horizons 共享。m=0 时没有冲击后证据；随着过去冲击变老，模型可以根据已出现的 follow-through / reversal 重新分配。**这就是本次唯一的附加机制**，不额外加入已否决的一致性罚、时钟、方差罚或广义 ridge。

这并不是新型信息：同一发行时点的原始 60lag 已经包含这些变化。提出的是低容量因果估计结构。年龄证据机制也可能已经落在非线性平滑器/神经滤波/混合专家先例中，因此新颖性仅作为待核假设。

### 2.3 固定门控后的三角 innovations observer

对于一个发行前缀，先计算所有 q_(j,t)，再固定它们解 innovations。定义 k(-1)=0，持久项 g_P(a)=1_(a≥0)，并令

G_(j,t)(a)=q_(j,t)g_P(a)+(1-q_(j,t))k(a)。

d_n=Σ_(j≤n)e_(j,t)[G_(j,t)(n-j)-G_(j,t)(n-j-1)]。

因 G(0)=1，所以对角系数固定为 1，按 n 从小到大直接求解：

e_(n,t)=d_n-Σ_(j<n)(1-q_(j,t))e_(j,t)[k(n-j)-k(n-j-1)]。

每一步最多访问前 12 个 innovations。初始暂态状态明确设为零；这是有限前缀边界近似，不是准确识别。不能因回构 d 完全一致就声称分解正确——该方程本来就是可逆重参数化。

### 2.4 多步均值响应

假定发行后新 innovations 的条件均值为零，则

ŷ_(t+h|t)-y_t=Σ_(j=1)^t(1-q_(j,t))e_(j,t)[k(t+h-j)-k(t-j)]。

未来持久冲击均值为零，旧持久冲击在相减时抵消。只有尚未消退的暂态参与修正。h≥12 后修正达到相同平台，**不是回到零修正**。没有逐 horizon 额外参数；所有 horizon 使用同一响应核。若实际可预测信号主要是持续漂移而非暂态回撤，本候选可能不适用。

总可学习参数=门控 25+核 5=**30**；尺度、固定节点、初始状态、温度等不能偷偷增加可学习参数。门控重算和 innovations 回溯导致发行成本约 O(59×12)，训练目标按全部已成熟 horizons 回传，模型容量和总训练 FLOPs 分开报告。

## 3. 与已知方法的区别与退化关系

| 邻居 | 本候选的实际区别 | 必须承认的重合/退化 |
|---|---|---|
| 受限 ARIMA/MLE、强 state linear | 输入条件化 q 随每个历史变化及其已观察年龄变化；有限响应核由直接多步预测目标学习 | q 固定后就是线性 innovations/state-space 估计；永久/暂态分解本身非创新。必须比较同核线性 MLE，而不是弱 ARIMA |
| N-BEATS | 没有自由 backcast/forecast 波形头；每个冲击同一有限核，锚定幅度、端点及方向；门控后先求 innovations | N-BEATS generic 可以表达类似函数；优势若有只能是有限样本估计结构，不是表达能力首创[1][5] |
| DLinear | 不用移动平均分解，不学习两套自由 60→H 权重；输出被单一冲击响应约束 | q 固定时仍是线性预测器，必须用同参数预算的低秩线性控制，不因不同命名就算区别[2][6] |
| 阻尼/指数 basis | 学节点下降质量，严格有限消退；允许肩部、没有自由极点或指数率 | 有限支撑仍是 basis，也可用 nilpotent shift-register SSM 精确实现；不能宣称不是 basis/SSM[4][8] |
| Koopa | 不做 FFT 时变/时不变分解，不估 local K；每次发行只是同一个小门控配单核 | 分段和稳定/变化分路早有邻居；不能将二者当作发现[3][7] |

**值得测试的最小主张**：在弱、偶发、需要几步后才分清的回撤信号下，年龄证据共享路由比同容量、同响应核、但只看冲击发生时信息的门控更容易估计多步均值。不是“用了非线性就优于 MLE”，也不是首次神经永久/暂态模型。

## 4. 参数与可辨识限制（失败可以直接停止）

1. 固定持久基 1、k(0)=1、k(12)=0、单调核，消除交换和幅度缩放自由度；但不能保证真实经济状态可辨识。
2. e、q、k 可通过三角逆变换补偿；仅靠 reconstruction 没有识别力。只有跨发行、已成熟多步预测能约束估计，低信号样本仍可能平坦。
3. k 自由形状与门控永久份额可能混淆。报告固定核 vs 学核、profile loss 和不同初始化的 forecast 分歧，而不是只展示一个解释漂亮的核。
4. 禁止把 q 当概率绘制可信区间。只有在已知 latent 标签的合成场景上校准检验后，才能讨论该场景的标签概率；不能推广为市场经济状态概率。
5. 初始零暂态可能经逆系统传播；合成门必须额外注入前缀外未消退状态，并与 120lag oracle 对照。不能用连续回构准确掩盖这个问题。
6. 训练和测试记录三角矩阵 inverse operator norm、condition number、max |e|/s。若 worst-case 放大超过预注册阈值 10，或 small perturbation 导致巨大预测变化，停止当前实现；不能事后添加 rescue clipping 然后宣布原机制成功。
7. q 的年龄信息与 shock evidence 的信息区间明确记录。年龄 12 以上的冲击仍影响逆解，但不直接参与未来 correction。不能使用未来观测日期或价格去判 q。

## 5. 训练及强对照（还未运行）

- 只在未来获准的训练前缀操作；本次不执行市场训练。发行 t 的 h 标签仅在 t+h 已观察后成熟。每个 h 使用所有可用成熟 origin，不能为了凑完整标签而少用短 horizon 数据。
- 直接变化 MSE：L=平均_h 平均_(s+h≤T) [(ŷ_(s+h|s)-y_(s+h))/s_train]²。平均 horizon 权重、训练范围和刷新政策必须与父实验一致；不得把 overlap targets 当独立似然并宣称 MLE。
- 同一 30 参数候选只允许 3 个种子；全参数优化预算固定为 500 epochs，Adam 学习率 1e-3，记录每个 epoch loss、gradient norm、有效更新次数。是否足够由合成学习曲线判断，不以预算跑完代替充分优化。协议若在训练前改变，保留原稿与理由；运行后不得救援调参。
- A：历史已经验证的强 MLE 和强 state-linear **按原实现**；父侧报告的 0.949793 / 0.942705 仅作为父提供背景，不是本任务独立重评分。
- B：同一有限核、固定 q、按匹配支持拟合的线性 innovations MLE，以及 exact-solved raw-state ridge。用这个对照识别“换了核而已”。
- C：固定有限核+年龄门控；学核+常数 q；固定指数核+同门控；学核+无年龄/冲击后证据 gate。最后一项以冲击时上下文补足四输入，保留相同 25 参数，不靠少参数制造优势。
- D：保留 age、移除 follow-through；保留 follow-through、移除 age；发行内因果地置换已观察 evidence 但保留年龄分布。年龄与证据必须分别消融，不能只做一个 full permutation。
- E：同参数规模直接小 MLP、同 rank 自由响应字典、同预算 N-BEATS generic、小型 DLinear、SegRNN 原骨干与父线性模型。自由模型的参数/ FLOPs 不可能全匹配时同时报告，不作“公平完全匹配”口号。
- 各方法同发行信息、60lag、per-h成熟支持、刷新政策、标签尺度、损失和搜索次数；额外年龄证据同样交给自由小 MLP。模型只比更弱基线好不算机制门通过。

## 6. 预注册合成门（尚未通过，不能替代真实市场）

所有 synthetic 完全程序生成，绝不来自或拟合市场价格。固定 seeds=(7,17,29,43,71)，长度 2400，train 1200 / validation 600 / test 600，h=(1,5,10)。split 边缘 purge 10 个发行 origin。各 h 单独成熟；参数只在训练数据拟合。主指标为逐种子 loss、seed 平均和与 strongest control 的成对差，不能平均预测来降低 MSE。

生成族：y_n=Σ_(j≤n)η_j+Σ_(j≤n)u_j k_true(n-j)+ε_n。η 是永久创新，u 为 Bernoulli(0.05)×Normal(0,1) 暂态冲击；ε~Normal(0,0.1²)。信号尺度以永久标准差 (0.5,1,2) 分别固定。另加入条件状态族：shock 是否 transient 的混合率依赖一个隐藏短状态，早期 permanent/transient 观测响应相似，后期 reversal 区分；所有控制拿到同样观察序列，不能把隐藏状态只交给本模型。

- G0：纯随机游走/白噪声，无暂态。平均 MSE 不能比 random walk 恶化超过 1%；检查 q-collapse/冗余核，禁止只报告正场景。
- G1：线性高斯 permanent+已知有限 transient 核。**同信息精确状态滤波器应是强对照**，不要求神经模型击败正确已知 DGP oracle。要求实现接近 oracle（差距≤5%），不得把打败错配 ARIMA 当成创新。
- G2：上述弱非线性混合冲击，核 shape 固定有 shoulder。要求在至少 4/5 个 seed 上相对最强可训练控制减少 MSE，平均至少 3%；同时比“无已观察年龄证据、相同 30 参数”版本至少改善 2%，并保留每 h 指标。否则停止年龄证据机制。
- G3：指数真核。比较同 gate 指数核和学有限核；有限核不应系统性输 3% 以上。此门检查 shape misspecification，不宣布有限核全面优越。
- G4：持久漂移可预测、没有暂态；明确属于模型的负适用范围。若神经 gate靠噪声在该场景过拟合，不能再称低信号稳健。
- G5：含前缀外 transient 状态，增加噪声并混合邻近 opposite shocks；检验 inverse 放大及 60lag/120lag oracle 差异。超过稳定阈值就停止当前构造。
- 工程门：未来 suffix 扰动不能改变早期发行 features/forecast；unmatured labels 扰动不能改变当次拟合；finite-difference 校验 gate/kernel gradient；固定 q 的预测叠加性、学 gate 的非叠加性分别验证。核单调、端点、h≥12平台必须精确检查。完整重构只是代数单元检查，不计为 G1成功。

G0/G1/G2/G5 全部通过，且机制消融有效，才有理由向父任务申请后续市场前缀试验；**本次没有这些结果**。

## 7. 研究结论和收据边界

保留一个具体候选，不继续列模型清单。目前能说明它与指定基线的约束/估计接口不同，不能说明它首次出现。最危险的近邻是 nonlinear innovations observer、switching SSM 和 conditional FIR mixture；8 次搜索请求和当前正文/代码核查不足以排除它们。没有因为已有部件就否决这个候选，但也没有把组合包装为首创。

receipts 中保存真实下载、HTTP 失败、PDF SHA256、Git commit/blob 比较和提取正文。正文只重点核查结构段落；没有重跑论文 benchmarks。检索共 8 queries（其中 2 次请求失败、2 次返回空结果），已达到 8 query 上限；原始返回保存 search_queries*.json。搜索工具的 openai-native 后端不可用，工具自动使用公开检索后端，失败收据完整保留；没有通过自行更换网络绕过来源 403。torch_segmented_backbone_20261014 的 backbone.py、market_parent.py 在此工作区实际存在；只检查文件名，没有读取 npz 或价格。没有因之前隔离视图误称父骨干不存在。

## Sources

[1] https://arxiv.org/pdf/1905.10437
[2] https://arxiv.org/pdf/2205.13504
[3] https://papers.neurips.cc/paper_files/paper/2023/file/28b3dc0970fa4624a63278a4268de997-Paper-Conference.pdf
[4] https://arxiv.org/pdf/2111.00396
[5] https://api.github.com/repos/ServiceNow/N-BEATS/contents/models/nbeats.py?ref=c746a4f13ffc957487e0c3279b182c3030836053
[6] https://api.github.com/repos/cure-lab/LTSF-Linear/contents/models/DLinear.py?ref=0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6
[7] https://api.github.com/repos/thuml/Koopa/contents/models/Koopa.py?ref=a2e0bb77ec7c1a25e8e0579ba517ffb41358b844
[8] https://api.github.com/repos/state-spaces/s4/contents/src/models/sequence/kernels/ssm.py?ref=e757cef57d89e448c413de7325ed5601aceaac13
[9] https://arxiv.org/pdf/2202.03416
[10] https://arxiv.org/pdf/2602.24252
[11] https://arxiv.org/pdf/2609.33566
[12] https://arxiv.org/pdf/2106.02329
[13] https://api.github.com/repos/FARKANE/APINN-Obs/contents/README.md?ref=b52795898fbe8eac5c0d2ea392a2c92ba59fb2df
