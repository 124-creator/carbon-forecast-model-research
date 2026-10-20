# 文献和作者代码驱动的模型选择

## 结论

唯一选择：**时序留尾交叉拟合的编码器元适配 Koopman 模型**。模型直接学习价格变化的非线性动力学，不接入旧预测的残差校准、门控或输出混合。两个待检验机制是：①元学习后在最近成熟支持上适配非线性编码器；②内层适配用“较早支持拟合算子、较晚支持评价预测”的留尾交叉拟合，避免编码器只降低同支持的算子拟合误差。它是一项明确的学习算法假设，**不是已确认首创，也不是已取得预测收益**。

MAML、Koopa、编码器适配的神经状态空间模型和 MetaKoopman 都已经存在；本轮查新排除了“元学习+Koopman本身就是创新”的说法。[1][3][16][17][18]

本轮没有价格训练，没有打开价格文件或2022边界，没有读取旧pickle、下载权重、克隆仓库、安装第三方依赖或执行作者代码。保存的是论文、API响应和静态读审文本，不是可运行的第三方代码目录。日期目录沿用用户指定的20261014，抓取时间保留实际UTC时间，不伪称已经是10月14日。950点与2020—2021的451个开发起点，以及MLE 0.949793、简单线性校正0.94270547，均来自任务上下文，本轮未复算。

## 一、三个不同学习机制及选择原因

### A. 小样本任务适配：MAML与神经状态空间元学习

MAML用支持集梯度更新θ′=θ−α∇L_S(θ)，再用查询集L_Q(θ′)训练初始化。这改变了训练目标与适配位置，不是对已有预测添加线性校准。[1] 作者代码cbfinn/maml的maml.py明确区分inputa/inputb和内外梯度，并有stop_grad的一阶变体。[5] 作者main.py含pickle和自动resume路径，因此不能直接运行默认入口或加载仓库权重。[6]

更直接的近邻是Chakrabarty等人的神经状态空间元学习：论文已研究只适配编码器、固定状态/输出矩阵；2025扩展进一步讨论MAML、ANIL与物理约束。因此“只适配编码器”也不能独占为新贡献。[16][17] 作者MERL实现MetaLIC的README说明Bouc-Wen系统数据与预训练权重；代码中存在神经编码器、可选线性状态转换和MAML二阶梯度。[19][22][23]

选择价值：少量近期标签可以更新**特征表示**，不只更新偏差或输出幅度。风险：同一市场的相邻窗口不是独立任务，多任务先验可能不足；嵌套梯度可能不稳定；元学习未必胜过同网络普通训练加微调。

### B. 结构约束非线性：ICNN/PICNN

ICNN通过非负隐藏到隐藏权重及凸、非递减激活，使能量E(x,y)对指定输入y凸，再通过argmin_y E(x,y)输出预测。凸性限制的是指定输入，不是参数训练问题；原文明确训练仍可非凸。[2] 作者locuslab/icnn的synthetic-cls/icnn.py用投影保证相关权重非负，README列出TensorFlow与RL环境依赖。[7][9]

潜在模型是对未来变化向量y定义输入凸能量，历史x允许非线性；不能未经证明便假定“价格对能源必然单调/凸”。不选择的理由：多步均值预测没有已验证的经济形状约束，能量推断与训练复杂度增加，旧TensorFlow移植工作较多。该方向真正不同于正则校准，但目前缺少值得强加的约束。

### C. 非平稳动力学学习：Koopa

Koopa学习非线性潜空间，局部算子由过去潜状态的最小二乘估计，时间不变部分另有可学习算子，并采用预测损失而不绑定传统重构损失。论文和README均给出作者thuml/Koopa实现。[3][12] 已读models/Koopa.py：TimeVarKP采用tanh MLP，KPLayer以torch.linalg.lstsq求局部K并递推；TimeInvKP采用全局线性K；这些机制已经存在。[11]

本轮不重新采用频率分解：不引入SSA、随机特征、分解修订或共享残差偏差。选择的是**任务适配的非线性动力学表示**，不是把傅里叶分解或局部最小二乘改名。

Koopa exp_main.py的频率掩码从train_loader计算；标准训练分割下不等于全样本频谱泄漏。但data_loader.py先pd.read_csv整个文件，预测专用分支还对整个输入拟合scaler；原入口不符合禁止读取2022后缀的严格边界。[14][15] 不执行该入口，未来使用独立的前缀输入器。

## 二、额外查新：已经存在的直接近邻

- **MetaLIC / 2023神经状态空间元学习及2025扩展**：已有编码器MAML/ANIL适配；其架构含状态/输出映射，并非本候选的“随编码器重新解析拟合局部K，再在留尾支持上更新编码器”。但本文献已覆盖宽泛的编码器元适配，不能声称这部分新颖。[16][17][22]
- **MetaKoopman，NeurIPS 2025**：已经元学习MNIW算子先验并进行闭式贝叶斯更新。会议主源摘要与作者实现README已读；作者README限定公开代码为MuJoCo模拟，不包含真实硬件实现。这里不能把所有论文实证都视为公开复现代码。[18][24]
- **MAKO作者仓库**：README明确Meta-Adaptive Koopman及控制应用，且环境涉及Mosek许可。只读README，未完成方法全文和训练代码审计，不能宣称排除等价；保留为后续强近邻。[25]

因此本候选的可检验差别窄化为：**近期编码器梯度针对时序留尾的多步价格预测，并通过较早支持的局部解析算子反传；外层又针对后续时间段优化这种适配。**若现有方法已经采用相同三段划分、同一解析局部算子依赖和相同梯度路径，则算法贡献撤回。金融场景、成熟标签掩码、tanh、ridge抖动、小维度以及CPU实现均不算新增机制。

## 三、明确数学模型

所有索引为交易记录序号，不是日历天。发行起点t可使用P_≤t；ΔP_s=P_s−P_(s−1)。取固定L=6，x_s为过去6个标准化变化。尺度由初始允许前缀估计后冻结，所有方法共用；s之后的数据不进入x_s。

非线性编码器仅有两维：

z_s=φ_θ(x_s)=tanh(Wx_s+b)，W∈R^(2×6)，b∈R²。

解码器c∈R²不带截距；不存在旧基线残差输入。对过去已完成的一步潜状态对A：

K_A(θ)=(Z_AᵀZ_A+εI)^(-1) Z_Aᵀ Z_A^+。

ε=10^(-4)固定，仅是数值抖动，不能算创新；全部Koopman对照匹配该抖动并报告条件数。若实际需要改变ε，先公开更改记录，不悄悄搜索。该模型采用行向量z，K为2×2。

预测原价：

P̂_(s+h)=P_s+σ₀ Σ_(j=1)^h cᵀ (z_s K_A^j)ᵀ。

同一动力学递推产生各步长，不用每个h另加校准头。有限维Koopman只是模型假设，不保证真实金融过程存在精确闭合潜空间；二阶维度也可能不足。

### 内层两个机制

每个历史任务τ由连续三段A_τ、B_τ、Q_τ组成：A最早，B随后，Q最晚。A包含32个完成的一步潜状态对；B包含8个已成熟预测起点；Q包含8个随后成熟起点。任何分割都需保证前段所有目标端点严格早于下一段第一个起点，不仅起点不重叠。为同支持消融，每个起点只保留H*全部步长均已成熟的标签。H*取既有预写协议的最大步长，本轮不自行更改。

定义原价共同尺度损失：

ℒ_C(θ,c;K)= |C|^(-1) Σ_(s∈C) Σ_(h∈H*) w_h [(P̂_(s+h)(θ,c,K)−P_(s+h))/σ₀]²。

**机制1：编码器元适配**。只更新W,b，不更新c：

θ′_τ=θ−α ∇_θ ℒ_(Bτ)(θ,c;K_(Aτ)(θ))。

梯度同时穿过tanh与解析K拟合；不是固定随机特征回归，也不是只调输出。α固定，先用无市场数据的数值检查选择可稳定尺度，不以开发区最优收益反复调节。

**机制2：时序留尾交叉拟合**。上述内层损失在B上计算，但K只用更早的A；适配后的K重新由A∪已完成的B转移对以θ′编码并拟合，然后预测Q。B的未来目标不能进入适配前的A算子。区别不在普通加权/惩罚，而在内层梯度的评价数据与拟合数据不同。交叉拟合本身已有广泛先例，因此只提出此处与局部潜动力学耦合的增量假设，不声称发明交叉拟合。

外层训练：

min_(θ,c) Σ_τ ℒ_(Qτ)(θ′_τ,c;K_(Aτ∪Bτ)(θ′_τ))。

外层必须包括∂θ′/∂θ与K对θ′的导数；冻结梯度的FOMAML只能列为近似对照，不能伪称与精确目标相同。每个任务的全部Q目标必须在当次元更新之前成熟。

发行t：以最近完整成熟B及更早A做一次上述编码器适配，随后仅用截至t的已观测一步潜转移重新拟合K_t(θ′)，从x_t递推各h。θ′每次从当前共享初始化θ开始，不在历史未来处回填发行特征。保存发行版本θ、c、K、尺度、支持最大目标时点与预测。

## 四、为什么不是普通校准或固定特征正则

固定θ时，K估计确实是普通带抖动线性回归，本轮不为该步骤申请创新。完整估计器却对θ作跨时间任务元优化，内层又对θ作适配，K依赖θ且预测包含K^j；因此参数到预测不是固定设计矩阵的单次ridge/Tikhonov解。不能把全部过程简化为给原预测乘幅度或加偏置。

存在退化：α=0时去掉适配；φ固定时退化为固定字典EDMD；tanh处于近线性区时近似线性动力学。这些正是强对照。只有若在匹配支持与预算下适配和留尾消融均改善，才能说额外机制有证据；仅胜过SSA/RFF失败原型不够。

## 五、创新假设与可否证结果

H1：近期市场状态变化表现为需要修改潜空间表示，而不仅修改局部K；适配编码器的多步预测损失优于冻结编码器、只刷新K。

H2：用同支持转移拟合误差适配编码器会出现表示过拟合；留尾多步梯度有更好的时间外预测。H2必须胜过同参数同预算但内层不留尾的版本，而不是多拿标签或少用困难起点。

这两点均是尚无结果的假设。现有元状态空间方法使H1宽泛形式不新；本轮没有证明H2组合不存在文献等价。若唯一收益来自ridge、初始尺度、线性变化或更长窗口，撤回新增机制主张。若仅元学习初始化有效而适配无效，不能给完整双机制命名背书。

## 六、强对照和同支持规则

所有方法使用相同发行起点、历史信息、标签成熟规则、初始尺度、H*和原价损失；保留全部种子损失而非只平均预测。2020、2021均为反复开发区，不能改称盲测。禁止读取2022后缀，即使只求全文件hash也不允许。

必须对照：

1. 随机游走、强MLE、既有简单线性校正；指标来自任务上下文，后续须从安全前缀重新训练/评分，不加载旧模型。
2. 同原始变化输入的小MLP，普通监督训练；同网络普通预训练+同一次近期微调，隔离元初始化作用。
3. 固定字典EDMD及线性φ版本；随机固定tanh字典只作机制对照，不能把它重新包装成RFF创新。
4. 同低维非线性编码器、同解析局部K，α=0的普通端到端预测训练；这是无元适配Koopa思想对照，不声称复现完整Koopa。
5. 同架构普通MAML、内层不留尾；仍严格时序且目标成熟，不能拿违规未来泄漏版本做弱对照。
6. 只适配c或K的ANIL/局部算子版本，与只适配θ版本公平比较。
7. MetaLIC式编码器适配而全局状态矩阵固定的干净重实现；MetaKoopman式闭式算子适配若实施，必须注明是否只重现后验均值而非完整作者模型。
8. 完整模型去掉机制1、去掉机制2、同时去掉两项的三项消融。

冻结窗口总信息预算、任务数、外层目标评价次数、内层更新次数与搜索个数；普通监督对照应允许相同计算预算而不只是相同epoch。共同成熟支持为主比较，再单列逐h全部成熟支持的附加分析，不能将后者额外标签量算作贡献。

最低结论门槛：开发总体原价损失优于0.94270547及重训MLE；2020/2021各自不劣于最强对照；两个新增机制各自消融有稳定增量；种子/优化诊断稳定。具体容忍值由既有验收口径给定，尚未从上下文取得，不编造。不能承诺达标或将开发结果等同外部泛化。

## 七、依赖、算力与安全审核

|作者实现|已核查许可|依赖和风险|本任务处理|
|---|---|---|---|
|cbfinn/maml|MIT，版权Chelsea Finn[4]|旧TensorFlow、二阶梯度；main有pickle/自动resume[5][6]|读审而不运行，不获取权重|
|locuslab/icnn|Apache-2.0；README说明DDPG部分MIT[7][8]|TensorFlow旧API、tflearn；RL还涉及Gym/MuJoCo[7][9]|不用RL，不直接移植入口|
|thuml/Koopa|MIT[10]|requirements钉torch2.0.1等；训练入口torch.load；整文件CSV读入[13][14][15]|不安装、不执行入口，不读取仓库数据|
|merlresearch/MetaLIC|AGPL-3.0-or-later；maml局部MIT、测试数据CC-BY-SA-4.0[19][20]|torch2.5.1；仓库含.mat数据与.pt/.pth预训练[19][21]|只阅读公式/代码，不复制实现到项目，不下载数据/权重|

许可核查是来源审计，不是专业法律意见。未来如复制MIT/Apache源码，必须保留相关版权/许可；MetaLIC的不同文件不能一概视为MIT。本候选采用独立NumPy/SciPy数学实现，是否形成衍生作品不能仅凭改写语法判断；本轮没有执行这种复制。

本地发现numpy/scipy可用、torch缺失。候选14个编码器参数+2个读出参数、2×2解析求解，适合先用NumPy有限差分做精确嵌套目标的小型实现，不需要大型安装或GPU。有限差分会放大计算成本；不能把“可用CPU”说成已测运行时。未来先计时单任务、检查内层导数和外层方向导数，再决定预算；如不稳定，不能静默换成只优化输出层。

**未运行市场训练。**下一实施检查应仅用显式合成的非线性潜动力学数据做梯度与机制检查，不能把合成改善写成市场收益。首要检查：外层方向导数；未来标签扰动不影响当前发行；前缀重放；同支持一致；固定θ退化等价；留尾与不留尾的计算预算。无信号/线性稳定系统用于观察额外适配是否伤害，不隐瞒不利结果。

## 八、获取收据与局限

retrieval_receipts.json保存每次正文请求的实际URL、状态码或错误、UTC时间、文件名、SHA256。成功和失败都保留；search_*.json保存真实搜索返回及其后端回退信息。没有伪写web.run收据。

最早OpenReview请求使用未经确认的ID，返回403，不能用作Koopa来源，不重试该ID、不通过代理/镜像绕过；Koopa身份及正文以独立公开NeurIPS会议论文和作者README核对。若403包含安全拒绝，本轮不尝试其同正文别名。GitHub raw若为连接错误（非安全/认证拒绝），通过公开API读取固定commit内容；此区别保留在原始错误中。MetaLIC LICENSE路径第一次404，之后按照真实树里的LICENSE.md读取，并保留原404。部分额外近邻README用分支引用而未钉commit，只提供范围提醒，不据此认定方法等价。

bs4不在执行解释器中；一次提取因此中断，已有结果留存，后续使用stdlib做HTML文本转换。PDF使用已安装pymupdf/fitz抽取文本，没有执行PDF内的程序。文献阅读覆盖正文的机制/相关工作与代码相关段落，不意味着重审所有原文实验或独立复现实验。

receipt_37的MetaKoopman arXiv页确实读取成功，但会议年份与arXiv上传年份不同，本报告以会议2025标识，不以URL年份推断首发。新颖性搜索仍有限；MAKO未取得方法全文、MetaKoopman未完整读审模型实现，不得声称查新穷尽。

## Sources

[1] https://proceedings.mlr.press/v70/finn17a/finn17a.pdf — Receipt 4
[2] https://proceedings.mlr.press/v70/amos17b/amos17b.pdf — Receipt 5
[3] https://proceedings.neurips.cc/paper_files/paper/2023/file/28b3dc0970fa4624a63278a4268de997-Paper-Conference.pdf — Receipt 9
[4] https://api.github.com/repos/cbfinn/maml/contents/LICENSE?ref=a7f45f1bcd7457fe97b227a21e89b8a82cc5fa49 — Receipt 23
[5] https://api.github.com/repos/cbfinn/maml/contents/maml.py?ref=a7f45f1bcd7457fe97b227a21e89b8a82cc5fa49 — Receipt 24
[6] https://api.github.com/repos/cbfinn/maml/contents/main.py?ref=a7f45f1bcd7457fe97b227a21e89b8a82cc5fa49 — Receipt 25
[7] https://api.github.com/repos/locuslab/icnn/contents/README.md?ref=762cc04992b6e649b4d82953d0c4a821912b1207 — Receipt 26
[8] https://api.github.com/repos/locuslab/icnn/contents/LICENSE?ref=762cc04992b6e649b4d82953d0c4a821912b1207 — Receipt 27
[9] https://api.github.com/repos/locuslab/icnn/contents/synthetic-cls/icnn.py?ref=762cc04992b6e649b4d82953d0c4a821912b1207 — Receipt 28
[10] https://raw.githubusercontent.com/thuml/Koopa/a2e0bb77ec7c1a25e8e0579ba517ffb41358b844/LICENSE — Receipt 18
[11] https://raw.githubusercontent.com/thuml/Koopa/a2e0bb77ec7c1a25e8e0579ba517ffb41358b844/models/Koopa.py — Receipt 20
[12] https://api.github.com/repos/thuml/Koopa/contents/README.md?ref=a2e0bb77ec7c1a25e8e0579ba517ffb41358b844 — Receipt 29
[13] https://api.github.com/repos/thuml/Koopa/contents/requirements.txt?ref=a2e0bb77ec7c1a25e8e0579ba517ffb41358b844 — Receipt 30
[14] https://api.github.com/repos/thuml/Koopa/contents/exp/exp_main.py?ref=a2e0bb77ec7c1a25e8e0579ba517ffb41358b844 — Receipt 31
[15] https://api.github.com/repos/thuml/Koopa/contents/data_provider/data_loader.py?ref=a2e0bb77ec7c1a25e8e0579ba517ffb41358b844 — Receipt 32
[16] https://arxiv.org/html/2211.07768v1 — Receipt 38
[17] https://arxiv.org/html/2501.06167v1 — Receipt 39
[18] https://proceedings.neurips.cc/paper_files/paper/2025/hash/2a5dd6bad2d10966a8bfa254dfb7dc76-Abstract-Conference.html — Receipt 40
[19] https://api.github.com/repos/merlresearch/MetaLIC/contents/README.md?ref=6ad9d060c45b6df17f19271fa956f37374364166 — Receipt 42
[20] https://api.github.com/repos/merlresearch/MetaLIC/contents/LICENSE.md?ref=6ad9d060c45b6df17f19271fa956f37374364166 — Receipt 44
[21] https://api.github.com/repos/merlresearch/MetaLIC/contents/requirements.txt?ref=6ad9d060c45b6df17f19271fa956f37374364166 — Receipt 45
[22] https://api.github.com/repos/merlresearch/MetaLIC/contents/networks/KoopmanNetMixed.py?ref=6ad9d060c45b6df17f19271fa956f37374364166 — Receipt 46
[23] https://api.github.com/repos/merlresearch/MetaLIC/contents/maml/maml.py?ref=6ad9d060c45b6df17f19271fa956f37374364166 — Receipt 47
[24] https://api.github.com/repos/Mahmoud-Selim/MetaKoopman/contents/README.md?ref=main — Receipt 36
[25] https://api.github.com/repos/hithmh/Meta-Koopman/contents/README.md?ref=master — Receipt 34
