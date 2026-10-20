# 多步累计变化的响应学习：一个可以检验、目前不能作为创新放行的候选

日期：2026-10-14（UTC+08）。状态：**NO-GO / 只允许作为可否证的学习机制实验，不允许写成已成立的新模型贡献。**

## 1. 直接结论

选择的研究候选是**有限幅度、带符号、多步响应对照训练（signed finite-response contrast training，简称 SFRC，仅为工作名）**。保留成熟 SegRNN 的编码器、GRU、并行多步解码器；学习整个累计变化向量对可预测状态改变的响应，而不是要求不同分段相位输出相同。不新增趋势专家、回复专家或指数基函数。

目前不放行的理由有三项，不能通过换名字绕过去：

1. 方向导数监督并不是新机制。Tangent Prop 原文不仅讨论零导数不变性，也允许指定非零目标方向导数；有限幅度差分对照在小扰动极限回到这一已有思路。[13]
2. 合成系统训练、成对反事实响应诊断也不是新方向。2026 年的 forced-system 论文已经明确使用这些设计，并强调相同合成数据训练的另一骨干可以接近微调结果。[16] 此文的外部输入系统与本候选的历史内部状态改变不同，但这种任务区别本身不够支撑新机制贡献。
3. 本设计目前没有证明，改进来自新的响应学习作用，而不是同一组模拟样本上的相关误差加权、数据增广或模拟器先验。即使得到更好的函数逼近，也不能跳过这一步。

**这次选择的是一个值得否证的学习机制假设，不是通过创新审查的模型。** 不因为函数类未改变就判死刑；也不因为增加一个损失就承认创新。若强邻居训练表现相同，应终止该候选，不追加第三个模块补救。

## 2. 预测目标和 h 响应到底有什么区别

给定长度 L 的价格历史 X_t，预测 C_t(h)=p_{t+h}-p_t，h=1,...,H。累计变化不是把未来价格再次累计；等价写法是 C_t(h)=sum_{j=1}^h r_{t+j}，其中 r_{t+j}=p_{t+j}-p_{t+j-1}。

以下模型只用于说明和合成控制，不是建议的新预测结构。设 p_t=ell_t+u_t，ell_{t+j}=ell_t+j m_t；零均值噪声下，u 的线性回复为 E[u_{t+h}|u_t]=rho^h u_t，0<rho<1。那么

    E[C_t(h)|m_t,u_t] = h m_t + (rho^h-1)u_t.
    d C_t(h)/d m_t = h.
    d C_t(h)/d u_t = rho^h-1.

趋势响应随 h 积累；一个当前正向偏离的回复贡献为负，短期逐步增大绝对值、长期趋于 -u_t。相邻步的趋势贡献为 m_t，回复贡献为 (rho-1)rho^(h-1)u_t，后者逐渐消退。这里区分的是**持久增长与暂时偏离**；不能把所有短期收益相关性都称为价格均值回复。

非对称例子可以令 u_{t+1}=rho_s u_t，s 为 u_t 的符号，rho_+ != rho_-。确定性且符号不跨越时，上下两侧的响应分别采用对应 rho_s；它不是奇函数。存在噪声、符号切换时，不能再套用一个固定 rho_s^h，必须计算条件期望或用模拟器重复采样。

真正实验不把上面两条曲线当作硬性模板。非线性回复可改变时间形状，趋势可转折，不同状态可能在相同 h 给出相反响应。GSTAR 原文已经研究非对称动态调整及多步预测，故“允许非对称”本身不是创新。[18]

价格历史不能唯一确定 ell、u、m。真实数据中不得把编码器某个坐标事后命名为“真实均值回复因子”。只有合成系统中的可控状态能提供已知成分响应；真实样本只评价预测表现和可观察诊断，不宣称因果识别。

## 3. 与成熟 SegRNN 的精确接口

作者代码是 segment Linear+ReLU embedding，单层循环编码，再以位置/通道嵌入执行 PMF；PMF 仍调用非线性 GRU，不能替换成线性探针后声称等价。[1][2] 默认非 RevIN 路径先减最后观测、最后加回。官方 README 的后续版本和原始 2023 预印本不是同一发表日期；此次以下载代码快照为实现依据，不声称运行了 IEEE 最终论文全套实验。[3]

定义现有成熟骨干输出 F_theta(X)_h=预测价格_h-X_last；如果骨干原本输出累计变化则不再相减。每一个反事实历史必须减**自身**末值。两条历史的末值可以不同，绝不能错误地共用 factual 的 p_t，否则回复响应会被常数偏移污染。

保留原分段策略、非线性 decoder 和按 h 输出的现有位置参数。h 依赖响应通过 F_theta(X')_h-F_theta(X)_h 直接学习；不是新增 h 基展开。模型在推理时只运行一次原骨干，不进行专家混合、不加残差校准、不做相位投票。

## 4. 最多两个新增部件及数学定义

### 部件 A：已知状态的成对合成生成协议（训练数据机制，非预测模块）

模拟器 S(z,a,omega) 返回 (X,C)，z 是共同上下文参数，a=(m,u) 是可改变的终端状态，omega 是共享随机数。通过改变初始状态/驱动并重新模拟**完整历史和未来**生成有效轨迹，不直接往最后一点拼接不符合动力学的扰动。配对需要每个历史含足够过去信息来识别自身状态；若不能做到，记录成不可辨识控制，不用 oracle 状态喂给模型。

对 j∈{m,u} 和有符号幅度 delta，令 a'=a+delta e_j，获得 (X,C),(X',C')。目标响应

    R_j(z,a,delta,h) = E_omega[C'(h)-C(h)].

正负 delta 分别采样，幅度独立变化；不要求 R(+delta)=-R(-delta)，不要求回复响应单调，不强制为指数。噪声系统使用共享 omega 的多次重复估计条件均值；训练与验证使用互不共享的随机种子和参数组合。不要把单次含噪响应声称为精确条件均值。

### 部件 B：全 h 非零响应对照损失（唯一额外训练项）

定义逐步误差 e_h=F_theta(X)_h-C(h)，e'_h=F_theta(X')_h-C'(h)，权重 w_h>=0 且 sum_h w_h=1，来自训练尺度的固定标准化，不由测试结果选择：

    L_pair = (1/2) sum_h w_h(e_h^2+e'_h^2).
    L_response = sum_h w_h[(F_theta(X')_h-F_theta(X)_h)-R_j(h)]^2.
    L = L_pair + lambda L_response.

若每对 C 和 C' 已是条件均值，R_j=C'-C。lambda 在验证集上选择，所有强邻居获得相同超参数搜索预算。delta 不趋于零也不把正负响应合并；这样监督整个 h 向量的有限幅度和方向，而非只惩罚输入敏感度。

真实数据阶段没有可验证的反事实目标，**不制造“真实回复标签”**；只用普通累计变化监督继续训练。合成预训练和混合训练必须分开报告。仅在某一模拟器上成立的结果只能支持该模拟器上的学习作用。

## 5. 必須正文披露的等价与边界

### 5.1 哪些伪结构直接排除

- h*a(X)+(1-rho(X)^h)*b(X)：它本来就是带条件系数的基展开；不能声称已排除 generic N-BEATS。N-BEATS 包含 generic 和解释型基扩展，作者代码提供相应实现。[4][5]
- sum_k gate_k(X)*expert_k(X,h)：普通专家混合，不因为专家名字叫趋势/回复而变成新机制。
- 增量 gamma^h*b(X) 或累计 sum_{j<=h}gamma^j*b(X)：指数阻尼。ETSformer 作者 README 关联其主源论文，DampingLayer 正是 powers、幂、cumsum 的实现。[6][7][8] 因而“长步趋于平台”并不独属于均值回复；阻尼增长也会趋于平台。
- rho_± 分段 AR、TAR/STAR/GSTAR、ARIMA 加漂移：均须作为已有统计邻居；新候选不能只胜过被限制的对称 ARIMA。[18]
- 学习 h 插值或多尺度投影：N-HiTS 已有多速率处理和层次插值；不能把平滑响应头单独当新意。[9][10]

SFRC 没有使用上述新结构，但**“没有使用它们”不等于排除了已有学习方法等价**。原 SegRNN 同样可以表达很多响应，SFRC 要证明的只能是训练中的响应估计改进。

### 5.2 配对损失的精确代数（不能绕过）

条件均值配对情形，L_response=sum_h w_h(e'_h-e_h)^2。令 c_h=(e_h+e'_h)/sqrt(2)，d_h=(e'_h-e_h)/sqrt(2)，则

    L = sum_h w_h[ (1/2)c_h^2 + (1/2+2lambda)d_h^2 ].

它是对共同误差和差异误差进行不同权重的相关监督，不是一个新函数类。在线性冻结特征下还是标准二次误差问题；不得包装成新型广义 ridge。端到端非线性训练可以改变参数学习，但此代数仍然成立。

小幅 delta 且可微时，响应差为 delta*D_{e_j}F_theta+o(delta)，目标同理；L_response/delta^2 趋于方向导数匹配。Tangent Prop 已包含指定目标导数，所以不能用“非零而非零不变性”绕过该邻居。[13]

普通成对增广的 lambda=0 不一定与上述 lambda>0 完全相同，但只证明目标函数不同不够。必须比较同数据、同次数前向、同训练时长的相关损失控制。差别若仅是已知目标权重选择，这次模型创新审查仍应拒绝。

### 5.3 反事实邻居并未缺席

ForecastCF 改变输入以得到指定输出范围，是预测模型解释，不是这里的已知真实响应监督。[11][12]

CounTS 是自解释和反事实推断模型；ConTex 学习共享干预策略。[14][15] 这些区别可以说明任务边界，不能证明本方法首创。

2026 forced-system 论文尤其需要保留：它已经把成对响应诊断、合成微调、相同数据的不同骨干控制放到同一研究中。[16] 其作者代码链接未在已读取正文中确认，不声称审计了该论文 GitHub。2024 causal-forecasting 论文采用正交学习训练因果预测，也提醒 observational forecast 不是 treatment-effect evidence。[17] 本候选不采用该因果识别假设，不把内生价格当可干预治疗变量。

## 6. 强邻居与消融：不是只对比弱 SegRNN

所有神经控制首先使用同一个成熟 SegRNN，相同初始化、训练样本、训练步数、参数数目和搜索预算。另报告等墙钟预算，避免三次前向比较一次前向造成误导。

| 必须项 | 检验问题 |
|---|---|
| 原骨干普通训练；同合成数据普通增广 | 是否只是样本/模拟器先验增加 |
| SFRC 的 lambda=0 | 配对组织而无响应监督是否已足够 |
| 同骨干目标方向导数匹配（Tangent Prop 的非零目标形式） | 有限幅度是否提供超出局部导数的信息 |
| 同配对样本、共同/差异误差权重的直接实现 | 必须数值等价；它是防止换名的阳性控制，不是可战胜的新对手 |
| 打乱 response 目标的配对归属 | 是否依赖真实成对方向关系，而非一般平滑/训练噪声 |
| 强制正负对称监督 | 在真实不对称 DGP 上是否丢失必要信息 |
| 仅 h=1 响应监督、全 h 响应监督 | 全 h 目标是否提供实际增量 |
| 只学习 m、只学习 u、联合配对 | 趋势和回复是否被混为一谈 |
| 线性漂移/ARIMA；TAR/STAR/GSTAR/NARX | 新机制是否只是解决统计模型本来会的任务 |
| 同骨干 generic basis head、指数阻尼 head、两专家 head | 等参数预算下，简单现有结构是否更有效 |
| N-BEATS generic+interpretable；N-HiTS；ETSformer | 外部成熟邻居，仅作为独立模型而非混装 |

若同配对的直接加权目标与 SFRC 结果不同，首先排查实现/随机性；不能把等价实现之间的偶然差距当机制优势。

## 7. 可否证协议与放行条件

只使用合成数据执行第一阶段；本次不读取市场价格。建议先固定 L=64,H=32,segment=8，并使用成熟底座支持的长度、CPU float32 和完整 PMF。此为预注册参数建议，不是已经完成的训练。

四个必须控制的 DGP：

1. 线性对称回复+稳定漂移：已有 AR/漂移正确设定，预期不必胜出；作为反例保留。
2. 不对称、幅度依赖的回复+状态依赖趋势：训练/测试分离正负侧速率、幅度和组合，检验非局部响应及 h 交互。
3. 仅趋势、仅回复、不可辨识历史：避免两个名称实际上拟合同一成分；不可辨识控制不应声称识别成功。
4. 趋势突变、回复机制变化的域外测试：没有真实响应标签的继续训练；若退化或遗忘，完整报告。

至少五个固定种子，时间顺序/模拟轨迹级切分，训练和验证不共享轨迹。报告按 h 累计变化 MSE/MAE、逐步变化误差、带符号响应误差、幅度比、正负侧分别指标以及训练时间。响应接近零时避免相对误差分母不稳定，以训练尺度归一化绝对误差。真实价格应用另行授权；不能为了这份设计偷偷读取现有价格文件。

允许放行到“进一步创新论证”的条件是：同数据等计算的普通增广和非零方向导数邻居不能解释优势；优势在不对称非局部 h 响应中存在、在线性负控制中不被夸大；打乱配对和只用一步的控制能否定对应机制；域外预测不系统退化；找到并明确检验比现有导数学习/合成微调更具体的可重复估计收益。即便上述都成立，也只支持学习贡献假设，仍需更完整的相关工作核查。

**目前结果：没有模型训练证据，没有满足上述条件，贡献差别不足，不能放行。** 若所有收益由相同模拟数据解释，或者非零目标 Tangent Prop 也获得相同收益，终止 SFRC，不宣称它是新的价格预测模型。

## 8. 检索收据、证据范围及本次验证

实际八条搜索调用的原始结构化返回存于 search_actual.json；search_receipts.json 是从该原始返回生成的摘要，不是检索工具完整日志的替代。部分配置的搜索后端失败，已有响应通过备用后端返回；两条搜索没有成功结果，保留失败，不额外超过八条预算。其余主源通过直接 URL 读取补足，直接读取不算新增搜索查询。

primary_receipts_first_attempt.json、primary_receipts.json、additional_receipts.json 保存原始成功和失败的 HTTP 状态、时间戳、字节数、SHA256 与重试。sources/ 保存实际下载的 PDF、正文提取和作者代码快照。Wiley 返回 403，CounTS README 的 main/master 路径返回 404；不能因此推断作者没有代码，也不能引用未读取的全文支持强断言。

已读取 SegRNN、N-BEATS、ETSformer、N-HiTS 的作者/维护仓库实现及 ForecastCF 作者 README；没有克隆整个仓库，没有运行它们的原始实验。主源已读全文并不等于复核所有实验结论。本次数学检查脚本只验证响应公式、非对称反例与配对误差代数，不冒充成熟底座训练结果。

## Sources

[1] https://arxiv.org/pdf/2308.11200 — segrnn_paper
[2] https://raw.githubusercontent.com/lss-1138/SegRNN/main/models/SegRNN.py — segrnn_model
[3] https://raw.githubusercontent.com/lss-1138/SegRNN/main/README.md — segrnn_readme
[4] https://arxiv.org/pdf/1905.10437 — nbeats_paper
[5] https://raw.githubusercontent.com/ServiceNow/N-BEATS/master/models/nbeats.py — nbeats_model
[6] https://arxiv.org/pdf/2202.01381 — etsformer_paper
[7] https://raw.githubusercontent.com/salesforce/ETSformer/main/README.md — etsformer_readme
[8] https://raw.githubusercontent.com/salesforce/ETSformer/main/models/etsformer/decoder.py — etsformer_decoder
[9] https://arxiv.org/pdf/2201.12886 — nhits_paper
[10] https://raw.githubusercontent.com/Nixtla/neuralforecast/main/neuralforecast/models/nhits.py — nhits_model
[11] https://arxiv.org/pdf/2310.08137 — forecastcf_paper
[12] https://raw.githubusercontent.com/zhendong3wang/counterfactual-explanations-for-forecasting/main/README.md — forecastcf_readme
[13] https://proceedings.neurips.cc/paper/1991/file/65658fde58ab3c2b6e5132a39fae7cb9-Paper.pdf — tangentprop_paper
[14] https://proceedings.mlr.press/v202/yan23d/yan23d.pdf — selfinterpret_paper
[15] https://arxiv.org/pdf/2606.18049 — contex_paper
[16] https://arxiv.org/pdf/2610.08118 — forced_system_paper
[17] https://arxiv.org/pdf/2411.00126 — causal_forecasting_paper
[18] https://repec.econ.au.dk/repec/creates/rp/13/rp13_32.pdf — gstar_paper
