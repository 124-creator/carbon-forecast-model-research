# 唯一设计：端点净变化—有符号桥接细节 NHITS

状态：DESIGN_REVIEW_ONLY / NOT_APPROVED_FOR_TRAINING。2026-10-14。没有读取价格数据，没有训练。本稿只提出一个编码与重构改造；不提交首创声明。父设计审核通过后才可讨论候选训练。

## 先查来源，再写设计
祖先 E:/碳价格/AGENTS.md、CLAUDE.md 已读取；新目录没有另外的祖先规则。官方原文先实际获取：sources/nhits.py（Nixtla commit 5372a33ae745a1b3f481827444315bb792253939）；首次出版社 PDF 失败，随后 arXiv 2201.12886 成功。已阅读此前本仓库 nhits_paper.txt 的方法正文，以及本次成功下载 PDF。source_receipts.json 保存 URL、失败与成功、SHA256；源码是静态阅读，不是本次执行的训练版本，也没有声称与父任务已冻结 wheel 相同。

官方 NHITSBlock lines 102–124 使用 ceil(input_size/k) 的输入宽度，pool stride=k、ceil_mode=True；lines 151–155 对整段 insample_y 池化；lines 181–183 MLP 输出 backcast/forecast；IdentityBasis 的 forecast knots 经 F.interpolate 恢复 H。官方已有 MaxPool1d 和 AvgPool1d，Max→Avg 不是创新。原论文 §3 把池化大小与输出插值表达率搭配，不能把每个分量称为已识别频带。后续 block 残差和未池化 k=1 分支可能保留信息，因此以下反例证明的是单个池化接口损失，不证明整个官方模型不可能区分。

## MaxPool 的精确问题及实际反例
输入增量 x_i=p_i-p_{i-1}。MaxPool 并非总抹掉符号：全负窗口输出仍为负；但只保留最大值，混合冲击时无法恢复负冲击大小、净变化与先后次序。对独立、等概率 ±1 的长度4窗口，枚举16种输入得到 E[max]=0.875、E[avg]=0。这是统计上非零位置偏移，不是未来收益偏差定理。

synthetic_counterexamples.py 已运行：
- [2,-2] 与 [2,2] 都输出 max=2；净变化分别0和4。Avg与signed extrema可区分。
- [2,-2,0,0] 与 [0,0,-2,2] 的 max=2、avg=0、固定正负极值=(2,-2)完全相同；桥接面积分别+0.5与-0.5，固定正弦核摘要分别+0.5857864376269051与-0.5857864376269051。
- [-3,-1] 与 [-1,-1] 的 max 均-1，净变化-4与-2。故不能说 Max 总删除负号。
这些输入显示历史回撤/反弹的存在与顺序，不证明均值回复、更不证明可预测性。净变化也不能直接解释为永久冲击。

## 完整编码：持久证据与反转证据分开，但不假设已识别永久/暂时
固定 L=64、J=3、k=2^J=8；仅使用发行时已观察增量。未来 H 长度独立按同样规则分块，末块使用实际长度，不复制伪增量；不使用未来数据补边界。历史块从发行端向左对齐，只取完整块；L=64没有余块。

每个历史块 x=(x_1,...,x_k)，定义
S_i=Σ_{a=1}^i x_a，m=S_k/k，B_i=S_i-i m，B_0=B_k=0。
x_i=m+(B_i-B_{i-1})。
这是精确、固定线性坐标变换：m保存块净变化，B保存相对端点直线的有符号路径。将 B_{1:k-1} 投影到固定离散正弦基
q_r(i)=sqrt(2/k) sin(π r i/k)，r=1,...,k-1，c_r=Σ_{i=1}^{k-1}q_r(i)B_i。
B_i=Σ_r q_r(i)c_r；故无丢失的编码 E(x)=(m,c_1,...,c_{k-1})。
按 r 组 {1},{2,3},{4,5,6,7} 把桥接细节送入粗、中、细三种预测分量；净变化通道进入粗尺度。所有 r 保留，不能宣称降低输入总信息维度。固定signed extrema e+=(max(0,max x)), e-=min(0,min x) 仅用于强对照，不额外加入提案。

预测：沿时间块排列 m 和每组 c_r；每组使用官方 NHITS 的 MLP / doubly residual block 与插值头，三个固定 n_freq_downsample=(4,2,1)，n_pool_kernel_size=(4,2,1)，不学习knots。对每个分支 a，r_a^(0)=其历史坐标序列；第b块
θ_ab=MLP_ab(Pool_ab(r_a^(b-1)))，(back_ab,f_ab)=IdentityBasis_ab(θ_ab)，r_a^(b)=r_a^(b-1)-back_ab，forecast_a=Σ_b f_ab。
提案分支内部使用AvgPool，保留对照完全相同的AvgPool分支。每个r各输出未来块的系数；沿未来块序号插值，而不是直接对价格插值。m分支预测未来块平均增量；c_r分支预测桥接系数。

对未来块实际长度 n（完整块 n=k；末块 n=H mod k），同样 q_r^(n)，需r=1,...,n-1；只保留相应前n-1个输出坐标，使用n长度正弦基，不把k基截断冒充n基。得到
Bhat_i=Σ_{r=1}^{n-1} q_r^(n)(i) chat_r，Bhat_0=Bhat_n=0，
xhat_i=mhat+Bhat_i-Bhat_{i-1}；
phat_(t+h)=p_t+Σ_{u=1}^h xhat_u。
块末累计变化恰为 n mhat；桥接通道只影响块内路径，不改变块末水平。这是硬结构约束，不是附加惩罚或事后残差校准。H=1时没有桥接分量，退化为净变化预测；不能要求短端桥接增益。
训练若以后获批，只使用重构价格变化的原始多h损失 Σ_{s,h:mature} w_h (phat_(s+h)-p_(s+h))²，不使用未来分解特征输入或人为永久冲击监督。不训练本稿。

## 与近邻的确切关系及不强提交理由
1. PN-Max（作者站点2013 ICCE primary）已保留正负最大值；固定signed extrema是既有思想。它仍对顺序置换不变；本稿桥接系数不同，但不主张signed pooling本身新颖。
2. Cui et al. CVPR2017 Kernel Pooling（primary PDF已存）是高阶交互/核显式近似，与这里固定位置正弦投影不是同一“核”；不要混用名称。本稿 sin 加权桥接摘要属于普通固定线性滤波，不是新的核学习。
3. Rippel et al. 2015 Spectral Representations（primary PDF已存）通过频域截断池化；本稿保留全部桥接频率、按组预测并端点重构。都属于固定谱表示，不能宣称发明谱池化。
4. WPMixer（作者PDF正文与author GitHub code成功获取）：原文§decomposition eq(1)、resolution branches、reconstruction；wp_decomposition.py 调用 DWT1DForward/Inverse；wp_wavelet_patch_mixer.py 的 forward 逐近似/细节分支预测再 inv_transform。与本稿最接近。区别只有：对价格增量先显式端点/桥接分离、细节硬零端点、替换branch为NHITS插值残差头，不是第一次多尺度分解-分支预测-重构。
5. TimeMixer 原文与author models/TimeMixer.py：已有多尺度下采样、趋势/季节方向混合与多预测器叠加。区别为可逆端点桥接、零块末贡献约束，而非多尺度本身。

该端点桥接可视为普通线性lifting/离散正弦表示，具有与wavelet近似/细节预测实质重合。参数类别相同不自动否决估计贡献；可实证贡献应限定为“端点净变化与块内回撤的结构性隔离能否改善有限样本NHITS估计”。本次有限检索没有排除更近的bridge/wavelet forecasting文献，不够支持原创投稿；不强提交，也不升级market training。

## 两项必要消融（以后审批才运行）
A1 去桥接信息：所有c输入置零，但保持分支、参数、梯度预算与输出重构。检查提升是否真的来自历史局部反转，而非仅多头参数。
A2 去端点路由：保留完全相同E(x)和所有参数，预测各通道后允许普通自由加法水平输出，不约束桥接块末=0；对比结构路由是否优于单纯可逆变换。报告与原式输出宽度/FLOPs不完全相等的部分，不能只用参数量替代计算公平性。

## 强baseline和门槛
同输入L、同成熟标签、同初始化种子、相同搜索/优化预算：官方NHITS Max / 官方Avg / 固定signed extrema（±极值两通道）先做最小对照；完整raw增量NHITS k=1；同维度固定Haar/DST分解 + NHITS系数预测逆重构；author WPMixer与TimeMixer；raw-lag ridge、random walk、历史rolling strong control。匹配有效参数与训练步数并另外报告FLOPs。历史冻结失败不等于rolling失败；未达到父任务公平rolling协议不得作机制优越声明。
先需要合成可学性、no-signal零预测伤害、同种变换对照、两个消融均支持机制，随后父审核明确授权才可能进行价格训练。当前数学枚举通过不等于该学习器通过。

## 来源/执行边界
最多4个搜索queries已用（signed pooling、kernel author code、spectral author code、WPMixer/TimeMixer/NHITS），这4个搜索queries实际发生在初稿写入之后；下载与实际原文review发生在初稿之前，不能声称搜索先于设计。现稿是post-search修正版，该搜索 chronology gate不通过。检索未核得spectral作者代码。PN-Max的原文Max是absolute-max稀疏编码语境，不是PyTorch MaxPool定义；仅引用正负双通道思想，不移植其符号丢失论证。Kernel/PN-Max author code未核得；WPMixer/TimeMixer author代码静态阅读已完成。WPMixer raw域两次EOF失败，正常GitHub contents API成功获取同一commit并验证Git blob；不是绕过身份或权限限制。下载脚本留下全部失败记录。所有第三方材料只保存在本地，未提交/发布。
