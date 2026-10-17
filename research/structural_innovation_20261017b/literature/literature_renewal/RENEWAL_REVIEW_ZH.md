# 重复净报价与 renewal 路线：有界查新、作者核心与数学边界

日期：2026-10-17。范围：一个完整主方案 DeepRenewal，加上直接解决“时间—mark 依赖”“金融变价事件”“特征函数评分”的近邻。本文没有读取报价、确认集或旧模型权重，没有市场拟合，也没有改变上一轮研究。

## 1. 结论与当前决定

**目前不支持把“年龄 × 上次变化符号状态、半 Markov 持有/跳转、signed reward 特征函数传播、跨期限 CRPS 训练”单独作为原创算法。** 给定有限状态与转移 reward 分布，其特征函数矩阵乘法就是标准 Markov additive / reward 计算。年龄扩状态、神经网络参数化和对矩阵乘法求导没有消除这一等价。CRPS 的特征函数积分也已有明确原文；最近的神经概率预测工作已直接使用特征函数训练。[S4–S6]

仍有一个可以被真实实验否定的**应用结构假设**：在报价净变化稀疏、变化方向或幅度确实依赖已知等待年龄的市场，用固定当前上下文加有限状态的过程约束，可能比自由的多期限分布头更节省样本。它与 DeepRenewal 的未来事件驱动 LSTM 确有模型类差异，但属于更强的有限状态假设；不是对 DeepRenewal 的通用表达能力升级，也不是新创 renewal、hurdle、特征函数或 proper scoring 原理。

当前决定分两层：**原创算法主张 No-go；应用诊断有条件保留。** 若普通同状态 Markov / hurdle 模型就能覆盖全部收益，不能把神经包装称为机制创新。若只有 CRPS 改善、均值 MSE 没有改善，应只报告分布预测结果，不能改写原点预测失败的结论。本轮没有授权或执行这项应用诊断。

## 2. 数据语义先于事件模型

令实际记录为日期与报价 $(d_t,p_t)$，$r_t=p_t-p_{t-1}$，并定义

$$
e_t=\mathbf 1\{r_t\ne0\}.
\tag{1}
$$

式（1）只识别**观察到的净报价变化**。$e_t=0$ 不识别没有交易、没有信息、没有日内价格波动或没有中间变化。例如潜在路径“没有跳变”与“先 +1、后 −1”给出相同端点报价。也不能把没有记录的一天补成已观察零变化。

如果预测目标仍是第 $h=1,5,20$ 个后续**观测**的报价，年龄首先应以观测步计数。已知过去日期间隔可以进入条件变量；未来记录日期在发行时未知，则不能使用真实未来间隔。仅凭稀疏端点数据拟合连续时间交易强度没有相同的观察语义。DeepRenewal 的连续时间节明确针对带粒度时间戳的单个事件；2026 modRHawkes 使用的金融变化事件也不同于本任务端点记录。[S1, S3]

事件概率可预测还不够支持点 MSE 优势。若 $J$ 是下一事件的 signed reward，且在每个可达状态下

$$
\mathbb E[J\mid e=1,\mathcal F,\text{state}]=0,
\tag{2}
$$

则即使等待危险率高度依赖年龄，未来累计变化的条件均值仍为零，RW 仍是相应平方损失的 Bayes 预测。年龄和事件概率此时可能改善方差、零质量和尾概率。相应纯合成枚举已通过，见 `RENEWAL_SYNTHETIC.json`；它是逻辑反例，不是市场结论。

## 3. 唯一完整主方案及作者源码

主方案是 Türkmen、Januschowski、Wang、Cemgil 的 *Forecasting intermittent and sparse time series: A unified probabilistic framework via deep renewal processes*，PLOS ONE 16(11):e0259764，2021-11-29，DOI 10.1371/journal.pone.0259764。[S1]

原文将间歇需求表示为等待区间与正需求量，用共享 LSTM 表示事件历史并产生下一组分布参数，允许跨序列学习。论文自己的 self-modulating DTRP 已被说明为 Markov renewal 实例。其强项是零需求与非零事件的完整概率建模，而非本任务 signed 价格变化的原生支持。本次阅读了问题定义、主要 renewal 扩展、§4.3–4.4 和相关实现说明；没有逐条重证全部附录或复现作者 benchmark。

### 3.1 精确来源与执行范围

论文指向的官方实现是 `awslabs/gluonts`，并非第三方同名复现。固定当前提交：

```text
repository https://github.com/awslabs/gluonts
commit     d60556ca71167a3ec1ea807b491dd982d1de2112
commit UTC 2026-10-13T12:42:28Z
tree       80252f53301922cb7a325aaef0464dd83b556ea1
license    Apache-2.0
```

这只是**作者指定仓库当前固定快照**，未证明与 2021 年论文实验代码逐字相同。四个 renewal 核心文件、两份直接变换实现、`pyproject.toml`、`LICENSE`、`NOTICE` 已按原字节保存于 `author_gluonts/`。九个文件全部通过 Git blob SHA 与本地 SHA256 校验，清单在 `AUTHOR_SOURCE_MANIFEST.json`。[C1]

| 文件 | 本次直接核查内容 |
|---|---|
| `_estimator.py` | 默认分布、数据变换、训练字段和预测入口 |
| `_network.py` | LSTM、似然、未来递归采样、首等待条件 |
| `_predictor.py` | 区间—大小样本向观察步序列的还原 |
| `_transform.py` | 输出变换与相关接口 |
| `transform/convert.py`、`feature.py` | 非零观测间距、丢首区间、末尾零计数 |

源码是 MXNet 实现；本轮没有安装 MXNet、导入实际作者类或运行其训练。当前包描述没有给这份旧 MXNet 核心一个已验证的独立可运行环境。保存的九个文件也不是完整运行包。后续若需要真实作者执行，必须另行冻结兼容环境与完整依赖；PyTorch 转写应称本方移植，并明确不是原类执行。

### 3.2 直接从核心得到的限制

**支持域。** 默认 interval 与 size 都用 `NegativeBinomialOutput`，训练的是 $Q-1$、$M-1$。因此原 size 是正整数需求量。把负价格增量直接塞入这两个分布没有合法的目标支持；用绝对增量训练则丢失方向。增加符号分布是必要适配，但普通 signed hurdle / marked renewal 已能表达这一变化，不能据此称原创。

**同一未来事件的依赖。** 核心似然是

$$
\log p(Q_i-1\mid h_i)+\log p(M_i-1\mid h_i).
\tag{3}
$$

虽然隐藏状态可使用过去 interval 与 size，同一事件的两个随机输出在给定该状态后仍按独立分布采样。共享历史不等于同一事件的条件依赖。让方向或幅度依赖本次等待时长确实改变这份源码的结构，但条件 time–mark 依赖本身已是直接近邻的研究对象；CIKM 2022 Eq5–8 已明确给出按 mark 条件化 time 的实现。[S2]

**窗口与未结束等待。** 核心先丢弃不完整的首区间，只从 past 中的已完成 interval–size 对计算训练似然。`time_remaining` 供预测使用，不是该训练损失字段。因此当前未结束零串没有单独的右删失生存项。补删失似然属于标准观察模型修正，不能当创新；也不能在没有分析抽样机制和边界条件时断言此实现所有估计必然有偏。

**未来状态与计算。** 每次采样下一事件后，核心把采样 interval、size 追加到序列，再推进 LSTM。其未来历史依赖比有限年龄/符号状态更丰富；代价是样本路径推理。换为固定有限状态允许矩阵传播，是模型类限制换取计算便利，不是同模型无损消去所有未来 LSTM 随机性。

### 3.3 首等待采样的可复核边界偏差

当前固定 `_network.py` 用末尾零观测数量 $a$ 条件化第一等待。训练分布变量是 $X=Q-1\ge0$，正确剩余等待应为

$$
W=Q-a=X+1-a,\qquad Q>a\iff X\ge a.
\tag{4}
$$

原分支保留满足 $X>a$ 的样本，最多重复 50 次；如果没有成功，强制 $X=a$，因此 $W=1$。这与式（4）不同。取 $X$ 为成功率 $p$、支持 $0,1,\ldots$ 的几何分布，则正确 $P(W=1)=p$；原分支给出

$$
P_{\rm branch}(W=1)=F_X(a)^{50},\qquad
\mathbb E_{\rm branch}W=1+\frac{1-F_X(a)^{50}}{p}.
\tag{5}
$$

在 $p=0.5$、有效过去非空时：

| 已观察末尾零数 $a$ | 正确 $P(W=1)$ | 当前分支 $P(W=1)$ | 正确均值 | 当前分支均值 |
|---:|---:|---:|---:|---:|
| 0 | 0.5 | $8.8818\times10^{-16}$ | 2 | 3.0000 |
| 5 | 0.5 | 0.455018 | 2 | 2.089964 |
| 20 | 0.5 | 0.999976 | 2 | 1.000048 |

这是**固定源码的分支语义与概率代数核验**，不是运行原 MXNet 类得到的输出。已独立核对目标偏移、`CountTrailingZeros` 定义、50 次循环及最后减 $a$，收据完整保留。它支持“后续执行前应处理并分开报告该边界问题”，不支持新模型创新或原论文整体失效的断言。

不能只把 `>` 改成 `>=`：原初始化为 0，$a=0$ 时会把没有抽样的初值当合法样本。需合法采样标志、无效哨兵或正确条件分布采样。修正版必须另存并与未修改作者核心分开，不能以修复有缺陷基线产生的优势作为新模型收益。

## 4. 所提有限状态模型的完整定义与已知等价

以下给出**候选应用结构**以便精确判定，而非授权实现或命名一个新模型。

在预测起点 $t$，由允许的过去报价与日期得到上下文 $c_t$。状态 $z=(a,s)$ 包括 capped 净变化等待年龄 $a\in\{0,\ldots,A\}$ 与上次非零变化符号 $s\in\{-1,+1\}$；没有历史事件时需额外初始状态，不能删除全零窗口。若年龄从比网络输入更长的历史中计算，所有对照都必须得到相同的信息。年龄上限后的自循环意味着尾部危险率不再随年龄增长，这是可检验假设。

令 $q_\theta(z,c_t)$ 为下一观测非零概率，$\pi_{\theta,k}(z,c_t)$ 为该事件符号 $k\in\{-1,+1\}$ 的概率。令非零幅度 $B>0$ 具有合法特征函数，reward 为 $J=kB$。可以用有有限二阶矩的 Gamma 等分布；分布选择本身没有新颖性。持有转移为 $(a,s)\to(\min(a+1,A),s)$，reward 为 0；变化转移为 $(a,s)\to(0,k)$。

定义 reward 特征函数转移矩阵

$$
K_\theta(u;c_t)_{zz'}
=P_\theta(z'\mid z,c_t)
\mathbb E_\theta[e^{iuJ}\mid z,z',c_t].
\tag{6}
$$

如果持有与变化抵达同一状态，应将对应概率乘特征函数的贡献相加。起点行分布为 $v_t$，则

$$
\phi_{t,h}(u)=v_tK_\theta(u;c_t)^h\mathbf 1,
\quad h=1,\ldots,20.
\tag{7}
$$

式（7）由对下一状态与 reward 逐步取条件期望直接得到，是标准 Markov additive 传播；时间非齐次时改成矩阵连乘，性质不变。设 $P=K(0)$，$r(z)=\mathbb E[J\mid z,c_t]$，均值为

$$
\widehat\Delta p_{t,h}=v_t\sum_{j=0}^{h-1}P^jr.
\tag{8}
$$

本方纯合成用另一条“状态、累计 reward 显式概率枚举”路径核验了式（7）–（8），最大特征函数差 $4.44\times10^{-16}$，最大均值差 $1.11\times10^{-16}$。这证明测试实例中的数学等价，不是新颖性证据。

### 4.1 能主张的相容性范围

同一个起点、同一个 $c_t$ 下，上述过程给所有期限一个共同随机路径模型。因此不仅有边缘分布，累计变量的联合特征函数也可写成

$$
\mathbb E\exp\!\left(i\sum_{h=1}^{H}u_h\Delta p_{t,h}\right)
=v_t\prod_{j=1}^{H}
K_\theta\!\left(\sum_{h=j}^{H}u_h;c_t\right)\mathbf 1,
\tag{9}
$$

乘积按时间从左到右。**只可以称同起点的过程相容性。** 下一真实起点若重新编码上下文，任意 encoder 不自动满足跨起点条件分布的 tower 一致性。仅用逐期限边缘 CRPS 也不能识别真实跨期限 copula；模型给出的联合结构来自假设，不能说已被边缘评分证明。

有限状态的代价同样具体：在固定起点上下文下，两条未来事件路径 $(+1,+1)$ 与 $(-1,+1)$ 都抵达“年龄 0、上次符号 +”，若真实下一 mark 取决于倒数第二个未来符号，则两个条件均值相反，本状态无法区分。未来事件驱动 LSTM 可以保留这种信息。这是条件过程类的反例，不是说任何有限边缘集合都不可能偶然匹配。

### 4.2 CRPS 与数值边界

有限一阶矩下，已知恒等式为

$$
\operatorname{CRPS}(F,y)
=\mathbb E|X-y|-\tfrac12\mathbb E|X-X'|
=\frac1\pi\int_0^\infty
\frac{|\phi_F(u)-e^{iuy}|^2}{u^2}\,du.
\tag{10}
$$

Jordan 2016 Eq6.6 / Eq6.22 明确给出这一形式，并讨论数值积分；它不是本方推导的新评分。[S5] 对模型的特征函数传播可以精确到浮点运算；一般 CRPS 积分、CDF 反演或 FFT 仍有截断、求积和混叠误差。不能把“无需 Monte Carlo”写成“无近似误差”。

有限频率集合的平方匹配更弱。在网格 $u_k=k\delta$ 上，点质量 $\delta_0$ 和 $\delta_{2\pi/\delta}$ 的特征函数完全相同，真实 CRPS 对 $y=0$ 分别是 0 和 $2\pi/\delta$。因此有限网格损失不能一般性地称为对任意分布严格 proper 的 CRPS。此反例已在合成收据验证。最近 DeepLévy 已采用每期限条件分布与特征函数匹配；这里只用其方法作为查新边界，不重复其优越性或校准结论，也不把预印本称已发表成果。[S6]

### 4.3 学习目标能检验什么

可以训练同一过程的多个期限 CRPS，并报告式（8）的原价点 MSE。若附加点损失，应给所有对照相同目标与预训练支持。一个普通的风险对齐参考为

$$
\mathcal L_{\rm point}
=\frac1{|\mathcal M|}\sum_m\frac1{|\mathcal H|}
\sum_{h\in\mathcal H}\frac1{n_m}
\sum_{t\in I_m^{\rm train}}
\frac{(\widehat\Delta p_{t,h}-\Delta p_{t,h})^2}
{D_{m,h}^{\rm train}},
\qquad
D_{m,h}^{\rm train}=\frac1{n_m}\sum_{t\in I_m^{\rm train}}(\Delta p_{t,h})^2.
\tag{11}
$$

分母只能由训练支持估计，零分母规则须先固定。CRPS 的货币尺度也应使用固定训练尺度控制跨市场权重。式（11）是普通损失加权基线，不是 innovation。它有助于排除旧式逐起点局部波动标准化带来的训练/评价权重差异；这一控制要求来自本轮诊断意见，本次未重新读取旧数值。

## 5. 最小对照与否证条件

若后续另行授权真实实验，最少需要以下三组强对照；所有模型使用同样的已知日期、年龄、市场身份、训练范围与成熟标签支持。

| 对照 | 要排除的替代解释 |
|---|---|
| 同年龄/符号状态、简单参数化的 signed Markov / hurdle 模型 | 收益全部来自标准有限状态结构，网络不是必要组成 |
| 直接概率 GRU，分布头或分位数头接受相同评分目标；另含普通风险对齐点 GRU | 收益来自概率任务或损失权重改变，并非 renewal 约束 |
| 事件驱动 RNN 的 signed renewal 适配，明确修改支持域、首等待、条件 time–mark 依赖 | 对 DeepRenewal 的限制有针对性，未利用不合法负值或有边界错误的弱基线 |

RW 应同时保留点基准及清楚定义的概率基准，如只在训练支持拟合的 signed increment 分布；不能只把退化的点 RW 当唯一概率竞争者。同一有限状态模型“Monte Carlo 路径 vs 特征函数传播”只能做数值等价与算力对照，不能作为不同统计模型的创新消融。

需要在读取数据前固定的可证伪判断包括：

1. 在 signed mark 条件均值为零、只有 hazard 改变的合成情形，不预期赢 RW 点 MSE；如果报告显著赢，先核验信息边界、选择偏差或统计误差。
2. 在年龄影响下一 mark 的符号/幅度、有限状态假设正确且事件样本少的情形，约束模型可能有分布样本效率优势；必须相对同状态参数模型和相同目标 GRU 验证。
3. 当真实下一 mark 依赖状态外的近期未来事件或幅度时，固定闭合模型应出现可测错设代价。
4. 若真实数据几乎没有零变化、年龄分组有效事件不足，或跨期符号关系反转，不能以不同状态格的稀疏拟合宣称机制稳定。

没有读取报价，因此这里没有报告零比例、事件数或市场适配性结论。最低数据条件不是某个通用“行数门槛”：必须有足够的**有效非零变化**覆盖两种方向与关键年龄，训练之外还要有独立时期覆盖。应报告状态占用、有效事件数和相应不确定性。单纯增加许多重叠零窗口不会按相同倍数增加方向学习的信息。

## 6. 能写与不能写的贡献句

在之后通过严格实证的前提下，可以写：

> 本研究检验一个固定上下文、有限年龄/方向状态的 signed renewal-reward 结构，在稀疏净报价的多步概率预测中，是否相对于同信息和同目标的直接模型获得样本效率收益；特征函数用于该已定义过程的确定性传播。

当前没有实证支持，不能写成已获得收益。不能声称“首次 renewal 价格模型”“首个 joint time–mark 模型”“首个特征函数概率预测”“首创精确 CRPS”“证明跨起点一致性”或“观测零代表没有交易”。也不能把作者采样边界修复、通用删失似然、归一化或普通风险对齐当核心原创。

**最终建议：本轮不为这个组合命名新的原创模型，不直接启动完整市场网格。** 如主任务选择概率预测应用研究，它可作为一个明确、可否证的结构基线候选；尚未发现足以支撑原创算法主张的额外学习规则缺口。

## 来源与阅读程度

- **[S1] 主完整方案。** Türkmen et al., PLOS ONE 2021, DOI [10.1371/journal.pone.0259764](https://doi.org/10.1371/journal.pone.0259764)。[正式全文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764)。已读定义与相关方法节、实现说明；作者指定核心全文读；未复现 benchmark。
- **[S2] 直接结构边界。** Waghmare et al., CIKM 2022, DOI [10.1145/3511808.3557399](https://doi.org/10.1145/3511808.3557399)。[作者全文 v2](https://arxiv.org/html/2210.15294v2)，已读 §3 Eq3–8 与作者代码链接。页面 2024 修订日期不是会议发表年份。[作者仓库](https://github.com/waghmaregovind/joint_tpp)仅定位，未下载、未核版本/许可、未执行。
- **[S3] 2026 直接金融边界。** Han, Stindl, Chen, *Disentangling endogeneity and exogeneity in cryptocurrency markets using the modulated renewal Hawkes process with marks*, Annals of the Institute of Statistical Mathematics，2026-04-29，DOI [10.1007/s10463-026-00976-x](https://doi.org/10.1007/s10463-026-00976-x)。[出版全文](https://link.springer.com/article/10.1007/s10463-026-00976-x)，已读 §1–2 强度定义及讨论。其 renewal age 是潜在 immigrant 的年龄，不是简单观察最后一跳；未读完整推断证明、未执行 R 程序。补充 ZIP 可能含数据，本轮未下载。
- **[S4] 历史数学边界。** McLean and Neuts, *The Integral of a Step Function Defined on a Semi-Markov Process*, SIAM Journal on Applied Mathematics 15(3):726–737，1967，DOI [10.1137/0115062](https://doi.org/10.1137/0115062)。[官方摘要](https://epubs.siam.org/doi/abs/10.1137/0115062?journalCode=smjmap)说明 semi-Markov reward 积分的矩阵变换。只读摘要，未访问付费全文；式（6）–（9）是本方直接条件期望推导，未冒称逐字取自未读全文。
- **[S5] CRPS 特征函数原文。** Alexander Jordan, *Facets of forecast evaluation*, KIT dissertation，2016，DOI [10.5445/IR/1000063629](https://doi.org/10.5445/IR/1000063629)。[机构全文](https://publikationen.bibliothek.kit.edu/1000063629/3972338)。已读 §6.1 Eq6.6 及 §6.2 Eq6.22，PDF 第94、102页附近；未通读整部论文。
- **[S6] 最近训练边界。** Yang, Yin, Xue, Salim, *DeepLévy: Learning Heavy-Tailed Uncertainty in Highly Volatile Time Series*, [arXiv:2605.10364v3](https://arxiv.org/html/2605.10364v3)，2026-05-14，**预印本**。已读摘要、§3、§4.2 及部分数值附录；未验证全部理论、代码、许可证或 benchmark。只支持“神经多期限 CF 学习已有直接方案”，不引用其性能为事实。
- **[C1] 作者代码。** [固定提交](https://github.com/awslabs/gluonts/tree/d60556ca71167a3ec1ea807b491dd982d1de2112/src/gluonts/mx/model/renewal)。精确文件链接、Git blob 和 SHA256 见 `AUTHOR_SOURCE_MANIFEST.json`。Apache-2.0 与 NOTICE 已保存。

访问错误与未读层级见 `SOURCE_READ_LOG.json`。本报告不把检索未发现解释为文献中不存在，也不保证本轮有界检索穷尽 2026 年所有研究。

## 2026-10-17 后续范围决定

在上述查新结论返回之后，主任务选择另行检验点预测结构候选 SMP（Signed Markov Propagation），保持价格 MSE 为主指标，并去掉 CF、CRPS 与 Monte Carlo 方案。该决定不把标准 Markov reward 递推升级为新数学。模型代码将在全新 `structural_innovation_20261017b` 目录实现，市场实验须由主任务另行冻结与执行；本阅读记录仍是 0 市场拟合、0 确认读取。本报告的 No-go 针对未经验证的原创算法主张，不妨碍清楚标注的有条件结构实验。
