# 线性族独立核查：NLinear / DLinear 与成熟标签学习规则

核查日期：2026-10-17。此文仅负责一个候选族；其他文献是直接近邻查重，不是新增候选。核查人没有读取或训练市场价格。本轮执行范围：检索、论文方法核对、固定提交源码阅读、原版模型类合成输入前向/反向，以及线性代数反例。

## 裁决

**NLinear 可以作为完整、可执行的已发表预测骨干；当前不支持把“分解、末值锚定、累计增量白化、成熟标签时间衰减或多提前期成熟残差”本身认定为我们的独立方法创新。** 候选骨干通过实现门，所讨论的新学习规则尚未通过新颖性和非等价性门。本核查没有发现可在当前证据下承诺的新规则。

如果主代理选择 NLinear 进行批准数据的小型重训，应明确标为已发表基线复现／开发诊断；不能把本文合成输入 smoke check 写成市场复现，也不能把已暴露的广州 2020/2021 重新称作独立测试。

## 1. 可核对的论文和源码

主论文：

- Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu. Are Transformers Effective for Time Series Forecasting? AAAI 2023, 37(9), 11121–11128。
- DOI：https://doi.org/10.1609/aaai.v37i9.26317
- 正式论文全文：https://ojs.aaai.org/index.php/AAAI/article/download/26317/26089
- 带附录版本：https://arxiv.org/pdf/2205.13504
- 作者仓库：https://github.com/cure-lab/LTSF-Linear
- 本轮固定提交：https://github.com/cure-lab/LTSF-Linear/commit/0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6
- 提交日期：2024-01-27T16:45:10Z；本轮 GitHub API 读取到的仓库默认分支 HEAD。
- 许可证：Apache-2.0，已实际读取 LICENSE；本地保留许可证。

全文证据定位：正式版 pp.11123–11125 的线性模型、DLinear/NLinear 定义及表2；附录版本 B.1/B.2（PDF 第9页）说明数据、moving average=25、Linear 默认 L=336，而 Transformer 默认 L=96。作者方案不是只给网络名：模型、数据加载、优化器、停止规则和预测输出均有公开代码。

已实际获取和阅读的源码文件（均在上述固定 commit）：

| 文件 | 核实内容 |
|---|---|
| README.md | 作者实现身份、数据链接、训练入口、三个线性模型的职责 |
| LICENSE | Apache-2.0 |
| models/DLinear.py | 端点重复填充、25点平均、两线性支路求和 |
| models/NLinear.py | detach末值、相减、线性映射、加回 |
| exp/exp_main.py | Adam、MSELoss、验证集early stopping、输出裁剪、测试指标 |
| data_provider/data_loader.py | train-only StandardScaler、连续窗口、自定义7/1/2划分 |
| data_provider/data_factory.py | 训练/验证shuffle与drop_last、测试顺序保留 |
| run_longExp.py | 种子2021、默认epochs=10、patience=3、学习率与入口 |
| scripts/EXP-LongForecasting/Linear/exchange_rate.sh | L=336；预测96/192/336/720；batch/lr配置 |
| requirements.txt | torch==1.9.0，其他依赖未锁版本 |

这里只恢复上述10个文件，**没有下载整个仓库**，没有宣称复现原论文九个数据集。GitHub 原 blob SHA 与本地文本 SHA 分别记录在 GITHUB_BLOB_SHAS.json 和 SOURCE_PROVENANCE.json。本地文本保存时统一换行，不能把其哈希误称为原始远端字节哈希。

## 2. 完整预测结构与关键实现细节

设单变量历史窗口为 x=(p_{t-L+1},…,p_t)，输出未来 H 个价格。

NLinear 为：

$$
\widehat{\boldsymbol p}_{t+1:t+H}
=p_t\boldsymbol1_H+
W(x-p_t\boldsymbol1_L)+b .
$$

原版已包含末值锚定，不是待我们补入的创新。原版参数数为 H(L+1)，包含 nn.Linear 默认 bias；论文附录的 HL 计数忽略了 bias。单变量下 individual=True/False 的函数结构相同。

DLinear 设 M 为端点重复填充的固定平均矩阵：

$$
\widehat{\boldsymbol p}
=W_s(I-M)x+W_tMx+b_s+b_t .
$$

它仍是仿射映射。两支路参数数为 2H(L+1)。平均窗口写死为25，而不是读取运行入口同名 moving_avg 参数。平均仅使用当前输入窗口；窗口内部的居中平滑并不自动意味着使用预测起点之后的价格。

源训练脚本使用未来价格水平的 MSE。市场内标准化在训练部分估计，测试时使用同一变换；NLinear 的相减发生于模型内部。线性模型实际只接收 batch_x，不使用 batch_x_mark 或 decoder 历史。原版测试函数直接在标准化后的数据上计算指标，没有在该函数中反标准化，因此广州评价必须明确回到批准协议的价格量纲与评分口径。

有两个需要修正的实施细节：

1. 原版训练循环每个 epoch 都读取并打印 test loss，虽然 early stopping 依据 val loss。严谨新实验应删除训练阶段的测试查询。
2. 原版自定义数据按70%/10%/20%固定划分；它不是广州项目规定的逐起点信息集与刷新制度。需要保留骨干与学习机制，替换成项目已经冻结的加载和评价边界，记录这种必要适配。

## 3. 对950点广州前缀的具体不足

这是从结构和样本规模作出的判断，不是本轮市场结果。

- 可训练完整标签窗口数为 n_train−L−H+1。不能照抄作者336点历史以及720步预测。在只有950点且还需开发期和预测区的场景，这些默认设置会用尽样本或产生极少窗口。
- 一个市场无法获得原文跨变量共享权重带来的横向样本量。增加高度重叠窗口不会制造同等数量的独立冲击。
- 全局线性权重难以表达输入状态依赖的动态改变；NLinear处理的是加性水平平移，不自动处理斜率、波动、机制转变。
- 25点均值没有广州特定的已验证经济依据；它是既有结构超参数。
- 原文在金融样本上击败复杂Transformer，并不等于击败RW。正式表2，Exchange-Rate在96步的MAE为Repeat 0.196、DLinear 0.203；192步为0.289、0.293；336步为0.396、0.414。不得仅以胜过旧Transformer作为广州发表依据。
- 2020/2021已被历史选模与诊断观察；换为NLinear不会使这些年份恢复“未看过”。

强对照应包含项目的RW、统计参考、同刷新政策的强GRU/LSTM，以及原版NLinear。若最终使用DLinear为改进骨干，必须包括同预算NLinear和普通线性模型，避免把优化参数化效果写成新表达能力。

## 4. 直接近邻查重：各自验证到哪一层

| 文献 | 与所讨论学习规则的直接接近处 | 本轮证据层级 |
|---|---|---|
| Better Batch for Deep Probabilistic Time Series Forecasting，AISTATS2024 | 连续样本的动态协方差、GLS/联合Gaussian likelihood、基于已观测误差的条件校准 | 已读原始论文§4.1–4.2及公式7–12；未核其代码 |
| Multivariate Probabilistic Time Series Forecasting with Correlated Errors，NeurIPS2024 | 多步误差协方差、低秩加对角参数化、误差条件校准 | 官方发表元数据已核；实际读arXiv v4方法；正式PDF请求超时 |
| TEFL: Prediction-Residual-Guided Rolling Forecasting for Multi-Horizon Time Series，2026预印本 | 可观察的成熟误差、完整H步延迟误差向量、低秩误差adapter、联合训练 | 已读arXiv v1 §3.1–3.3；未核代码；不宣称已正式发表 |
| Beyond Uniform Updates: Drift Pattern Aware Online Time Series Forecasting Under Delayed Feedback（PADRE），IJCAI2026 | 根据当前上下文、延迟残差轨迹和模式库调节梯度更新门 | 仅官方摘要与DOI已核；未读全文，不把摘要用于细粒度优越性判断 |
| The Label Horizon Paradox: Rethinking Supervision Targets in Financial Forecasting，ICML2026 | 学习不同监督期限权重的双层优化 | 正式PMLR元数据及arXiv v3 §4公式6–14已读；未核代码 |

直接来源：

- https://proceedings.mlr.press/v238/zheng24a/zheng24a.pdf
- https://proceedings.neurips.cc/paper_files/paper/2024/hash/619b8e3ead58dce90bc615f2a7d5d102-Abstract-Conference.html
- https://arxiv.org/html/2402.01000v4
- https://arxiv.org/html/2602.22520v1
- https://www.ijcai.org/proceedings/2026/503
- https://doi.org/10.24963/ijcai.2026/503
- https://proceedings.mlr.press/v306/song26l.html
- https://arxiv.org/html/2602.03395v3

以上足以否定“此前没有人学习相关误差／只用成熟误差／学习期限权重／延迟残差控制更新”的宽泛首创表述。它们并不能证明任何细节组合都不存在；不应反过来把未搜到完全同名算法当成创新证据。

## 5. 唯一审查的改进方向：成熟标签相关性驱动的参数更新

研究问题有现实意义：长提前期标签成熟慢，多个提前期还共享市场冲击，直接将它们当独立监督可能失真。但以下五个代数事实必须先跨过。这里是独立推导，不是引用某篇论文的定理。

### 5.1 分解线性不增加函数类

$$
A=W_s(I-M)+W_tM,\qquad
\widehat{\boldsymbol p}=Ax+b.
$$

反过来对任意 A 取 W_s=W_t=A 就能实现该映射。因此DLinear相对无约束单层线性的差异在参数化、初始化、正则化和优化路径，不在扩大单变量仿射函数类。

### 5.2 固定的跨期限GLS可能与OLS完全相同

设所有输出共享满列秩设计矩阵 X，完整标签 Y，系数 B 不受约束；固定正定期限协方差 Σ。

$$
\min_B\operatorname{tr}
\big[(Y-XB)\Sigma^{-1}(Y-XB)^\top\big].
$$

驻点为 Xᵀ(XB−Y)Σ⁻¹=0，故 B=(XᵀX)⁻¹XᵀY，与OLS相同。这个结论仅覆盖上述共同设计、完整输出、无正则和固定协方差条件；加入缺失标签、不同设计、低秩约束或参数相关协方差可以打破等价，但需要重新说明贡献，不能直接借GLS之名声称改进。

### 5.3 累计噪声白化可能只是普通增量损失

对于逐步增量的累加矩阵 C，若采用 Σ=σ²CCᵀ，则

$$
e^\top\Sigma^{-1}e
=\sigma^{-2}\|C^{-1}e\|_2^2.
$$

C⁻¹就是差分算子。以此解释重叠并无问题，但这一变换本身是既有的目标重参数化。

### 5.4 指数成熟新鲜度可能只差一个常数

若起点 i、提前期 h、刷新时点 t，使用

$$
w^{mature}_{i,h}
=\exp[-(t-i-h)/\tau]
=\exp(h/\tau)\exp[-(t-i)/\tau].
$$

每个h单独且无正则拟合时，前面的期限常数会消掉。若共享参数或有固定惩罚，差异对应期限加权或相对正则强度，应与这种直接对照比较。

### 5.5 同一目标日的多提前期成熟误差只有一个新增随机冲击

令所有已发出预测均固定存档，

$$
e_{t,h}=p_t-\widehat p_{t\mid t-h}.
$$

在 F_{t-1} 中，旧预测都已知，所以

$$
\operatorname{Cov}(e_t\mid F_{t-1})
=\operatorname{Var}(p_t\mid F_{t-1})\boldsymbol1\boldsymbol1^\top,
$$

秩至多为1。同时，

$$
e_{t,h}-e_{t,k}
=\widehat p_{t\mid t-k}-\widehat p_{t\mid t-h}
$$

在新价到达前就已经确定。多维误差向量由一个新价格冲击和旧预测分歧组成，不能被当成H个独立“漂移新证据”。即使随机训练算法产生预测，在条件信息集中包含已经抽取的种子和存档权重后，结论也成立。

线性高斯模型配合二次参数漂移惩罚会回到带相关观测噪声的RLS/Kalman系数跟踪；学习更新映射则必须和在线元学习、PROCEED及PADRE比较。本文没有建立一个超出这些近邻的非等价机制。因此**暂否决把这一方向直接命名为我们的新算法**。

可否证门槛很具体：如果声称利用多个成熟提前期的新信息，必须证明在每个目标日只新增一个价格的情况下，所学映射的收益来源区别于RLS/Kalman、期限权重及共享参数约束。任一代数退化对照能复现全部预测，或改进仅由不同刷新频次/隐含正则预算导致，就应撤回新学习规则主张。若只在已暴露2020/2021上好看，也不构成独立泛化证据。

## 6. 本轮真实执行记录

本地先确认默认Python没有torch；随后复用主代理安装的 carbon_deps 中 torch 2.3.1+cpu。没有安装其他市场数据依赖，没有调用作者完整训练脚本。

执行作者原版模型类：L=13，H=10，单变量，batch=3，individual=False/True，共4个case。前向输出形状均为(3,10,1)，输出和反向梯度均有限。DLinear的仿射折叠最大误差不超过5.0×10⁻¹⁶，NLinear平移等变误差不超过1.8×10⁻¹⁵。只验证实现行为，不构成预测准确度结果。

独立随机矩阵代数核查：固定种子17017，GLS与OLS系数差1.04×10⁻¹⁵；累计白化损失差7.11×10⁻¹⁵；DLinear仿射折叠及任意A可表示的误差均在2×10⁻¹⁵内。

输出：official_class_synthetic_smoke.json、algebra_audit.json。两份结果都限定为合成验证。最终研究判断仍是：可复现骨干成立，独立模型创新未成立。

