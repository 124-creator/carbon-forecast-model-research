# 分解预测方向决策（2026-10-13，未训练）

## 结论
仅建议把“原始目标风险驱动的分解版本修订可靠性”作为有条件主线。不是把因果分解、uncertainty、gate拼接；要检验同一个历史观测被新增样本重新分解后，表示修订能否预示可避免的真实预测损失，并据此决定是否使用分解表示。这里“因果”仅指过去可用、无未来泄漏，不是因果效应识别。未经实验和全文差异核对，不称已成立创新，更不保证发表。

## 证据范围与门禁
仓库根目录无AGENTS.md/CLAUDE.md；读取上级E:/碳价格/CLAUDE.md及本库README/START_HERE。上级旧项目已封盘，本库独立研究，不改旧项目。
实际执行19条搜索，原始返回（含失败）在ACTUAL_SEARCH_RECEIPTS、ROUND2、ROUND3；PRIMARY_SEARCH_EVIDENCE仅机械摘出主源网址搜索段落。16次正文抽取均被工具private/internal-address安全拒绝：不绕过，不经代理，不将搜索长摘要冒充全文。PRIMARY_01—14部分URL是待核实候选/历史线索，不是身份或访问成功证据；PRIMARY_15—16同样未通过正文访问。先前聊天URL、CaStNet不纳入认证源。
本轮有主源主机搜索摘录，但没有独立核读方法全文，近邻排除不完整。确切首次在线日期、期刊卷期与预印本版本未独立认证；搜索显示的“小时前”不作为出版日期。vintage-aware首轮失败后换词/再次检索，有结果不等于该精确命名无先例。

## 直接近邻（只写实际摘录支持的内容）
- A consistent framework for decomposition-based time series forecasting：递归分解、训练/推理任务一致、泄漏与边界问题已经是直接研究对象。滚动分解本身不能称创新。[1]
- Decompose-learn-revise（作者单位主源）：padding-free分解、边界无失真监督、选择性学习、相似记忆检索式预测revision。它直接覆盖“可靠性+边界+选择学习+修订”模块组合。其prediction revision不能未经全文核对等同于历史分量vintage修订。[2]
- ADNet（主源arXiv搜索摘要）：频率软分配给互补分量、独立预测分支、与预测目标联合学习，直接占据任务驱动分解。[3]
- AdaptiveFrequencyNet（出版社主源摘录）：可学习时域多尺度滤波组、分支softmax、跨尺度融合。换成轻量滤波器或attention并不消除这一近邻。[4]
- FoReco官方文档：跨截面/时间最小二乘一致性投影、误差协方差与收缩已有实现。简单和约束、协方差融合并非新机制。[5]
- VINTAGE-TS：观察时间与信息时间分离，历史revision channels，首次与固定延迟值联合预测。属于原始统计数据版本，不是算法分解分量；它使“revision-aware forecasting”泛称不可用作新颖性。[6]
- Revision Risk in Real-Time Macroeconomic Forecasting：早期误差、修订风险和协方差的MSE分解已有邻近理论。不能把风险恒等式本身当创新。[7]

特别核查boundary uncertainty：精确词返回不少水文空间边界/物理外部驱动不确定性，不等于一维信号分解端点。真正直接的端点近邻是[1][2]及搜索返回2021 adaptive ICEEMDAN-ICA。后者出版社搜索仅指出适应边界和避免未来信息，具体算法未读，须纳入后续必要基线，不声称被排除。adaptive decomposition forecasting是成熟拥挤领域，不能将adaptive等同于新贡献。

## 三条路线的项目比较
|方向|可检验差异（假设，不是成果）|成本与风险|裁决|
|---|---|---|---|
|版本修订风险学习|固定分解器，保存历史as-of版本；只用已成熟的版本差异估计“分解加入相对原始目标基线的损失增量”，而非重构误差或未来分量误差；检验风险信号的增量决策价值|无需新外生/价格下载；CPU多次短窗口重算，低维回归/收缩；需保存版本并对齐成分；模式交换和边界修订可能只反映波动|唯一有条件主线|
|任务驱动分解|若只是分解参数跟随预测loss优化，已被ADNet/AFNet直接覆盖；必须给出小样本可识别的约束或估计论贡献|小样本950/451协议下，滤波器与预测器联学增加自由度；更多种子、容量匹配与调参；GPU可选但工程较高|不作为主线|
|跨分量/步长约束|只对真实线性聚合关系施加约束；任意不同h价格不是求和层级。若变化累计路径满足求和，普通投影/广义岭已是近邻|低维CPU可行，但K×H残差协方差受成熟样本限制；高维全协方差不稳；固定二次惩罚需证明非Tikhonov等价|不作为主线，最多公平基线|

计算预算是后续设计上限而非本轮实测：固定2—3分量，窗口只预定2个，最多3个成熟修订年龄，不做分解器/元启发式/网络全网格。滚动前缀全重算可能总成本随样本长度近二次增长，优先固定短窗口；缓存仅保存末端必要版本。今天没有CPU耗时、内存或训练估计实测，不报假分钟数。

## 主线的数学对象与时序
记d_s^[t]=D(x_≤t)在s位置的分量，r_(s,k,a)=d_(s,k)^[s+a]−d_(s,k)^[s]。只在s+a≤t时能作为历史观测。固定年龄a的版本不是最终真分量；禁止全样本最终分解当监督。当前分解的不确定性只能由历史已成熟修订规律预测，不得读取它自己的未来修订。
候选风险rho_(t,h)=E[(x_(t+h)−f_dec)^2−(x_(t+h)−f_raw)^2 | 截止t可得修订统计]；只在历史s+h≤t且所需修订年龄已成熟时更新。预测可采用有限集合的“启用/收缩分解增量”决策；gate不是创新，能否在同信息/容量/支持预算下估准增量风险才是问题。
分量之间修订常互相抵消，不能只加各分量方差；应保留联合修订或直接对预测增量的修订统计建模。若不同版本K变化、频率排序交换，先固定分解器与频带定义或宣布对象不可比，不事后挑匹配。

## 必须区别于近邻的结果与否决门
1. 首先全文核对[1][2]是否已经使用分解版本三角、成熟修订统计及相对原始目标风险监督；若三者已有同构方法，则撤回方法新颖性，只可考虑协议审计/负结果贡献。
2. 对照原始强MLE（提供上下文0.949793）、F交互（0.956715）、RW、固定因果分解、递归一致分解、固定收缩、普通已成熟残差gate、只波动率/边界距离gate、修订统计gate。上述数值只是既有项目上下文，今天未重算；不把F当强于MLE。
3. 成熟支持与调参预算完全相同。修订打乱/无修订控制、未来扰动不变性、分量对齐失败控制是机制鉴别；没有这些不能归因为revision。
4. 验证风险排序/条件校准是否有增量，同时以原始价格目标MSE为主要判决，分量重构分数、置信区间变窄不能替代收益。相对强MLE无稳定改善或收益由波动率解释即停止，不转头堆attention。
5. 冻结2020—2021开发协议；2022 StageC已见，不能重新包装盲测，且本任务禁止读取2022价格。本轮不读任何价格、不训练、不下载数据。后续如要发表须预注册新的未见期或独立目标，另获授权；不能从现有两个开发年宣称外部普适性。

## 当前不能承诺的事
本轮已经足以否决“滚动分解就是新”“联合优化分解就是新”“增加和约束就是新”的浅层论证；不足以认证唯一候选全球首次。数据版本与算法表示版本、预测revision与分量revision必须分别命名。有限样本可能让修订风险不可预测，也可能伤害强基线；失败完全允许。

## Sources

[1] https://www.sciencedirect.com/science/article/abs/pii/S0045790625007165 — A consistent framework for decomposition-based time series forecasting
[2] https://scholar.xjtlu.edu.cn/en/publications/decompose-learn-revise-a-reliability-aware-framework-for-robust-t — Decompose-learn-revise: A reliability-aware framework for robust time series forecasting
[3] https://arxiv.org/abs/2609.25777 — Disentangling Heterogeneous Traffic Dynamics for Multi-Step Traffic Forecasting via Adaptive Spectral Decomposition
[4] https://link.springer.com/article/10.1038/s41598-026-56619-x — AdaptiveFrequencyNet: a long-term time series forecasting approach based on learnable frequency decomposition and cross-scale attention fusion
[5] https://search.r-project.org/CRAN/refmans/FoReco/html/octrec.html — R: Optimal combination cross-temporal forecast reconciliation
[6] https://arxiv.org/html/2609.28576 — Time-Series Foundation ModelsThat Understand Data Revisions
[7] https://arxiv.org/abs/2607.05882 — Revision Risk in Real-Time Macroeconomic Forecasting
