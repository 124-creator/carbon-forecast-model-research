# 预测导向共同信号：否决泛化创新，保留一个待检验估计目标

日期：2026-10-14。范围：纯设计、先验检索；未读取市场数据、未训练、未提交或 push。
结论：novelty_gate=BLOCKED；不得作为新算法放行。潜在共同因子、异质载荷、测量误差、直接多步、Kalman 条件均值/协方差分别不是创新；组合也不足以证明创新。下面唯一保留项是任务定制估计目标，不是原创性声明。

## 实际检索证据与访问失败（不报告全文成功）

严格只使用6次搜索，原始返回保存在 search_ledger.json。搜索原生后端不可用，实际由 keyless rescue 返回；第4次失败、第6次为空；MCP 不在可用工具中，单渠道降级。

搜索实际返回的主源摘要：
- Stock & Watson，Macroeconomic Forecasting Using Diffusion Indexes：作者 PDF 搜索摘要明确 PCA/approximate dynamic factor 和6/12/24个月预测。候选 https://princeton.edu/~mwatson/papers/Stock_Watson_JBES_2002.pdf 。只确认搜索摘要，不声称全文。
- Hemati、Rowley、Deem、Cattafesta，De-biasing the dynamic mode decomposition：作者机构 PDF 搜索返回含联合 snapshot TLS/EIV 去偏方法。主源候选 https://dept.aem.umn.edu/~./faculty/mhemati/documents/HeRoDeCaTCFD2017.pdf 。只确认搜索返回的方法片段，不声称阅读全文。
- Dawson、Hemati、Williams、Rowley：搜索返回作者机构 PDF 摘要，列出已知噪声、forward-backward、TLS 三种修正。候选 https://dept.aem.umn.edu/~./faculty/mhemati/documents/DaHeWiRoExIF2016.pdf 。全文未验证。
- Doz、Giannone、Reichlin：机构主源搜索摘要明确 PCA/OLS 后 Kalman smoother 的两阶段 DFM。候选 https://iris.luiss.it/handle/11385/93991 。全文未验证，预测部署不得用包含未来数据的 smoother。

待确认而非已接受来源：Burnham/MacGregor/Viveros 查询没有准确主论文返回；Serena Ng 查询失败；Rauhala 查询空结果。PLS/RRR/EIV 等价风险作为数学比较任务保留，不伪造论文精读佐证。预设 PDF、Rauhala 页、JMLR 候选亦尝试读取，结果均失败，见 source_access.json；没有核准作者、页数、DOI 或章节，不能引用为已读文献。

### 作者 GitHub 实际访问结果
对 https://github.com/cwrowley 、https://github.com/cwrowley/dmdtools 、https://github.com/cwrowley/dmd 和 python 子目录已发起实际读取，全部因 “Blocked: URL targets a private or internal network address” 被拒绝。作者归属、库存在性、代码、许可证未确认；不得写 success 或404。该拒绝属于工具安全边界，停止，不使用 curl/browser 绕过。待可合法访问时优先核验作者主页互链、精确 commit、TLS-DMD 实现，而不是凭库名认定作者代码。

此次全15个直接来源访问均收到相同拒绝。source_access.json 保存每个原始 error。因此全文门未满足，结论只允许 BLOCKED；没有任何来源被标注完整精读成功。搜索片段足够提出既有方法警告，不足以确立原创或证明完全等价。

## 只保留一个数学目标（待检验、可能完全等价于 EIV-RRR）

对每个截至 t 可见的跨源历史向量 x_t∈R^p，以及同时堆叠所有目标和步长的未来向量 y_t∈R^q（q=M×H），只允许在训练段中使用未来标签。u_t 是无误差的自身历史/截距控制。输入 x_t=x*_t+e_t，E[e_t|x*_t,u_t]=0，Cov(e_t)=R_t，且 Cov(e_t,y_t)=0。R_t 必须来自独立重复观测/外部可靠性校准，不得由价格波动、预测残差或缺失率直接解释成测量误差。存在跨时相关时 R_t 必须覆盖历史堆叠块；只有单源观测且没有识别约束，R_t 不可识别，停止 EIV 分支。

唯一目标，令 B=WV，rank(B)≤r，D 为训练段确定并固定的正定目标权重：

J(A,B)= n^-1 Σ_t [(y_t−A^T u_t−B^T x_t)^T D (y_t−A^T u_t−B^T x_t) − tr(D B^T R_t B)] + λ tr(D B^T B).

共同预测分量 z_t=W^T x_t 的学习目标是未来损失，不是 min||X−ZΛ^T||² 的观测重构；但这与 predictive RRR 同族，预测监督同 PLS 已有思想。异质市场只是 y 的多列、D 的权重和载荷区别，不是额外新模块。R=0、λ=0 时严格退化为普通多响应 RRR；R=0 时为 ridge-RRR；正定已知 R 下为 corrected-moment EIV-RRR。因子 W 的旋转不识别，只比较 B/预测，不宣传可解释的唯一潜在信号。标量响应时 rank 限制基本不发挥降秩作用，需要堆叠多目标多步。

推导：在独立零均值经典测量误差条件下，E[观测平方损失]比无噪声输入损失多 tr(D B^T R B)，因此减去该项。该目标估计的是 clean-input 回归系数，不自动优化 noisy-input 部署预测风险。直接用 B^T x 可能扩大部署 MSE，这是主要反证条件；不能隐藏。可另以已有 Gaussian 条件均值 m_t=E[x*_t|x_{≤t}] 部署 B^T m_t，但这是已有滤波/收缩，不算第二个创新，必须单独消融，且预测条件均值与真实状态 oracle 严格区分。

## 可执行的估计步骤（next coder 按此实现，当前没有执行模型）

1. 在训练段中心化；用训练段 u 对 X,Y 做投影残差化，记 Xr,Yr。残差化改变输入误差，必须同步用投影后的平均误差协方差 Rr；首轮合成单元测试仅设置截距，以免忽略此项。
2. G=Xr^T Xr/n−Rr+λI，C=Xr^T Yr/n。要求 G 正定，最小特征值>1e−8×max(1,最大特征值)。不满足则当前 λ 候选 invalid；禁止静默裁剪 G 后仍声称同一目标。λ 由前向验证选取。
3. T=G^(-1/2) C D^(1/2)，对 T 做截断 r 阶 SVD 得 T_r；B=G^(-1/2) T_r D^(-1/2)。回代 A 拟合自身历史控制。记录 rank、条件数和 objective。
4. 验证部署 noisy x 与已知模型条件均值 m 两条已有输入处理方式；选择只能依验证损失，测试段封存。
5. 此闭式解本身是广义低秩二次优化，不可包装为新算法。只有先完成 state-space/EIV/RRR/PLS 全文、代码近邻核验，证明实际结构差异且胜过强近邻，才重新讨论贡献。当前状态始终 BLOCKED。

## 全文门与否决规则

来源获取成功≠完整阅读≠等价性排除。全部直接全文和作者 GitHub 读取被安全边界拒绝；6次检索存在失败与空返回，不能保证穷尽；全文门未满足，不放行。即便测试有改善，也只能暂称任务定制估计器的经验表现。已有等价式对应明确的部分（RRR 退化、TLS/EIV 噪声修正、监督因子）直接否决创新命名；整体是否与具体异方差 EIV-RRR 全等仍待全文核验，不能虚构最终证明。
