# 局部基函数直接多步候选审查

## 结论与证据级别

本轮只保留一个待核候选：**成熟标签覆盖约束下的局部样条复杂度分配**。这是研究假设，不是新模型首创声明，不是已经满足创新标准的方案。没有训练、读取价格、加载权重、提交或推送。旧 Koopman 实现没有修改。

重要阻塞：本轮全部论文全文提取返回 `Blocked: URL targets a private or internal network address`。保留 PRIMARY_01—08 与 PUBLISHER_ACTUAL 原始返回，没有通过代理、镜像或浏览器绕过。PUBLISHER_ACTUAL 中两个事先未核身份的链接仅是失败请求，不能用来确认论文身份。真正完成的是作者/维护方 GitHub 的静态源码和许可证核对，不是出版方全文精读。SEARCH_ACTUAL 包含真实搜索返回及失败；不把搜索摘要当全文。WEB_RUN_SEARCH_RECEIPT 是撤回记录，不是可引用证据。候选查新尚未完成，特别是自适应样条、局部回归、非平稳多任务复杂度分配近邻，不能宣布不存在先例。

## 主源已核对的已有机制

- N-BEATS 作者仓库 `models/nbeats.py`：MLP 输出 backcast/forecast 系数，残差逐块相减，预测逐块相加；TrendBasis 使用多项式，SeasonalityBasis 使用正余弦，GenericBasis 直接切分系数。局部分解与神经基展开已经存在，不能把它们改名当贡献。[14]
- N-HiTS 的 Nixtla 当前维护实现：输入池化、MLP 系数、低维预测结点插值，并从最后观察值开始累计块预测。源码明确直接预测而非递归；默认插值为线性，也提供 nearest/cubic。这里只核对维护方实现，不把它等同于原论文实验逐字复现。[5]
- 原始 KAN 作者 pykan：样条函数位于网络边；网格由样本分位点与均匀网格混合更新，随后重拟合样条系数以保持已有函数。自适应结点和样条激活不是新贡献。[8]
- TimeKAN 官方实现使用 Chebyshev 多项式层，不是把所有 KAN 都实现成 B-spline。模型含平均池化、多频率分解/插值、不同阶数 KAN 和混合；“分频后分别分配表达能力”已经是直接近邻。[13][16]
- RevIN 作者实现分别移除输入实例均值/标准差，再在输出恢复，统计量 detach。局部归一化不是贡献，也不能声称处理了所有条件分布变化。[10]
- Non-stationary Transformers 作者实现以原始窗口和均值/标准差学习 tau/delta，向注意力传入，再恢复输出尺度。以状态调节表达能力并不是无人研究的概念；本轮没有执行其训练代码。[20]

## 仅一个任务定制假设（自行提出，尚未证明新颖）

假设：小样本直接多步学习中，不同预测步的成熟标签对某个局部状态区间的覆盖相差很大。与全预测步共用固定样条复杂度相比，**使用各步已成熟样本对局部样条支撑的有效覆盖，收缩或开放该支撑上的非线性自由度**，可能减少长预测步稀疏状态的方差，同时保留短预测步的非线性。机制作用于原始多步预测函数的估计，不是在旧模型预测误差上追加校准。

### 可实施定义

固定有限状态特征 z_s，例如原始滞后变化的两个预先定义聚合量。特征和局部中心/尺度均只能从发行时 s 的输入窗口构造；本轮不计算这些量。令 y_{s,h}=(p_{s+h}-p_s)/sigma_s，I_{t,h}={s:s+h<=t}。冻结的结点 K 在初始开发前缀确定，此后不使用任何未来样本更新结点。用低阶 B-spline B_{j,k}(z_{s,j})，固定 horizon basis q_r(h/H)，定义：

f_h(z_s)=a_h + b_h^T z_s + sum_{j,k,r} theta_{jkr} B_{j,k}(z_{s,j}) q_r(h/H)。

p_hat_{t+h}=p_t+sigma_t f_h(z_t)。这里 p_t 是目标参数化的平移锚点，不是另一个训练好的主干；所有参数共同通过真实成熟目标估计。

输入 B-spline 特征须在初始前缀对常数/线性空间投影中心化，以区分 b 与 theta。预测步基 q 和输入结点都固定，小样本第一实现不增加深层 KAN 或神经生成结点。

w_{t,s}=rho^(t-s)。对每个支撑与预测步，定义 c_{s,jk}=w_{t,s} B_{j,k}(z_{s,j})^2，n_eff(t,h,jk)=(sum_{s in I} c_s)^2/(epsilon+sum_{s in I} c_s^2)。当 sum c_s=0 时，n_eff=0。局部系数 v_{h,jk}=sum_r theta_{jkr}q_r(h/H)。目标函数：

J_t=sum_h alpha_h [sum_{s in I_{t,h}} w_{t,s}(y_{s,h}-f_h(z_s))^2 / sum_{s in I_{t,h}} w_{t,s}] + lambda sum_{h,j,k} [(n0+n_eff(t,h,jk))/(epsilon+n_eff(t,h,jk))] v_{h,jk}^2 + lambda_s sum_{j,r} ||D2 theta_{j,:,r}||^2 + lambda_l sum_h ||b_h||^2。

空 I 的预测步禁止估计并记为不可用。alpha_h 固定且归一化。n0、epsilon、rho、lambda、lambda_s、lambda_l 预先限定，不能在已反复观察的评估段搜索。这个目标在特征固定时是带数据依赖对角惩罚的凸二次问题，可用增广最小二乘精确求解。它不是新函数类；潜在贡献最多是覆盖依赖的估计约束及其在异步成熟多步任务中的作用。不能把普通 generalized ridge 包装成新的分解定理。

### 直接近邻与贡献界限

最接近的已核部件分别是 pykan 样本自适应网格、N-HiTS 预测步低维插值、TimeKAN 不同频段不同阶数，以及 RevIN/非平稳 Transformer 的输入实例适配。[8][5][13]本候选不宣称其中任何部件原创。

统计近邻还必须核对：局部有效样本量驱动的正则化、adaptive ridge、P-spline/GAM、varying-coefficient 多任务模型、在线非线性样条，以及 delayed-label forecasting。搜索已经出现相关样条非平稳工作，但全文没有通过，本轮不填写未经全文确认的模型细节。若已有同一覆盖惩罚或等价形式，只能成为任务验证而不是方法创新。

## 强对照和可证伪条件（仅设计，不执行）

1. 随机游走、已有低维 MLE 和已知线性校正作为固定历史成绩门槛；旧校正不是候选的一部分。不重新把开发过的 451 段称为未见测试。
2. 完全相同样条、相同 horizon basis、相同 rho 的普通 ridge/P-spline：去掉 n_eff 因子。这是核心机制消融。
3. 每个 h 一个常数惩罚，与按 n_eff 分配但总惩罚强度匹配的对照，区分整体正则加强与局部覆盖作用。
4. 保留各步样本数量而打乱局部支撑覆盖分配，检验覆盖位置是否真的有效；打乱仅发生在允许的训练支撑，不涉及未来标签。
5. 使用所有 h 完全成熟的交集训练，与逐步成熟训练比较，同时另设相同支撑对照，不能把更多短步标签解释成机制贡献。
6. GAM/低阶样条、同预算小 MLP、压缩 N-BEATS、压缩 N-HiTS、小 KAN、TimeKAN 对照。先报告实际参数数目、计算与搜索预算，而非只称“规模相同”。线性子模型和非线性部分各自消融。
7. 年份分组、每 h 损失与原始单位价格误差；保持原任务完整评估权重。不得仅选择某一个表现良好的 h。
8. 合成机制先检验：固定非线性加状态覆盖漂移、仅噪声无条件均值信号、线性真值与不漂移对照。验证凸解独立重算、未来标签扰动不影响当前参数、同前缀重放一致。此次没有生成或训练合成数据。

拒绝机制的条件：相同特征的普通 ridge 已解释增益；按局部覆盖分配无优于按 h 常数分配；仅短步增益掩盖长步恶化；无信号也因搜索制造收益。失败实现可以终止，但不能把该候选失败泛化成所有局部可学习模型都失败。

## 许可、数据可用性与执行边界

N-BEATS 作者代码是 CC BY-NC 4.0，明确商业使用须另许可且不授予专利许可，不能作为无条件 Apache/MIT 代码重分发。[2]Nixtla neuralforecast、TimeKAN 根目录 LICENSE 为 Apache-2.0。[4][12]pykan 与 RevIN 为 MIT；RevIN 文件名是 `License` 而不是 `LICENSE`，初次 404 已保留，后通过目录树核对真实文件。[7][15]Nonstationary_Transformers 根目录为 MIT。[19]各仓库依赖和借用部件仍需单独核验，不由根许可证自动覆盖。

TimeKAN README 指向 Autoformer 数据来源；这仅证明提供入口，不证明数据下载可用、样本小或数据可再分发。[11]其 data_loader 使用整文件 pd.read_csv，随后才切分和拟合训练 scaler；因此绝不能直接用于“不读取 2022 边界”的本任务。[17]公开 benchmark 入口与广州价格的许可是两件事。广州样本可用性、950/451 与历史得分来自上级任务说明，本轮没有打开文件核验，也不引入外部价格。当前不需要下载任何数据或权重。

## 后续准入

本轮完成源码静态审查和可实现假设定义，**出版方全文核对未完成，查新也未完成**。在全文合法可访问、近邻等价形式核对并通过贡献界限审查之前，不应进入价格训练。下一步是补全合法主源全文与覆盖自适应样条近邻，不是修补旧 Koopman，也不是立即跑更多模型。

## Sources

[2] https://api.github.com/repos/ElementAI/N-BEATS/contents/LICENSE?ref=c746a4f13ffc957487e0c3279b182c3030836053
[4] https://api.github.com/repos/Nixtla/neuralforecast/contents/LICENSE?ref=dd2c92d42ea40237743cc58e92603bb8b3838fd2
[5] https://api.github.com/repos/Nixtla/neuralforecast/contents/neuralforecast/models/nhits.py?ref=dd2c92d42ea40237743cc58e92603bb8b3838fd2
[7] https://api.github.com/repos/KindXiaoming/pykan/contents/LICENSE?ref=ecde4ec3274d3bef1ad737479cf126aed38ab530
[8] https://api.github.com/repos/KindXiaoming/pykan/contents/kan/KANLayer.py?ref=ecde4ec3274d3bef1ad737479cf126aed38ab530
[10] https://api.github.com/repos/ts-kim/RevIN/contents/RevIN.py?ref=fee40bc6c87cb536d048bcf1c14c4ed644b875e1
[11] https://api.github.com/repos/huangst21/TimeKAN/contents/README.md?ref=3a7c366a9e8547fd8840c5d27f25ee3e30615e33
[12] https://api.github.com/repos/huangst21/TimeKAN/contents/LICENSE?ref=3a7c366a9e8547fd8840c5d27f25ee3e30615e33
[13] https://api.github.com/repos/huangst21/TimeKAN/contents/models/TimeKAN.py?ref=3a7c366a9e8547fd8840c5d27f25ee3e30615e33
[14] https://api.github.com/repos/ElementAI/N-BEATS/contents/models/nbeats.py?ref=c746a4f13ffc957487e0c3279b182c3030836053
[15] https://api.github.com/repos/ts-kim/RevIN/contents/License?ref=fee40bc6c87cb536d048bcf1c14c4ed644b875e1
[16] https://api.github.com/repos/huangst21/TimeKAN/contents/layers/ChebyKANLayer.py?ref=3a7c366a9e8547fd8840c5d27f25ee3e30615e33
[17] https://api.github.com/repos/huangst21/TimeKAN/contents/data_provider/data_loader.py?ref=3a7c366a9e8547fd8840c5d27f25ee3e30615e33
[19] https://api.github.com/repos/thuml/Nonstationary_Transformers/contents/LICENSE?ref=c4ec40675d11d50b3d9923657f408d0db6f90f56
[20] https://api.github.com/repos/thuml/Nonstationary_Transformers/contents/ns_models/ns_Transformer.py?ref=c4ec40675d11d50b3d9923657f408d0db6f90f56
