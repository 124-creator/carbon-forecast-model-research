# Auto-Lambda 主近邻、具体不足与本轮创新边界

核查日期：2026-10-18。范围：一个完整学习方法主线；其他工作仅作直接边界核对。本文没有读取市场 CSV、预测 NPZ、模型检查点或确认留出，没有运行市场拟合。

## 1. 结论

本轮最贴切的已发表主近邻是 **Auto-Lambda: Disentangling Dynamic Task Relationships**，Shikun Liu、Stephen James、Andrew J. Davison、Edward Johns，TMLR 2022。它已经通过主任务的元损失学习辅助任务权重，理想内层变量明确同时包含共享表征和各任务头。因此，**“按目标任务适配后的风险决定迁移”不构成超出其一般双层目标的新目标**。作者实际算法只作一步联合虚拟更新，没有单独把目标头充分优化；这一近似在错配的任务头和重叠标签下存在具体风险。[作者论文](https://arxiv.org/pdf/2202.03091v2)、[作者项目页](https://shikun.io/projects/auto-lambda)、[作者仓库](https://github.com/lorenmt/auto-lambda)。

把目标头换成可微精确岭回归，是已知变量消去与双层学习的组合。本轮保留它作为强对照。待验证差异只限定为：**共享特征改变后，在固定目标支持集的预测空间内继承旧二次先验，在新出现的正交残差方向补充惩罚，再用该头的目标元风险学习源权重**。这仍使用标准最小二乘、二次型拉回、岭回归及元梯度，不认证新数学或已成立的原创贡献。

该规则有重要反面边界：沿特征子空间变化一圈回到相同终点，先验可能已经改变；收益可能仅来自正则强度漂移。必须用固定岭头 profile、初始有效自由度配平的协方差先验、可学习对角先验及常规预测基线区分。本文交付时，方法证据仅为合成数学/接口检查，尚无由本文执行的市场性能证据。

## 2. 版本、访问与阅读层级

| 对象 | 精确定位 | 实际读取程度 |
|---|---|---|
| 正式发表身份 | [OpenReview](https://openreview.net/pdf?id=KKeCMim5VN)；作者仓库引用块也列 TMLR 2022 | 正式入口出现浏览器验证页；未取得该入口 PDF，未继续重试 |
| 作者公开稿 | [arXiv v2](https://arxiv.org/pdf/2202.03091v2)，2022-06-02；DOI `10.48550/arXiv.2202.03091` | 下载成功。完整读方法 §3–4、分析 §7.1–7.3、限制 §8、训练附录 A，并读相关背景/引用；没有声称逐页完整审核所有实验附录 |
| 作者说明 | [项目页](https://shikun.io/projects/auto-lambda) | 已读算法说明与边界讨论 |
| 实现 | 固定 commit `bdba6365953f7ebb3a8ae33fabfecd3f933ecc27` | 已读 `auto_lambda.py` 全部、两个训练器的配对和优化流程、README、utils 的任务损失接口及 LICENSE |
| 本轮运行 | 本方解析反例、独立学习器合成检查 | **没有导入或执行 Auto-Lambda 作者软件**。新学习器使用的另一个作者数值核是父任务单独核验的 R2D2 岭回归方法，须分别记账 |

OpenReview 实际转向 `https://openreview.net/challenge?redirect=/pdf?id%3DKKeCMim5VN`。这是被观察到的验证阻断，不是论文不存在。另一个工具读取 raw/PDF 返回 `DisabledError`；它是工具能力错误，不是服务器 HTTP 403。之后按正常网络权限取得的是独立作者 arXiv 版本和公开 GitHub 文件，均记录 HTTP 200；没有重试被验证阻断的 OpenReview 地址，也不声称 arXiv 文件与正式版本逐字节一致。

作者代码 commit 的时间是 `2023-02-05T01:47:58Z`，tree 为 `5cb9979ef35d10b86f4b4b10ff77938021d1e325`。论文 v2 文件 SHA256 为 `97ddca9f16cc575b20e18be7e30318819e781643c8f9016e1e9fab777a070bc7`。逐文件 URL、字节数和 SHA 见 [AUTOLAMBDA_ACCESS.json](AUTOLAMBDA_ACCESS.json)，公共归纳见 [AUTOLAMBDA_SOURCE_MANIFEST.json](AUTOLAMBDA_SOURCE_MANIFEST.json)。其中原本地 `reference/` 路径只记录阅读归档，不指向本包内文件。

## 3. 作者数学与代码的精确区别

设任务集合为 \(1,\ldots,K\)，主任务集合为 \(P\)，共享参数及所有任务头合并为 \(\theta\)。作者 Eq. (3) 是

\[
\min_{\lambda}\sum_{i\in P}L_i^{\mathrm{meta}}(\theta^*(\lambda)),
\qquad
\theta^*(\lambda)=\arg\min_\theta\sum_i\lambda_iL_i^{\mathrm{train}}(\theta).
\tag{1}
\]

实际 Eq. (4)–(6) 交替进行

\[
\theta'=\theta-\alpha\nabla_\theta\sum_i\lambda_iL_i^{\mathrm{train}}(\theta),
\quad
\lambda^+=\lambda-\beta\nabla_\lambda L_P^{\mathrm{meta}}(\theta'),
\quad
\theta^+=\theta-\alpha\nabla_\theta\sum_i\lambda_i^+L_i^{\mathrm{train}}(\theta).
\tag{2}
\]

对于这一单步普通梯度更新，且元损失不直接依赖权重，精确超梯度为

\[
\partial_{\lambda_i}L_P^{\mathrm{meta}}(\theta')
=-\alpha\left\langle
\nabla_\theta L_i^{\mathrm{train}}(\theta),
\nabla_{\theta'}L_P^{\mathrm{meta}}(\theta')
\right\rangle.
\tag{3}
\]

作者以有限差分 Hessian-vector product 近似混合导数。代码里的 `virtual_step` 更新所有参数，因此它不是“只更新共享层而永远固定错误头”的方法；但也没有在虚拟共享层上把目标头重新求到最优。论文 §4 的多次采样指每次抽取部分任务来减少内存，不能据此声称已实现多步内层头收敛。[方法正文与附录](https://arxiv.org/pdf/2202.03091v2)、[固定核心](https://github.com/lorenmt/auto-lambda/blob/bdba6365953f7ebb3a8ae33fabfecd3f933ecc27/auto_lambda.py)。

本轮代码核对还确认以下具体语义：作者权重初始值为 0.1，核心没有 softmax、非负投影或权重上界；虚拟步是含可用 momentum buffer 与 weight decay 的 SGD 形式，不能直接把给它传入 Adam 的操作称为精确 Adam 虚拟步。有限差分扰动尺度为元梯度范数的倒数比例，核心没有零范数分支。`model_fit` 依赖作者 `utils.compute_loss` 中的任务类型，碳价回归需要明确适配。作者 README 写 PyTorch 1.7；本轮没有声称原完整训练驱动已在当前环境兼容执行。[固定训练器](https://github.com/lorenmt/auto-lambda/blob/bdba6365953f7ebb3a8ae33fabfecd3f933ecc27/trainer_cifar.py)、[固定 README](https://github.com/lorenmt/auto-lambda/blob/bdba6365953f7ebb3a8ae33fabfecd3f933ecc27/README.md)。

## 4. 对小样本多期限迁移最具体的两个不足

### 4.1 头尚未适配，可以改变辅助任务超梯度的符号

取独立、零均值、单位方差的 \(x_1,x_2\)，目标 \(y=x_2\)，特征 \(\phi_c=x_1+cx_2\)。平方风险采用 \(1/2\) 约定。固定错误头 \(\beta=-1\) 时，

\[
R(c,-1)=\tfrac12[1+(c+1)^2],\qquad \partial_cR=c+1.
\tag{4}
\]

精确重新拟合头则得到

\[
\beta^*(c)=\frac{c}{1+c^2},\qquad
R(c,\beta^*)=\frac{1}{2(1+c^2)},\qquad
\frac{dR}{dc}=-\frac{c}{(1+c^2)^2}.
\tag{5}
\]

当 \(c>0\) 时，同一个增大 \(c\) 的源方向，对错误头有害而对充分适配头有益。本方脚本还把主任务头自己的单步更新纳入虚拟步骤：取 \(c=.5,\beta=-1,\alpha=.05\)，源任务 \(y_A=x_1+2x_2\)，可以得到相反符号的单步与精确 profile 超梯度；无岭和岭强度 0.1 两种情况均以精确总体矩和有限差分验证。见 `check_transfer_counterexamples.py` 和 `TRANSFER_COUNTEREXAMPLES.json`。

这个反例只否定不足适配的近似，不否定式 (1)，也不证明本轮从正确精确头温启动的 head1/head5 必然失效。稳定多步 GD 若完整求导并趋于收敛，会接近同一个 profile 解。精确 profile 因而必须作为已有强方法组合，而不能直接命名为原创迁移机制。

### 4.2 时间窗口重叠会把共同噪声误读成正迁移证据

作者正文和两个训练器使用同一训练集的不同随机 batch 作为训练/元 batch，而不是永久分开的目标元集。这在其任务设置中是论文明确的策略；直接套到累计价格变化窗口会产生不同的信息关系。[论文 §4](https://arxiv.org/pdf/2202.03091v2)。

例如纯随机游走，两个相邻起点的 20 步累计标签各除以 \(\sqrt{20}\)，其协方差为 \(19/20\)。在标量预测器零点，虚拟更新 \(\theta'=\alpha\lambda y_S\) 后，元平方风险的期望超梯度为

\[
\mathbb E\,\partial_\lambda L_Q
=\alpha^2\lambda\operatorname{Var}(y_S)
-\alpha\operatorname{Cov}(y_S,y_Q).
\tag{6}
\]

第二项可以奖励纯共同噪声。标签支持不重叠时，该共同未来增量项消失；但这只是必要的时间边界控制，并非新的学习算法，也不意味着所有潜在时序依赖都被消除。脚本的 14 项解析检查全部通过，没有市场取样或优化器拟合。

## 5. 本轮候选的可辩护差异与标准等价

在目标固定支持特征 \(X\) 和提议后的 \(X'\) 上，设

\[
A=X^+X',\quad E=(I-P_X)X',\quad
\Lambda'=A^\top\Lambda A+\lambda_0E^\top E/n,
\quad P_X=XX^+.
\tag{7}
\]

这里 \(X\) 满列秩，\(X^+=(X^\top X)^{-1}X^\top\)，初始 \(\Lambda_0=\lambda_0I\)。新头使用 \(\Lambda'\) 精确求解；元梯度通过 \(A,E,\Lambda'\) 和新头，实参更新后同步保存实际提议的 \(\Lambda\)。

精确可以主张的是：若更新只是 \(X'=XA\) 的可逆坐标改变，且元/验证特征也服从同一改变，则 \(\Lambda'=A^\top\Lambda A\)、\(W'=A^{-1}W\)，预测保持不变。固定每次重置 \(\lambda_0 I\) 不具有这个性质。**不过，初始先验本身仍在既定初始特征坐标定义；任意改基后重新初始化为 \(\lambda_0 I\) 并不等价。**

式 (7) 也等价于标准二次形式

\[
\Lambda'=X'^\top M_XX',\qquad
M_X=X^{+\top}\Lambda X^+
+\frac{\lambda_0}{n}(I-P_X).
\tag{8}
\]

因此不能把二次型拉回、最小二乘投影、正交残差或可微岭回归分别宣布为新理论。如果 \(\Lambda=\mu X^\top X/n\)，且残差系数也取同一 \(\mu\)，则运输严格化为 \(\Lambda'=\mu X'^\top X'/n\)。本轮协方差对照用初始有效自由度配平得到 \(\mu\)，一般不等于运输残差系数 0.1，所以两臂不是逐步相同实现，但该等价特例必须承认。

此外，若一维等范数特征的协方差为 \(c^2\)，先旋转角度 \(\alpha\) 再回到原特征，则

\[
\Lambda_{\mathrm{back}}
=\Lambda_0\cos^4\alpha
+\lambda_0c^2(1-\cos^4\alpha).
\tag{9}
\]

终点特征相同，精度却可能改变。对 \(\Lambda_0=\lambda_0\) 且 \(c^2<1\)，这表现为惩罚减弱。把这种正则漂移带来的收益称为新增可迁移信息是不充分的。方程 (8)–(9) 在此作为代数推导；完成学习器既定合成检查后，本轮已停止扩大测试，没有为这两个新表述再运行脚本或追加市场拟合。

## 6. 必须保留的对照和证伪条件

| 对照/检查 | 对应问题 |
|---|---|
| local_profile 与 uniform_profile | 共享源信息是否有必要；仅加更多源样本是否已经足够 |
| 精确 profile + 源权重元学习 | 不把可微局部头及已有辅助权重学习的组合误作运输贡献 |
| head1/head5 | 区分头适配近似与先验规则；明确它们是本方匹配控制，不是作者 Auto-Lambda 原样复现 |
| transport_uniform | 分离先验规则与权重元学习的作用 |
| 初始有效自由度配平的 covariance_profile | 排除预测空间各向同性二次先验这个直接等价近邻 |
| diagonal_profile | 排除普通可学习岭先验/正则强度改善；不将其计为新原创模型 |
| 常规本地/共享网络与可调岭回归 | 方法需先达到有意义的预测水平，不能只赢弱化对照 |

若运输仅在一两个种子或单一小支持预算获利、相对精确 profile 的差异在确认集消失、协方差/对角先验解释了全部收益，或多次出现秩/正定性失败，应停止正向创新主张。坐标不变性合成通过只说明实现满足所给代数，不能证明市场可预测性。固定 meta 块虽没有时间穿越，仍会因数百次自适应访问被过拟合；验证块只能选择检查点，最终测试不能反向改权重、支持或协议。

旧 Stage O 已包含可微岭读出与前向块风险、预测作用上的二次优化；本轮独立历史审计也明确：仅把有限源提议重新 profile 后再作标量二次组合，主要仍是标准 stacking/约束回归，不足作为新的主机制。该候选没有沿那条已停止路线重新命名。[本轮历史审计](../HISTORY_AND_UNTESTED_NEED_ZH.md)。

## 7. 有界查新边界，不扩展完整候选池

以下只读一手摘要/作者页面，未据其作精确公式等价断言，也没有下载或执行这些作者实现。

| 工作 | 阅读层级与直接边界 |
|---|---|
| [AuxiLearn, ICLR 2021](https://arxiv.org/abs/2007.02693)；[作者页面](https://avivnavon.github.io/AuxiLearn/) | 一手摘要/作者页面。隐式求导学习辅助目标和小数据动机已经存在；不能声称所有既有辅助学习只有一步梯度对齐 |
| [A Challenge in Reweighting Data with Bilevel Optimization](https://arxiv.org/abs/2310.17386) | 2023 作者预印本摘要。对样本重加权的共同演化动力学作警示；没有把其结论直接当作本轮任务权重方法的定理 |
| [Self-Supervised Auxiliary Task Discovery for Stable Reinforcement Learning in Stock Trading](https://arxiv.org/abs/2608.15841) | 2026 作者预印本摘要；未核实正式发表。金融交易中的辅助任务元梯度思想并非空白；其强化学习设置不同，未读全文，不声称实现等价 |

R2D2 的完整近邻核查和 MIT 数值核执行由父任务独立完成；这里不把它包装成本轮新原创模块。OL-AUX 仅完成初始一手书目信息定位，未再作为第二个完整方法展开。

## 8. 许可证与公共交付

Auto-Lambda 的仓库 LICENSE 是 Imperial 的特定协议，核心不能按 README 中的通用开源印象处理为 MIT/BSD。文本限制在非商业内部或学术研究范围，并含用户、转让/再许可和第三方使用条款；另有针对 BSD 元件的单独部分，但本次核心文件未见相应 BSD 标头。协议里还保留了另一软件名称的模板文字，不能据此扩大许可。[固定 LICENSE](https://github.com/lorenmt/auto-lambda/blob/bdba6365953f7ebb3a8ae33fabfecd3f933ecc27/LICENSE)。

因此本轮只留来源身份、访问事实和本方说明；**作者源码、LICENSE 全文及论文 PDF/提取文本均不进入本轮公共源码包**。本文没有把这种许可状态写成“作者不允许发表任何研究”，也没有实际运行该作者软件。本方 `learning.py` 为独立公式实现；使用 R2D2 数值核的许可及范围单独记录，不能混同为 Auto-Lambda 作者复现。

公共 allowlist：本文件、`AUTOLAMBDA_SOURCE_MANIFEST.json`、`AUTOLAMBDA_ACCESS.json`、`check_transfer_counterexamples.py`、`TRANSFER_COUNTEREXAMPLES.json`。访问过程的历史 API JSON 可保留为出处审计附件，`reference/` 不发布。
