# Deep Factors：共享与局部预测的一项完整方法核查

日期：2026-10-18。任务父提交：`1cba2eba`（PR18）。本目录只用于本轮文献、作者代码静态核查与数学边界分析；没有读取市场 CSV、旧确认分区、市场预测或模型参数，没有进行模型拟合。

## 结论

Deep Factors 已经把共享神经因子、序列载荷与局部随机过程写成完整预测模型。因此，共享骨干加局部输出头、随机效应、GP 或 LDS 本身不能作为本轮创新。当前 GluonTS 作者实现也不是可直接照搬的碳价增量骨干：它是依赖日历与 ID 的 DF-RNN，均值限制为正数，且给定训练参数后的预测并不消费接口中的历史目标值。

本轮更窄的问题——“局部输出头完成适配后，辅助源是否仍有增量价值”——值得做识别性实验，但单纯对最优输出头求隐式导数，再按目标验证损失选择辅助任务，仍属于既有双层优化原理。详见 [PROFILED_TRANSFER_BOUNDARY_ZH.md](PROFILED_TRANSFER_BOUNDARY_ZH.md)。本核查不认证新算法，也不建议为了运行作者名义模型而另外移植完整 MXNet 栈。

## 1. 唯一完整阅读方案

Yuyang Wang, Alex Smola, Danielle C. Maddix, Jan Gasthaus, Dean Foster, Tim Januschowski. **Deep Factors for Forecasting**. ICML 2019, PMLR 97:6607–6617。

- [正式出版页](https://proceedings.mlr.press/v97/wang19k.html)
- [正文 PDF](https://proceedings.mlr.press/v97/wang19k/wang19k.pdf)
- [补充材料 PDF](https://proceedings.mlr.press/v97/wang19k/wang19k-supp.pdf)
- [arXiv 1905.12417](https://arxiv.org/abs/1905.12417)
- [作者页面及 GluonTS 代码归属说明](https://dcmaddix.github.io/publications/deep_factors)

正文 §§1–6、补充材料 A/B 均已通过公开网页工具的 PDF 文本阅读；没有以仅摘要代替方法阅读。正式 PMLR 页面未给出 DOI，本记录不补造 DOI。本分工占一个完整方案；AuxiLearn 只核其作者主页和 README 的相关边界，不算已完成其论文全文或实验审查。

### 主要结构

用本文统一记号表示作者 §3 的结构：

\[
g(x;\theta)\in\mathbb R^K,\qquad
f_i(x)=w_i^\top g(x;\theta),\qquad
u_i(x)=f_i(x)+r_i(x),\qquad
z_{it}\sim p(z_{it}\mid u_i(x_{it})).
\]

共享因子是确定性 RNN；局部随机项可为异方差 Gaussian 噪声、LDS 或 GP。Gaussian GP 情形有 \(z_i\sim N(f_i,C_i)\)、\(C_i=K_i+\sigma_i^2I\)，并联合最大化各序列对数似然之和。非 Gaussian 情形通过结构化变分推断处理。正文也允许自回归输入；不能把下面这个软件版本的日历输入限制说成整个论文框架的限制。[依据：正文 §3、§4。]

以下式子是由该 Gaussian 似然直接展开的本方分析：

\[
\ell_i=\frac12(z_i-f_i)^\top C_i^{-1}(z_i-f_i)
+\frac12\log|C_i|+\frac{T_i}{2}\log(2\pi).
\]

在固定 \(C_i\) 时，共享均值的梯度含 \(J_i^\top C_i^{-1}(f_i-z_i)\)。因此，“按局部噪声协方差降低某些误差对共享表示的影响”已经存在于此似然；不能把普通 GLS、逆方差加权或局部噪声头重新命名为新学习机制。

### 证据适用范围

补充材料说明实证数据为 electricity 370、traffic 963、taxi 140、uber 114 条小时序列；每条只用一周训练，评估 24/72 小时预测，主实验采用 DF-RNN。补充 A.1 给出对角 Gaussian 局部噪声，A.2/A.3 则描述 LDS/GP；后两者不能冒充当前公共 estimator 已实现。作者还展示了一个信息不足时失效的 taxi 冷启动例子。这些是需求/使用量预测证据，没有提供本项目五序列碳价、弱增量均值或负迁移保护的保证。[依据：补充 A、B.2、Figure B.5。]

## 2. 作者源码与正文的区别

固定仓库：[awslabs/gluonts](https://github.com/awslabs/gluonts)，commit：

`d60556ca71167a3ec1ea807b491dd982d1de2112`

commit 日期为 2026-10-13。这是本轮固定读取的公共版本，**不是已证明与 2019 论文实验逐字相同的发布快照**。阅读阶段的本地归档取得原始四个 Python 文件及 Apache-2.0 LICENSE、NOTICE；四个源码 Git blob SHA-1 均与 GitHub 目录元数据核对一致。公共交付只保留下表固定链接和 [作者文件身份记录](DEEP_FACTORS_SOURCE_MANIFEST.json)，不附该阅读归档，也不需要安装 MXNet。身份记录内的 `reference/` 是当时的本地归档路径，不是本包缺失的运行依赖。

| 核查对象 | 实際代码行为与边界 |
|---|---|
| [`_estimator.py`](https://github.com/awslabs/gluonts/blob/d60556ca71167a3ec1ea807b491dd982d1de2112/src/gluonts/mx/model/deep_factor/_estimator.py) | 明确实现 DF-RNN；DF-GP/DF-LDS 留为 TODO。构造一个共享 global RNN、一个由 ID 和时间条件化的 local RNN，并非给每个市场另建一个独立噪声网络。 |
| [`_network.py`](https://github.com/awslabs/gluonts/blob/d60556ca71167a3ec1ea807b491dd982d1de2112/src/gluonts/mx/model/deep_factor/_network.py) | \(\mu_{it}=\exp(w_i^\top g_t(\text{time}))\)，\(\sigma_{it}=\operatorname{softplus}(\operatorname{RNN}_{local}([\mathrm{ID}_i,\text{time}]))\)。训练和采样实际使用 Normal。 |
| 预测输入 | `past_target` 出现在 prediction 的接口中，但函数体未用于预测计算；历史目标在训练似然中使用，不能误写成训练完全未使用价格。 |
| 声明分布 | estimator 的 `distr_output=StudentTOutput()` 被保存为属性，但创建 train/prediction network 时未传入；不能仅凭参数文档声称该固定代码运行了 Student-t。 |
| 时间信息 | transformation 从给定 frequency 构造过去与未来日历特征。对“未来第 H 条不规则报价”的日期未知问题，需要明确另行改编。 |
| [`RNNModel.py`](https://github.com/awslabs/gluonts/blob/d60556ca71167a3ec1ea807b491dd982d1de2112/src/gluonts/mx/model/deep_factor/RNNModel.py) | MXNet Gluon 的 RNN 加 Dense decoder；不是 PyTorch 核心。 |

执行层级只到：取得源码、阅读、Python AST 解析、字节哈希和 Git blob 身份核对。没有 import 作者模块、没有 forward/backward、没有 synthetic 模型执行、没有市场拟合。当前检查用 Python 环境找不到 `mxnet` module spec；这不是对所有可能环境的兼容性判定。

两次普通命令行 PMLR PDF 保存尝试均被本地网络沙箱拒绝，收据见 [DEEP_FACTORS_ACCESS.json](DEEP_FACTORS_ACCESS.json)。没有重试或绕过；正文阅读来自已成功的网页 PDF 文本。因没有取得 PDF 原字节，本目录**不提供虚构的 PDF SHA-256 或 PDF 本地副本**。

## 3. 对碳价任务可具体检验的不足

以下是依据方法结构作出的任务适配判断，不是已测出的碳价缺陷。

1. **共享不等于避免负迁移。** 联合似然可让局部大方差误差影响较小，却不保证辅助序列的表示更新在目标局部头重新适配后有正价值。固定点附近的 head 失配和表示真正损害需要分开识别。
2. **当前实现的局部均值缺少历史创新适应。** 日历与 ID 模式可能适合重复需求；碳价增量若主要接近零、偶有制度性变化，过去价格创新应否改变预测需要实际比较。给当前作者均值加自回归输入是任务改编，论文框架已经容许它，不构成新方法。
3. **小样本是两种不同问题。** 作者的“每条序列短”仍配有 114–963 条相关序列；本项目约五条异质序列不能假定拥有同等跨序列信息。EU_REPORTED 与 EEX 还不是独立市场复制。
4. **目标和时间口径必须统一。** 现货报告价与一级拍卖加权清算价应保持工具定义；日历 cutoffs 可以共用，但不能默认同日发布先后、补出未观察报价，或把观测步误叫自然日。
5. **正均值头与增量目标不兼容。** 若预测 signed \(\Delta P\)，\(\exp(\cdot)\) 直接排除了非正均值。若改为预测价格水平，再与 RW 比较，则基准和损失支持应一致，不能借这个软件差异创造被削弱的对照。

## 4. 本轮方法选择建议

保留 Deep Factors 为已发表的全局/局部近邻和源码边界证据。若根任务继续局部最优 head 后的辅助价值，应先用同一因果骨干、相同输出头解法和相同正则的对照完成有限机制检验；不必把不匹配的作者日历模型强行移植成新名称。

具体数学反例、应匹配的直接近邻、必要控制与停止条件在第二份说明中。本分工没有发现“共享＋最优 ridge head＋验证辅助权重”已经足以独立成立的新颖性；给它增加一个名称不能改变这个结论。

