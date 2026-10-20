# FITS 独立核查：第三候选，暂不启动价格训练

核查日期：2026-10-17。范围仅 FITS 的正式论文、作者实现及合成输入验证。本目录没有新增市场数据，也没有市场训练、样本外预测结果或收益声明。

## 1. 裁决

FITS 可保留为低参数频域对照与第三备用基座，当前不建议升为主方向。它有正式发表与作者源码证据，但在短回看窗、非稳定周期价格场景的适配证据不足。已找到可复现的代码适配问题和表示缺口；修复这些问题并不自动形成方法创新。

本轮不提出“低频 FITS + 当前价格 skip/anchor”的新名字，也不将普通差分、残差预测、切换原论文已有训练损失包装为新方法。FITS 是否值得再推进，须由完整原模型按项目既定协议重训后的证据决定；本轮遵守“第三候选暂不启动”。

## 2. 来源、版本与证据层级

| 项目 | 已核实内容 | 证据层级 |
|---|---|---|
| 正式题名 | FITS: Modeling Time Series with 10k Parameters | ICLR 正式 proceedings 全文 |
| 作者 | Zhijian Xu、Ailing Zeng、Qiang Xu | 论文首页 |
| 发表 | ICLR 2024；作者 README 标注 Spotlight | proceedings + 作者仓库 |
| 官方实现 | VEWOXIC/FITS | 论文指向作者仓库 |
| 本次固定版本 | d040bb015b6299da26d879b90dd19c80fb72c160 | GitHub commit API |
| 提交时间 | 2025-06-04T13:16:38Z | GitHub commit API |
| 该次提交内容 | 将 requirements 的 torch 从 2.4.1 更新到 2.7.0 | commit diff |
| 源码许可证 | Apache-2.0 | 固定提交的 LICENSE |
| DOI | 10.48550/arXiv.2307.03756 | arXiv 页面；这是 arXiv DOI，不冒称会议 DOI |
| 本项目指标证据 | 无；本分工未训练价格 | 明确未知 |

全文核查覆盖论文方法、预测实验、短期 M4 附录、分布漂移讨论及其他附录文本；异常检测波形不用于价格适配结论。OpenReview 页面出现浏览器验证，正式 proceedings 的 24 页全文可正常读取。执行环境直接下载 PDF 受网络限制，因此没有在本目录保存 PDF，不影响通过正式 proceedings 读取已引用的内容。

## 3. 与作者代码一致的模型定义

取单变量输入窗 x∈R^L，M=L+H，η=M/L。按 `models/FITS.py`：

\[
\mu(x)=\frac1L\sum_{j=0}^{L-1}x_j,\qquad
s(x)=\sqrt{\frac1{L-1}\sum_{j=0}^{L-1}(x_j-\mu)^2+10^{-5}},
\]

\[
z=(x-\mu\mathbf1_L)/s,\qquad
Z=P_K\operatorname{rFFT}_L(z),\qquad K_o=\lfloor\eta K\rfloor,
\]

\[
U=WZ+b,\quad W\in\mathbb C^{K_o\times K},\ b\in\mathbb C^{K_o},
\]

\[
\widehat x=\mu\mathbf1_M+s\eta\operatorname{irFFT}_M\{P_0(U)\},\qquad
\widehat y=\widehat x_{L:L+H}.
\]

其中 P_K 保留 rFFT 的前 K 个系数，**作者实现包含索引 0**；P_0 在高频末端补零至 floor(M/2)+1。论文叙述有去掉/补回零频项的说法，但实际模型的 `0:K` 包含中心化后接近零的 DC 输入，复线性偏置可生成非零 DC 输出，应以固定作者代码说明复现口径。

上式使用明确的 `irFFT_M` 描述论文意图和本次孤立长度补丁；**原作者类没有传 n=M**，其实际输出长度见第 5 节。

作者代码按 `train_mode` 提供三种训练途径：0 为整个回看窗与未来窗一起 MSE（B+F）；1 仅预测窗 MSE（F）；2 先 B+F 再 F。`etth1_best.sh` 使用 train_mode=1。增加 backcast 或改用 forecast-only 都是原方案已有选项。

单变量共享复层的参数量为 K_o(K+1) 个复数存储元素，等于 2K_o(K+1) 个实数自由度。原训练器的 `p.numel()` 把一个复数参数计作一个元素；和纯实数模型比较时须同时报告实数自由度。模型不是固定恰好 10k 参数，参数数目取决于 L、H、K。

## 4. 数学等价与表示缺口

### 4.1 归一化并未产生任意非线性学习能力

将 FFT、截频、W、补零、逆 FFT 组合为实线性算子 A，将偏置变换记作 c_b，可直接展开得：

\[
f(x)=\mu(x)\mathbf1_M+A\{x-\mu(x)\mathbf1_L\}+s(x)c_b.
\]

因此，除标准差乘偏置这一项外，其预测映射就是受频域参数化约束的时域线性映射。若 b=0，则严格为线性映射。均值恢复赋予平移等变性，不意味着模型只能输出历史均值；它仍能学习低频趋势、相位与幅度变化。将 FITS 描述成任意非线性深度模型，或描述成只能重复原来的周期，都不准确。

独立代数展开与作者补丁类前向在 float32 下最大误差为 3.814697265625e-06；这是等价核查，不是预测实验。

### 4.2 丢弃频率后不一定保留当前价格

考虑 L=20、K=4，j=0,…,19：

\[
x^+_j=30+2(-1)^j,\qquad x^-_j=30-2(-1)^j.
\]

两窗均值、方差相同，差异仅在被滤掉的 Nyquist 模式。因此保留的归一化低频表征相同。对任意 W、b，都有 f(x^+)=f(x^-)，而两窗最后价格分别为 28、32；RW 的预测会相差 4。

作者原类及仅修长度的类，在种子 0、17、42 对应的任意复权重下均实测预测差为 0，而当前价格差为 4。这个反例证明特定截频表示不能无损表达所有 RW 映射，**不证明真实碳价格中这些高频成分一定可预测，也不证明 FITS 必然输给 RW**。

它提供的可否证诊断是：若完整原模型的短期误差主要发生在当前价偏离其低通重建值较大的起点，低频信息损失可能是实际缺口；若误差与该量无关，便不能据此设计模型。直接把当前价接回去只修补该反例，仍属于普通 skip/anchor 方案，不构成足够的新颖性证据。

## 5. 可运行性缺陷与合成实测

| 问题 | 固定源码位置 | 本次核查与必要处理 |
|---|---|---|
| 奇数总长少一个输出 | models/FITS.py 第 54 行 | 原 `irfft` 未传 n；改为 `n=self.seq_len+self.pred_len`，独立副本保存；这是 bug fix |
| 短窗默认 cutoff 越界 | run_longExp_F.py 第 104–113 行 | 默认 K=(L//24+1)×H_order+10；L=20 默认 K=12，超过 rFFT 的 11 个系数，实测矩阵维度错误；须显式给合法 K |
| 验证集随机打乱且丢尾批 | data_provider/data_factory.py 第 29–30 行 | val 落入通用 else；短验证集可能被大幅删减甚至为空；统一评价器须固定顺序、保留全量验证点 |
| 每轮读取并打印测试损失 | exp/exp_main_F.py 第 206–209 行 | 本项目必须关闭逐 epoch 测试读取；early stopping 只依据合法内部验证 |
| batch_size 隐式翻倍 | run_longExp_F.py 第 120–124 行 | 无增强时乘 2；复现预算应记录实际批大小 |

cutoff 还需满足两个约束：

\[
1\le K\le\lfloor L/2\rfloor+1,\qquad
\lfloor K(L+H)/L\rfloor\le\lfloor(L+H)/2\rfloor+1.
\]

合成形状测试覆盖 L∈{20,21,60}、H∈{1,5,20} 共 9 格。原类长度恒为 2 floor((L+H)/2)；补丁类全部为 L+H。4 个偶数总长案例中，补丁输出与相同权重原类逐元素差为 0。L=20/H=5 时，原类输出 24 点，B+F 的 25 点目标引发 RuntimeError；补丁输出 25 点且损失可计算。仅预测最后 H 点的模式可能不报 shape 错误，但使用了错误的输出长度与 Fourier 网格，不能当成忠实的 L+H 插值实现。

这些合成测试没有拟合参数，没有使用 950 条价格，没有产生预测准确率。

## 6. 对 950 条非稳定周期价格的适配判断

论文主预测基准为 ETT、Electricity、Traffic、Weather；附录 D.3 承认短期 M4 不是最优，短窗难以提取有效频域表示是原因之一。其 M4 Yearly/Quarterly/Monthly SMAPE 为 14.00/10.72/13.49；表中 TimesNet 对应 13.38/10.10/12.67。附录 E.1 还将部分数据上更长回看窗的退化归因于分布漂移。

对当前项目的以下判断属于推断，须经内部历史验证：

- 单序列、小样本与 H=1/5/20 的重点，和原论文大规模周期基准不同；小参数量降低估计负担，但不保证频域归纳偏置匹配。
- 观测步频率不是自然日频率。默认 base_T=24 来自原基准设置，不能转写成碳价格的 24 日周期假设。
- 输入周期稳定性不足、末端跳变包含有效价格信息、结构变化使长窗混合多个状态，都是最可能失败的途径；不能只用“学习率再调小”掩盖它们。
- 全窗 B+F 在 H 远小于 L 时给回看重建分配大量损失权重。F 模式原论文已经提供，应当视为原方案选项，而非我们的创新。
- 新规则必须在完整作者类重训并定位可重复失效以后才设计。本轮没有这种价格证据，因此不推出命名创新模型。

## 7. 依赖与工件

固定作者 requirements：einops 0.8.0、matplotlib 3.9.2、numpy 1.26.4、pandas 2.2.3、pmdarima 2.0.4、ptflops 0.7.4、scikit_learn 1.5.2、thop 0.1.1、torch 2.7.0、tqdm 4.66.5。单独导入原 Model 时实际依赖 torch、numpy 与作者 models/NLinear.py（后者是未使用的导入，但须存在）。

本次实际合成执行为 torch 2.3.1+cpu，来自父任务已获正常审批的 carbon_deps。它和当前作者 pin 不同，只用于明确列出的形状与代数性质验证；不能据此声称完成作者全环境复现。

- `official/`：从固定作者提交取得的原始模型、训练器、数据工厂、运行脚本、示例脚本、requirements、LICENSE。
- `patched/FITS_lengthfix.py`：带修改声明的孤立补丁类；唯一功能变化为 irFFT 明确输出总长。
- `torch_synthetic_audit.py`：作者类合成核查脚本。
- `torch_synthetic_receipt.json`：完整结果、软件版本、原始与补丁文件 SHA256。
- `numpy_operator_check.py` / `.json`：此前 NumPy 代数核查，明确不是作者模型训练。

复核命令：

```bash
PYTHONPATH=/workspace/scratch/b31c5edbd0de/carbon_deps python literature_fits/torch_synthetic_audit.py
```

## 8. 精确来源 URL

1. 正式全文：https://proceedings.iclr.cc/paper_files/paper/2024/file/701251e1db4a2e4dd2ef23f5265d5936-Paper-Conference.pdf
2. 正式题录：https://proceedings.iclr.cc/paper_files/paper/2024/hash/701251e1db4a2e4dd2ef23f5265d5936-Abstract-Conference.html
3. 作者 arXiv/DOI：https://arxiv.org/abs/2307.03756 ，https://doi.org/10.48550/arXiv.2307.03756
4. 作者仓库：https://github.com/VEWOXIC/FITS
5. 固定提交：https://github.com/VEWOXIC/FITS/commit/d040bb015b6299da26d879b90dd19c80fb72c160
6. 原 Model：https://github.com/VEWOXIC/FITS/blob/d040bb015b6299da26d879b90dd19c80fb72c160/models/FITS.py
7. 原训练器：https://github.com/VEWOXIC/FITS/blob/d040bb015b6299da26d879b90dd19c80fb72c160/exp/exp_main_F.py
8. 原数据工厂：https://github.com/VEWOXIC/FITS/blob/d040bb015b6299da26d879b90dd19c80fb72c160/data_provider/data_factory.py
9. 原运行脚本：https://github.com/VEWOXIC/FITS/blob/d040bb015b6299da26d879b90dd19c80fb72c160/run_longExp_F.py
10. 原脚本示例：https://github.com/VEWOXIC/FITS/blob/d040bb015b6299da26d879b90dd19c80fb72c160/scripts/FITS/etth1_best.sh
11. 原依赖：https://github.com/VEWOXIC/FITS/blob/d040bb015b6299da26d879b90dd19c80fb72c160/requirements.txt
12. 许可证：https://github.com/VEWOXIC/FITS/blob/d040bb015b6299da26d879b90dd19c80fb72c160/LICENSE
13. PyTorch irFFT 输出长度规定：https://docs.pytorch.org/docs/2.14/generated/torch.fft.irfft.html

供父任务自行打开后引用的检索 refs：正式全文 turn3view0；M4 附录 turn12view1/turn24view0；分布漂移 turn27view4；作者仓库 turn2search0；arXiv DOI turn27view2；PyTorch 长度文档 turn27view1。没有使用搜索聚合页推定源码、许可或市场表现。
