# 直接近邻与可复现强对照审查

日期：2026-10-19。本文是新一轮方法研究的文献与实现审查，不是正收益结果报告。上一轮反射控制候选已经停止；本文不修改该轮实验，也不使用 2024 年及以后的保留市场值。最终执行设置以本轮冻结协议和代码为准。

## 1. 本轮结论

1. **先补真正缺失的强对照。** GAS-Norm 的动态位置、尺度过滤，以及预测未来归一化统计量后的反变换，没有被上一轮 EWMA 加有界单尺度因子完整覆盖。本轮适合完成 GAS-Norm、EWMA、Global 三种归一化配合同一 GRU、同一混合密度头、同一 CRPS 目标的受控实验。这是论文方法的任务适配，不是复现原论文全部数值结果。
2. **碳价概率预测已经是成熟题材，单纯把点预测改成分布预测不足以构成方法贡献。** 碳价已有在线 CRPS 分位数组合、QTCN、混频 Transformer 分位数预测及分位数与 conformal 区间。对只能读到摘要的论文，下表不补猜日期、切分或损失细节。
3. **“共享神经状态转移并传播完整预测分布”本身已有直接先例。** DeepState、DS³M 和 D2D 分别覆盖解析高斯状态空间预测、神经切换生成路径、递归分布映射与多期限联合训练。DS³M 的作者代码甚至显式加入了保持原状态的转移概率。
4. 当前三态候选只能保留一个较窄的问题：**在固定单步分布和全部期限二阶矩的条件下，学习起点条件化的持续性，是否改善多期限端点分布的 CRPS？** 受限分布族、精确计算和较少自由度是可以验证的结构差异；尚未确证其原创性，也尚未证明具有市场价值。先做低成本合成筛选是合理的。

## 2. 阅读深度与比较规则

**A**：已读取原始正文中直接相关的方法、实验和／或附录；有作者代码时单独标明。**B**：只读取出版商、作者机构或原始书目数据库的摘要及元数据，不能据此声称完整复现。每行均给可独立打开的原始链接；没有原始全文支持的字段标为“未核实”。下面报告论文设计，不搬用论文的优胜结论作为本项目证据。

### 2.1 碳价概率预测的直接比较

| 论文与证据深度 | 数据、输入与日期 | 预测对象、损失与切分 | 代码／数据可复现性 | 结构贡献与本项目缺口 |
|---|---|---|---|---|
| **Berrisch & Ziel, CRPS Learning**，Journal of Econometrics 237(2), 105221，2023；方法预印本 2021。**A**：原文 §4–5、§7 和算法；[DOI](https://doi.org/10.1016/j.jeconom.2021.11.008)、[作者预印本 v3](https://arxiv.org/pdf/2102.00968v3)。 | ICE EUA 月前期货，2013-01 至 2020-12，2,092 个日观测。目标为对数价格。 | 一步分位数预测；99 个分位水平。五个 ETS／QR／ARMA-GARCH／EGARCH 等专家各用滚动 250 日估计；形成 1,841 个样本外预测。在线 Bernstein 聚合可按分位水平变化并作平滑。 | [作者 profoc](https://github.com/BerriJ/profoc)，GPL≥3。此前固定版本树未找到该 2,092 条原始应用数据。 | 学习分位水平相关的组合权重，具有在线学习方法内容。本项目前轮只有相同两个专家、19 原子、不作样条平滑的适配；不能写成已经完整覆盖原文的五专家、99 分位方案。 |
| **Cao, Zha, Wang & Wen, Probabilistic carbon price prediction with quantile temporal convolutional network considering uncertain factors**，J. Environmental Management 342, 118137，2023。**B**：[DOI](https://doi.org/10.1016/j.jenvman.2023.118137)、[原始 PubMed 摘要](https://pubmed.ncbi.nlm.nih.gov/37178463/)。 | 摘要为湖北市场；列出能源、经济、国际碳价、环境、公众关注、地缘风险与政策不确定性等信息。精确样本日期未核实。 | 碳价分位数／概率预测，QTCN。确切损失公式、分位网格和时间切分未核实。 | 未找到可核的作者公开实现；全文未取得。 | 已覆盖 TCN 加分位数概率输出的基本思路。额外外生输入与本项目现有单变量信息集不同；不能仅对照论文报表的数值。 |
| **Han et al., Causal neural network for carbon prices probabilistic forecasting**，Applied Energy 397, 126343，2025。**B**：[DOI](https://doi.org/10.1016/j.apenergy.2025.126343)、[出版商](https://www.sciencedirect.com/science/article/pii/S0306261925010736)。 | 出版商可索引摘要／正文片段说明广东与欧盟市场，能源和经济因素。日期未核实。 | EPPT、单调复合分位数 causal TCN，继而用 KDE 形成密度；确切损失、切分与预处理范围未核实。 | 未找到公开作者代码；未读到完整算法。 | 非线性 Granger 关系和单调分位数约束是直接近邻。“causal”名称不等于本文核实了干预识别；也不能仅凭分解模块认定泄漏。 |
| **Ji, Du, Du, Niu & Wang, A novel probabilistic carbon price prediction model: Integrating the transformer framework with mixed-frequency modeling at different quartiles**，Applied Energy 391, 125951，2025。**B**：[DOI](https://doi.org/10.1016/j.apenergy.2025.125951)、[出版商](https://www.sciencedirect.com/science/article/pii/S0306261925006816)。 | 摘要为上海、湖北，使用混频信息；精确观测日期和完整输入表未核实。 | QRTransformer–MIDAS，点／区间预测；精确训练损失、分位网格、切分未核实。 | 未找到公开作者代码或可直接合法取得的全套原数据。 | 主要处理混频输入与条件分位输出。当前信息集中缺乏同口径混频驱动；删去 MIDAS 后不能仍称原方法复现。 |
| **Wu, Abedin, Zeng & Lucey, European Union Allowance price forecasting with Multidimensional Uncertainties: A TCN-iTransformer Approach for Interval Estimation**，Journal of Forecasting 45(1), 88–113，2026；2025-09-21 首次在线。**B**：[DOI／出版商](https://onlinelibrary.wiley.com/doi/10.1002/for.70024)。 | EUA 和多类不确定性变量；摘要提及期限利差和地缘风险。样本日期未核实。 | TCN 与 iTransformer、QR、conformal 区间；确切校准、切分及训练损失未核实。 | 出版商说明数据需向作者申请，因隐私／伦理限制不公开；未找到作者代码。 | 概率／区间模块已有先例。没有全文不能判断其 conformal 时间依赖处理，更不能直接移植独立同分布覆盖保证。 |

### 2.2 与本轮学习机制更接近的非碳方法

| 论文与证据深度 | 数据与预测任务 | 方法、损失与切分 | 复现性和本项目的实质关系 |
|---|---|---|---|
| **Urettini, Atzeni, Ramjattan & Carta, GAS-Norm**，CIKM 2024。**A**：§3–5、Student 推导、两套作者实现。[DOI](https://doi.org/10.1145/3627673.3679822)、[正文](https://arxiv.org/html/2410.03935v1)。 | AR 合成；VIX 1990-01 至 2024-05 的一阶差分；ECL 第一客户前 10,000 小时；另有 NN5 Weekly、M4 Weekly、FRED-MD。读取论文设计不等于开启本项目保留值。 | Student score 过滤位置／尺度，先拟合静态参数并冻结，再训练神经残差、反归一化。第一组 LSTM 预测 50 步，报告 MASE；作者第一仓库训练用 MAE，按 70/10/20 划分。 | 两仓库公开可读，但本次未找到覆盖核心代码的明确复用许可证。独立数学实现，见 §3。上一轮只有 EWMA 输入及尺度头，未完整覆盖此法。 |
| **Zhao, Zhu, Ng & Lee, From GARCH to Neural Network for Volatility Forecasting**，AAAI 2024。**A**：方法 Eq.1–7、实验设计。[DOI](https://doi.org/10.1609/AAAI.V38I15.29643)、[正文](https://arxiv.org/html/2402.06642v1)。 | S&P500、DJI、NASDAQ、EUR/USD、Gold，约 2,514–2,603 个日收盘收益；正文所读部分未给可核完整日历区间。预测波动。 | GARCH 作为递归神经核，并用 LSTM 门调节；Gaussian／Student NLL，约 8/1/1 顺序划分，验证早停。评估代理由最近五个平方收益构成。 | 未找到作者仓库。不能用第三方 garchai 代替作者原实现。其波动代理含已知历史量，与未来 h 日端点分布不是相同任务；公式可独立适配。 |
| **Buczynski & Chlebus, GARCHNet: Value-at-Risk Forecasting with GARCH Models Based on Neural Networks**，Computational Economics 63, 1949–1979，2024；2023 在线。**A**：§2.3、§3。[DOI／全文](https://doi.org/10.1007/s10614-023-10390-7)。 | WIG20、S&P500、FTSE100 日收益；四个 1,250 点窗口始于 2005、2007、2013、2016，测试对应 2009、2011、2017、2020。 | LSTM 输出正波动、t 自由度或偏度，条件 NLL；滚动 1,000 点训、一步预测，重复 250 次；固定 300 epoch，无验证集；2.5% VaR。 | 作者数据／代码按需申请。精确原训练预算较大；同 CPU 预算缩小必须标适配。动态分布参数头已有直接先例，不能把“神经估计密度参数”当新机制。 |
| **Livieri & Palmari, Proper-Score Observation-Driven Filters: Local Geometry, Estimation, and Continuous-Time Limits**，2026 预印本。**A**：核心递推、§8–9、附录 G。[正文 v3](https://arxiv.org/html/2608.02828v3)。 | 12 个国际股指 2000–2024，EURO STOXX50 自 2007；70% 估计、30% 测试。原始平方收益 QLIKE 为主，OHLC 范围代理为敏感性。 | 同一密度下以 log、density-power、CRPS、MMD 驱动状态更新；另换 t 密度。固定起点优化，事件分析严格在事件前估计。 | 未找到作者代码；附录给显式更新，可独立实现。该文是预印本，不能写已被期刊验证。它直接覆盖“把 GAS 的 log-score 换为 CRPS-score”的宽泛创新。 |
| **Xu, Peng & Chen, Deep switching state space model for nonlinear time series forecasting with regime switching**，International Journal of Forecasting 42(1), 85–98，2026；预印始于 2021、2025 更新。**A**：§3、§4、附录和作者路径采样代码。[DOI](https://doi.org/10.1016/j.ijforecast.2025.05.001)、[原始预印本](https://arxiv.org/abs/2106.02329)。 | Sleep；UNRATE 1948-01—2021-03；杭州 2019-01；Seattle 2015-01；Pacific 1970—2002；法国电力 2012—2019。测试分别用最后区间。 | 静态 Markov 离散状态 + 神经连续 latent transition／Gaussian emission；ELBO，验证早停；未来递归采样完整路径。主要报告 RMSE／MAPE，不是端点 CRPS 训练。 | [作者代码](https://github.com/Sherry-Xu/Deep-Switching-State-Space-Model)，Apache-2.0，固定版本见 §5。已覆盖神经 Markov 状态生成概率路径；本轮受限三态、无连续 latent、解析多期限 CRPS 是较窄差异。 |
| **Yang, Du & Aslett, A Distribution-to-Distribution Neural Probabilistic Forecasting Framework for Dynamical Systems**，2026 预印本。**A**：§II–III。[正文](https://arxiv.org/html/2603.25370v1)。 | Lorenz63，RK4 步长 .01、每 .05 观测并加高斯噪声；示例训练 30,000 样本。完整独立切分计数未核实。 | 五分量边际 GMM→核均值嵌入→LSTM→GMM，同一算子迭代；多期限 log-score 和 curriculum。 | 未找到作者公开实现。明确只传播各状态坐标边际，不是完整联合状态分布；不能把“递归分布传播 + 多期限联合训练”作为本项目宽泛创新。 |
| **Rangapuram et al., Deep State Space Models for Time Series Forecasting**，NeurIPS 2018。**A**：§3–5。[原文](https://proceedings.neurips.cc/paper/8004-deep-state-space-models-for-time-series-forecasting.pdf)。 | Electricity、Traffic、M4、Tourism、Parts；采用指定顺序留出或公开标准切分；并非碳价。 | RNN 生成线性高斯状态空间参数；Kalman likelihood 可微训练，未来完整联合高斯分布可解析计算，也可路径采样。 | 这里核了原论文，未做特定 GluonTS 代码版本复现。它已覆盖神经参数化状态空间与解析联合概率预测；三态厚尾是受限分布族差异，不是新框架。 |
| **Wang & Rachev, Neural Lévy SDE**，预印本记录需要澄清。**A（方法局部）**：§3–4.2。[原文](https://arxiv.org/html/2509.01041v1)。 | 美国股票横截面，日／周价格量、部分高频 RV 与结构特征；确切全样本日历和切分未核实。 | 共享 encoder、各 h 独立的 drift／diffusion／jump 头；加权 QML 与正则。§4 明确不依赖连续路径积分。 | 未找到作者代码。arXiv v1 的 2025 版本标识与稿内 2026 日期不一致，暂不作精确先后优先权依据。独立期限头没有实现本轮共享未来状态转移。 |

## 3. GAS-Norm：可核的数学接口与本轮适配

### 3.1 来源与版本

本次只读取作者源码，不复制代码：

- [GAS 参数拟合与更新](https://github.com/edo-urettini/GAS_Norm/blob/cc63080958ebc9bfbb4bc95c2333c76a2070e607/normalizers/GAS_norm.py)：`SD_Normalization_Student`、`Update_function_Student`、`neg_log_likelihood_Student`、`Optimized_params_Student`。
- [预测和反归一化](https://github.com/edo-urettini/GAS_Norm/blob/cc63080958ebc9bfbb4bc95c2333c76a2070e607/models/GAS_LSTM.py)：`AR_GAS.forward`、`GAS_LSTM.forward`。
- [时间切分](https://github.com/edo-urettini/GAS_Norm/blob/cc63080958ebc9bfbb4bc95c2333c76a2070e607/utils/data_prep.py)：`prepare_dataset`；[训练目标与强度选择](https://github.com/edo-urettini/GAS_Norm/blob/cc63080958ebc9bfbb4bc95c2333c76a2070e607/main_exp.py)。
- [第二套作者实现](https://github.com/daniele-atzeni/GAS-Norm-gluonTS/blob/814441c6912c2aeea812593ecf18a47f86c80aa3/normalizer.py)：`GASTStudent`。该仓库包含概率预测网络，但不能因此把论文的 MASE 结果改写成已完成本任务的 CRPS 比较。

两套版本分别固定为 `cc63080958ebc9bfbb4bc95c2333c76a2070e607` 和 `814441c6912c2aeea812593ecf18a47f86c80aa3`。本次 GitHub 元数据未标明核心仓库许可证；第一仓库某第三方子目录有许可证，不据此推断全部 GAS 实现可直接复用。

### 3.2 状态、更新与真正的标准差

用 (m_t,s_t^2) 表示看见 (x_t) 前的 Student 工作分布位置和 **scale squared**，

\[
 x_t\mid\mathcal F_{t-1}\sim t_\nu(m_t,s_t),\qquad
 v_t=\frac{\nu}{\nu-2}s_t^2.
\]

令 (e_t=x_t-m_t)、(\eta_j=\gamma_j/(1-\gamma_j))。先更新，再产生下一观测的先验：

\[
\begin{aligned}
m_t^+ &=m_t+\eta_\mu\alpha_\mu
       \frac{e_t}{1+e_t^2/(\nu s_t^2)},\\
(s_t^2)^+&=s_t^2+\eta_\sigma\alpha_\sigma
       \left\{\frac{(\nu+1)e_t^2}{\nu+e_t^2/s_t^2}-s_t^2\right\},\\
m_{t+1}&=\omega_\mu+\beta_\mu m_t^+,\\
s_{t+1}^2&=\omega_\sigma+\beta_\sigma(s_t^2)^+.
\end{aligned}
\]

归一化历史输入是 (z_t=(x_t-m_t)/\sqrt{v_t})，使用观测前的状态。原论文 Eq.(4) 第二行印为 (\beta_\mu)，本实现采用作者代码的 (\beta_\sigma)。未来未知观测没有实现值，采用原论文 §3.3 明列的纯 AR 预测：

\[
m_{t+j+1\mid t}=\omega_\mu+\beta_\mu m_{t+j\mid t},\quad
s^2_{t+j+1\mid t}=\omega_\sigma+\beta_\sigma s^2_{t+j\mid t}.
\]

这不是把未来 (x\) 设成预测均值后再代回 score：那样尺度 score 不为零，会多出负的尺度修正。作者 `AR_GAS` 代码采用了后者，而论文预测式采用前者。本轮固定前者，并写入 metadata；不宣称二者数值相同。

### 3.3 固定拟合设置与已声明的偏离

本轮无超参数搜索，固定 (\nu=20\)、(\gamma_\mu=\gamma_\sigma=.1\)。静态参数为八个：两组 (\alpha,\beta,\omega\) 和初始 (m_0,s_0^2\)。沿用作者数值边界：四个 (\alpha,\beta\in[0,1]\)，(\omega_\sigma,s_0^2\ge10^{-6}\)，位置截距与初值无符号限制。工作尺度单位先用训练前缀的均值和总体标准差固定；该标准化同样是训练数据内估计。

原作者实现的目标把每次更新后状态用于下一真实观测的 Student log likelihood，并以相邻状态的平方变化惩罚；它不是论文 Eq.(2) 所写一般 Fisher 矩阵惩罚的逐字实现。本轮独立实现明确使用：

\[
\frac1{T-1}\sum_{t=0}^{T-2}\Big[
-(\gamma_\mu+\gamma_\sigma)\log p_\nu(x_{t+1};m_{t+1},s_{t+1})
+\frac{1-\gamma_\mu}{2}(m_{t+1}-m_t)^2
+\frac{1-\gamma_\sigma}{2}(s_{t+1}^2-s_t^2)^2\Big].
\]

作者源码会复制最后一个观测来凑最后一项。本轮删去此人工重复，命名 `author_penalty_terminal_fixed_v1`。单个确定起点、Powell、最多 400 次迭代／4,000 次目标计算；优化是否收敛、目标值、计算次数和训练数据哈希均保存。以上是明确选择的任务适配，不以测试结果决定用哪一种版本。

当 (\gamma\le.5\)、(\alpha_\sigma\le1\) 时，尺度更新中的最小系数 (1-\eta_\sigma\alpha_\sigma\ge0\)，加正截距后保持正性。本实现不裁剪异常状态来隐藏发散。也不把 (\gamma=0\) 一概说成静态归一化：任意 (\omega,\beta\) 仍可产生 AR 变化；Global 臂单独定义。

### 3.4 和本项目完全相同目标的适配

对每个 (h\in\{5,20\}\)，构造**截至 t 已观测的**目标过程

\[
x_t^{(h)}=\log(P_t/P_{t-h}),\quad
x_{o+h}^{(h)}=\log(P_{o+h}/P_o).
\]

这样，原模型“预测自己所过滤的过程”的设计得到保留；从起点 (o\) 对 (x^{(h)}\) 预测 (h\) 步，正好是本项目未来 (h\) 步收益。无需把日方差乘以 (h\)，也无需宣称重叠收益独立。

三个归一化臂使用相同的历史 20 个 (x^{(h)}\)、GRU16、相同 GMM 输出参数数目、seeds 17/41/89、40 epochs。若神经头输出 \((\pi_k,\widetilde m_k,\widetilde s_k)\)，GAS 的最终分量为

\[
m_k=m_{o+h\mid o}+\sqrt{v_{o+h\mid o}}\,\widetilde m_k,
\qquad s_k=\sqrt{v_{o+h\mid o}}\,\widetilde s_k.
\]

训练损失统一是反变换后收益分布的 CRPS，再除每个起点的公共单位
\(u_o=\widehat\sigma_{o,\mathrm{daily\ EWMA}}\sqrt h\)。不同臂不能各用自己的尺度来改变评分权重。验证选择用同一指标；所有标签必须在规定边界前成熟，测试参数冻结，仅按允许的已观测历史过滤。

最终输出保持 Gaussian mixture，因此价格 (P_o\exp(R_h)\) 是 lognormal mixture，有有限的任意固定阶矩。Student 只用作历史工作过滤器。若把无界 Student 直接作为对数收益最终密度，正指数矩通常发散，不能同时报告有限解析价格均值；本轮没有采用这种变换。

独立文件接口：

```python
fit = fit_gas_normalizer(x_train)  # 仅成熟训练前缀
history = filter_gas(x_observed, fit)
mean_h, std_h = forecast_gas(fit, history, horizon=h)
```

每个输出行 t 仅使用输入截至 t 的值。`history` 已保存 t+1 的状态，所以 `forecast_gas` 只追加 (h-1\) 次纯 AR 推进；归一化返回真正标准差，不混用 t scale。

## 4. 第二个低成本近邻：proper-score 状态过滤

原始 2026 预印本附录 G 给出零中心高斯的 log-variance \(\lambda=\log v\)、标准化残差 \(z=y/\sqrt v\) 递推。适当选择其列明的缩放后，

\[
\lambda_{t+1}=\omega+\varphi\lambda_t+\alpha g(z_t),\quad
g_{\log}(z)=z^2-1,\quad
g_{\mathrm{CRPS}}(z)=\pi^{-1/2}-2\phi(z).
\]

CRPS 的原始负导数为 \(\sqrt v[\pi^{-1/2}-2\phi(z)]/2\)，上述 driver 使用该文的缩放 \(2/\sqrt v\)。这是有界、尾部趋于非零常数的更新，不是“极端值完全不影响更新”。[原始公式：附录 G](https://arxiv.org/html/2608.02828v3#A7)

它适合将来以相同密度比较 log 与 CRPS 的**状态更新算子**，不能与“都用 CRPS 训练一个静态神经尺度头”混为一谈。本轮优先完成已锁定的三归一化比较，不临时增加该实验。若将来用它预测累计收益，也需要像 §3.4 那样说明目标过程与期限；一步状态方差并不自动定义真实多步收益密度。

## 5. 三态持续性候选与最近邻的准确边界

### 5.1 已存在的部分

Calvet–Fisher 的多频率切换波动模型中，每个乘子以一定概率从固定分布重新抽取，否则保持。单个乘子写成 (P=\rho I+(1-\rho)\mathbf1\pi^\top\) 正是相同保持／重抽结构。原文还研究持续性、幂次协方差与多期限波动预测。因此保持核、持续性造成尾形变化，都不能称本轮的新理论。[2002 作者工作稿 §2.1](https://archive.nyu.edu/bitstream/2451/26508/2/02-64.pdf)

DS³M 的一般结构为

\[
d_t\sim\Gamma_{d_{t-1},\cdot},\quad
z_t\sim N(f_{d_t}(z_{t-1},h_t),\Sigma_{d_t}),\quad
y_t\sim N(g_{d_t}(z_t,h_t),V_{d_t}),
\]

其中 RNN 状态 (h_t\) 同时进入 transition 和 emission。作者 `src/DSSSMCode.py` 的 `TransitionMatrix` 使用 (0.5\operatorname{softmax}(A)+0.5I\)，Sleep 为 (0.2\operatorname{softmax}(A)+0.8I\)；`_forecastingMultiStep` 从离散状态、连续状态、观测逐层采样未来，训练用 ELBO。[固定作者实现](https://github.com/Sherry-Xu/Deep-Switching-State-Space-Model/blob/f649ee04f63a32aff6fde9be19f384de187e249a/src/DSSSMCode.py)

作者仓库固定版本 `f649ee04f63a32aff6fde9be19f384de187e249a`，已核 [Apache-2.0 LICENSE](https://github.com/Sherry-Xu/Deep-Switching-State-Space-Model/blob/f649ee04f63a32aff6fde9be19f384de187e249a/LICENSE)。因此不是只在摘要中泛泛出现 Markov 一词，而是已有可运行的神经切换路径模型。

### 5.2 尚可检验的窄子问题

本轮暂限三态、固定 \(\pi=(.45,.4,.15)\)、固定已归一化波动乘子 f、起点平稳状态，并让可观测历史决定单一 \(\rho_o\)。在每个起点生成时冻结该 \(\rho_o\)，完整未来使用同一保持／重抽核。若 \(r_j=\sqrt{v_o f_{S_j}}\epsilon_j\) 且 \(\epsilon_j\) 独立标准高斯，则改变 \(\rho_o\) 不改变单步混合分布，亦不改变任意累计期限方差 \(h v_o\)。改变的是累计随机方差的分布，因而是端点分布的高阶形状。

与一般神经状态空间相比，这一受限子族没有连续 latent、没有自由的各期限密度头；可通过有限状态／占用次数递推精确组成端点高斯混合，并对实际端点 CRPS 求导。潜在贡献只能落在这个约束及其学习效果上：控制自由度是否使小样本条件持续性能够被估计，而不是增加拟合噪声。

合成否决条件应包括：固定单步及二阶矩的数值等式不成立；无条件 \(\rho\) 已达到与条件网络相当的风险；独立期限同参数预算头无劣势；真 \(\rho\) 的 oracle 优势本身小到无法支持开发目标；正效应仅来自误设同一信息预算或期限标签成熟规则。若这些条件触发，不应转而宣传“分布一致性”弥补预测收益的缺失。

## 6. 证据限制与实际交付

这次已读取多篇原始正文及两套 GAS-Norm、DS³M 作者源码。几篇 Applied Energy／Journal of Forecasting 碳价论文仅取得官方摘要／元数据；这里没有据摘要编造数据日期、分割、公式或作者实现。不存在把公开源码自动等同可自由复制、把预印本自动等同期刊定稿、或者把不同输入条件下论文表格的分数直接拿来排名的做法。

本轮先交独立数学过滤器，冻结后完成三归一化的同骨干实验；三态方法先经过合成识别与效应规模筛选。文献相似度、合成可识别性、市场样本外收益是三类不同证据，必须分别报告。
