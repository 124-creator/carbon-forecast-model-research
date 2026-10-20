# 阶段A：主源核读完成，算法候选暂不提交

## 审核结论

本次仅做检索、原文与代码静态核读。基础模型优先选择官方 NeuralForecast 的完整 NHITS，而不是重新编写一个线性修正器；但阶段A整体 **BLOCKED**，不授权阶段B。原因不是训练失败，而是以下两项前置证据尚不足：两篇论文均不能直接证明“单条短序列/极少训练窗口”的优势；本次找到的自适应尺度、时间扭曲和可学习样条位置已有明显先例，尚不能给出可信的实质创新差异。不得把未核实的差异包装成贡献假设。

本报告在真实检索、下载、阅读和引用登记之后创建。累计6个检索query，其中2个失败，未增加第7个。原始返回分别保存于 search_receipts.json 和 search_receipts_second.json。没有训练、没有模型实现、没有市场数据读取或修改、没有2022样本实验、没有提交或推送。

## 1. 两篇有作者代码的结构预测论文

### NHITS — Challu等，AAAI 2023

读取范围：出版社9页PDF全部文本；重点核读§Methodology、§Experimental Results、§Training、§Ablation，PDF第3–7页。摘要和论文第1页直接指向作者仓库；README也明确自称官方NHITS实现。[1][5]

原文定位：第4页Eq.(2)–(4)，先由MLP产生前后向系数，再由插值函数合成预测；第3页Eq.(1)为MaxPool。第4页原公式为 `yhat_tau,l = g(tau, theta_l^f)`，系数数目 `ceil(r_l H)`，时间划分采用固定尺度。输入残差依次减去backcast，forecast依次相加。这是可解释的尺度分解，不是外接预测残差收缩。[1]

小样本证据边界：正文实验自称六个large-scale benchmark；ILI为较短周序列，但这不等于few-shot训练消融。没有读到固定单条序列仅少量窗口时与完整骨干比较的学习曲线。不能将“参数减少”改称“已证明小样本有效”。[1]

代码与论文必须分开：现行代码 `_IdentityBasis.forward` 将backcast直接切片，只有forecast调用`F.interpolate`；`create_stack`使用`max(h // n_freq_downsample[i], 1)`，并不是正文`ceil(r_l H)`的完全逐字复现；`NHITS.forward`还初始化最后观测值为level。因此后续基准应使用本次完整库实现作为共同骨干，不能混用纸面公式和另写实现后声称原模型复现。[3]

### N-BEATS — Oreshkin、Carpov、Chapados、Bengio，ICLR 2020

读取范围：arXiv v4的31页全部文本（分页工具覆盖1–4034行）；重点核读§3、§5.2、Appendix A/B/D。官方仓库README将模型与原论文对应，但明确说明代码是lighter版本，论文正文使用TensorFlow，而该仓库提供PyTorch实现。[2][8]

原公式定位：PDF第3页Eq.(1)为四层FC与前后向系数；第4页§3.2为 `x_l=x_(l-1)-backcast_(l-1)`、`yhat=sum_l yhat_l`；第5页Eq.(2)/(3)分别将预测映射到低阶多项式与Fourier基。预测分解来自固定基和串联残差，不来自惩罚项。[2]

小样本证据比NHITS更接近要求，但不能夸大：第8页报告同一架构在M3 Others的174条序列上训练；Appendix A表3给出该子集长度71–104。TOURISM yearly表4最短长度11。但短序列通过面板共享、窗口采样和集成训练，不等于仅一个市场的一条短序列。论文常用180模型集成，较小消融仍用18模型，后续不能拿单模型运行直接承接全部榜单结论。[2]

## 2. 许可与预训练污染核查

NHITS作者库本次取得的LICENSE为Apache-2.0；分发改动须保留许可和适用声明、标明改动。此处仅报告文件文本，不替代法律意见。[4]

N-BEATS作者库为CC BY-NC 4.0，并明确商业用途需额外许可、未授予专利许可。不能把第三方MIT实现的许可套给作者代码。研究可用性仍须遵守该文本限制。[7]

本次两份模型文件均从构造器创建层，未看到默认下载预训练权重。NHITS基类已核读构造器359–578行，并检索其权重加载相关路径；该库允许显式加载checkpoint，故“没有默认预训练”不等于“任何使用方式都无污染”。本次未实例化、未加载权重、未下载任何checkpoint。后续只允许从头初始化、训练边界内fit，不允许仓库示例预训练/迁移权重、不允许全样本预分解或全样本缩放。这里是静态检查与拟定约束，不是已经完成训练管线污染审计。[3][6][11]

版本边界：当前main/master源码按SHA256固定了本次取得的内容；两个GitHub commit API返回已存档，但没有逐个用commit URL重新取得并比对全部文件，因此不声称这些浮动分支文件已经绑定到同一准确提交。完整库安装、依赖锁定和运行可用性尚未验证，阶段A禁止借此启动训练。

## 3. 最近邻核查：为什么不能提交“自适应插值”作为创新

已核读的WGP，Tolpin，PDF共6页：第2页Eq.(5)通过正距离增量改变观测时间；第3页§3.3讨论未来时间映射，并采用最后时距变化率延续；§3.4/Eq.(9)分别使用原时间处理周期、扭曲时间处理局部非平稳性。因此，“波动越大时间越快”“历史时间与季节时间分开”不是新机制。只把GP换成NHITS，尚没有算法上的新增必要性论证。[9]

已核读SplineNet，Biloš等，PDF共6页：§2.2已经使用可学习knot位置和多项式系数、通过曲线距离得到可解释特征，并允许最终取样位置可学习。论文预测部分是Physionet与Speech的10步重建误差，不是小样本金融价格日序列；作者代码及许可本次未核，所以仅作为创新最近邻，不计入“两篇有作者代码已核”名额。[10]

Pathformer检索返回指出按样本动态选择多尺度路径。**只读到了检索返回，没有核读全文**；本次尝试的OpenReview PDF返回403，记录于source_receipts.json。拒绝后未访问该目标的代理、镜像或其他绕行地址，不以该摘要补足全文门槛。其更细粒度router公式与本任务的差异仍未知。

因此下列包装均不通过：固定插值改为可学习位置；把波动时间扭曲接到NHITS；新增多尺度router后宣称“不同函数类”。前两者已有读到的直接机制先例；第三者最近邻全文尚未核查。不能仅靠组合、替换骨干、降低参数量来证明贡献。

## 4. 诚实阻塞与父审核范围

本次提交 **零个算法候选**，不提供“我们改公式”，不预填参数/复杂度或同骨干消融为已经成立的设计。原公式、源码差异与最近邻均已列出，足以防止继续沿惩罚、收缩或简单模块并列路线训练。

父审核可确认的成果：NHITS官方完整库可作为优先骨干；N-BEATS提供更明确的短序列面板证据，但作者代码许可和lighter实现需保留；时间扭曲与可学习节点已有近邻，不能冒称新颖。未解决的前置条件：第二篇直接小样本结构预测证据，以及一个经过最近邻全文核查的新增算法机制。该阻塞不表示停止整体研究，只表示本次阶段A没有足够证据放行阶段B。

## 5. 访问失败与保留范围

首次requests下载发生TLS客户端EOF，随后使用本机curl正常证书校验取得公开原文，不关闭证书、不更改网络安全配置。配置中的web_extract为search-only backend，真实返回“不支持extract”，没有把它记为全文访问。两个检索query失败已保留。OpenReview 403保留为拒绝，不重试绕行。

PDF、全文摘取和第三方源码仅为本地核读证据，不运行、不重新授权。`.download`/PDF存在重复下载副本；本次不删除证据、不复制到其他实验目录，也不自动纳入提交。

## Sources

[1] https://ojs.aaai.org/index.php/AAAI/article/download/25854/25626 — nhits_paper
    > "Our code is available at"
[2] https://arxiv.org/pdf/1905.10437 — nbeats_paper
    > "The same architecture is successfully trained on the M4 Monthly subset with 48k"
[3] https://raw.githubusercontent.com/Nixtla/neuralforecast/main/neuralforecast/models/nhits.py — nhits_code
    > "forecast = F.interpolate("
[4] https://raw.githubusercontent.com/Nixtla/neuralforecast/main/LICENSE — nhits_license
    > "Apache License Version 2.0, January 2004"
[5] https://raw.githubusercontent.com/Nixtla/neuralforecast/main/README.md — nhits_readme
    > "Official `NHITS` implementation, published at AAAI 2023."
[6] https://raw.githubusercontent.com/ServiceNow/N-BEATS/master/models/nbeats.py — nbeats_code
    > "forecast = x[:, -1:]"
[7] https://raw.githubusercontent.com/ServiceNow/N-BEATS/master/LICENSE — nbeats_license
    > "requires an explicit license."
[8] https://raw.githubusercontent.com/ServiceNow/N-BEATS/master/README.md — nbeats_readme
    > "This is a "lighter" version of original N-BEATS"
[9] https://arxiv.org/pdf/1912.02527 — warp_gp_paper
    > "the time runs faster in areas with high"
[10] https://arxiv.org/pdf/2210.10630 — splinenet_paper
    > "The positions of these points can also be learnable."
[11] https://raw.githubusercontent.com/Nixtla/neuralforecast/main/neuralforecast/common/_base_model.py — nhits_base
    > "Allows instantiation from a checkpoint from class"
