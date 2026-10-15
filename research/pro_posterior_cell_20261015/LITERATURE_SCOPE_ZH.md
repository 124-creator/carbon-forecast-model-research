# Stage J 文献核收与阅读范围

本轮为定向查新，不是系统综述、原作者实验复现或新颖性认证。没有重新取得此前隔离PDF，也没有再分发论文全文。

## 设计相关来源

1. Martin & Elster, Aleatoric Uncertainty for Errors-in-Variables Models in Deep Regression, DOI10.1007/s11063-022-11066-3。2022-11-01在线、2023卷期。本轮实际核读出版社HTML第1—2节，包括输入不确定性、边际似然和预测矩。其目标与假设不同，不能据此宣称本候选必然改善碳价点预测。https://link.springer.com/article/10.1007/s11063-022-11066-3
2. Recurrent Kalman Networks, PMLR97, 2019。本轮只核官方摘要；主线另有作者单元前后向检查，不能变成本轮原论文复现。显式循环不确定性表示已有研究。https://proceedings.mlr.press/v97/becker19a.html
3. Mamba time series forecasting with uncertainty propagation, arXiv2503.10873v1, 2025-03-13。本轮取得HTML并阅读均值/方差双网络框架及相关训练段落，不声称全文逐式审查。协议对“设计前已读方法”的措辞过宽：最初显示的是HTML资源与摘要，明确框架段落的展开在协议提交后、训练前。该时间顺序单独保留，不倒推为先完成全部核读。https://arxiv.org/html/2503.10873v1
4. Mip-NeRF, ICCV2021。实际读作者项目页的摘要和Integrated Positional Encoding说明，未下载PDF或执行作者代码。Gaussian区域上的正弦期望、按方差与频率衰减特征已有直接代数先例；本轮闭式恒等式绝不称新定理。https://jonbarron.info/mipnerf/
5. Uncertainty propagation networks for neural ordinary differential equations, arXiv2508.16815。只取得作者摘要/检索信息；HTML v1为404，未改取PDF或镜像。不能声称核完其均值/协方差动力学公式。

## 后续补充检索：不能倒推为设计前完成

- Hybrid Feature-Selection and Multi-Scale Deep Learning for Carbon Price Forecasting in Emerging Carbon Markets, DOI10.1155/er/1314913，2026-08-13在线。仅出版社检索摘要，ET/MVMD/GRU与多市场研究已存在。没有完整管道审计，不借摘要改善数字为本项目背书，也不据摘要完成泄漏定罪。
- European Union Allowance price forecasting with Multidimensional Uncertainties: A TCN-iTransformer Approach for Interval Estimation, DOI10.1002/for.70024，2025-09-21首次在线，检索列2026卷期。只有出版社检索摘要，PDF链接未打开。外部不确定性指标、预测区间与本轮内部工作状态方差不是同一个对象。
- Rollcast: Proper-Score Gated Rolling Anchors for Adaptive Probabilistic Time-Series Forecasting, arXiv2609.05561，2026-09-03预印本。仅作者摘要/检索，发生在首轮成绩后。统计锚点混合与递归不确定性传播已有近邻；未核全部方程或提出第二候选。https://arxiv.org/abs/2609.05561

## 本候选的新颖性边界

Gaussian sine期望是已有恒等式。INTEGRATED与FREE_DAMP在d=a时逐值相同，候选是自由衰减循环单元的参数绑定子模型，不是已证明的新通用函数类。允许由已有部件形成有限样本结构贡献，但必须有匹配对照的独立作用；本轮尚未通过。

工作后验只对选定估计模型条件化，忽略模型选择和参数不确定性。把其方差直接叫观测噪声、真实经济状态的不确定性或可信预测区间均不成立。递推使用局部提议函数的期望，并没有精确传播整个随机隐藏轨迹。以上限制不因代数测试通过而解除。
