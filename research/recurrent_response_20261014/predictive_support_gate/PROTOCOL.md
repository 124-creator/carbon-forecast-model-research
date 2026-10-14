# 固定开发实验：预测支持子空间与循环参数更新

状态：候选机制验证，非成立创新。2026-10-14，在新训练前冻结。此版本为预设计，近邻检索尚未执行。上一版误写了尚未取得的访问状态，现撤回；训练必须等待实际检索完成并记录结果，不追认前版为成功核读。
来源：GPM https://arxiv.org/abs/2103.09762 ，PSD https://papers.neurips.cc/paper/6717-predictive-state-decoders-encoding-the-future-into-recurrent-networks.pdf ，空间交集近邻 https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Rethinking_Gradient_Projection_Continual_Learning_Stability__Plasticity_Feature_Space_CVPR_2023_paper.pdf 。未来监督、SVD和梯度投影本身已有；下面只有一个待证规则，不认证新颖性。

复用本轮已审核GRUfree的17/29种子、12个刷新块的数值NPZ初始化（allow_pickle=False）；不加载pickle，不使用任何旧实验checkpoint。训练前从950条许可前缀重建数组、目标和尺度。451个开发起点，h1/5/20观测步；2020/2021六格原价MSE/RW等权。
在每次训练截止内，将原训练子集分成早/晚两块，二者之间purge20个起点。固定初始化的隐藏状态到20期限净变化的岭映射，ridge=0.1，取左奇异向量最多3维。晚块支持作为single；早晚主夹角cos>=0.9的方向构成shared（取晚块侧主向量）。不采用外层评分定阈值、秩或岭参数；子空间几何相同不意味着未来响应方向/系数稳定，必须保留符号反转反例。
用16个等间距训练样本的平均末状态对全部encoder参数的Jacobian，将支持方向拉回参数空间并用SVD提取非零奇异值对应的正交基。初始协议写QR，实际实现采用SVD，此处按独立审查纠正；没有改变算法或重新训练。只在该固定低秩空间内更新encoder；head使用普通更新。该Jacobian是平均状态的局部映射，不是全部样本或全路径的可识别性证明，也不是天然坐标不变估计。
四模式：raw普通SGD、single单块投影、shared跨块支持投影、freeze冻结encoder。每个分支独立复制同一初始化，用SGD lr0.003，无动量/权重衰减，最多20步；共同原20期限MSE，梯度范数裁剪1。投影非零时匹配原encoder梯度范数，秩零则冻结，不补随机方向；head更新相同规则。按原验证区选择模型，允许选第0步，禁止外层调参。这是额外微调对照，不是从头训练原AdamW的替代。
总计96段新微调；普通原GRU无微调作为第五参考。披露Jacobian额外计算；预算不称同等FLOPs。总计训练预算600秒，不追加种子/阈值扫描。为seed17/block0四个模式实际重算未来扰动下支持、Jacobian和微调，比较发行458预测及状态；这只认证额外机制，不追认全部初始化重训。
门：先测试精确交集、零秩、正交子空间、符号反转、Jacobian有限差分；若实现核验失败不评分。市场候选只有共同分数优于raw、single、freeze及原初始化，且每年总体方向都改善，才允许继续研究；并须最终面对已有强预测组合和近邻贡献审核。否则停止该固定规则，不调整阈值/秩救成绩。
