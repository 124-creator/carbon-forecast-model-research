# 完整循环骨干与期限响应固定试验

来源先于实现：父实际gh search NeuralLaplace并read samholt/NeuralLaplace README，固定HEAD4cc654161f61c776054c594c4f545a954083b25a。作者README说明隐状态参数化时间响应，通过ILT重构；未读全论文/核心实现，不能称完整查新。状态条件时间响应有明确近邻，候选不登记首创。前述衰减响应仅待证small-data结构估计假设。

价格only广州950截至2021精确前缀，451同yearH20，459初始、40刷新、最多480完整成熟训练起点、输入60差分。所有5模型共同样本、训练尺度、原价净变化loss全20步、同purged时间验证。不开外生分支，已有煤FX未建立稳定增量，不意味着都无用。

模型GRUfree/LSTMfree/GRUfixed/GRUglobal/GRUadaptive。原生nn.GRU/nn.LSTM，1layer hidden16；free线性20输出，响应为b*h+a*(1-exp(-h/tau))，h=0硬锚0，tau固定6/全局可学习/[2,60]状态条件。不称参数完全相同；free头更多参数，报告counts。固定与global是必备消融，不能只胜free算机制证据。

种子17/29固定，无种子挑选。AdamW lr.003 wd.001 max200updates patience20，全批，梯度clip1，内部最后60成熟窗口验证，purge20。输入和目标同一个训练创新std缩放以保留全20原价MSE权重，不逐h方差加权。step0作为可选择checkpoint，报告是否选初始但不当stationarity唯一门。最大记录成功fit累积800秒，未完成如实partial，不会合并不同epoch配置。全部fitlogs每完成一次落盘，数值参数仅本地不用pickle。未来扰动首块代表必须真实refit。

主score等年等h1/5/20，all20附加，yearcells全部。开发区已重复用，不是未见测试。新机制必须胜fixed/global/free和强MLE/state历史非本轮重训才支持扩展；若没有差异不救参数。本轮是候选开发筛查，不保证论文。
