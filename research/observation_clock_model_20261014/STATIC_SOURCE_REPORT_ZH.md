# 独立静态源码与移植范围报告

## 证据
已独立读取固定作者 `models/SegRNN.py` 全203行、Apache-2.0 LICENSE全文、README主要结构与作者引用、上一审设计/收据。固定提交 `8e869ecfdf1daab3a0ba14d1d620796c1a5d2c4f`；源码SHA256 `b7dd6ba15600fbb67f5137e58d45795c75a13a1661bc615ffe9c7b145bf1da95`，Git blob `f1e168e95a67fea1852c2747b38b4fc487434e96`。本地六份作者对象与上一审SHA256全部一致。模型、许可、README独立blob重算记录于SOURCE_REVIEW.json。没有执行、导入作者模型/runner/loader，没有权重。论文正文和直接先例正文仍为上一审blocked，不能声称全文精读或首创。

LICENSE随本目录保留，model.py有上游作者/提交归属及修改声明；完整上游tree的NOTICE尚未审查，不能据此宣布完整再发布合规审计。代码许可不证明论文、市场数据、权重许可。

## 精确保留的数学结构
- 单变量、revin=False、channel_id=False、GRU、PMF子集。
- 60点末值中心化，按12×5保留段内位置；5→16线性+ReLU；一个共享单层GRU编码12段。
- PyTorch GRU的r/z/n门形式、两个bias、reset作用在循环候选项（含bias）、`(1-z)n+zh`，由numpy/autograd重写。
- 编码/解码同一套GRU权重；解码输入选原4段位置embedding，对h=1/5/20分别选择段0/0/3及输出偏移0/4/4；16→5输出。
- 公共gap输入以5→16投影加入段embedding的ReLU之前。候选q(h,k)为共享编码状态的16→9线性头；类别代表值投影进相应段查询；预测Σqμ。不输入未来真实gap。

## 与作者实现不同，不能叫完整复现
- 本环境torch缺失。无需安装，使用已装numpy/autograd；没有作者torch同权重数值比对。结构数学移植不是论文性能复现。
- 只返回3个监督预测增量，不返回完整20点价格；输入尺度与训练标签尺度由合成训练段各自估计。末价恢复可由`last+increment`完成，但模型函数直接返回增量。
- 正态fan-in初始化/零bias不同于作者PyTorch默认初始化；没有RNN/LSTM/RMF、多变量channel embedding/RevIN组件。
- 所有模式实际计算9个解码查询，包括price-only/gap的重复查询；对应相同h的无条件输出相等，和原4段PMF所选输出函数相同，但计算更浪费。训练dropout .1在一次调用中对9查询共用batch×16 mask，不是上游独立query dropout。必须披露，不能声称训练完全复现。
- 候选按每个h复制段查询、3个类别，共9查询，没有独立类专家。公共编码12步；每样本21个GRU单步等效调用，9查询每个5输出；未测真实FLOPs，仅报告查询/参数/墙钟。
- 所有模式分配同字典，但original休眠gap/clock/q，gap休眠clock/q，auxiliary/no_condition休眠clock；实际可训练容量不相等。总参数不能冒称有效参数相等。
- auxiliary与no_condition当前固定方差条件MSE/CE损失完全等价，无条件类均值相等。两者相同初始化/预算得到相同预测，是强消融的代数重复，不构成两份独立支持。
- 合成类别support由生成器明确给出，不实现市场内折分位/小支持合并；没有市场rolling loader/成熟标签/日期认证/强线性比较。语义门未过，故不开展这些市场步骤。

## 最先可否证的是任务而非分数
未来工作周、提前公告的休市计划可为签发可知信息；未知记录选择与未来日历未知不能互换。此次广州历史日历搜索失败，尝试全国公告正文被安全屏障拒绝，均有收据；仅生态环境部官方PDF搜索索引出现“交易时段由交易机构公布”，不认证广州日期。既有作者快照含周末与缺项，日期不认证交易日；没有证据将缺报价解释为流动性、无成交或经济钟。语义门=不通过/未认证，市场执行=禁止。

合成known_calendar中计划由episode编号确定且同一过去gap揭示，所有gap模型信息相同；不把“使用已知计划”称未知时钟边缘化贡献。unknown_state里未来类随机但概率随过去可识别状态变化；这是设计出来的机制，不是市场事实。即使正控成功，直接均值模型理论上也可以学习同一条件期望，潜在收益是有限样本结构/目标假设，不是新增信息，也不是已有GRU mixture的创新认证。

## 复核命令
在本目录：`python test_model.py`、`python -O test_model.py`、`python verify_saved.py`、`python -O verify_saved.py`。重新训练不是父级验收必需，若执行`python run_synthetic.py`会覆盖本目录同名结果，当前全部首次结果已保留，不执行二次挽救。
