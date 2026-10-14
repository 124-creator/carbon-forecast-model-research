# 一页机制立项与冻结实验：多期限共享状态梯度冲突

状态：假设未证，不是成立创新。本人真实取得PCGrad官方README/完整PCGrad_tf.py，commitc5fbd7c856526373828074f06875230f7f3ee79e、MIT，再本契约。方法对冲突task gradient投影已有，不能首创；本轮仅必要机制反证，不承诺其效果或与碳价任务直接匹配。

问题：原20h共享GRU，任务梯度可能相互抵消。假设是这种冲突限制原价预测，不是只观察负cos就宣布原因。学习对象仍既有GRU编码器与20h输出，未新分解/外生/标签。

固定对照：原普通GRU此前逐点审核预测作为plain；新增norm-matched-PCGrad，两seed17/29×12刷新=24正式fit。input60diff、H20、原base.arrays训练窗口/scale/purge不改；AdamW.003/.001、clip1、max200/patience20、仅主验证选择step，全部保持。任务固定h1/h2..5/h6..20，每task损失sum/(N*20)，三项相加严格等价原main平均loss，不改变期限权重。

每step三个task梯度，既计算共享encoder和全参数cos。以同seed固定NumPy随机排列投影顺序，对每task负内积部分沿其他原taskgradient投影。sum后将范数匹配未投影三task梯度和，再原clip1。normmatch不是作者原样PCGrad，需披露，不声称原作者完整复现。共三次autograd代替原一次，计算不同；maxepoch相同不等于同actualstep/FLOPs。记录原norm、投影norm、pairconflicts及所有loss。

正控/反例：零梯度、无冲突方向需保持原sum，冲突fixture不输出nonfinite；gradient normmatch满足定义。代表seed17初始future459扰动真实fit尺度/参数/截至458预测不变。normal/O独立统计、原价主year×h1/5/20两seedloss均权451核验。

机制判定：负cos仅描述，不证明负迁移；norm-matched对照若不改善，不以新分组/学习率救结果。即使胜plain仍非创新，需超历史强统计与强简单组合（本轮历史比较不混称公平重训），后续独立算法必须区别成熟label支持、共享/私有结构或明确估计规则，不能把PCGrad改名。若失败结束该具体假设，不继续其系列门控。新方法尚不存在，不为凑论文宣称成功。

预算：正式fit600秒，逐块数值NPZallow_pickleFalse和完整日志保存；超预算记录不完整。开发仅950许可prefix截至2021，不读后缀，无子agent，无原始预测Gitpush。
