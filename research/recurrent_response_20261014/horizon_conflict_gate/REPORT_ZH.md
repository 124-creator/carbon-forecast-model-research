# 一页机制立项实际结论：冲突投影假设不升级

## 事实而非创新包装
本人真实核读PCGrad作者源码，再冻结horizon-group及norm-matched修改，运行24新fit/2seed/12刷新；plain使用此前逐点审核GRU预测。本轮不声称新方法，PCGrad已有冲突投影。[1] 本地PyTorch转写有范数匹配改动，不是作者TF原样网络或完整实验复现。

实际模型仍原GRU60diff/H20，无新监督/分解。h1、h2..5、h6..20三组以原目标分母N*20计算，共享原loss期限权重；投影后匹配未投影sum梯度范数，再相同clip1。额外三次autograd与原一次不同计算，不伪称公平FLOPs。norm matching检查normal/O通过。

## 真实结果
plain .9753725253647837，projected .9768725393802057，未改善。共享encoder1601/4008组对负cos=39.94510978%，这些训练step并非独立样本，该比例是描述性，不作为显著性或负迁移因果证据。总1336optimizer steps/4008autograd calls，24完整fits，451主起点。正式fit99.4572411秒不含额外future refit。保存全部epoch validation、grad cos及选step，不用最后trainloss冒充外推。

零梯度/no-conflict/冲突数值fixture通过；代表seed17初始from459未来扰动重fit参数/scale和截至458预测一致。normal/O独立评分、时间purge、计数、范数匹配检查通过。没有2022price，没有子agent，没有新外生信息。

## 决策
存在冲突不意味着它是成绩不足原因；在本固定norm-matched干预下，没有主预测收益证据。不能说PCGrad所有变体都失败或共享状态绝对正确，但不足以把冲突管理作为核心新算法立项。停止本具体分组/干预，不追加adaptive weights/新分组/阈值救结果。若以后提出其他规则，必须有不同问题及直接证据，而非这轮失败后换名称。

这次真正执行了“具体问题→冻结已有机制反证→判断”的流程，但没有创造新算法。方法贡献门未打开、效果门未过，无创新/未见验证/发表资格。未把失败改阴性论文，目标不变。本輪并不承诺下一轮一定成功。

## Sources
[1] https://github.com/tianheyu927/PCGrad/blob/c5fbd7c856526373828074f06875230f7f3ee79e/PCGrad_tf.py
