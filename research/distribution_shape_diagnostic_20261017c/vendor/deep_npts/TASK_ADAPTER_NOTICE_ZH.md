# DeepNPTS 原始核心与本任务适配边界

本目录保留 GluonTS 作者仓库 [awslabs/gluonts](https://github.com/awslabs/gluonts) 的固定提交 `d60556ca71167a3ec1ea807b491dd982d1de2112`。`_network.py`、`scaling.py`、`discrete_distribution.py`、`LICENSE` 和 `NOTICE` 的原始字节均通过固定 Git blob 身份核对；各文件的来源、字节数和 SHA256 见 `SOURCE_MANIFEST.json`。许可证为 Apache-2.0，原许可证和通知完整保留。

## 实际执行范围

实际执行的是原始 `DeepNPTSNetworkDiscrete` 网络及 `DiscreteDistribution` 分布类。本项目没有执行完整 GluonTS estimator、作者数据处理流水线或论文基准实验，也没有声称复现论文成绩。

`runtime.py` 在加载前校验上述五份原文件。它以局部导入映射连接原分布和缩放模块，把用于构造参数登记的 `validated` 装饰器替换为恒等装饰器。原构造函数、断言和网络/分布数学代码保持原字节。该兼容层不安装包，不注册全局 GluonTS 模块或全局导入钩子。它需要单独披露，不能描述为完全未适配的 GluonTS 运行环境。

## 共同的任务适配

两个作者核心诊断臂 `deepnpts_rps` 和 `deepnpts_crps` 采用相同适配。

1. 原作者网络保留两层宽度 32 的 MLP 及原始概率输出头，`use_softmax=False`。两臂均不是 GRU。
2. 原网络的 31 个 `past_target` 支撑点改为该预测期限训练前缀内固定的 FHS 支撑点。20 个期限直接分别预测；没有执行原作者递归多步路径。
3. 当前已观测的 `x[20,3]` 展平为 60 个数，尾部补两个零，再整理成 `31×2` 输入原 `past_time_feat` 槽位。这是特征槽位适配，不是未来日期特征。静态实数特征为 `h/20`，类别为唯一虚拟类别。
4. 原作者嵌入保存在普通 Python 列表，原优化器不会更新它。本封装注册同一个嵌入对象并冻结，保存原数值，使 dtype、device 和 checkpoint 状态完整；没有重新初始化它。
5. 对原始概率 `p_author` 使用固定混合下界 `p = 0.03/31 + 0.97*p_author`。随后按这个实际输出概率对支撑点进行加权重心与方差标准化，使每个预测分布均值为零、方差为一。这一步用于隔离分布形状，并避免极小概率把极远支撑点隐藏在方差中。下界与标准化都是本任务适配，不属于原作者实现。

## 配对的唯一训练评分差异

调用方必须在构造两臂前重置同一个随机种子。合成检查已验证，两臂初始 state_dict、支撑点和概率逐项相同。

- `deepnpts_rps`：调用原始 `DiscreteDistribution.rps`，作用于实际发出的、已经概率加权标准化的分布。
- `deepnpts_crps`：对同一分布使用有限离散分布的精确 CRPS。

所有正式验证和最终比较应采用共同的精确 CRPS，不能把原始 `rps` 值当作精确 CRPS或对数似然。`source/models.py` 的 `training_loss` 返回每个样本、每个期限的损失；外层训练器负责冻结的单位换算与聚合。

## 证据与局限

`provenance/MODELS_SYNTHETIC.json` 记录了原始源码评分反例、梯度、配对初态、分布矩、支撑点顺序和 checkpoint 往返检查。反例针对这一固定版本的源码评分，不外推为每个论文公式、软件版本或完整 estimator 都存在同样问题。精确 CRPS 本身是既有评分规则，采用它不是新的预测算法。

31 个有限支撑点、概率下界和严格正间隔限制了本诊断所能表达的形状；这不是任意连续密度或任意零点原子概率模型。一次形状实验的失败不能证明所有条件分布形状都不可预测。
