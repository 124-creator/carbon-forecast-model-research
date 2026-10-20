# 一次有界优化路线选择

## 结论
放弃当前手写 GRU 作为可信的可训练研究底座；保留本目录的 autograd + SciPy L-BFGS-B 作为诊断工具，不称模型创新，也不声明其市场有效。成熟优化器可以实际执行训练，但两种子没有同时优于现有方案，不能用继续加预算掩盖问题。没有访问市场数据，因此没有执行条件性市场刷新。

## 静态根因检查
历史源码只读来源：E:/碳价格/独立数据归档/carbon-forecast-model-research/research/segmented_price_backbone_20261014/model.py。本次工作目录与真实历史代码目录不同，使用绝对路径只读导入；源码 SHA256 和不变检查见 decision.json。

- 输入减末价格后分为 12 个五点片段。合成训练前缀统一尺度后标准差为 1，嵌入标准差 0.465/0.726，初始门 logits 超过绝对值 5 的比例 0/0.001275；未支持“大规模初始门饱和”这一解释。此诊断不代表市场门尺度已验证。
- 权重是按输入维数缩放的正态初始化，并非标准 torch GRU 初始化；位置向量为单位正态。保持相同初始化而不是额外搜索初始化，避免优化与架构改变混在一起。
- 每个 horizon 独立标准化，目标标准差设下限；损失按 horizon 有效观测数均值后平均。合成数据没有缺失；市场缺失 horizon 在小 batch 内仍存在分母为零风险，本轮未触发、未修补。
- 原方案为 8 batch/epoch，60 epoch/480 updates；全局梯度范数截断至 1 后加入 1e-4 参数项，再进入手写 Adam。这不是 AdamW，且截断后的数据梯度与参数项相对权重会变化。原实现来自 autograd 自动微分，不是手写反向传播；本轮没有反复修补它。
- 针对小样本、确定性可微目标，单一候选采用成熟 SciPy L-BFGS-B 全 batch、线搜索和曲率近似，去掉 minibatch 噪声、截断与耦合衰减。这是一个优化方案包，不是纯 optimizer 单因素消融。CPU torch 未安装，没有安装或假称执行 torch。

## 固定实验
复用原 nonlinear/null 生成器、种子 7/17、原架构与参数初始化。顺序划分 192 train /64 validation /128 test，尺度只由 train 拟合，两方案相同。测试集不参与选点。原方案固定预算终止；候选 maxiter=200/maxfun=400、gtol=1e-4、ftol=1e-10，并在 validation 连续 20 次没有超过 1e-5 的改进时停止、回取 validation 最佳参数。既报告停止类别，也报告梯度，不把下降称为收敛。

|任务|种子|原方案 test MSE|L-BFGS-B test MSE|候选停止|
|---|---:|---:|---:|---|
|nonlinear|7|0.01144358|0.01104238|validation patience，68次迭代|
|nonlinear|17|0.01212770|0.01213863|validation patience，65次迭代|
|null|7|0.01026202|0.01020426|validation patience，23次迭代|
|null|17|0.01121928|0.01110569|validation patience，24次迭代|

候选无信号两种子均通过不超过零预测 MSE 的 1.1 倍检查；正信号没有同时改进，因此拒绝升级底座。候选选点梯度 L2 分别为 0.0993/0.0868/0.0283/0.0851，并非 stationarity。原方案末梯度分别为 0.3155/0.1649/0.0700/0.0961。候选快且能够有效降低目标，但不是证实已解决市场训练根因。

成功完整运行 39.34 秒。首次执行出现日志字典 history 重复键，修正仅本次实验脚本后重新完成；没有修改历史 model.py。八个保存参数/预测文件重载后预测最大误差为 0；verification.json 验证八组齐全且 MSE 有限。未提交、未 push。未读取 2022 年或任何市场样本。

## 成熟优化依据
SciPy 官方文档明确给出 L-BFGS-B 的 gtol/ftol 以及 minimize 的 callback StopIteration 终止语义；实际环境 SciPy 1.17.1、NumPy 2.4.6。
参考：`https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html`、`https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html`。

## 边界
未进行 torch 方程数值对应验证；当前结论不是“所有 GRU 都应放弃”，而是停止为这份手写实现继续补预算或修优化。历史滚动 1.093/冻结 1.074 来自委托上下文，本轮不重新运行或改写其结果。条件性市场测试因正信号双种子门未通过而不执行。
