# 固定三市场历史训练试验：实际执行记录

原设计 DESIGN_ZH.md 已完整读取并原样迁移；它的“design only”等历史状态不是本次执行状态。仅实现设计原有小型三块多尺度骨干、低秩私有响应A和时间顺序适应B，没有第三个模块。创新仍为假设，全文查新缺口未解决，不声称first。

## 完成范围
- 5模型 × 2seed（17、29）× 12个滚动重估块 = 120次正式从头训练；另有1次全候选未来扰动真实重训。
- 每40观察点刷新，全部451个2020—2021广州同年H20起点（2020:224；2021:227）均有2seed完整20步预测。
- 实际累计fit时间49.389秒，预算上限900秒；预算包含真实扰动重训，未耗满预算，不说明优化充分。
- 最初完整成熟池广州379、湖北405、EU400。预留40个成熟验证起点并清除20个重叠端点，实际优化池319、345、340。来源同样预留尾部，验证分数仅广州；固定一个配置，不根据开发分数选模型/参数。
- 各市场MAD差分尺度只在清除后训练标签范围估计，不用整个prefix，更不使用全历史尺度。第一次尺度末价格索引398、424、419。
- source标签日期严格早于广州fit最后标签日期；snapshot历史模拟不是real-time。EU产品、单位、venue未认证。数据引用Fan (2025), DOI 10.6084/m9.figshare.28682348.v1，历史CC BY4元数据；本轮未重新联网认证。

## 固定预算的准确含义与局限
日志字段gradient_calls实际计数是每市场、每目标的一阶梯度计算贡献/目标评估次数，而不是torch.autograd调用次数。每次fit均120个此类目标：独立120次单市场更新，普通三市场池化40次市场等权更新；episodic19轮×(3 support+3 query)+6目标私有适应=120。独立/普通/episodic实际反向或autograd调用次数分别120/40/82；optimizer步数也不相等，不声称相同FLOPs或相同墙钟。每个目标batch≤64；共同900秒累积fit上限。该预算匹配是受限先导，不足以排除迭代数、优化器和优化程度混杂。

B使用明确的一阶近似：support仅更新私有参数，query通过detach后的私有更新量计算共享骨干及初始化梯度；最终6步私有适应仅广州训练池，不使用validation。历史support端点严格早于query起点。静态私有控制省略Q，参数量略少，不声称精确参数匹配。训练池总参数：独立/pooled5663；static6131（私有468）；dynamic6203（私有540，含三市场分配）。

## 结果（年份×全部20步MSE/RW比例等权平均，种子平均损失）
|模型|比例|
|---|---:|
|独立广州|1.077758|
|普通pooled shared|1.000062|
|shared+staticprivate|1.004022|
|dynamicprivate去episodic|1.001444|
|dynamicprivate+episodic|1.001786|

完整模型没有超过普通pooling或RW。h1/5/20逐年原价MAE/RMSE均在scores.json中；2020完整候选RMSE为0.430918/0.605963/0.947293，2021为0.650076/1.303364/3.862447。所有20步逐年明细保留，未平均预测后评分。该结果不支持本次候选增量，不能宣称永久否定机制或停止整体研究。

## 验证与文件
- test_contract.py先真实失败（实现不存在），实现后通过。
- verify.py和python -O verify.py均核验120训练记录、模型权重文件、完整预测、原价标签、源截止和200个year×h评分行；normal/O输出分别保存verify_normal.json、verify_optimized.json。
- 第一个dynamic_episodic seed17 fit之后，各市场允许fit端点之后的全部内存价格受到大幅扰动；真实重新训练后的参数SHA256和截止内签发预测完全一致。未来source推理不输入任何外市场价格。禁止区2022未被读取；旧历史审计接触过2022日期边界的事实不撤回。
- 已保存每次fit的loss日志、seed权重、fit_receipts.json、causality.json、protocol.json、budget.json、predictions_LOCAL_ONLY.npz、scores.json、execution.log和复制的设计/资格/查新记录。未加载任何旧权重，未安装或修改官方环境包，未提交或push。
- MLE、stronglinear仅可作为既有历史对照，明确不是本轮重训；本次未伪造其分数，也未声称完成设计列出的全部独立强基线、逐源敏感性或所有扩展消融。

执行环境：repo/research/official_nhits_baseline_20261014/.venv/Scripts/python.exe。父任务可直接执行verify.py或run.py --replay；run.py拒绝覆盖已有结果，重新训练需另设新的输出目录而不能覆盖本轮。
