# 成交量条件创新分配：部分 v1 结果之后的新开发扩展

2026-10-19。该候选在已看到广州／湖北部分 v1 开发结果后提出，是**结果之后的新开发假说**，不是 v1 拟合前预注册内容，也不是 v1 的数值修复。原 `innovation_models.py` 与已启动的 v1 模型／结果保持原样。新模块为 [liquidity_models.py](../source/liquidity_models.py)；正式数据准备、拟合与共同基线由另一个运行器负责。

## 机制与完整公式

沿用 [v1 设计](SHOCK_ALLOCATION_DESIGN_ZH.md) 的当前价格创新、永久／两个暂时状态、全局衰减系数、能源系数与解析 20 期均值读出。唯一的新算子是把 16 单元 MLP 改成由两类已观察信息驱动的紧凑门：

\[
s_t=\log(1+|e_t|),\qquad
q_{t,k}=2\tanh\{\gamma_k s_t-\delta_k v_t m_t\},
\]

其中 \(v_t\) 是当前 \(\log(1+V_t)\) 相对**此前 60 个碳价记录内有效量记录**的 median／standard-deviation 标准化异常度，\(m_t\in\{0,1\}\) 是该特征有效掩码。统一扩展运行器要求至少 10 个过去有效量记录，标准差下限 0.25，标准化后截到 [−5,5]。这些是本轮新输入定义，不是通过结果选择的值。具体成交量字段、零量／非交易记录及发布时间由数据协议核定；模型不重新计算这些统计量，也不把缺失填充值当真实零成交。缺失时运行器提供 \(v_t=0,m_t=0\)，门只保留创新幅度项。

64 个输入 token 中，每个量异常度可额外用该 token 以前 60 个目标记录，所以底层量历史最多回看至末端以前第 123 个记录；不能把它描述为严格只观察 64 个原始记录。同样的量特征交给所有强对照。

约束版本用 \(\gamma_k=\operatorname{softplus}(g_k)>0\)、\(\delta_k=\operatorname{softplus}(d_k)>0\)。取 logits \((0,b_1+q_{t,1},b_2+q_{t,2})\) 的 softmax，将暂时份额乘 0.95、永久份额取余数；再沿用

\[
e_t=r_t-(\rho-1)^\top c_{t-1}-\beta^\top u_t,\qquad
c_t=\rho\odot c_{t-1}+w_t e_t,
\qquad \widehat m_{o,h}=(\rho^h-1)^\top c_o.
\]

基类本来就执行 `2*tanh`，所以自定义 gate 只返回括号内的线性式，**没有重复施加 tanh**。变量以规范化原价变化计量，训练与评分继续使用共同的成熟原价 h1/h5/h20 风险，不另加重建目标、成交量预测目标或未成熟标签。

## 经济符号的含义与缺陷

固定成交量时，正 \(\gamma\) 使较大的绝对创新提高暂时分量 logits；固定创新幅度时，正 \(\delta\) 使更高的成交量异常度降低暂时 logits。总暂时份额因此分别增加／减少；两个暂时模态之间还会相互竞争，不能简单声称每一个暂时份额都分别单调。

“放量的价格变化更持久”及“大创新在量未增加时更可能回落”是可以被碳价数据否定的模型假说，不是由市场效率或成交量定义推出的定理。价格与成交量来自同一天也不等于同一合约或同一披露时点；数据侧必须核对同源对应关系。过去曝光的开发年份不会因为增加成交量而变成新盲测。

v1 的条件均值解释在这里还需加强：目标未来创新须在给定过去、当期能源和**当期成交量**后条件对称，能源变化仍须满足相应条件零均值；否则幅度／成交量依赖的门可能使 \(E[w_te_t]\ne0\)。现实未被认证满足这些假设时，输出仍是受限零新息预测规则。孤立目标创新不能继续同向扩大的 v1 结构限制也仍存在。

## 对照、初始化与容量

| 版本 | 条件门 | 暂时状态／读出位置 | 参数数（3 能源） |
|---|---|---|---:|
| `full` | 正 gamma、正 delta | 每个历史观察分配 | 11 |
| `readout_gate` | 完全相同的四个门参数 | 常数库存，只在最后调幅 | 11 |
| `shock_size_only` | 两个正 gamma，无 delta 参数 | 每个历史观察分配 | 9 |
| `unconstrained` | 两组自由符号系数 | 每个历史观察分配 | 11 |
| `constant_allocation` | 无条件门 | v1 的常数分配 | 7 |

全局参数仍为两衰减系数、两个分配 logits、三个能源系数，共 7；没有残留被替换 MLP 的参数。full 对 readout 是同容量位置控制；full 对 shock-size-only 检验新增量信息／两个系数；full 对 unconstrained 检验经济符号限制。更少参数的消融不冒称同容量。强 GRU／LSTM／Ridge／HGB 应取得相同有效成交量信息。

有效 gamma／delta 均初始化为约 0.1。softplus 在有限参数下严格为正，因此 full 与 readout 虽然初始参数相同，其初始预测一般不同；也不与常数臂精确相同。这个差异会由检查明确记录，不能沿用 v1 的初始预测完全相等结论。unconstrained 用经过相同初始有效系数的仿射参数化，在相同 dtype 下初始 full 输出逐值一致，同时训练后允许系数跨过 0；这不是把 softplus 的导数作为自由版本的导数。

## API、复杂度与资格检查

```python
from liquidity_models import LiquidityAllocationForecaster

model = LiquidityAllocationForecaster(n_exog=3, horizon=20, variant="full")
prediction = model(x)  # x: (B,64,6), output: (B,20)
# x columns: target change, three energy changes, volume surprise, valid mask
# Alternatively: model(r[B,L], energy[B,L,3], volume_context[B,L,2])
```

统一 runner 的别名 `liquidity_full`、`liquidity_readout`、`liquidity_unconstrained` 也直接支持；`shock_size_only` 不变，别名不表示额外配置。

模型不读取日期、数据文件或未来变量。成交量 context 必须有限、valid 必须为 0/1；缺失值的含义由 mask 控制。运算量为 \(O(BL(D+K)+BHK)\)，K=2，没有 MLP、矩阵分解或抽样传播。相对于千条级日／报价数据，11 参数降低了估计方差的潜在成本，仍不能预先保证有效。

必要检查为：与独立 NumPy 标量门／递推逐值比较；gamma/delta 的方向导数；有效量变大时总暂时份额下降的约束；缺失 mask 使 volume placeholder 无效；取消符号约束后确实允许反号；所有训练参数产生真实梯度并改变；full 与 readout 的初始差异及有条件历史区别。检查只使用合成输入，不算市场拟合，也不设置任意“必须改善 1%”的合成性能门。
