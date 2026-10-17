# SOLID 作者适配层与有限运行协议：独立前置审查

## 结论与范围

**结论：PASS，可以按最终冻结的 SOLID_B1 协议执行。** 本审查没有读取市场 CSV、市场模型权重、市场预测数组或确认期数值，没有执行市场训练。实际作者函数只在独立生成的合成数组上运行。正式预测的实际得分和市场逐点数组不在本次前置审查范围内。

执行证据见同目录 `SOLID_EXECUTION_INDEPENDENT_AUDIT.json`；可复核脚本为 `audit_solid_execution.py`。脚本运行了 **32 个合成查询、320 次合成邻居反向传播**，进行了 **513 项检查、12,580 个数值比较**；最大绝对误差 **4.440892098500626e-16**。没有新增骨干拟合。总运行时间约 2.59 秒，实际环境为 Python 3.12、NumPy 2.5.4、PyTorch 2.3.1+cpu，float64、单线程。

这份 PASS 表示时间边界、执行路径和所声明的公式相符，不能用于证明预测优势或方法创新。该实验仍是对已发表方法的本任务适配性检验。

## 固定来源与执行身份

| 对象 | SHA256 |
|---|---|
| 最终 `provenance/SOLID_B1_PROTOCOL.json` | `b44357f30ce5c7f50f2023ebb0fd3fcc17e1598038724738b0b64b56c7ca76fd` |
| `source/run_solid_reference.py` | `856f2fd24f8d5863dae533ca1bc1eeb4b1c9e981ea9675ed893b25679ec7bfea` |
| `source/solid_author_adapter.py` | `2e38d06e39928e771f8e09ed50add4f8ffb291ec2dd6918e2b7c6dc4469648ac` |
| `source/rrc_core.py` | `0ee05e9ad74062d7d5467e7a89efc601931be59c5737d2dea90cc9c9182c20d5` |
| 作者 `exp/exp_main.py` | `d0087fac4a2dd4b9d3b5650b7d9330715be12d371f203b8935f6440eb8777d56` |
| 作者 `LICENSE` | `74323c2d2119ab99a00c4d0bda4f72c4ff3e3aca49af17eaf62aec593f2af1b2` |

作者仓库为 [HALF111/calibration_CDS](https://github.com/HALF111/calibration_CDS/tree/3c74e5f8b3c921a2e7cda30f36117c295bf8c112)，固定提交 `3c74e5f8b3c921a2e7cda30f36117c295bf8c112`，MIT。`Exp_Main.select_with_distance` 从已核 Git blob 的作者文件提取，算法函数的 AST 未改动。数据、模型前向、内存 checkpoint、诊断和文件输出接口由本任务桥接。原作者的完整数据处理、模型架构和整套 benchmark 没有在这里复现。

原协议 `c6110f66ba887e9f083a0e54b5458ab99044f22363faf0389f6b07735623ba9c` 在正式执行前增加了 recency 精确等价复用附录；原字节另行保留。首次独立审计在旧协议 SHA 检查处停止，当时尚未执行任何合成操作或读取市场数据。审计随后针对最终固定协议重新运行；审计者没有改动训练源、作者适配层或协议。

## 时间边界与原作者更新语义

设当前查询起点为 (q)，预测长度为 (H=20)，候选池配置为 (C\in\{128,512\})。实际候选集合为

\[
\mathcal S_q=\{\max(60,q-H-C+1),\ldots,q-H\}.
\]

候选数量不足 10 时明确停止。每个历史样本的完整目标最晚到 (s+H\le q)，输入和按原点计算的波动尺度也只使用其自身已观察前缀。查询 (q) 的真实未来标签不传给作者函数；作者所需的查询目标接口返回零占位，依赖真实查询标签的诊断分支关闭，原作者 metric hook 也关闭。正式得分由运行器在预测完成后另行计算。

独立检查使用了会拒绝任何大于当前 (q) 索引访问的价格、日期间隔和协变量数组。该检查通过；将最大查询之后的价格、协变量及日期间隔大幅扰动，所有已签发合成预测逐位不变。

作者非 AMP 路径对每个查询复制同一个固定离线模型。编码器冻结，只对线性头求导。设 (z_s\) 为固定 GRU 特征，(f_s=W_0z_s+b_0\)，归一化标签为 (y_s\)。实际更新为

\[
G_W=\sum_{s\in N_q}\frac{2}{H}(f_s-y_s)z_s^\top,
\qquad
G_b=\sum_{s\in N_q}\frac{2}{H}(f_s-y_s),
\]

\[
W_q=W_0-\eta G_W,\qquad b_q=b_0-\eta G_b.
\]

这是在共同初始头部计算 10 个邻居梯度后求和，再进行一次手工更新。没有除以邻居数 10，也没有逐邻居更新参数。本次原作者 `test_train_epochs=1`、period=1、非 AMP 的设定已核对。原始循环实际被调用，梯度 hooks 捕获其反向传播，适配后头部与记录梯度总和相符。

独立 NumPy 实现从 GRU 的 reset/update/new 门开始重算隐藏状态，再按上式计算全部梯度、头参数、归一化预测以及

\[
\widehat p_{q+h}=p_q+\sigma_q\sqrt h\,(W_qz_q+b_q)_h.
\]

计算还保留了作者 `pairwise_distance` 的默认 (10^{-6}\) epsilon 约定。每个合成查询的上述数组都与实际作者路径相符。cached 与 uncached 的预测和梯度逐位一致；多查询流中的每个预测与独立单查询重置运行逐位一致；原离线模型及冻结编码器保持不变。

## recency 等价复用与真实计数

period=1 时所有整数相位差均通过筛选。固定 (k=10)，只要两个候选池都包含至少 10 个成熟起点，recency 排序均选择相同的有序集合

\[
q-20,q-21,\ldots,q-29.
\]

它们使用相同的原始模型、输入、标签、尺度、学习率以及求和顺序；候选池大小不会进入 MSE 分母。因而 context128 与 context512 的适配头和预测完全相同。独立合成检查也得到了逐位相同的结果。

最终协议据此保留所有 **80 条配置预测曲线、480 个市场/年份/期限/seed 指标格**，实际执行 **60 条作者适配流**，另有 **20 条明确标注的等价派生曲线**。协议声明的正式预算为 **25,332 次 query-local 头部更新、253,320 次邻居反向传播，0 次新增骨干预训练**。这些总查询计数来自冻结协议，本前置审计没有打开市场数据重新计算其日期数量。

运行器对派生曲线写出 `equivalent_prediction_from`，并将私有适配状态保存为指向实际执行数组的 ALIAS 记录；不会捏造第二次梯度调用。候选池大小、编码器缓存次数等执行元数据不可假称相同。本次等价结论只支持预测和适配参数复用。

## 正式运行器检查与解释边界

运行器只引用已批准的 development 文件，设置数值截断日期为 2024-01-01；加载的模型必须来自 D1 离线 `pretrain_shrink*.npz` 中的 `T.` 状态，输入文件 SHA、checkpoint SHA 和实际 teacher state hash 都需要匹配。每条适配流执行后再次检查原 teacher 没有被修改。公开 receipt 使用明确字段白名单，逐查询梯度、适配头和预测数值保存在私有目录。

评分起点必须位于 2022 或 2023，且该起点的完整第 20 步目标仍处于同一年。该支持函数已用独立合成日期枚举检查；MSE、相同起点 RW 分母、相对 MSE、MAE、方向准确率和每格数量也用合成价格独立重算通过。正式数据上的 480 格指标及选择结果需要完成后的独立统计复核。

唯一 pilot 分支只取首 20 个查询，关闭预测得分；其市场执行收据不属于本次合成重跑。正式运行拒绝已有输出目录，遇到非有限预测、错误或过期 deadline 会停止并保留失败；没有自动追加配置或重训。deadline 是查询/前向处的协作检查，不是操作系统强制终止，正在进行的 checkpoint 读取或结果文件写入可能稍后返回。

本轮是已经反复使用的开发年份上的有限备选检查。只有既有方法的收益不能构成本方创新；执行正确也不能反推性能有效。原 RRC 失败结论和确认区限制保持原样。

## 复核入口

在具有固定作者文件和冻结源的 study 目录之外，也可明确指定路径运行：

~~~bash
PYTHONPATH=/path/to/carbon_deps python -B backup_review/audit_solid_execution.py \
  --study /absolute/path/to/event_feedback_innovation_20261017 \
  --output /absolute/path/to/new_synthetic_receipt.json
~~~

需要 NumPy、CPU PyTorch 和本 study 中的三份 source 及固定作者 `exp_main.py`/`LICENSE`；不需要任何市场文件或 checkpoint。脚本的协议 SHA 门槛绑定本次冻结版本，若来源改变，应另做明确的新版本审核，不能直接将本次 PASS 沿用到其他源代码。
