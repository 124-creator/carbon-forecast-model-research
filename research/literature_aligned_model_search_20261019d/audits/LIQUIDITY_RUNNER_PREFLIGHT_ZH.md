# 成交量扩展运行器预检

**PASS_READ_ONLY_PREFLIGHT**。未发现阻断该固定输入与本轮启动方式的错误。此次只读源码和输入、独立重建量特征，并检查工作进程的 hook；新增市场拟合与预测评分均为 0。本文件在扩展已经由主代理及模型代理完成先验检查并启动后归档，不冒称新增事前注册。完整机器回执为 `LIQUIDITY_RUNNER_PREFLIGHT.json`。

## 执行入口与源身份

从 `source/run_liquidity_extension.py` 作为脚本启动时，其模块导入会替换本进程内 `run_model_search` 的 `prepare`、`make_model`、`freeze`、市场列表及六个神经家族。原 v1 源文件保持不变。`BASE_PREPARE` 与 `BASE_FREEZE` 在替换前保存原函数，避免递归调用；基类函数查找的全局 hooks 指向扩展实现。

独立审查分别创建了 **fork 与 spawn** 单工作进程，核对这三个函数身份、两市场、六家族及辅助目录，均与父进程一致。`--auxiliary-root` 在建进程池前转为绝对路径并放入 `CARBON_VOLUME_AUX_DIR`，spawn 导入可读取同一目录。没有依赖工作进程继承临时交互式变量。

| 文件 | SHA-256 |
|---|---|
| v1 `run_model_search.py` | `26347433fe6416298e4969da618cbba291999f3a66131172bc8f2e510f3633bf` |
| v1 `innovation_models.py` | `cca83f45474c5110f180ae8b7763131f33596a664939c4ebd34c32db24b4dc1e` |
| `run_liquidity_extension.py` | `95665858a1945b00bba6d6233022c34c0a58993899ccadda8c895f64ffe69807` |
| `liquidity_models.py` | `91dd66c0b33b8d75100e37cc382f01be44021ae538f6cbbedddda7f9b9386e0b` |

上述哈希在本次预检前后相同。模型公式和合成梯度检查另见 `LIQUIDITY_MODEL_CHECK.json`；本运行器审查不代替该检查。

## 数据、有效性与标签成熟

独立标量实现重新计算了每一行量异常度：当前 `log1p(volume)` 减去此前最多 60 个价格记录内有效量的中位数，除以该历史样本总体标准差与 0.25 的最大值，截到 [−5,5]，并要求至少 10 个历史有效量记录。四个市场年度单元全部量特征与运行器 **逐值差为 0**。

广州 1,434 行，量有效 1,434 行，具备足够历史的 context 为 1,424 行；湖北 1,408 行，量有效 1,378 行，可用 context 为 1,368 行。湖北 30 个无效量行保留为 NA，不删旧价格行：23 个无对应日期、3 个重复日期、4 个价格冲突。独立检查证实所有 `valid_volume=1` 都有唯一对应日期、`price_match=1`、有限非负量，无效行未填造量。

运行器读取已记录哈希的 `valid_volume` 标记；它自身没有重新审计作者工作簿的价格匹配。因此本 PASS 针对这两份固定、已审计辅助文件。复现时若更换输入，必须重新执行对齐和有效性审计，不能仅改文件路径。

六通道输入的前四列与 v1 原数组逐值相等，目标、train/val/final/test 掩码完全相等。量及掩码为新增两列；其可用性来自当前日已报告成交量的假设。原始作者字段的单位、准确公布时间和历史版本未认证，不能把收盘后假设表述为日内提前可得。全部输入有限。

| 市场／预测年 | 选择训练起点 | 验证起点 | 最终重训起点 | 测试起点 |
|---|---:|---:|---:|---:|
| 广州 2022 | 766 | 80 | 866 | 222 |
| 广州 2023 | 1,008 | 80 | 1,108 | 222 |
| 湖北 2022 | 752 | 80 | 852 | 217 |
| 湖北 2023 | 989 | 80 | 1,089 | 215 |

选择训练最后 H20 标签严格早于首个验证 origin；验证和最终重训最后标签严格早于首个测试 origin。未来量扰动不改变过去输入的检查已由主代理保存于 `LIQUIDITY_INPUT_PREFLIGHT.json`，四个标量重建单元补充验证了固定输入规则。64 个 token 中每个量 token 还使用此前 60 行作参考，量信息最远可到预测 origin 之前 123 行，不能说原始信息只覆盖 64 行。

## 比较与计数边界

四个紧凑门分别为 constrained full、同容量 readout、shock-only 和 unconstrained。GRU/LSTM 的输入为相同六列；`ridge_full` 取得 64 期六列、`hgb_full` 取得 20 期六列，`ridge_price` 仍是 20 期价格基线。共同信息截止时点相同，不应声称各家族容量与历史长度都相同。沿用 v1 constant 结果作为额外控制时，必须匹配同一市场、年度、origin 和 horizon，不能并入新拟合计数。

计划的 120 是 `2 市场 × 2 年 × 6 神经家族 × (2 次学习率选择 + 3 次最终重训)` 的调用数；进程池有 28 个 job，其中 4 个为经典方法 job。最终若选 epoch 0，对应重训没有梯度更新，仍须按实际日志披露。另有 132 次经典 `.fit` 调用（Ridge 72、单 horizon HGB 60）；RW 与 drift 不属于梯度拟合。

协议提出的四紧凑模型固定初始化评分不在共享主循环里另行输出，须由单独的无拟合诊断完成；不能用“某些家族选到 epoch 0”替代所有四个固定先验的比较。该项不影响已启动训练的输入与优化正确性，但应在完整交付中兑现。

本版本是在部分 v1 结果之后提出的新开发假说。2022/2023 年标签此前已参与开发；增加成交量不会产生新的独立市场，也不把既有年份变成新盲测。

## 复现入口

在完整仓库和所述依赖环境中，从本研究目录启动。所有线程设为 1；`--out` 应指定全新的私有目录。运行器会拒绝已有 `PROTOCOL_BEFORE_RUN.json` 的输出，但不会把一个包含其他文件的目录自动清空，因此调用者应使用真正的新目录。

```bash
env PYTHONPATH=/tmp/carbon-forecast-20261019d-deps \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/carbon-research-env/bin/python source/run_liquidity_extension.py \
  --auxiliary-root /tmp/carbon-literature-data-audit-20261019d/prepared \
  --markets GUANGZHOU HUBEI --years 2022 2023 --workers 4 \
  --out /tmp/carbon-liquidity-reproduction-new
```

以上为未来完整复现实验命令；本审计未执行它。本次实际执行的是 `audits/check_liquidity_runner_preflight.py`，其 `--out` 同样拒绝覆盖旧审查回执。
