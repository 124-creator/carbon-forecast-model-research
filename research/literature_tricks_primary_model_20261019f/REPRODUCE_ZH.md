# 阶段 f：公开数值重放与重新训练说明

## 1. 可以直接复现什么

本目录提供 [source/replay_public_states.py](source/replay_public_states.py)：在完整 checkout 与两个同 SHA256 标准行情文件可用时，从公开的自行训练数值 JSON 重新生成本轮全部 **90 个确定性评价预测面板**，并逐项核对公开汇总。它不需要本会话的旧 `.pt`、旧 `.npz` 或原临时路径。

| 内容 | 评价预测面板数 | 数值来源 |
|---|---:|---|
| 三种 EWT 设置 × 五模型 × 两市场 | 30 | 阶段 f 的公开 `REFIT.json`；选中验证状态也重放 |
| 密集路径监督 × 四模型 × 两市场 | 8 | 阶段 f 的公开 `REFIT.json`；完整 H1…20 输出后取 H1/5/20 |
| 五种成熟读出策略 × 四模型 × 两市场 | 40 | 阶段 e 的公开训练选择及最终数值状态；按成熟规则重新求解读出 |
| 复用的原始响应、价格 Ridge 和 RW | 12 | 阶段 e 公开数值状态、Ridge 的 `SELECTION.json` 最终系数/尺度，及零变化 RW |

这是**数值预测重放**，不是重新执行 598 次离线拟合。在线读出是一种随成熟标签更新的预测器，重放它必须重复声明的 **13,632 次三维线性读出求解**；这些求解不更新原非线性参数。模型使用冻结实现的基函数与前向函数，因此也不把本入口称为独立算法重写。

EWT 的 `window_past`、`train_joint_test_window`、`full_series_joint` 信息范围保持原声明。尤其全序列分解仍属于完整序列可用的回顾基准，不会因为采用重放入口而变成逐时点预测。两个市场在阶段 f 都是已暴露的开发数据。

## 2. 冻结对象与依赖

完整实验计划为 [audits/EXPERIMENT_PLAN_BEFORE_RUN.json](audits/EXPERIMENT_PLAN_BEFORE_RUN.json)，SHA256：

`78893370f6862b1295ba9cfcc5e71fc89776944aeae39346de788a4e09cc8880`

本地拟合源冻结提交为 `9803953b7927d232608710aef21dafe870cf6cbc`，内容树为 `08e25bb2cd1ec3387e2c14c0ac8cd016dfbb89d3`。公开数值重放器与导出器在拟合源冻结之后编写，属于交付工具；它们不会改写这四个冻结拟合源。

必须使用完整 `carbon-research` checkout。阶段 f 引用阶段 e 的原模型与数据准备，阶段 e 还核对阶段 d 的源文件身份，不能只拷贝本目录后删除旧阶段。

实际运行环境是 Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0、PyTorch 2.14.1+cpu、pandas 2.2.3、scikit-learn 1.8.0、ewtpy 0.2 与 PyWavelets 1.8.0。拟合及重放使用 float64。ewtpy 的 `EWT1D` 源文件 SHA256 为 `35cb4532ae35a41867ee1411cb61a33165f7eebab3ebf70a58b47c74c6e6a289`，入口会核对这一实现。

本会话的解释器和额外依赖路径如下；其他机器应使用自己的已安装环境，不要求存在这些临时路径。

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=/tmp/carbon-forecast-20261019d-deps
```

## 3. 获取并核对相同数据

数据来自官方行情页面 <https://www.chinatcx.com.cn/tshq> 的历史快照。数据本体没有收入公开代码树；公开的提取器与来源记录在阶段 d/e 中。原始 Body.js 字节 SHA256 为 `c3e4c075832f6e5087cf416adecbae9d3c1b61859a6ad09fdf84294197393238`。

| 标准文件 | 记录数 | SHA256 |
|---|---:|---|
| `TIANJIN_OFFICIAL_MEAN.csv` | 1,077 | `85ca07648855db00fbf67c4e1c083db715f62ab829f74c143c3b3f67da21252a` |
| `CEA_OFFICIAL_MEAN.csv` | 1,251 | `f9f0e09b7500b2bdb4ec8b16c61d5d5fa2b64de63a24c0cf7bd1d269aa2be75d` |

如果持有本轮原始快照，可分别提取到新的研究用目录：

```bash
/tmp/carbon-research-env/bin/python \
  research/literature_aligned_model_search_20261019d/provenance/prepare_tianjin_official_snapshot.py \
  --raw-file /tmp/carbon-next-market-data-20261019d/tianjin_body.js \
  --out-dir /tmp/carbon-stage-f-replay-tianjin

/tmp/carbon-research-env/bin/python \
  research/signed_response_search_20261019e/provenance/prepare_cea_official_snapshot.py \
  --raw-file /tmp/carbon-next-market-data-20261019d/tianjin_body.js \
  --out-dir /tmp/carbon-stage-f-replay-cea
```

没有原始文件时，可省略 `--raw-file` 让提取器执行一次普通官方 GET，并检查返回字节。上游可能修改历史快照或停止服务；若 SHA 不符或访问失败，程序不会用新数据冒充旧快照。此时精确重放仍缺相同原始数据，必须单独登记新快照复现。

将两个同名标准 CSV 放入一个 checkout 之外的目录，例如 `/tmp/carbon-stage-f-replay-data`，作为下面 `--data-root`。文件顺序与精度不可改写；目标是官方公布的每日**总成交均价**，不改成收盘价、挂牌均价或成交额除成交量重算价。不补自然日，不删除评价原点。H1/5/20 是后续观测记录步长。

## 4. 只核身份的公开预检

下列命令要求本轮最终 JSON 结果已经导出到 checkout。预检只验证源、计划、公开导出身份与标准文件哈希；不生成市场预测，不求解读出，不计算预测分数。

```bash
/tmp/carbon-research-env/bin/python \
  research/literature_tricks_primary_model_20261019f/source/replay_public_states.py \
  --data-root /tmp/carbon-stage-f-replay-data \
  --out /tmp/carbon-stage-f-public-replay-preflight \
  --preflight-only
```

输出目录必须不存在或为空，并位于 checkout 之外。通过后得到 `PREFLIGHT.json`；它证明当前输入与公开物证身份相符，不证明预测数值已经重放。

## 5. 完整公开数值重放

使用另一个全新输出目录：

```bash
/tmp/carbon-research-env/bin/python \
  research/literature_tricks_primary_model_20261019f/source/replay_public_states.py \
  --data-root /tmp/carbon-stage-f-replay-data \
  --out /tmp/carbon-stage-f-public-replay
```

这个入口执行以下核对：

1. 验证全部 752 份导出 JSON 的字节身份，以及原拟合源、固定计划、同 SHA 数据。
2. 从同一标准数据重新构造原始输入、尺度、切分和目标；EWT 四个子集的输入与目标哈希必须与原预处理清单一致。
3. 从公开数值状态生成全部正式评价预测。EWT 与密集监督还回放已选训练状态的验证损失，并检查记录中的固定网格选择顺序；不重新优化所有网格状态。
4. 对成熟读出独立重算全部五种策略的验证路径与损失，先写本次验证选择，再生成评价路径。它核对全部五个验证损失、所选策略、冻结策略与原始响应预测的一致性。
5. 将重新计算的 MSE、RMSE、MAE、偏差、RW 比率及 Q 与公开记录中的对应指标比较，保留每个面板的差异，不重新挑选更好模型。

数值比较阈值为绝对 `1e-10`、相对 `1e-9`；实际最大差异写入 `REPLAY_AUDIT.json`。本会话已完整重放通过，90 个面板的公开指标最大差异为 **0**；额外与原私有预测数组比较的最大差异为 `1.7763568394002505e-15`，二者不是同一种核对量。源代码、数据、计划和被读取的公开 JSON 在重放完成后再次核验。入口不会伪造旧 NPZ、PT 的文件哈希，也不会复用原运行时间戳。

主要输出为：

- `REPLAY_PROTOCOL_BEFORE_PREDICTION.json`：本次重放身份、数据和依赖。
- `ALL_REPLAY_SUMMARIES.json`：完整 90 个评价面板及其公开指标差异。
- `REPLAY_AUDIT.json`：是否完整通过、13,632 次成熟读出求解计数和验证一致性。
- 每个面板的 `predictions.npz`，以及成熟读出的 `paths.npz`：逐原点数据保留在研究用输出目录，不加入公开树。

`REPLAY_AUDIT.json` 中的 PASS 是程序真实完成后产生的状态，不能用运行前语法检查代替。实际交付后的回放审查记录见 `audits/PUBLIC_STATE_REPLAY_AUDIT.json`（由真实运行结果逐字复制）。

## 6. 原物证运行器与完整重新训练

本轮原始拟合命令如下。它们记录真实研究设置；运行时应使用新的空目录，不能覆盖已完成实验。

```bash
/tmp/carbon-research-env/bin/python \
  research/literature_tricks_primary_model_20261019f/source/run_primary_model_tricks.py \
  --datasets research/literature_tricks_primary_model_20261019f/provenance/RUN_DATASETS.json \
  --plan research/literature_tricks_primary_model_20261019f/audits/EXPERIMENT_PLAN_BEFORE_RUN.json \
  --expected-plan-sha 78893370f6862b1295ba9cfcc5e71fc89776944aeae39346de788a4e09cc8880 \
  --raw-response-run /tmp/carbon-signed-response-runs-20261019e/budget320 \
  --raw-classical-run /tmp/carbon-signed-response-runs-20261019e/core \
  --out /tmp/carbon-primary-model-tricks-runs-20261019f/ewt --workers 2

/tmp/carbon-research-env/bin/python \
  research/literature_tricks_primary_model_20261019f/source/run_dense_path_supervision.py \
  --datasets research/literature_tricks_primary_model_20261019f/provenance/RUN_DATASETS.json \
  --plan research/literature_tricks_primary_model_20261019f/audits/EXPERIMENT_PLAN_BEFORE_RUN.json \
  --expected-plan-sha 78893370f6862b1295ba9cfcc5e71fc89776944aeae39346de788a4e09cc8880 \
  --out /tmp/carbon-primary-model-tricks-runs-20261019f/dense_path --workers 2

/tmp/carbon-research-env/bin/python \
  research/literature_tricks_primary_model_20261019f/source/run_matured_readout.py \
  --budget-out /tmp/carbon-signed-response-runs-20261019e/budget320 \
  --datasets research/literature_tricks_primary_model_20261019f/provenance/RUN_DATASETS.json \
  --plan research/literature_tricks_primary_model_20261019f/audits/EXPERIMENT_PLAN_BEFORE_RUN.json \
  --expected-plan-sha 78893370f6862b1295ba9cfcc5e71fc89776944aeae39346de788a4e09cc8880 \
  --out /tmp/carbon-primary-model-tricks-runs-20261019f/matured_readout
```

这些冻结入口绑定本会话原物证：EWT 复用基线检查旧 NPZ 哈希；成熟读出还检查旧上游协议、完成记录、PT 与 NPZ 哈希。公开完整数值 JSON 可以还原模型，重新序列化或重新训练却不保证得到旧二进制字节。因此上述原物证命令**不等于仅下载公开 checkout 就能完成的全量重新训练入口**。

新环境若需要从头重训 598 次，应保留原科学设置，另外建立带新路径、运行时间和上游绑定的 replication 计划，或由单独适配器组织原训练函数；不能改写旧协议时间或以新文件冒充旧物证哈希。本轮公开入口已经解决相同数据下的全部正式预测数值重放，尚未把全量重新训练也封装成不依赖原物证的命令。

## 7. 完成后导出全部证据

[source/export_results.py](source/export_results.py) 对三个实验块同时检查完成状态、固定计划、完整模型与策略数目、逐模型汇总和顶层汇总一致性。先检查候选：

```bash
/tmp/carbon-research-env/bin/python \
  research/literature_tricks_primary_model_20261019f/source/export_results.py \
  --run-root /tmp/carbon-primary-model-tricks-runs-20261019f \
  --plan research/literature_tricks_primary_model_20261019f/audits/EXPERIMENT_PLAN_BEFORE_RUN.json \
  --expected-plan-sha 78893370f6862b1295ba9cfcc5e71fc89776944aeae39346de788a4e09cc8880 \
  --check-only
```

去掉 `--check-only` 后逐字节复制全部正式 JSON，生成 `provenance/RESULT_EXPORT_RECEIPT.json`。实际已经完成 EWT 557、密集监督 157、成熟读出 38，共 **752 份**；程序逐路径要求完整，不用分数过滤模型或策略。自行训练的数值参数和训练轨迹保留；原始数据、逐原点预测、每期在线头、缓存及二进制状态不复制。已存在且不同的证据会触发拒绝覆盖。

导出完成后，才对公开副本运行第 5 节的数值重放。公开重放审查与独立统计审查承担不同作用：前者证明参数、处理和汇总可以衔接，后者评价效果差异与不确定性。前者通过不会改变两个市场已经暴露的研究身份，也不自动证明条件持续时间机制具有独立增益。

## 8. 本次已经执行的检验

真实公开状态重放始于 2026-10-19 15:02:09 UTC（北京时间 23:02:09），使用适配器 SHA256 `f69dea055d3fc8de227e873c80b3c982ed71a20e5e11f2490f8350e4ceeb4463`。下列命令已经成功执行，旧输出目录中保留了结果，因此再次执行应更换 `--out`。

```bash
PYTHONPATH=/tmp/carbon-forecast-20261019d-deps \
PYTHONDONTWRITEBYTECODE=1 \
/tmp/carbon-research-env/bin/python \
  research/literature_tricks_primary_model_20261019f/source/replay_public_states.py \
  --data-root /tmp/carbon-stage-f-public-state-replay-data-20261019f \
  --out /tmp/carbon-stage-f-public-state-replay-20261019f
```

| 实际核对项 | 结果 |
|---|---:|
| 公开 JSON 完整清单及逐字节一致 | 752 / 752 |
| 重放的正式评价面板 | 90 / 90 |
| 全部公开评价指标最大绝对差 | 0 |
| EWT/密集监督选中状态验证损失最大绝对差 | `2.220446049250313e-16` |
| 在线全部五策略验证损失最大绝对差 | 0 |
| 在线成熟读出求解 | 13,632 |
| 读出正规方程最大绝对残差 | `2.220446049250313e-16` |
| 本次离线重新拟合 / 非线性更新 | 0 / 0 |
| 另行与原私有运行核对的数组 | 482 |
| 原私有数组与重放数组最大绝对差 | `1.7763568394002505e-15` |

公开重放结果在 [audits/PUBLIC_STATE_REPLAY_AUDIT.json](audits/PUBLIC_STATE_REPLAY_AUDIT.json)，SHA256 `6dcd1704b0295188cd73a4465cf34b9b4ce223e7aa405fc20c42148a957f6492`；它是实际运行审查记录的逐字节副本。另见 [audits/PUBLIC_DELIVERY_SCOPE_AND_ARRAY_AUDIT.json](audits/PUBLIC_DELIVERY_SCOPE_AND_ARRAY_AUDIT.json) 与可重复运行的 [audits/audit_public_delivery.py](audits/audit_public_delivery.py)。后一项原物证审查需要保有私有 NPZ；第 5 节面向公开代码使用者的重放入口不需要它们。

原正式运行的 64 个 NPZ、408 个 PT，以及本次重放新生成的 98 个 NPZ 均保留在 checkout 之外。公开结果包含自行训练状态及其缓冲量的 11,152 个数值标量，以及 Ridge 系数、截距和尺度的 21,762 个标量；这些不是独立样本数或模型参数自由度。逐日标准行情的 2,328 条原始行签名在受检公开快照中命中 0。
