# 私有证据包复现说明

## 1. 可以复现什么

本入口使用包内现成的 prepared 输入、正式预测、验证预测、模型参数和原运行收据。默认无需重新下载原始 EEX 工作簿，也无需重新训练，即可核验文件身份、复算正式价格指标及回放保存的模型。可选训练入口会在全新目录复制冻结研究结构，完整执行指定版本的经典和神经训练，再评分、回放。

三个版本均以 **2021、2022、2023 年作为开发评价年份**，预测期为未来第 1、5、20 个已计划拍卖事件。这些年份已经反复用于研究开发，不能称为从未查看的最终测试集。2024/2025 确认数据未进入本 prepared 包，**确认分区没有评分**；本入口不会去读取确认文件。

| 版本名 | 包内相对目录 | 神经拟合次数 | 经典学习拟合次数 | 正式预测 NPZ | 独立保存的验证预测 NPZ |
|---|---|---:|---:|---:|---:|
| `original` | 研究根目录 | 120 | 27 | 84 | 48 |
| `corrected` | `calendar_correction_20261018/rerun/` | 120 | 27 | 84 | 48 |
| `fdmc` | `scale_separated_revision/` | 105 | 27 | 75 | 42 |
| 合计 | 三个真实运行 | **345** | **81** | **243** | **138** |

345 次神经拟合包含 138 次候选调参与 207 次最终 seed 重拟合。每版有 12 个经典 model×year 任务，其中 Ridge 的 18 次与 HistGB 的 9 次拟合合计 27；RW 和 LastBidMedian 是解析规则，不计为学习拟合。243 份正式 NPZ 包含 207 份神经预测和 36 份经典预测。经典候选验证预测也保存在各自正式 NPZ 内，不能再次算成 138 份独立神经验证文件。

三次神经 seed 构成一个平均点预测和等权混合分布，不构成三个独立市场样本。每模型每版共有 1,933 个已发预测记录，其中 1,927 个具有可评分价格；它们包含重叠预测期。

## 2. 四个入口文件

| 文件 | 用途 |
|---|---|
| `verify_evidence_bundle.py` | 仅用 Python 标准库检查私有正向清单、所有文件 SHA/大小、三份冻结清单、345 个 checkpoint 与 receipt 的 SHA 链、真实任务计数。不会反序列化模型。 |
| `reproduce.py` | 在新目录复制必要研究结构，执行校验、指标复算、回放，或明确选择的训练。 |
| `requirements-pinned.txt` | 本次实际 CPU 环境的固定 Python 包版本；不会安装或读取数据。 |
| `PRIVATE_EVIDENCE_INDEX.json` | 完整私有 prepared 证据包的逐文件相对路径、字节数、SHA、用途及总计数。由上述校验器生成，包交付时提供。 |

索引采用正向清单，**不会递归打包整个工作区**。索引与校验报告是外层元数据，不在索引内部递归对自身求哈希。私有包应额外携带索引文件和交付校验报告。

## 3. 环境

实际运行环境为 Python **3.12.14**、Linux x86_64、CPU PyTorch **2.5.1+cpu**、NumPy **2.3.5**、pandas **2.2.3**、SciPy **1.17.0**、scikit-learn **1.8.0**。完整固定版本见 requirements；原冻结清单也保留关键版本信息。训练与回放使用 2 个 CPU 计算线程，神经 interop 为 1；launcher 设置相应线程环境变量。

在私有研究根目录中，例如执行：

```bash
python3.12 -m venv ../carbon_repro_env
../carbon_repro_env/bin/python -m pip install -r reproduction/requirements-pinned.txt
```

requirements 中的 CPU PyTorch 额外索引只用于安装依赖，复现程序本身不联网。固定 Python 包版本不等于固定操作系统、BLAS 二进制和 CPU 指令实现；跨平台重训可能有细小数值差异。历史文件 SHA 用于确认原证据身份，不能要求新训练产生相同的序列化时间或文件 SHA。若严格回放在另一环境失败，应保留失败记录并恢复同等环境，不改变冻结模型或放宽门槛来取得通过。

## 4. 最轻量：不训练、不评分的完整证据核验

解包后保持三个研究版本的相对层级，不需要保留原工作区的绝对根路径。先在研究根目录执行：

```bash
../carbon_repro_env/bin/python reproduction/verify_evidence_bundle.py \
  --study-root . \
  --index reproduction/PRIVATE_EVIDENCE_INDEX.json \
  --report ../carbon_private_hash_check_01.json
```

输出报告路径必须尚不存在。校验器检查全部正向清单文件的字节数和 SHA，并重新核查正式/验证文件计数、冻结关系和 checkpoint/receipt 对应关系。此检查只读取证据，不执行 pickle/joblib/PT 模型，不计算模型性能。

维护者需要从完整私有目录重新生成清单时，可选择一个新输出文件名：

```bash
../carbon_repro_env/bin/python reproduction/verify_evidence_bundle.py \
  --study-root . \
  --write-index ../private_evidence_index_new.json \
  --report ../private_evidence_index_new_check.json
```

已有发布包应优先验证随包提供的原索引，不能通过重新生成索引来掩盖证据变更。

## 5. 不训练：复算正式指标与保存模型回放

先查看计划：

```bash
../carbon_repro_env/bin/python reproduction/reproduce.py \
  --study-root . --action audit --studies all \
  --output ../carbon_audit_01 --plan-only
```

移除 `--plan-only` 执行同一计划：

```bash
../carbon_repro_env/bin/python reproduction/reproduce.py \
  --study-root . --action audit --studies all \
  --output ../carbon_audit_01
```

`--output` 必须是**证据树以外、尚不存在的目录**。launcher 先核验源索引，再复制文件到 `study_snapshot/`；所有指标、审核及日志只写到这个新目录。它不会通过软链接或硬链接共享可写的原证据，也不会修改任何冻结脚本来改变路径。

动作可以分别选择：

- `verify`：仅完成私有索引核验及元数据报告，不创建运行 clone。
- `score`：三个版本的独立正式评分；可用 `--studies original corrected` 等缩小版本范围。
- `replay`：各所选版本的经典模型回放；所选版本包含 fdmc 时，再回放 **63 个 v2 正式神经 checkpoint**。
- `audit`：`score` 与 `replay` 的完整组合。

正式评分使用现有独立评分器，覆盖原价 MSE/MAE、proper CRPS、预测区间、三 seed 混合、2,000 次固定种子的按年分层配对区块重抽样及冻结门槛。均值、概率分布和 RW 分母按原定义处理；不会把验证预测流混入正式评价。完成后会把主要汇总指标和门槛决定与源包原正式结果比较。

回放范围必须区分：三版经典模型均检查保存 pipeline、训练内 scaler、候选验证损失、残差原子和直接原价评分；**神经 forward checkpoint 回放入口仅涵盖已有并通过的 v2 63 个最终模型**。original/corrected 的神经 checkpoint 全部纳入哈希与运行收据校验，并参与正式预测评分，但本交付没有新增其 144 个最终模型的 forward 回放声称。点神经基线按原审核范围核均值，其残差分布的正式 CRPS 由独立评分器复算。

输出组织如下：

```text
carbon_audit_01/
  PLAN.json
  SOURCE_PRIVATE_EVIDENCE_INDEX.json
  SOURCE_VERIFICATION.json
  FORMAL_SCORE_REPRODUCTION_COMPARISON.json
  REPRODUCTION_RECEIPT.json
  logs/
  study_snapshot/
```

`study_snapshot/` 中保持研究原层级。各版本的新评分位于 `results/independent_metrics/`，经典回放位于 `results/classical/CLASSICAL_PREDICTION_AUDIT.json`，v2 神经回放位于 `scale_separated_revision/results/NEURAL_REPLAY_AUDIT.json`。失败时保留日志和 `REPRODUCTION_FAILURE.json`，不自动改变配置或补跑。

## 6. 可选：在新 clone 中完整重跑训练

例如仅复跑 fdmc：

```bash
../carbon_repro_env/bin/python reproduction/reproduce.py \
  --study-root . --action train --studies fdmc \
  --output ../carbon_fdmc_training_01 --plan-only
```

确认计划中的版本与目录正确后，移除 `--plan-only` 即执行。三个版本一起重跑使用 `--studies all`，将产生新的 **345 次神经拟合及 81 次经典学习拟合**，并随后评分和按上述范围回放。

训练 clone 只携带现有冻结 prepared 输入、核心源文件和必要来源审核，不复制原模型输出到训练目的目录。经典审核包装脚本在经典训练完成后才放入 `local_only/classical/`，避免触发原 runner 对非空结果目录的保护。所有原有参数、候选、划分、seed、训练 epoch 选择及失败条件保持原协议。

训练使用原冻结协议 ID 标识所复现的协议；新的 `REPRODUCTION_RECEIPT.json` 明确记录本次时间及“这是复现输出，不是历史原运行记录”。它不构成新预注册，也不改变看到原 v1 结果后进行数据修正、再进行一次模型修订的研究时间顺序。新训练不会以复现成绩决定是否追加候选或打开确认集。

## 7. 路径可移植性与保留的限制

launcher 解决的是**prepared 证据的定位和安全新输出**：可以用 `--study-root` 指向任意解包位置，用 `--python` 指定已安装固定依赖的 Python，并完整保留原目录相对关系。v2 模型对上一层 `clearing_models.py` 的导入、跨版本冻结引用和各审核脚本根据 `__file__` 定位的行为因此保持有效。

现有独立评分器虽然有 `--study-dir`，但把输出固定为该目录中的 `results/independent_metrics/`；部分审核脚本没有 CLI 输出参数，直接根据自身位置写结果。请通过新 launcher 调用，不要直接在原证据树运行这些脚本。

原始工作簿来源字段仍包含原工作区的绝对 `source_path`，这是冻结证据的一部分。默认评分、回放和训练仅读取 prepared 数据，不沿这些字符串打开原件，所以不需要伪造旧目录或改 CSV。**从官方工作簿重新解析到 prepared 输入的端到端再生成不由默认入口完成**：原 parser 的 source-root 默认位置、修正 parser 对源 SHA/原路径的验证以及原始来源可取得性仍是外部依赖。原件不随默认私有 prepared 包收录；如另行取得，应使用新的非冻结迁移包装流程，并区分数据值复现与因来源路径变化而不同的 CSV 字节哈希。不要修改被冻结的路径字符串来冒充原证据。

v2 的 `prepare_factorized_examples.py` 支持 `--source-study`，但默认训练入口仍复用已冻结 prepared 文件，不重新生成并覆盖它们。既有来源、因果性和算子检查随包保留；本轮交付只验证 launcher 的帮助、plan-only 及真实证据哈希/计数，没有为测试 launcher 再跑一遍训练。

## 8. 私有包与公开 Git 的边界

建议私有完整包包含索引 `files` 列出的全部文件，并额外包含 `PRIVATE_EVIDENCE_INDEX.json` 与交付校验报告。该清单包含三版 prepared 数据、243 份正式预测、138 份验证预测、345 份神经 PT、36 份经典 joblib、训练行身份、调参/拟合收据、冻结源码、正式汇总及复现工具。

**公开 Git 不应默认接收这一私有清单。** prepared 价格输入、逐起点预测、权重/checkpoint 和训练行明细属于私有证据交付；论文全文、原始 EEX 工作簿及外部档案也不在默认公开或本 prepared 包清单内。公开交付仅采用另行审核的代码、协议、来源元信息和聚合结果清单。索引中的哈希是完整性说明，不是对外再分发许可。
