# 在新隔离目录复现本轮实验

本入口恢复批准材料和运行代码；它不训练模型、不安装依赖，也不复制本轮已有的
拟合结果。下面的训练命令只应在入口创建的**新隔离目录**中执行。
2020/2021 年仍是已暴露的开发期；复现不会使它们成为独立测试集。

## 1. 固定输入与恢复规则

`prepare_workspace.py` 从最终仓库的
`research/web_pro_execution_20261017/provenance/WEB_READ_INDEX.json` 读取原始索引。
索引本身固定为 SHA-256：

```text
f7f9d0e822c2459529cebbfafe7d5f66719fb1da2fbe62bd95811b9e95a1ef9e
```

它只允许索引中提交 `c78846539e79dc459d3559ae5d0a410ba15fc0d5` 的 21 项材料。
每项按 `bytes` 和 `sha256` 校验，随后写入索引规定的 `runtime_path`。
21 项中包括固定的历史报告、配置与源码证据；这些文件的存在不算本轮重新训练。
唯一市场数据文件是 950 条广州价格，29204 字节，SHA-256：

```text
1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070
```

网络模式逐项请求固定 `raw_url`，不跟随重定向，不递归抓取仓库，不获取原仓库的
完整价格 CSV。响应必须具有匹配的 `Content-Length`；不匹配、缺失或哈希失败均停止。
脚本不更换 URL 或放宽校验。若代理去掉长度头，可使用下述已验证本地来源模式。

本轮运行代码、协议、作者 vendor、必要来源证明和审计器从 `--source-dir` 本地复制，
按 `REPRODUCTION_SOURCES.json` 核验。使用明确清单，任何额外文件都不会自动进入
输出。Stage A 原始冻结代码的哈希还在入口中单独固定；Stage B 源哈希保存在对应
正式运行冻结记录导出的清单中。代码不会从网络另行下载。

输出路径必须不存在，父目录必须已存在，并且位于来源目录之外。即使输出只是一个
空目录，也会拒绝覆盖。所有来源先通过校验，再创建输出；写入途中失败会留下失败
标记，禁止把该目录当作可继续运行的工作区。路径穿越、符号链接和文件/目录碰撞均拒绝。

## 2. 先查看计划，再恢复

将下列路径改为自己的绝对路径。`CARBON_SOURCE_ROOT` 是**最终仓库根目录**，
不是其中的实验子目录；其中必须含本次交付的完整源码清单。

```bash
set -euo pipefail
CARBON_SOURCE_ROOT="/absolute/path/carbon-forecast-model-research"
CARBON_REPRO_ROOT="/absolute/path/new-carbon-reproduction"
python3.12 -B "$CARBON_SOURCE_ROOT/research/web_pro_execution_20261017/prepare_workspace.py" \
  --source-dir "$CARBON_SOURCE_ROOT" \
  --output "$CARBON_REPRO_ROOT" \
  --dry-run
```

`--dry-run` 不发网络请求，也不创建输出目录。网络恢复命令为：

```bash
python3.12 -B "$CARBON_SOURCE_ROOT/research/web_pro_execution_20261017/prepare_workspace.py" \
  --source-dir "$CARBON_SOURCE_ROOT" \
  --output "$CARBON_REPRO_ROOT"
```

若已有之前按索引恢复的批准材料，可全程离线恢复：

```bash
CARBON_APPROVED_ROOT="/absolute/path/previous-approved-runtime"
python3.12 -B "$CARBON_SOURCE_ROOT/research/web_pro_execution_20261017/prepare_workspace.py" \
  --source-dir "$CARBON_SOURCE_ROOT" \
  --approved-local-dir "$CARBON_APPROVED_ROOT" \
  --output "$CARBON_REPRO_ROOT"
```

`--approved-local-dir` 中的文件布局必须是索引的 `runtime_path`。
本地模式先检查文件大小，再打开读取；若其中是旧完整 CSV，会在读取内容前拒绝。
它不会截取旧完整文件的前 950 条来冒充已批准来源。本地模式也可搭配 `--dry-run`，
届时会校验全部 21 项本地文件，但不写输出。

成功时根目录生成 `WORKSPACE_PREPARATION.json`，状态应为 `COMPLETE`，
`approved_files_verified` 应为 21；本地模式网络请求数为 0，网络模式为 21。
新目录同时生成审计器所需的根 `provenance/SOURCE_READ_MANIFEST.json`。
这些恢复路径不会改动最终仓库中的旧历史文件。

## 3. 使用实际记录的运行环境

正式环境为 **Linux x86_64、CPython 3.12.14、NumPy 2.5.4、torch 2.3.1+cpu、
transformers 4.40.2**。`ENVIRONMENT_LOCK.json` 和 `requirements.txt` 锁定了
24 个已安装分发包的版本。核心版本有 Stage A/B 的正式进程记录；传递依赖版本来自
正式运行完成后的安装元数据核对。历史批准材料里的 `requirements.lock.txt`
描述另一套历史环境，不能用于覆盖本轮环境锁。

可以使用已经具备这些版本的解释器。如果依赖放在独立目录，例如执行时采用的
`carbon_deps`，则显式设置该目录：

```bash
CARBON_PYTHON="/absolute/path/python3.12.14"
export PYTHONPATH="/absolute/path/carbon_deps"
```

若要从零准备环境，可以在新目录建立独立 venv，先从 PyTorch 的 CPU 专用索引
安装精确 CPU 构建，再从 PyPI 安装锁定依赖。这里的 `python3.12` 本身应为
3.12.14。依赖安装是单独的环境准备步骤，会访问下列软件包索引；恢复入口自身的
网络请求仍严格限于 21 项批准材料。

```bash
python3.12 -m venv "$CARBON_REPRO_ROOT/.venv"
CARBON_PYTHON="$CARBON_REPRO_ROOT/.venv/bin/python"
unset PYTHONPATH
"$CARBON_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.3.1+cpu"
"$CARBON_PYTHON" -m pip install --index-url https://pypi.org/simple \
  -r "$CARBON_REPRO_ROOT/research/web_pro_execution_20261017/requirements.txt"
```

本轮实际依赖安装已成功，正式模型已使用上文记录的核心版本运行。为保持正式环境
稳定，没有再次创建新 venv 或重装来测试上述文档命令；实际校验的是已有环境中
版本、导入、合成运行以及恢复后的源代码布局。

若已有完整 Linux/CPython 3.12 CPU wheel 目录，可选择下面的离线路径，
无需运行上面的联网安装命令。wheel 目录须含 `torch==2.3.1+cpu` 及全部锁定依赖。

```bash
CARBON_WHEELHOUSE="/absolute/path/verified-wheelhouse"
python3.12 -m venv "$CARBON_REPRO_ROOT/.venv"
CARBON_PYTHON="$CARBON_REPRO_ROOT/.venv/bin/python"
unset PYTHONPATH
"$CARBON_PYTHON" -m pip install --no-index --find-links "$CARBON_WHEELHOUSE" \
  -r "$CARBON_REPRO_ROOT/research/web_pro_execution_20261017/requirements.txt"
```

进入新实验目录后校验版本和实际导入，并检查 NumPy/PyTorch 的 float64 转换。
环境校验不读市场数据、不训练模型，也不会访问模型中心：

```bash
cd "$CARBON_REPRO_ROOT/research/web_pro_execution_20261017"
"$CARBON_PYTHON" -B check_environment.py --output ENVIRONMENT_REPRODUCED.json
```

只有 `passed: true` 才继续。本轮模型均使用 CPU、float64、单数学线程、单工作进程。
版本锁不是 wheel 二进制哈希或操作系统/BLAS 快照；其他机器可能存在细小数值与计时
差异。不要为了匹配某个结果而改种子、选模型或重调参数。原始 Stage A 三次计时
未记录 NumPy，不能声称重现了其未被记录的完整环境；24 次正式拟合记录的版本是明确的。

## 4. Stage A：按顺序执行，每条命令一次

```bash
"$CARBON_PYTHON" -B stage_a.py check
"$CARBON_PYTHON" -B audit/verify_stage_a.py --support-only
"$CARBON_PYTHON" -B stage_a.py timing
"$CARBON_PYTHON" -B stage_a.py run
"$CARBON_PYTHON" -B stage_a.py summarize
"$CARBON_PYTHON" -B audit/verify_stage_a.py
```

`check` 只运行已知斜坡信号、形状/梯度、成熟边界与后缀扰动检查。
`--support-only` 只核对批准数据和评分支持，不进行拟合。
`timing` 是 3 次真实短拟合：NLinear、DLinear、c003_GRU，cut459、seed113，
各最多 3 epoch，不计算开发年度评分。

`run` 是 24 次新的正式拟合：4 个模型 × 两年 cut459/703 × seed17/29/43，
一个进程从写入 `STAGE_A_PRE_RUN.json` 时起使用同一个 600 秒预算。
训练与验证方案保持 `STAGE_A_PROTOCOL.md` 的冻结规定。
`summarize` 在全部拟合完成后一次性生成完整每 seed/每格聚合，随后独立审计器
从批准 CSV 和本机新产生的权重/预测重算结果。审计默认以独占创建方式写入
`audit/AUDIT_STAGE_A.json`，不覆盖已有文件。

## 5. Stage B：Stage A 完整完成后执行

Stage B 作者已确认如下 CLI。代码自身也要求 Stage A 的 24 次正式拟合完整结束，
才允许市场计时与正式运行。

```bash
"$CARBON_PYTHON" -B stage_b.py check
"$CARBON_PYTHON" -B audit/verify_stage_b.py --support-only
"$CARBON_PYTHON" -B stage_b.py timing
"$CARBON_PYTHON" -B stage_b.py run
"$CARBON_PYTHON" -B stage_b.py summarize
"$CARBON_PYTHON" -B audit/verify_stage_b.py
```

三臂是 NLinear_OGD、完整 PROCEED 和独立训练的 PROCEED_no_drift。
`timing` 恰 3 次短拟合；正式阶段恰 18 次新拟合与 4239 次按成熟标签进行的在线更新，
一个进程共用从正式开始计起的 900 秒预算。在线顺序、冻结参数和具体消融含义以
`STAGE_B_PROTOCOL.md` 为准。PROCEED 的本地依赖导入不需要预训练权重；代码启用
Hugging Face 离线标志，缓存路径为本机 `local_only/huggingface_cache`。

本轮评价在完整 H20 落在同年的 451 个起点上进行：2020 年 224 个、2021 年 227 个。
h=1/5/20 都是观测步；先计算六格各自的 MSE/RW MSE，再对六格等权平均。
三 seed 描述初始化波动，不是三个独立市场。Stage A 年度冻结预测与 Stage B
逐步在线更新的机制比较应保持报告中的限定。

## 6. 结果、失败与验证范围

新的 `STAGE_*_PRE_RUN.json`、计时计划、拟合回执、聚合 CSV/JSON 和审计报告会在
新实验目录中产生。逐点预测、训练/验证逐点输出、在线逐点日志和模型权重只写入
`local_only`。入口不复制任何已有 Stage A/B 结果或逐点预测，也不把这些本机文件
放入公共交付包。新目录的 `.gitignore` 排除 `local_only`，但复现者发布时仍应按
明确文件清单处理，而不是上传整个工作目录。

已有计时计划或 `PRE_RUN` 会触发原程序的禁止重启检查；不要删除标记以重置预算。
异常或超时保留已完成回执与失败记录，不能从部分结果挑选赢家。程序不自动重试，
也不提供改变超参数的复现参数。一次新的复现属于单独记录的运行，不改写本轮原始记录。

入口自身的验证脚本为：

```bash
"$CARBON_PYTHON" -B verify_prepare_workspace.py \
  --source-dir "$CARBON_SOURCE_ROOT" \
  --approved-local-dir "$CARBON_APPROVED_ROOT" \
  --output /absolute/path/new-prepare-validation.json
```

它在临时新目录验证本地 21 项恢复、路径/哈希/尺寸拒绝、不覆盖、严格 URL/重定向
处理，并运行环境、两阶段合成检查与支持核对。它**不运行市场 timing/run**。
本次实际记录见公共 `REPRODUCTION_VALIDATION.json`；网络下载路径只做受控响应测试，
未将其描述为重新下载 21 项或第三方完整训练复现。完整优化器轨迹的重新运行需要执行
上面的正式命令，独立审计通过也不等于论文原创性得到证明。
