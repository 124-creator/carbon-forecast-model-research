# 固定未来预测几何实验：完整复现说明

研究目录：`research/forecast_geometry_filter_20261019c/`。

本轮已经完整执行并独立复核，原协议决策为 **`STOP_FIXED_GEOMETRY_CANDIDATE`**。这里提供两种用途：核查已经生成的四个合成 NPZ，或从固定种子完整重跑原协议。两者都不是追加模型搜索，也不打开碳市场或 2024 年以后的确认数据。

## 1. 环境与获取源码

正式 `results/synthetic/START.json` 记录的运行环境为 **CPython 3.12.14、NumPy 2.3.5、SciPy 1.17.0**。80 位高精度诊断使用 **mpmath 1.3.0**，结果图使用 **Matplotlib 3.10.8**；固定依赖见 `requirements.txt`。没有 PyTorch 或其他神经网络训练依赖。

以下命令使用 Linux／POSIX shell。本次核实仓库可公开克隆。请选择尚不存在的克隆目录：

```bash
git clone --branch research/forecast-geometry-filter-20261019c --single-branch https://github.com/124-creator/carbon-forecast-model-research.git carbon-research-geometry
cd carbon-research-geometry
git rev-parse HEAD
export CARBON_GEOM_STUDY="$PWD/research/forecast_geometry_filter_20261019c"
export CARBON_GEOM_GIT_STUDY="$CARBON_GEOM_STUDY"
export CARBON_GEOM_REPRO_WORK="$(mktemp -d /tmp/carbon-geometry-repro.XXXXXX)"
python3.12 -m venv "$CARBON_GEOM_REPRO_WORK/venv"
. "$CARBON_GEOM_REPRO_WORK/venv/bin/activate"
python -m pip install -r "$CARBON_GEOM_STUDY/requirements.txt"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
python - <<'PY'
import sys
import numpy, scipy, mpmath, matplotlib
assert sys.version_info[:3] == (3, 12, 14), sys.version
expected = {'numpy': '2.3.5', 'scipy': '1.17.0',
            'mpmath': '1.3.0', 'matplotlib': '3.10.8'}
actual = {'numpy': numpy.__version__, 'scipy': scipy.__version__,
          'mpmath': mpmath.__version__, 'matplotlib': matplotlib.__version__}
assert actual == expected, actual
print(sys.version)
print(actual)
PY
```

`python3.12` 必须实际对应补丁版本 3.12.14；`pip` 不能安装或切换 Python 解释器。三项线程环境变量应在启动 Python 之前设置。

正式原始执行时的本地 Git HEAD 是 `51ce5582c3b747f4829432c1fbb8f5e114969318`。GitHub 交付的提交包含后续报告和审计，未必与此本地提交相同；不要假定该本地 SHA 一定可以从远端检出。应记录实际检出的 GitHub HEAD，并核对下列冻结源文件哈希。

## 2. 先核对冻结输入

| 文件 | 正式修复后 SHA-256 |
|---|---|
| `source/geometry_filter.py` | `a6602b273f10fcba6f1862153bd93453d6b7212a4a10d36611d6aa5a44caad66` |
| `source/run_synthetic.py` | `4ac6a99a415ffa50ddec266faee42c5037aedd8f8cfb3d1b988ea5872bf84b8f` |
| `provenance/SYNTHETIC_PROTOCOL.json` | `b5c6b2ac088d7c82ee765ecc8064d3322e07512a798c20fc26743b0cbea85676` |

完整正式输入及预运行审计绑定在 `provenance/SOURCE_FREEZE.json`。下面同时检查其全部文件以及前缀审计后来封存的依赖：

```bash
python - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['CARBON_GEOM_STUDY'])
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
freeze = json.loads((root/'provenance/SOURCE_FREEZE.json').read_text())
for relative, expected in freeze['sha256'].items():
    assert digest(root/relative) == expected, relative
for relative in freeze['required_prefit_audits']:
    audit = json.loads((root/relative).read_text())
    assert audit['status'] == 'PASS', relative
    for source, expected in audit['source_hashes'].items():
        assert freeze['sha256'][source] == expected, (relative, source)
prefix = json.loads((root/'audits/PREFIX_REPLAY_DESIGN_FREEZE.json').read_text())
for relative, expected in prefix['source_hashes'].items():
    assert digest(root/relative) == expected, relative
print('PASS: frozen source and audit bindings')
PY
```

不要先重跑会将回执写回固定路径的旧 core、runner、recovery 或 solver-diagnostic 脚本，再期望原冻结继续有效。它们生成的新时间戳会改变审计 JSON 的哈希。原稿、原回执、`attempts/01_kl_dual_stagnation/`、`ORIGINAL_SOURCE_FREEZE.json` 及正式结果都应保留。下面两个事后回放工具有独立的 `--out` 参数，可以直接使用新输出位置。

另有一处已明确记录的冻结文本勘误：`methods/GEOMETRY_DERIVATION_ZH.md` 第 4 节将两个中间状态的“当前方差同为 1”写成归一化前值；协议和实际代码一直使用归一化后的 **0.64**，完整方差向量为 `[0.16, 0.64, 0.64, 2.56]`。见新增 `methods/FROZEN_TEXT_ERRATA_ZH.md`；它说明历史文字差异，不修改冻结公式、源码或结果，也不要求重跑实验。

## 3. 核查现成的纯合成轨迹包

本轮轨迹包包含这四个 NPZ 对应的**合成观察、隐藏状态、污染标记、验证分数、选择结果、过滤权重和测试评分**。它不包含旧轮次市场原始数据、神经模型权重或逐点真实市场预测，也不代表此前那些市场档案的保存问题已经处理。

正式包名为 **`carbon_geometry_synthetic_run_20261019.zip`**。内部布局是 `study/` 存放完整本研究目录（不含 `.git`），`traces/` 存放四个 NPZ，顶层另有 `ARCHIVE_README_ZH.md` 和 `ARCHIVE_MANIFEST.json`。下面将下载包解压到一个新目录，使用包内研究快照和公共结果完成审计：

```bash
export CARBON_GEOM_ARCHIVE='/absolute/path/carbon_geometry_synthetic_run_20261019.zip'
export CARBON_GEOM_BUNDLE_DIR="$(mktemp -d /tmp/carbon-geometry-bundle.XXXXXX)"
sha256sum "$CARBON_GEOM_ARCHIVE"
python -m zipfile -e "$CARBON_GEOM_ARCHIVE" "$CARBON_GEOM_BUNDLE_DIR"
export CARBON_GEOM_STUDY="$CARBON_GEOM_BUNDLE_DIR/study"
export CARBON_GEOM_TRACE_DIR="$CARBON_GEOM_BUNDLE_DIR/traces"
export CARBON_GEOM_PUBLIC="$CARBON_GEOM_STUDY/results/synthetic"
export CARBON_GEOM_AUDIT_OUT="$(mktemp -d /tmp/carbon-geometry-audit.XXXXXX)"
```

将 `/absolute/path/...` 替换成实际下载包路径，并将 `sha256sum` 输出与 GitHub 交付清单公布的包级 SHA-256 对照；包内 `ARCHIVE_MANIFEST.json` 记录成员清单。**只核查现成档案时可直接从解压后的 `study/` 执行，无需 Git 仓库。** 此时可省略第 1 节克隆和 `git rev-parse`，将上述解压路径设为 `CARBON_GEOM_STUDY` 后照第 1 节安装依赖和设置线程，再执行第 2 节的冻结哈希检查。下表及 `COMPLETE.json` 给出精确的逐 NPZ 校验值：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `clean_correct.npz` | 19,003,745 | `db96beebd483c702a9d100a7dc46277df5515be58b2a7454a856f2d8bab2ffb7` |
| `rare_large_shocks.npz` | 19,165,865 | `901aa7e79b5304878857725994a51cef7ab59ca50eb5e67fa5f94c19c6bae54b` |
| `frequent_moderate_shocks.npz` | 19,129,526 | `68fdec7a8e17fdd92b603fb0118637724362de0cfd15e9389123d1a819da9476` |
| `wrong_transition_geometry.npz` | 17,016,984 | `7f9a3459b911b315ef75504cbc352e032dc76a476db2e3adab7ae4b531211e11` |

SHA 校验命令同时核对公共汇总、四轨迹、各场景回执和预测前选择文件：

```bash
python - <<'PY'
import hashlib, json, os
from pathlib import Path
public = Path(os.environ['CARBON_GEOM_PUBLIC'])
traces = Path(os.environ['CARBON_GEOM_TRACE_DIR'])
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
complete = json.loads((public/'COMPLETE.json').read_text())
assert complete['complete'] is True
assert complete['source_hashes_unchanged'] is True
assert digest(public/'SUMMARY.json') == complete['summary_sha256']
for row in complete['trace_files']:
    path = traces/row['trace_file']
    assert path.stat().st_size == row['trace_bytes'], path
    assert digest(path) == row['trace_sha256'], path
    scenario = path.stem
    receipt = json.loads((public/scenario/'EXECUTION_RECEIPT.json').read_text())
    assert receipt['trace_sha256'] == row['trace_sha256'], scenario
    assert digest(public/scenario/'SELECTION_BEFORE_TEST.json') == receipt['selection_file_sha256'], scenario
print('PASS: complete results, four synthetic traces, selections and receipts')
PY
```

执行两类核查，**每次指定新回执路径**：

```bash
python "$CARBON_GEOM_STUDY/audits/independent_trace_replay.py" \
  --traces "$CARBON_GEOM_TRACE_DIR" \
  --public "$CARBON_GEOM_PUBLIC" \
  --out "$CARBON_GEOM_AUDIT_OUT/INDEPENDENT_TRACE_REPLAY.json"

python "$CARBON_GEOM_STUDY/audits/independent_prefix_replay.py" \
  --traces "$CARBON_GEOM_TRACE_DIR" \
  --public "$CARBON_GEOM_PUBLIC" \
  --out "$CARBON_GEOM_AUDIT_OUT/INDEPENDENT_PREFIX_REPLAY.json"
```

**不要向前缀脚本传 `--seal`。** 仓库已经携带 `PREFIX_REPLAY_DESIGN_FREEZE.json`；`--seal` 只供当时记录审计设计，已有记录时应拒绝覆盖。日常复现使用现有设计冻结，只把新核查回执写到新目录。两脚本需要 `COMPLETE.json` 和全部场景的公共 CSV／JSON，单独四个 NPZ 不足以核对全部选择和汇总。

独立性边界也应保留：trace 回放不导入 production core／runner，独立检查 Fourier 分布、原目标 KKT、oracle、评分、汇总及 gate；prefix 回放复用冻结 production 更新和一条原评分算式来核对严格平局，同时另用 Fourier／`erf` 独立评分。后者不是另一种优化器的独立求解。

## 4. 从固定种子重新执行全部合成实验

这一步不需要现成四 NPZ，也不读取任何市场数据，但需要完整冻结源码、审计回执及一个**有有效 HEAD 的 Git 工作树**。`run_synthetic.py` 通过 `git rev-parse HEAD` 将实际代码提交记入 `START.json`；只有源码 ZIP、没有 `.git` 的目录不能直接完成此启动记录。

选择一个新的父目录，并让 `--out` 的目标子目录在启动时不存在：

```bash
export CARBON_GEOM_STUDY="$CARBON_GEOM_GIT_STUDY"
git -C "$CARBON_GEOM_STUDY" rev-parse HEAD
export CARBON_GEOM_NEW_RUN="$(mktemp -d /tmp/carbon-geometry-run.XXXXXX)"
python "$CARBON_GEOM_STUDY/source/run_synthetic.py" \
  --out "$CARBON_GEOM_NEW_RUN/traces" \
  --public "$CARBON_GEOM_NEW_RUN/public"
```

这里的 `CARBON_GEOM_GIT_STUDY` 由第 1 节克隆后设置。若此前仅使用合成包而未克隆，请先完成第 1 节；不要把完整 runner 指向解压包中没有 `.git` 的 `study/`。

`--out` 采用 `mkdir(exist_ok=False)`，因此不能预先创建 `traces` 子目录。`--public` 必须使用新目录：脚本发现已有 `START.json` 会拒绝启动。不要删除正式研究目录的 START 来继续运行，也不要省略 `--public` 而写进已完成的 `results/synthetic/`。本脚本没有从部分运行状态无痕续跑的接口；失败应保留失败输出并使用另一个新运行记录。

运行中将依次保存各场景选择和回执，全部完成后才写 `SUMMARY.json` 与 `COMPLETE.json`。完成后用上节相同审计命令，改为：

```bash
export CARBON_GEOM_TRACE_DIR="$CARBON_GEOM_NEW_RUN/traces"
export CARBON_GEOM_PUBLIC="$CARBON_GEOM_NEW_RUN/public"
export CARBON_GEOM_AUDIT_OUT="$(mktemp -d /tmp/carbon-geometry-new-run-audit.XXXXXX)"

python "$CARBON_GEOM_STUDY/audits/independent_trace_replay.py" \
  --traces "$CARBON_GEOM_TRACE_DIR" --public "$CARBON_GEOM_PUBLIC" \
  --out "$CARBON_GEOM_AUDIT_OUT/INDEPENDENT_TRACE_REPLAY.json"

python "$CARBON_GEOM_STUDY/audits/independent_prefix_replay.py" \
  --traces "$CARBON_GEOM_TRACE_DIR" --public "$CARBON_GEOM_PUBLIC" \
  --out "$CARBON_GEOM_AUDIT_OUT/INDEPENDENT_PREFIX_REPLAY.json"
```

新运行的时间戳、耗时、实际 Git HEAD 及回执哈希会变化，不应与原回执文件做整份字节相等比较；新运行应由自己的 COMPLETE／回执和同一数值审计建立一致性。固定版本及单线程有助于复现，但并不保证不同硬件、操作系统或 BLAS 构建具有逐位相同算术。尤其 prefix 对原算式分数和先验使用零容差；如跨平台失败，应保留差异并定位环境，不能悄悄放宽原审计阈值或改写历史结果。

## 5. 预期规模与本轮结论

正式规模为 4 场景 × 20 种子 × 10 家族 × 7 参数，即 **5,600 条验证配置流**；选定家族流为 800 条，加入 nominal Bayes 的选定递推流为 880 条。80 个场景—种子序列共 136,240 个合成观察，但**独立随机种子只有 20**，因为同种子场景共用基础随机数。主区间按 20 个种子成对重采样；验证网格、诊断和审计回放不能相加包装成更多独立实验。

本轮新增市场模型、神经训练以及发射／转移参数拟合均为 **0**。方法研究的是已知名义状态模型下的过滤更新规则。

原协议 STOP 已由完整结果及独立复算确认。除相对预选强对照没有显示实际平均改善外，还应保留资源门槛的设计缺陷：在本轮固定测试轨迹、相同历史信息和相同预选对照上，真实 oracle 的期望 CRPS 改善上界约 **0.559361%**，小于原先要求的 **1%**。这是从已保存结果计算的事后可改善空间诊断，不是未来所有问题的普遍上界，也不允许事后降低门槛重判本轮。

后续不得通过改变此候选的持续性、状态数、污染强度、损失权重、参数网格、种子或通过门槛来寻求有利结果；按本协议不继续市场拟合、不消费 2024+ 确认数据。新方向必须另有新科学假说与预先检查过可达性的资格方案，不能将本轮局部几何效应直接写成有效的新碳价模型。

## 6. 结果和历史入口

- `results/synthetic/SUMMARY.json`、`SELECTED_RESULTS.csv`、`TRIAL_VALIDATION.csv`：完整正式比较、选择和门槛。
- `provenance/SYNTHETIC_PROTOCOL.json`：固定 v1.1 科学协议。
- `provenance/SOURCE_FREEZE.json`：数值修复后的正式输入冻结。
- `attempts/01_kl_dual_stagnation/`：旧 KL 容差失败及完整原始冻结。
- `audits/RECOVERY_PIPELINE_AUDIT_ZH.md`、`SOLVER_REPAIR_REPORT_ZH.md`：修复、原目标一致性及恢复执行资格。
- `audits/INDEPENDENT_TRACE_REPLAY.json`、`INDEPENDENT_PREFIX_REPLAY_ZH.md`：测试和完整验证前缀复核。
- `audits/INDEPENDENT_RESULT_INTERPRETATION_ZH.md`：主要失败、局部效应、oracle 剩余空间和解释边界。
- `methods/FROZEN_TEXT_ERRATA_ZH.md`：冻结推导中归一化前后方差措辞的独立勘误，不改历史输入。
- `HANDOFF_CONTINUATION_ZH.md`：供后续会话使用的明确衔接点。
