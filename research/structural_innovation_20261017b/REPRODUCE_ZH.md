# SMP 冻结开发实验的可移植复现

本说明对应研究目录 `research/structural_innovation_20261017b`。复现入口是 [source/prepare_reproduction.py](source/prepare_reproduction.py)，它验证本地开发数据的固定字节身份，创建一个新的工作目录，并打印实验及独立审核命令。准备入口本身不训练、不运行评分、不请求网络。

本次交付保留 2026-10-17 14:03:49 UTC 冻结的 [provenance/PROTOCOL.json](provenance/PROTOCOL.json) 原始字节：

```text
SHA256 b10409de072efaba5fce1d570c15e6e0b4128f5234f037c25afeecfd90efc503
```

这是在同一开发数据、同一模型定义及同一任务清单下进行的**新运行**。保留历史协议并不使新运行成为原始运行，也不构成外部预注册；原运行的数值结果、审核结论或确认资格均不能直接继承。D1 使用 2022 年，条件式 D2 使用 2023 年；这两个年份及五份开发文件此前均已暴露，不能称为未见测试集。

## 1. 需要的源码与环境

从本次发布中取得整个 `structural_innovation_20261017b` 目录，以下用 `SMP_SOURCE` 表示其绝对路径。旧父提交只用于定位既有数据和 EEX 数据复原包，不包含本轮新写的 SMP 源码。

准备入口仅依赖 Python 标准库。运行训练和选择需要 NumPy 与 PyTorch；独立结果审核仅依赖 NumPy 与标准库。原开发运行使用 Python 3.12、NumPy 2.5.4、PyTorch 2.3.1+cpu。冻结协议规定 CPU、float64、两个进程且每进程一个计算线程。跨 Python、BLAS 或 PyTorch 版本不承诺逐位相同；重放应保留新运行实际记录的环境、时间、源文件身份和结果。

先执行只读源码检查：

```bash
python -B SMP_SOURCE/source/prepare_reproduction.py check-sources
```

也可以从其他位置运行该入口，并显式传入 `--source-root SMP_SOURCE`。该命令只读取协议、源码和许可证；不会读取数据映射、CSV、模型权重或确认文件。

入口检查并复制的内容如下：

| 内容 | 处理方式 |
| --- | --- |
| 原协议声明的 12 个训练源文件 | 逐个核对原 `source_sha256`，保持原相对路径及原字节 |
| `provenance/PROTOCOL.json` | 对照入口内固定 SHA256，按原字节复制；历史绝对数据路径不改写 |
| `source/vendor/LICENSE` | 作为许可证辅助文件单独固定 SHA256；保留 PatchTST 作者 Apache-2.0 许可证 |
| `source/independent_results.py` | 作为正式训练冻结后交付的审核辅助文件单独固定 SHA256 |
| 新的数据映射与准备收据 | 由入口生成，记录新路径、身份和命令 |
| 原结果、逐起点预测、权重、旧审核结果及旧运行标记 | 不复制 |

许可证 SHA256 为 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`；独立审核脚本 SHA256 为 `669a7dc6fb74d5724ce194d6424a76758b9aa4fb84f0719fedd99bdf2311fb52`。这两个辅助文件不被追加入原 12 源训练清单，也不回写原协议。原协议中的历史计时、预审及设计文件哈希仍然是历史引用；新工作目录不会复制那些历史结果，训练器不需要把它们当成本次运行产物。

本方 PatchTST 对照使用未修改的作者核心及统一任务适配，包括多输入到单输出的监督读出和 `RevIN=False`。重放本研究不等于复现作者论文原始 benchmark。SMP 的 Markov 奖励递推及基础组件也不因可运行而获得原创性证明。

## 2. 五份开发文件的精确身份

数据范围固定为日期严格早于 `2024-01-01`。文件应保持原始 CSV 字节，包括换行、列顺序和数值文本；入口拒绝哈希不符的文件，不做清洗、重导出、自动换源或放宽校验。

| data-map 键 | 本地文件名 | 固定 SHA256 |
| --- | --- | --- |
| `EEX_EUA_T3PA` | `EEX_EUA_T3PA_daily_development.csv` | `5a0c7580edebc2a7a5e66d7487568f6d99a8e08cca0785676c2aaeccad7db6bb` |
| `GUANGZHOU` | `GUANGZHOU_development.csv` | `8d78a4f13c5a799519546fd3bfa0e0ac5eaa52b0441a983a95ef06f5e5e34051` |
| `HUBEI` | `HUBEI_development.csv` | `04655d59c1b585f6c35c02a43469a965fe3589eb613d7d8d8e46996751b7d797` |
| `SHANGHAI` | `SHANGHAI_D4_development.csv` | `425424353bb3ac09e3d96b291680b4179e86f180da626a5d19e5c1e58ed960a1` |
| `EU_REPORTED` | `EU_REPORTED_development.csv` | `ca7c192380897076ef57516a11ea2ccefda6ba92ab9823d9aa8f2bb281cefedc` |

四份已公开开发 CSV 可从仓库 `124-creator/carbon-forecast-model-research` 的固定父提交 `b953fb52c9e2b1f02fdf4f4c7d9eb9b574455aeb` 取得：

- [广州开发 CSV](https://github.com/124-creator/carbon-forecast-model-research/blob/b953fb52c9e2b1f02fdf4f4c7d9eb9b574455aeb/research/cross_market_innovation_20261017/data/GUANGZHOU_development.csv)
- [湖北开发 CSV](https://github.com/124-creator/carbon-forecast-model-research/blob/b953fb52c9e2b1f02fdf4f4c7d9eb9b574455aeb/research/cross_market_innovation_20261017/data/HUBEI_development.csv)
- [上海开发 CSV](https://github.com/124-creator/carbon-forecast-model-research/blob/b953fb52c9e2b1f02fdf4f4c7d9eb9b574455aeb/research/cross_market_innovation_20261017/data/SHANGHAI_D4_development.csv)
- [既有 EU 报价开发 CSV](https://github.com/124-creator/carbon-forecast-model-research/blob/b953fb52c9e2b1f02fdf4f4c7d9eb9b574455aeb/research/cross_market_innovation_20261017/data/EU_REPORTED_development.csv)

应下载文件原始内容，不要把 GitHub HTML 页面保存成 CSV。EEX 与 `EU_REPORTED` 都涉及 EU ETS，不能将它们当作两个统计独立市场。原始来源及使用条件继续以主报告和数据归属说明为准。

### EEX：优先直接使用已复原的开发 CSV

若本地已有上表固定 SHA256 的 EEX 开发文件，直接将它写入 data-map 即可。SMP 准备入口只打开显式映射的这份开发文件，不寻找其相邻目录，也不枚举确认文件。

若需要从官方历史归档重新复原，请取得父提交中完整的 [data_reproduction 包](https://github.com/124-creator/carbon-forecast-model-research/tree/b953fb52c9e2b1f02fdf4f4c7d9eb9b574455aeb/research/event_feedback_innovation_20261017/data_reproduction)，不能只复制一个脚本。该包包含 `EEX_REPRODUCTION_SOURCES.json` 和 13 个固定运行文件。下面的 CLI 已通过读取实际源码核对，`EEX_REPRO_SOURCE` 表示该包的绝对路径。

```bash
python -B EEX_REPRO_SOURCE/reproduce_eex_data.py check
python -B EEX_REPRO_SOURCE/reproduce_eex_data.py prepare --output NEW_EEX_WORK --archive-local /absolute/path/eex_primary_auction_2012_2025.zip
```

第二条适用于已经正常取得的本地归档。固定 ZIP 字节数为 `784392`，SHA256 为 `c3ff6ffaaf348cbd17f61d4726fdbd45a7ead089a0006568a490dd63bb080231`。若尚无归档，既有程序支持对固定官方地址进行一次正常下载：

```bash
python -B EEX_REPRO_SOURCE/reproduce_eex_data.py prepare --output NEW_EEX_WORK --download
```

本地归档与下载选项互斥；错误时停止，不改用其他来源。成功后的开发文件位于：

```text
NEW_EEX_WORK/local_only/prepared/development/EEX_EUA_T3PA_daily_development.csv
```

**既有 EEX 复原程序的范围比本轮模型输入更宽：**实际 `prepare` 会解析完整归档，并机械生成及校验开发、2024 确认、2025 确认共六份输出文件。这些动作不等于将确认样本用于模型，却也不能描述成“从未解析或读取确认字节”。本轮 SMP 仅映射固定开发 CSV，`confirmation_access=false`，没有确认训练或评分入口。本次交付验证没有再次执行 EEX 数据复原、下载归档或读取任何真实 EEX/确认数据。EEX 报价及派生逐行数据不随公开代码重新分发。

EEX 的 `check` 仅检查源码；其 `prepare --dry-run --download` 不请求网络、不创建目录；`prepare --dry-run --archive-local ZIP` 仍会读取并校验 ZIP。请不要混淆这些不同命令的读取范围。

## 3. 创建独立工作目录

先自行写一个 JSON 映射，例如 `/absolute/path/smp_development_map.json`：

```json
{
  "EEX_EUA_T3PA": "/absolute/path/EEX_EUA_T3PA_daily_development.csv",
  "GUANGZHOU": "/absolute/path/GUANGZHOU_development.csv",
  "HUBEI": "/absolute/path/HUBEI_development.csv",
  "SHANGHAI": "/absolute/path/SHANGHAI_D4_development.csv",
  "EU_REPORTED": "/absolute/path/EU_REPORTED_development.csv"
}
```

必须恰有这五个键；重复键、缺失键或额外键均被拒绝。输入相对路径相对于该 JSON 所在目录解析，生成的映射使用绝对路径，避免运行目录改变含义。源协议中的历史路径始终保留，通过训练器已有的 `--data-map` 参数在内存中重定位。

可先查看准备计划：

```bash
python -B SMP_SOURCE/source/prepare_reproduction.py prepare --work-root NEW_SMP_WORK --data-map /absolute/path/smp_development_map.json --dry-run
```

这里的 `--dry-run` **仍会读取并哈希显式映射的五份开发文件**，但不解析 CSV、创建目录或执行训练。若只想检查代码，请使用第 1 节的 `check-sources`。

实际准备：

```bash
python -B SMP_SOURCE/source/prepare_reproduction.py prepare --work-root NEW_SMP_WORK --data-map /absolute/path/smp_development_map.json
```

`NEW_SMP_WORK` 必须不存在。已经存在的目录、文件或悬空符号链接都会在读取映射数据前被拒绝。所有源文件与五份数据的身份均通过后才创建目录；写入失败会保留未完成目录，不自动删除、重试或覆盖。开始新运行时请使用另一个全新目录并保留失败记录。

成功后生成：

- 原字节协议、12 个冻结训练源文件、许可证、独立审核单文件。
- `data_map.json`：重定位后的五条显式开发路径。
- `provenance/PREPARATION_RECEIPT.json`：新准备时间、入口哈希、源与数据哈希、路径及范围。
- `REPRODUCTION_COMMANDS.json`：参数数组、带 shell 引号的可复制命令和每一步条件。

此时状态为 `PREPARED_NO_FITS`。入口不会创建原运行的 `STARTED`、训练结果、模型权重或审核通过记录。

## 4. 手动执行 D1、独立审核及条件式 D2

以下命令在 `NEW_SMP_WORK` 目录执行。准备入口也会打印使用当前 Python 解释器和绝对路径的等价命令，因此无需硬编码原研究机器路径。

### D1：先运行，再选择，再独立重算

```bash
python -B source/run_development.py --protocol provenance/PROTOCOL.json --phase D1 --data-map data_map.json
python -B source/selection.py --protocol provenance/PROTOCOL.json --phase D1
python -B source/independent_results.py --protocol provenance/PROTOCOL.json --phase D1 --output audits/D1_independent
```

逐条执行，前一步完整成功后再执行下一步。D1 固定 160 个任务：120 个神经拟合、25 个 ridge 拟合、5 个 Markov MLE、5 个项目既有参考预测、5 个 RW 预测。神经模型只用 seed 17；全部拟合、零轮检查点选择和失败状态均由原训练器记录。

生产选择写入 `results/D1_SELECTION.json`。独立审核先验证整个阶段的完成记录，再读取本次 `local_only` 预测 NPZ，独立重算支持、价格误差、分母、种子聚合和选择条件，输出 `METRIC_AUDIT.json`、`RECOMPUTED_METRICS.json`、`SELECTION.json` 等记录。它不导入生产数据、模型、评分或选择实现。

审核输出目录必须不存在。审核失败、任务未完成、两份选择不一致或 `advance_to_D2=false` 时停止该固定候选，保留全部结果。不得通过改变 alpha、种子、损失、样本范围或新增网格绕过门槛。

### 仅在 D1 完整通过时运行 D2

先确认本次独立 D1 审核为 PASS、独立选择与生产选择一致，且 `advance_to_D2=true`。然后执行：

```bash
python -B source/run_development.py --protocol provenance/PROTOCOL.json --phase D2 --selection results/D1_SELECTION.json --data-map data_map.json
python -B source/selection.py --protocol provenance/PROTOCOL.json --phase D2 --selection results/D1_SELECTION.json
python -B source/independent_results.py --protocol provenance/PROTOCOL.json --phase D2 --d1-selection audits/D1_independent/SELECTION.json --output audits/D2_independent
```

D2 使用 seeds 17、29、43，按冻结规则取每个模型自身 D1 选中配置与 SMP-full 选中配置的并集，不重复相同配置。最多 345 个神经拟合，加固定 40 个统计或参考任务，实际总数由本次 D1 选择决定。

D2 的独立审核另输出 `PAIRED_UNCERTAINTY.json`。点门槛失败时不启动 bootstrap；点门槛通过后执行冻结的联合日历分块不确定性分析。种子先在损失层面平均，不能冒充更多独立市场或独立测试年份。

原训练器的预算分别为 D1 2700 秒、D2 7200 秒。它们是协作式截止：检查后不启动新的任务或训练 batch，已经开始的验证、恢复、预测或闭式计算可能稍后完成；不宣称硬中断。新运行不允许自动续期、静默续跑或覆盖既有结果。

本交付没有确认阶段命令。即使本次 D2 通过，也需按主协议另行冻结确切确认数据、模型及拟合数量后才能开展新的确认操作；准备程序和本次重放本身不构成该操作的授权。

## 5. 交付验证的实际范围

本次仅核对真实的冻结源码、协议、许可证、审核脚本与 EEX CLI，并以人工创建的五文件数据映射检查路径解析、哈希拒绝、重复键、目录拒绝和字节复制。没有执行真实市场的 `prepare`、没有读取真实开发或确认 CSV、没有重复训练或把旧结果移入新目录。

验证入口为 [source/check_prepare_reproduction.py](source/check_prepare_reproduction.py)，本次 63 项检查全部通过。收据和交付身份见 `provenance/REPRODUCTION_STATIC_CHECK.json` 与 `provenance/REPRODUCTION_DELIVERY.json`。若需要在本地重新执行这种源码及人工文件检查，指定一个尚不存在的输出文件：

```bash
python -B SMP_SOURCE/source/check_prepare_reproduction.py --output /absolute/path/new_source_fixture_check.json
```

这种检查验证了准备流程的边界，不能替代新运行完成后的独立数值审核，也不证明统计优势或发表结论。
