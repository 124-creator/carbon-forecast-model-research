# 便携复现：在新目录重建数据并重跑研究

本入口不依赖原工作区的 `/workspace/...` 路径，不需要账号或凭据。使用全新的数据目录和全新的研究副本；已有目录一律拒绝覆盖。创建副本不会训练模型。只有显式执行后面的 `run` 命令才开始该阶段拟合。

## 1. 数值环境

原研究关键环境为 Linux、Python **3.12.14**、NumPy **2.5.4**、PyTorch **2.3.1+cpu**；读取上海原工作簿使用 openpyxl **3.1.5**、et_xmlfile **2.0.0**。当前四市场研究源码不依赖 transformers；以前 Stage B 的 transformers 环境不属于此入口的必要依赖。

在已提供 Python 3.12.14 的机器上建立独立环境，例如：

```bash
python3.12 -m venv /your/path/carbon_reproduction_venv
source /your/path/carbon_reproduction_venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1+cpu
python -m pip install -r /path/to/portable_helpers/requirements-portable.txt
python -c 'import platform; print(platform.python_version())'
```

这些是未来用户的安装命令，本次编写入口没有重新安装环境。requirements 文件固定关键直接依赖，不冒充完整传递依赖锁。离线安装可以使用包含相同版本及其传递依赖的 wheelhouse：`python -m pip install --no-index --find-links /path/to/wheels -r requirements-portable.txt`。

## 2. 准备五个派生文件

安装后的相对布局为：

| 路径 | 记录数 |
|---|---:|
| `development/GUANGZHOU.csv` | 1,434 |
| `development/HUBEI.csv` | 1,408 |
| `development/EU_REPORTED.csv` | 1,515 |
| `development/SHANGHAI_D4.csv` | 1,507 |
| `confirmation_sealed/SHANGHAI_D4.csv` | 261 |

如果最终公开仓库提供这些按原样保存的派生 CSV，优先验证并复制：

```bash
python -B /path/to/portable_helpers/install_main_data.py --package-dir /path/to/published_data_package --output /your/path/new_carbon_data
```

`--package-dir` 支持上表的相对路径，也支持最终仓库 `research/cross_market_innovation_20261017/data/` 的五个平铺文件：`GUANGZHOU_development.csv`、`HUBEI_development.csv`、`EU_REPORTED_development.csv`、`SHANGHAI_D4_development.csv`、`SHANGHAI_D4_confirmation.csv`。两种布局的内容身份要求完全相同。此模式只复制核对后的数据字节，不联网。若同一数据的两个别名同时存在，会拒绝歧义，不任意选择。

若已有原始文件缓存，目录内放置 `guangzhou_reported.csv`、`hubei_reported.csv`、`eu_reported.csv`、`mendeley_d4_data.zip`，再运行：

```bash
python -B /path/to/portable_helpers/install_main_data.py --source-cache /path/to/raw_cache --output /your/path/new_carbon_data
```

如果没有本地数据，可以显式下载四个固定公开文件：

```bash
python -B /path/to/portable_helpers/install_main_data.py --download --output /your/path/new_carbon_data
```

此命令只访问固定提交上的三个 GitHub 原始 CSV 和上海 Mendeley ZIP。没有搜索、认证、替代镜像、自动重试或官方接口抓取。文件大小或哈希不符、访问被拒或出现非预期重定向都会停止并保留失败收据，不自动换源。本地模式不会静默转为联网模式。

加 `--dry-run` 只展示计划，不创建目录、不读价格文件、不联网；目标目录仍必须不存在。格式提取不打印确认价格、价格分布或模型分数。由原始文件提取时，程序会为校验和导出读取价格；这一过程不被描述为新的盲测。

## 3. 创建独立研究副本

将本入口各文件放在同一 `portable_helpers` 目录。然后指定最终仓库根目录，或直接指定其 `research/cross_market_innovation_20261017` 研究目录：

```bash
python -B /path/to/portable_helpers/prepare_reproduction_workdir.py --study-dir /path/to/carbon-forecast-model-research --data-dir /your/path/new_carbon_data --output /your/path/new_carbon_study
```

加 `--dry-run` 可先核查历史协议与源码身份；该模式不打开数据文件。正式准备会按字节验证并复制五个 CSV，创建相对路径 registry、源码/vendor 副本和文件清单。D1 使用 `frozen_d1_source`，D2、C 分别使用各自冻结源，额外学习诊断使用根目录的 `learning_diagnostic.py`，原模型文件逐字节不变。四份历史计划只保存到 `historical_protocols/` 供参数核对；不复制历史结果、权重、逐点预测、开始/完成标记，也不把历史训练算作新的训练。制图脚本 `make_figures.py` 同样复制并记录身份，是否执行制图由用户选择。

若已经使用最终仓库的五个公开派生文件，也可跳过第二步，直接把 `--data-dir` 指向上述仓库 `data/` 目录；生成器同样按固定字节数与 SHA256 校验，并将其复制为新工作区的标准相对布局。

通过启动器运行时，当前目录自动切到新研究根目录，因此整个新目录在阶段开始前移动到其他路径仍可运行。C 的确认文件位置只在内存中绑定为相对路径，不修改原 `run_confirmation.py` 文件。

## 4. 按顺序执行研究

以下命令从任意目录均可调用。每个命令是一个独立进程，不把多个市场拟合任务并行执行。

```bash
python -B /your/path/new_carbon_study/reproduce_study.py environment
python -B /your/path/new_carbon_study/reproduce_study.py d1 check
python -B /your/path/new_carbon_study/reproduce_study.py d1 init
python -B /your/path/new_carbon_study/reproduce_study.py d1 run
python -B /your/path/new_carbon_study/reproduce_study.py d1 summarize
python -B /your/path/new_carbon_study/reproduce_study.py d2 check
python -B /your/path/new_carbon_study/reproduce_study.py d2 init
python -B /your/path/new_carbon_study/reproduce_study.py d2 run
python -B /your/path/new_carbon_study/reproduce_study.py aggregate
python -B /your/path/new_carbon_study/reproduce_study.py learning init
python -B /your/path/new_carbon_study/reproduce_study.py learning run
python -B /your/path/new_carbon_study/reproduce_study.py c init
python -B /your/path/new_carbon_study/reproduce_study.py c run
python -B /your/path/new_carbon_study/reproduce_study.py aggregate-c
```

完整流程共 **385 次正式预测器拟合**：D1 **141** 次，D2 **158** 次，额外学习诊断 **24** 次，C **62** 次。另有 D2 **9** 次、C **6** 次插补器拟合，共 **15** 次。每阶段的原始总时限、单进程、单线程、float64、种子、最多轮数与早停规则均保留；没有追加自动重试。学习诊断保留独立 **600 秒** 总时限。若机器太慢导致阶段时限内无法完成，会保留未完成状态并停止，不扩大预算或选择部分结果。

`learning init/run` 直接调用冻结原脚本同名函数，使用 D2 训练实现。它只检验开发期历史长度与已标注窗口数量的关系，不读取确认数据，不修改 `DEVELOPMENT_AGGREGATE.json` 或 C 的选择。即使学习诊断完成顺序改变，C 的入口仍只依赖 D1、D2 和开发汇总。学习诊断计划中的 D2 哈希会绑定为新工作区已核验的 D2 计划，并再次检查该依赖，不能用任意新哈希跳过核验。

每次 `init` 先用冻结源码重建新计划，更新当前 registry、源码与依赖文件哈希，然后核对历史完整任务表与参数。新旧计划的时间戳、运行路径及结果依赖哈希自然不同。C 的开发分数可以因数值环境存在差异，但所选配置与所有确认任务必须匹配；不匹配就停止，不替换候选或修改已生成计划。新 C 计划保守标注已公开数据复现，不声称再产生新的未见确认集。

原时序诊断的 `TIMING_PLAN.json` 只作为历史设置依赖复制，此入口**没有重跑那三次短计时**。它重跑的是上述正式阶段；不要把复制的计时方案描述为本次新计时结果。

如果源仓库包含 `aggregate_confirmation.py`，生成器会显式复制它并记录哈希，最后的 `aggregate-c` 调用其正式统计 CLI。当前已从源码核对其路径相对研究目录，依赖 NumPy 及 C 源中的 `paired_statistics.py`；本次未执行该统计汇总。其生成的逐点 CSV/NPZ 留在新工作区的 `results/`，该目录与 `local_only/` 均由新副本 `.gitignore` 排除。没有该可选脚本时，入口会明确停止，不代造统计结果。

独立审计器可依最终仓库说明运行。审计应核对保存的数组、支持与指标；不能只比较压缩 NPZ 文件自身字节，因为归档容器元数据可能不同。

### 可选：生成原研究图与学习曲线表

完成以上四阶段及 `aggregate-c` 后，安装本次实际制图环境所用的 Matplotlib **3.10.8**，再显式制图：

```bash
python -m pip install -r /your/path/new_carbon_study/requirements-figures.txt
python -B /your/path/new_carbon_study/reproduce_study.py figures
```

启动器调用原 `make_figures.py` 的 `main()`；原脚本没有子命令。此步读取新生成的 D1、D2、学习诊断和 C 汇总，不重新拟合市场模型。输出 `figures/confirmation_results.png`、`figures/confirmation_results.svg`、`figures/learning_curves.png`、`figures/learning_curves.svg` 和 `results/LEARNING_CURVES.csv`。任一目标已存在即停止，不覆盖已有图或表。Matplotlib 不是训练必需依赖，本次更新入口没有再次安装或执行它。

## 5. 可复现性的准确范围

原始字节、派生 CSV 字节、冻结源码以及训练配置可以严格核对；关键数值版本由启动器检查。固定种子与确定的任务顺序保留，但不同 CPU、数学库或运行时细节不自动保证所有浮点数组逐位一致。应报告实际差异，不以重新选择配置来消除差异。

此便携入口在编写期间只做合成数据、路径与元数据验证；未重新下载公开市场文件，未读取市场确认 CSV，未重新执行市场训练。既有来源曾成功获取、既有导出有独立核验，是原研究的证据；它们不等于新入口已经完成了一轮端到端市场复现。实际验证明细见 `PORTABILITY_VALIDATION.json`。
