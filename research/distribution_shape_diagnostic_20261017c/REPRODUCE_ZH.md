# 复现本轮条件分布形状必要性实验

本目录属于独立新研究，父提交为 `e2d99b8a5c26c4d7b363ec728eadab6d3fbabbcd`。不覆盖 SMP 或其他历史研究。正式协议是 `provenance/PROTOCOL.json`；所有数值以本轮真实运行结果为准，不复制历史 PASS 来冒充执行。

## 环境

实际训练使用 Python 3.12.14、NumPy 2.5.4、SciPy 1.17.0、PyTorch 2.3.1+cpu，CPU float64、两个进程、每进程一个数学线程，关闭 MKLDNN 并启用 PyTorch 确定性算法。图表由 Matplotlib 3.10.8 导出。其他环境的数值可能有微小差异，不能预先保证逐位一致。

`vendor/deep_npts/` 已含 Apache-2.0 许可、NOTICE、固定作者文件和只替换导入/配置验证的本地加载器，不需要安装完整 GluonTS。这里执行的是作者数学核心的直接多期限任务适配，没有复现原论文整套数据、递归估计器及训练协议。

## 准备五个开发文件

四个公开作者开发 CSV 在仓库的 `research/cross_market_innovation_20261017/data/`。EEX 文件须由仓库 `research/event_feedback_innovation_20261017/data_reproduction/` 的官方档案准备器生成，或使用已经留存且 SHA-256 完全一致的开发 CSV。准备器和数据权利、归档版本、参数说明见该目录 `EEX_DATA_METHODS_ZH.md`。

本轮不会提供 EEX 原始报价，不读取已有 C 文件；EEX 历史准备器可能机械导出 D/C 文件，这不同于把 C 交给模型。此模型入口只接受协议中五个开发文件的精确摘要，并拒绝包含 2024 年起后缀的 CSV。新找到的深圳附件不在本次拟合输入中。

创建 `data_map.json`，只包含下列五个键，值替换为对应开发 CSV 的绝对路径：

```json
{
  "EEX_EUA_T3PA": "/your/path/EEX_EUA_T3PA_daily_development.csv",
  "GUANGZHOU": "/your/repo/research/cross_market_innovation_20261017/data/GUANGZHOU_development.csv",
  "HUBEI": "/your/repo/research/cross_market_innovation_20261017/data/HUBEI_development.csv",
  "SHANGHAI": "/your/repo/research/cross_market_innovation_20261017/data/SHANGHAI_D4_development.csv",
  "EU_REPORTED": "/your/repo/research/cross_market_innovation_20261017/data/EU_REPORTED_development.csv"
}
```

不要改写冻结协议里的源摘要来迁就不同数据。`--data-map` 只允许位置迁移，保留原始协议和数据身份。

## 真实重新运行

在本研究目录、已经准备好上述依赖的 Python 环境中执行。`fresh_run/results` 和 `fresh_run/local_only` 两个输出目录必须尚不存在：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B source/run.py \
  --protocol provenance/PROTOCOL.json \
  --data-map data_map.json \
  --output fresh_run/results \
  --private-output fresh_run/local_only
```

该命令实际重新拟合 30 次神经网络和 10 次 GARCH，生成 65 组预测；不是读取历史汇总。每个工作进程只在当前序列训练/验证信息上估参，2022 年内不更新参数。输出保留全部拟合、早停轨迹、失败及真实运行时间；无自动重试。预算为 2,400 秒的协作式截止检查，已经开始的保存/预测工作可能略晚完成，不声称硬实时终止。

`fresh_run/local_only` 含逐点概率分布、目标、尺度、验证预测和权重，不能将其混入公开交付。公开误差表可由 `source/export_results.py` 从聚合 JSON 导出，不读取报价或重新训练。

## 接着独立验分

`run.py` **不会自动增加 `D1` 子目录**。上一节的两个实参就是本次新运行的公开结果根目录和私有预测根目录；独立验分必须使用完全相同的两个目录：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B source/independent_results.py audit \
  --protocol provenance/PROTOCOL.json \
  --results fresh_run/results \
  --private-output fresh_run/local_only \
  --freeze provenance/INDEPENDENT_RESULTS_FREEZE.json \
  --output fresh_run/results/INDEPENDENT_D1_AUDIT.json
```

| 用途 | 上述新运行的实际路径 |
|---|---|
| 运行身份与协议摘要 | `fresh_run/results/PRE_RUN.json` |
| 各序列的结果与支持声明 | `fresh_run/results/<market>.json` |
| 主聚合结果 | `fresh_run/results/RESULTS.json` |
| 逐起点预测 | `fresh_run/local_only/<market>/<family>.npz` |
| 新独立验分收据 | `fresh_run/results/INDEPENDENT_D1_AUDIT.json` |

`--results` 接收包含 `PRE_RUN.json`、五份市场 JSON 和 `RESULTS.json` 的**目录**，不是 `RESULTS.json` 文件路径。`--private-output` 接收直接包含五个市场目录的父目录，不要再追加 `/D1`。原始实验使用 `results/D1` 和另一处 `local_only/D1`，是因为当时显式传入了这些路径，不是程序隐式创建的层级。

原 freeze 文件验证审核器的源字节身份；对新运行必须实际执行上述命令，生成新的 PASS 或错误记录。原研究的审计 PASS 不能作为新运行已经验分的证明。该审核器读取保存预测及公共元数据，不重新拟合、不读取源价格 CSV 或 checkpoint；其完整范围见 `INDEPENDENT_AUDIT_ZH.md`。

## 从聚合结果导出图表与摘要

以下两项只读取新的 `RESULTS.json`，无需私有 NPZ、报价文件或重新训练。选用新的 `fresh_export` 目录保存导出；它不需要包含 `source/`。

```bash
mkdir -p fresh_export/provenance
MPLCONFIGDIR=/tmp/carbon_shape_mpl python -B source/export_results.py \
  --result fresh_run/results/RESULTS.json \
  --study fresh_export

python -B source/summarize_report.py \
  --result fresh_run/results/RESULTS.json \
  --output fresh_export/results/REPORT_SUMMARY.json
```

图表写入 `fresh_export/figures`，误差表写入 `fresh_export/results/tables`，图表导出身份写入 `fresh_export/provenance/TABLE_FIGURE_EXPORT.json`。前置的 `mkdir` 为该身份文件准备目录。摘要命令输出新的 `fresh_export/results/REPORT_SUMMARY.json`。这些都是新运行的聚合导出，不复制原研究的表格或摘要充当新结果。

## 解释边界

CRPS 对实际发布的 31 原子有限分布精确计算，不把 31 个原子或 GARCH-FHS 的 2,048 条模拟路径当作新增观测。所有主诊断分布的均值固定为当前价格，因此均值 MSE 等于 RW；这不构成点预测增益。2022 年属于已经暴露的开发区，当前协议没有 D2、置信区间确认或 C 运行。即便必要性门槛通过，也还需要真正的新机制、近邻比较和独立数据验证。
