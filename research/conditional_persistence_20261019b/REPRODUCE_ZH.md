# 本轮实验复现说明

本文件说明如何运行代码和核对档案。研究结论见 `REPORT_ZH.md`；完整数学定义见 `methods/METHOD_ZH.md`。这是一次有固定模型、固定训练预算和明确失败条件的方法实验，不能通过更换日期、种子、目标权重来“复现”一个更好结果。

## 环境和输入

实际环境记录在 `provenance/ENVIRONMENT.json`，精确依赖版本在 `requirements.txt`。本次使用 Python 3.12.14、CPU PyTorch 2.14.1+cpu，每个进程的 Torch/BLAS 线程数为 1；三个候选 shard 和三个 GAS worker 分别并行。`+cpu` 是实际轮子的版本后缀，复现环境须取得相应发行文件。代码未使用 GPU、numba 或未公开的网络模型服务。

需要四个固定开发 CSV 和 PLOS S1 附件。其名称、来源链、字节哈希、报价缺失规则见 `data_sources/SOURCE_AUDIT.json`、`data_sources/SOURCE_AUDIT_ZH.md`。精确输入及本次运行的模型权重、预测分布保存在独立运行档案；公开仓库包含获取来源、代码和聚合证据。不能将另一作者的同名序列、不同合约或经填值的数据直接当作同一输入。

本次实际输入位置如下；这些是归档前的源路径，解压后的目录按档案清单映射，不应直接假定文件名相同：

| 输入 | 本次实际源路径 | 建议归档相对路径 |
|---|---|---|
| 四个开发 CSV | `/tmp/carbon-distribution-update-data/development/{EUA_LONG,GUANGZHOU,HUBEI,SHANGHAI}_development.csv`，花括号表示四个文件 | `inputs/development/` 下保留各自文件名 |
| 深圳 PLOS S1 | `/tmp/carbon-conditional-persistence-runs/sources/plos_0296105_s001.xls` | `inputs/plos_shenzhen/plos_0296105_s001.xls` |

四个 CSV 合计 187,754 字节，附件 13,625,856 字节。归档时只纳入明确列出的四个 CSV，不扫描其父目录中的其他来源或保留数据。附件在 `inputs/` 保存一份即可，base 运行目录的 `sources/` 可只保留两份下载回执，避免重复打包同一附件。

深圳 S1 的 SHA-256 为 `90c49501afa9beb7370c4409084a28729daa4904a7e7e5de310d0f6bd7a3ba10`。只取 `shenzhen` 的日期和价格列，只有截至 2021 年的有效正价格进入数组。使用 xlrd 时其内部可以解码该 sheet 的其他单元格；实验数组不包含 2022 年。所有市场的 2024 年及以后价格均不进入本轮。

## 在新的工作副本中复现

以下命令从仓库根目录执行。用新的工作副本和新的私有输出目录，保留已发布结果供逐项比较。runner 不会静默覆盖已存在但不完整的拟合。

```bash
set -euo pipefail
export CARBON_PY=/path/to/experiment-python
export CARBON_STUDY=research/conditional_persistence_20261019b
export CARBON_INPUT=/path/to/extracted-private-archive/inputs
export CARBON_RUN=/tmp/carbon-persistence-independent-reproduction
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/carbon-persistence-reproduction-mpl

# 只在新的复现工作副本中移动结果；原始结果作为参考保留。
mv "$CARBON_STUDY/results" "$CARBON_STUDY/published_results_reference"
# 数值审计有拒绝覆盖的保护；同样保留已发表回执。
mv "$CARBON_STUDY/audits/LOGNORMAL_CRPS_CANCELLATION_AUDIT.json" \
  "$CARBON_STUDY/published_results_reference/LOGNORMAL_CRPS_CANCELLATION_AUDIT.json"
mv "$CARBON_STUDY/audits/STABLE_PRICE_CRPS_CELLS.csv" \
  "$CARBON_STUDY/published_results_reference/STABLE_PRICE_CRPS_CELLS.csv"
mv "$CARBON_STUDY/audits/STABLE_PRICE_CRPS_RECOMPUTATION.json" \
  "$CARBON_STUDY/published_results_reference/STABLE_PRICE_CRPS_RECOMPUTATION.json"

"$CARBON_PY" "$CARBON_STUDY/source/run_synthetic.py" \
  --local-dir "$CARBON_RUN/synthetic" \
  --public-dir "$CARBON_STUDY/results/synthetic" --workers 3

"$CARBON_PY" "$CARBON_STUDY/data_sources/prepare_data.py" \
  --data-dir "$CARBON_INPUT/development" \
  --local-dir "$CARBON_RUN/base" \
  --shenzhen-workbook "$CARBON_INPUT/plos_shenzhen/plos_0296105_s001.xls" \
  --market-protocol "$CARBON_STUDY/provenance/MARKET_PROTOCOL.json" \
  --market-protocol-sha256 f2ac9d5aac5a8bbc7d6f7a785e943e307b903d2fc3ab56cc8484b9e1375adc9d

# 完整 checkpoint 审计固定核对本次三份 COMPLETE；按真实三分片执行。
run_carbon_candidate_shard() {
  "$CARBON_PY" "$CARBON_STUDY/source/run_persistence_market.py" \
    --prepared-dir "$CARBON_RUN/base/prepared" \
    --private-dir "$CARBON_RUN/persistence" \
    --public-dir "$CARBON_STUDY/results/market/persistence" \
    --protocol "$CARBON_STUDY/provenance/MARKET_PROTOCOL.json" \
    --threads 1 --panels "$@"
}
run_carbon_candidate_shard EUA_LONG_2023 GUANGZHOU_2021 HUBEI_2021 \
  SHANGHAI_2022 SHENZHEN_PLOS_2020 &
CARBON_SHARD_A=$!
run_carbon_candidate_shard EUA_LONG_2022 GUANGZHOU_2022 HUBEI_2022 \
  SHENZHEN_PLOS_2021 &
CARBON_SHARD_B=$!
run_carbon_candidate_shard EUA_LONG_2021 GUANGZHOU_2023 HUBEI_2023 \
  SHANGHAI_2023 &
CARBON_SHARD_C=$!
wait "$CARBON_SHARD_A"
wait "$CARBON_SHARD_B"
wait "$CARBON_SHARD_C"

"$CARBON_PY" "$CARBON_STUDY/source/run_gas_benchmark.py" \
  --prepare-only --workers 3 \
  --base-prepared-dir "$CARBON_RUN/base/prepared" \
  --local-dir "$CARBON_RUN/gas" \
  --expected-source-hash 8bc1e84129b65fc30db010839523a448e605fae07581fb224a1b7b5e4c2dbf23

"$CARBON_PY" "$CARBON_STUDY/source/run_gas_benchmark.py" \
  --train-only --workers 3 --local-dir "$CARBON_RUN/gas" \
  --expected-source-hash 8bc1e84129b65fc30db010839523a448e605fae07581fb224a1b7b5e4c2dbf23

"$CARBON_PY" "$CARBON_STUDY/source/run_gas_benchmark.py" \
  --aggregate-only --local-dir "$CARBON_RUN/gas" \
  --expected-source-hash 8bc1e84129b65fc30db010839523a448e605fae07581fb224a1b7b5e4c2dbf23

"$CARBON_PY" "$CARBON_STUDY/source/aggregate_persistence_market.py" \
  --private-dir "$CARBON_RUN/persistence"

"$CARBON_PY" "$CARBON_STUDY/source/compare_benchmark_families.py" \
  --persistence-private "$CARBON_RUN/persistence" --gas-private "$CARBON_RUN/gas"

"$CARBON_PY" "$CARBON_STUDY/audits/replay_market_checkpoints.py" \
  --base-prepared "$CARBON_RUN/base/prepared" \
  --candidate-private "$CARBON_RUN/persistence" \
  --candidate-public "$CARBON_STUDY/results/market/persistence" \
  --gas-private "$CARBON_RUN/gas" \
  --cache-dir "$CARBON_RUN/independent-checkpoint-replay" \
  --output "$CARBON_STUDY/audits/INDEPENDENT_CHECKPOINT_REPLAY.json"

"$CARBON_PY" "$CARBON_STUDY/audits/replay_market_scores_qualified.py" \
  --persistence-private "$CARBON_RUN/persistence" --gas-private "$CARBON_RUN/gas" \
  --cache "$CARBON_RUN/independent-score-replay"

"$CARBON_PY" "$CARBON_STUDY/audits/lognormal_crps_cancellation_audit.py" \
  --persistence-private "$CARBON_RUN/persistence" --gas-private "$CARBON_RUN/gas" \
  --private-output "$CARBON_RUN/price-crps-numerical-audit"

"$CARBON_PY" "$CARBON_STUDY/audits/recompute_stable_price_crps.py" \
  --persistence-private "$CARBON_RUN/persistence" --gas-private "$CARBON_RUN/gas" \
  --private-output "$CARBON_RUN/stable-price-crps"

"$CARBON_PY" "$CARBON_STUDY/source/make_figures.py"
```

上方命令使用下列三个真实、互不重复的 `--panels` 清单，共用同一个 private/public 目录；三个进程各自保留 START 和 COMPLETE。当前完整检查点审计固定要求三份完成清单，因此不要以单个全 13-panel 调用替代。若资源只能顺序运行，也应保留这三次独立调用。执行顺序不改变每项实验的数据、种子或预算。

| shard | 面板 |
|---|---|
| A | EUA_LONG_2023、GUANGZHOU_2021、HUBEI_2021、SHANGHAI_2022、SHENZHEN_PLOS_2020 |
| B | EUA_LONG_2022、GUANGZHOU_2022、HUBEI_2022、SHENZHEN_PLOS_2021 |
| C | EUA_LONG_2021、GUANGZHOU_2023、HUBEI_2023、SHANGHAI_2023 |

正式汇总要求这些清单覆盖全部 13 个面板，但不把“三个并行进程”当作统计复制。每个模型的三次训练种子才是本次记录的初始化复制。

## 审计和数值对照

本次档案的数学、时间边界、权重重放、评分、统计汇总和同步日期抽样审计均在 `audits/`。支持命令行参数的审计脚本可先查看 `--help`，再指向解压出的私有目录。`replay_gas_aggregates.py` 与 `replay_synthetic_screen.py` 使用模块内固定的 `PRIVATE` 路径，没有命令行解析；向它们传 `--help` 不会显示帮助，应先阅读源码，在单独复现副本中调整审计路径并记录审计源码身份变化，或恢复到记录的目录。不要为此修改冻结的训练源码。

完整性应同时查看任务数量、缺失和失败字段及状态含义，不能只匹配一个状态前缀。检查点完整通过为 `PASS_ALL_510_CHECKPOINT_REPLAYS`；统计汇总有各自的通过状态。原冻结评分审计发现价格 CRPS 数值抵消而停止，保留原脚本和缓存。完成版为 `COMPLETE_WITH_PRICE_CRPS_NUMERICAL_QUALIFICATION`：其余指标仍须满足原阈值，价格 CRPS 超阈值点如实记为数值资格限制，不是全指标无条件通过。`PARTIAL_TECHNICAL_CHECK` 始终只是当时已完成文件的技术检查。

`replay_market_scores.py` 及有数值资格说明的完成版不导入生产评分模块，独立计算正态绝对差期望、对数正态截断矩和经验原子分布距离。原脚本全量执行预计会重现深圳 h20 的价格 CRPS 阈值失败；不能以加大阈值消除该证据。`lognormal_crps_cancellation_audit.py` 用先符号消去大均值项的公式扫描，并对选定点作 80/120 位高精度和定义 CDF 积分验证。`replay_market_checkpoints.py` 独立重建网络及状态传播；统计重放也不导入生产汇总函数。角色披露保留在回执中：这是内部交叉复核，不是外部同行评审。

合成识别的 720 次低维拟合与正式市场的 510 次神经拟合分别计数。GAS 的六次完整合成神经 smoke、三次合成静态检查，以及候选的 16 个合成优化步骤也另行列出，不加入正式市场拟合数。六次神经与其中两次静态 smoke 的私有产物位于 `/tmp/carbon-gas-pipeline-synthetic-t7ppvkav`；另一次静态检查只在 `audits/formula_reviews.json` 留有回执。候选 16 步 smoke 与后续零梯度接口检查的数组和模型仅存在于内存，没有遗漏的私有权重目录；保留其公开回执即可。

训练预算耗尽时取得有限有效解，不等于优化器收敛。回执保留 success/status/nfev，不追加优化或换模型。全部 40 epoch 都执行，即使验证最后选择 epoch 0；不能将“选择初始模型”误报为“没有训练”。

## 如何解释结果

主统计量是每个起点先平均种子的**评分**，再取市场年度／期限单元均值，计算候选／匹配对照 − 1，最后等权平均单元。其中每个单元内部确实计算候选与对照均值之比；最终统计量不是把所有市场堆叠后计算两组总体均值之比，也不是三个预测分布的 ensemble 得分。

主开发层为 11 个旧面板的 22 个 h5/h20 单元；深圳 4 个单元单列。主区间按固定面板循环块抽样；统一日期的抽样是后来在查看比较结果之前说明的次要敏感性检查。它不改变已冻结的六项继续研究条件，也不证明对未见市场的推广能力。

本轮源码冻结和阶段 START 都发生在相应拟合前；GitHub 上传发生在全部实验和复核完成之后。因此，这些记录是可核对的本地执行溯源，不应称作外部预注册。
