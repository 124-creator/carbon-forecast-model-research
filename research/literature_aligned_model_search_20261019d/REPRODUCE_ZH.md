# 复现入口

以下命令重跑四个较早完成的阶段，以及随后完成的官方天津 V4，并把新输出放在独立目录。V1/V2 是已暴露的开发年份；V3 的原主候选为 `liquidity_full`，原次候选为 `allocation_readout`；SWEWT 是看过部分 V3 结果后增加的描述性对照。官方天津 V4 在这些结果之后指定 `allocation_readout` 为新阶段主候选。重跑不改变这些历史身份。

## 1. 路径与固定输入

先进入**完整仓库根目录**，选择装好依赖的 Python；输出目录应为新目录。

```bash
CARBON_REPRO_REPO="$PWD"
CARBON_REPRO_STUDY="$CARBON_REPRO_REPO/research/literature_aligned_model_search_20261019d"
CARBON_REPRO_OUT=/tmp/carbon-reproduction-20261019d
CARBON_REPRO_PY=python
CARBON_REPRO_SCIENCEDB="$CARBON_REPRO_STUDY/data/sciencedb_819_v1/prepared"
mkdir -p "$CARBON_REPRO_OUT"
```

本次记录的 Python 为 3.12.14。包版本见 [RUNTIME.json](provenance/RUNTIME.json)：NumPy 2.3.5、pandas 2.2.3、SciPy 1.17.0、scikit-learn 1.8.0、PyTorch 2.14.1+cpu；工作簿解析还需 xlrd 2.0.1、openpyxl 3.1.5。SWEWT 使用 ewtpy 0.2，其具体模块 SHA 另见补充运行协议。Python 必须能正常导入这些依赖；不同环境下的数值结果应重新核验，不能仅凭种子相同承诺逐位一致。

程序读取以下已有固定输入，并将其 SHA 写入各次运行的 manifest：

- `research/cross_market_innovation_20261017/data/{GUANGZHOU,HUBEI,SHANGHAI_D4,EU_REPORTED}_development.csv`；
- `data/parsed/{brent,wti,henryhub}_eia_daily.csv`；
- V3 的 ScienceDB 三市场表，默认使用本研究包的 `data/sciencedb_819_v1/prepared/`。

上述输入缺失时，应先恢复相应固定来源。不能把不同数据库的价格静默替换后称为相同实验的复现。公开 `results/` 保留协议、摘要等轻量记录，不包含原运行的 NPZ 或 checkpoint；独立审计完整成绩需要下述重跑产生的完整输出，或原有完整本地运行目录。

### V2 作者成交量辅助表

固定作者原件与对齐规则见 [PUBLIC_AUTHOR_REPRODUCTION_ZH.md](provenance/PUBLIC_AUTHOR_REPRODUCTION_ZH.md)。下面两步只下载/解析数据，不拟合模型：

```bash
"$CARBON_REPRO_PY" "$CARBON_REPRO_STUDY/provenance/download_public_author_data.py" \
  --out-dir "$CARBON_REPRO_OUT/author_raw"

"$CARBON_REPRO_PY" "$CARBON_REPRO_STUDY/provenance/prepare_auxiliary_alignment.py" \
  --raw-root "$CARBON_REPRO_OUT/author_raw" \
  --repo-root "$CARBON_REPRO_REPO" \
  --out-dir "$CARBON_REPRO_OUT/volume_prepared" \
  --receipt "$CARBON_REPRO_OUT/volume_alignment.json"
```

如需独立重建 ScienceDB 包，可先执行下列纯离线命令，再将三个模型阶段共用的路径变量指向其输出。原件和数据归属见 [数据 README](data/sciencedb_819_v1/README_ZH.md)。

```bash
"$CARBON_REPRO_PY" "$CARBON_REPRO_STUDY/provenance/reproduce_sciencedb_transfer.py" \
  --repo-root "$CARBON_REPRO_REPO" \
  --raw-file "$CARBON_REPRO_STUDY/data/sciencedb_819_v1/raw/sciencedb_819_v6.xls" \
  --out-dir "$CARBON_REPRO_OUT/sciencedb"
CARBON_REPRO_SCIENCEDB="$CARBON_REPRO_OUT/sciencedb/prepared"
```

## 2. 四轮模型命令

每一轮均使用既定默认模型、学习率和 seed 预算。只有等待该轮 `EXECUTION_COMPLETE.json` 正常完成，才应审计或使用其全部成绩。统计上的主指标为 **mean-seed MSE/RW**；三种子先平均预测的集成结果另列为次要指标。

### V1：四市场 × 2022/2023 开发

```bash
"$CARBON_REPRO_PY" "$CARBON_REPRO_STUDY/source/run_model_search.py" \
  --out "$CARBON_REPRO_OUT/main_v1" \
  --markets GUANGZHOU HUBEI SHANGHAI_D4 EU_REPORTED \
  --years 2022 2023 --workers 4
```

预期 88 份 NPZ，240 项神经任务；任务数包含最终选择 epoch 0 的零更新重训，不能等同于 240 次均有梯度更新的训练。

### V2：广州/湖北成交量机制扩展

```bash
"$CARBON_REPRO_PY" "$CARBON_REPRO_STUDY/source/run_liquidity_extension.py" \
  --out "$CARBON_REPRO_OUT/liquidity_v2" \
  --auxiliary-root "$CARBON_REPRO_OUT/volume_prepared" \
  --markets GUANGZHOU HUBEI --years 2022 2023 --workers 4
```

预期 44 份新 NPZ，120 项神经任务；统计比较另外复用 V1 中相同四个市场/年份单元的 `allocation_constant` 输出。

### V3：ScienceDB 北京/天津复制、重庆压力测试

```bash
"$CARBON_REPRO_PY" "$CARBON_REPRO_STUDY/source/run_new_market_replication.py" \
  --out "$CARBON_REPRO_OUT/new_market_v3" \
  --data-root "$CARBON_REPRO_SCIENCEDB" \
  --markets BEIJING_MEAN TIANJIN_MEAN CHONGQING_MEAN --workers 4
```

预期 39 份 NPZ，120 项神经任务。必须保留原 [V3 统计计划](audits/V3_STATISTICAL_AUDIT_PLAN.json) 的字节身份；`fixed602020` 是分区标签，不是年份。

### SWEWT-GRU：独立补充结构对照

```bash
"$CARBON_REPRO_PY" "$CARBON_REPRO_STUDY/source/run_swewt_gru_comparison.py" \
  --out "$CARBON_REPRO_OUT/swewt_gru_supplement" \
  --data-root "$CARBON_REPRO_SCIENCEDB" \
  --reference-v3-run "$CARBON_REPRO_OUT/new_market_v3" \
  --markets BEIJING_MEAN TIANJIN_MEAN CHONGQING_MEAN --workers 4
```

预期 6 份 NPZ，30 项神经任务。使用同一 `data-root` 下刚生成的 V3 reference run；其 manifest 包含绝对路径，不能直接以旧目录的归档 manifest 代替新路径下的参照。该对照是固定参数与预算的多步受控改编，不是作者数值复现，也不进入原 V3 六项检验家族。

## 3. 独立误差与执行审计

这些脚本不启动模型训练。V1、V2、V3 的统计脚本包含各自已固定的块法；SWEWT 脚本只复算新预测并作描述性比较。审计输出另放目录，保持既有报告原样。

```bash
"$CARBON_REPRO_PY" "$CARBON_REPRO_STUDY/audits/independent_v1_statistics.py" \
  --run "$CARBON_REPRO_OUT/main_v1" --out "$CARBON_REPRO_OUT/audit_v1"

"$CARBON_REPRO_PY" "$CARBON_REPRO_STUDY/audits/independent_v2_statistics.py" \
  --run "$CARBON_REPRO_OUT/liquidity_v2" \
  --v1-run "$CARBON_REPRO_OUT/main_v1" \
  --v1-audit "$CARBON_REPRO_OUT/audit_v1/V1_INDEPENDENT_STATISTICAL_AUDIT_ZH.json" \
  --auxiliary-root "$CARBON_REPRO_OUT/volume_prepared" \
  --out "$CARBON_REPRO_OUT/audit_v2"

"$CARBON_REPRO_PY" "$CARBON_REPRO_STUDY/audits/independent_v3_statistics.py" \
  --run "$CARBON_REPRO_OUT/new_market_v3" \
  --data-root "$CARBON_REPRO_SCIENCEDB" --out "$CARBON_REPRO_OUT/audit_v3"

"$CARBON_REPRO_PY" "$CARBON_REPRO_STUDY/audits/independent_swewt_statistics.py" \
  --run "$CARBON_REPRO_OUT/swewt_gru_supplement" \
  --reference-v3-run "$CARBON_REPRO_OUT/new_market_v3" \
  --v3-audit "$CARBON_REPRO_OUT/audit_v3/V3_INDEPENDENT_STATISTICAL_AUDIT_ZH.json" \
  --data-root "$CARBON_REPRO_SCIENCEDB" --out "$CARBON_REPRO_OUT/audit_swewt"
```

每份独立结果 JSON 记录实际路径、源文件及预测哈希、所用审计脚本 SHA、逐市场/步长误差与执行核数。正梯度次数来自哈希匹配执行收据核对；这些脚本不会把 checkpoint 字节检查改称反向传播或模型预测重演。V1 readout 排名后发现性诊断和初始化归因另见各自报告，不混入上述原统计家族。

## 4. 官方天津 V4：固定较近时期检验

[V4 固定统计计划](audits/V4_STATISTICAL_AUDIT_PLAN.json) 已给出官方 1077 行样本、2023-04-19 起的尾段、196 个共同 H20 成熟测试原点，以及在旧结果后指定的主 `allocation_readout`、次 `liquidity_full`。五项对比、20/40 块长和多重校正规则已固定；此前 V3 身份不变。

固定官方快照可按 [数据收据](provenance/TIANJIN_OFFICIAL_CANONICAL_QC.json) 由下列解析脚本重建。已有原件时传 `--raw-file`，完全离线；省略该参数会读取固定官方公开地址，并拒绝与冻结 SHA 不同的内容。

```bash
"$CARBON_REPRO_PY" "$CARBON_REPRO_STUDY/provenance/prepare_tianjin_official_snapshot.py" \
  --raw-file /absolute/path/to/tianjin_body.js \
  --out-dir "$CARBON_REPRO_OUT/tianjin_official_data"
```

运行器使用 `--data-file` 指定单个固定四列 CSV。它会在拟合前把原 V4 统计计划 SHA、源码与数据身份绑定到新协议。普通输入和 SWEWT 输入在同一市场目录分别写 `INPUT_MANIFEST.json` 与 `SWEWT_INPUT_MANIFEST.json`。

```bash
"$CARBON_REPRO_PY" "$CARBON_REPRO_STUDY/source/run_official_tianjin_extension.py" \
  --out "$CARBON_REPRO_OUT/official_tianjin_v4" \
  --data-file "$CARBON_REPRO_OUT/tianjin_official_data/TIANJIN_OFFICIAL_MEAN.csv" \
  --workers 4

"$CARBON_REPRO_PY" "$CARBON_REPRO_STUDY/audits/independent_v4_statistics.py" \
  --run "$CARBON_REPRO_OUT/official_tianjin_v4" \
  --data-file "$CARBON_REPRO_OUT/tianjin_official_data/TIANJIN_OFFICIAL_MEAN.csv" \
  --out "$CARBON_REPRO_OUT/audit_v4"
```

预期 15 份 NPZ、50 项神经任务（20 次选择、30 次重训）。独立审计只会处理有正常完成标记的运行，按冻结的五项对比、20/40 原点块各 50000 次计算区间；主次身份不随新尾段成绩改变。纯合成统计自检可单独运行，不读取市场数据或训练模型：

```bash
"$CARBON_REPRO_PY" "$CARBON_REPRO_STUDY/audits/independent_v4_statistics.py" \
  --self-test --out "$CARBON_REPRO_OUT/audit_v4_selftest"
```

## 5. 导出汇总、完整训练日志与文件身份

[export_research_results.py](source/export_research_results.py) 读取完整运行目录，导出协议、全部选择/训练日志、原单位汇总及 NPZ/checkpoint 哈希。原件、逐点预测 NPZ 和 torch checkpoint 不会被复制到公开结果目录。以 V4 为例：

```bash
"$CARBON_REPRO_PY" "$CARBON_REPRO_STUDY/source/export_research_results.py" \
  --run-root "$CARBON_REPRO_OUT" \
  --runs official_tianjin_v4 \
  --official-data-root "$CARBON_REPRO_OUT/tianjin_official_data"
```

导出目的地固定为当前 checkout 的 `research/literature_aligned_model_search_20261019d/results/<run-name>/`，同名汇总文件会更新；需要保留原交付快照时，在独立 checkout 中执行。`--runs` 也接受 `main_v1 liquidity_v2 new_market_v3 swewt_gru_supplement`。导出器的 ScienceDB 参照仍读取研究包内 `data/sciencedb_819_v1/prepared/` 的固定副本，官方天津参照由 `--official-data-root` 指定；这些源值必须与对应运行 manifest 一致。

## 6. 独立的信息范围诊断

这是另外的单步 Ridge 实验，原始滞后、历史滑窗 EWT、全序列事后 EWT 共84次估计器拟合，不是五轮神经模型的指标。为保留当前研究结果并满足审计的目录接口，建立独立输出目录并复制冻结源码的同一字节版本：

```bash
mkdir -p "$CARBON_REPRO_OUT/information_replay/source"
cp "$CARBON_REPRO_STUDY/source/literature_diagnostic.py" \
  "$CARBON_REPRO_OUT/information_replay/source/literature_diagnostic.py"

"$CARBON_REPRO_PY" "$CARBON_REPRO_STUDY/source/literature_diagnostic.py" \
  --repo "$CARBON_REPRO_REPO" \
  --out "$CARBON_REPRO_OUT/information_replay/results/information_diagnostic"

"$CARBON_REPRO_PY" "$CARBON_REPRO_STUDY/audits/independent_ewt_diagnostic.py" \
  --repo "$CARBON_REPRO_REPO" \
  --study "$CARBON_REPRO_OUT/information_replay" \
  --predictions "$CARBON_REPRO_OUT/information_replay/results/information_diagnostic/PREDICTIONS.csv" \
  --out "$CARBON_REPRO_OUT/information_audit.json"
```

诊断逐点 CSV 含目标价格路径，应保留在新的本地运行目录。原交付的该 CSV 已迁出公开结果目录，位置与 SHA 见 `results/information_diagnostic/PRIVATE_PREDICTIONS_MANIFEST.json`；不能省略路径然后误用一个不存在的原公开 CSV。

## 7. 回执与图表

对提供 `--preflight` 与 `--preflight-report` 的运行器，如需复跑预检，应将后者显式指定为新的回执路径，避免覆盖当时的原预检证据。不要为了重跑而修改冻结源码或统计计划。

当完整汇总已导出到独立 checkout 后，下面命令重新生成全部四张静态研究图的 PNG/PDF 及哈希 manifest，不拟合模型：

```bash
"$CARBON_REPRO_PY" "$CARBON_REPRO_STUDY/source/make_research_figures.py"
```

模型初始化回放与独立误差审计的范围不同：`diagnose_*` 脚本使用 torch checkpoint 和模型前向；`independent_*_statistics.py` 从保存预测重建指标、选择和区间。前者中部分旧阶段数据路径遵循当时运行的固定位置；迁移时必须恢复记录的输入或明确适配路径，不能用另一份价格/量表代替后仍称原实验。
