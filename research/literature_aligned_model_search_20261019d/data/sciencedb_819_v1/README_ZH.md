# ScienceDB 819：北京、天津、重庆日成交均价数据

本包固定保存作者发布的 ScienceDB 原工作簿和本研究所用的三个市场表。原 XLS 有七个市场 sheet，本次新增转移评价只取北京、天津、重庆。目标为作者整理的**配额日成交均价（元/吨）**，同时保留日成交量（吨）与成交额（元）。价格直接抄录原工作簿；成交额除以成交量与三个市场的价格逐行相等，比例关系仅用于核验。[字段与数量核验](prepared/DATA_MANIFEST.json)

## 来源、归属和版本

建议引用：张倩云、李瑾、张瑶、赵东、刘琛（2019）. *A dataset for the trade information during 2014–2017 of seven pilot carbon markets in China*. Science Data Bank, V1. [DOI:10.11922/sciencedb.819](https://doi.org/10.11922/sciencedb.819)。

官方元数据标注发布日期为 2019-08-02、许可为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。证据见 [官方元数据](provenance/SCIENCEDB_OFFICIAL_METADATA_CURRENT.json)、[本次读取收据](provenance/LICENSE_METADATA_RETRIEVAL.json) 及 [ScienceDB 发布页](https://www.scidb.cn/detail?dataSetId=633694461347495939)。使用时保留作者、DOI、许可链接及本研究的格式转换说明。

平台引用版本是 V1；本次固定附件的文件名含“v6”，包内保存为 `raw/sciencedb_819_v6.xls`：

- [官方原件下载](https://china.scidb.cn/download?fileId=5eeeecf15e972ddb1b4881d7)。
- 大小：374,272 bytes。
- SHA256：`5d362db0f8e0a7609d35a04afb25a8bcbbc67167b56c41c3800febf2364cf223`。
- MD5：`59f01f50f6e9a8f1b063b9adf2c3e9c1`。
- 获取凭据：[ACQUISITION_RECEIPT.json](provenance/ACQUISITION_RECEIPT.json)。

## 内容和历史划分

| 市场 | 全表行数 | 日期范围 | 原始 train / validation / remainder 行数 | 64 历史与 H20 成熟后的 train / val / test origins |
|---|---:|---|---|---|
| 北京 | 673 | 2014-01-02—2017-12-29 | 403 / 134 / 136 | 319 / 114 / 116 |
| 天津 | 457 | 2014-01-02—2017-06-30 | 274 / 91 / 92 | 190 / 71 / 72 |
| 重庆 | 240 | 2014-06-19—2017-12-29 | 144 / 48 / 48 | 60 / 28 / 28 |

完整 CSV 的四列是 `date, price, volume_tons, turnover_yuan`。新导出统一日期格式、按日期升序并加入原作者数量字段；没有重新清洗、修复价格、补齐日历或插值。上述日期、数量和列定义均来自 [DATA_MANIFEST.json](prepared/DATA_MANIFEST.json)。

划分沿用 2026-10-17 的顺序规则：前 `N*3//5` 行训练，随后 `N//5` 行验证，其余为末段。九个旧分区的 `date/mean_price/source_row` 重新序列化后，旧格式哈希逐个精确一致。新四列表增加了数量字段，因此保留独立的新文件哈希。H1/H5/H20 共用最长标签已成熟的 origins，步数按观测记录计。

`confirmation_sealed` 是保留的历史目录名称。此次新市场模型运行前，本项目已有获取、解析、质量核验与划分，未找到北京、天津、重庆的模型拟合或预测评分记录；模型运行后，该测试段的模型表现即已暴露。当前暴露状态应结合最新运行记录解释。[既有范围](provenance/LEGACY_INDEPENDENT_SOURCE_SCOPE.json)；[既有暴露账本](provenance/LEGACY_EXPOSURE_LEDGER.json)。

## 作者处理与预测含义

作者删除非交易日期，并报告基于月度参考加权均价筛查异常，确认异常后以前一交易日数据替换；逐点替换标志未取得。月内参照可能使用某日起点之后的同月信息，因此本次评价对应**明确界定的作者清洗基准**，不等于原始实时发布版本的部署检验。[解析协议](prepared/PROTOCOL_BEFORE_PARSE.json)

应保留这个来源限制，但不能仅因目标经过作者处理，就把目标本身自动判为预测输入泄露。模型按当前日成交均价和报告成交量已知的时点构造历史输入，要求标签在分区内成熟。`turnover_yuan` 用于来源核验，本轮不把未来成交额作为输入。`price` 应始终表述为日成交均价，不能写成收盘价。

## 复现

[reproduce_sciencedb_transfer.py](../../provenance/reproduce_sciencedb_transfer.py) 下载或核验固定原件，再调用 [prepare_sciencedb_transfer.py](../../provenance/prepare_sciencedb_transfer.py)。解析需要 Python 和 `xlrd==2.0.1`；应保留完整仓库，因为程序还读取旧研究的固定范围与分区清单。输出目录必须尚不存在。

本工作区的纯离线命令如下；仓库位于其他目录时统一替换根路径。

```bash
python /workspace/scratch/562d14c1693a/carbon-research/research/literature_aligned_model_search_20261019d/provenance/reproduce_sciencedb_transfer.py \
  --repo-root /workspace/scratch/562d14c1693a/carbon-research \
  --raw-file /workspace/scratch/562d14c1693a/carbon-research/research/literature_aligned_model_search_20261019d/data/sciencedb_819_v1/raw/sciencedb_819_v6.xls \
  --out-dir /tmp/sciencedb-819-offline-reproduction
```

重新获取官方固定原件时，使用同一命令并省略 `--raw-file`。程序核对大小与 SHA256，来源不一致时停止，不静默使用新版本。成功后写 `REPRODUCTION_COMPLETE.json` 和 `prepared/DATA_MANIFEST.json`。两个脚本均不调用模型拟合或预测评分。

实际离线复现已通过：三个完整表与九个新分区 CSV 共 12 个文件逐字节重建一致，且九个旧格式分区哈希精确匹配。[PACKAGE_MANIFEST.json](provenance/PACKAGE_MANIFEST.json) 记录 `PASS_OFFLINE_REPRODUCTION_AND_PUBLIC_COPY`、`all_twelve_csvs_reproduced_exactly=true`、12 个逐文件比较，以及包装器测试没有联网。清单中的历史绝对路径保存原执行位置；对应文件按本包当前相对目录提供。

