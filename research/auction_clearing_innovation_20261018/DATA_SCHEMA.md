# EEX T3PA 事件级开发数据字段

本文件由 `parse_auction_events.py` 生成。只读 2012–2023 的 12 个年度原件；每一行是原表一条 T3PA 拍卖记录。保留取消、缺失价格和所有原表异常，不按成功状态筛选事件，不生成下一事件标签。

## 来源、平台与时间

来源为既有已批准 EEX 历史拍卖档案的开发年度成员；精确文件名、字节哈希、表头、列索引见 `DATA_QUALITY.json`。原件可定位于官方档案：https://www.eex.com/fileadmin/EEX/Downloads/Markets/Environmentals/EUA_Emission_Spot_Primary_Market_Auction_Report/Archive_Reports/emission-spot-primary-market-auction-report-2012-2025-data.zip 。本程序不下载或打开完整跨确认年份档案。

`platform` 统一为 CAP、DE、PL、NI：EU/TCAP/CAP2-EU/CAP3 EU → CAP；德国 → DE；波兰及 CAP2-PL/CAP3 PL → PL；CAP3 NIR → NI。`auction_name` 与 `country_raw` 保留原文，平台制度代际未被删去。

`date`、`auction_time` 分别来自原表日期与 Time/Auction Time。Excel 时钟原样解码；不据此认证 UTC、首次结果发布时刻或事前公告。相同日期多拍卖保留独立行；同一时钟值不虚构可观察先后。`same_time_event_count` 提醒需要同时事件处理。排序只是文件布局，不代表信息先后已认证。

## 数值字段

空字符串表示源列不存在、原单元为空或不是有限数值；由 absent_source_columns、missing_fields、nonnumeric_fields 区分，原始单元保存在私有 JSONL。数值由 Excel 缓存值解码为十进制文本，不插补、不平滑、不舍入价格、不更正异常。

| 字段 | 含义 |
|---|---|
| `clearing_price_eur_per_tco2` | 原表 Auction Price；EUR/tCO2，取消或缺失保留空。 |
| `supply_tco2` | 原表 Auction Volume，即记号 S；tCO2。实际供给/成交量的事件含义按原表与状态保留，不能自动当作事前公告量。 |
| `total_bid_volume_tco2` | 原表 Total Amount of Bids，即记号 B；总投标量，非投标笔数。 |
| `cover_ratio_reported` | 原表 Cover Ratio，保留原报告精度，不用 B/S 覆写。 |
| `bid_min_eur_per_tco2` | 原表 Minimum Bid/Minimal Price；EUR/tCO2。 |
| `bid_max_eur_per_tco2` | 原表 Maximum Bid/Maximal Price；EUR/tCO2。 |
| `bid_mean_eur_per_tco2` | 原表 Mean；EUR/tCO2。加权方式未认证，不认定数量加权。 |
| `bid_median_eur_per_tco2` | 原表 Median；EUR/tCO2。加权方式未认证，不认定数量加权。 |
| `bid_count` | Number of bids submitted；投标笔数；2012–2016 原列缺失。 |
| `bidder_count` | Total Number of Bidders；参与者人数，不等于投标笔数。 |
| `successful_bid_count` | Number of successful bids；成功投标笔数；2012–2016 原列缺失。 |
| `successful_bidder_count` | Number of Successful Bidders；成功参与者人数。 |
| `average_bids_per_bidder` | Average number of bids per bidder；2017 起提供。 |
| `average_bid_size_tco2` | Average bid size；投标数量，tCO2；2017 起提供。 |
| `average_bid_volume_per_bidder_tco2` | Average volume bid per bidder；tCO2；2017 起提供。 |
| `bid_volume_std_per_bidder_tco2` | Standard deviation of bid volume per bidder；数量标准差，tCO2，绝不是价格标准差。 |
| `average_volume_won_per_bidder_tco2` | Average volume won per bidder；tCO2；2017 起提供。 |
| `volume_won_std_per_bidder_tco2` | Standard deviation of volume won per bidder；数量标准差，tCO2，绝不是价格标准差。 |
| `revenue_eur` | 原表 Total Revenue；EUR，不用价格乘量覆写。 |

## 状态与标识

- `event_id` 是日期、时钟、Auction Name、Contract 的哈希标识，不是假定存在的官方 ID；`official_auction_id_raw` 在本批原表中缺列时为空。
- `source_status` 保留原文；2012–2019 没有 Status 列。`event_status` 区分 reported_successful、reported_cancelled_or_failed、legacy_price_observed_status_absent 等，不把早期推断说成官方成功标签。
- `status_column_present`、`price_observed` 是原表事后观测标志；`failure_markers_json`、`anomaly_flags` 保留取消/异常标记。取消场次价格空值不能填零。
- `completed_match_eligible`、`old_completion_evidence`、`old_component_id` 只用于与旧日表核对；不得用这些事后标志定义下一次可预测拍卖或筛选预报签发事件。
- `duplicate_event_identity`、`within_date_event_count`、`same_time_event_count` 是结构标记。发现重复不静默删除；来源文件、工作表、Excel 行号始终保留。
- `source_year/source_file/source_sheet/source_excel_row/source_header_excel_row/source_file_sha256/source_row_sha256` 给出精确定位。私有 `auction_event_source_rows.jsonl` 保存该来源行全部解码单元，包括取消说明、源国别分项、原文本与 Excel 错误。

## 信息边界与公开范围

本行清算价、实际投标量、覆盖率、投标价格统计、人数、笔数、实际结果状态与收入是本次拍卖结果信息，不能直接作为本次拍卖开始前的输入。事前供给须由独立日历版本按发布时点连接；这里不会从实际成功/取消结果回填已公告日历。研究模型需自行按已冻结截止规则构造过去已公布输入。

本解析器不把 bid-volume 标准差当价格标准差，不认证 Mean/Median 的加权方式，也不把未来成功标签作为下一事件的定义。没有价格分布统计、模型拟合或确认期读取。

公开交付：本脚本、DATA_SCHEMA.md、DATA_QUALITY.json。私有：local_only/ 下事件 CSV 与原始行 JSONL。质量收据只输出源身份、日期/计数、空值与状态及核对误差，不公开逐次价格/投标数值。

## 年份与行数

| 年份 | T3PA 事件行数（保留取消） | 平台 |
|---|---:|---|
| 2012 | 19 | {"CAP": 11, "DE": 8} |
| 2013 | 202 | {"CAP": 142, "PL": 14, "DE": 46} |
| 2014 | 194 | {"CAP": 144, "DE": 47, "PL": 3} |
| 2015 | 195 | {"CAP": 142, "DE": 47, "PL": 6} |
| 2016 | 187 | {"DE": 48, "CAP": 132, "PL": 7} |
| 2017 | 201 | {"DE": 45, "CAP": 137, "PL": 19} |
| 2018 | 203 | {"CAP": 139, "PL": 23, "DE": 41} |
| 2019 | 208 | {"CAP": 140, "DE": 44, "PL": 24} |
| 2020 | 210 | {"CAP": 140, "DE": 46, "PL": 24} |
| 2021 | 223 | {"CAP": 132, "DE": 45, "PL": 46} |
| 2022 | 215 | {"CAP": 143, "DE": 46, "PL": 24, "NI": 2} |
| 2023 | 215 | {"CAP": 143, "DE": 47, "PL": 24, "NI": 1} |

## 运行

```bash
python parse_auction_events.py --source-dir /absolute/path/eex_development_originals --output-dir /absolute/path/new_study --old-daily /absolute/path/EEX_EUA_T3PA_daily_development.csv
```

输出目录中允许存在其他研究文件；只写本解析器负责的四个交付文件及 local_only 的忽略规则。重写已有解析输出需显式 `--overwrite-owned-outputs`。
