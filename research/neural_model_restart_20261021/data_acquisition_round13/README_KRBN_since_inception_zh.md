# KRBN 成立以来扩展：独立公开日频数据

本次沿用同一 Yahoo Finance 来源，请求从 2020-01-01 起的数据，实际返回 **2020-07-31—2026-10-20，共 1,563 个完整交易日**。首行与 Yahoo 元数据 `firstTradeDate`（2020-07-31 09:30 纽约时间）一致；这里将其称为“首个可得交易日”，不把基金成立日期与上市交易日期混同。

推荐文件：`KRBN_ETF_USD_daily_completed_20200731_20261020.csv`。

SHA-256：`ff99c14e84153717be0d72506b0c8b581605495288849823e4bf219ab1462f11`。

## 来源、身份与完整日筛选

- 实际请求：https://query1.finance.yahoo.com/v8/finance/chart/KRBN?period1=1577836800&period2=1792540800&interval=1d&events=div%2Csplits%2CcapitalGains
- 抓取时间：2026-10-21T14:17:13.361232+00:00，即北京时间 22:17:13，纽约时间 10:17:13。
- 请求的 `period2` 为 2026-10-21 00:00 UTC（不包含该时刻），并另以纽约交易日期 `<=2026-10-20` 且早于抓取时纽约日期筛选。返回的 1,563 行全为截止日前的数据，没有 2026-10-21 盘中行。
- KRBN 为 KraneShares Global Carbon Strategy ETF，NYSEArca，USD/ETF 份额，成交量单位为 ETF 份额；追踪多个碳配额市场的期货组合，基准为 S&P Global Carbon Credit Index。不能命名为 EUA 配额、EUA 现货或单一 EUA 期货。
- 身份复用已抓取的发行人官网 https://kraneshares.com/etf/krbn/ 及本目录 `kraneshares_KRBN_identity_raw.txt`，原快照抓取时间为 2026-10-21T14:11:14.669324+00:00，SHA-256 已重新核验。

## 字段与公司行动

CSV 字段：`date,timestamp_utc,open,high,low,close,volume,adj_close`。日期按 America/New_York 转换，时间戳保留 Yahoo 原始 epoch。

OHLCV 保留原始 quote 字段，不进行人工重构。`adj_close` 为 Yahoo 独立提供的复权收盘字段，不能直接与未复权 OHLC 混用。原始收盘跨分红日的变化包含分红影响，若研究目标是投资总回报，需明确采用相应复权/总回报口径。

`KRBN_since2020_corporate_actions.json` 保留返回的 6 笔 `dividends`，含 Yahoo epoch 时间、纽约日期及金额；未返回 split 或 capitalGains 事件。较大一笔为 2022-12-28 的 8.355 USD/份额（Yahoo 事件标签为 dividends），应在解释原始收盘收益时注意该公司行动。

## 质量与覆盖

| 年份 | 完整日行数 |
|---|---:|
| 2020（07-31 起） | 107 |
| 2021 | 252 |
| 2022 | 251 |
| 2023 | 250 |
| 2024 | 252 |
| 2025 | 250 |
| 2026（截至 10-20） | 201 |
| 合计 | 1,563 |

- 所有字段缺失/非有限数值为 0，重复日期 0，周末日期 0，日期严格递增。
- OHLC 顺序违规 0，非正价格/非正成交量为 0。
- 有 4 个 `high == low` 的交易日，保留原值；计算区间或其变换时需考虑零区间。
- 日成交量范围为 100—4,304,000 份额。
- `adj_close` 与 `close` 不同的行数为 1,355。
- 未使用独立交易所日历或第二行情商核验，不声称已经证明无漏失交易日。

相比原 703 行快照，新增 **2024 年以前 860 行**。重叠的 703 个日期，open/high/low/close/volume/adj_close 各字段逐值完全一致；旧文件的 SHA-256 在执行前后均为 `d1ff17cfb541809385d3ba6e918c61ce147682a91ea13bea93a426c33474d42a`。

## 新文件

- `KRBN_ETF_USD_daily_completed_20200731_20261020.csv`：完整日数据。
- `yahoo_KRBN_since2020_raw.json`：完整原始响应。
- `KRBN_since2020_acquisition_log.json`：URL、抓取时间、响应头、HTTP 状态、原始与 CSV 哈希、instrument 元数据。
- `KRBN_since2020_instrument_manifest.json`：标的口径与复用发行人身份快照的验证。
- `KRBN_since2020_corporate_actions.json`：6 笔分红。
- `KRBN_since2020_quality_report.json`：覆盖、质量及旧快照一致性核验。
- `acquire_KRBN_since2020.py`：Python 标准库复现脚本；使用排他新建模式，避免覆盖既有文件。

本次没有新增来源组、安装包、登录付费或绕过验证；仅在指定归档目录创建新文件，未修改此前交付的 703 行数据及其说明，未读取任何模型测试结果。
