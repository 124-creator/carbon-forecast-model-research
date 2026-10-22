# Round 13：独立公开日频数据获取与可用性

本轮取得 **KRBN ETF 日 OHLCV，703 个完整交易日，2024-01-02 至 2026-10-20**。这是可交易碳期货策略 ETF 的独立数据，不能命名为 EUA，也不能直接拼接到已有 EUA close 序列中。未取得可直接使用的 EUA 配额/单一 EUA 期货 OHLCV。

所有文件仅保存在本目录。未修改 legacy，未安装包，未登录或付费，未向第三方发送消息。独立公开来源共三个：Yahoo Finance、Stooq、KraneShares 发行人官网。

## 可直接使用的数据

- `KRBN_ETF_USD_daily_completed_20240102_20261020.csv`：推荐输入；703 行。
- 字段：`date,timestamp_utc,open,high,low,close,volume,adj_close`。
- `date`：美国纽约交易日期；`timestamp_utc`：Yahoo 原始 epoch 时间戳。
- 标的：KraneShares Global Carbon Strategy ETF，代码 KRBN，NYSEArca，价格单位 USD/ETF 份额，成交量单位为 ETF 份额。
- ETF 的敞口为多个碳配额市场的期货组合。发行人注明其基准为 S&P Global Carbon Credit Index；并非 EUA 现货、实物配额或单一 EUA 期货。
- `open/high/low/close/volume` 保留 Yahoo quote 字段原值，无人工重构。`adj_close` 为 Yahoo 独立提供的复权收盘字段；不能将它与原始 OHLC 混合计算蜡烛图、区间或缺口。返回的 2 笔分红记录另存于 `KRBN_corporate_actions.json`。

获取请求：

https://query1.finance.yahoo.com/v8/finance/chart/KRBN?period1=1704067200&period2=1792627200&interval=1d&events=div%2Csplits

抓取时间：**2026-10-21 14:11:14.667661 UTC**，即北京时间 22:11:14、纽约时间 10:11:14。当天美国市场尚未收盘。原始响应含 704 行、末行 2026-10-21 为盘中数据；该行仅保留在原始快照与 `yahoo_KRBN_daily.csv` 中，已从推荐 CSV 排除。

发行人身份来源：https://kraneshares.com/etf/krbn/ 。HTML 快照、抓取时间及 SHA-256 均已保留。

## 数据核验

完整日按年：2024 年 252 行；2025 年 250 行；2026 年截至 10-20 为 201 行。

- 703 行 OHLCV 均非缺失且为有限数值；日期严格递增，重复日期 0，周末记录 0。
- 703 行均满足 `low <= min(open,close) <= max(open,close) <= high`；价格和成交量非正值 0；高低价相等行 0。
- 成交量范围 2,300—885,700 份额/日。
- 495 行的 `adj_close` 与 `close` 不同，分红/复权口径应在研究中明确。
- 完整日 CSV SHA-256：`d1ff17cfb541809385d3ba6e918c61ce147682a91ea13bea93a426c33474d42a`。
- 未使用独立交易所日历核实是否漏失交易日，也未与第二行情供应商交叉校验，因此不声称已证明交易日历完整或交易所级精度。

详见 `KRBN_quality_report.json`。

## 其他标的与阻断

| 来源/标的 | 实际结果 | 判断 |
|---|---|---|
| Yahoo `CFI=F` | HTTP 404，正文 `No data found, symbol may be delisted` | 未能核实该代码的碳标的身份，未使用，不能把它当作已确认的 EUA 期货 |
| Yahoo `KEUA` | HTTP 404，正文相同 | 未取得日线；发行人页面明确其已停止交易 |
| Stooq `CK.F` | CSV 请求及身份页面均 HTTP 200，但正文为 796 字节的 JavaScript 浏览器验证页 | 未返回 CSV/报价；未尝试绕过验证，未把 HTTP 200 当作数据获取成功 |
| KraneShares 官方 `KEUA` | 发行人身份与清算公告 HTML 成功获取 | KEUA 跟踪 EUA 期货，为 ETF；公告称 2026-03-13 收盘后停止交易，清算日为 2026-03-20 |
| KraneShares 官方 `KRBN` | 身份 HTML 成功获取 | 确认为全球碳期货组合 ETF，不能改名为 EUA |

Stooq 的公开搜索摘要将 `CK.F` 标注为 ICE CO2 EU Allowances，但直接访问已被浏览器校验页阻断，合约细节、连续合约换月口径和日线数值均未进一步验证。

相关 URL：

- https://query1.finance.yahoo.com/v8/finance/chart/CFI%3DF?period1=1704067200&period2=1792627200&interval=1d&events=div%2Csplits
- https://query1.finance.yahoo.com/v8/finance/chart/KEUA?period1=1704067200&period2=1792627200&interval=1d&events=div%2Csplits
- https://stooq.com/q/d/l/?s=ck.f&d1=20240101&d2=20261021&i=d
- https://stooq.com/q/?s=ck.f
- https://kraneshares.com/etf/keua/

KEUA 清算公告位于本地 `kraneshares_KEUA_identity_raw.txt` 第 3119—3121 行；这比仅凭 Yahoo 404 推断退市更有证据。

## 归档文件与复现

- `acquisition_log.json`：逐请求 URL、UTC 抓取时间、HTTP 状态、原始响应路径、响应头及 SHA-256；Yahoo instrument 元数据也在此保留。
- `instrument_and_availability_manifest.json`：标的身份、单位、价格口径、ETF/EUA 区别、可用性与阻断结论。
- `KRBN_quality_report.json`：数据质量、覆盖范围、剔除的盘中行、哈希。
- `KRBN_corporate_actions.json`：Yahoo 返回的 2 笔分红原始记录。
- `yahoo_KRBN_raw.json`：Yahoo 完整原始快照；`yahoo_KRBN_daily.csv` 为 704 行原样解析版本，包含盘中末行，请勿直接作为完整日实验输入。
- `yahoo_CFI_F_error.txt`、`yahoo_KEUA_error.txt`：HTTP 404 的原始 JSON 正文。
- `stooq_CK_F_raw.txt`、`stooq_CK_F_identity_raw.txt`：浏览器验证阻断响应。
- `kraneshares_KRBN_identity_raw.txt`、`kraneshares_KEUA_identity_raw.txt`：发行人原始 HTML。
- `acquire_public_daily.py`、`finalize_availability.py`：使用 Python 标准库的抓取与整理脚本，无新增包。

本目录支持将 KRBN 单独报告为 2024 年以来的碳相关 ETF OHLCV 稳健性/识别实验。它补充价格区间与 ETF 成交量维度，但不解决 EUA 本身仍只有收盘价的事实；ETF 的资产组合、美元汇率、分红、费用及市场交易时段也与 EUA 不同。
