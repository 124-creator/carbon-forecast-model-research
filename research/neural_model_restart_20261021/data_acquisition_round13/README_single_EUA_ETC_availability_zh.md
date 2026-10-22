# 单一 EUA 敞口 ETC：身份与 OHLCV 可用性核验

本轮严格只请求三个 ticker：CO2.L、CO2P.L、CARB.L，以及对应发行人身份资料。三个 Yahoo chart 请求均成功，原始响应、按完整日解析的原值 CSV、公司行动和质量报告均已保存。**没有任何候选通过本轮 OHLC 结构可用性核验**：上市后和 2024 年以来仍有大量收盘价超出当日 high/low 的记录，不能直接用于需要真实蜡烛几何结构的实验。

这不等同于证明全部价格无效；有可能涉及供应商对成交高低价与收盘报价的不同口径，但本轮没有取得能解释或协调这些字段的权威定义。因此未人工将 high/low 改成包住 open/close，未填补，也未删除异常日来生成选择过的训练样本。

## 标的身份与单位

| Yahoo ticker | 身份 | 报价单位 | 结论 |
|---|---|---|---|
| CO2.L | SparkChange Physical Carbon EUA ETC；ISIN XS2353177293；实物 EUA 支持的债务型 ETC | EUR/ETC 证券 | 发行人身份已核实；Yahoo OHLC 结构不通过 |
| CO2P.L | 同上，同一 ISIN 的英镑交易线 | Yahoo 为 GBp，即英镑便士/ETC 证券 | 身份已核实；不是第二个独立资产；Yahoo OHLC 结构不通过 |
| CARB.L | 候选 WisdomTree Carbon，当前官方 factsheet 搜索索引显示 ISIN JE00BP2PWW32、EUA 期货指数敞口及 swap 结构 | Yahoo 为 USD/ETC 证券 | 完整权威资料读取受阻，历史接续未核实，保持候选、不使用 |

GBp 转 GBP 时，所有价格字段（包括 adj_close）除以 100，成交量不变。本轮原始 CSV 保持 Yahoo GBp 数字，**没有做单位转换**。CO2.L 原值为欧元，不需要除以 100；CARB.L 原值为美元。

三个 Yahoo `instrumentType` 都写 EQUITY，这是供应商通用分类，不能据此称其为普通股票或 ETF；实物产品法律结构由发行人确认为 ETC。

CO2.L/CO2P.L 的发行人页面明确：基础币种 EUR、实物复制、100% EUA 支持（扣除费用影响）、发行人为 HANetf ETC Securities Plc。2026-10-20 的每证券 EUA entitlement 为 0.9566130510；所以 ETC 份额价格也不能等同为每吨 EUA 现货价格。成交量为 ETC 证券数量，不能解释成 EUA 吨数或 EUA 期货手数。

成功读取的发行人来源：

https://hanetf.com/fund/co2-sparkchange-physical-carbon-eua-etc/

该页面通过本会话 `functions.web_fetch` 返回 HTTP200 文本，关键原文摘录保存在 `HANetf_CO2_issuer_verified_webfetch_excerpt.json`。稍后的 Python 直接 GET 返回 HTTP403，原始错误正文也已保存；没有尝试绕过浏览器验证。网页工具未提供精确请求时间，因此摘录档分别说明工具来源、时间范围和实际归档时间，未伪造精确抓取秒数。

## 实际数据与 SHA-256

| CSV | 原样解析行数 | 实际日期范围 |
|---|---:|---|
| CO2_L_ETC_daily_completed_20211018_20261020.csv | 1,264 | 2021-10-18—2026-10-20 |
| CO2P_L_ETC_daily_completed_20211018_20261020.csv | 1,264 | 2021-10-18—2026-10-20 |
| CARB_L_ETC_daily_completed_20200102_20261020.csv | 1,718 | 2020-01-02—2026-10-20 |

对应 CSV SHA-256，顺序同上：

```text
CO2.L  8feecd0706b0f10dfa3e9dfa2569df816fa5b68b592cc92e2c50f41e9ff72160
CO2P.L 52b775b972630d96c465b8321a7e1a8997122e5d6ca00c397bbb379faa4c9fdb
CARB.L ca3b402ed1937cf0e8e3096774072258052a2f69a6868188c7a26b85e9f721e0
```

CSV 字段为 `date,timestamp_utc,open,high,low,close,volume,adj_close`，日期按 Europe/London 交易所时区生成。请求起点为 2020-01-01，截止 `period2=2026-10-21T00:00:00Z`（不包含），再按交易日期不晚于 2026-10-20 且早于抓取时交易所本地日期筛选。没有 2026-10-21 盘中行。抓取时间为 2026-10-21 14:38:07 UTC 左右，逐请求精确时间见日志。

Yahoo URL 形式（ticker 只取上表三个）：

`https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1=1577836800&period2=1792540800&interval=1d&events=div%2Csplits%2CcapitalGains`

## 上市日期与异常

发行人明确 CO2.L 与 CO2P.L 的 LSE 上市日期为 **2021-11-04**，而 Yahoo 从基金 inception date 2021-10-18 就给出了记录。前 13 行全部零成交量，不能当作上市交易 OHLCV。去掉这 13 行也不能解决后续结构问题。

| 范围/标的 | 行数 | 至少一个字段缺失 | OHLC 范围不一致 | 零成交量 | high=low |
|---|---:|---:|---:|---:|---:|
| CO2.L 上市后 | 1,251 | 0 | 478 | 66 | 128 |
| CO2.L 2024 年起 | 710 | 0 | 255 | 43 | 60 |
| CO2P.L 上市后 | 1,251 | 1 | 581 | 226 | 402 |
| CO2P.L 2024 年起 | 710 | 1 | 327 | 123 | 195 |
| CARB.L 2021-08-27 起（候选分段） | 1,299 | 2 | 556 | 395 | 597 |
| CARB.L 2024 年起（候选） | 710 | 2 | 280 | 273 | 379 |

OHLC 范围要求 `low <= min(open,close) <= max(open,close) <= high`。CO2.L 的 478 个问题均为 close 超出 high/low，且 472 个偏离大于当日 close 的 1 个基点，不能归因于单纯浮点微差。例如 2024-01-03：O=72.69、H=72.80、L=72.69、C=73.32。

Yahoo 对三个候选均返回空的公司行动列表，且有效行 adj_close 与 close 相同。这只代表本次响应没有返回事件，不是对历史上不存在公司行动的独立证明。

CARB.L 当前官方 factsheet 的公开搜索索引给出 LSE USD 交易线及 2021-08-27 上市日期：

https://www.wisdomtree.eu/de-de/-/media/eu-media-files/key-documents/factsheet/etf-securities/eng-carb-ln-wisdomtree-carbon.pdf

官方产品页直接请求 HTTP403，文档网页抓取工具失败。Yahoo 却给出 2020 年以来历史，未能核实其是否含此前产品、旧 ISIN 或重上市前记录；不能将所有历史直接认定为当前同一 EUA 期货 ETC。

## 归档入口

- `EUA_ETC_candidates_acquisition_log.json`：每个请求 URL、时间、HTTP 状态、原始响应哈希、Yahoo 元数据和日线哈希。
- `EUA_ETC_identity_units_segmented_quality.json`：最终身份/单位、上市前与上市后/2024 年起分段质量、例子及不使用结论。
- `HANetf_CO2_issuer_verified_webfetch_excerpt.json`：已成功读取的发行人身份原文摘录与来源说明。
- 各自的 `yahoo_*_etc_raw.json`、`*_ETC_quality_report.json`、`*_ETC_corporate_actions.json`：原始响应、全样本质量和公司行动列表。
- `acquire_single_EUA_ETC_candidates.py`、`assess_single_EUA_ETC_quality.py`：标准库获取与核验脚本。

本轮没有获取更多 ticker、安装包、登录付费、发送第三方消息或绕过验证。所有输出均为指定目录的新文件，没有覆盖 KRBN 文件，没有读取任何模型测试标签或评分。**本轮的实质结果是核实了更接近单一 EUA 的实物 ETC 身份，但公开 Yahoo 数据未满足所需 OHLC 结构质量。**
