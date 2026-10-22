# KCCA：单一 CCA 期货策略 ETF 日 OHLCV

本轮仅获取 **KCCA**，成功取得 **2021-10-05—2026-10-20，共 1,266 个完整交易日**。数据通过基础 OHLCV 质量检查，可作为独立列示的不同碳市场 ETF 外部实验对象。这里的“独立列示”仅指不同证券/数据文件；**KCCA 与 KRBN 可能共有 CCA 底层敞口，不能声称二者是独立风险来源或统计独立证据**。

推荐 CSV：`KCCA_ETF_USD_daily_completed_20211005_20261020.csv`。

SHA-256：`17c93aeca744b91b041a419413fae105c80aca8c9b214c625c7ab6b47d2cae68`。

## 标的与来源

发行人官网 https://kraneshares.com/etf/kcca/ 明确：

- 官方名称：KraneShares California Carbon Allowance Strategy ETF。
- 提供 California Carbon Allowance（CCA）期货敞口；正文基准为 S&P Carbon Credit CCA Excess Return Index，追踪最活跃的 CCA 期货。官网资料表将指数简写为 S&P Carbon Credit CCA Index。
- 官网当前期货持仓示例为 California Carbon Allowance (CCA) Vintage 2026 Future、KBCZ26 Comdty，约 100.03% NAV。当前持仓仅用于核对期货敞口身份，不代表整个历史始终持有同一个合约。
- 发行人成立日期是 2021-10-04；Yahoo 首个交易日期为 2021-10-05，实际数据从后者开始，不伪造前一日记录。

Yahoo 元数据为 ETF、NYSEArca、USD、America/New_York。价格单位是 **USD/ETF 份额**，成交量是 **ETF 份额**。这不是 CCA 现货、每吨实物配额价，也不是直接上市的单一期货合约价格，更不能命名为 EUA。

本次 Yahoo 请求：

https://query1.finance.yahoo.com/v8/finance/chart/KCCA?period1=1577836800&period2=1792540800&interval=1d&events=div%2Csplits%2CcapitalGains

- 抓取时间：2026-10-21T14:53:18.603787+00:00，即北京时间 22:53:18、纽约时间 10:53:18。
- 起点请求为 2020-01-01，实际首行是 2021-10-05，与 Yahoo firstTradeDate 一致。
- period2 为 2026-10-21 00:00 UTC（不包含），另按纽约交易日期不晚于 2026-10-20 且早于抓取当天筛选，没有 2026-10-21 的盘中行。
- 官网身份 HTML 直接抓取 HTTP200，时间为 2026-10-21T14:53:18.606363+00:00；原始页面与哈希均已保留。此前网页抓取工具一次超时，没有进行验证绕过。

## 数据口径与公司行动

字段为 `date,timestamp_utc,open,high,low,close,volume,adj_close`。日期按纽约时区转换，时间戳保留 Yahoo epoch。OHLCV 使用原始 quote 字段，不修补或人工重构；`adj_close` 是独立复权字段，不能直接混用未复权 OHLC 计算蜡烛几何结构。

返回的 6 笔 Yahoo `dividends` 已完整保存在 `KCCA_corporate_actions.json`，含事件时间戳、本地日期和金额。较大一笔为 **2024-06-27：5.028 USD/份额**；原 close 的跨日变动会包含分配影响，不能解释为投资总回报。共有 1,058 行 adj_close 与 close 不同。本次未计算任何收益、预测目标或模型评分。

官网说明分配可来源于资本利得、股息及 PFIC 相关因素，因此 Yahoo 的 dividends 是供应商事件标签，不能据此将所有款项解释为普通股息。响应未返回 split 或 capitalGains 记录，不等同于独立证明历史不存在其他事件。

## 质量

| 年份 | 完整日行数 |
|---|---:|
| 2021（10-05 起） | 62 |
| 2022 | 251 |
| 2023 | 250 |
| 2024 | 252 |
| 2025 | 250 |
| 2026（截至 10-20） | 201 |
| 合计 | 1,266 |

全样本检查：

- 所有字段缺失/非有限值 0；重复日期 0；周末记录 0；日期严格递增。
- 全部满足 `low <= min(open,close) <= max(open,close) <= high`；OHLC 范围违规 0。
- 非正价格、零/负成交量均为 0。日成交量范围 300—4,972,600 份额。
- 有 1 个 `high == low` 的有效交易日：2023-05-11。保留真实零区间，没有人工放大高低价或删除该日。
- 2024 年起有 703 行，无缺失、OHLC 范围违规、非正价格或成交量、零高低价区间；该分段仅为数据质量核验，不是任何模型测试结果。

未使用独立交易所日历或第二行情商核实，不能声称已证明交易日历无漏日或交易所级行情精度。通过这些基础检查不代表有预测收益、因果识别能力或与 KRBN 独立。

## 本轮新文件

- `KCCA_ETF_USD_daily_completed_20211005_20261020.csv`：完整日数据。
- `yahoo_KCCA_since2020_raw.json`：Yahoo 完整原始响应。
- `kraneshares_KCCA_identity_raw.html`：发行人原始身份 HTML。
- `KCCA_acquisition_log.json`：逐请求 URL、精确 UTC 抓取时间、HTTP 状态、响应头、原始与 CSV 哈希及 Yahoo 元数据。
- `KCCA_corporate_actions.json`：6 笔分配事件原始记录。
- `KCCA_quality_report.json`：全样本及 2024 年起质量、覆盖与零区间日期。
- `KCCA_instrument_and_availability_manifest.json`：币种、单位、期货 ETF 身份、官网证据位置及敞口重叠限制。
- `acquire_KCCA_since2020.py`：Python 标准库抓取与检查脚本，排他新建模式。

所有文件均为指定目录中的独立新文件，未覆盖既有 KRBN 或单一 EUA 候选文件。本轮只尝试 KCCA，没有新增 ticker，没有登录、付费、安装包或绕过验证，没有读取或计算任何模型测试目标/评分。
