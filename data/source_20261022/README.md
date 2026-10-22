# 实验输入数据（2026-10-22 归档）

本目录补齐此前只存在于本机、未随仓库分发的**原始输入数据**。所有实验脚本读取的是这些文件或由它们派生的中间表，派生表在各 `research/*/` 目录内。

逐文件大小与 SHA256 见同目录 `MANIFEST_SHA256.csv`（45 个文件，合计约 9.9 MB）。

## 目录说明

### `yahoo_ohlc/`
三只碳 ETF 与三只能源/碳相关合约的日频 OHLC（含调整收盘价），是全部波动率实验的**真实输入**。

| 文件 | 行数（约） | 覆盖 |
|---|---|---|
| `KRBN_ohlc.csv` | 1,470 | 2020-07-31 → 2026-06-08，KraneShares Global Carbon ETF |
| `KEUA_ohlc.csv` | 1,122 | → 2026-03-20，KraneShares European Carbon Allowance ETF（该日之后停止更新） |
| `KCCA_ohlc.csv` | 1,173 | → 2026-06-08，KraneShares California Carbon Allowance ETF |
| `ECF=F_ohlc.csv` | 极少行 | 几乎为空，不作为可用序列 |
| `CL=F_ohlc.csv` / `NG=F_ohlc.csv` | 数千 | WTI 原油、Henry Hub 天然气，仅作对照 |

下载来源：Yahoo Finance 公开行情。**Yahoo 条款限制再分发**，此处仅为结果复核与研究审阅保留；若要公开发表或再分发，请自行替换为持牌数据源。

### `expanded_proxy/`
EUA 欧洲交易时段代理标的的日频序列（Xetra 上市的 EUA ETC `DE000SL0E4X3.SG`、韩国 `400570.KS`），以及各自抓取时的 Yahoo chart JSON 原件与质量汇总表。用于检验"是否存在比美股 ETF 更贴近 EUA 的可得代理"，结论见对应研究目录；本轮未作为主模型输入。

### `eex_auction/`
EEX 公开的 EU 排放配额一级市场拍卖报告：

- `eua_auction_2012_2025.zip`（SHA256 `c3ff6ffaaf348cbd17f61d4726fdbd45a7ead089a0006568a490dd63bb080231`）为官方打包下载件
- 解包后的 2012–2019 `.xls` 与 2020–2026 `.xlsx` 逐年文件
- 2026 年度文件另有一个独立下载版本（`emission-spot-primary-market-auction-report-2026-data.xlsx`）

原始下载地址：

- `https://www.eex.com/fileadmin/EEX/Downloads/Markets/Environmentals/EUA_Emission_Spot_Primary_Market_Auction_Report/Archive_Reports/emission-spot-primary-market-auction-report-2012-2025-data.zip`
- `https://public.eex-group.com/eex/eua-auction-report/emission-spot-primary-market-auction-report-2026-data.xlsx`

**注意**：2012–2019 为 `.xls`，当前环境未安装 `xlrd`，因此这部分**从未被解析**，也不在任何实验样本内。审计结论见 `research/eua_auction_20261022/audit_report.md`。

### `eex_calendar/`
EEX 拍卖日历（2024 年两个修订版本，`.pdf` 与 `.xlsx`）、EEX Group DataSource REST API 用户指南、历史页截图、2026 年 54e6fb64 快照。用于核验"拍卖日程在事前是否可得"，是判断 `days_to_next_auction` 能否使用的依据。

### `dehst/`
德国联邦环境署（DEHSt）2024 年报 PDF 与下载收据，仅作市场制度背景，未进入任何模型。

## 许可与再分发状态（必读）

| 来源 | 状态 |
|---|---|
| Yahoo Finance 行情 | 条款限制再分发；本仓库仅作研究复核留存 |
| EEX 拍卖报告与日历 | 报告公开可下载；**再分发与商用条款未确认** |
| DEHSt 年报 | 公开发布，署名保留 |
| 中国配额（GDEA/HBEA/CEA）原始响应 | **不在本目录**，此前已按许可未确认排除；仅有派生面板 `research/fullcycle_h20_20261020/PANEL_4MARKET.csv` |

本目录按项目所有者要求整体随仓库公开，用于让接手方直接复核。若任一权利方提出异议，应立即删除对应子目录，仓库中不保留其副本；实验结论不依赖这些文件的存在与否，`MANIFEST_SHA256.csv` 可证明被删除的是哪一份。
