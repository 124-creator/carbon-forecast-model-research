# 三个额外碳资产产品核验 — Round 17

结论：**GRN 可保留为有覆盖缺口与流动性限制的额外产品候选；CARB.TO 已终止，XCO2.AX 当前行情不可获取，均不纳入持续前瞻输入。**本轮最多核验这三个新 instrument；另取得 KRBN/KCCA 的 2026-10-21 已完成 bar，分别另存新文件。没有读取模型结果或生成回报、2026 标签、预测、损失、评分。

全部新增文件位于本目录。没有触碰 legacy 或覆盖原有数据，也没有重试 China CRC 路线。遇到 SEC 和 VanEck origin 的直接 HTTP403 后，对应路线均停止，未换 UA、代理、浏览器或申请 API key；未安装任何包。

## 1. 身份与生命周期

| 产品 | 身份/币种/敞口证据 | 起始与生命周期 | 本轮处置 |
|---|---|---|---|
| GRN | Barclays Bank PLC 的 iPath Series B Carbon ETN；USD；2019 年监管文书列 Barclays Global Carbon II TR USD Index，EU ETS/CDM 期货，当时 EU 权重>99.9% | 首售/法律 inception 2019-09-09，首次发行 09-11；发行人发布稿索引称预期 09-10 在 NYSE Arca 交易。CUSIP06747C322、ISINUS06747C3227 | Yahoo 返回对应 Series B 名称及 2019-09-18 起历史；保留 available-window CSV，不称完整成立期历史 |
| CARB.TO | Global X Carbon Credits ETF，原 Horizons Carbon Credits ETF；管理人 Global X Investments Canada Inc.；CAD；Horizons Carbon Credits Rolling Futures Index (Excess Return)，非 CAD 敞口对 CAD 对冲 | 官方 ETF Facts：2022-02-10 开始；2025-03-18 收盘摘牌，2025-03-24 终止 | 官方身份与终止已证，Yahoo404；不造历史 CSV，不作持续前瞻产品 |
| XCO2.AX | VanEck Global Carbon Credits ETF (Synthetic)；VanEck Investments Limited；官方旧 snapshot 搜索索引列 AUD、2022-10-11；官方博客列 ICE Global Carbon Futures Index，EU/California/RGGI/UK 四市场 | 原 snapshot 当前直接403；公开 ASX 公告镜像索引显示2025-03-28终止通知、最后交易日2025-04-28，但尚未取得官方交易所 PDF 原件完成此日期的直接认证 | Yahoo404；不造 CSV，不纳入当前前瞻输入；身份部分原件/生命周期证据仍有限 |

GRN 的官方类型是 **ETN**，不是基金 ETF。Yahoo 将 instrumentType 标成 ETF，这个供应商分类差异已保留并在 manifest 中更正身份解释；没有篡改原始响应。GRN 有 Barclays 的无担保发行人信用与 bail-in 风险，市场价格与 indicative value 不同，不能直接称 EUA 现货或期货合约价格。

## 2. 官方证据与访问记录

- [GRN 的 SEC 定价说明书](<https://www.sec.gov/Archives/edgar/data/312070/000110465919049393/a19-18373_31424b2.htm>) 经 web_fetch200 可读。它明确首售/发行日期、CUSIP/ISIN、USD、发行人和当时指数构成。本地直接获取 HTML 返回403，已停止；[本地身份摘录](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/data_acquisition_round13/GRN_regulatory_identity_round17_webfetch_excerpt.json>)明确是工具抽取证据，不假称 SEC 原始 HTML。
- [GRN 发行人 launch PDF](<https://barxis.barcap.com/file.app?action=shared&path=iPath/US/Press/GRN%20Launch%20Press%20Release.pdf>) 的搜索索引列2019-09-10及预期首个交易日，直接 URL 当前404。没有拿2008旧同名 GRN 拼接或补齐历史。
- [CARB 官方 ETF Facts](<https://www.globalx.ca/wp-content/uploads/2025/01/CARB_FundSummary.pdf>) 返回200且具有 PDF 文件头。已保存[原始 PDF](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/data_acquisition_round13/CARB_TO_issuer_facts_round17_raw.pdf>)，使用现有 pdftotext 提取并阅读[文本](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/data_acquisition_round13/CARB_TO_issuer_facts_round17_text.txt>)，第19–37行给出开始日期、CAD、指数和货币对冲，第10–11行给出预计终止安排。
- [Global X 最终终止公告](<https://www.globalx.ca/news/press-releases/global-x-investments-canada-inc-announces-final-valuation-for-the-terminating-etfs>) 的工具与直接获取均200，明确 CARB 于2025-03-18摘牌、03-24终止，最终NAV7.11651356 CAD。[原始 HTML](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/data_acquisition_round13/CARB_TO_issuer_termination_round17_raw.html>)已保留。
- [VanEck 官方 XCO2 博客](<https://www.vaneck.com.au/blog/carbon-credits/carbon-markets-explained-for-investors/>)工具200可读，发布日期2024-04-26，说明合成 ETF 和四市场碳期货敞口；它不能证明2026仍在存续。官方旧 snapshot 的日期/币种仅由搜索索引取得，当前 origin 直接403后停止。
- [XCO2 facts URL](<https://www.vaneck.com.au/library/vaneck-etfs/XCO2-fact-sheet.pdf>)虽然返回200，却重定向发行人首页并返回 HTML。其响应字节原样保存，但 **拒绝作为 PDF 或基金身份文件使用**；完整 redirect、SHA 和格式失败在 manifest 中列出。
- [XCO2 ASX 终止公告镜像](<https://www.marketindex.com.au/asx/xco2/announcements/notice-of-termination-2A1587424>)检索索引列最后交易日2025-04-28；本地保存了该网页 HTML，尚未取得公告 PDF 本体。不能把镜像 HTML 或检索片段当官方交易所原始公告。

## 3. GRN 可见历史与质量

Yahoo 使用既有 query1/chart 公开请求方式，要求从实际 Series B 首售日2019-09-09开始，到 exclusive period2=2026-10-22T00:00Z。实际返回 **2019-09-18—2026-10-21，共1,783行**；供应商 firstTradeDate 也为2019-09-18。预期挂牌日09-10至首 bar 之前有6个未表示的工作日候选（09-10/11/12/13/16/17）；未独立验证交易所日历，也不能断言这些天一定有遗漏成交。

因此**完整成立期覆盖门未通过**。本轮保存的是明确命名的可见窗口；不补前期行、不导入旧 GRN、不伪造 OHLC，不把它升级为 inception-complete 的锁定数据集。CSV 保留原始 quote O/H/L/C/V，adj_close 单列；没有手工前后复权。Yahoo 返回2021-06-04的拆分事件 numerator=1、denominator=5，仅原样归档，尚未独立核对发行人拆分规则和历史 quote 调整口径。

| 可见数据段 | 行数 | 缺失/非有限 | 非正价格 | 包络异常 | 零 range | 零量 | 两者交叉 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2019-09-18—2025-12-31 | 1,581 | 0 | 0 | 0 | 342 | 102 | 102 |
| 2026-01-02—2026-10-21，仅原始质量 | 202 | 0 | 0 | 0 | 69 | 0 | 0 |
| 全部响应的质量元数据 | 1,783 | 0 | 0 | 0 | 411 | 102 | 102 |

日期严格递增、无重复、无周末行，volume 无负值。2019年73行中65天零range、38天零量，2020年253行中214天零range、64天零量；均原样保留。2026年仅作为质量元数据，其 volume 中位数为300个ETN单位，不能因此当作高流动性资产，也未据此构造任何目标或评分。

- [GRN 2025年底以前可见历史](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/data_acquisition_round13/GRN_ETN_USD_available_history_20190918_20251231_round17.csv>)：SHA256 `52cba5fd28beb71995ce4b0092cd7b0b41ece97aa1b28c0d369d90cd04c57e9b`。
- [GRN 2026原始质量隔离数据](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/data_acquisition_round13/GRN_ETN_USD_2026_raw_quality_only_20260102_20261021_round17.csv>)：SHA256 `d15ea14503a6362512b697a0b3ab00fc1375ecc74d8149bcc30552b2affff821`。
- [GRN 完整质量报告](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/data_acquisition_round13/GRN_available_window_quality_round17.json>)包含逐年质量、异常/零range/零量日期、覆盖缺口与调整边界。

CARB.TO、XCO2.AX 的 Yahoo 响应均为404，内容为 `No data found, symbol may be delisted`；这是本次供应商路线的结果，不表示过去不存在该产品，也不授权拼接其他 ticker 或伪造历史。

## 4. 2026-10-21 已完成的两 ETF 输入

KRBN、KCCA 的请求分别在2026-10-21T21:08:15Z、21:08:16Z取得，即 New York 当日17:08，晚于已明确的16:00常规收盘。只保存该日新增 bar；原2026-10-20冻结 CSV 保持不变，前后 SHA 已核对。

| 产品 | O | H | L | C | Volume |
|---|---:|---:|---:|---:|---:|
| KRBN | 33.99000168 | 34.08000183 | 33.88999939 | 34.00999832 | 13,395 |
| KCCA | 17.02000046 | 17.06999969 | 17.01000023 | 17.01000023 | 10,450 |

两行均正价、字段完整、OHLC包络有效、range非零；原始浮点值全部保留，表格仅作显示。这里只确认已完成输入的质量，不把已发生行情称为前瞻预测。

- [KRBN 新 bar](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/data_acquisition_round13/KRBN_new_completed_bar_20261021_round17.csv>)，SHA256 `dd94ed9d77c401476dcb11cd9f90fbd933b2b26b56517cb335bb6c3576093d44`。
- [KCCA 新 bar](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/data_acquisition_round13/KCCA_new_completed_bar_20261021_round17.csv>)，SHA256 `f7af649709126b3658eeb1d8f938b663d73821a563fac035f8e9496f9fc6e1b2`。
- [输入 manifest](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/data_acquisition_round13/KRBN_KCCA_latest_complete_inputs_20261021_round17_manifest.json>)记录请求URL、UTC抓取时间、完成规则、原始响应SHA、bar质量及冻结文件前后SHA。

## 5. 验证独立性与前瞻边界

这些是额外的**交易产品**，不能直接算新增独立宏观市场。GRN 的原始条款高度偏重 EUA，CARB 的期货和对冲组合包含 EUA 风险，XCO2 的四市场组合与 KRBN 重叠且包含 KCCA 的 California/WCI 风险。USD/CAD/AUD计价、基金/ETN结构及汇率对冲不同，会增加产品与微观结构差异；它们不消除底层风险重叠，也不能代替独立经济市场验证。

未来若另行授权前瞻 logging，需要在真实预测起点记录数据 as-of、模型/配置版本与实际产生的预测，再等待未来观察发生；本轮没有产生预测，不能回填预测时间。没有设置每天自动运行或提醒任务。

总的文件入口是 [三产品 manifest](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/data_acquisition_round13/additional_carbon_products_round17_manifest.json>)，SHA256 `01d56bc680839b9672901fbf7c4f93a585377ea996700560677521bae54be52d`。本轮仅完成数据与身份核验，未改变研究路线或运行模型。
