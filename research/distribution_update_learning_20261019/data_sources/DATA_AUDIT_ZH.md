# 新一轮碳价格数据：可用性、暴露边界与确认候选

审计日期：2026-10-19 UTC。交付范围：公开来源查找、真实下载、固定文件身份、字段标签和日期审计；**没有拟合预测模型，没有读取旧 EEX 2024–2025 确认价格，没有修改旧仓库文件，也没有向 GitHub 写入。** 本次新下载的所有原始文件位于 `/tmp/carbon-next-data/`。

## 1. 可执行结论

本次最有用的进展不是新增一个“已经完全认证的碳价数据库”，而是取得了三个不同用途的实文件候选：

1. **较长 EUA 训练历史及新后期候选：Mendeley d4 的连续期货结算价/量子文件。** 5,262 个唯一日期，2005-04-22—2025-10-02；其中 2023 年以前及 2023 年共 4,808 行，2024 年 259 行、2025 年 195 行。文件内有价格类型、单位和指标码，优于此前 Quantum 的无日期问题。但来源文字存在层级不一致，连续合约换月仍不明确。它是旧存档中本轮首次专项核验的 EUA 子文件，不应把全部 5,262 行称为全新独立样本。
2. **尚未使用的新深圳作者序列：Mendeley fsdf v2。** 实下载得到 `datetime,price`，1,901 个唯一日期，2015-05-18—2023-02-24。它与此前已取得、但没有用于模型的 PLOS 深圳 1,893 条可形成来源一致性核对。两者是同一市场的不同作者样本，不能算两个独立市场。
3. **较新时期的全国 CEA 外部确认候选：Mendeley c287 作者复现包中的官方日报提取结果。** 289 个唯一日期，2025-05-27—2026-07-31，含报告 OHLC、分交易方式价量及 `no_trade`。这段目前只做了字段/日期检查，数值保持未读。它不够独立承担长历史训练，但时间上可以补充较新的外部评价。原始 HTML 没有随包保存，作者提取器及官方文章 ID 仍需按预定规则复核。

**建议先以 d4 的历史段进行下一轮开发设计；将 d4 的 2025 年和 c287 的较新 CEA 段列为候选确认范围，暂不打开价格。** 在做出最终确认承诺前，需明确同一资产已暴露时期、数据语义、跨来源重叠、缺失值和无成交处理。若新方法本身需要长历史 CEA 训练或严格同一 session 的 OHLC，本次数据尚不能支持这个前提。

Ding 的后缀只有 39 条，既短又有上一轮接触披露，不建议把它作为主要独立确认。上海 2024–2025 已经完成旧研究确认评分，不能再贴封存标签。

## 2. 审计方式及“样本量”的准确含义

本轮对新 CSV 只提取日期列；对 XLSX 直接读取工作表标签及日期列 XML。日期的 Excel 序列值会转换为日历，但**价格、成交量及其他数值列没有转换、统计、画图或输入模型**。因此下表的新文件计数是有效日期行数，不是已经验证为有效、正数或可交易的价格样本数。没有对确认期检查价格分布、包含性、重复价格或预测误差。

保存了原始 ZIP/XLSX/CSV、下载 URL、响应状态、字节数、SHA-256、部分代码文件和日期审计 JSON。总清单见 `/tmp/carbon-next-data/DATE_ONLY_DISCOVERY_RECEIPT.json`。没有运行下载包中的 notebooks、训练代码或行情获取脚本。

本轮曾对 Ding 日期字符串直接排序，临时标准输出给出了字典序边界；随后立即用 `%Y/%m/%d` 实际解析校正。本文只采用校正结果：2024-01-02—2024-02-23。旧记录和文件没有因此被改写。

## 3. 本轮实际核验的新文件

| 文件/用途 | 日期行及范围 | 字段证据 | 本轮判断 |
|---|---|---|---|
| d4 EUA 连续期货子文件 | **5,262**；2005-04-22—2025-10-02 | 日期、连续期货结算价、连续期货成交量；明确欧元/吨 CO₂ 当量、吨 CO₂ 当量 | 最优长历史候选；换月及数据供应链尚未完整认证 |
| fsdf 深圳 CSV | **1,901**；2015-05-18—2023-02-24 | `datetime,price` | 可用于新一轮作者来源实验；价格类型/配额年份欠明确；19 条周末日期需解释 |
| c287 CEA CSV | **289**；2025-05-27—2026-07-31 | 日期、`no_trade`、OHLC、挂牌/大宗/单向竞价/总价量 | 新后期候选；时间短，缺原始 HTML，不宣称完整 2021–2026 历史 |
| g56 深圳 XLSX | **2,077**；2013-06-18—2022-02-10 | A 列 `date`、B 列 `Shenzhen`；另有无标签数字块 | 辅助来源核对；不能使用无标签附加列；21 条周末日期 |
| g56 China XLSX | **186**；2021-07-16—2022-04-22 | `date,China` | 太短、价格类型不足，不能补成完整全国市场历史 |
| g56 Composite China XLSX | **281**；2023-08-28—2024-10-31 | `date,Composite price` | 综合价格候选；与前一个文件有很长缺口及口径变化，不得直接拼成连续序列 |
| peltwtcn EUA 原始缓存 | **4,346**；2007-09-10—2024-06-04 | `Date,Open,High,Low,Close,Volume` | 不作为首选新确认：42 条周末日期；软件层补自然日后才得到 6,113 行；spot/futures 描述冲突 |

### 3.1 d4：5,262 个日期的 EUA 连续结算价与成交量

作者存档：[Datasets for Carbon Pricing and Energy Markets in Sustainable Production (2003–2025)](https://data.mendeley.com/datasets/d4bv52fgmd/1)，DOI **10.17632/d4bv52fgmd.1**，2026-02-23 发布，页面声明 CC BY 4.0。

原始 ZIP 的固定公开下载地址：

<https://data.mendeley.com/public-files/datasets/d4bv52fgmd/files/4935504f-cf2d-45c0-add4-d6c20d667a45/file_downloaded>

- ZIP：508,904 字节；SHA-256 `b4ddba5c3609dfa3c13ed4a0c5998c952225c69dc1bc6c8a8f4c73a5fdfd2862`。本轮重新下载为 HTTP 200，与旧来源清单哈希一致。
- ZIP 内真实成员名：`data/期货结算价(..A).xlsx`。工作表名称为 `期货结算价(连续)_欧盟排放配额(EUA)`。
- 工作簿：115,000 字节；SHA-256 `f4d04a394806caa0c4c8f8de02954f41c38dcc3dc3f27a8d28aaa161a6223d5d`。
- 第 1–5 行为指标说明；2026 年没有日期记录。日期均唯一、递增，未出现周末日期。
- 最后一行是“数据来源：Wind”页脚，不能计作价格缺失观测。

| 工作簿标签 | 价格列 B | 成交量列 C |
|---|---|---|
| 指标名称 | 期货结算价(连续):欧盟排放配额(EUA) | 期货成交量(连续):欧盟排放配额(EUA) |
| 频率 | 日 | 日 |
| 单位 | 欧元/吨二氧化碳当量 | 吨二氧化碳当量 |
| 指标 ID | S5100012 | S5100010 |
| 来源行 | 根据新闻整理 | 根据新闻整理 |

日期结构：≤2023 为 4,808 行；2024 为 259 行；2025 为 195 行，最后日期 2025-10-02。如果只允许起点及其 20 步未来标签都在 2025 段内，最多 175 个成熟 H20 起点；不同期限和相邻起点不独立。若允许段前起点，数量会依具体协议改变，不能预报为另一个固定样本量。

来源证据有层级差异：landing 和 README 将该文件写为 Wind 的 EUA 连续结算价，landing 后段又称 EUA 来自 Investing.com；工作簿正文写“根据新闻整理”，页脚写 Wind。可以准确称作**作者存档的 Wind 指标格式连续 EUA 结算价/成交量**，不能据此认证其逐笔来源、交易所成交价、固定到期合约或具体换月算法。

旧 Chen EUA 数据的 2006–2023 年逐年日期数与此文件一致，且区间高度重合；这提示很可能存在共同底层行情，但本轮没有读取价格来证明逐点同一性。旧 Chen 元数据记录到 2024-04-08，本轮保守地不把 d4 的此前时期称为新外部证据。d4 与旧上海使用同一存档，也不使 EUA 和上海的来源记录相互独立。

**最有价值的新增部分是明示日期、价格类型和单位，以及此前未在本次审计中观察数值的后期记录，不是总行数本身。** 公开再分发不能只根据作者 CC BY 声明自动覆盖 Wind/Investing 的第三方权利。本轮只保留本地原文件；后续宜发布下载脚本、来源/哈希和聚合结果。

日期回执：`/tmp/carbon-next-data/d4_eua_dates_audit.json`。

### 3.2 fsdf：深圳 1,901 个日期，原始 CSV 可下载

作者存档：[A CNN-LSTM based deep learning model with high accuracy and robustness for carbon price forecasting: A case of Shenzhen's carbon market in China](https://data.mendeley.com/datasets/fsdf4f4yv5/2)，DOI **10.17632/fsdf4f4yv5.2**，2024-04-18，CC BY 4.0。对应论文 DOI [10.1016/j.jenvman.2024.120131](https://doi.org/10.1016/j.jenvman.2024.120131)。

固定下载：

<https://data.mendeley.com/public-files/datasets/fsdf4f4yv5/files/198be635-b685-47df-b110-0e189d9aee09/file_downloaded>

- ZIP：680,872 字节；SHA-256 `fd003757ff358bca432814984de7bff94a64a39340f487dee273057a06d5eca1`。
- `shenzhen.csv`：37,249 字节；SHA-256 `faff19bd92b4277c09cb15d51807a6eac0879c82f38612d4dd1bea4a98fcbd6e`。
- 字段为 `datetime,price`，日期唯一、递增；2015–2023 逐年数量为 158、245、246、245、247、238、245、243、34。
- 有 19 条周末日期。中国调休工作日不等于证券/碳市场交易日，故不能自动删除，也不能自动认定合法交易日；需按具体交易所日历或原公告解释。
- 包含多个模型 notebook。本轮没有打开 notebook 输出，没有接触这些模型已存的预测或调参成绩。

存档元数据说明“深圳碳价”，CSV 仅写 `price`；还没有认证日均/收盘、具体 SZA 年份、成交方式、合并规则或无成交日处理。它适合作为作者报告价格的透明实验，暂不能写成官方、单一资产的认证收盘序列。

旧研究只记录 fsdf API 的 403 失败。**本轮对相同公开 API 的有效参数请求返回 200，随后固定公开文件 GET 成功**；未更换身份、代理、凭证或绕过挑战。首次本轮文件列表请求漏了 `folder_id=root`，返回 `{error:400}`，修正参数后成功；这是请求格式修正，不能把前者记为数据不存在。

日期回执：`/tmp/carbon-next-data/shenzhen_fsdf_dates_audit.json`。

### 3.3 c287：较新全国 CEA 289 日期的确认候选

作者存档：[Talk, not numbers: entering China's emissions trading system and the carbon content of annual reports](https://data.mendeley.com/datasets/c287n97rz5/1)，DOI **10.17632/c287n97rz5.1**，2026-09-11，CC BY 4.0。

固定下载：

<https://data.mendeley.com/public-files/datasets/c287n97rz5/files/55e5074a-735e-40fd-95dc-8bab84aa79a0/file_downloaded>

- ZIP：6,969,331 字节；SHA-256 `7a38a24f1bec7f8eea2dcd20664a2d37b3b88227854552c38917bc94b38b6806`。
- 成员 `raw/cea/cea_daily.csv`：29,470 字节；SHA-256 `83fe0bfb330e8eaf21604406cf69bcf633901ea017d92563dddc6fc2068df7ef`。
- **289 个唯一日期**，2025-05-27—2026-07-31；2025 年 150 行、2026 年 139 行；无周末日期。
- 字段：`date,no_trade,open,high,low,close,listed_volume,listed_amount,block_volume,block_amount,auction_volume,auction_amount,total_volume,total_amount`。
- `raw/cea/index.json` 为 289 个日期到官方文章 ID 的映射；包中另有 `src/fetch_cea.py`、`src/carbon_price.py`。没有随包保存这 289 页日报 HTML。

作者提取器指向上海环境能源交易所全国碳市场栏目及 `/c/{date}/{article_id}.shtml` 公告，价格正则匹配“元/吨”。其代码区分无交易和缺失：字段缺失为 None；只有文字明确无某类交易时把该类价量记零。该逻辑优于无条件填零，但**静态代码合理不等于这 289 行已与源文逐值对齐**。本轮没有执行获取代码、请求这段官方日报、计算价格有效性或包含性。

全国 CEA 2023 年以来使用综合报价，旧审计已有官方综合收盘不在综合高低价范围内的实例。新段不能未经认证套上单一成交路径的硬 OHLC 包含关系。进一步还要按事先规则处理无交易日、单向竞价、只有大宗成交以及报价缺失，不能将 `no_trade` 或同日成交量当作预测时已经知道的信息。

这只是 **2025–2026 的较短新时期**，不是一个已取得的全国 CEA 千条全历史数据库。与其他来源续接需要验证目标定义和缺口，不能把相隔数月的相邻文件行当作相邻交易日。若后续冻结使用，可以将它作为一个完全保留数值的外部评价候选；本轮日期审计本身不授权提前看它的效果来选模型。

日期回执：`/tmp/carbon-next-data/c287_cea_dates_audit.json`。

### 3.4 g56：能下载，但长深圳与短全国文件必须分开

存档：[Data of carbon price in China](https://data.mendeley.com/datasets/g56r5bn92t/1)，DOI **10.17632/g56r5bn92t.1**，2024-11-12，CC BY 4.0；元数据只说明“Shenzhen and China national”。

| 文件 | 固定下载 URL | SHA-256 |
|---|---|---|
| Shenzhen.xlsx | <https://data.mendeley.com/public-files/datasets/g56r5bn92t/files/e2d2bb92-5cab-4697-ba42-113997b6ce76/file_downloaded> | `13f6b1c9dca2725056296df85b9f9fbc545bf09853e537a47c988fb9e6ab5808` |
| China.xlsx | <https://data.mendeley.com/public-files/datasets/g56r5bn92t/files/544dbc64-253c-40b8-933f-0f5952759fe9/file_downloaded> | `bffc023f685ad26e5762f54728ece125840132cca773f09e57db3c3d90d026bb` |
| Composite China.xlsx | <https://data.mendeley.com/public-files/datasets/g56r5bn92t/files/dc65a274-8ad7-4dd4-8d69-2636455d85fc/file_downloaded> | `3cb6780b183a32d24c577a801cba2c8909c0bb51098117f2abe2a7dd01abef34` |

深圳表有 2,077 个日期，另有 H–O 等列的无标签数值块；本轮没有查看那些数值，不将其误当已知协变量。China 仅 186 个日期，Composite China 仅 281 个日期，且分别结束于 2022-04、开始于 2023-08。不能把全国两个文件拼成 467 天连续同口径价格。

其日期跨度可以补充来源比对，但标签、供应链和配额定义比 d4 更弱。不能仅因深圳有 2,077 行就优先于日期与目标定义更清晰的来源。日期回执：`/tmp/carbon-next-data/g56_dates_audit.json`。

## 4. 未纳入首选的其他核验

### peltwtcn 的“6,113 日数据”是补齐自然日后的数量

[包说明](https://pypi.org/project/peltwtcn/1.0.0/)称 6,113 日，并将目标称 EUA spot；但它是 Merwan Roudane 的第三方实现，不是 Ren 等原论文作者直接认证的原始存档。固定仓库提交：`2fcdcdd8e781cc3f157595fd30dbe6d73d90ff6a`。

本轮实下载原始缓存：

<https://raw.githubusercontent.com/merwanroudane/peltwtcn/2fcdcdd8e781cc3f157595fd30dbe6d73d90ff6a/data/eua_investing_2007-09-10_2024-06-04.csv>

364,914 字节；SHA-256 `3cd3e80f34f10e4e52a630cf4666923ee5e2405636bbefa86826c10bff4dbeef`。日期数实际 **4,346**，含 42 条周末日期。

静态获取代码：

<https://raw.githubusercontent.com/merwanroudane/peltwtcn/2fcdcdd8e781cc3f157595fd30dbe6d73d90ff6a/peltwtcn/datasets.py>

其默认 `load_eua(frequency="calendar")` 使用 `pd.date_range(...,freq="D")`、`reindex(...).ffill().bfill()`，代码将 Investing 工具 8848 标为 **Carbon Emissions Futures**。因此 6,113 不是独立真实交易日数；spot/futures 描述也未一致。MIT 是软件许可，包说明明确底层数据受各供应商条款约束。

本轮打开 PyPI 说明页时，该页自动展示了作者第三方复现的区间与模型成绩。这些不是本项目新实验结果，也未输入模型选择；但不应再把该来源描述成所有历史信息均未接触的盲确认。没有运行其联网获取或填值代码。

### r68 更新包主要是事件研究产物

[Response of Chinese Carbon Markets to Green and Low Carbon Development Policies](https://data.mendeley.com/datasets/r68f5g56pz/2)，CC BY 4.0。本轮下载 868,741 字节的 Updated data of event study.zip，SHA-256 `d37dee7f2c9c94ea918a8f4de93a9c1d94de513a46cab8f1074000fb315db75c`：

<https://data.mendeley.com/public-files/datasets/r68f5g56pz/files/2416bda3-914c-4ed8-9a32-91bd44568dfd/file_downloaded>

目录为 51 个事件窗口和聚合表；只检查最后一个事件工作簿的文本标签，见 AR、CAR、Abnormal Return、事件相对日等。不能把这些模型估计后的异常收益输出当作新的原始逐日价格。较大的原始 Data and Figures.zip 为 474,263,419 字节，未下载、未作可用样本量承诺。

### CRPS Learning / profoc

[profoc 官方项目站](https://profoc.berrisch.biz/)提供 GPL≥3 方法代码。对 GitHub 固定主分支 `b5073ca70aea60ce209676faa99e04da725ffe04` 的递归树审计未发现 CSV/XLSX/RDS/RDA/RData 碳价输入。同行代理查到 CRPS Learning 论文描述 2013–2020 的 EUA month-ahead future 共 2,092 个观测，但本轮没有获得可下载且已验日期的数据文件。现阶段它是方法和实验设计参考，不应写成数据已接入。

### Ember 当前工具

[官方方法说明](https://ember-energy.org/data/european-electricity-prices-and-costs/)明确 EU/UK ETS 成本使用 Montel 的 front December contract；目前页面的数据下载项是电力价格。没有找到已可下载并许可清楚的碳价原始长序列。本轮没有从其图像/嵌入图表反向抽取价格，也没有将网站总体 CC BY 自动扩展到 Montel 行情。

## 5. 现有项目数据库存与暴露状态

以下是已有清单和已完成报告的核对，不重新打开原始或封存价格。旧来源目录写“当时未拟合”时，必须再看后来主报告；不能让旧回执覆盖后续已发生的实验。

| 既有来源 | 已有数量/字段 | 可继续做什么；不能声称什么 |
|---|---|---|
| Quantum | 1,805；报告 Price/O/H/L/Vol；无日历日期 | 已完成上一轮开发；1,025 个 Price 超出报告 H/L，不施加硬 close 包含约束；不能作新盲确认 |
| Ding | ≤2023 共 772；date/close/open/high/low/volumek | 已评分；两条 close 超报告范围；保留作者原值，不改成认证交易收盘 |
| Ding 2024 | **39**；2024-01-02—2024-02-23 | 本轮只日期复核；若起点及20步标签均限在这段，只剩最多19个成熟H20起点；上一轮已有单值接触披露，不作主要新确认 |
| EEX T3PA | 2012–2023 开发 **2,255** 个日聚合拍卖观察；2024–2025 **426** 个封存日观察 | 原有开发多次使用；此次完全不打开确认文件；拍卖价不是二级市场期货结算价 |
| Figshare 28682348 | 广州1,492、湖北1,466、EU1,577；合计4,535 | 旧开发分别1,434、1,408、1,515；长期多次模型实验；其他后缀状态不能只凭更换下载网址重置 |
| 上海 d4 | 1,768；2013-12-19—2025-09-25；日期、收盘价、配额 | ≤2023为1,507；2024–2025共261已在cross_market完成确认；现在全部是已暴露资料 |
| Mendeley LNCEA | 760；2021-07-16—2024-08-30；LNCEA及其他对数因子 | ≤2023为598；后缀162在来源清单曾封存，未在针对性报告中找到其后评分证明；不据此宣称历史从未接触；对数取指数不等于恢复未经舍入原价 |
| ScienceDB 819 | 七试点4,196；2014–2017；配额日成交均价元/吨 | 上海595、北京673、广东732、深圳596、湖北903、天津457、重庆240；有作者清洗/潜在后视预处理，不能当原始实时 close 面板 |
| PLOS 0296105 S1 | 深圳1,893；广东1,706；湖北1,868；上海1,198；日期和 close price | 旧机械核验合计6,665；深圳在 distribution_shape 和 prior_transport 报告明确模型使用0次。可作新一轮作者来源候选，仍与ScienceDB及本轮深圳文件重叠 |
| Chen-zhongxiu EUA | 4,861；2005-05-19—2024-04-08；日期、EU ETS类型、price | 无仓库许可；产品、spot/futures、单位、换月未认证；不能把更换为d4就算全新的早期EUA观察 |
| Wang 2020 作者 EUA | 2,059；2010-12-13—2018-12-27；date/price | 作者存档CC BY-NC 3.0记录；不是已认证交易所行情；旧清单曾标training_approved=false，本轮未改变该记录 |
| CEA 官方直取审计 | 414份日报；2021年114、2022年150、2023年150 | 后两年档案不完整，综合价格口径变化；上一轮降为审计附录，不能算本轮已经可训练的完整CEA OHLC |

这些条数不能相加成“独立训练样本”。同一市场不同作者、不同价格定义、共同日期和高度重叠预测目标会重复信息。

关键内部证据：

- `research/cross_market_innovation_20261017/data_provenance/SOURCE_CATALOG.json`
- `research/cross_market_innovation_20261017/data_provenance/INDEPENDENT_DATA_MANIFEST.json`
- `research/cross_market_innovation_20261017/data_provenance/SCIENCEDB_DATA_MANIFEST.json`
- `research/cross_market_innovation_20261017/REPORT_ZH.md`：明确上海确认已完成，旧来源目录仅保留当时状态。
- `research/event_feedback_innovation_20261017/provenance/DATA_QUANTITY_SUMMARY.json`
- `research/event_feedback_innovation_20261017/provenance/EEX_SOURCE_ADDENDUM.json`
- `research/distribution_shape_diagnostic_20261017c/data_sources/plos_shenzhen/DATA_DISCOVERY_ZH.md`
- `research/distribution_shape_diagnostic_20261017c/REPORT_ZH.md` 与 `research/prior_transport_learning_20261018/REPORT_ZH.md`：新深圳未用于模型。
- `research/stageC_20261013_review/NEW_EUA_ACQUISITION_AND_AUDIT.json`
- `research/ohlc_path_learning_20261019/provenance/DATA_AUDIT_ZH.md`、`SOURCE_MANIFEST.json`。

## 6. 给下一轮协议的具体建议

**训练与确认分离应先按日期和资产写清，再按来源选文件。** d4 的早期 EUA 与旧作者和拍卖研究显著关联，只适合作为开发证据或同源敏感性。新目录、新下载和新许可都不会自动产生新市场信息。

可行的暂定方案是：用已暴露 EUA 及明确纳入的深圳历史开发；候选模型、损失、强基线和缺失/无交易规则冻结后，才决定按统一读取程序开放 d4 的新后期和 c287 CEA 段。对于 d4，可以保守把2025年195日期列作待用确认；2024年的定位须结合旧Chen和其他EUA接触记录，不宜直接称全年新鲜。对 c287，不需为获得一个“第二市场”而假装已有同定义全国长训练历史；若需要跨市场迁移，应把迁移本身及目标市场前缀使用写成明确任务。

fsdf、PLOS、g56 深圳的优势是独立作者链条可相互核对，不是可以叠加成三倍样本。优先核实同日期是否为同类价格、周末记录如何形成、缺失交易日及不同 SZA 年份的处理。若核查差异，需要保留原值、源记录与有依据的解释；不要用插值、裁剪或自动移位把数据修成喜欢的结构。

原价预测仍需原价持久性 RW、带漂移/适当差分回归、同信息集控制和固定时间评价。是否加入报告成交量，取决于 lagged volume 的时间可用性和定义。不要把未来同日成交量、事后年内标签或用全样本构造的情绪/政策变量当作历史信息。

本次没有找到同时满足“长期日频、完全认证合约、明示可再分发、全新跨市场确认”四个条件的完美来源。**可以继续研究的资料已比 Quantum/Ding 更充分，但正式成果仍要按作者报告序列和有限确认样本来限定主张。**

## 7. 追加：获准读取的 ≤2023 开发数值准备与审计（2026-10-19）

本节记录后续明确授权后的新工作。第 1–6 节的“只检查日期、不读价格”描述的是发现阶段；此后仅对 **d4 EUA 的 2023 年及以前部分、四个已有开发文件和 Ding 的已有 ≤2023 开发文件**读取价格并做来源对比。本节没有拟合模型，没有计算任何预测结果，没有打开 EEX 2024/2025，PLOS 深圳继续留待独立确认。d4 混合时期 XLSX 的字节用于哈希和 XML 日期解析，但程序先读取 A 列日期、先排除 `date >= 2024-01-01`，再访问 B/C 价格和量；后期 B/C 未转换、未统计、未输出。

### 7.1 可直接供模型开发的文件

目录为 `/tmp/carbon-next-data/prepared/`。价格一律保留原始文本，d4 CSV 为 `date,price,volume`；四个既有开发 CSV 按字节复制，保留 `source_excel_row/source_date_kind` 或 `source_row` 等原有追溯字段。上海输出别名由 `SHANGHAI_D4` 简化为 `SHANGHAI`，文件内容不改。

| 文件 | 行数 | 开发日期 | SHA-256 |
|---|---:|---|---|
| EUA_LONG_development.csv | 4,808 | 2005-04-22—2023-12-29 | `aa4ca31146d872a39628d2e0de8c14a0c4bcdfd3b801cb6e50601b714aafbdc2` |
| GUANGZHOU_development.csv | 1,434 | 2018-01-03—2023-12-29 | `8d78a4f13c5a799519546fd3bfa0e0ac5eaa52b0441a983a95ef06f5e5e34051` |
| HUBEI_development.csv | 1,408 | 2018-01-02—2023-12-29 | `04655d59c1b585f6c35c02a43469a965fe3589eb613d7d8d8e46996751b7d797` |
| SHANGHAI_development.csv | 1,507 | 2013-12-19—2023-12-29 | `425424353bb3ac09e3d96b291680b4179e86f180da626a5d19e5c1e58ed960a1` |
| EU_REPORTED_development.csv | 1,515 | 2018-01-02—2023-12-29 | `ca7c192380897076ef57516a11ea2ccefda6ba92ab9823d9aa8f2bb281cefedc` |

d4 新 CSV 为 122,757 字节。其原始 XLSX 和 ZIP 身份仍为第 2 节列出的固定哈希：XLSX `f4d04a394806caa0c4c8f8de02954f41c38dcc3dc3f27a8d28aaa161a6223d5d`，ZIP `b4ddba5c3609dfa3c13ed4a0c5998c952225c69dc1bc6c8a8f4c73a5fdfd2862`。本轮实际确认 ZIP 成员名是 `data/期货结算价(..A).xlsx`，其字节哈希与独立提取的 XLSX 完全一致，不能把 README 中较完整的文件名当作实际成员名。

Ding 仅供本节来源对比，继续使用 `/tmp/carbon-path-local/ding.csv` 的 772 行；哈希为 `6cba3c37d990fb18c7de09a3aae2a7644a2067e382ce4983960c9ab97c074ab4`。没有重新打开带 2024 后缀的 Ding 原始 CSV。

可复核运行入口：

```bash
python -B /tmp/carbon-next-data/prepare_development.py \
  --repo /workspace/scratch/5a60a5e17bce/carbon-research \
  --data-root /tmp/carbon-next-data \
  --ding-development /tmp/carbon-path-local/ding.csv
```

脚本只接受清单中的输入哈希，生成开发 CSV、`EUA_LONG_date_lineage.csv`、`DEVELOPMENT_NUMERIC_AUDIT.json` 和 `PREPARED_SOURCE_MANIFEST.json`。这些文件中没有模型权重或预测。原始及逐行衍生价格均留在本地；此脚本和本节不授权重新发布供应商行情。

### 7.2 数值质量与日历审计

五个开发文件日期均严格递增、无重复日期；所有 `price` 均为有限正数，无价格缺失、零价或负价。此结论是对作者文件完整性的核查，不是对官方行情正确性的认证。

| 数据 | 价格最小 / 最大（原文件尺度） | 相邻同价次数 | 价格相对变化绝对值 >20% 的记录数 | 周末日期数 |
|---|---|---:|---:|---:|
| d4 EUA 全部 ≤2023 | 0.01 / 97.67 | 235 | 24 | 0 |
| d4 EUA 2013–2023 | 2.70 / 97.67 | 73 | 4 | 0 |
| GUANGZHOU | 12.00 / 95.26 | 47 | 0 | 14 |
| HUBEI | 14.07 / 61.48 | 56 | 0 | 0 |
| SHANGHAI | 4.20 / 74.71 | 201 | 16 | 0 |
| EU_REPORTED | 7.62 / 97.67 | 14 | 0 | 0 |

变化旗标定义为相邻作者报价 `abs(p_t / p_previous - 1) > 0.2`，只是描述性回查清单。**旗标不导致删除、截尾、修价、移位、补齐或按模型成绩选择样本。** d4 早期极低价格也不能直接判成错误：欧洲委员会的[历史说明](https://climate.ec.europa.eu/areas-action/carbon-markets/eu-emissions-trading-system-eu-ets/development-eu-ets-2005-2020_en)确认第一阶段配额不可结转到第二阶段且 2007 年市场价格曾降至接近零的状态；但这段一般市场历史也不能逐行认证本工作簿。

d4 成交量有 **1 个缺失、25 个原值零、0 个负数**。缺失日是 **2022-04-18**，输出仍为空单元格。2013–2023 子段有 1 个缺失、17 个原值零。价格预测可以保留全部价格记录；若以后加入滞后成交量，须在预先协议中采用明确的缺失指示与仅过去可用的处理，不得把缺失擅自解释成零成交。不应从日期接近节假日就自动删除零量记录。量的完整缺失/零量日期清单已写入机器审计文件。

日历并不等价于各自交易所完整交易日历。广州 14 个周末日期按作者原样保留；湖北相邻最大日期间隔为 **60 天（2020-01-23—2020-03-23）**；上海最大间隔为 **248 天（2021-03-23—2021-11-26）**，2021 年全年仅有 47 条作者报价。EU_REPORTED 在 2018/2019 年分别只有 241/240 行，而 d4 同两年各为 258 行；在这些年份 d4 多出的 35 个日期不能用插值或日历补齐伪造为 EU_REPORTED 原始记录。当前数据的 `h=20` 应称 **20 个作者报价观测步**，除非另有足够官方证据，不能统一写成 20 个日历日或 20 个交易日。

### 7.3 两个 EUA 作者来源的高度重合，以及 Ding 的差异

使用 ISO 日期内连接，按原始十进制数值判断相等；没有尝试优化比例因子、时间偏移或自动修正价格。

| 比较 | 共同日期 | 精确同价 | 绝对差中位数 | 最大绝对差 | 含义 |
|---|---:|---:|---:|---:|---|
| d4 EUA 与 EU_REPORTED | 1,515 | **1,335** | 0 | 0.38 | 很大部分是同一历史价格信息，不能视为两个独立市场 |
| d4 EUA 与 Ding | 772 | **0** | 0.38 | 3.36 | 同市场、不同作者报告价格；不认证为同一合约/时点 |
| EU_REPORTED 与 Ding | 772 | **0** | 0.39 | 3.36 | 不应因日期相同就合并标签或平均价格 |

更关键的是：d4 与 EU_REPORTED 在 **2018–2022 年的全部 1,257 个共同日期逐值完全相等**。2023 年共同 258 日中 78 日相等，180 日不同；首次不同日期为 **2023-04-10**，180 个差异均为 d4 更高，差值最大 0.38。差异不是每一个后续日期都必定出现。现在没有证据区分修订、连续合约处理、结算/收盘定义、来源更新或其他原因，不能擅自归因为某一机制，更不能把较新的工作簿称为逐日真实版本。

据此建议：保留五个命名文件供审计，但主要市场面板中 **EUA_LONG 与 EU_REPORTED 只选一个作为 EUA 主目标**，另一个作为同市场来源敏感性；不要将二者相互作为所谓独立市场迁移来源。d4 的新增价值主要是更长历史及明确的列定义。与 Ding 不精确相等同样不构成“不同市场”的证据。不同作者不等于独立原始观察。

### 7.4 价格类型、单位和公开回查边界

d4 工作簿明确将 B 列写为连续 EUA **期货结算价**，单位为 **欧元/吨二氧化碳当量**，指标 `S5100012`；C 列为连续期货成交量，单位为 **吨二氧化碳当量**，指标 `S5100010`。但[作者存档页面](https://data.mendeley.com/datasets/d4bv52fgmd/1)前段和 README 指向 Wind，页面后段又写 EUA 来自 Investing.com；工作簿正文来源为“根据新闻整理”，页脚为“数据来源：Wind”。README 还保留 `[Your Paper Title]` 和 `[Your Email]` 占位符。这些文本不能支撑“经交易所认证、指定到期合约、既定换月算法”的说法。

[Figshare 原始存档](https://figshare.com/articles/dataset/Daily_Carbon_Price_Data_from_Hubei_Guangzhou_and_EU_Emissions_Trading_Systems_2018_2024_/28682348)称三个市场数据来自官方交易所并作为收盘价处理，但本次检查的官方存档介绍没有给出 EU 的币种、合约及换月规则，旧项目审计也明确这些信息未核实。它与 d4 大量数值相等是重要来源关联证据，**不能替代独立的币种或结算口径认证**。因此不能把 Figshare 的“收盘价”说明移植到 d4“期货结算价”，也不能默认为两个来源已有完全一致的价格类型。上海作者工作簿中的 `收盘价` 同样未完成交易所逐行及币种的完整认证；本次没有强行将其标为已验证的元/吨原始日线。

本次没有发现可支持对 EU_REPORTED 做美元/欧元换算的可靠声明，也没有采用任何估计汇率修正。论文中 d4 可以注明“工作簿报告为 EUR/tCO₂e”；EU_REPORTED、上海等应继续以作者报告价格及原文件尺度限定表述。跨市场原始单位未充分认证时，不把各市场未经处理的 MSE 直接相加并标为统一货币平方。

d4 与 Figshare 的存档均声明 CC BY 4.0；这是作者存档许可，不能自动覆盖 Wind/Investing 的第三方数据许可，也不能补齐官方行情溯源。Ding 无仓库数据许可。保持公开代码、输入下载地址/版本/哈希、处理规则和聚合审计；原始/逐行衍生价格留在本地。没有可公开无条件访问的逐日对应合约与换月清单，因此本轮不会以自动修价模拟官方认证。

### 7.5 训练年代与 2021/2022/2023 的可评价数量

建议在模型训练前固定：EUA 主开发采用 **2013 年起**的 2,839 条记录，2021/2022/2023 分别作为前向年度开发评价；2005–2012 的 1,969 条仍保留，若协议预定做全历史敏感性，就完整报告它。2013 起点有制度理由：[欧洲委员会历史说明](https://climate.ec.europa.eu/areas-action/carbon-markets/eu-emissions-trading-system-eu-ets/development-eu-ets-2005-2020_en)明确第三阶段为 2013–2020，并说明该阶段采用统一欧盟排放上限等较大规则变化；[官方总页](https://climate.ec.europa.eu/areas-action/carbon-markets/eu-emissions-trading-system-eu-ets_en)明确第四阶段为 2021–2030。这个建议不是根据任何预测分数作出的。全 2005 历史和 2013 子段都不保证跨时期同分布。

三个中国试点可从各自作者开发序列的首日开始使用：广州/湖北为 2018 年，上海为 2013-12-19。不要为了统一面板删掉上海早期历史，也不要填补其 2021 大缺口。上海 2021 数量很小，应单独清楚列示其结果及不确定性。EU_REPORTED 2018 起仍可作为同源敏感性，但不能与 EUA_LONG 相加来宣称多出一个独立市场。

以下仅为**数据容量**：历史窗口为 `p[t-20],...,p[t]` 共 21 个节点；起点日期须属于评价年份 Y，目标 `p[t+h]` 必须存在并在同一 Y 年内成熟；全部所需价格须有限。未按交易所日历补齐，h 表示作者报价观测步。尚未扣除之后模型协议规定的验证集、purge/embargo 或训练预算，重叠窗口也不是独立样本。

| 开发序列 | 2021：H1 / H5 / H20 起点 | 2022：H1 / H5 / H20 起点 | 2023：H1 / H5 / H20 起点 |
|---|---|---|---|
| EUA_LONG（2013 起；全历史相同年度评价数） | **258 / 254 / 239** | **257 / 253 / 238** | **257 / 253 / 238** |
| GUANGZHOU | 246 / 242 / 227 | 241 / 237 / 222 | 241 / 237 / 222 |
| HUBEI | 241 / 237 / 222 | 236 / 232 / 217 | 234 / 230 / 215 |
| SHANGHAI | 46 / 42 / **27** | 214 / 210 / 195 | 195 / 191 / 176 |
| EU_REPORTED（同市场敏感性） | 258 / 254 / 239 | 257 / 253 / 238 | 257 / 253 / 238 |

在各年 1 月 1 日以前、所有 H20 标签均已成熟时，可用于训练的最大监督起点数为：

| 训练序列定义 | 2021 评价之前 | 2022 评价之前 | 2023 评价之前 |
|---|---:|---:|---:|
| EUA_LONG：2013 起 | **2,024**（2,064 价格） | **2,283**（2,323 价格） | **2,541**（2,581 价格） |
| EUA_LONG：2005 起完整敏感性 | 3,993（4,033 价格） | 4,252（4,292 价格） | 4,510（4,550 价格） |
| GUANGZHOU：2018 起 | 663（703 价格） | 910（950 价格） | 1,152（1,192 价格） |
| HUBEI：2018 起 | 654（694 价格） | 896（936 价格） | 1,133（1,173 价格） |
| SHANGHAI：2013-12 起 | 1,009（1,049 价格） | 1,056（1,096 价格） | 1,271（1,311 价格） |
| EU_REPORTED：2018 起 | 700（740 价格） | 959（999 价格） | 1,217（1,257 价格） |

EUA H20 的首末评价起点分别为 2021-01-04—2021-12-03、2022-01-03—2022-12-01、2023-01-02—2023-11-30；每年最终标签分别成熟于 2021-12-31、2022-12-30、2023-12-29。其他序列和期限的精确首末日期在 `DEVELOPMENT_NUMERIC_AUDIT.json`。这些年份早已不同程度用于此前模型研究，因此本轮应称开发评价，不能更换来源后改称全新盲测试。

### 7.6 EEX ≤2023 本地日价定位结果

已有开发文件的**准确身份**可从项目元数据恢复，但当前 scratch 没有找到这个独立开发 CSV，因此本次没有展示它的开发头，也没有为定位而打开混合年份归档。

- 文件名：`EEX_EUA_T3PA_daily_development.csv`。
- 已有清单数量：2,255 个按拍卖量加权的日聚合，2012-10-26—2023-12-18。
- 字节数：373,159。
- SHA-256：`5a0c7580edebc2a7a5e66d7487568f6d99a8e08cca0785676c2aaeccad7db6bb`。
- 字段清单：`date,price,auction_volume,total_bid_volume,cover_ratio,bidder_count,successful_bidder_count,auction_count,contract,auction_names,completion_evidence,missing_fields,component_ids`。
- 旧来源路径 `/workspace/scratch/b31c5edbd0de/scratch/next_data_20261017/local_only/eex_prepared/development/EEX_EUA_T3PA_daily_development.csv` 当前不存在。
- 旧恢复副本 `/workspace/scratch/be2216688c09/recovery_audits/delivery/portable_replay/local_only/inputs/EEX_EUA_T3PA_daily_development.csv` 当前不存在。
- 标准复现相对路径是 `local_only/prepared/development/EEX_EUA_T3PA_daily_development.csv`；当前 `eex_local_replay` 对应路径也不存在。

证据为 `research/event_feedback_innovation_20261017/data_reproduction/provenance/EEX_PREPARATION_AUDIT.json` 的 development 输出项，以及 `research/prior_transport_learning_20261018/audits/recovery_delivery/` 中旧命令回执。EEX 是一级市场拍卖日聚合价，不能用当前 d4 二级市场连续期货列替代后仍称同一目标。若之后恢复该开发副本，应先核对上面的固定哈希，并只把独立开发文件提供给模型。EEX 2024/2025 本节完全未开封。

## 8. 当前正式面板与按观测分块的滚动协议（后续冻结规则，2026-10-19）

本节是数据准备任务后续收到并采用的正式规则，优先于前文的候选建议。第 7.5 节的 21 节点容量表和先按日历年分段的方案只保留为筛选诊断。**当前正式实验以第 40 条报价（索引 39）作为最早可用起点；40 条是 EWMA 初始化及特征暖机的最小初始前缀，实际网络输入为连续 20 行历史特征。按观测起点构造 126 个验证、252 个校准起点，扩展训练。** 本节只准备数据和计算日期容量；没有拟合碳价预测模型或计算模型评价分数。

### 8.1 独立获取入口与四市场主面板

公开脚本：`research/distribution_update_learning_20261019/data_sources/acquire_development.py`。它仅依赖 Python 标准库，可用普通 HTTPS GET 下载前文的固定 Mendeley ZIP；先核对 ZIP、目标 XLSX 和衍生 EUA 主 CSV 的 SHA-256，再接受输出。四个旧开发 CSV 均必须逐字节匹配既有清单哈希。输出目录必须位于仓库外且为新目录或空目录；脚本不覆盖已有非空目录。

```bash
python -B /path/to/carbon-research/research/distribution_update_learning_20261019/data_sources/acquire_development.py \
  --download \
  --repo-root /path/to/carbon-research \
  --output-dir /tmp/carbon-development-new
```

已持有完全相同 ZIP 时，以 `--archive-local /absolute/path/to/mendeley_d4_data.zip` 替代 `--download`。无需打开 EEX，也不需要 PLOS 深圳、Ding 2024、d4 2024/2025 价格或 c287 CEA 价格。

本轮已经用该脚本实际完成一次公开网络下载，HTTP 200，固定字节身份全部通过。产物位于 `/tmp/carbon-distribution-update-data/`：

| 主面板 | 私有相对路径 | 行数 | 主要角色 |
|---|---|---:|---|
| EUA_LONG | development/EUA_LONG_development.csv | **2,839** | 2013-01-02—2023-12-29 的作者连续 EUA 结算价 |
| GUANGZHOU | development/GUANGZHOU_development.csv | **1,434** | 既有开发文件，原字节不改 |
| HUBEI | development/HUBEI_development.csv | **1,408** | 既有开发文件，原字节不改 |
| SHANGHAI | development/SHANGHAI_development.csv | **1,507** | 既有 SHANGHAI_D4 开发文件，原字节不改 |

四市场共 7,188 条作者报价记录；这是记录总数，不是独立训练样本数。主 EUA CSV 为 **71,995 字节**，SHA-256 `142f391305c482eb3bd58dcf8df477db90cf54402495527749a53e126cdf0e7f`。它与第 7 节保留的 4,808 行全历史文件是不同、明确标记的派生文件。该新入口对 2013 年以前以及 2024 年以后的 EUA 行只处理日期，只有通过 `2013-01-01 <= date < 2024-01-01` 的行才访问价格/成交量单元格；没有价格修正。

EU_REPORTED 经过原哈希核验，放在私有 `comparison_only/EU_REPORTED_development.csv`。它不进入默认目标面板或相互迁移源；d4 与其在 2018–2022 年全部 1,257 个共同日期精确同价的审计结论保留。其他原始归档、逐行价格、逐点预测或权重均不复制到本研究的公开数据来源目录。

### 8.2 正式分块定义

索引从 0 开始，对每个市场、评价年份 Y 和期限 h∈{5,20} 独立定义：

```text
test_start = 当年首条作者报价的索引
cal_end    = test_start - h - 1
cal_start  = cal_end - 251
val_end    = cal_start - h - 1
val_start  = val_end - 125
train_end  = val_start - h - 1
train_start = 39
```

端点均为包含端点。`train_start=39` 保证最早起点的初始 EWMA 已可用，且最早被网络使用的特征行是索引 20。实际网络在每个起点 t 使用 t−19,…,t 共 20 行、每行 3 个特征；这些增量需要 p[t−20],…,p[t]，EWMA 还递归依赖更早过去。40 条是最小初始暖机前缀，不是每个起点的 40 步网络窗口或 EWMA 截断长度。训练必须至少有 128 个起点，验证和校准分别完整保留 126、252 个起点。若训练不满 128 或所需历史/完整角色块不可用，整项案例按事先规则排除，不能见到结果再调整。评价起点及 t+h 终点须同属 Y 年。两个期限有各自的角色端点，不把未来未成熟标签越过边界。

这组定义使训练最后一个目标刚好在验证首起点之前成熟，验证最后一个目标在校准首起点之前成熟，校准最后一个目标在评价首起点之前成熟。数值准备只保留原价；没有补齐交易日历或按同价/跳变删除记录。 h 是作者报价观测步。

**上海 2021 年的整年评价仍预定排除，h=5 和 h=20 均排除。** 原始 47 条报价和可用的 42/27 个年度成熟起点保留在审计里，不能把这 27 个高度重叠的 H20 目标当作充分的独立可靠证据。该排除规则不改变原始文件。

### 8.3 正式容量与边界核验

| 市场 | 年份 | h | 训练起点 | 验证 | 校准 | 拟纳入评价起点 | 校准起点日历跨度（含首尾，天） |
|---|---:|---:|---:|---:|---:|---:|---:|
| EUA_LONG | 2021 | 5 | 1632 | 126 | 252 | 254 | 353 |
| EUA_LONG | 2021 | 20 | 1587 | 126 | 252 | 239 | 357 |
| EUA_LONG | 2022 | 5 | 1891 | 126 | 252 | 253 | 353 |
| EUA_LONG | 2022 | 20 | 1846 | 126 | 252 | 238 | 355 |
| EUA_LONG | 2023 | 5 | 2149 | 126 | 252 | 253 | 353 |
| EUA_LONG | 2023 | 20 | 2104 | 126 | 252 | 238 | 353 |
| GUANGZHOU | 2021 | 5 | 271 | 126 | 252 | 242 | 378 |
| GUANGZHOU | 2021 | 20 | 226 | 126 | 252 | 227 | 378 |
| GUANGZHOU | 2022 | 5 | 518 | 126 | 252 | 237 | 372 |
| GUANGZHOU | 2022 | 20 | 473 | 126 | 252 | 222 | 370 |
| GUANGZHOU | 2023 | 5 | 760 | 126 | 252 | 237 | 376 |
| GUANGZHOU | 2023 | 20 | 715 | 126 | 252 | 222 | 376 |
| HUBEI | 2021 | 5 | 262 | 126 | 252 | 237 | 428 |
| HUBEI | 2021 | 20 | 217 | 126 | 252 | 222 | 435 |
| HUBEI | 2022 | 5 | 504 | 126 | 252 | 232 | 379 |
| HUBEI | 2022 | 20 | 459 | 126 | 252 | 217 | 379 |
| HUBEI | 2023 | 5 | 741 | 126 | 252 | 230 | 383 |
| HUBEI | 2023 | 20 | 696 | 126 | 252 | 215 | 383 |
| SHANGHAI | 2021 | 5 | 617 | 126 | 252 | 0（预定排除；可用42） | 579 |
| SHANGHAI | 2021 | 20 | 572 | 126 | 252 | 0（预定排除；可用27） | 583 |
| SHANGHAI | 2022 | 5 | 664 | 126 | 252 | 210 | 864 |
| SHANGHAI | 2022 | 20 | 619 | 126 | 252 | 195 | 875 |
| SHANGHAI | 2023 | 5 | 879 | 126 | 252 | 191 | 705 |
| SHANGHAI | 2023 | 20 | 834 | 126 | 252 | 176 | 736 |

共 24 个计划组合，22 个具备正式模型评价资格；除上海 2021 两个期限外没有其他排除，所有组合的训练数均超过最低 128。24 个组合的 72 个角色边界检查全部通过。这里“拟纳入”是数据容量，不代表已经发生任何模型训练。

校准块数相同不表示时间长度相同，也不表示 252 个独立样本。上海的校准起点窗口跨越明显较长的日历期：

| 上海评价年 | h | 校准首起点 | 校准末起点 | 最后目标成熟日期 | 起点跨度（天） | 首起点至最后成熟跨度（天） |
|---|---:|---|---|---|---:|---:|
| 2022 | 5 | 2019-08-12 | 2021-12-22 | 2021-12-31 | 864 | 873 |
| 2022 | 20 | 2019-07-09 | 2021-11-29 | 2021-12-31 | 875 | 907 |
| 2023 | 5 | 2021-01-18 | 2022-12-23 | 2022-12-30 | 705 | 712 |
| 2023 | 20 | 2020-11-26 | 2022-12-01 | 2022-12-30 | 736 | 765 |

这种分块解决“某日历年只有 27 个校准起点”的数量问题，但没有解决上海缺期和历史分布变化问题。长校准窗口和重叠目标的依赖性必须在方法及结果限制中保留，不能据固定 252 个起点自动宣称独立同分布或覆盖保证。

完整逐角色、逐组合的起止索引、历史参考日期、起止日期、目标成熟日期和数量见 `SPLIT_DATE_COUNTS.json` 中的 `formal_observation_blocks.cases`，以及同目录的 `FORMAL_SPLIT_ROLE_TABLE.csv`。该 CSV 只有分块日期/索引/数量，不含价格或预测值。

### 8.4 日历三段方案只作筛选诊断

以下使用与正式实验一致的最早起点索引 39（启动时至少 40 条原价前缀）作为日期容量门槛，以及各角色内完整成熟标签，展示此前候选日历方案的容量。它与第 7.5 节 21 节点的一般容量表有明确区别。日历方案没有作为正式默认，也没有根据其预测效果选择。每格顺序均为 **H5 / H20**；评价年中的上海 2021 仍统一排除。

| 市场 | 固定 train≤2018 | tuning2019 | calibration2020 |
|---|---:|---:|---:|
| EUA_LONG | 1503/1488 | 253/238 | 254/239 |
| GUANGZHOU | 170/155 | 240/225 | 239/224 |
| HUBEI | 198/183 | 239/224 | 203/188 |
| SHANGHAI | 702/687 | 137/122 | 156/141 |

扩展日历方案按评价年 Y 使用 train≤Y−3、tuning=Y−2、calibration=Y−1。Y=2021 与上表相同；其余年份：

| 评价年 | 市场 | 训练 H5/H20 | 调参 H5/H20 | 校准 H5/H20 |
|---:|---|---:|---:|---:|
| 2022 | EUA_LONG | 1761/1746 | 254/239 | 254/239 |
| 2022 | GUANGZHOU | 415/400 | 239/224 | 242/227 |
| 2022 | HUBEI | 442/427 | 203/188 | 237/222 |
| 2022 | SHANGHAI | 844/829 | 156/141 | 42/27 |
| 2023 | EUA_LONG | 2020/2005 | 254/239 | 253/238 |
| 2023 | GUANGZHOU | 659/644 | 242/227 | 237/222 |
| 2023 | HUBEI | 650/635 | 237/222 | 232/217 |
| 2023 | SHANGHAI | 1005/990 | 42/27 | 210/195 |

日期审计显示，日历方案会将上海 2021 的 42/27 个 H5/H20 起点用于 2022 校准、2023 调参。因此后续规则统一改为第 8.2 节按观测分块，不以模型成绩作选择。

### 8.5 公开文件、复现和保留期状态

本目录新增的公开文件只包括数据获取代码、来源与处理限制、聚合数值审计、日期分块数量和报告。`SOURCE_MANIFEST.json` 记录实际公开下载、源文件哈希、主输出哈希、单位声明、相关来源重复、零量/缺失和稀疏限制；`DEVELOPMENT_NUMERIC_AUDIT.json` 保留先前已获准的 ≤2023 审计结果；`SPLIT_DATE_COUNTS.json` 给出当前正式规则及已降级为诊断的日历方案。没有向本目录复制原始 ZIP/XLSX、逐行行情或逐点预测/权重。

新 EUA 2025、d4 2024 后缀、c287 CEA、PLOS 深圳等未来确认候选的价格均没有在此获取流程中读取或用于训练。EEX 不再继续寻找、恢复或打开。本次数据准备不会把先前已经暴露的 2021–2023 开发年份重新标成新盲测；独立确认仍须在方法、选择规则及评价承诺冻结后另行处理。

### 8.6 启动后的历史窗口表述澄清

独立读取实际 `source/run_market.py`、`market_models.py` 和全部 22 个准备数组后确认：网络输入严格为 `20×3`，协议的 `history_length=20` 与之相符。先前将最早允许索引 39 简写成“40 个历史报价节点”的说明容易被误解为网络 40 步输入，本报告现已更正。这是文档澄清，没有改价格、分块起止索引、任何输入数组、神经结构或已拟合参数。

分块模板 JSON/CSV 中既有 `first_history_index` 和 `first_history_date` 数值按起点减 39 计算，保留它们作为原先暖机约定的历史参考，不能拿它们推断实际特征张量的窗口。实际首特征行是 t−19，首个相关价格增量需要 p[t−20]，EWMA 初始后递归使用全部较早过去。正式角色索引、成熟日期与计数完全未变。

时间防护另由 root 在启动后做了一处修正：先只读日期、拒绝 2024+，再读价格。独立审查反向还原该代码块后的哈希与启动前源码完全相等；全部 22 个实际输入的固定哈希和日期均属 ≤2023，所有准备数组独立复算一致，因而这项防护更正不影响当前结果，也不需重做已有 44 次统计拟合。详见 `audits/INDEPENDENT_TIME_DATA_AUDIT_ZH.md`。
