# 新一轮碳价数据发现与机械核验

核验日期：2026-10-17。工作范围为来源、许可、文件身份、字段、日期和有效行数；本记录不包含价格统计、预测结果或新的市场拟合。

## 1. 可用结论

本轮成功取得并固定了一个新的作者公开来源：Wei、Ouyang 的 PLOS One 论文附件。其深圳工作表有 **1,893 条有效日期与数值收盘价记录，2013-08-05 至 2022-03-11**。这满足“单个系列超过 1,000 条”的文件规模要求，但深圳在既有 ScienceDB 来源中已出现，因此应称为**新的作者来源及更长的深圳历史候选**，不能称为一个从未出现的市场，也不能把全部 1,893 条计作完全新增的独立观测。[P1–P4；M1]

本轮没有完成一个“超过 1,000 条、全新、全国 CEA、可下载并明确原始价格口径”的文件核验。北京的作者附件有明确论文列表和样本量线索；正常请求 Wiley 正文 URL 返回 HTTP 403，附件点击仅得到检索工具 InternalError，未成功解析最终文件 URL。尚不能交付为已核验数据，不能将正文响应写成附件的 HTTP 响应。

当前深圳候选仍有具体边界：附件字段只标为 `close price`，未明确 SZA 配额年份、不同年份配额如何汇总或衔接，也没有成交量列。可以按作者发布的深圳收盘价序列进入后续预先说明口径的研究；目前不能将其写成已经逐项验证的交易所特定合约原始行情。

## 2. 已下载文件及机械结果

| 项目 | 固定值 |
|---|---|
| 论文 | Xiaolu Wei、Hongbing Ouyang，*Carbon price prediction based on a scaled PCA approach* |
| 论文 DOI | `10.1371/journal.pone.0296105` |
| 附件 DOI | `10.1371/journal.pone.0296105.s001` |
| 发布平台 | PLOS One，作者提供的 S1 XLS |
| 完整文件字节数 | 13,625,856 |
| SHA256 | `90c49501afa9beb7370c4409084a28729daa4904a7e7e5de310d0f6bd7a3ba10` |
| 成功响应 | HTTP 200，读取至 EOF，长度与响应声明一致 |
| 发现工作区中的原文件（不在公开包内） | `downloads/plos_0296105_s001_complete.xls` |
| 输出字段规则 | 仅第 0 列日期与第 1 列 `close price`，公开 CSV 列名为 `date,price` |
| 本轮 CSV 导出 | **未执行**；未将新价格文件放入任何模型目录 |

稳定下载地址：[PLOS S1 附件](https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0296105.s001)。下载后应校验上述 SHA，不能仅凭相同 URL 或文件名接纳以后发生变化的响应。

工作表物理尺寸均为 2,295 行、73 列。前两行为头部；有效观测定义为“可解析日期且 `close price` 是有限数值”。工作表尺寸不能作为样本量。有效日期均保持原始顺序、严格递增且无重复。[M1]

| 工作表 | 有效记录 | 首日 | 末日 | 有日期但无有限数值价格的行 |
|---|---:|---|---|---:|
| shenzhen | 1,893 | 2013-08-05 | 2022-03-11 | 398 |
| guangdong | 1,706 | 2013-12-18 | 2022-03-22 | 585 |
| hubei | 1,868 | 2014-04-28 | 2022-03-23 | 423 |
| shanghai | 1,198 | 2013-11-26 | 2022-03-23 | 1,093 |

四个工作表合计 6,665 条有效记录，不代表 6,665 条全新的数据。广东、湖北、上海均已有其他来源被使用；未经独立口径比较，不能把不同作者文件直接拼接、去重后视为官方扩展。

正文的湖北主实验明确说明去掉零成交量数据。该说明不构成对深圳每一个空价格行的逐行解释。这里只记录空单元格与日期，不把空值、未记录日期或连续同价解释成无成交。提取器不填充日历、不插值、不排序、不删除数值零，也不读取 71 列技术指标作为模型输入。[P1；M1]

## 3. 与已见来源的关系

去重仅阅读了既有来源目录的元数据，没有打开旧市场 CSV、NPZ 或确认留出文件。读取的目录及身份为：

| 目录 | SHA256 |
|---|---|
| `research/cross_market_innovation_20261017/data_provenance/SOURCE_CATALOG.json` | `2e7ac7f029ac13a9dd16404dc1b5b05a8cf695ac1b3156aaf8cb02de81b4c3c1` |
| `research/event_feedback_innovation_20261017/provenance/SOURCE_OPTIONS.json` | `b766cccfbdfe3077536e8dc00583412e64f64a0e9a5ab7ea11ea3421c4303e12` |

既有 ScienceDB `10.11922/sciencedb.819` 覆盖七试点的 2014—2017 年，包含深圳，字段为配额日成交均价。新 PLOS 深圳表有 911 条落在 2014—2017 年；另有 982 条位于这些年份之外：2013 年 89 条，2018—2022 年 893 条。这是**基于日期年份的覆盖核对**，不是两来源原价或逐日记录的相等性核验。均价和收盘价也不是天然同一口径。[M1；既有来源目录]

既有的广州、湖北、上海、EU、EEX，以及全国 CEA 的 760 条对数价格作者数据均不能重新计作本轮的新数据。已见 CEA 对数文件不应通过对已舍入对数取指数而被描述成获得了未经变换的原始价格。旧官方 CEA/GDEA/HBEA 接口的访问失败没有在本轮换入口重试。

## 4. 使用条件与文章版本

论文页面声明 CC BY 4.0；PLOS 的政策将许可用于其发表的文章及其他作品，要求保留作者与原始来源署名。本文作者明确提供 S1 中的技术数据，同时说明 Wind 的金融和商品指标属于第三方数据，作者不能分享。因此本次可复现范围仅限该公开附件，不包括 Wind 输入或原论文完整多变量实验。[P1、P3]

推荐署名为：Wei X, Ouyang H. *Carbon price prediction based on a scaled PCA approach*. PLOS One. 2024;19(1):e0296105. DOI: 10.1371/journal.pone.0296105。另列 S1 DOI、文件 SHA、CC BY 4.0，以及“只提取日期和收盘价、未采用原作者技术指标”的处理说明。

该文章有明确版本更正：出版社将早期不完整稿件误作正式版本发布，已于 **2025-06-05** 重新发布；更正公告于 **2025-08-05** 刊出，DOI `10.1371/journal.pone.0329804`。这是发布版本错误的更正，**不是撤稿，也不是一项数据造假的认定**。本次下载来自当前论文页；本记录没有比较更正前后附件的逐字节差异。[P2]

## 5. 未达到可接入状态的候选

| 候选 | 已读程度及具体证据 | 本轮结论 |
|---|---|---|
| Wang 等，`10.1002/ese3.1380` | 已读作者全文数据段、Table 3 和附件列表；报告北京 1,250、深圳 1,921 条，数据截至 2022-06-02；Crossref 的出版社元数据声明 CC BY 4.0 | `ese31380-sup-0001-Beijing.xlsx`、`ese31380-sup-0005-Shenzhen.xlsx` 文件名存在；正常请求正文 URL 返回 403，附件点击为检索工具 InternalError，最终文件 URL 未解析成功；没有附件 HTTP 状态、文件 SHA、逐行日期或机械行数，暂不接入 |
| Sun 等，`10.1002/ese3.799` | 作者页面列出 `ese3799-sup-0001-AppendixS1.xlsx`，八试点论文；仅页面及附件元数据 | 未取得附件、未核许可与实际日期，不称可用数据 |
| Mendeley `c287n97rz5/1` | 新的作者存储库元数据称 `raw/` 有全国 CEA 日行情，声明 CC BY 4.0；主研究是企业年报 | 正常元数据和文件 API 均 403；没有核验实际文件、行数和字段 |
| Fujian，`10.3389/fenrg.2022.939602` | 全文数据段为 2017-01-09—2022-02-25、640 条，Wind/CSMAR 来源 | 样本量不到 1,000，未发现可核验独立数据附件 |
| `njuxukai/cn_carbon_emission_market_data` | 已读 README 字段说明，正常取得仓库和树元数据；存在 SQLite 文件，未下载 | API `license=null`；源码说明部分市场缺收盘价时回退均价，口径并非未经处理的官方收盘价；不采用其数据库 |

Wiley 文章对应网址：<https://scijournals.onlinelibrary.wiley.com/doi/full/10.1002/ese3.1380>。其附件的最终可用下载地址没有被成功解析，本记录不会根据常见目录格式拼出一个未经验证的地址。

全国 CEA 新候选网址：<https://data.mendeley.com/datasets/c287n97rz5/1>。旧的 `g56r5bn92t/1`、`fsdf4f4yv5/2` 和 `gmsxtcrfgt/2` 访问失败来自既有目录，本轮没有重试。

另排除两条表面相关的 PLOS 线索：`e0294269` 页面有 2025 年撤稿公告；`e0298811` 为月度数据且有 2025 年关注声明。未下载其数据。它们与上述可用候选 `e0296105` 的普通版本更正是不同事件。

## 6. 访问失败与下载修正

首次执行器网络请求受到本地沙箱 `EPERM` 限制。获得正常联网批准后进行了相同的公开元数据获取；本地权限失败不应记录成站点拒绝。其后真实 HTTP 失败包括福建和深圳官方落地页 503、Wiley 正文 403、Mendeley 两个 API 403；停止相应入口，不切换镜像、账户或绕过挑战。北京官方落地页在浏览检索工具返回 521。详细条目见 `SOURCE_ACCESS_LOG.json`。

PLOS 附件首个 HTTP 200 下载由于本方代码只读取 10,000,000 字节而被截断，不能作为完整数据。该文件 SHA 为 `d4d840dde53372148000d31f52424bd52f5e5acd29bd0e6a60f3189f3c812181`，机械解析失败。随后从同一个允许公开访问的稳定链接读取至 EOF，得到本记录固定的完整 13,625,856 字节文件。这是修正成功响应的本地读取上限，未重试被拒绝资源。原始回执和无效文件保留；公开的 `DOWNLOAD_HISTORY.json` 清楚区分两次结果，不包含临时签名重定向参数。

## 7. 公开可复现入口

需要 Python 标准库和 `xlrd`。本轮只执行了机械检查，没有执行 `extract`。

```bash
python -m pip install xlrd
python -B reproduce_plos_supplement.py download \
  --out /your/new/path/plos_0296105_s001.xls \
  --receipt /your/new/path/DOWNLOAD.json
python -B reproduce_plos_supplement.py inspect \
  --workbook /your/new/path/plos_0296105_s001.xls \
  --metadata /your/new/path/MECHANICAL.json
```

只有在后续明确决定使用该候选之后，才需要显式执行日期和价格导出：

```bash
python -B reproduce_plos_supplement.py extract \
  --workbook /your/new/path/plos_0296105_s001.xls \
  --sheet shenzhen \
  --out /your/new/path/shenzhen.csv \
  --metadata /your/new/path/EXTRACTION.json
```

三个子命令均拒绝覆盖现有输出。下载只尝试一次，不自动重试或换源，文件变化会因 SHA 不匹配而失败。检查固定工作表和列名、有效行数及日期边界；输出只保留有效日期和原数值收盘价，不制造缺失日数据，也不使用任何模型结果选择数据。再次下载与导出将是新的执行记录，不能沿用本次检查回执假称已运行。

## 8. 机制变量与本轮暴露边界

没有取得可与该深圳序列逐日对齐并经过口径核验的成交量。本轮识别过全国碳市场年度运行公告和深圳配额拍卖公告的官方页面，但只作出处和发布时间线索，未建立履约或供给特征。未来若采用公告，必须以实际公布日为可用时点，不能把事后公告中的年份标签当作当年已知信息。

严格陈述本轮暴露范围：未打开旧 C 文件，未下载或使用新候选的确认划分，未进行价格诊断或市场拟合。公开官方年度公告页面曾返回包含年度价格、交易量的文字摘要，因此不能把“未读取旧确认文件”夸写成“从未看见任何网页上的价格数字”。

## 9. 证据与文件

- **P1**：论文与 Data Availability，<https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296105>。已读正文有关来源及数据的部分、许可和附件链接；下载的 HTML 身份记入 JSON。
- **P2**：出版社版本更正，<https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0329804>。已读完整更正说明。
- **P3**：PLOS 许可政策，<https://journals.plos.org/plosone/s/licenses-and-copyright>。已读文章及其他发表作品的 CC BY、第三方材料和署名条款。
- **P4**：S1 附件稳定地址和 DOI，见第 2 节。已完整下载并通过机械检查。
- **M1**：`REPRODUCIBLE_MECHANICAL_CHECK.json`，由公开检查脚本对完整固定文件生成；`explicit_export=null`。
- **M2**：公开包中的 `DOWNLOAD_HISTORY.json` 区分截断与完整下载；原始 `PLOS_COMPLETE_DOWNLOAD_RECEIPT.json` 仅保存在发现工作区。
- **M3**：`DATA_DISCOVERY_MANIFEST.json`、`SOURCE_ACCESS_LOG.json`，记录候选、访问层级、失败和既有来源排除；范围有限，不声称全球数据穷尽检索。


## 10. 交付文字更正记录

2026-10-17T16:03:43.156972+00:00：将此前“正常下载入口返回 403”的含糊表述精确为：HTTP 403 属于 `https://scijournals.onlinelibrary.wiley.com/doi/full/10.1002/ese3.1380` 正文请求；附件点击只返回检索工具 InternalError，未取得最终文件 URL 或附件的 HTTP 状态。这是对既有 URL 级回执的表述更正，没有发起新请求。`SOURCE_ACCESS_LOG.json`、下载器、原机械检查与价格文件身份保持原字节。修改前本文 SHA256 为 `2469d45394f266db5b8ceb309353a9f5acc9bb61029ea871704526ef5cf2650e`。
