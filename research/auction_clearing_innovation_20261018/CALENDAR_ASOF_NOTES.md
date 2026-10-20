# 开发期拍卖计划的版本与使用边界

本表由官方 EEX 历史日历中的 2017—2023 年 XLSX 日历首页解析，包含 32 个可辨认版本、6,608 条 T3PA 计划记录。重复版本行不是新增独立训练样本。2012—2016 年原件已获取，但本次未将缺乏可靠日期或尚未完成解析的 PDF 硬转为计划表。

## 输入与输出

- 解析器：`parse_calendar_vintages.py`。可用 `--source-root` 指定历史文档根目录；默认查找工作区 `research_planning/data/source_documents/calendar_vintages_2012_2023`。
- 数据：`local_only/calendar_vintages_development.csv`。
- 逐版本来源、SHA-256、拒绝原因与覆盖：`local_only/CALENDAR_VINTAGE_COVERAGE.json`。
- 日期切换、缺口及删除拍卖的实测校验：`validate_calendar_asof.py`、`local_only/CALENDAR_ASOF_VALIDATION.json`。

| 日历年 | 采用版本数 | 含重复版本的计划行 |
| --- | ---: | ---: |
| 2017 | 5 | 996 |
| 2018 | 5 | 1,040 |
| 2019 | 5 | 933 |
| 2020 | 6 | 1,260 |
| 2021 | 2 | 446 |
| 2022 | 7 | 1,503 |
| 2023 | 2 | 430 |

## 日期证据的强度

`revision_date` 来自官方工作簿的 Document History；2017 年第一版以稍后官方版本明确记载的初次发布日期佐证。`revision_available_date` 统一为该日期加一自然日，不假定公告当天可用。`publication_date_is_explicit=True` 表示文内有明确版本日期证据，不表示已经取得当年的网页抓取时间戳。来源行和证据说明保存在每条记录中。

两份 2023 日历还与官方带日期的客户公告一致：[2022-07-28 原始发布](https://www.eex.com/fileadmin/Global/News/EEX/EEX_Customer_Information/2022/20220728_CI_Auction_Calendar_2022_and_2023.pdf)、[2023-06-21 修订发布](https://www.eex.com/fileadmin/Global/News/EEX/EEX_Customer_Information/2023/20230621_CI__Publication_of_revised_Emissions_Auction_Calendar.pdf)。其余日期主要依赖 EEX 文内版本历史，而非逐一认证过的独立历史网页快照。档案于 2026 年重新打包的时间不是这些版本的首次发布时间。

## 必须执行的整版选择

对每个预测起点和日历年，先选 `revision_available_date <= origin` 的最新完整年度版本，再过滤 `scheduleddate > origin`，同时要求 `origin < valid_until_exclusive`（空值表示无已知后续版本边界）。可直接调用 `select_asof_plan(rows, origin_date, calendar_year)`。不能对每个拍卖日期单独回填最后一条记录，否则会把新版已经删除的未来拍卖复活。

`platform` 统一为 CAP、DE、PL、NI，保留 `auction_name` 和 `platform_raw` 原文。连接开发期事件优先使用日期与平台，原名称作核对信息。计划量单位为 EUA，不将 CAP2/CAP3 制度变化解释为独立资产。`volume_is_indicative` 仅在来源明确说明九月至十二月计划量仍待调整时置真；它不承诺未标记者的日期或拍卖结果必然实现。2018 年德国合约结束后的日期曾为临时安排，后续被删除。

## 主动保留的缺口

2019-06-25 的取消修订在官方历史中存在，但对应完整版本未在本次档案中取得，因此从 2019-06-26 至 2019-07-11 不返回 2019 计划。

两份 2020 工作簿的最新历史都记为 2020-03-17，但其中一份文件名为 2020-04-03，且其脚注及至少一个平台名称发生变化；没有充分证据把这份更晚文件的内容追溯到三月。解析器排除同日冲突组，从 2020-03-18 至 2020-07-01 不返回 2020 计划。

2018 年 2017-11-27 的初版没有对应采用的工作簿，首个可用版本是 2017-11-28 的修订；2017 年 2016-12-01 只纠正表头年份的版本不构成计划内容失效。

未知年份、未知发布日期和已知缺失版本应形成缺失指示或限制所用样本，不能改用事后实际成交量、成功拍卖间距或尚未发布的最终年度计划填补。最终失败或取消的拍卖也不能用于提前挑选预测目标日期。本脚本不读取价格数据或 2024/2025 日历内容。

官方历史档案：[EEX 历史拍卖日历 ZIP](https://www.eex.com/fileadmin/EEX/Downloads/Trading/Calendar/Auction_Calendar/20260217_History_EUA_Primary_Auction_Calendar_2011-2025.zip)。本次只提取其中 2012—2023 年的 135 个成员，逐个 CRC 检查；本次表只消费其中 2017—2023 的 34 份 XLSX，排除上述两份冲突文件。
