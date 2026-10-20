# EEX 官方 EUA 一级拍卖数据：来源、处理与隔离复现

本批数据用于下一轮独立研究。它包含 **2,681 个有实际拍卖结果的日期、2,695 场纳入拍卖**；其中开发集为 2,255 日，2024 年与 2025 年各封存 213 日。本数据处理任务没有执行任何市场模型拟合、调参或成绩计算。

## 1. 数据对象和一手来源

目标是 **EEX 发布的 EUA 一级市场拍卖清算价**，单位 EUR/tCO₂。它不是连续交易收盘价，也不是某个月份的 EUA 期货结算价；与旧研究中来源口径不同的 `EU_REPORTED` 应分别标识。每个 EUA 对应 1 tCO₂；拍卖采用统一清算价机制。本批仅覆盖 EEX 本报表所收录的拍卖，不表示整个欧洲二级市场或其他拍卖平台的全部成交。

权威入口：

- [EEX 官方 EU ETS1 拍卖介绍](https://www.eex.com/en/markets/environmentals/eu-ets1-eu-ets2-auctions/eu-ets1-auctions)。
- [EEX 官方拍卖设计说明](https://www.eex.com/en/markets/environmentals/eu-ets1-eu-ets2-auctions/eu-ets1-auctions/auction-design)：统一清算价及拍卖量、报价需求之间的规则。
- [EEX 官方数据下载目录](https://www.eex.com/en/market-data/market-data-hub/environmentals/eex-eua-primary-auction-spot-download)。
- [固定 2012–2025 年官方 ZIP](https://www.eex.com/fileadmin/EEX/Downloads/Markets/Environmentals/EUA_Emission_Spot_Primary_Market_Auction_Report/Archive_Reports/emission-spot-primary-market-auction-report-2012-2025-data.zip)。
- [EEX 免责声明及使用条款](https://www.eex.com/en/disclaimer)。

ZIP 为 **784,392 bytes**，SHA256：

```text
c3ff6ffaaf348cbd17f61d4726fdbd45a7ead089a0006568a490dd63bb080231
```

压缩包含 2012–2019 年 8 份 XLS 和 2020–2025 年 6 份 XLSX，主表均为 `Primary Market Auction`。2012–2019 年的 `Caption` 附表把 T2PA、T3PA 标作 EUA，把 EAA2、EAA3 标作航空配额。2021 年以后的官方拍卖名称改为第四阶段，而合约代码仍为 T3PA；这里保留来源代码，并不把它误写成固定交割月份的期货合约。

这是 2026-10-17 下载的官方历史报表快照。原报表未提供每项历史更正的首次发布时间，因此不能宣称已恢复当年实时可见的数据版本。

## 2. 先冻结后处理的边界

| 事件 | UTC 时间 / 证据 |
|---|---|
| 元数据 HEAD 结束 | 2026-10-17 09:42:07.651802；读取响应正文 0 bytes |
| 初始来源范围冻结 | 09:42:34.184591；`EEX_SOURCE_SCOPE.json` |
| 第一次且唯一一次官方价格 ZIP GET | 09:49:30.912176–09:49:33.355858；HTTP 200 |
| 合约与旧版成交证据附录冻结 | 09:56:11.049268；`EEX_SOURCE_ADDENDUM.json` |
| 准备代码、字段规则及环境冻结 | 10:00:40.902358；`EEX_PREPARATION_PROTOCOL.json` |

HEAD 实际先于范围文件，不能写成范围冻结先于 HEAD。关键区别是：HEAD 没有价格正文，第一次价格下载确实晚于范围冻结。后续合约/状态附录依据结构和机械一致性审计制定；根研究者没有在此过程中查看确认期报价、投标分布或模型成绩。

| 分区 | 固定日期边界 | 实际首末日期 | 日记录 | 原拍卖场次 |
|---|---|---|---:|---:|
| 开发 | 2012-01-01 ≤ date < 2024-01-01 | 2012-10-26—2023-12-18 | 2,255 | 2,269 |
| 封存确认 2024 | 2024-01-01 ≤ date < 2025-01-01 | 2024-01-15—2024-12-16 | 213 | 213 |
| 封存确认 2025 | 2025-01-01 ≤ date < 2026-01-01 | 2025-01-07—2025-12-15 | 213 | 213 |

2026 年不纳入。较早网页搜索偶然显示过一条 2026 年拍卖新闻中的价格，已在来源暴露记录中标记。旧上海确认集已经用于上一轮评分，不能替代本批新的确认区。本处的封存是研究流程中的文件隔离和访问纪律，不是加密或不可访问的保险库。

## 3. 纳入、剔除和身份核验

全部原表共有 2,844 条带有效日期的拍卖行。主序列只保留 **T3PA**；排除 T2PA、EAA2、EAA3 共 146 行。T2PA 与 T3PA 在 2012 年有 2 个同日出现的日期、2013 年有 6 个，故不能把二者默认为同质标的混合。

2012–2019 年没有 `Status` 列。按追加冻结规则，只有清算价、拍卖量、总收入均为正有限数，且 `abs(revenue - price × volume) ≤ 0.01 EUR`，同时整行没有取消/失败文字，才纳入；证据标签为 `inferred_completed`。所有这些年份的 T3PA 行满足条件。这个标签是对官方结果的完成证据推断，并不是伪造一个原表不存在的 `successful` 状态。

2020–2025 年必须具有原表明确的 `successful` 状态，并通过相同正值及收入恒等式检查。3 场明确取消的 T3PA 拍卖被排除，日期为 2020-03-17、2022-02-01、2022-03-02；其清算价及收入为空。取消拍卖不作为价格为零的观察。旧无状态报表未出现取消行，不代表当年从未发生过取消，而是这些报表可用记录的范围。

去重身份为原表的 `(date, time, auction_name, contract)`。实际没有重复身份；复现代码仅接受内容完全一致的重复，冲突重复会停止，不能任意保留一行。所有原始主表行和逐场纳入结果留在本地，可追溯到年度文件、工作表和 Excel 行号。

## 4. 日聚合与字段定义

若同日有多场纳入拍卖，记其清算价为 `p_j`，已报告拍卖量为 `v_j`，总投标量为 `b_j`。日价格为 `sum(p_j × v_j) / sum(v_j)`。本批只有 2013 年的 14 个日期需要合并，每日两场；其他纳入日期均为一场。

| 日 CSV 字段 | 定义及处理 |
|---|---|
| `date` | 官方拍卖日期，升序、唯一；不添加无拍卖日期 |
| `price` | 拍卖量加权的清算价，EUR/tCO₂ |
| `auction_volume` | 纳入场次的官方拍卖量之和，不使用总投标量作权重 |
| `total_bid_volume` | 原表 `Total Amount of Bids` 之和；该字段位于报表的 Volumes 分组，与投标笔数分列 |
| `cover_ratio` | `sum(total_bid_volume) / sum(auction_volume)`；重新计算，不平均原表已四舍五入的覆盖倍数 |
| `bidder_count` | 原表总投标人数的拍卖量加权平均；不是同日去重后的自然人/机构总数 |
| `successful_bidder_count` | 原表成功投标人数的拍卖量加权平均，解释同上 |
| `auction_count` | 当日纳入场数 |
| `contract` | 固定为 T3PA |
| `auction_names` | 原表拍卖项目名称的排序集合；所有来源场次均在 EEX，名称不等于不同交易所 |
| `completion_evidence` | 原始明确状态或旧表完成证据推断的标签 |
| `missing_fields` | 如任何组成场次的可选字段缺失，该日对应聚合字段也缺失；不填 0 |
| `component_ids` | 逐场身份索引，用于回溯组件 CSV |

所有纳入日的这些核心字段在当前快照中完整。没有删除异常值、平滑、插值、向后填值或补交易日。数值由原单元格缓存值转成十进制表示，聚合使用 50 位十进制精度；不会因显示格式把价格再舍入一次。逐场文件另保留原表的 `reported_cover_ratio`、总收入及全部来源定位字段。

这些量是本次拍卖结束后才形成的结果。在以某个过去日期为预测起点时，只能使用截至该起点已经完成的拍卖；未来实现的拍卖量、投标量、覆盖倍数、人数和未来实际间隔都不能当作已知特征。预测步长应解释为未来有结果的拍卖日记录，而非固定自然日。此准备程序不构造未来日历特征。

## 5. 实际验证与公开交付范围

准备前 9 组合成检查通过，覆盖成交证据、取消与金额恒等式、严格合约筛选、已知双拍卖加权、缺失传播、重复冲突、分区边界及十进制输出。

随后另一个代码实现直接重新读取 XLS/XLSX，使用精确分数从源记录复算全部逐场与日数据，不导入准备器、结构检查器或下载器。**53,760 次数值与元数据比较通过**。这是独立算法实现的来源核验，不是第二次训练，也不暗示已由另一个研究者完成独立盲审。相关回执只记录日期、数量、通过状态和哈希，不包含确认价格统计。

EEX 条款不是 CC BY。其免责声明对个人非商业复制和公开再利用有明确限制；本包不提供原始或派生报价的再分发许可。ZIP、完整源行、组件 CSV 和日 CSV 均留在 `local_only/`。可交付获取与处理代码、官方链接、来源/结果哈希和研究者自己的合规汇总结果；不得把历史其他 CC BY 数据的许可套到本来源上。实际对外使用仍应遵循 [EEX 原始条款](https://www.eex.com/en/disclaimer)。

第一次结构检查误把部分开发期 `Auction Details` 收益文字归入结构 JSON。原字节已移到本地私有目录，另存修正收据；公开结构审计已换成仅表头、日期、计数、状态与布尔核验信息。没有确认报价在该次输出中显示。

## 6. 无绝对工作区路径的复现

保留本附录所在目录的完整代码及 `EEX_REPRODUCTION_SOURCES.json`，可从任意本地路径执行。代码/协议 allowlist 验证固定字节与 SHA256；结果目录必须此前不存在，不会覆盖旧结果。下载只访问上列固定官方 ZIP 一次，不换代理、不重试、不自动改用新版本。若官网以后替换同名文件，哈希检查将停止，不能默默改变样本。

实际解析环境为 **Python 3.12.14、xlrd 2.0.1、openpyxl 3.1.5、et_xmlfile 2.0.0**，无需 PyTorch、预训练模型、API key 或私人凭据。可以在新的独立环境安装本目录的固定依赖；本次交付期间没有重复安装环境。

```bash
python3.12 -m venv /path/to/new_eex_venv
/path/to/new_eex_venv/bin/python -m pip install -r /path/to/eex_bundle/requirements-eex.txt

/path/to/new_eex_venv/bin/python -B /path/to/eex_bundle/reproduce_eex_data.py check

# 已有同字节官方 ZIP：完全本地复现。
/path/to/new_eex_venv/bin/python -B /path/to/eex_bundle/reproduce_eex_data.py prepare --archive-local /path/to/eex_primary_auction_2012_2025.zip --output /path/to/new_eex_workspace

# 或从固定官方入口获取一次；不要同时运行上面的本地命令到相同输出目录。
/path/to/new_eex_venv/bin/python -B /path/to/eex_bundle/reproduce_eex_data.py prepare --download --output /path/to/another_new_eex_workspace
```

给 `prepare` 增加 `--dry-run` 可校验代码和计划；本地模式还校验 ZIP 哈希。dry-run 不创建输出目录、不访问网络、不解析价格。

入口把运行代码和历史冻结协议复制到新目录的 `runtime_source/`，把报价留在 `local_only/source/` 与 `local_only/prepared/`。随后执行同一准备程序、比对六份输出 CSV 的已知哈希，并执行精确分数核验。确认数据仍放在独立 `sealed_confirmation/` 子目录；数据准备成功不等于获准开展确认期选模或评分。

轻量复现入口检查可单独运行，完全不读价格档案：

```bash
python3 -B /path/to/eex_bundle/check_eex_reproduction.py --output /path/to/new_source_guard_receipt.json
python3 -B /path/to/eex_bundle/check_eex_preparation.py --output /path/to/new_synthetic_receipt.json
```

本轮没有新增市场拟合。后续模型、超参数、评价方法和打开确认分区的条件由研究主协议单独冻结。
