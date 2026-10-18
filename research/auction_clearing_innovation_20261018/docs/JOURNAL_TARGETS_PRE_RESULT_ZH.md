# 随机拍卖清算预测研究：主投稿与两项备选

核查日期：2026-10-18。当前状态：方法和独立数学检查已可交接，真实预测结果由主实验提供。本文件没有据期刊名替代创新性判断，也不承诺录用、分区或所有情形免收费。

## 1. 明确推荐

**主投稿：Journal of Forecasting。备选一：International Journal of Forecasting。备选二：Expert Systems with Applications。**

这是一套按当前论文贡献组织的选刊建议，不是录用难度排序。Journal of Forecasting 适合把问题写成“小中样本、已公告供给与未观测需求下的结构预测学习”；International Journal of Forecasting 适合在结果足够稳健、贡献能够超出单一碳拍卖数据集时考虑；Expert Systems with Applications 适合突出可实现的随机清算学习算法、同信息强对照和完整消融。不能因为主刊认为方法增量不足，就把同一缺陷原封不动地转投备选。

| 角色 | 官方范围中与本研究直接相关的部分 | 本研究应交付的实质内容 | 当前主要风险 |
|---|---|---|---|
| 主投稿：Journal of Forecasting | 多学科预测，覆盖理论、实践、计算和方法问题。[J1] | 随机清算诱导的价格分布、与直接预测的准确数学边界、严格时间外评价及何时有收益 | 单一市场上的弱增益，或收益完全来自新增竞价字段，不能支持模型创新 |
| 备选一：International Journal of Forecasting | 覆盖各类预测，强调实证评价、实现与预测实践改进；明确包括金融、能源、机器学习和预测方法。[J4] | 跨时期可靠的预测证据、目标/评分一致、数据时点与评估可复现；共享结构的约束必须有直接对照 | 仅更换网络解码器而缺乏一般预测见解；这一备选并不意味着更低的方法要求 |
| 备选二：Expert Systems with Applications | 智能系统的设计、开发、测试和应用，领域包括经济、金融、能源；官方明确反对给已有概念改名。[J6] | 清晰的学习问题与算子、与相同信息/相同辅助监督直接模型的比较、计算成本和数值稳定性 | 多组件堆叠或仅称首次用于碳价；这些不能替代形式差异与实验贡献 |

上表的匹配顺序和风险判断是研究判断，不是期刊编辑部的预审意见。

Journal of Forecasting 曾发布“Forecasting Carbon Prices in an Integrated Market Network”专题征稿，说明碳价格预测有直接主题契合证据；该文件截止日期为 **2025-05-30，已经过去**，不能作为当前仍开放的投稿通道。[J10]

## 2. Applied Energy 为什么暂未列入三个目标

Applied Energy 的官方范围明确包含能源相关建模、预测、决策、市场设计与运行，并要求侧重能源应用。[J7] 因此不能说碳价预测不在其范围内。近期可微/单调竞价曲线近邻也已经发表于该刊，见 `METHOD_FOUNDATIONS.md` 的 R7、R9。

当前研究的中心问题是预测学习规则，而已授权的首轮实验是单一拍卖目标与结构消融。就这篇尚未有真实结果的稿件而言，先按预测方法论文写更一致。如果后续证据自然形成具体的能源市场应用价值，Applied Energy 可以重新考虑；不应为选刊临时添加没有数据支撑的减排效应、政策因果结论或采购收益。这个取舍是当前稿件定位的判断，不是该刊范围排斥。

## 3. 出版模式与费用：已核实和未核实分开记录

| 期刊 | 已核实的订阅 / OA 路径 | 本次能确认的作者费用信息 | 还不能承诺什么 |
|---|---|---|---|
| Journal of Forecasting | 官方作者指南明确为订阅期刊，可在接受后选择 OA。[J2] | 官方明确无投稿费；选择 OA 时当前列示 **USD 3,750 / GBP 2,500 / EUR 3,130**，未含适用税费；机构、基金或学会协议可能覆盖部分或全部。[J2–J3] | 标准订阅路线不需购买这个可选 OA APC，但不能由此保证所有可选服务或其他情形均无费用；也不能保证作者有协议资格 |
| International Journal of Forecasting | 出版社官方页面有机构订阅入口，并指向个刊 OA 政策。[J4] | Elsevier 官方区分由订阅读者付费的订阅文章与由作者/机构/基金支付 APC 的 OA 文章。[J8] | 本次未成功核实该刊当前精确 APC、可能的投稿费及所有附加费用，不填第三方数值 |
| Expert Systems with Applications | 出版社官方页面有订阅入口，并指向个刊 OA 政策。[J6] | 适用 Elsevier 的订阅 / OA 收费区分；协议减免取决于实际资格。[J8–J9] | 本次未成功核实该刊当前精确 APC 和全部个刊附加费用，不能写“保证免费发表” |
| Applied Energy（范围比较） | 出版社官方页面有订阅入口，并指向个刊 OA 政策。[J7] | 同样区分订阅与 OA APC。[J8] | 本次未核实精确 APC；不以第三方报价作为选刊证据 |

Elsevier 官方说明 APC 会变化，按期刊及可能的文章类型制定，并在个刊页或中央清单提供价格。[J8] 本次个刊 ScienceDirect 指南多次无法读取；从官方价格页指向的 XLSX 清单请求只返回了 195 字节的“Site Unavailable”HTML。该响应已准确保留为 `source_documents/elsevier_apc_download_response_20261018.html`，没有把它当成有效工作簿解析，也没有据此填写报价。查不到精确金额不影响当前完成模型研究和稿件准备；实际选择 OA 时再核实当时个刊报价与资助要求。

费用控制的可执行选择是：在作者资助要求允许的前提下，优先考虑期刊提供的标准订阅发表路线；如要求 OA，则按接受时的正式报价和机构协议处理。这里不替用户付款、选许可或提交论文。

## 4. 当前稿件应怎样限定贡献

可暂用工作题目 **Learning Stochastic Auction-Clearing Distributions from Public Bid Summaries**。这只是描述研究对象，尚不是经过实验支持的创新结论。

论文的核心问题应是：在只观察公开汇总、无法恢复私人完整订单簿的条件下，能否通过一个低容量随机需求法则，把未来总投标量的辅助学习与已公告供给驱动的价格分布绑定起来，并改善真正的下一场价格预测？

理论部分须同时保留正面性质和反例。正面性质包括非负清算质量、随机总量积分、正确的成功后混合、正价格 RW 嵌入以及同一订单实现上的供给单调性。边界包括：单一供给下的 Gamma 清算可重参数化任意正离散 CDF；汇总量不点识别完整曲线；条件成功均值不由路径单调性自动保证单调；仅 3 个真实取消不支持识别保密筛选；当前复合边际损失不是联合似然。详见 `METHOD_FOUNDATIONS.md`。

这组边界不会自动消除研究价值，但使创新主张只能落在共享参数化与估计问题、可复现的结构学习收益上。DSAA 2023 的 P&O 已有端到端潜在订单簿清算，且其纯 P&O 不要求订单簿监督；随机订单产生清算分布也已有原始论文。论文不能以“可微清算”“没有真曲线标签”“Gamma”或“首次应用于碳价”单独声称创新。

真实实验需要回答的核心问题仍然只有一个：该结构是否在统一预测原点、同信息和同监督预算下，稳定优于 RW、普通直接模型、同多任务分量内去清算模型与确定性近似？价格 MSE 是已冻结的主目标；CRPS 或 B 辅助任务改善不能替代主目标结论。取消状态概率只是补充诊断。发表判断要等待这些真实结果，本文件不预先把数学 PASS 写成投稿通过。

## 5. 复现与数据材料的投稿准备

Journal of Forecasting 的作者指南要求数据可用性声明并期待数据共享；也允许已发布预印本的稿件。[J2] IJF 官方出版者页面说明相关数据和程序作为在线补充材料提供；IIF 官方页面搜索结果进一步声明，自 2023 年 7 月起数据和代码须提供，例外需向主编作可接受的解释。[J4–J5] 后一条本次读到官方搜索索引内容，但直接页面未能打开，提交前应再核其当前表述。

针对本项目，宜提交代码、解析规范、文件哈希、可得时点规则、冻结协议、全部原点预测和可合法分享的派生结果。原始 EEX 表格的公开可读性不自动等于再分发授权；受限制的原文件应提供官方获取路径及复跑说明。不能为满足共享要求重命名或匿名复制受限原始数据。关于历史版本缺口和偶然搜索暴露，按照 `research_planning/data/RESEARCH_SOURCE_HANDOFF_20261018.md` 与 `EXPOSURE_LEDGER_20261018.json` 如实披露。

## 6. 官方来源与核查深度

- **J1，Journal of Forecasting 范围**：[Wiley 期刊主页](https://onlinelibrary.wiley.com/journal/1099131x)。官方页面已读取。
- **J2，Journal of Forecasting 投稿费、订阅 / OA、数据与预印本规则**：[Wiley Author Guidelines](https://onlinelibrary.wiley.com/page/journal/1099131x/homepage/forauthors.html)。官方全文已读取。
- **J3，Journal of Forecasting 可选 OA 金额、税与协议**：[Wiley Open Access / APC](https://onlinelibrary.wiley.com/page/journal/1099131x/homepage/fundedaccess.html)。官方全文已读取，金额仅代表核查时页面。
- **J4，IJF 范围、订阅入口与复现材料**：[Elsevier 官方期刊介绍](https://shop.elsevier.com/journals/international-journal-of-forecasting/0169-2070)。官方全文已读取。
- **J5，IJF 数据代码政策**：[International Institute of Forecasters 官方 IJF 页](https://forecasters.org/ijf/)。官方搜索索引可读，直接访问失败；没有把该页视为已完整读取。
- **J6，ESWA 范围与反对概念改名的说明**：[Elsevier 官方期刊介绍](https://shop.elsevier.com/journals/expert-systems-with-applications/0957-4174)。官方全文已读取。
- **J7，Applied Energy 范围与订阅入口**：[Elsevier 官方期刊介绍](https://shop.elsevier.com/journals/applied-energy/0306-2619)。官方全文已读取。
- **J8，Elsevier 定价和 APC 规则**：[Elsevier Journal Pricing Policy](https://www.elsevier.com/about/policies-and-standards/pricing)。官方全文已读取；其[中央 APC 工作簿链接](https://legacyfileshare.elsevier.com/els_com_pricing/article-publishing-charge.xlsx)下载失败，不能用于确认个刊金额。
- **J9，Elsevier 开放获取选项及协议**：[Elsevier Open Access](https://www.elsevier.com/open-access)。官方页面已读取，具体资格未替作者核验。
- **J10，已经截止的碳价专题征稿**：[Wiley 官方 PDF](https://onlinelibrary.wiley.com/pb-assets/assets/1099131x/Proposal-for-SI-JoF-07May24-1734467232.pdf)。官方 3 页 PDF 已读取；只证明历史主题匹配，不是当前开放征稿。

本文件没有核实或宣称任何期刊当前分区，也没有用影响因子、网站平均接受率或平均处理时长推算本稿录用概率。
