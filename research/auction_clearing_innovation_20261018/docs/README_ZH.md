# 研究文档的时间状态

最终解释以研究根 `REPORT_ZH.md` 及三轮 `results/independent_metrics/` 为准。

| 文档 | 阅读方式 |
|---|---|
| `METHOD_FOUNDATIONS_PRE_RESULT.md` | 原 SDMC 方法提案及独立数学分析；“建议运行”是当时状态，不能覆盖后来失败结论。原文按字节保留。 |
| `V2_FACTORISATION_REVIEW_PRE_RESULT_ZH.md` | v2 实现前审查建议；实际线性 basis、位置/跨度边界和损失以 `scale_separated_revision/PRE_RESULT_SPEC_ZH.md` 与冻结代码为准。 |
| `JOURNAL_TARGETS_PRE_RESULT_ZH.md` | 最初选刊依据与费用核查；后来实验未通过，不能把选刊建议理解为已可投稿。 |
| `PUBLICATION_CLAIM_REQUIREMENTS_ZH.md` | 完成评分后的声明边界：两版均停止，尺度条件不足以支持模型创新。 |
| `SINGLE_ROUTE_PROPOSAL_ZH.md` | 唯一未来提案，新增二级价格和部分数量观测均未落实；不是已运行的第三个模型。其中二级价格记为 R，主报告为避免与指标 R 混淆记为 A。 |
| `HISTORICAL_ROUTE_AND_TASK_DIAGNOSIS_ZH.md` | 旧项目任务和失败路线的回顾，只用于防止重复研究；不同任务的分数不拼接排名。 |
| `CONFIRMATION_EXPOSURE_STATUS_ZH.md` | 既有确认年份接触和评分状态，保留精确的可声称范围。 |
| `LITERATURE_FULLTEXT_ACCESS_RECEIPT.json` | Steinbakk 与 Sinha 原始全文的实际取得、SHA 与阅读范围；全文通过一手链接取得，不在本包复制。 |
| `check_random_clearing_math.py` / `MATH_CHECK_RECEIPT.json` | 不读取市场数据的 15 项理论/模拟检查；数学 PASS 不是预测优势或新颖性 PASS。 |

较早文档中的 `research_planning/` 和绝对路径是形成该记录时的来源位置，可能不在公开包中。公开包保留可审查的方法文字、代码、来源 URL 和汇总收据；复现实验使用主目录的可迁移入口，不要求恢复这些旧绝对路径。

