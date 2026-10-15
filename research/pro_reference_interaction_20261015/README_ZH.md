# Stage I：参考状态与循环记忆的锚定交互

**科学裁决：STOP_DEVELOPMENT_GATE_FAILED。完整源码与完整远端数值材料已进入本分支；未合并main。**

唯一候选MIXED没有超过普通ADDITIVE/FULL和简单预测组合。本地先完成固定实验，随后GitHub新环境完整重训；两端都失败，但不能称逐值复现：候选最优lambda从10变为0，80个汇总指标最大差0.0026273，624次拟合中25次内部验证checkpoint选择不同。不要用较好的远端结果覆盖首轮结果，也不通过再训练寻找一个更好版本。精确记录见`REPRODUCTION_CHECK.json`。

## 先读哪些文件

- `REPORT_ZH.md`：本地首轮数学、全部配对结论和范围；其表格不是远端表。
- `REPRODUCTION_CHECK.json`：本地和远端的具体差异、环境、执行收据与解释限制。
- `local_first_summary/all_scores.csv`：首次本地全部40配置的主分数及集成分数。
- `results/RESULTS.json`、`results/all_scores.csv`：远端完整重放自己的结果，明确其选参。
- `PROTOCOL.json`：新预测成绩出现前冻结，提交49721fcf833ce1ffd14f3317661d0c9b62c2677d。
- `SOURCE_LEDGER.json`、`SOURCE_ROUNDTRIP.json`、`EXPOSURE_LEDGER.json`：实际阅读层级、执行源字节身份与价格访问边界。
- `VISIBILITY_NOTICE.json`：当前GitHub元数据标记仓库公开；本轮未改变可见性。原协议的private措辞在此更正。

## 实际执行

源main为c024f140e28753bcec47f82b44300787b7f35c6b。首次本地624次新神经拟合、180条传统配置刷新记录、4条新全参数在线GRU轨迹；40个配置预测档案。远端按相同源码、同数据、同种子与预算重新完成同一实验。

每个环境中20项测试的普通/优化模式通过；624份保存参数全部精确重建该环境内的预测；独立算术重算80个主/集成指标；5次代表性完整未来扰动重训通过。跨环境不一致原因未被逐项隔离，不能把它归因于某一个已证明的软件/硬件因素。

只读取现有广州作者CSV前950条、截至2021-12-31。不请求下一行、不全文件hash、不新下载行情；2020/2021仍是反复暴露开发区。2022—2024、湖北/EU不建模。作者价格、版本和发布时间限制继承原合同；源行号和Excel格式不作特征。

## 完整数值材料已提交

`results/`含1315个结果文件（manifest另列其余1314个）：624份拟合NPZ、624份逐轮JSON、40份完整配置预测、在线过程与权重、参考状态、全配置/逐年逐期限/逐种子表、独立审计、环境、远端执行日志和哈希清单。它们已在Git提交7692853e4ae33e35f15bee413ee6ca620312b757保存，不依赖临时artifact有效期。

远端运行：38681433247，执行源码提交88de468765b9a119f1b176a9656833d3952f4016。artifact11948413214同时保留完整快照；artifact已下载做CRC与1314项哈希复核。完整本地首轮参数、预测与长日志另在对话交付ZIP的local_first中原样保留，不伪称它们与远端文件相同。

## 复现

在独立工作副本/虚拟环境运行，避免改动原有实验环境：

```bash
python -m pip install -r research/pro_reference_interaction_20261015/requirements.txt
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python research/pro_reference_interaction_20261015/run_all.py --out /path/to/NEW_results --workers 3
```

本地Python3.13.5，远端Python3.13.16；两端torch2.10.0+cpu、numpy2.3.5、scipy1.17.0、sklearn1.8.0。版本一致不代表跨环境位级结果保证。本任务不再运行调参扫描。

输出目录必须不存在。`run_all.py`完成测试、全部新训练、评分和审计，不覆盖首轮结果。`compare_remote.py LOCAL REMOTE output.json`比较两个已完成目录，不训练、不选新参数。实验代码本身无网络访问，既有价格文件可由严格前缀读取器使用。

## 解释边界

mean-seed loss与prediction ensemble分开；八网络快慢集成不等于四网络预算。FULL=ADDITIVE+MIXED只对相同固定参数成立。锚定主效应消除不是总体正交，不识别真实经济状态。统计神经混合、结构化交互和anchored decomposition均已有研究。本轮不将FULL对照或成熟误差组合改名为新模型，也未达到独立确认或发表资格。
