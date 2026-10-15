# Stage I：参考状态与循环记忆的锚定交互

科学裁决：**STOP_DEVELOPMENT_GATE_FAILED**。这不是结果论文；唯一候选MIXED没有超过简单对照。普通FULL混合与简单预测组合保留为强基线，不改名为新模型。

## 已运行的本地首轮
源main：c024f140e28753bcec47f82b44300787b7f35c6b。协议先于新预测成绩提交49721fcf833ce1ffd14f3317661d0c9b62c2677d。
624次新神经拟合，180个传统模型配置刷新记录，4条新全参数在线GRU轨迹；40组完整预测档案。20项测试普通/优化模式通过；624份参数全部重建预测一致；独立算术重算80个主/集成指标；5次代表性完整未来扰动重训通过。

只读取既有广州作者CSV前950条，截止2021-12-31；不请求第951条，不全文件hash，不新下载行情。2020/2021仍为反复暴露的开发区。2022—2024、湖北/EU均不读取、不评分。作者数据的产品、版本、发布时间及命名限制继承原数据合同；不把源行号和Excel格式当特征。

## 复现
请在独立工作副本/虚拟环境运行，避免改动已有研究环境。

```bash
python -m pip install -r research/pro_reference_interaction_20261015/requirements.txt
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python research/pro_reference_interaction_20261015/run_all.py --out /path/to/NEW_results --workers 3
```

本地执行环境Python3.13.5、torch2.10.0+cpu、numpy2.3.5、scipy1.17.0、sklearn1.8.0。输出目录必须不存在；脚本拒绝覆盖首次结果。所有依赖只在用户选定环境安装，实验本身无网络调用。

## 材料位置
- `PROTOCOL.json`：运行前冻结的规则和判据。
- `model.py`：完整31成员因果参考库、模型和严格前缀读取。
- `experiment.py`：全部新训练、基线、在线轨迹、成熟反馈组合。
- `analyze.py`：固定主指标、全配置和逐年逐期限评分、配对块区间。
- `audit.py`、`test_model.py`：独立算术、参数重建、代表重训和20项测试。
- `SOURCE_LEDGER.json`、`REPORT_ZH.md`：实际文献阅读层级与裁决。
- `results/`：GitHub工作流重放后提交的完整数值预测NPZ、每次拟合参数NPZ、逐轮JSON日志、在线权重、全配置表、审计与哈希清单。不是只提交摘要。
- GitHub Actions artifact同时保留本目录完整快照；其保留期限不能替代Git内的结果文件。

工作流只由本研究分支的`REQUEST_REPLAY.txt`变更触发；它只向本分支的本实验结果目录提交，不合并main。远端是否成功及执行SHA以工作流状态和`results/REMOTE_EXECUTION.json`为准，不把CI成功当作创新成功。结果目录存在时不要直接重跑同一工作流覆盖它。

## 关键界限
mean-seed loss与prediction ensemble分开。FULL=ADDITIVE+MIXED只对相同固定参数成立，不等于分别训练得到的预测相同。零锚点主效应消除不是总体L2正交，不识别真实市场状态。统计神经混合、结构化交互、anchored decomposition已有文献，不宣称这些部件首次提出。
