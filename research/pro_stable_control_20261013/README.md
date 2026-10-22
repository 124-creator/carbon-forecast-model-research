# Stage D：低维稳定估计与单一规则检验

入口：[完整中文裁决与数学定义](REPORT_ZH.md)。数据来源保持 `main@fddf3e69` 的 `research/pro_data_packet_20261013`，不重新采集价格。

## 已执行结论
- 最小合成控制通过；有信号时低维ML相对RW改善10.9208%，恢复约90.605%的参数oracle预测增益。
- 唯一候选为剖面似然集合上的联合预测投影；其通用优化与Magging等价。
- 广州2020—2021调参最优为阈值0，等同普通低维ML；对RW改善5.0138%但对Cstar新增收益0。
- 停止在调参筛选；2022验证、2023历史段和其他两个市场都未评分。不进入结果论文。

`RESULTS_SUMMARY.json` 保存主要数值，`market_tuning_all.csv` 保存全部68组配置；详细逐样本数组及日志见私有Actions运行38348341926的artifact和下载交付包。artifact保留14天，之后仍可由代码和原CSV完整重现。

## 复现
```bash
python -m pip install -r research/pro_stable_control_20261013/requirements.txt
python research/pro_stable_control_20261013/run_all.py --root . --out /tmp/stageD-replay
```
输出目录必须不存在。程序有阶段日志和状态文件，绝不自动打开验证或新历史段。17项测试在本地及远端、普通及优化模式均通过。远端完整重放的68组分数、68份预测数组及两阶段合成详细记录与本地逐值一致，见 `REPRODUCTION.json`。CI通过表示执行成功，不表示科学候选过关。

## 来源和限制
清单注明Yingying Fan (2025)，DOI 10.6084/m9.figshare.28682348.v1，CC BY 4.0。作者冻结快照不等于交易所历史定价或实时版本认证。source_excel_row、source_date_kind只供追溯，不进入特征。文献核收、Magging阅读范围和实现修正分别见 `LITERATURE_SCOPE.json`、`IMPLEMENTATION_CHANGE_LOG.json`。
