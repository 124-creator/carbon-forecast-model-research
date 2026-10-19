# 阶段 g 复现入口

先读 REPORT_ZH.md 和 PLAN_BEFORE_RUN_ZH.md。当前是旧数据上的已完成开发实验，不是天津/CEA原快照的新确认。

GitHub中直接提供 `source/` 的实际模型、训练与合成检查源码，以及核心汇总。完整补充材料位于本次另行交付的 `carbon_stage_g_evidence_20261019.zip`；解压可获得 audits/、utilities/、results/、original_summaries/ 和全部来源/执行记录。没有原数据，不能仅靠汇总复跑市场成绩。

## 环境

本轮：Python 3.13.5，numpy 2.3.5，scipy 1.17.0，pandas 2.2.3，scikit-learn 1.8.0，torch 2.10.0+cpu。需有相应可用环境；未在其他版本承诺位级重现。

## 数据恢复

从本项目原有私有归档 carbon_distribution_update_private_run_20261019.zip 恢复指定四个CSV，保持原字节。这个归档已经存在于作者Library，本轮未把它重新公开。

```bash
python utilities/prepare_archive.py --archive /path/to/carbon_distribution_update_private_run_20261019.zip --out /path/outside-public-repo/data_g
python source/test_model.py
```

## 从头运行

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python source/run_experiment.py --data-dir /path/outside-public-repo/data_g --out /path/outside-public-repo/g_new_run --workers 2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python audits/audit_and_summarize.py --run /path/outside-public-repo/g_new_run --data-dir /path/outside-public-repo/data_g --out /path/new_audit_results
python utilities/export_native_metrics.py --run /path/outside-public-repo/g_new_run --data-dir /path/outside-public-repo/data_g --out /path/new_audit_results/NATIVE_METRICS.csv
```

输出目录必须新建且为空；不要覆盖旧结果。正式三份source源码SHA在SOURCE_FREEZE.json和运行协议中。审计与恢复/导出辅助程序是之后单独添加的代码，不混入训练代码冻结计数。

## 包内记录

- results/SUMMARY.json：全部均值、市场分组、两块长八比较区间与计数。
- results/AUDITED_SCORES.csv：88个单元×模型记录。
- results/NATIVE_METRICS.csv：264个单元×模型×期限的MSE/RMSE/MAE/MAPE/价格水平R²。
- results/CONFIGURATIONS.csv：381个梯度模型选择/最终任务的配置和初始/最后/所选损失摘要，不是全部逐epoch曲线。
- original_summaries/*/RESULTS.json：全部8单元，包含Ridge/HGB全部网格、每模型全部原始成绩和实际选择。
- results/FINAL_STRUCTURAL_STATES.json：所有最终结构模型数值参数，warm重复者明确指向同一模型。
- results/REPLAY_AUDIT.json：原私有逐点数组及完整训练日志的SHA与最大重放差。
- SOURCE_FREEZE.json / RUNTIME_PROTOCOL.json：输入、切分和执行身份。

完整逐epoch曲线文件名 FULL_TRAINING_HISTORIES.json.gz，SHA256 f3b9116cdf966c1b985ffa78c1a212f32df59fec5ba3c2ce6952d2ada8cbb46e。它是本轮另行提供的无原价、无模型权重的训练曲线附件，未嵌入GitHub复现包；从头运行也会自动产生各单元的完整日志。

附件包中的PUBLIC_MANIFEST.json列出支持文件哈希。参数、误差摘要不能替代原数据；没有原归档不能仅靠摘要重现市场成绩。
