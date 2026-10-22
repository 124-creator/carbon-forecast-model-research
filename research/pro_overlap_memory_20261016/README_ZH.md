# Stage K：重叠依赖神经检索

**当前科学裁决：STOP_DEVELOPMENT_GATE_FAILED。** 唯一候选OVERLAP不是已经成立的新模型。首轮主相对MSE 0.9283070903，最接近DIAGONAL 0.9283481586，不带检索FULL 0.9272222496。新增机制没有独立优势。不打开后续年份或其他市场追分。

## 阅读顺序

`REPORT_ZH.md` → `PROTOCOL.md` / `PREFIT_CLARIFICATIONS.md` → `LOCAL_FIRST_SUMMARY.json` → `LITERATURE_SCOPE_ZH.md` → `REPRODUCTION_CHECK.json`（只有实际完成远端比较后才存在）。

源码：`core.py`、`experiment.py`、`baselines.py`、`analyze.py`、`audit.py`、`test_core.py`、`run_all.py`。`diagnose_fixed.py`为明确的结果后、固定参数诊断，不追加训练或另一个模型成绩。`compare_runs.py`对两个实际输出目录比较，不以CI绿色状态替代数值复现。

## 独立环境复现

安装requirements.txt所列CPU依赖；实际首轮Python3.13.5。不要覆盖用户现有虚拟环境。克隆本研究分支，在新的输出目录运行：

```bash
python research/pro_overlap_memory_20261016/run_all.py --out /path/to/NEW_results --workers 3
```

默认从本仓库现有广州CSV只读取已许可前950条/29204字节，并核对前缀哈希；不会请求下一字节、下载价格、使用旧权重或自动打开2022—2024。原始源日期不等于历史首次发布时刻；这是作者快照上的开发模拟。输出目标是1..20观测步净变化，正式评价h1/5/20。Ridge/树其余期限为NaN，不能作为预测。

如需重放后验固定参数诊断（不改模型），执行：

```bash
python research/pro_overlap_memory_20261016/diagnose_fixed.py --out /path/to/NEW_results
```

完整训练结果以`results_remote/`及其清单为准，必须在该目录实际上传后才称完整交付；全部首轮数据另外保留下载包`local_first/`，不被较好的重跑结果替换。工作流仅写本研究分支，不合并main、不改变可见性。精确远端比较由REPRODUCTION_CHECK披露，不预先保证相同。

论文近邻原代码未在本轮复现，原始全文不随包分发。数据署名和许可沿用仓库原有ATTRIBUTION记录；本轮不重新认证作者权属或官方市场语义。
