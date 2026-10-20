# 复现与资料说明

2026-10-20。所有结果限定已暴露2022/2023开发数据，原始行情私有，仅供用户在原已获授权范围内恢复。基于用户此前上传的私有运行档案 `carbon_distribution_update_private_run_20261019.zip`，取ZIP中的 `inputs/development/EUA_LONG_development.csv`, `GUANGZHOU_development.csv`, `HUBEI_development.csv`, `SHANGHAI_development.csv` 到本文件同级 `private_inputs/`；**该目录绝不可公开上传**。

运行环境 Python 3.13.5、numpy 2.3.5、pandas 2.2.3、scipy 1.17.0。安装 `requirements.txt` 并按以下顺序在本目录运行：

```bash
python run_experiment.py
python independent_audit.py
python conditional_statistics.py
python gaussian_mint_benchmark.py
RECON_WINDOW=126 python run_experiment.py
RECON_WINDOW=504 python run_experiment.py
```

默认方案 `run_experiment.py` 运行固定 W252方案，输出 `results/SUMMARY.json`、`results/SCORES_BY_CELL_HORIZON.csv`；选定权重只用测试年之前最近80个H20标签已成熟起点。完整20步历史情景只抽取t时已成熟的过去路径，分位相容桥与Gaussian WLS投影的加总约束都通过数值检验。`independent_audit.py` 不引用主实验函数，独立重算全部实验单元的直接经验分位和评分。

`conditional_statistics.py` 中逐期限和3中国市场统计是结果后分析；`results_sensitivity_W126` 和 `W504` 是结果后补做的窗口敏感性，不能替换主结果。`gaussian_mint_benchmark.py` 是已在主结果产生后另行冻结的基准补强，不作为原始W252预选模型。

每一输出 JSON 的 `status` 明确标记历史时期已暴露；2024+没有被本实验读取。不存在新年份独立确认或论文录用保证。

公开仓库只可包含 `*.py`、`PLAN_BEFORE_FIT_ZH.md`、`GAUSSIAN_BENCHMARK_PLAN_ZH.md`、`REPORT_ZH.md`、`REPRODUCE_ZH.md`、`requirements.txt`、`results/` 中去原价的聚合数值，以及后验敏感性聚合摘要。严禁提交 `private_inputs/`、NPZ逐点路径、原始交易所Excel、二进制模型、用户私有ZIP。来源和原始哈希见SUMMARY.json。

## 结果文件

- `REPORT_ZH.md`: 核心表、统计、证据强度和目标刊差距
- `results/SUMMARY.json`: W252完整8单元汇总、验证网格、配对Bootstrap
- `results/SCORES_BY_CELL_HORIZON.csv`: 8×方法×期限的Pinball、覆盖和宽度
- `results/INDEPENDENT_AUDIT.json`: 第二套实现独立数值审计
- `results/POSTHOC_BY_HORIZON_BOOTSTRAP.json`: 额外逐期限后验统计
- `results/GAUSSIAN_MIN_T_BENCHMARK.json`: Gaussian基准8单元完整指标与加总约束
- `results_sensitivity_W126/SUMMARY.json` 和 `results_sensitivity_W504/SUMMARY.json`: 历史窗口敏感性

在已曝光数据上这仍是开发科研，**即使数值复演完全相同也不构成新样本外确认**。
