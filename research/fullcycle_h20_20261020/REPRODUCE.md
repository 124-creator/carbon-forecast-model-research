# 可复现性清单

本文每个出现在正文或图表中的数字，都由下表中的脚本从 `PANEL_5MARKET.csv` （sha256 `d470238386e201e6...`，8912 行）重新生成。

## 运行环境

```
Python 3.11.15
numpy 2.4.6, pandas 3.0.6, scipy 1.17.1
scikit-learn 1.9.1, statsmodels 0.15.0
```

## 数字 → 脚本 对照表

| ID | 论文中的内容 | 脚本 | 输入 | 输出 | 重跑命令 |
|---|---|---|---|---|---|
| R1 | 主切分上的 (h0,H) 与 A5/A14 的 CRPS | `SCRIPTS_A14.py` | `PANEL_5MARKET.csv` | `FULL_ROSTER.json, A14.json` | `python SCRIPTS_A14.py` |
| R2 | 五切分稳健性（规则 R 完整链条） | `SCRIPTS_EXP17_FINAL_SPEC.py` | `PANEL_5MARKET.csv` | `FINAL_SPEC.json` | `python SCRIPTS_EXP17_FINAL_SPEC.py` |
| R3 | 主切分上的规则 R → h_sat=50 | `SCRIPTS_EXP20_PAPER_CLAIMS.py` | `PANEL_5MARKET.csv` | `PAPER_CLAIMS.json` | `python SCRIPTS_EXP20_PAPER_CLAIMS.py` |
| R4 | 最终证据表（A14 vs 各对照，200 次打乱） | `SCRIPTS_EXP10_FINAL_EVIDENCE.py` | `PANEL_5MARKET.csv` | `FINAL_EVIDENCE.json` | `python SCRIPTS_EXP10_FINAL_EVIDENCE.py` |
| R5 | DM 检验的零假设校准 | `SCRIPTS_EXP8_DM_CALIB.py` | `PANEL_5MARKET.csv` | `DM_CALIB.json` | `python SCRIPTS_EXP8_DM_CALIB.py` |
| R6 | 合并检验的统计效力 | `SCRIPTS_EXP9_POWER_TEST.py` | `PANEL_5MARKET.csv` | `POWER_TEST.json` | `python SCRIPTS_EXP9_POWER_TEST.py` |
| R7 | 安慰剂电池（校准版） | `SCRIPTS_EXP18_PLACEBO_CALIB.py` | `PANEL_5MARKET.csv` | `PLACEBO_CALIB.json` | `python SCRIPTS_EXP18_PLACEBO_CALIB.py` |
| R8 | 16 期限密网格 | `SCRIPTS_EXP2_HORIZON_GRID.py` | `PANEL_5MARKET.csv` | `HORIZON_GRID.json` | `python SCRIPTS_EXP2_HORIZON_GRID.py` |
| R9 | 多指标一致性 | `SCRIPTS_EXP3_MULTIMETRIC.py` | `PANEL_5MARKET.csv` | `MULTIMETRIC.json` | `python SCRIPTS_EXP3_MULTIMETRIC.py` |
| R10 | 数据口径稳健性 | `SCRIPTS_EXP5_DATAROBUST.py` | `PANEL_5MARKET.csv` | `DATAROBUST.json` | `python SCRIPTS_EXP5_DATAROBUST.py` |
| R11 | 经济价值与履约决策 | `SCRIPTS_EXP6_ECONVALUE.py` | `PANEL_5MARKET.csv` | `ECONVALUE.json` | `python SCRIPTS_EXP6_ECONVALUE.py` |
| R12 | 市场级诊断量与留一市场检验 | `SCRIPTS_EXP19_MECHANISM.py` | `PANEL_5MARKET.csv` | `MECHANISM.json` | `python SCRIPTS_EXP19_MECHANISM.py` |
| R13 | H 的跨市场迁移（否） | `SCRIPTS_H_TRANSFER.py` | `PANEL_5MARKET.csv` | `H_TRANSFER.json` | `python SCRIPTS_H_TRANSFER.py` |
| R14 | 深度学习序列基准 | `SCRIPTS_DL_SEQ.py` | `PANEL_5MARKET.csv` | `DL_SEQ.json, DL_SEQ_AGG.json` | `python SCRIPTS_DL_SEQ.py` |
| R15 | 独立复现检验 | `SCRIPTS_VERIFY.py` | `PANEL_5MARKET.csv, FULL_ROSTER.json, A14.json` | `VERIFY.json` | `python SCRIPTS_VERIFY.py` |
| R16 | 论文图 1–5 | `SCRIPTS_FIGURES.py` | `HORIZON_GRID.json, MULTIMETRIC.json, FINAL_EVIDENCE.json, MULTISPLIT_A14.json` | `figures/*.png` | `python SCRIPTS_FIGURES.py` |

## 两条必须写进论文的口径

```python
# 测试原点（注意是从 t0 起步长 2，不是从 W+h+5 起步长 2）
ts_te = range(max(W + h + 5, t0), len(lp) - h, STEP)   # STEP = 2

# 有效性过滤
if len(win) < 120 or len(rr) < 40 or np.std(rd, ddof=1) <= 0:
    ok = False   # 该原点被剔除
```

这两条若不写清，数字无法复现。§24.41 的独立复现检验正是因此发现过一处奇偶性口径差异。

## 论文图

- `figures/fig1_horizon_profile.png`
- `figures/fig2_pit_calibration.png`
- `figures/fig3_coverage.png`
- `figures/fig4_calibrated_pvalues.png`
- `figures/fig5_multisplit.png`
