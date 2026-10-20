# -*- coding: utf-8 -*-
"""审计 SCRIPTS_BENCH_FAMILY.py 的 train/test 原点划分是否泄漏。"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

P = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020\PANEL_5MARKET.csv"
W, STEP, HOR = 250, 2, [5, 20, 60, 120]
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2022-01-01", "EU": "2022-01-01"}
PANEL = pd.read_csv(P)
print("市场  n   t0    h   |ts_tr| |ts_te|  重叠数  重叠占比  测试起始t  t0")
for m in ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]:
    d = PANEL[PANEL.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float))
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    for h in HOR:
        ts_tr = set(range(W + h + 5, t0 - h - 1, STEP))
        ts_te = set(range(W + h + 5, len(lp) - h, STEP))
        ov = len(ts_tr & ts_te)
        print(f"{m:10s} {len(lp):5d} {t0:5d} {h:4d} |{len(ts_tr):6d} |{len(ts_te):6d} "
              f"{ov:7d} {ov/max(1,len(ts_tr)):8.1%} "
              f"{min(ts_te) if ts_te else -1:10d} {t0:6d}")
print()
print("说明：ts_te 若从 260 附近开始而 t0 在数百到上千，则测试集包含全部训练原点。")
