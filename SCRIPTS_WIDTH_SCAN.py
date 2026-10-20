# -*- coding: utf-8 -*-
"""
决定性诊断：位置块 vs 宽度块，谁主导 CRPS？
------------------------------------------------------------------
A15 族（A15a/b/c）与 ZERO 的结果：LCRPS（固定高斯形状、只比位置）的中位比值
恒为 1.0000、0/25 胜 —— 说明**在 A5 的宽度下，位置块对 CRPS 几乎没有贡献**。
§24.27 之所以四种 λ 全部"失败"，是因为它们都在做一件 CRPS 看不见的事。

那 CRPS 看见的是什么？本脚本逐格扫描"宽度乘子" c：
  Q = loc + c · σ̂_d · h^H · G(τ)
在 c ∈ [0.4, 2.0] 上找每格的 CRPS 最优 c，看最优 c 随 (市场, h) 怎么走。
若最优 c 系统性偏离 1，且有可解释的规律 → 宽度块才是真正可创新的部件。
同时报告：**位置误差在给定最优宽度下还剩多少**（即位置块是否还有独立价值）。
"""
import numpy as np, pandas as pd, os, sys, json
from scipy.stats import norm, t as tdist
sys.stdout.reconfigure(encoding="utf-8")
OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
CS = np.round(np.arange(0.40, 2.01, 0.05), 3)


def crps_q(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


RES = {}
print("=" * 120)
print("一、宽度乘子 c 的最优值（测试段最优 = 事后上界，仅作诊断，不作为模型）")
print("=" * 120)
print(f"{'市场':<8}{'h':>5}{'A5':>10}{'c*=0.6':>10}{'c*=0.8':>10}{'c*=1.0':>10}"
      f"{'c*=1.2':>10}{'c*=1.6':>10}{'最优c':>8}{'CRPS@c*':>10}{'改善':>8}{'n':>5}")
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    for h in HOR:
        ts_te = [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP)]
        Y, LOC, S5, SD = [], [], [], []
        for t in ts_te:
            win = lp[max(0, t - W):t]
            rr = win[h:] - win[:-h]
            sd = np.std(rr, ddof=1)
            if sd <= 0:
                continue
            rd = np.diff(win); sdd = np.std(rd, ddof=1)
            Y.append(lp[t + h] - lp[t]); LOC.append((h0 / (h0 + h)) * rr.mean())
            S5.append(sdd * h ** Hv); SD.append(sd)
        Y, LOC, S5, SD = map(np.array, (Y, LOC, S5, SD))
        if len(Y) < 25:
            continue
        cr = {}
        for c in CS:
            cr[c] = float(crps_q(Y, LOC[:, None] + c * S5[:, None] * GV).mean())
        cb = min(cr, key=cr.get)
        a5 = cr[1.0]
        RES[f"{m}|{h}"] = dict(n=len(Y), c_star=float(cb), crps_at_c=float(cr[cb]),
                               crps_at_1=float(a5), curve={str(k): v for k, v in cr.items()},
                               loc_rmse=float(np.sqrt(np.mean((Y - LOC) ** 2))),
                               zero_loc_crps=float(crps_q(Y, S5[:, None] * GV).mean()),
                               # 最优宽度下，位置块还有多少价值
                               loc_zero_at_c=float(crps_q(Y, cb * S5[:, None] * GV).mean()))
        print(f"{MN[m]:<8}{h:>5}{a5:>10.5f}{cr[0.6]:>10.5f}{cr[0.8]:>10.5f}{cr[1.0]:>10.5f}"
              f"{cr[1.2]:>10.5f}{cr[1.6]:>10.5f}{cb:>8.2f}{cr[cb]:>10.5f}"
              f"{(cr[cb]/a5-1)*100:>7.1f}%{len(Y):>5}")

print("\n" + "=" * 120)
print("二、最优宽度下，位置块还有多少价值？（c* 下 A5 位置 vs 零位置）")
print("=" * 120)
print(f"{'市场':<8}{'h':>5}{'c*':>7}{'有位置':>10}{'零位置':>10}{'差':>9}{'相对':>9}")
for m in CUT:
    for h in HOR:
        k = f"{m}|{h}"
        if k not in RES:
            continue
        r = RES[k]
        rel = r["loc_zero_at_c"] / r["crps_at_c"] - 1
        print(f"{MN[m]:<8}{h:>5}{r['c_star']:>7.2f}{r['crps_at_c']:>10.5f}"
              f"{r['loc_zero_at_c']:>10.5f}{r['loc_zero_at_c']-r['crps_at_c']:>9.5f}{rel:>8.2%}")

print("\n" + "=" * 120)
print("三、c* 的结构：c* 随 h 的走向（每市场）")
print("=" * 120)
for m in CUT:
    xs = [h for h in HOR if f"{m}|{h}" in RES]
    cs = [RES[f"{m}|{h}"]["c_star"] for h in xs]
    print(f"{MN[m]:<8}" + "  ".join(f"h{h}:c={c:.2f}" for h, c in zip(xs, cs)))
print("\n  按期限的中位 c*：")
for h in HOR:
    v = [RES[f"{m}|{h}"]["c_star"] for m in CUT if f"{m}|{h}" in RES]
    print(f"  h={h:<5} 中位 c* = {np.median(v):.2f}   各市场 {[RES[f'{m}|{h}']['c_star'] for m in CUT if f'{m}|{h}' in RES]}")

json.dump(RES, open(os.path.join(OUT, "WIDTH_SCAN.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 WIDTH_SCAN.json")
