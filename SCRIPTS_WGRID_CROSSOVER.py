# -*- coding: utf-8 -*-
"""
W 网格交叉点迁移检验（$h^* \propto W^{\\alpha}$）
==================================================
设计要点：把 (h0, H) 固定在 W=250 的训练估计值，只改变**估计量窗口** W。
这样隔离出的正是理论关心的量：窗口经验分位的抽样方差 ∝ 1/W。
且所有 W 使用**完全相同的测试原点集合**（起点取 max(Wmax+h+5, t0)），
避免"测试样本不同"的混杂。
预测：W 增大 -> EMP 更准 -> 模型优势缩小 -> 交叉点 h* 右移（alpha > 0）。
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
HOR = [5, 20, 40, 60, 120]; STEP = 2
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
WS = [125, 250, 500]
WMAX = max(WS)


def crps(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


DATA = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float))
    dt = pd.to_datetime(d.date).to_numpy()
    cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
    DATA[m] = dict(lp=lp, t0=int(np.searchsorted(dt, np.datetime64(cut))))

print("=" * 150)
print("表 1  A5 相对 EMP 的 CRPS 变化（%），同一测试原点集合，只改估计窗口 W")
print("      (h0, H) 固定在 W=250 的训练估计值，故差异纯粹来自窗口抽样误差")
print("=" * 150)
RES = {}
for m in MK:
    lp, t0 = DATA[m]["lp"], DATA[m]["t0"]
    h0, Hv = PAR[m]
    print(f"\n--- {m}  (len={len(lp)}, t0={t0}) ---")
    print(f"{'W':>5}{'h':>5}{'n_te':>6}{'CRPS:EMP':>10}{'CRPS:A5':>10}{'delta':>8}")
    for W in WS:
        for h in HOR:
            lo = max(WMAX + h + 5, t0)
            rec = []
            for t in range(lo, len(lp) - h, STEP):
                win = lp[t - W:t]
                rr = win[h:] - win[:-h]; rd = np.diff(win)
                sdw = np.std(rr, ddof=1); sdd = np.std(rd, ddof=1)
                if sdw <= 1e-12 or sdd <= 1e-12:
                    continue
                rec.append((t, float(rr.mean()), sdw, sdd,
                            np.quantile((rr - rr.mean()) / sdw, TAUS)))
            if len(rec) < 40:
                continue
            ts = np.array([r[0] for r in rec])
            mu = np.array([r[1] for r in rec]); sd = np.array([r[2] for r in rec])
            sdd = np.array([r[3] for r in rec]); xi = np.vstack([r[4] for r in rec])
            y = lp[ts + h] - lp[ts]
            L = h0 / (h0 + h)
            Qe = mu[:, None] + sd[:, None] * xi
            Qa = mu[:, None] * L + (sdd * h ** Hv)[:, None] * np.tile(GV, (len(mu), 1))
            ce, ca = float(crps(y, Qe).mean()), float(crps(y, Qa).mean())
            dd = (ca / ce - 1) * 100
            RES[f"{m}|{W}|{h}"] = dict(n=len(rec), emp=ce, a5=ca, d=dd, W=W, h=h, m=m)
            print(f"{W:>5}{h:>5}{len(rec):>6}{ce:>10.5f}{ca:>10.5f}{dd:>7.1f}%")

print("\n" + "=" * 150)
print("表 2  按期限汇总（5 个市场的 A5 相对 EMP 的 CRPS 变化）")
print("=" * 150)
print(f"{'h':>5}" + "".join(f"{'W='+str(W):>22}" for W in WS))
print(f"{'':>5}" + "".join(f"{'中位':>10}{'胜':>12}" for W in WS))
PROF = {}
for h in HOR:
    line = f"{h:>5}"
    for W in WS:
        dd = [RES[f"{m}|{W}|{h}"]["d"] for m in MK if f"{m}|{W}|{h}" in RES]
        if not dd:
            line += f"{'-':>10}{'-':>12}"
            continue
        PROF[f"{W}|{h}"] = dict(med=float(np.median(dd)), win=int(sum(1 for x in dd if x < 0)), n=len(dd))
        line += f"{np.median(dd):>9.1f}%{sum(1 for x in dd if x < 0):>7}/{len(dd):<5}"
    print(line)

print("\n" + "=" * 150)
print("表 3  交叉点 h*：A5 首次低于 EMP 的最小期限（按市场）")
print("=" * 150)
print(f"{'市场':>11}" + "".join(f"{'W='+str(W):>10}" for W in WS))
HS = {}
for m in MK:
    line = f"{m:>11}"
    for W in WS:
        hs = [h for h in HOR if f"{m}|{W}|{h}" in RES and RES[f"{m}|{W}|{h}"]["d"] < 0]
        hs_ok = [h for h in hs if all(RES[f"{m}|{W}|{hh}"]["d"] >= 0 for hh in HOR
                                      if hh < h and f"{m}|{W}|{hh}" in RES)]
        star = min(hs_ok) if hs_ok else None
        HS[f"{m}|{W}"] = star
        line += f"{str(star) if star else '>120':>10}"
    print(line)
print("\n  （只接受首次穿越即保持为负的市场；非单调的记为 >120）")

print("\n" + "=" * 150)
print("表 4  单调性检验：h=120 处的优势是否随 W 增大而缩小")
print("=" * 150)
for m in MK:
    row = [RES[f"{m}|{W}|120"]["d"] if f"{m}|{W}|120" in RES else np.nan for W in WS]
    mono = "↓单调缩小" if all(x < y for x, y in zip(row, row[1:]) if not (np.isnan(x) or np.isnan(y))) else "非单调"
    print(f"  {m:<11} " + "  ".join(f"W={W}: {x:+7.1f}%" for W, x in zip(WS, row)) + f"   {mono}")

print("\n" + "=" * 150)
print("表 5  h=120 处的聚合：中位 delta 与胜格数")
for W in WS:
    dd = [RES[f"{m}|{W}|120"]["d"] for m in MK if f"{m}|{W}|120" in RES]
    print(f"  W={W:>4}  中位 {np.median(dd):+7.2f}%   胜 {sum(1 for x in dd if x<0)}/{len(dd)}")

json.dump(dict(cells=RES, profile=PROF, hstar=HS,
               note="(h0,H) fixed at W=250 train estimates; common test origins per (m,h)"),
          open(os.path.join(OUT, "WGRID_CROSSOVER.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 WGRID_CROSSOVER.json")
