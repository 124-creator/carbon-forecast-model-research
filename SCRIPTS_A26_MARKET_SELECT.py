# -*- coding: utf-8 -*-
"""
A26（最后实验）：市场级的"形状族选择"—— 用训练段可估的市场属性决定用哪种形状
=====================================================================
HOLDOUT 的教训：凡是**每市场固定**的成分都会在 EVAL 上失效（A21/A23/A24 族）。
最后的可能是：某个每市场固定的**二值选择**（而非连续参数）足够稳健。
候选属性（全部在**训练段**估计，量纲无关）：
  P1  偏度的**半窗一致率**（窗内前后半偏度同号的频率）—— 广州 0.66~0.72 vs 上海 0.21~0.31
  P2  偏度的**自相关**（偏度本身的持续性）
  P3  窗内日收益的**一阶自相关 |ρ|**
规则（事先写定，不看结果调）：
  若 P1 ≥ 0.5 ⇒ 用"混合 w(ξ) 且 w 随 h 衰减"（A14 家族）
  否则 ⇒ 用纯高斯（A5 家族）
即：**A5 与 A14 之间的二值选择，由训练段估计的市场属性决定。**
这没有连续参数，只有一个阈值，且阈值 0.5 是中位自然值。
检验：训练段估 P1 → SEL 验证选择是否正确 → EVAL 报告最终表现。
"""
import numpy as np, pandas as pd, os, sys, json
from scipy.stats import norm, t as tdist
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(DIR, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}


def crps_q(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


def dm(a, b, L=10):
    d = np.asarray(a) - np.asarray(b); n = len(d)
    if n < 20:
        return np.nan
    m_ = d.mean(); dc = d - m_
    lrv = dc @ dc / n
    for l in range(1, L + 1):
        lrv += 2 * (1 - l / (L + 1)) * (dc[l:] @ dc[:-l] / n)
    if lrv <= 0:
        return np.nan
    s = m_ / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    return float(2 * (1 - tdist.cdf(abs(s), df=n - 1)))


# ---- 训练段：估每市场的偏度半窗一致率 ----
print("=" * 112)
print("训练段：偏度半窗一致率（市场属性 P1）")
print("=" * 112)
P1, RHO = {}, {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    for h in HOR:
        cs, rh = [], []
        for t in range(W + h + 5, t0 - h - 1, STEP):
            win = lp[max(0, t - W):t]
            rr = win[h:] - win[:-h]
            if len(rr) < 3 or np.std(rr, ddof=1) <= 0:
                continue
            rd = np.diff(win)
            if np.std(rd, ddof=1) <= 0:
                continue
            half = len(rd) // 2
            s1 = float(pd.Series(rd[:half]).skew()); s2 = float(pd.Series(rd[half:]).skew())
            cs.append(1.0 if (s1 * s2 > 0) else 0.0)
            rh.append(float(pd.Series(rd).autocorr(1)))
        if len(cs) >= 30:
            P1[(m, h)] = float(np.mean(cs))
            RHO[(m, h)] = float(np.mean(rh))
    v = [P1[(m, h)] for h in HOR if (m, h) in P1]
    vr = [RHO[(m, h)] for h in HOR if (m, h) in RHO]
    P1m = float(np.median(v)) if v else np.nan
    print(f"  {MN[m]:<8} 逐期限 P1 = {[round(P1[(m,h)],3) for h in HOR if (m,h) in P1]}  "
          f"中位 {P1m:.3f}   |ρ| 中位 {np.median(np.abs(vr)):.3f}")
print("  → 按事先写定的规则 P1 ≥ 0.5 ⇒ A14 家族，否则 A5（纯高斯）")
USE14 = {}
for m in CUT:
    v = [P1[(m, h)] for h in HOR if (m, h) in P1]
    USE14[m] = bool(v) and float(np.median(v)) >= 0.5
print(f"  选择结果: " + "  ".join(f"{MN[m]}→{'A14' if USE14[m] else 'A5'}" for m in CUT))

# ---- 测试段 ----
RES = {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    for h in HOR:
        ts_te = [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP)]
        Y, LOC, S, XI = [], [], [], []
        for t in ts_te:
            win = lp[max(0, t - W):t]
            rr = win[h:] - win[:-h]
            if len(rr) < 3 or np.std(rr, ddof=1) <= 0:
                continue
            rd = np.diff(win); sdd = np.std(rd, ddof=1)
            if not np.isfinite(sdd) or sdd <= 0:
                continue
            Y.append(lp[t + h] - lp[t]); LOC.append((h0 / (h0 + h)) * rr.mean())
            S.append(sdd * h ** Hv); XI.append(np.quantile((rd - rd.mean()) / sdd, TAUS))
        if len(Y) < 25:
            continue
        Y, LOC, S, XI = map(np.array, (Y, LOC, S, XI))
        w14 = min(1.0, h / 50.0)
        G = np.tile(GV, (len(Y), 1))
        shpA5 = G
        shpA14 = (1 - w14) * XI + w14 * G
        shpSEL = shpA14 if USE14[m] else shpA5
        L = {"A5": crps_q(Y, LOC[:, None] + S[:, None] * shpA5),
             "A14": crps_q(Y, LOC[:, None] + S[:, None] * shpA14),
             "SEL": crps_q(Y, LOC[:, None] + S[:, None] * shpSEL)}
        RES[(m, h)] = dict(L=L, n=len(Y), half=len(Y) // 2)
cells = sorted(RES.keys())

for which in ("SEL", "EVAL"):
    print("\n" + "=" * 112)
    print(f"{which} 段")
    print("=" * 112)
    print(f"{'模型':<8}{'中位/A5':>10}{'胜':>7}{'显著胜':>8}{'显著负':>8}")
    for nm in ("A5", "A14", "SEL"):
        v, b = [], []
        sw = sl = 0
        for mh in cells:
            R = RES[mh]; hh = R["half"]
            a = R["L"][nm][:hh] if which == "SEL" else R["L"][nm][hh:]
            c = R["L"]["A5"][:hh] if which == "SEL" else R["L"]["A5"][hh:]
            v.append(a.mean()); b.append(c.mean())
            pv = dm(a, c)
            if pv == pv and pv < .05:
                if a.mean() < c.mean():
                    sw += 1
                else:
                    sl += 1
        v, b = np.array(v), np.array(b)
        print(f"{nm:<8}{np.median(v/b):>10.4f}{int((v<b).sum()):>4}/25{sw:>8}{sl:>8}")

print("\n按市场的 EVAL 比（SEL 模型的每格）：")
for m in CUT:
    vs = [RES[(m, h)]["L"]["SEL"][RES[(m, h)]["half"]:].mean() /
          RES[(m, h)]["L"]["A5"][RES[(m, h)]["half"]:].mean() for h in HOR if (m, h) in RES]
    print(f"  {MN[m]:<8} 选 {'A14' if USE14[m] else 'A5 '}  EVAL 中位比 {np.median(vs):.4f}  "
          f"逐格 {[round(v,3) for v in vs]}")
json.dump({"P1": {f"{m}|{h}": v for (m, h), v in P1.items()},
           "USE14": USE14,
           "cells": {f"{m}|{h}": {k: float(RES[(m, h)]["L"][k].mean()) for k in RES[(m, h)]["L"]}
                     for (m, h) in cells}},
          open(os.path.join(DIR, "A26.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 A26.json")
