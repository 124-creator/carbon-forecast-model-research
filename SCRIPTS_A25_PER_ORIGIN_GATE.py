# -*- coding: utf-8 -*-
"""
A25：完全逐原子的形状块（无任何每市场固定成分）—— A14 的唯一可能改进方向
=====================================================================
§24.29 的全部证据链：
 A21/A23/A24 族：SEL 0.95~0.97 → EVAL 1.01~1.02，**不成立**（前半段现象）
 A14：SEL 0.9998（8 显著胜 0 显著负）→ EVAL 1.0000（3 显著胜 2 显著负），**唯一稳定**
 差别：A14 的形状 = **逐原点重算**的窗内经验分位 ξ_d，无任何每市场固定成分。
 ⇒ 若要改进 A14，只能在"逐原点重算"这个约束内做。

A25 候选（全部逐原点重算，零拟合参数）：
  A25a  形状 = ξ_d，但**按窗内偏度的可靠性加权**与高斯混合：
         w_t = min(1, h/h_s) · g_t，  g_t = min(1, |skew_t|/se)  （可靠性门控，逐原点）
  A25b  形状 = ξ_d 与"偏度校准后的 ξ_d"混合：把 ξ_d 的偏度修正到"h 步聚合后应有的偏度"
         （h 步聚合使偏度按 h^{-1/2} 衰减？—— 这是可检验的解析命题）
  A25c  **最简**：A14 的 w 换成逐原子的 w_t = min(1, h/h_s)·(1 - ρ_t)，
         其中 ρ_t = 窗内日收益的一阶自相关（逐原点可观测量，度量窗口内的可预测性）
判据：在 SEL/EVAL 分离下，SEL 与 EVAL 都必须 < 1，且 EVAL 不能显著负。
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


RES = {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    for h in HOR:
        ts_tr = [t for t in range(W + h + 5, t0 - h - 1, STEP)]
        ts_te = [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP)]
        assert not (set(ts_tr) & set(ts_te))
        Y, LOC, S, XI, SKE, RHO = [], [], [], [], [], []
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
            SKE.append(abs(float(pd.Series(rd).skew())))
            RHO.append(float(pd.Series(rd).autocorr(1)))
        if len(Y) < 30:
            continue
        Y, LOC, S, XI, SKE, RHO = map(np.array, (Y, LOC, S, XI, SKE, RHO))
        w14 = min(1.0, h / 50.0)
        se = float(np.sqrt(6.0 / 249.0))
        g = np.minimum(1.0, SKE / se)
        rho_c = np.clip(RHO, -0.5, 0.5)
        G = np.tile(GV, (len(Y), 1))
        cands = {
            "A5": G,
            "A14": (1 - w14) * XI + w14 * G,
            # A25a：逐原子的可靠性门控
            "A25a": (1 - w14 * g)[:, None] * XI + (w14 * g)[:, None] * G,
            # A25c：逐原子的可预测性门控（|rho| 大 => 窗口内趋势/偏度可能是结构性的）
            "A25c": (1 - w14 * np.minimum(1.0, 2 * np.abs(rho_c)))[:, None] * XI
                    + (w14 * np.minimum(1.0, 2 * np.abs(rho_c)))[:, None] * G,
            # A25d：两者同时
            "A25d": (1 - w14 * g * np.minimum(1.0, 2 * np.abs(rho_c)))[:, None] * XI
                    + (w14 * g * np.minimum(1.0, 2 * np.abs(rho_c)))[:, None] * G,
        }
        L = {nm: crps_q(Y, LOC[:, None] + S[:, None] * shp) for nm, shp in cands.items()}
        RES[(m, h)] = dict(L=L, n=len(Y), half=len(Y) // 2)
cells = sorted(RES.keys())
FAM = ["A5", "A14", "A25a", "A25c", "A25d"]

for which in ("SEL", "EVAL"):
    print("\n" + "=" * 110)
    print(f"{which} 段")
    print("=" * 110)
    print(f"{'模型':<8}{'中位/A5':>10}{'胜':>7}{'显著胜':>8}{'显著负':>8}{'逐期限/A5':>36}")
    for nm in FAM:
        def seg(mh):
            R = RES[mh]; hh = R["half"]
            return R["L"][nm][:hh] if which == "SEL" else R["L"][nm][hh:]
        v = np.array([seg(mh).mean() for mh in cells])
        b = np.array([(RES[mh]["L"]["A5"][:RES[mh]["half"]] if which == "SEL"
                       else RES[mh]["L"]["A5"][RES[mh]["half"]:]).mean() for mh in cells])
        sw = sl = 0
        for mh in cells:
            R = RES[mh]; hh = R["half"]
            a = R["L"][nm][:hh] if which == "SEL" else R["L"][nm][hh:]
            c = R["L"]["A5"][:hh] if which == "SEL" else R["L"]["A5"][hh:]
            pv = dm(a, c)
            if pv == pv and pv < .05:
                if a.mean() < c.mean():
                    sw += 1
                else:
                    sl += 1
        per = " / ".join(f"{np.median([(RES[mh]['L'][nm][:RES[mh]['half']] if which=='SEL' else RES[mh]['L'][nm][RES[mh]['half']:]).mean() / (RES[mh]['L']['A5'][:RES[mh]['half']] if which=='SEL' else RES[mh]['L']['A5'][RES[mh]['half']:]).mean() for mh in cells if mh[1]==h]):.3f}" for h in HOR)
        print(f"{nm:<8}{np.median(v/b):>10.4f}{int((v<b).sum()):>4}/25{sw:>8}{sl:>8}{per:>36}")

print("\n逐格 A25a（SEL比 / EVAL比）：")
for mh in cells:
    R = RES[mh]; hh = R["half"]
    s = R["L"]["A25a"][:hh].mean() / R["L"]["A5"][:hh].mean()
    e = R["L"]["A25a"][hh:].mean() / R["L"]["A5"][hh:].mean()
    print(f"  {MN[mh[0]]:<6} h={mh[1]:<4} SEL {s:.3f}  EVAL {e:.3f}")
json.dump({"cells": {f"{m}|{h}": {k: float(RES[(m, h)]["L"][k].mean()) for k in RES[(m, h)]["L"]}
                     for (m, h) in cells}}, open(os.path.join(DIR, "A25.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 A25.json")
