# -*- coding: utf-8 -*-
"""
A23：带可靠性门控的观测偏度形状（零参数门控）
=====================================================================
A21b 的机制边界已定位：
  广州/湖北（受益）：窗内日收益的负偏度与 h 步实现分布的负偏度**一致**
  上海（受损，H=0.30 强均值回复）：窗内偏度是噪声，远期分布不跟随
  EU/CE（中性）
⇒ 门控必须回答："窗内偏度是不是可重复的（真偏度）还是噪声？"
零参数门控的候选（都不引入拟合参数）：
  G1  偏度的**自相关**：skew(窗内后半) 与 skew(窗内前半) 的相关性/一致性
  G2  偏度的**样本标准误**：skew 的渐近标准误 ≈ sqrt(6/n)；若 |skew|/se 小 ⇒ 噪声
  G3  偏度与 h 步实现偏度的**符号一致率**（需要未来数据 ⇒ 不可用于预测，但可作诊断）
  G4  偏度的**窗口稳定性**：skew 在最近 125 天 vs 前 125 天的一致性
最简洁且可解释的是 G2 + G4：把"偏度估计的可信度"映射成权重。
最终形状： Q = λμ̂ + σ̂_d h^H · [ (1-w_g)·ξ_d + w_g·SN(α_obs) ]
  其中 w_g = 可靠性门控（0..1），w(h)=min(1,h/50) 仍控制长端退回。
若门控有效：A21b 的上海损失应消失，广州/湖北的收益保留。
"""
import numpy as np, pandas as pd, os, sys, json
from scipy.stats import norm, t as tdist, skewnorm
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
ALG = np.round(np.arange(-8, 8.01, 0.5), 2)
_M = np.stack([(lambda q: (q - q.mean()) / q.std(ddof=0))(skewnorm.ppf(TAUS, a)) for a in ALG])


def shape_alpha(al):
    al = np.clip(np.asarray(al, float), ALG[0], ALG[-1])
    i = np.clip(np.searchsorted(ALG, al) - 1, 0, len(ALG) - 2)
    lo, hi = ALG[i], ALG[i + 1]
    w = (al - lo) / np.where(hi - lo < 1e-12, 1.0, hi - lo)
    return _M[i] * (1 - w)[:, None] + _M[i + 1] * w[:, None]


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


# ---- 诊断：偏度可靠性 vs 门控效果 ----
print("=" * 118)
print("诊断：窗内日收益偏度的可靠性代理 与 市场")
print("=" * 118)
print(f"{'市场':<8}{'h':>5}{'|skew|中位':>11}{'se=sqrt(6/n)':>14}{'|skew|/se':>11}"
      f"{'前后半一致':>11}{'实现偏度':>10}{'符号一致率':>11}")
REL = {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    for h in HOR:
        SKE, SE, CONS, ZSKE, SIGN = [], [], [], [], []
        for t in range(max(W + h + 5, t0), len(lp) - h, STEP):
            win = lp[max(0, t - W):t]
            rr = win[h:] - win[:-h]
            if len(rr) < 3 or np.std(rr, ddof=1) <= 0:
                continue
            rd = np.diff(win); sdd = np.std(rd, ddof=1)
            if not np.isfinite(sdd) or sdd <= 0:
                continue
            sk = float(pd.Series(rd).skew())
            n = len(rd); se = np.sqrt(6.0 / n)
            half = len(rd) // 2
            s1 = float(pd.Series(rd[:half]).skew()); s2 = float(pd.Series(rd[half:]).skew())
            cons = 1.0 if (s1 * s2 > 0) else 0.0
            y = lp[t + h] - lp[t]
            z = (y - (h0 / (h0 + h)) * rr.mean()) / (sdd * h ** Hv)
            zsk = float(pd.Series(z).skew())
            SKE.append(sk); SE.append(se); CONS.append(cons); ZSKE.append(zsk)
            SIGN.append(1.0 if sk * zsk > 0 else 0.0)
        if len(SKE) < 25:
            continue
        REL[(m, h)] = dict(abs_skew=float(np.median(np.abs(SKE))), se=float(np.median(SE)),
                           ratio=float(np.median(np.abs(SKE)) / np.median(SE)),
                           cons=float(np.mean(CONS)), zskew=float(np.median(ZSKE)),
                           sign=float(np.mean(SIGN)))
        print(f"{MN[m]:<8}{h:>5}{np.median(np.abs(SKE)):>11.3f}{np.median(SE):>14.4f}"
              f"{np.median(np.abs(SKE))/np.median(SE):>11.2f}{np.mean(CONS):>11.3f}"
              f"{np.median(ZSKE):>10.3f}{np.mean(SIGN):>11.3f}")
    print()

# ---- A23 候选 ----
print("=" * 118)
print("A23：带可靠性门控的观测偏度形状")
print("=" * 118)
print(f"{'模型':<8}{'/A5中位':>10}{'胜':>7}{'显著胜':>8}{'显著负':>8}{'逐期限/A5':>34}")
LOSS = {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    for h in HOR:
        Y, LOC, S, XI, SK, CONS, ABS = [], [], [], [], [], [], []
        for t in range(max(W + h + 5, t0), len(lp) - h, STEP):
            win = lp[max(0, t - W):t]
            rr = win[h:] - win[:-h]
            if len(rr) < 3 or np.std(rr, ddof=1) <= 0:
                continue
            rd = np.diff(win); sdd = np.std(rd, ddof=1)
            if not np.isfinite(sdd) or sdd <= 0:
                continue
            Y.append(lp[t + h] - lp[t]); LOC.append((h0 / (h0 + h)) * rr.mean())
            S.append(sdd * h ** Hv); XI.append(np.quantile((rd - rd.mean()) / sdd, TAUS))
            SK.append(float(pd.Series(rd).skew()))
            half = len(rd) // 2
            s1 = float(pd.Series(rd[:half]).skew()); s2 = float(pd.Series(rd[half:]).skew())
            CONS.append(1.0 if (s1 * s2 > 0) else 0.0)
            ABS.append(abs(float(pd.Series(rd).skew())))
        if len(Y) < 25:
            continue
        Y, LOC, S, XI, SK, CONS, ABS = map(np.array, (Y, LOC, S, XI, SK, CONS, ABS))
        w14 = min(1.0, h / 50.0)
        shp_obs = shape_alpha(1.0 * SK)
        # 门控 1：可靠性 = min(1, |skew|/se / c)，c=1 ⇒ |skew| 超过 1 个标准误才算数
        se = float(np.sqrt(6.0 / 249.0))   # 偏度渐近标准误 sqrt(6/n)，n=249 个日收益
        g1 = np.minimum(1.0, ABS / (1.0 * se))
        # 门控 2：前后半一致性（0 或 1），按 |skew|/se 软化
        g2 = CONS * np.minimum(1.0, ABS / (2.0 * se))
        cands = {
            "A5": np.tile(GV, (len(Y), 1)),
            "A14": (1 - w14) * XI + w14 * GV[None, :],
            "A21b": (1 - w14) * XI + w14 * shp_obs,
            "A23a": (1 - w14) * XI + w14 * (g1[:, None] * shp_obs + (1 - g1[:, None]) * GV[None, :]),
            "A23b": (1 - w14) * XI + w14 * (g2[:, None] * shp_obs + (1 - g2[:, None]) * GV[None, :]),
        }
        L = {nm: crps_q(Y, LOC[:, None] + S[:, None] * shp) for nm, shp in cands.items()}
        LOSS[(m, h)] = L
        print(f"  {MN[m]:<6} h={h:<4} A5 {L['A5'].mean():.5f} A14 {L['A14'].mean():.5f} "
              f"A21b {L['A21b'].mean():.5f} A23a {L['A23a'].mean():.5f} A23b {L['A23b'].mean():.5f} "
              f"| g1中位 {np.median(g1):.2f} g2中位 {np.median(g2):.2f}")

cells = sorted(LOSS.keys())
print("\n" + "=" * 118)
print("汇总（相对 A5）")
print("=" * 118)
print(f"{'模型':<8}{'/A5中位':>10}{'胜':>7}{'显著胜':>8}{'显著负':>8}{'逐期限/A5 5/20/40/60/120':>42}")
for nm in ("A14", "A21b", "A23a", "A23b"):
    rr = np.array([LOSS[(m, h)][nm].mean() / LOSS[(m, h)]["A5"].mean() for (m, h) in cells])
    sw = sum(1 for (m, h) in cells if dm(LOSS[(m, h)][nm], LOSS[(m, h)]["A5"]) < .05 and LOSS[(m, h)][nm].mean() < LOSS[(m, h)]["A5"].mean())
    sl = sum(1 for (m, h) in cells if dm(LOSS[(m, h)][nm], LOSS[(m, h)]["A5"]) < .05 and LOSS[(m, h)][nm].mean() > LOSS[(m, h)]["A5"].mean())
    per = " / ".join(f"{np.median([LOSS[(m,h)][nm].mean()/LOSS[(m,h)]['A5'].mean() for m,_ in cells if h==h]):.3f}" for h in HOR)
    print(f"{nm:<8}{np.median(rr):>10.4f}{int((rr<1).sum()):>4}/25{sw:>8}{sl:>8}{per:>42}")

print("\n按市场的 A21b/A14 与 A23b/A14（>1 表示比 A14 差）：")
for nm in ("A21b", "A23a", "A23b"):
    print(f"  {nm}: " + "  ".join(
        f"{MN[m]} {np.median([LOSS[(m,h)][nm].mean()/LOSS[(m,h)]['A14'].mean() for h in HOR]):.3f}"
        for m in CUT if any((m, hh) in LOSS for hh in HOR)))
json.dump({"REL": {f"{m}|{h}": v for (m, h), v in REL.items()},
           "cells": {f"{m}|{h}": {k: float(LOSS[(m, h)][k].mean()) for k in LOSS[(m, h)]}
                     for (m, h) in cells}},
          open(os.path.join(DIR, "A23.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 A23.json")
