# -*- coding: utf-8 -*-
"""
自检：形状族的实现是否正确 + 训练段选参为何与实现偏度符号相反。
-------------------------------------------------------------------
关键疑问：实现残差偏度是**负**的（−0.1~−1.46），但训练段 CRPS 选出的 skew-normal
α = +2（**右**偏）。两者矛盾，必须查清是实现的 bug 还是我的推理错误。
检验：
 1) 直接构造"经验形状"（标准化后的实现分位）与各候选形状的逐点比较
 2) 对训练段的实现残差，直接算其**最优 α**（把实现 z 投到 skew-normal 上）
 3) CRPS 偏好：skew-normal 的 α=+2 与 α=−2 哪个 CRPS 更低？用同一批数据
"""
import numpy as np, pandas as pd, os, sys, json
from scipy.stats import norm
from scipy.special import owens_t
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(DIR, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
W = 250; STEP = 2
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}


def sn_cdf(z, a):
    if abs(a) < 1e-9:
        return norm.cdf(z)
    return norm.cdf(z) - 2 * owens_t(z, a)


def sn_ppf(u, a):
    out = np.zeros_like(u)
    for i, uu in enumerate(u):
        lo, hi = -20.0, 20.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if sn_cdf(mid, a) < uu:
                lo = mid
            else:
                hi = mid
        out[i] = (lo + hi) / 2
    return out


def crps_q(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


print("=" * 112)
print("1) skew-normal 形状表：α=+2 与 α=−2 的分位值（单位方差）")
print("=" * 112)
for a in (2.0, -2.0, 0.0):
    q = sn_ppf(GV, a)
    q = (q - q.mean()) / q.std(ddof=0)
    print(f"  α={a:+.1f}: " + " ".join(f"{v:+.3f}" for v in q))
print(f"  正态    : " + " ".join(f"{v:+.3f}" for v in GV))

print("\n" + "=" * 112)
print("2) 训练段实现残差：偏度、最优 α、CRPS 偏好")
print("=" * 112)
print(f"{'市场':<8}{'h':>5}{'n':>6}{'偏度':>8}{'α=+2 CRPS':>12}{'α=−2 CRPS':>12}"
      f"{'α=0 CRPS':>12}{'实现形状CRPS':>14}{'实现形状/α=0':>15}")
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    for h in (5, 20, 120):
        Y, L, S, Z = [], [], [], []
        for t in range(W + h + 5, t0 - h - 1, STEP):
            win = lp[max(0, t - W):t]
            rr = win[h:] - win[:-h]
            if len(rr) < 3 or np.std(rr, ddof=1) <= 0:
                continue
            rd = np.diff(win); sdd = np.std(rd, ddof=1)
            if not np.isfinite(sdd) or sdd <= 0:
                continue
            Y.append(lp[t + h] - lp[t]); L.append((h0 / (h0 + h)) * rr.mean())
            S.append(sdd * h ** Hv)
            Z.append((lp[t + h] - lp[t] - (h0 / (h0 + h)) * rr.mean()) / (sdd * h ** Hv))
        if len(Y) < 30:
            continue
        Y, L, S, Z = map(np.array, (Y, L, S, Z))
        from scipy.stats import skew
        q2 = sn_ppf(GV, 2.0); q2 = (q2 - q2.mean()) / q2.std(ddof=0)
        qm2 = sn_ppf(GV, -2.0); qm2 = (qm2 - qm2.mean()) / qm2.std(ddof=0)
        c2 = crps_q(Y, L[:, None] + S[:, None] * np.tile(q2, (len(Y), 1))).mean()
        cm2 = crps_q(Y, L[:, None] + S[:, None] * np.tile(qm2, (len(Y), 1))).mean()
        c0 = crps_q(Y, L[:, None] + S[:, None] * np.tile(GV, (len(Y), 1))).mean()
        # 实现形状：把 Z 的经验分位当作形状
        emp = np.quantile(Z, TAUS)
        cE = crps_q(Y, L[:, None] + S[:, None] * np.tile(emp, (len(Y), 1))).mean()
        print(f"{MN[m]:<8}{h:>5}{len(Y):>6}{skew(Z):>8.3f}{c2:>12.5f}{cm2:>12.5f}"
              f"{c0:>12.5f}{cE:>14.5f}{cE/c0:>15.4f}")
