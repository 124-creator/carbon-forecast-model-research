# -*- coding: utf-8 -*-
"""
A13 = 去掉位置块（参数 10->5）+ 形状按 w(h)=min(1,h/50) 混合
============================================================
两个部件各自已被验证：
  1. lambda0：位置块三种改法全失效、训练段单调偏好 kappa=1 而测试段五五开
     => 位置块结构性不可辨，去掉后中位不变而最坏格子被修好（LOC_MONO / KAPPA_FIT）
  2. 形状混合 w(h)=min(1,h/50)：§24.25 的 A8a，零自由参数，短端靠经验日频形状、
     长端退回高斯；A5 最差的 7 格里 3 格是 h=5
A13 把两者叠加，只保留 H 一个逐市场参数（5 个），另加一个全局常数 50（须披露其
来自 oracle w 剖面的观测值，不是拟合出来的）。

对比：A5（10 参数）、λ0（5 参数）、A13（5 参数）、EMP，以及 FULL_ROSTER 的 11 个基准。
输出 A13.json
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm, t as tdist
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS); JM = 7
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
HSAT = 50.0


def crps_q(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


def dm(a, b, L=10):
    d = a - b; n = len(d)
    if n < 20:
        return np.nan
    m = d.mean(); dc = d - m
    lrv = dc @ dc / n
    for l in range(1, L + 1):
        lrv += 2 * (1 - l / (L + 1)) * (dc[l:] @ dc[:-l] / n)
    if lrv <= 0:
        return np.nan
    s = m / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    return float(2 * (1 - tdist.cdf(abs(s), df=n - 1)))


RES = {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    RES[m] = {}
    for h in HOR:
        ts_te = [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP)]
        mu, sd, y, xi_d, ok = [], [], [], [], []
        for t in ts_te:
            win = lp[max(0, t - W):t]
            rr = win[h:] - win[:-h]
            rd = np.diff(win)
            if len(win) < 120 or len(rr) < 40 or np.std(rd, ddof=1) <= 0:
                ok.append(False); continue
            sdd = np.std(rd, ddof=1)
            mu.append(rr.mean()); sd.append(sdd); y.append(lp[t + h] - lp[t])
            xi_d.append(np.quantile((rd - rd.mean()) / sdd, TAUS)); ok.append(True)
        mu = np.array(mu); sd = np.array(sd); y = np.array(y)
        xi_d = np.array(xi_d); ok = np.array(ok)
        w = min(1.0, h / HSAT)
        xi = (1 - w) * xi_d + w * GV[None, :]
        sc = (sd * h ** Hv)[:, None]
        Q5 = (h0 / (h0 + h)) * mu[:, None] + sc * GV[None, :]
        Q0 = sc * GV[None, :]
        Q13 = sc * xi
        l5 = crps_q(y, Q5); l0 = crps_q(y, Q0); l13 = crps_q(y, Q13)
        RES[m][h] = dict(n=int(ok.sum()), A5=float(l5.mean()), lam0=float(l0.mean()),
                         A13=float(l13.mean()), p13=dm(l13, l5), p13_0=dm(l13, l0),
                         w=w)
        print(f"{m:<11}h={h:<4}n={int(ok.sum()):<4} A5 {l5.mean():.5f}  "
              f"λ0 {l0.mean():.5f}  A13 {l13.mean():.5f}  "
              f"p(A13 vs A5)={RES[m][h]['p13']:.3f}  p(A13 vs λ0)={RES[m][h]['p13_0']:.3f}")

R = json.load(open(os.path.join(OUT, "FULL_ROSTER.json"), encoding="utf-8"))["res"]
RM = {m: {int(h): v for h, v in D.items()} for m, D in R.items()}
cells = [(m, h) for m in CUT for h in HOR]
med = lambda a: float(np.median(a))

print("\n" + "=" * 120)
print("放进 11 模型名册")
print("=" * 120)
print(f"{'模型':<16}{'参数':>6}{'相对最优中位':>14}{'胜出':>8}{'平均排名':>10}{'第1名':>8}{'前三':>7}{'最差':>9}")
for nm, key, npar in (("A5", "A5", "10"), ("λ0", "lam0", "5"), ("A13", "A13", "5+1")):
    bo, rk = [], []
    for m, h in cells:
        c = dict(RM[m][h]["crps"]); c["MINE"] = RES[m][h][key]
        bo.append(c["MINE"] / min(v for k, v in c.items() if k != "MINE"))
        rk.append(sorted(c, key=c.get).index("MINE") + 1)
    bo = np.array(bo); rk = np.array(rk)
    print(f"{nm:<16}{npar:>6}{med(bo):>14.4f}{int((bo<=1.0).sum()):>5}/{len(bo):<2}"
          f"{rk.mean():>10.2f}{int((rk==1).sum()):>8}{int((rk<=3).sum()):>7}{bo.max():>9.4f}")

print("\n逐期限")
print(f"{'h':>5}{'λ0/A5':>9}{'A13/A5':>9}{'A13/A5 胜':>11}{'A13显著胜':>10}{'A13显著负':>10}")
for h in HOR:
    ks = [(m, h) for m in CUT]
    r0 = med([RES[m][h]["lam0"] / RES[m][h]["A5"] for m, _ in ks])
    r13 = med([RES[m][h]["A13"] / RES[m][h]["A5"] for m, _ in ks])
    win = sum(1 for m, _ in ks if RES[m][h]["A13"] < RES[m][h]["A5"])
    sig = sum(1 for m, _ in ks if RES[m][h]["p13"] < 0.05 and RES[m][h]["A13"] < RES[m][h]["A5"])
    bad = sum(1 for m, _ in ks if RES[m][h]["p13"] < 0.05 and RES[m][h]["A13"] > RES[m][h]["A5"])
    print(f"{h:>5}{r0:>9.4f}{r13:>9.4f}{win:>8}/{len(ks):<2}{sig:>10}{bad:>10}")

r13 = np.array([RES[m][h]["A13"] / RES[m][h]["A5"] for m, h in cells])
print(f"\n全 25 格：A13/A5 中位 {med(r13):.4f}  胜 {int((r13<1).sum())}/25  "
      f"显著胜 {int(sum(1 for m,h in cells if RES[m][h]['p13']<0.05 and r13[cells.index((m,h))]<1))}  "
      f"显著负 {int(sum(1 for m,h in cells if RES[m][h]['p13']<0.05 and r13[cells.index((m,h))]>1))}")
json.dump({m: {str(h): RES[m][h] for h in RES[m]} for m in RES},
          open(os.path.join(OUT, "A13.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("已写出 A13.json")
