# -*- coding: utf-8 -*-
"""
A14 = A5 原样（位置块 + h^H 尺度块）+ 形状按 w(h)=min(1,h/50) 混合
=================================================================
A13 把两件事绑在一起：去位置块（伤 GZ h=60/120，p=.006/.000）+ 形状混合
（h=5 五格全胜 0.9728）。但 w(h)=min(1,h/50) 在 h>=50 时恒为 1，混合后的
形状就是 GV，与 A5 逐位相同。所以形状混合根本不影响 h=60/120 —— 两个部件
应当解耦。A14 只加形状混合，其余与 A5 完全一致。

  A5  : Q = lam_mu + sd*h^H*GV
  A14 : Q = lam_mu + sd*h^H*[(1-w) xi_d + w GV],  w=min(1,h/50)
  A13 : Q = sd*h^H*[(1-w) xi_d + w GV]          （无位置块，作对照）

w 在 h=5/20/40/60/120 取 0.10/0.40/0.80/1.00/1.00 => A14 与 A5 只在 h<=40 不同。
参数：A14 仍是每市场 2 个（h0,H）= 10 个；常数 50 须披露为来自 oracle w 剖面，
不是拟合值。另报一处敏感性（h_sat=30/50/70）。

排名基准统一为：FULL_ROSTER 的 11 个基准模型（剔除 A5）+ 候选模型 = 12 个。
输出 A14.json
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm, t as tdist
warnings.filterwarnings("ignore")
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


CELL = {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    for h in HOR:
        ts_te = [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP)]
        mu, sd, y, xi, ok = [], [], [], [], []
        for t in ts_te:
            win = lp[max(0, t - W):t]
            rr = win[h:] - win[:-h]
            rd = np.diff(win)
            if len(win) < 120 or len(rr) < 40 or np.std(rd, ddof=1) <= 0:
                ok.append(False); continue
            sdd = np.std(rd, ddof=1)
            mu.append(rr.mean()); sd.append(sdd); y.append(lp[t + h] - lp[t])
            xi.append(np.quantile((rd - rd.mean()) / sdd, TAUS)); ok.append(True)
        mu = np.array(mu); sd = np.array(sd); y = np.array(y)
        xi = np.array(xi); ok = np.array(ok)
        sc = (sd * h ** Hv)[:, None]
        loc = (h0 / (h0 + h)) * mu[:, None]
        rec = dict(n=int(ok.sum()), y=y, ok=ok)
        for tag, hs in (("A5", None), ("A13", 50.0), ("A14", 50.0), ("A14_30", 30.0), ("A14_70", 70.0)):
            if hs is None:
                Q = loc + sc * GV[None, :]
            else:
                w = min(1.0, h / hs)
                xi_m = (1 - w) * xi + w * GV[None, :]
                Q = (sc * xi_m) if tag == "A13" else (loc + sc * xi_m)
            rec[tag] = crps_q(y[ok], Q[ok])
        rec["p_A14"] = dm(rec["A14"], rec["A5"])
        rec["p_A14_13"] = dm(rec["A14"], rec["A13"])
        CELL[(m, h)] = rec

R = json.load(open(os.path.join(OUT, "FULL_ROSTER.json"), encoding="utf-8"))["res"]
RM = {m: {int(h): v for h, v in D.items()} for m, D in R.items()}
BASE = [k for k in RM["CEA"][5]["crps"] if k != "A5"]
cells = [(m, h) for m in CUT for h in HOR]
med = lambda a: float(np.median(a))

print("=" * 126)
print("一、A14 相对 A5（w(h)=min(1,h/50)；h>=50 时与 A5 逐位相同）")
print("=" * 126)
print(f"{'市场':<12}{'h':>5}{'n':>5}{'A5':>10}{'A14':>10}{'变化':>9}{'p':>8}{'w':>6}")
for m, h in cells:
    c = CELL[(m, h)]
    a5 = c["A5"].mean(); a14 = c["A14"].mean()
    print(f"{m:<12}{h:>5}{c['n']:>5}{a5:>10.5f}{a14:>10.5f}"
          f"{(a14/a5-1)*100:>+8.2f}%{c['p_A14']:>8.3f}{min(1.0,h/50.0):>6.2f}")

r = np.array([CELL[(m, h)]["A14"].mean() / CELL[(m, h)]["A5"].mean() for m, h in cells])
sw = int(sum(1 for m, h in cells if CELL[(m, h)]["p_A14"] < .05 and CELL[(m, h)]["A14"].mean() < CELL[(m, h)]["A5"].mean()))
sl = int(sum(1 for m, h in cells if CELL[(m, h)]["p_A14"] < .05 and CELL[(m, h)]["A14"].mean() > CELL[(m, h)]["A5"].mean()))
print(f"\n25 格：中位 {med(r):.4f}  胜 {int((r<1).sum())}/25  显著胜 {sw}  显著负 {sl}")

print("\n" + "=" * 126)
print("二、统一基准排名（11 个名册基准 + 候选 = 12 个；A5 不重复计入）")
print("=" * 126)
print(f"{'模型':<10}{'参数':>6}{'相对最优中位':>14}{'胜':>8}{'平均排名':>10}{'第1名':>8}{'前三':>7}{'最差':>9}")
RANK = {}
for nm, tag, npar in (("A5", "A5", "10"), ("A13", "A13", "5"), ("A14", "A14", "10")):
    bo, rk = [], []
    for m, h in cells:
        mine = CELL[(m, h)][tag].mean()
        c = dict(RM[m][h]["crps"])
        vals = [c[k] for k in BASE]
        bo.append(mine / min(vals))
        rk.append(sum(1 for v in vals if v < mine) + 1)
    bo = np.array(bo); rk = np.array(rk)
    RANK[nm] = dict(bo=bo.tolist(), rk=rk.tolist())
    print(f"{nm:<10}{npar:>6}{med(bo):>14.4f}{int((bo<=1.0).sum()):>5}/{len(bo):<2}"
          f"{rk.mean():>10.2f}{int((rk==1).sum()):>8}{int((rk<=3).sum()):>7}{bo.max():>9.4f}")

print("\n逐期限")
print(f"{'h':>5}{'A14/A5':>10}{'A14/A13':>10}{'A14胜A5':>10}{'显著胜':>8}{'显著负':>8}{'A14排名':>9}{'A5排名':>9}")
for h in HOR:
    ks = [(m, h) for m in CUT]
    print(f"{h:>5}"
          f"{med([CELL[(m,h)]['A14'].mean()/CELL[(m,h)]['A5'].mean() for m,_ in ks]):>10.4f}"
          f"{med([CELL[(m,h)]['A14'].mean()/CELL[(m,h)]['A13'].mean() for m,_ in ks]):>10.4f}"
          f"{sum(1 for m,_ in ks if CELL[(m,h)]['A14'].mean()<CELL[(m,h)]['A5'].mean()):>7}/5"
          f"{sum(1 for m,_ in ks if CELL[(m,h)]['p_A14']<.05 and CELL[(m,h)]['A14'].mean()<CELL[(m,h)]['A5'].mean()):>8}"
          f"{sum(1 for m,_ in ks if CELL[(m,h)]['p_A14']<.05 and CELL[(m,h)]['A14'].mean()>CELL[(m,h)]['A5'].mean()):>8}"
          f"{np.mean([RANK['A14']['rk'][cells.index((m,h))] for m,_ in ks]):>9.2f}"
          f"{np.mean([RANK['A5']['rk'][cells.index((m,h))] for m,_ in ks]):>9.2f}")

print("\n" + "=" * 126)
print("三、w 饱和点敏感性（50 是 oracle 剖面观测值，不是拟合值）")
print("=" * 126)
print(f"{'h_sat':>7}{'相对A5中位':>13}{'胜A5':>8}{'显著胜':>8}{'显著负':>8}{'相对最优中位':>15}{'平均排名':>10}")
for tag, hs in (("A14_30", 30), ("A14", 50), ("A14_70", 70)):
    rr = np.array([CELL[(m, h)][tag].mean() / CELL[(m, h)]["A5"].mean() for m, h in cells])
    s1 = sum(1 for m, h in cells if CELL[(m, h)]["p_A14"] < .05 and CELL[(m, h)][tag].mean() < CELL[(m, h)]["A5"].mean())
    s2 = sum(1 for m, h in cells if CELL[(m, h)]["p_A14"] < .05 and CELL[(m, h)][tag].mean() > CELL[(m, h)]["A5"].mean())
    bo, rk = [], []
    for m, h in cells:
        mine = CELL[(m, h)][tag].mean()
        vals = [RM[m][h]["crps"][k] for k in BASE]
        bo.append(mine / min(vals)); rk.append(sum(1 for v in vals if v < mine) + 1)
    print(f"{hs:>7}{med(rr):>13.4f}{int((rr<1).sum()):>5}/25{s1:>8}{s2:>8}"
          f"{med(bo):>15.4f}{np.mean(rk):>10.2f}")

print("\nA14 取得第 1 名的格子：")
for i, (m, h) in enumerate(cells):
    if RANK["A14"]["rk"][i] == 1:
        print(f"  {m} h={h}  A14 {CELL[(m,h)]['A14'].mean():.5f}  "
              f"A5 {CELL[(m,h)]['A5'].mean():.5f}  最优基准 {min(RM[m][h]['crps'][k] for k in BASE):.5f}")
print("A14 排名最差的 4 格：")
for i in sorted(range(len(cells)), key=lambda i: -RANK["A14"]["rk"][i])[:4]:
    m, h = cells[i]
    print(f"  {m} h={h}  A14 {CELL[(m,h)]['A14'].mean():.5f}  排名 {RANK['A14']['rk'][i]}  "
          f"相对最优 {RANK['A14']['bo'][i]:.4f}")

OUTJ = {"cells": {f"{m}|{h}": {k: (float(np.mean(v)) if isinstance(v, np.ndarray) else float(v))
                               for k, v in CELL[(m, h)].items()} for m, h in cells},
        "rank": {k: dict(bo=v["bo"], rk=v["rk"], mean_rank=float(np.mean(v["rk"])),
                         firsts=int(sum(1 for x in v["rk"] if x == 1)),
                         top3=int(sum(1 for x in v["rk"] if x <= 3)))
                 for k, v in RANK.items()},
        "base": BASE, "h_sat": 50.0}
json.dump(OUTJ, open(os.path.join(OUT, "A14.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 A14.json")
