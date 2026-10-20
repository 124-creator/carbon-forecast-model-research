# -*- coding: utf-8 -*-
"""
A12：位置块的全局收缩系数 kappa（1 个参数，不是每市场 1 个）
==========================================================
诊断：位置块在训练段上四种函数形式的差异 <0.3%（不可辨），而测试段
  A5 (kappa=1) 在 EU h=120 排 11/12，lambda0 (kappa=0) 在 GZ h=120 丢第一名。
=> 两个端点各有胜负，逐市场拟合 h0 等于拟合噪声。改用单一全局标量：
      lambda(h) = kappa * h0/(h0+h)
  kappa 在**训练段**（跨 5 市场 25 格合并）上估出，测试段只做外推评价。

注意与 §24.x 已弃用的 "kappa 宽度乘子" 区分：那个作用在区间宽度上，本节的
kappa 作用在位置项上，是完全不同的对象。

输出 KAPPA_FIT.json
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
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
KS = np.round(np.arange(0.0, 1.001, 0.05), 2)


def crps_q(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


B = {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    B[m] = {}
    for h in HOR:
        for tag, ts in (("tr", [t for t in range(W + h + 5, t0 - h - 1, STEP)]),
                        ("te", [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP)])):
            mu, sd, yy, ok = [], [], [], []
            for t in ts:
                win = lp[max(0, t - W):t]
                rr = win[h:] - win[:-h]
                rd = np.diff(win)
                if len(win) < 120 or len(rr) < 40 or np.std(rd, ddof=1) <= 0:
                    mu.append(np.nan); sd.append(np.nan); yy.append(np.nan)
                    ok.append(False); continue
                mu.append(rr.mean()); sd.append(np.std(rd, ddof=1))
                yy.append(lp[t + h] - lp[t]); ok.append(True)
            B[m][(h, tag)] = (np.array(mu), np.array(sd), np.array(yy), np.array(ok))
    assert not (set(range(W + HOR[0] + 5, t0)) & set(range(t0, len(lp)))), "泄漏"


def cell_crps(m, h, tag, kap):
    mu, sd, y, ok = B[m][(h, tag)]
    h0, Hv = PAR[m]
    lam = kap * (h0 / (h0 + h))
    Q = lam * mu[:, None] + (sd * h ** Hv)[:, None] * GV[None, :]
    return float(crps_q(y[ok], Q[ok]).mean())


print("=" * 132)
print("一、kappa 在训练段的最小值（合并 5 市场 × 5 期限，按格数加权）")
print("=" * 132)
tr = {}
for kap in KS:
    tot = 0.0
    for m in CUT:
        for h in HOR:
            n = int(B[m][(h, "tr")][3].sum())
            tot += cell_crps(m, h, "tr", kap) * n
    tr[float(kap)] = tot
kb = min(tr, key=tr.get)
print(f"{'kappa':>7}{'训练目标':>14}{'相对 kappa=1':>14}")
for kap, v in tr.items():
    star = " <<<" if kap == kb else ""
    print(f"{kap:>7.2f}{v:>14.3f}{(v/tr[1.0]-1)*100:>13.2f}%{star}")
print(f"\n训练段最优 kappa* = {kb}   （训练目标 {tr[kb]:.3f}，kappa=1 时 {tr[1.0]:.3f}，"
      f"改善 {(tr[kb]/tr[1.0]-1)*100:+.2f}%）")

print("\n" + "=" * 132)
print("二、测试段：kappa* 与两个端点对比（25 格）")
print("=" * 132)
R = json.load(open(os.path.join(OUT, "FULL_ROSTER.json"), encoding="utf-8"))["res"]
RM = {m: {int(h): v for h, v in D.items()} for m, D in R.items()}
cells = [(m, h) for m in CUT for h in HOR]
te = {kap: {(m, h): cell_crps(m, h, "te", kap) for m, h in cells} for kap in (0.0, 1.0, kb)}
med = lambda a: float(np.median(a))

print(f"{'kappa':>7}{'/A5 中位':>11}{'胜A5':>8}{'相对最优中位':>14}{'胜':>7}{'平均排名':>10}{'第1名':>8}")
for kap in (1.0, 0.0, kb):
    r = np.array([te[kap][(m, h)] / te[1.0][(m, h)] for m, h in cells])
    bo, rk = [], []
    for m, h in cells:
        c = dict(RM[m][h]["crps"]); c["MINE"] = te[kap][(m, h)]
        bo.append(c["MINE"] / min(v for k, v in c.items() if k != "MINE"))
        rk.append(sorted(c, key=c.get).index("MINE") + 1)
    bo = np.array(bo); rk = np.array(rk)
    lab = {1.0: "1.00 (=A5)", 0.0: "0.00 (λ0)"}.get(kap, f"{kap:.2f} (kappa*)")
    print(f"{lab:>12}{med(r):>11.4f}{int((r<1).sum()):>5}/{len(r):<2}"
          f"{med(bo):>14.4f}{int((bo<=1.0).sum()):>5}/{len(bo):<2}{rk.mean():>10.2f}{int((rk==1).sum()):>8}")

print("\n逐期限（相对 kappa=1）")
print(f"{'h':>5}{'kappa*':>10}{'lambda0':>10}{'kappa* 胜':>11}")
for h in HOR:
    ks = [(m, h) for m in CUT]
    print(f"{h:>5}{med([te[kb][(m,h)]/te[1.0][(m,h)] for m,_ in ks]):>10.4f}"
          f"{med([te[0.0][(m,h)]/te[1.0][(m,h)] for m,_ in ks]):>10.4f}"
          f"{int(sum(1 for m,_ in ks if te[kb][(m,h)]<te[1.0][(m,h)])):>8}/{len(ks)}")

print("\n逐格：kappa* 明显变化（>1%）的格子")
print(f"{'市场':<12}{'h':>5}{'A5':>10}{'kappa*':>10}{'变化':>9}{'名册最优':>10}{'A5排':>7}{'k*排':>7}")
for m, h in cells:
    ch = te[kb][(m, h)] / te[1.0][(m, h)] - 1
    if abs(ch) > 0.01:
        c = dict(RM[m][h]["crps"])
        bv = min(c.values())
        ra = sorted(c, key=c.get).index("A5") + 1
        c2 = dict(c); c2["MINE"] = te[kb][(m, h)]
        r0 = sorted(c2, key=c2.get).index("MINE") + 1
        print(f"{m:<12}{h:>5}{te[1.0][(m,h)]:>10.5f}{te[kb][(m,h)]:>10.5f}"
              f"{ch*100:>+8.1f}%{bv:>10.5f}{ra:>7}{r0:>7}")

json.dump(dict(kappa=kb, train={str(k): v for k, v in tr.items()},
               test={m: {str(h): dict(a5=te[1.0][(m, h)], lam0=te[0.0][(m, h)],
                                      kstar=te[kb][(m, h)]) for h in HOR} for m in CUT}),
          open(os.path.join(OUT, "KAPPA_FIT.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 KAPPA_FIT.json")
