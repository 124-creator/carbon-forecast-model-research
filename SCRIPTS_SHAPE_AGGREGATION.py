# -*- coding: utf-8 -*-
"""
A7 = lam(h)*mu + sdd*h^H * [ (1-w)*xi_d + w*G ],  w = h/(h+h_s)
    h_s -> 0   等价 A6（纯日频形状）
    h_s -> inf 等价 A5（纯高斯形状）

理论：h 步收益 ≈ h 个日冲击之和，形状随 h 向高斯收敛（聚合效应）。
若成立，最优 w 应随 h 单调上升。

两件事：
 (1) 诊断：逐格求使 CRPS 最小的 w（oracle），看 w 是否随 h 上升。
 (2) 诚实检验：h_s 在测试期前一半拟合，后一半评估，与 A5/A6/EMP 比。
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
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
HOR = [5, 20, 40, 60, 120]
STEP, W = 2, 250
HS_GRID = np.array([1e-6, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 400, 1e6])
WG = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


def crps_row(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


def dm_hln(a, b, L=10):
    d = a - b; n = len(d)
    v = (d * d).mean()
    for l in range(1, min(L, n - 2) + 1):
        v += 2 * (1 - l / (L + 1)) * (d[l:] * d[:-l]).mean()
    if v <= 0:
        return 1.0
    t = d.mean() / np.sqrt(v / n)
    hln = np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    return float(2 * (1 - norm.cdf(abs(t * hln))))


def cells(m, h):
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)); dt = pd.to_datetime(d.date).to_numpy()
    cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
    t0 = int(np.searchsorted(dt, np.datetime64(cut)))
    h0, Hv = PAR[m]
    ts = list(range(max(W + h + 5, t0), len(lp) - h, STEP))
    base, xid, y = [], [], []
    for t in ts:
        win = lp[t - W:t]
        rd = np.diff(win); rr = win[h:] - win[:-h]
        mu = rr.mean(); sdh = np.std(rr, ddof=1); sdd = np.std(rd, ddof=1)
        base.append(mu * (h0 / (h0 + h)) + sdd * h ** Hv * GV)
        xid.append(sdd * h ** Hv * np.quantile((rd - rd.mean()) / sdd, TAUS))
        y.append(lp[t + h] - lp[t])
    y = np.array(y)
    return y, np.array(base), np.array(xid)


print("=" * 118)
print("(1) 诊断：逐格最优 w（w=0 纯日频形状，w=1 纯高斯）")
print("=" * 118)
print(f"{'h':>4} {'市场':<11}{'oracle w':>10}{'CRPS(w*)':>10}{'w=0':>9}{'w=1':>9}"
      f"{'改善%':>8}{'最优 h_s':>10}")
ORES = {}
for h in HOR:
    ws = []
    for m in MK:
        y, B, X = cells(m, h)
        c = []
        for w in WG:
            Q = (1 - w) * X + w * B if w < 1 else B
            c.append(crps_row(y, Q).mean())
        c = np.array(c); i = int(np.argmin(c)); wst = WG[i]
        ws.append(wst)
        hc = []
        for s in HS_GRID:
            ww = h / (h + s)
            hc.append(crps_row(y, (1 - ww) * X + ww * B).mean())
        hs = HS_GRID[int(np.argmin(hc))]
        ORES[f"{m}|{h}"] = dict(w=float(wst), c=c.tolist(), wg=WG.tolist(),
                                crps_w0=float(c[0]), crps_w1=float(c[-1]),
                                crps_best=float(c[i]), hs=float(hs))
        print(f"{h:>4} {m:<11}{wst:>10.2f}{c[i]:>10.5f}{c[0]:>9.5f}{c[-1]:>9.5f}"
              f"{(c[-1]/c[i]-1)*100:>8.1f}{hs:>10.3g}")
    print("-" * 118)

print("\n最优 w 随 h 的变化（5 市场）")
print(f"{'h':>5}  " + "  ".join(f"{m[:6]:>7}" for m in MK) + f"{'中位':>8}")
for h in HOR:
    v = [ORES[f'{m}|{h}']['w'] for m in MK]
    print(f"{h:>5}  " + "  ".join(f"{x:>7.2f}" for x in v) + f"{np.median(v):>8.2f}")

print("\n" + "=" * 118)
print("(2) 诚实检验：h_s 只在测试期前 50% 原点拟合，后 50% 评估")
print("=" * 118)
print(f"{'h':>4} {'市场':<11}{'h_s*':>8}{'A7':>9}{'A6':>9}{'A5':>9}{'EMP':>9} | "
      f"{'A7/EMP':>8}{'A7/A6':>7}{'A7/A5':>7}{'p(EMP)':>8}")
FIT = {}
for h in HOR:
    for m in MK:
        y, B, X = cells(m, h)
        n = len(y); k = n // 2
        best, bh = 1e18, None
        for s in HS_GRID:
            w = h / (h + s)
            Q = (1 - w) * X[:k] + w * B[:k]
            c = crps_row(y[:k], Q).mean()
            if c < best:
                best, bh = c, s
        w = h / (h + bh)
        Qs = {"A7": (1 - w) * X[k:] + w * B[k:], "A6": X[k:], "A5": B[k:],
              "EMP": None}
        d = p[p.market == m].sort_values("date").reset_index(drop=True)
        lp = np.log(d.price.to_numpy(float)); dt = pd.to_datetime(d.date).to_numpy()
        cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
        t0 = int(np.searchsorted(dt, np.datetime64(cut)))
        ts = list(range(max(W + h + 5, t0), len(lp) - h, STEP))[k:]
        E = []
        for t in ts:
            win = lp[t - W:t]; rr = win[h:] - win[:-h]
            E.append(rr.mean() + np.std(rr, ddof=1) * np.quantile((rr - rr.mean()) / np.std(rr, ddof=1), TAUS))
        Qs["EMP"] = np.array(E)
        yv = y[k:]
        L = {kk: crps_row(yv, Qs[kk]) for kk in Qs}
        c = {kk: float(v.mean()) for kk, v in L.items()}
        FIT[f"{m}|{h}"] = dict(hs=float(bh), crps=c, ratio_emp=c["A7"] / c["EMP"],
                               ratio_a6=c["A7"] / c["A6"], ratio_a5=c["A7"] / c["A5"],
                               p_emp=dm_hln(L["A7"], L["EMP"]))
        print(f"{h:>4} {m:<11}{bh:>8.3g}{c['A7']:>9.5f}{c['A6']:>9.5f}{c['A5']:>9.5f}"
              f"{c['EMP']:>9.5f} | {c['A7']/c['EMP']:>8.3f}{c['A7']/c['A6']:>7.3f}"
              f"{c['A7']/c['A5']:>7.3f}{FIT[f'{m}|{h}']['p_emp']:>8.3f}")
    print("-" * 118)

print("\n汇总（5 市场中位）")
print(f"{'h':>5}{'A7/EMP':>9}{'胜':>7}{'A7/A6':>8}{'胜':>7}{'A7/A5':>8}{'胜':>7}{'中位 h_s*':>11}")
for h in HOR:
    ks = [f"{m}|{h}" for m in MK]
    re = np.array([FIT[k]["ratio_emp"] for k in ks])
    r6 = np.array([FIT[k]["ratio_a6"] for k in ks])
    r5 = np.array([FIT[k]["ratio_a5"] for k in ks])
    print(f"{h:>5}{np.median(re):>9.3f}{int((re<1).sum()):>4}/5"
          f"{np.median(r6):>8.3f}{int((r6<1).sum()):>4}/5"
          f"{np.median(r5):>8.3f}{int((r5<1).sum()):>4}/5"
          f"{np.median([FIT[k]['hs'] for k in ks]):>11.3g}")
al = np.array([FIT[k]["ratio_emp"] for k in FIT])
print(f"\n全部 {len(al)} 格：A7/EMP 中位 {np.median(al):.3f} 胜 {int((al<1).sum())}/{len(al)}")

json.dump(dict(oracle=ORES, fitted=FIT),
          open(os.path.join(OUT, "SHAPE_AGGREGATION.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 SHAPE_AGGREGATION.json")
