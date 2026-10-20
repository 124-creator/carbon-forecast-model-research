# -*- coding: utf-8 -*-
"""修正 MCS bootstrap 索引；同时输出 A5 与 A6 的区间覆盖率对照。"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm, chi2
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
STEP = 2
MODS = ["A6", "A5", "EMP", "RW"]


def crps_row(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


def dm_hln(a, b, L):
    d = a - b; n = len(d)
    v = (d * d).mean()
    for l in range(1, min(L, n - 2) + 1):
        v += 2 * (1 - l / (L + 1)) * (d[l:] * d[:-l]).mean()
    if v <= 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt(v / n)
    hln = np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    return float(t * hln), float(2 * (1 - norm.cdf(abs(t * hln))))


def build(m, h, W):
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)); dt = pd.to_datetime(d.date).to_numpy()
    cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
    t0 = int(np.searchsorted(dt, np.datetime64(cut)))
    h0, Hv = PAR[m]
    ts = list(range(max(W + h + 5, t0), len(lp) - h, STEP))
    if len(ts) < 30:
        return None
    Q = {k: [] for k in MODS}; y = []
    for t in ts:
        win = lp[t - W:t]
        rd = np.diff(win); rr = win[h:] - win[:-h]
        mu = rr.mean(); sdh = np.std(rr, ddof=1); sdd = np.std(rd, ddof=1)
        loc = mu * (h0 / (h0 + h)); sc = sdd * h ** Hv
        Q["A6"].append(loc + sc * np.quantile((rd - rd.mean()) / sdd, TAUS))
        Q["A5"].append(loc + sc * GV)
        Q["EMP"].append(mu + sdh * np.quantile((rr - rr.mean()) / sdh, TAUS))
        Q["RW"].append(sdd * np.sqrt(h) * GV)
        y.append(lp[t + h] - lp[t])
    y = np.array(y)
    return y, {k: np.array(v) for k, v in Q.items()}


def cov(y, Q, a, b):
    return float(np.mean((y >= Q[:, a]) & (y <= Q[:, b])))


def mcs(loss, alpha=0.10, B=400, block=10, seed=7):
    M = list(loss.keys()); S = np.column_stack([loss[k] for k in M])
    n = len(S); cur = list(range(len(M)))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(B, nb * block)
    elim = []
    while len(cur) > 1:
        sub = S[:, cur]; d = sub.mean(0)
        D = np.array([sub[i].mean(0) for i in idx])
        se = D.std(0, ddof=1); se[se <= 0] = 1e-12
        ts = (d - d.min()) / se
        tb = (D - D.min(axis=1)[:, None]) / se
        crit = np.quantile(tb.max(1), 1 - alpha)
        if ts.max() <= crit:
            break
        j = int(np.argmax(ts)); elim.append(M[cur[j]]); cur.pop(j)
    return [M[i] for i in cur], elim


R = {}
print("=" * 128)
print("覆盖率对照：A6(日频形状) vs A5(高斯形状) vs EMP    W=250")
print("=" * 128)
print(f"{'h':>4} {'市场':<11} | {'A6c80':>7}{'A5c80':>7}{'EMPc80':>8} | "
      f"{'A6c95':>7}{'A5c95':>7}{'EMPc95':>8} | {'A6/EMP':>7}{'A6/A5':>7}")
for h in HOR:
    for m in MK:
        b = build(m, h, 250)
        if b is None:
            continue
        y, Q = b
        L = {k: crps_row(y, Q[k]) for k in MODS}
        cr = {k: float(v.mean()) for k, v in L.items()}
        r = dict(crps=cr, n=len(y), neff=len(y) * STEP / h,
                 c80={k: cov(y, Q[k], 2, 12) for k in ("A6", "A5", "EMP")},
                 c95={k: cov(y, Q[k], 0, -1) for k in ("A6", "A5", "EMP")},
                 p_emp=dm_hln(L["A6"], L["EMP"], 10)[1],
                 p_rw=dm_hln(L["A6"], L["RW"], 10)[1],
                 p_a5=dm_hln(L["A6"], L["A5"], 10)[1])
        r["ratio_emp"] = cr["A6"] / cr["EMP"]; r["ratio_rw"] = cr["A6"] / cr["RW"]
        r["ratio_a5"] = cr["A6"] / cr["A5"]
        R[f"{m}|{h}"] = r
        print(f"{h:>4} {m:<11} | {r['c80']['A6']:>7.3f}{r['c80']['A5']:>7.3f}"
              f"{r['c80']['EMP']:>8.3f} | {r['c95']['A6']:>7.3f}{r['c95']['A5']:>7.3f}"
              f"{r['c95']['EMP']:>8.3f} | {r['ratio_emp']:>7.3f}{r['ratio_a5']:>7.3f}")
    print("-" * 128)

print("\n" + "=" * 112)
print("覆盖率汇总（5 市场中位，名义 0.80 / 0.95）")
print("=" * 112)
print(f"{'h':>5}{'A6c80':>8}{'A5c80':>8}{'EMPc80':>8} | {'A6c95':>8}{'A5c95':>8}{'EMPc95':>8}")
for h in HOR:
    ks = [k for k in R if k.endswith(f"|{h}")]
    g = lambda d, kk: np.median([R[k][d][kk] for k in ks])
    print(f"{h:>5}{g('c80','A6'):>8.3f}{g('c80','A5'):>8.3f}{g('c80','EMP'):>8.3f} | "
          f"{g('c95','A6'):>8.3f}{g('c95','A5'):>8.3f}{g('c95','EMP'):>8.3f}")

print("\n" + "=" * 128)
print("4 模型 MCS（A6/A5/EMP/RW），alpha=0.10, B=400, block=10")
print("=" * 128)
SURV = {}
for h in HOR:
    for m in MK:
        y, Q = build(m, h, 250)
        L = {k: crps_row(y, Q[k]) for k in MODS}
        surv, elim = mcs(L)
        SURV[f"{m}|{h}"] = dict(surv=surv, elim=elim)
        print(f"h={h:>3} {m:<11} 存活 {surv}   淘汰 {elim}")

print("\n" + "=" * 112)
print("W 网格稳健性（中位 A6/EMP | A5/EMP）")
print("=" * 112)
WG = [125, 250, 500]
print(f"{'h':>5}" + "".join(f"{'W='+str(w):>19}" for w in WG))
for h in (20, 40, 60, 120):
    row = f"{h:>5}"
    for W in WG:
        v6, v5 = [], []
        for m in MK:
            b = build(m, h, W)
            if b is None:
                continue
            y, Q = b
            c = {k: crps_row(y, Q[k]).mean() for k in ("A6", "A5", "EMP")}
            v6.append(c["A6"] / c["EMP"]); v5.append(c["A5"] / c["EMP"])
        row += (f"{np.median(v6):>10.3f}/{np.median(v5):<8.3f}" if v6 else f"{'—':>19}")
    print(row)

json.dump(dict(cells=R, mcs=SURV),
          open(os.path.join(OUT, "A6_BATTERY.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 A6_BATTERY.json")
