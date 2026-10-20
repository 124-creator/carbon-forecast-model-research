# -*- coding: utf-8 -*-
"""
DM 检验的带宽稳健性 + Holm
==========================
前文只用 L=10 的 Bartlett HAC。审稿人会问"带宽是不是挑出来的"。
这里对 L in {6,10,20,40} 各算一次 HLN 校正的 DM，并在 25 格上做 Holm，
看"对 RW 无显著格"这个结论是否依赖带宽。
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
HOR = [5, 20, 40, 60, 120]
W, STEP = 250, 2
LS = [6, 10, 20, 40]
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}


def crps_row(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


def dm_hln(a, b, L):
    d = a - b; n = len(d); db = d.mean()
    v = (d * d).mean()
    for l in range(1, min(L, n - 2) + 1):
        v += 2 * (1 - l / (L + 1)) * (d[l:] * d[:-l]).mean()
    if v <= 0:
        return np.nan, 1.0
    t = db / np.sqrt(v / n)
    hln = np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    return float(t * hln), float(2 * (1 - norm.cdf(abs(t * hln))))


def holm(ps):
    m = len(ps); order = np.argsort(ps); adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(run, 1.0)
    return adj


print("=" * 130)
print("DM（HLN 校正 + Bartlett HAC）在不同带宽下的显著性，25 格")
print("=" * 130)

CELLS, ROWS = [], {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float))
    dt = pd.to_datetime(d.date).to_numpy()
    cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
    t0 = int(np.searchsorted(dt, np.datetime64(cut)))
    h0, Hv = PAR[m]
    for h in HOR:
        start = max(W + h + 5, t0)
        ts = list(range(start, len(lp) - h, STEP))
        if len(ts) < 40:
            continue
        Qa, Qe, Qw, y = [], [], [], []
        for t in ts:
            win = lp[t - W:t]
            rr = win[h:] - win[:-h]; rd = np.diff(win)
            mu = rr.mean(); sdh = np.std(rr, ddof=1); sdd = np.std(rd, ddof=1)
            xi = np.quantile((rr - rr.mean()) / sdh, TAUS)
            Qa.append(mu * (h0 / (h0 + h)) + sdd * h ** Hv * GV)
            Qe.append(mu + sdh * xi)
            Qw.append(sdd * np.sqrt(h) * GV)
            y.append(lp[t + h] - lp[t])
        y = np.array(y)
        La = crps_row(y, np.array(Qa)); Le = crps_row(y, np.array(Qe))
        Qw = np.array(Qw).reshape(len(y), -1)
        Lw = crps_row(y, Qw)
        cell = f"{m}|{h}"
        CELLS.append(cell)
        ROWS[cell] = dict(n=len(y), crps_a5=float(La.mean()), crps_emp=float(Le.mean()),
                          crps_rw=float(Lw.mean()),
                          **{f"dme_L{L}": dm_hln(La, Le, L) for L in LS},
                          **{f"dmw_L{L}": dm_hln(La, Lw, L) for L in LS})

print(f"\n{'市场':<12}{'h':>4}{'A5':>10}{'EMP':>10}{'RW':>10} | "
      + "".join([f"DM-EMP L{L:<3}" for L in LS]) + " | " + "".join([f"DM-RW L{L:<3}" for L in LS]))
for c in CELLS:
    m, h = c.split("|"); r = ROWS[c]
    print(f"{m:<12}{h:>4}{r['crps_a5']:>10.5f}{r['crps_emp']:>10.5f}{r['crps_rw']:>10.5f} | "
          + "".join([f"{r[f'dme_L{L}'][1]:>10.3f}" for L in LS]) + " | "
          + "".join([f"{r[f'dmw_L{L}'][1]:>10.3f}" for L in LS]))

print("\n" + "=" * 130)
print("汇总：p<0.05 的格数（raw）与 Holm 后")
print("=" * 130)
for base, lab in (("dme", "A5 vs EMP"), ("dmw", "A5 vs RW")):
    print(f"\n  {lab}")
    print(f"  {'L':>4}{'raw p<0.05':>12}{'Holm<0.05':>12}{'A5 更好(Holm)':>16}{'A5 更差(Holm)':>16}")
    for L in LS:
        ps = np.array([ROWS[c][f"{base}_L{L}"][1] for c in CELLS])
        ts = np.array([ROWS[c][f"{base}_L{L}"][0] for c in CELLS])
        adj = holm(ps)
        sig = adj < 0.05
        print(f"  {L:>4}{int((ps<0.05).sum()):>12}{int(sig.sum()):>12}"
              f"{int((sig & (ts<0)).sum()):>16}{int((sig & (ts>0)).sum()):>16}")

json.dump(ROWS, open(os.path.join(OUT, "DM_BANDWIDTH.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 DM_BANDWIDTH.json")
