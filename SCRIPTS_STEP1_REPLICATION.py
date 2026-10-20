# -*- coding: utf-8 -*-
"""
STEP=1 加固：把原点密度翻倍，有效样本量翻倍
==========================================
24.20 指出 n_eff ~= n_te*STEP/h，h=120 时每格只有 1.2-4.5 个有效独立观测。
最直接的补救是把原点从隔 2 天改为逐日（STEP=1），n_eff 翻倍。
本节检验：结论是否随原点密度改变。

(A5,H) 沿用 24.3 的估计值（PAR），不重新调参——本节只检验"原点密度"这一个维度，
若重调参就会混淆两件事。
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
W = 250
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


print("=" * 146)
print("STEP=1 加固：A5 vs EMP vs RW，以及与 STEP=2 的对照")
print("=" * 146)

R = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float))
    dt = pd.to_datetime(d.date).to_numpy()
    cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
    t0 = int(np.searchsorted(dt, np.datetime64(cut)))
    h0, Hv = PAR[m]
    print(f"\n--- {m} ---")
    print(f"{'h':>4}{'n_te2':>7}{'n_te1':>7}{'neff2':>7}{'neff1':>7} | "
          f"{'A5/EMP 2':>10}{'A5/EMP 1':>10}{'Δ':>7} | {'A5cov 2':>9}{'A5cov 1':>9} | "
          f"{'EMPcov 2':>9}{'EMPcov 1':>9} | {'pRW@L=h':>9}")
    for h in HOR:
        out = {}
        for STEP in (2, 1):
            start = max(W + h + 5, t0)
            ts = list(range(start, len(lp) - h, STEP))
            if len(ts) < 40:
                out[STEP] = None; continue
            Qa, Qe, Qw, y = [], [], [], []
            for t in ts:
                win = lp[t - W:t]
                rr = win[h:] - win[:-h]; rd = np.diff(win)
                mu = rr.mean(); sdh = np.std(rr, ddof=1); sdd = np.std(rd, ddof=1)
                xi = np.quantile((rr - rr.mean()) / sdh, TAUS)
                c = mu * (h0 / (h0 + h)); s = sdd * h ** Hv
                Qa.append(c + s * GV)
                Qe.append(mu + sdh * xi)
                Qw.append(sdd * np.sqrt(h) * GV)
                y.append(lp[t + h] - lp[t])
            y = np.array(y)
            La = crps_row(y, np.array(Qa)); Le = crps_row(y, np.array(Qe))
            Lw = crps_row(y, np.array(Qw))
            Qa = np.array(Qa); Qe = np.array(Qe)
            cova = np.mean((y >= Qa[:, 0]) & (y <= Qa[:, -1]))
            cove = np.mean((y >= Qe[:, 0]) & (y <= Qe[:, -1]))
            out[STEP] = dict(n=len(y), ca=La.mean(), ce=Le.mean(), cw=Lw.mean(),
                             cova=cova, cove=cove, La=La, Lw=Lw,
                             dmw=dm_hln(La, Lw, h), dme=dm_hln(La, Le, h))
        a2, a1 = out.get(2), out.get(1)
        if a1 is None:
            print(f"{h:>4}  STEP=1 样本不足"); continue
        r2 = a2["ca"] / a2["ce"] if a2 else np.nan
        r1 = a1["ca"] / a1["ce"]
        R[f"{m}|{h}"] = dict(
            n2=a2["n"] if a2 else None, n1=a1["n"],
            neff2=(a2["n"] * 2 / h) if a2 else None, neff1=a1["n"] / h,
            ratio2=float(r2) if a2 else None, ratio1=float(r1),
            cova2=float(a2["cova"]) if a2 else None, cova1=float(a1["cova"]),
            cove2=float(a2["cove"]) if a2 else None, cove1=float(a1["cove"]),
            dmw1_p=float(a1["dmw"][1]), dmw1_t=float(a1["dmw"][0]),
            dme1_p=float(a1["dme"][1]))
        print(f"{h:>4}{a2['n'] if a2 else 0:>7}{a1['n']:>7}"
              f"{(a2['n']*2/h) if a2 else 0:>7.1f}{a1['n']/h:>7.1f} | "
              f"{r2:>10.3f}{r1:>10.3f}{r1-r2:>+7.3f} | "
              f"{a2['cova'] if a2 else 0:>9.3f}{a1['cova']:>9.3f} | "
              f"{a2['cove'] if a2 else 0:>9.3f}{a1['cove']:>9.3f} | "
              f"{a1['dmw'][1]:>9.3f}")

print("\n" + "=" * 146)
print("按期限汇总")
print("=" * 146)
print(f"{'h':>4}{'格数':>6}{'n_eff':>8}{'A5/EMP(STEP1) 中位':>20}{'A5 胜':>8}"
      f"{'A5cov 中位':>12}{'EMPcov 中位':>13}{'A5 显著胜 RW':>14}")
for h in HOR:
    ks = [k for k in R if k.endswith(f"|{h}")]
    if not ks:
        continue
    rr = np.array([R[k]["ratio1"] for k in ks])
    cc = np.array([R[k]["cova1"] for k in ks])
    ee = np.array([R[k]["cove1"] for k in ks])
    sw = sum(1 for k in ks if R[k]["dmw1_p"] < 0.05 and R[k]["dmw1_t"] < 0)
    print(f"{h:>4}{len(ks):>6}{np.mean([R[k]['neff1'] for k in ks]):>8.1f}"
          f"{np.median(rr):>20.3f}{(rr<1).sum():>5}/{len(ks)}"
          f"{np.median(cc):>12.3f}{np.median(ee):>13.3f}{sw:>10}/{len(ks)}")

rr = np.array([R[k]["ratio1"] for k in R])
print(f"\n全部 {len(R)} 格（STEP=1）：A5/EMP 中位 {np.median(rr):.3f}，A5 胜 {(rr<1).sum()}/{len(R)}")
print(f"STEP=2 时对应值：中位 "
      f"{np.median([R[k]['ratio2'] for k in R if R[k]['ratio2']]):.3f}，"
      f"胜 {sum(1 for k in R if R[k]['ratio2'] and R[k]['ratio2']<1)}/{len(R)}")

json.dump(R, open(os.path.join(OUT, "STEP1_REPLICATION.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 STEP1_REPLICATION.json")
