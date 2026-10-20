# -*- coding: utf-8 -*-
"""
完整期限剖面：h = 1,2,3,5,10,20,40,60,120
==========================================
中心命题是"建模收益随期限单调"。前面所有表都是 5 个期限的稀疏点，
这一节把剖面补密，并加入 h=1,2,3 三个短期限作为**安慰剂检验**：

  h=1 时 h^H = 1，A5 的尺度块就是 sdd*1*xi，与 EMP 的 sdh*xi 在数值上几乎相同
  （h=1 时 h 步收益就是日收益）。所以 A5 在 h=1 上**应该没有优势**。
  如果 A5 在 h=1 上也大幅优于 EMP，说明所谓"长期限增益"其实是普遍性的拟合优势，
  机制解释就是错的。这是对本文主张最强的一次可证伪检验。
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
HOR = [1, 2, 3, 5, 10, 20, 40, 60, 120]
W, STEP = 250, 2
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
        return 0.0, 1.0
    t = db / np.sqrt(v / n)
    hln = np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    return float(t * hln), float(2 * (1 - norm.cdf(abs(t * hln))))


print("=" * 150)
print("完整期限剖面 + h=1,2,3 安慰剂检验")
print("=" * 150)
R = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float))
    dt = pd.to_datetime(d.date).to_numpy()
    cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
    t0 = int(np.searchsorted(dt, np.datetime64(cut)))
    h0, Hv = PAR[m]
    print(f"\n--- {m} ---")
    print(f"{'h':>4}{'n_te':>6}{'neff':>7} | {'EMP':>9}{'A5':>9}{'RW':>9} | "
          f"{'A5/EMP':>8}{'A5/RW':>7}{'p(EMP)':>8}{'p(RW)':>8} | "
          f"{'covA5':>7}{'covEMP':>7}{'sigA5':>7}")
    for h in HOR:
        lo = max(W + h + 5, t0)
        ts = list(range(lo, len(lp) - h, STEP))
        if len(ts) < 40:
            print(f"{h:>4}  跳过 n_te={len(ts)}")
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
        y = np.array(y); Qa = np.array(Qa); Qe = np.array(Qe)
        La, Le, Lw = crps_row(y, Qa), crps_row(y, Qe), crps_row(y, Qw)
        cova = np.mean((y >= Qa[:, 0]) & (y <= Qa[:, -1]))
        cove = np.mean((y >= Qe[:, 0]) & (y <= Qe[:, -1]))
        siga = np.mean((y >= Qa[:, 13]) & (y <= Qa[:, 14]))
        pa = dm_hln(La, Le, 10); pw = dm_hln(La, Lw, 10)
        R[f"{m}|{h}"] = dict(n_te=len(y), neff=len(y) * STEP / h,
                             crps_emp=float(Le.mean()), crps_a5=float(La.mean()),
                             crps_rw=float(Lw.mean()), cov_a5=float(cova),
                             cov_emp=float(cove), cov_a5_upper=float(siga),
                             p_emp=pa[1], p_rw=pw[1], t_rw=pw[0], t_emp=pa[0])
        print(f"{h:>4}{len(y):>6}{len(y)*STEP/h:>7.1f} | {Le.mean():>9.5f}{La.mean():>9.5f}"
              f"{Lw.mean():>9.5f} | {La.mean()/Le.mean():>8.3f}{La.mean()/Lw.mean():>7.3f}"
              f"{pa[1]:>8.3f}{pw[1]:>8.3f} | {cova:>7.3f}{cove:>7.3f}{siga:>7.3f}")

print("\n" + "=" * 150)
print("汇总：中心命题的单表证据")
print("=" * 150)
print(f"{'h':>4}{'平均neff':>9}{'A5/EMP 中位':>13}{'A5 胜':>8}{'A5/RW 中位':>12}"
      f"{'A5 胜RW':>9}{'A5cov 中位':>12}{'EMPcov 中位':>13}")
for h in HOR:
    ks = [k for k in R if k.endswith(f"|{h}")]
    if not ks:
        continue
    ra = np.array([R[k]["crps_a5"] / R[k]["crps_emp"] for k in ks])
    rw = np.array([R[k]["crps_a5"] / R[k]["crps_rw"] for k in ks])
    ca = np.array([R[k]["cov_a5"] for k in ks])
    ce = np.array([R[k]["cov_emp"] for k in ks])
    print(f"{h:>4}{np.mean([R[k]['neff'] for k in ks]):>9.1f}{np.median(ra):>13.3f}"
          f"{(ra<1).sum():>5}/{len(ks)}{np.median(rw):>12.3f}{(rw<1).sum():>6}/{len(ks)}"
          f"{np.median(ca):>12.3f}{np.median(ce):>13.3f}")

print("\n" + "=" * 150)
print("安慰剂检验：h=1 上 A5 是否也赢？")
print("=" * 150)
for h in (1, 2, 3, 5):
    ks = [k for k in R if k.endswith(f"|{h}")]
    ra = np.array([R[k]["crps_a5"] / R[k]["crps_emp"] for k in ks])
    print(f"h={h}: A5/EMP 中位 {np.median(ra):.4f}，范围 [{ra.min():.4f}, {ra.max():.4f}]，"
          f"胜 {(ra<1).sum()}/{len(ks)}")

json.dump(R, open(os.path.join(OUT, "HORIZON_PROFILE.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 HORIZON_PROFILE.json")
