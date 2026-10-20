# -*- coding: utf-8 -*-
"""
HAC 修正的覆盖率检验（修正版）
==============================
前版两个错，已改：
 (1) 旧版拿"单尾 97.5% 分位的超出率"去对原假设 0.05，而该量的名义值是 0.025；
 (2) 旧版与 24.14 的"双侧 95% 覆盖率 PICP95"不是同一个量，无法对照。

本版直接检 24.14 的那个量：
    I_t = 1{ y_t 落在 [q_{0.025}, q_{0.975}] 之外 },   H0: E[I_t] = 0.05
    带宽 L = ceil(h/STEP)（重叠 h-2 天 ⇒ 原点滞后 h/STEP）
并给出上下尾各自的超出率（名义 0.025），用来说明偏离方向。

不重叠子样本的二项检验同时给出，作为交叉验证；h>=60 时该子样本只剩 2-11 个点，
仅作参考，不作结论。
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm, binomtest
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
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}


def hac_z(I, p0, L):
    n = len(I)
    if n < 10:
        return np.nan, np.nan
    d = I - I.mean()
    v = (d * d).mean()
    for l in range(1, min(L, n - 2) + 1):
        v += 2 * (1 - l / (L + 1)) * (d[l:] * d[:-l]).mean()
    v = max(v, 1e-12)
    z = (I.mean() - p0) / np.sqrt(v / n)
    return float(z), float(2 * (1 - norm.cdf(abs(z))))


print("=" * 152)
print("HAC 修正的 PICP95 校准检验（双侧，原假设 0.05），并给出上下尾方向")
print("=" * 152)

RES = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float))
    dt = pd.to_datetime(d.date).to_numpy()
    cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
    t0 = int(np.searchsorted(dt, np.datetime64(cut)))
    h0, Hv = PAR[m]
    print(f"\n--- {m} ---")
    print(f"{'h':>4}{'n':>5}{'L':>4}| {'A5cov':>7}{'A5z':>7}{'A5p':>7}{'':>5}"
          f"{'低尾':>6}{'高尾':>6}| {'EMPcov':>7}{'EMPz':>7}{'EMPp':>7}{'':>5}"
          f"{'低尾':>6}{'高尾':>6}| {'非重n':>6}{'二项p':>7}")
    for h in HOR:
        start = max(W + h + 5, t0)
        ts = list(range(start, len(lp) - h, STEP))
        if len(ts) < 40:
            continue
        out_a, out_e, lohi_a, lohi_e = [], [], [], []
        for t in ts:
            win = lp[t - W:t]
            rr = win[h:] - win[:-h]; rd = np.diff(win)
            mu = rr.mean(); sdh = np.std(rr, ddof=1); sdd = np.std(rd, ddof=1)
            xi = np.quantile((rr - rr.mean()) / sdh, TAUS)
            c = mu * (h0 / (h0 + h)); s = sdd * h ** Hv
            lo_a, hi_a = c + s * GV[0], c + s * GV[-1]
            lo_e, hi_e = mu + sdh * xi[0], mu + sdh * xi[-1]
            y = lp[t + h] - lp[t]
            out_a.append(1.0 if (y < lo_a or y > hi_a) else 0.0)
            out_e.append(1.0 if (y < lo_e or y > hi_e) else 0.0)
            lohi_a.append((y < lo_a, y > hi_a))
            lohi_e.append((y < lo_e, y > hi_e))
        A = np.array(out_a); E = np.array(out_e)
        LA = np.array(lohi_a); LE = np.array(lohi_e)
        L = max(int(np.ceil(h / STEP)), 1)
        za, pa = hac_z(A, 0.05, L)
        ze, pe = hac_z(E, 0.05, L)
        sub = A[::L]; nsub = len(sub)
        pb = binomtest(int(sub.sum()), nsub, 0.05).pvalue if nsub else np.nan
        RES[f"{m}|{h}"] = dict(n=len(A), L=L, cov_a5=float(1 - A.mean()), cov_emp=float(1 - E.mean()),
                               za=za, pa=pa, ze=ze, pe=pe,
                               lo_a5=float(LA[:, 0].mean()), hi_a5=float(LA[:, 1].mean()),
                               lo_emp=float(LE[:, 0].mean()), hi_emp=float(LE[:, 1].mean()),
                               nsub=nsub, pbinom=float(pb) if nsub else None)
        r = RES[f"{m}|{h}"]
        print(f"{h:>4}{len(A):>5}{L:>4}| {1-A.mean():>7.3f}{za:>7.2f}{pa:>7.3f}"
              f"{'PASS' if pa>0.05 else 'FAIL':>6}"
              f"{LA[:,0].mean():>6.3f}{LA[:,1].mean():>6.3f}| "
              f"{1-E.mean():>7.3f}{ze:>7.2f}{pe:>7.3f}"
              f"{'PASS' if pe>0.05 else 'FAIL':>6}"
              f"{LE[:,0].mean():>6.3f}{LE[:,1].mean():>6.3f}|{nsub:>7}{pb:>7.3f}")

print("\n" + "=" * 152)
print("汇总")
print("=" * 152)
for lab, k in (("A5", "pa"), ("EMP", "pe")):
    print(f"\n  {lab}  HAC-PICP95 通过格数（p>0.05，原假设 cov=0.95）")
    print(f"  {'h':>4}{'通过':>8}{'平均覆盖率':>12}{'低尾均':>9}{'高尾均':>9}")
    for h in HOR:
        ks = [f"{m}|{h}" for m in MK if f"{m}|{h}" in RES]
        if not ks:
            continue
        ck = "cov_a5" if lab == "A5" else "cov_emp"
        lk = "lo_a5" if lab == "A5" else "lo_emp"
        hk = "hi_a5" if lab == "A5" else "hi_emp"
        print(f"  {h:>4}{sum(1 for x in ks if RES[x][k]>0.05):>5}/{len(ks)}"
              f"{np.mean([RES[x][ck] for x in ks]):>12.3f}"
              f"{np.mean([RES[x][lk] for x in ks]):>9.3f}"
              f"{np.mean([RES[x][hk] for x in ks]):>9.3f}")
    print(f"  全部 {sum(1 for x in RES if RES[x][k]>0.05)}/{len(RES)}")

print(f"\n  对照 24.14 的 iid 版 Kupiec POF：A5 12/25、EMP 5/25")
print(f"  本表 HAC 版（双侧 PICP95）：A5 {sum(1 for x in RES if RES[x]['pa']>0.05)}/25、"
      f"EMP {sum(1 for x in RES if RES[x]['pe']>0.05)}/25")

json.dump(RES, open(os.path.join(OUT, "COVERAGE_HAC.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 COVERAGE_HAC.json")
