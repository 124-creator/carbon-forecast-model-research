# -*- coding: utf-8 -*-
"""
标度 × 形状 的 2x2 分解
=======================
A5 现在是 sdd * h^H * G，G 是对称高斯的。24.24.4 显示实际分布在 h>=40 左偏 -0.12。
问题：剩下的缺口是标度还是形状？

四个变体，同一信息集（窗口内 250 日）：
  A5G    = lam*mu + sdd*h^H * G            当前正式规格（形状=高斯，0 参数）
  A5d    = lam*mu + sdd*h^H * xi_d          形状 = 日频标准化经验分位（窗口内 249 个，稳定）
  A5h    = lam*mu + sdd*h^H * xi_h          形状 = h 步窗口标准化经验分位（EMP 用的那个）
  EMP    = mu + sdh * xi_h                  参照

对照关系：
  A5h vs A5G  -> 只换形状（用 h 步形状）
  A5d vs A5G  -> 只换形状（用日频形状）
  A5h vs EMP  -> 只换标度（形状相同）
若 A5h 明显优于 A5G 且接近/优于 EMP，说明形状是主因；
若 A5h 与 A5G 持平，说明 h^H 标度是主因、形状无关。
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
W, STEP = 250, 2
HS = [1, 5, 20, 40, 60, 120]


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
    t = (d.mean()) / np.sqrt(v / n)
    hln = np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    return float(2 * (1 - norm.cdf(abs(t * hln))))


R = {}
print("=" * 130)
print("标度 x 形状 2x2 分解（W=250, STEP=2, L=10）")
print("=" * 130)
print(f"{'h':>4} {'市场':<11} {'A5G':>9}{'A5d':>9}{'A5h':>9}{'EMP':>9} | "
      f"{'A5d/A5G':>8}{'A5h/A5G':>8}{'A5h/EMP':>8}{'A5G/EMP':>8}")
for h in HS:
    agg = []
    for m in MK:
        d = p[p.market == m].sort_values("date").reset_index(drop=True)
        lp = np.log(d.price.to_numpy(float)); dt = pd.to_datetime(d.date).to_numpy()
        cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
        t0 = int(np.searchsorted(dt, np.datetime64(cut)))
        h0, Hv = PAR[m]
        lo = max(W + h + 5, t0)
        ts = list(range(lo, len(lp) - h, STEP))
        if len(ts) < 30:
            continue
        Qg, Qd, Qh, Qe, y = [], [], [], [], []
        for t in ts:
            win = lp[t - W:t]
            rr = win[h:] - win[:-h]; rd = np.diff(win)
            mu = rr.mean(); sdh = np.std(rr, ddof=1); sdd = np.std(rd, ddof=1)
            xi_h = np.quantile((rr - rr.mean()) / sdh, TAUS)
            xi_d = np.quantile((rd - rd.mean()) / sdd, TAUS)
            sc = sdd * h ** Hv; loc = mu * (h0 / (h0 + h))
            Qg.append(loc + sc * GV)
            Qd.append(loc + sc * xi_d)
            Qh.append(loc + sc * xi_h)
            Qe.append(mu + sdh * xi_h)
            y.append(lp[t + h] - lp[t])
        y = np.array(y)
        L = {k: crps_row(y, np.array(Q)) for k, Q in
             (("A5G", Qg), ("A5d", Qd), ("A5h", Qh), ("EMP", Qe))}
        r = {k: float(v.mean()) for k, v in L.items()}
        r["p_d_vs_g"] = dm_hln(L["A5d"], L["A5G"])
        r["p_h_vs_g"] = dm_hln(L["A5h"], L["A5G"])
        r["p_h_vs_e"] = dm_hln(L["A5h"], L["EMP"])
        R[f"{m}|{h}"] = r
        print(f"{h:>4} {m:<11} {r['A5G']:>9.5f}{r['A5d']:>9.5f}{r['A5h']:>9.5f}{r['EMP']:>9.5f} | "
              f"{r['A5d']/r['A5G']:>8.3f}{r['A5h']/r['A5G']:>8.3f}"
              f"{r['A5h']/r['EMP']:>8.3f}{r['A5G']/r['EMP']:>8.3f}")
        agg.append((r["A5d"] / r["A5G"], r["A5h"] / r["A5G"], r["A5h"] / r["EMP"],
                    r["A5G"] / r["EMP"]))
    a = np.array(agg)
    print(f"{'':>4} {'中位':<11}{'':>9}{'':>9}{'':>9}{'':>9} | "
          f"{np.median(a[:,0]):>8.3f}{np.median(a[:,1]):>8.3f}"
          f"{np.median(a[:,2]):>8.3f}{np.median(a[:,3]):>8.3f}")
    print("-" * 130)

print("\n" + "=" * 130)
print("汇总：三种形状在各期限的比值（5 市场中位）")
print("=" * 130)
print(f"{'h':>5}{'A5d/A5G':>10}{'A5h/A5G':>10}{'A5h/EMP':>10}{'A5G/EMP':>10}"
      f"{'  A5d胜A5G':>11}{'  A5h胜A5G':>11}{'  A5h胜EMP':>11}")
for h in HS:
    ks = [k for k in R if k.endswith(f"|{h}")]
    if not ks:
        continue
    v = lambda f: np.median([R[k][f] for k in ks])
    nd = sum(R[k]["p_d_vs_g"] < 0.05 and R[k]["A5d"] < R[k]["A5G"] for k in ks)
    nh = sum(R[k]["p_h_vs_g"] < 0.05 and R[k]["A5h"] < R[k]["A5G"] for k in ks)
    ne = sum(R[k]["p_h_vs_e"] < 0.05 and R[k]["A5h"] < R[k]["EMP"] for k in ks)
    print(f"{h:>5}{v('A5d')/v('A5G'):>10.3f}{v('A5h')/v('A5G'):>10.3f}"
          f"{v('A5h')/v('EMP'):>10.3f}{v('A5G')/v('EMP'):>10.3f}"
          f"{nd:>7}/{len(ks):<4}{nh:>7}/{len(ks):<4}{ne:>7}/{len(ks):<4}")

json.dump(R, open(os.path.join(OUT, "SHAPE_SCALE_2X2.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 SHAPE_SCALE_2X2.json")
