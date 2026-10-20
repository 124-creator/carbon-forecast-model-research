# -*- coding: utf-8 -*-
"""
A16 族：有效样本量加权的标度收缩（新的标度块）
=====================================================================
动机（来自 §24.29 的诊断，不是猜测）：
 1) 位置块在 CRPS 里几乎不起作用（除广州），宽度乘子 c* 不可迁移 —— 剩下唯一
    可动的部件是**标度块**，而 A5 用的是纯幂律 σ̂_d · h^H。
 2) 标度还有第二个估计量：**窗内 h 步收益的经验离散度 ŝ_h**。两者偏差方向相反：
      - 幂律 σ̂_d·h^H 用 H（由 VR 估）从**日频**方差外推，忽略 (a) h 步收益的
        重叠导致的方差结构，(b) 有限窗口的估计误差。
      - 经验 ŝ_h 直接由窗内 W/h 个（重叠的）h 步收益算出，**无外推**，但
        有效自由度只有 W/h：h=5 时 50 个（可靠），h=120 时 2.1 个（几乎无信息）。
 3) 因此正确的标度应当是**两者的收缩组合，权重由有效样本量 W/h 决定**：
        s(h) = w(h)·ŝ_h + (1-w(h))·σ̂_d·h^H ,   w(h) 由 W/h 与噪声水平共同决定
    这是一个**参数少、机制明确、且此前从未被检验过**的标度块。

本脚本在**训练段**拟合收缩规则（只用训练数据），到测试段评估。
候选：
  A16a  固定 w(h) = W/(W+h)          —— 有效样本量权重，零参数
  A16b  固定 w(h) = (W/h)/(W/h+κ)     —— κ 由训练段估（每市场 1 个参数）
  A16c  训练段最优 w(h) 查表（每市场 5 个）—— 作为上界
  A16d  收缩到**去均值**的经验离散度（去掉窗内漂移），并用 A16b 的权重
对照：A5（σ̂_d·h^H）、EMP（纯经验分位数）、SD250（σ̂_d·√h）
"""
import numpy as np, pandas as pd, os, sys, json
from scipy.stats import norm, t as tdist
sys.stdout.reconfigure(encoding="utf-8")
OUTDIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUTDIR, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
KS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def crps_q(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


def dm(a, b, L=10):
    d = np.asarray(a) - np.asarray(b); n = len(d)
    if n < 20:
        return np.nan
    m_ = d.mean(); dc = d - m_
    lrv = dc @ dc / n
    for l in range(1, L + 1):
        lrv += 2 * (1 - l / (L + 1)) * (dc[l:] @ dc[:-l] / n)
    if lrv <= 0:
        return np.nan
    s = m_ / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    return float(2 * (1 - tdist.cdf(abs(s), df=n - 1)))


RES = {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    print(f"\n=== {m} ===")
    for h in HOR:
        ts_tr = [t for t in range(W + h + 5, t0 - h - 1, STEP)]
        ts_te = [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP)]
        assert not (set(ts_tr) & set(ts_te))

        def pack(ts):
            Y, LOC, PL, EM, EMD, S5 = [], [], [], [], [], []
            for t in ts:
                win = lp[max(0, t - W):t]
                rr = win[h:] - win[:-h]
                if len(rr) < 3:
                    continue
                sd = np.std(rr, ddof=1)
                if sd <= 0:
                    continue
                rd = np.diff(win); sdd = np.std(rd, ddof=1)
                if not np.isfinite(sdd) or sdd <= 0:
                    continue
                Y.append(lp[t + h] - lp[t])
                LOC.append((h0 / (h0 + h)) * rr.mean())
                PL.append(sdd * h ** Hv)                     # A5 幂律标度
                EM.append(sd)                                # 经验 h 步离散度
                EMD.append(np.std(rr - rr.mean(), ddof=1))   # 去均值经验离散度
                S5.append(sdd)
            return [np.array(x) for x in (Y, LOC, PL, EM, EMD, S5)]

        Ytr, Ltr, PLtr, EMtr, EMDtr, _ = pack(ts_tr)
        Yte, Lte, PLte, EMte, EMDte, _ = pack(ts_te)
        if len(Ytr) < 30 or len(Yte) < 25:
            continue
        nind = max(1.0, W / h)

        # --- 训练段：估 kappa（对 A16b），并给出 A16c 上界 ---
        def crps_of(scale, Y, LOC):
            return crps_q(Y, LOC[:, None] + scale[:, None] * GV)
        kappa_grid = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        best_k, best_v = None, 1e18
        for kap in kappa_grid:
            wtr = np.minimum(1.0, (nind / (nind + kap)))
            sc = wtr * EMDtr + (1 - wtr) * PLtr
            v = float(crps_of(sc, Ytr, Ltr).mean())
            if v < best_v:
                best_v, best_k = v, kap
        wA16a = W / (W + h)
        wA16b = nind / (nind + best_k)
        wA16c = KS[int(np.argmin([float(crps_of(k * EMDtr + (1 - k) * PLtr, Ytr, Ltr).mean())
                                  for k in KS]))]
        row = dict(n=len(Yte), n_tr=len(Ytr), kappa=best_k,
                   w_a16a=float(wA16a), w_a16b=float(wA16b), w_a16c=float(wA16c),
                   nind=float(nind))
        specs = {
            "A5":   lambda Y, L, PL, EMD: PL,
            "A16a": lambda Y, L, PL, EMD: wA16a * EMD + (1 - wA16a) * PL,
            "A16b": lambda Y, L, PL, EMD: wA16b * EMD + (1 - wA16b) * PL,
            "A16c": lambda Y, L, PL, EMD: wA16c * EMD + (1 - wA16c) * PL,
            "EMPSCALE": lambda Y, L, PL, EMD: EMD,
            "PLONLY": lambda Y, L, PL, EMD: PL,
        }
        losses = {}
        for nm, f in specs.items():
            sc = f(Yte, Lte, PLte, EMDte)
            l = crps_q(Yte, Lte[:, None] + sc[:, None] * GV)
            losses[nm] = l
            row[nm] = float(l.mean())
        for nm in ["A16a", "A16b", "A16c", "EMPSCALE"]:
            row["p_" + nm] = dm(losses[nm], losses["A5"])
            row["r_" + nm] = row[nm] / row["A5"]
        # 覆盖率
        for nm in ["A5", "A16b", "A16c"]:
            sc = specs[nm](Yte, Lte, PLte, EMDte)
            Q = Lte[:, None] + sc[:, None] * GV
            row["c95_" + nm] = float(np.mean((Yte >= Q[:, 0]) & (Yte <= Q[:, 14])))
            row["c80_" + nm] = float(np.mean((Yte >= Q[:, 2]) & (Yte <= Q[:, 12])))
        RES[f"{m}|{h}"] = row
        print(f"  h={h:<4} n={row['n']:<4} kappa*={best_k:<5} w_a16b={wA16b:.3f} | "
              f"A5 {row['A5']:.5f} | A16a {row['A16a']:.5f} ({row['r_A16a']:.3f},p={row['p_A16a']:.3f}) | "
              f"A16b {row['A16b']:.5f} ({row['r_A16b']:.3f},p={row['p_A16b']:.3f}) | "
              f"A16c {row['A16c']:.5f} ({row['r_A16c']:.3f},p={row['p_A16c']:.3f}) | "
              f"EMPscale {row['EMPSCALE']:.5f} | c95 A5 {row['c95_A5']:.3f} A16b {row['c95_A16b']:.3f}")

json.dump(RES, open(os.path.join(OUTDIR, "A16.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)

cells = [(m, h) for m in CUT for h in HOR]
cells = [(m, h) for m, h in cells if f"{m}|{h}" in RES]
print("\n" + "=" * 118)
print("汇总（相对 A5）")
print("=" * 118)
print(f"{'模型':<10}{'中位':>9}{'胜':>7}{'显著胜':>8}{'显著负':>8}{'逐期限中位 h=5/20/40/60/120':>40}")
for nm in ["A16a", "A16b", "A16c", "EMPSCALE"]:
    rr = np.array([RES[f"{m}|{h}"][nm] / RES[f"{m}|{h}"]["A5"] for m, h in cells])
    sw = sum(1 for m, h in cells if RES[f"{m}|{h}"]["p_" + nm] < .05 and RES[f"{m}|{h}"][nm] < RES[f"{m}|{h}"]["A5"])
    sl = sum(1 for m, h in cells if RES[f"{m}|{h}"]["p_" + nm] < .05 and RES[f"{m}|{h}"][nm] > RES[f"{m}|{h}"]["A5"])
    per = " / ".join(f"{np.median([RES[f'{m}|{h}'][nm]/RES[f'{m}|{h}']['A5'] for m,_ in cells if h==h]):.3f}"
                     for h in HOR)
    print(f"{nm:<10}{np.median(rr):>9.4f}{int((rr<1).sum()):>4}/25{sw:>8}{sl:>8}{per:>40}")

print("\n覆盖率对比（A5 vs A16b 的 95% 覆盖，目标 0.95）：")
for h in HOR:
    ks = [(m, h) for m, h in cells if h == h]
    print(f"  h={h:<5} A5 {np.mean([RES[f'{m}|{h}']['c95_A5'] for m,_ in ks]):.3f}   "
          f"A16b {np.mean([RES[f'{m}|{h}']['c95_A16b'] for m,_ in ks]):.3f}   "
          f"A16c {np.mean([RES[f'{m}|{h}']['c95_A16c'] for m,_ in ks]):.3f}")
print("\n已写出 A16.json")
