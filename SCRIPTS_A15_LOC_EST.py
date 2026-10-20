# -*- coding: utf-8 -*-
"""
A15 族：估计误差感知的漂移估计（替换 A5 的固定 λ 收缩）
=====================================================================
诊断（§24.29.1）：A5 的位置项在长端系统性偏高。
  残差均值 z̄ 在 h=120 达 -1.09(湖北) / -0.84(上海) / -0.83(EU)，
  PIT 中位数 湖北 h=120 = 0.156。
  而 λ(h)=h0/(h0+h) 在上海/湖北的 h=120 只有 0.0008 / 0.0244 —— 位置项几乎为零，
  偏差却仍在。=> 病根不是 λ 的函数形式，而是**漂移估计量 μ̂ 本身**：
  μ̂ = 窗口内所有 h 步收益的均值，窗口只有 W/h 个 h 步收益（h=120 时仅 2.1 个），
  估计误差极大。

§24.27 已证明四种"改 λ"（递增/对数/精度加权/全局 κ）全部失败，
因为它们都在**保持 μ̂ 不变**的前提下重新加权一个噪声估计量。

本脚本检验三种**换估计量**的做法（参数只在训练段拟合）：
  A15a  多尺度漂移：mu* = sum_k w_k mu_k，k in {20,60,120,250}，w 由训练段
        逐 (市场, h) 的岭回归系数给出（截距项另给一个常数漂移 a）。
  A15b  信噪收缩（无拟合参数）：loc = mu * s2/(s2+v)，
        v = mu 的估计方差 = sig_d^2 * h^H / (W/h) （窗口内独立 h 步收益个数）
  A15c  稳健 + 信噪：mu 用窗口内 h 步收益的**中位数**，再乘信噪收缩因子。

评估除全 CRPS 外，另报**位置块专属指标**：
  - LRMSE：位置误差的均方根（标准差单位）
  - LCRPS：固定形状（高斯）+ 候选位置 的 CRPS —— 把位置贡献与宽度贡献分开
  - 位置块 DM
"""
import numpy as np, pandas as pd, os, sys, json
from scipy.stats import norm, t as tdist
sys.stdout.reconfigure(encoding="utf-8")
OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
KS = [20, 60, 120, 250]
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}


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
        assert not (set(ts_tr) & set(ts_te)), "LEAK"

        def feats(t):
            win = lp[max(0, t - W):t]
            rr = win[h:] - win[:-h]
            sd = np.std(rr, ddof=1)
            if sd <= 0 or len(rr) < 3:
                return None
            rd = np.diff(win)
            sdd = np.std(rd, ddof=1)
            out = dict(mu=float(rr.mean()), med=float(np.median(rr)),
                       sdd=float(sdd), sd=float(sd), y=float(lp[t + h] - lp[t]))
            # 多尺度：k 日窗口内的 h 步收益均值
            for k in KS:
                w2 = lp[max(0, t - k):t]
                if len(w2) < h + 3:
                    out[f"m{k}"] = np.nan
                else:
                    r2 = w2[h:] - w2[:-h]
                    out[f"m{k}"] = float(r2.mean())
            return out

        FTR = [f for f in (feats(t) for t in ts_tr) if f is not None]
        FTE = [f for f in (feats(t) for t in ts_te) if f is not None]
        if len(FTR) < 30 or len(FTE) < 25:
            continue
        # ---- 训练段拟合 A15a 的岭系数 ----
        X = np.column_stack([np.ones(len(FTR))] + [np.array([f[f"m{k}"] for f in FTR]) for k in KS])
        yv = np.array([f["y"] for f in FTR])
        lam_r = 1.0
        A = X.T @ X + lam_r * np.eye(X.shape[1]); A[0, 0] -= lam_r
        beta = np.linalg.solve(A, X.T @ yv)

        def locs(fs):
            """返回各模型的 location（标度单位无关的绝对对数收益）"""
            out = dict()
            lam = h0 / (h0 + h)
            out["A5"] = np.array([lam * f["mu"] for f in fs])
            # A15a 多尺度
            Xs = np.column_stack([np.ones(len(fs))] + [np.array([f[f"m{k}"] for f in fs]) for k in KS])
            out["A15a"] = Xs @ beta
            # A15b 信噪收缩：v = sd^2 / (独立 h 步收益个数 ~ W/h)
            nind = max(1.0, W / h)
            v = np.array([f["sd"] ** 2 / nind for f in fs])
            s2 = np.array([f["sd"] ** 2 for f in fs])
            shr = s2 / (s2 + v)
            out["A15b"] = out["A5"] * shr
            # A15c 稳健中位数 + 信噪
            lamc = lam
            out["A15c"] = lamc * np.array([f["med"] for f in fs]) * shr
            return out

        LTR, LTE = locs(FTR), locs(FTE)
        yte = np.array([f["y"] for f in FTE])
        s5 = np.array([f["sdd"] * h ** Hv for f in FTE])
        s4 = np.array([f["sd"] for f in FTE])       # 窗口内 h 步收益 sd（用于 A15b 的 SNR 定义一致性）

        row = dict(n=len(FTE), n_tr=len(FTR))
        base = None
        for nm in ["A5", "A15a", "A15b", "A15c"]:
            loc = LTE[nm]
            Q = loc[:, None] + s5[:, None] * GV
            l = crps_q(yte, Q)
            # 位置专属：固定高斯形状、只比位置
            lq = crps_q(yte, s5[:, None] * GV)
            row[nm] = dict(crps=float(l.mean()), lcrps=float(lq.mean()),
                           lrmse=float(np.sqrt(np.mean((yte - loc) ** 2))),
                           zbar=float(np.mean((yte - loc) / s5)))
            if nm == "A5":
                base = (l, lq, np.sqrt(np.mean((yte - loc) ** 2)))
        for nm in ["A15a", "A15b", "A15c"]:
            loc = LTE[nm]
            Q = loc[:, None] + s5[:, None] * GV
            l = crps_q(yte, Q)
            lq = crps_q(yte, s5[:, None] * GV)
            row[nm]["p_crps"] = dm(l, base[0])
            row[nm]["p_lcrps"] = dm(lq, base[1])
            row[nm]["r_crps"] = row[nm]["crps"] / row["A5"]["crps"]
            row[nm]["r_lcrps"] = row[nm]["lcrps"] / row["A5"]["lcrps"]
        # 位置块"零位置"对照
        l0 = crps_q(yte, s5[:, None] * GV)
        row["ZERO"] = dict(crps=float(l0.mean()), lcrps=float(l0.mean()),
                           lrmse=float(np.sqrt(np.mean((yte - 0.0) ** 2))),
                           zbar=float(np.mean(yte / s5)),
                           p_crps=dm(l0, base[0]))
        RES[f"{m}|{h}"] = row
        print(f"  h={h:<4} n={row['n']:<4} A5 crps {row['A5']['crps']:.5f} zbar {row['A5']['zbar']:+.3f} "
              f"| A15a {row['A15a']['crps']:.5f} zbar {row['A15a']['zbar']:+.3f} p={row['A15a']['p_crps']:.3f} "
              f"| A15b {row['A15b']['crps']:.5f} zbar {row['A15b']['zbar']:+.3f} p={row['A15b']['p_crps']:.3f} "
              f"| A15c {row['A15c']['crps']:.5f} zbar {row['A15c']['zbar']:+.3f} p={row['A15c']['p_crps']:.3f} "
              f"| ZERO {row['ZERO']['crps']:.5f}")

json.dump(RES, open(os.path.join(OUT, "A15.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)

# ---------------- 汇总 ----------------
cells = [(m, h) for m in CUT for h in HOR]
cells = [(m, h) for m, h in cells if f"{m}|{h}" in RES]
print("\n" + "=" * 126)
print("汇总（相对 A5，中位比值；胜/显著胜/显著负）")
print("=" * 126)
print(f"{'模型':<8}{'CRPS 中位':>11}{'胜':>7}{'显著胜':>8}{'显著负':>8}"
      f"{'LCRPS 中位':>12}{'LCRPS 胜':>10}{'zbar 均值':>11}{'LRMSE 中位':>12}")
for nm in ["A15a", "A15b", "A15c", "ZERO"]:
    rr = np.array([RES[f"{m}|{h}"][nm]["crps"] / RES[f"{m}|{h}"]["A5"]["crps"] for m, h in cells])
    sw = sum(1 for m, h in cells if RES[f"{m}|{h}"][nm]["p_crps"] < .05 and RES[f"{m}|{h}"][nm]["crps"] < RES[f"{m}|{h}"]["A5"]["crps"])
    sl = sum(1 for m, h in cells if RES[f"{m}|{h}"][nm]["p_crps"] < .05 and RES[f"{m}|{h}"][nm]["crps"] > RES[f"{m}|{h}"]["A5"]["crps"])
    rl = np.array([RES[f"{m}|{h}"][nm]["lcrps"] / RES[f"{m}|{h}"]["A5"]["lcrps"] for m, h in cells])
    zb = np.mean([RES[f"{m}|{h}"][nm]["zbar"] for m, h in cells])
    lr = np.median([RES[f"{m}|{h}"][nm]["lrmse"] / RES[f"{m}|{h}"]["A5"]["lrmse"] for m, h in cells])
    print(f"{nm:<8}{np.median(rr):>11.4f}{int((rr<1).sum()):>4}/25{sw:>8}{sl:>8}"
          f"{np.median(rl):>12.4f}{int((rl<1).sum()):>7}/25{zb:>11.3f}{lr:>12.4f}")

print("\n逐期限（CRPS 中位比值）")
print(f"{'h':>5}{'A5 zbar':>10}{'A15a':>9}{'A15b':>9}{'A15c':>9}{'ZERO':>9}")
for h in HOR:
    ks = [(m, h) for m, h in cells if h == h]
    if not ks:
        continue
    row = [str(h), f"{np.mean([RES[f'{m}|{h}']['A5']['zbar'] for m,_ in ks]):+.3f}"]
    for nm in ["A15a", "A15b", "A15c", "ZERO"]:
        row.append(f"{np.median([RES[f'{m}|{h}'][nm]['crps']/RES[f'{m}|{h}']['A5']['crps'] for m,_ in ks]):.3f}")
    print("".join(f"{x:>9}" for x in row))
print("\n已写出 A15.json")
