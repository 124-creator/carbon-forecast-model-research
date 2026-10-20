# -*- coding: utf-8 -*-
"""
为什么只有广州受益？—— 幂律标度 vs 经验标度 的比值诊断
=====================================================================
定义 R = ŝ_h / (σ̂_d · h^H)：经验 h 步离散度 与 幂律外推 的比值。
R < 1 ⇒ 幂律高估了宽度（宽度该更窄）；R > 1 ⇒ 幂律低估。
若 A16 的增益来自"幂律系统性有偏"，则 R 应系统偏离 1，且方向与增益一致。
检验：
  1) R 在训练段/测试段/各市场/各期限的取值
  2) R 的均值是否可从**训练段**估出并用于测试段（可迁移性）
  3) 在训练段上"经验标度 vs 幂律"谁更好，是否与测试段一致
  4) κ 的可辨性：把网格扩到 1000，看 κ* 是否在边界
"""
import numpy as np, pandas as pd, os, sys, json
from scipy.stats import norm
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(DIR, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}


def crps_q(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


print("=" * 118)
print("一、R = ŝ_h / (σ̂_d · h^H)  的结构（训练段 / 测试段）")
print("=" * 118)
print(f"{'市场':<8}{'h':>5}{'R_train':>10}{'R_test':>10}{'R_te 中位':>11}"
      f"{'n_eff=W/h':>11}{'CRPS 幂律/经验 tr':>21}{'CRPS 幂律/经验 te':>21}")
RR = {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    for h in HOR:
        def pack(ts):
            Y, LOC, PL, EM, R = [], [], [], [], []
            for t in ts:
                win = lp[max(0, t - W):t]
                rr = win[h:] - win[:-h]
                if len(rr) < 3 or np.std(rr, ddof=1) <= 0:
                    continue
                rd = np.diff(win); sdd = np.std(rd, ddof=1)
                if not np.isfinite(sdd) or sdd <= 0:
                    continue
                Y.append(lp[t + h] - lp[t]); LOC.append((h0 / (h0 + h)) * rr.mean())
                pl = sdd * h ** Hv
                PL.append(pl); EM.append(np.std(rr - rr.mean(), ddof=1))
                R.append(np.std(rr - rr.mean(), ddof=1) / pl)
            return [np.array(x) for x in (Y, LOC, PL, EM, R)]

        Ytr, Ltr, PLtr, EMtr, Rtr = pack([t for t in range(W + h + 5, t0 - h - 1, STEP)])
        Yte, Lte, PLte, EMte, Rte = pack([t for t in range(max(W + h + 5, t0), len(lp) - h, STEP)])
        if len(Ytr) < 30 or len(Yte) < 25:
            continue
        ctr = crps_q(Ytr, Ltr[:, None] + PLtr[:, None] * GV).mean()
        etr = crps_q(Ytr, Ltr[:, None] + EMtr[:, None] * GV).mean()
        cte = crps_q(Yte, Lte[:, None] + PLte[:, None] * GV).mean()
        ete = crps_q(Yte, Lte[:, None] + EMte[:, None] * GV).mean()
        RR[f"{m}|{h}"] = dict(R_tr=float(np.median(Rtr)), R_te=float(np.median(Rte)),
                              r_tr=float(ctr / etr), r_te=float(cte / ete),
                              n_tr=len(Ytr), n_te=len(Yte))
        print(f"{MN[m]:<8}{h:>5}{np.median(Rtr):>10.3f}{np.median(Rte):>10.3f}"
              f"{np.median(Rte):>11.3f}{W/h:>11.1f}{ctr/etr:>21.3f}{cte/ete:>21.3f}")

print("\n" + "=" * 118)
print("二、可迁移性：R 的训练段中位 与 测试段中位")
print("=" * 118)
tr = np.array([v["R_tr"] for v in RR.values()]); te = np.array([v["R_te"] for v in RR.values()])
print(f"  Pearson {np.corrcoef(tr, te)[0,1]:.4f}  Spearman "
      f"{np.corrcoef(np.argsort(np.argsort(tr)), np.argsort(np.argsort(te)))[0,1]:.4f}")
print(f"  |R_tr - R_te| 中位 {np.median(np.abs(tr-te)):.3f}   "
      f"R_tr 中位 {np.median(tr):.3f}  R_te 中位 {np.median(te):.3f}")

print("\n" + "=" * 118)
print("三、'幂律 vs 经验'的胜负是否可迁移（r<1 表示幂律更好）")
print("=" * 118)
rt = np.array([v["r_tr"] for v in RR.values()]); re = np.array([v["r_te"] for v in RR.values()])
print(f"  Pearson {np.corrcoef(rt, re)[0,1]:.4f}   "
      f"符号一致 {int((np.sign(rt-1)==np.sign(re-1)).sum())}/25")
print(f"  训练段幂律更好 {int((rt<1).sum())}/25   测试段幂律更好 {int((re<1).sum())}/25")

print("\n" + "=" * 118)
print("四、κ 的可辨性：网格扩到 1000，逐格看 κ*")
print("=" * 118)
KG = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    line = []
    for h in HOR:
        Y, LOC, PL, EM = [], [], [], []
        for t in range(W + h + 5, t0 - h - 1, STEP):
            win = lp[max(0, t - W):t]
            rr = win[h:] - win[:-h]
            if len(rr) < 3 or np.std(rr, ddof=1) <= 0:
                continue
            rd = np.diff(win); sdd = np.std(rd, ddof=1)
            if not np.isfinite(sdd) or sdd <= 0:
                continue
            Y.append(lp[t + h] - lp[t]); LOC.append((h0 / (h0 + h)) * rr.mean())
            PL.append(sdd * h ** Hv); EM.append(np.std(rr - rr.mean(), ddof=1))
        if len(Y) < 30:
            continue
        Y, LOC, PL, EM = map(np.array, (Y, LOC, PL, EM))
        nind = max(1.0, W / h)
        vals = []
        for kap in KG:
            w = nind / (nind + kap)
            vals.append((kap, float(crps_q(Y, LOC[:, None] + (w * EM + (1 - w) * PL)[:, None] * GV).mean())))
        bk = min(vals, key=lambda x: x[1])
        line.append(f"h{h}:κ*={bk[0]:g}")
    print(f"{MN[m]:<8}" + "  ".join(line))

json.dump(RR, open(os.path.join(DIR, "RATIO_DIAG.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 RATIO_DIAG.json")
