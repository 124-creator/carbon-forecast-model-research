# -*- coding: utf-8 -*-
"""
关键一步：c* 的期限结构在**训练段**是否存在？
若训练段的 c*(h) 曲线与测试段一致 → 这是一个可估计、可外推的真实结构，
   且能导出"标度不是纯幂律"这一模型创新。
若训练段 c*(h) 与测试段不一致 → c* 是测试段噪声，前面整条线索作废。
同时检验：c* 的期限形状能否被一个**两参数**函数解释。
"""
import numpy as np, pandas as pd, os, sys, json
from scipy.stats import norm
sys.stdout.reconfigure(encoding="utf-8")
OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
DIRP = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
CS = np.round(np.arange(0.40, 2.01, 0.05), 3)


def crps_q(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


OUT = {}
print("=" * 116)
print("训练段 vs 测试段：宽度乘子最优 c*")
print("=" * 116)
print(f"{'市场':<8}{'h':>5}{'c*_train':>10}{'c*_test':>10}{'差':>8}{'n_tr':>7}{'n_te':>6}")
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    for h in HOR:
        def pack(ts):
            Y, LOC, S5 = [], [], []
            for t in ts:
                win = lp[max(0, t - W):t]
                rr = win[h:] - win[:-h]
                if len(rr) < 3 or np.std(rr, ddof=1) <= 0:
                    continue
                rd = np.diff(win); sdd = np.std(rd, ddof=1)
                if not np.isfinite(sdd) or sdd <= 0:
                    continue
                Y.append(lp[t + h] - lp[t]); LOC.append((h0 / (h0 + h)) * rr.mean())
                S5.append(sdd * h ** Hv)
            return map(np.array, (Y, LOC, S5))

        def best_c(Y, LOC, S5):
            if len(Y) < 25:
                return np.nan, np.nan
            cr = {c: float(crps_q(Y, LOC[:, None] + c * S5[:, None] * GV).mean()) for c in CS}
            b = min(cr, key=cr.get)
            return float(b), cr[b]

        Ytr, Ltr, Str = pack([t for t in range(W + h + 5, t0 - h - 1, STEP)])
        Yte, Lte, Ste = pack([t for t in range(max(W + h + 5, t0), len(lp) - h, STEP)])
        ctr, _ = best_c(Ytr, Ltr, Str)
        cte, cte_v = best_c(Yte, Lte, Ste)
        if not (np.isfinite(ctr) and np.isfinite(cte)):
            continue
        OUT[f"{m}|{h}"] = dict(c_tr=ctr, c_te=cte, n_tr=len(Ytr), n_te=len(Yte))
        print(f"{MN[m]:<8}{h:>5}{ctr:>10.2f}{cte:>10.2f}{cte-ctr:>8.2f}{len(Ytr):>7}{len(Yte):>6}")

print("\n" + "=" * 116)
print("c*(h) 的期限结构：训练段与测试段的对照（逐期限中位）")
print("=" * 116)
print(f"{'h':>5}{'中位 c*_train':>15}{'中位 c*_test':>15}{'差':>8}{'相关':>9}")
for h in HOR:
    tr = [v["c_tr"] for k, v in OUT.items() if k.endswith(f"|{h}")]
    te = [v["c_te"] for k, v in OUT.items() if k.endswith(f"|{h}")]
    print(f"{h:>5}{np.median(tr):>15.2f}{np.median(te):>15.2f}{np.median(te)-np.median(tr):>8.2f}"
          f"{np.corrcoef(tr, te)[0,1]:>9.3f}")

print("\n" + "=" * 116)
print("全期限一致性：训练段 c* 与测试段 c* 的 25 格相关")
print("=" * 116)
tr = np.array([v["c_tr"] for v in OUT.values()])
te = np.array([v["c_te"] for v in OUT.values()])
print(f"  Pearson {np.corrcoef(tr, te)[0,1]:.4f}   Spearman "
      f"{np.corrcoef(np.argsort(np.argsort(tr)), np.argsort(np.argsort(te)))[0,1]:.4f}")
print(f"  训练段 c* 分布: {np.round(np.percentile(tr,[10,25,50,75,90]),3).tolist()}")
print(f"  测试段 c* 分布: {np.round(np.percentile(te,[10,25,50,75,90]),3).tolist()}")
print(f"  |c_tr - c_te| 中位 {np.median(np.abs(tr-te)):.3f}")

print("\n" + "=" * 116)
print("结构检验：c* 能否被单一 h 的函数解释？（按市场拟合 c*(h) = h^a · b）")
print("=" * 116)
for m in CUT:
    ks = [h for h in HOR if f"{m}|{h}" in OUT]
    for tag in ("c_tr", "c_te"):
        v = np.array([OUT[f"{m}|{h}"][tag] for h in ks])
        lh = np.log(ks)
        A = np.column_stack([np.ones(len(ks)), lh])
        beta, *_ = np.linalg.lstsq(A, np.log(v), rcond=None)
        pred = np.exp(A @ beta)
        print(f"{MN[m]:<8}{tag}: c* ≈ {np.exp(beta[0]):.3f}·h^{beta[1]:+.3f}   "
              f"R2={1-((np.log(v)-A@beta)**2).sum()/((np.log(v)-np.log(v).mean())**2).sum():.3f}")
    print()

json.dump(OUT, open(os.path.join(DIRP, "CSTAR_TRAIN_TEST.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("已写出 CSTAR_TRAIN_TEST.json")
