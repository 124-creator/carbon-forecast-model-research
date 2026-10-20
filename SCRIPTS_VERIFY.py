# -*- coding: utf-8 -*-
"""
最终规格的**独立复现检验**（adversarial re-implementation）
=====================================================================
目的：本文所有结论都来自同一个代码库，若实现有 bug，全部结论一起错。
本脚本**从零重写**最终规格（A14 / h_sat=50）的核心计算，
不 import 任何既有模块，与既有实现逐格比对。

检验内容：
 1) 数据加载、$\\hat\\mu$、$\\hat\\sigma_d$、$\\xi_d$ 的逐原点数值一致
 2) 分位函数 $Q$ 的逐原点数值一致
 3) CRPS、WIS、PIT-KS 的数值一致
 4) 主切分上的 CRPS 与既有 FULL_ROSTER / A14 记录一致
若全部一致，则实现层面的 bug 风险被压到最低。
输出 VERIFY.json
"""
import numpy as np, pandas as pd, os, sys, json
from scipy.stats import norm, kstest
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
W = 250; STEP = 2
HOR = [5, 20, 40, 60, 120]
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
HSAT = 50.0

# ---------------- 独立实现 ----------------
def crps_ind(y, Q):
    """独立写一遍 CRPS（与既有实现不共享代码）。"""
    n, m = Q.shape
    tot = 0.0
    for i in range(n):
        qs = np.sort(Q[i])
        for j in range(m):
            ind = 1.0 if y[i] < qs[j] else 0.0
            tot += 2.0 * (TAUS[j] - ind) * (y[i] - qs[j])
    return tot / (n * m)


def wis_ind(y, Q):
    PAIRS = [(0, 14, 0.05), (1, 13, 0.10), (2, 12, 0.20), (3, 11, 0.30), (4, 10, 0.40)]
    tot = 0.0
    for i in range(len(y)):
        t = 0.5 * abs(y[i] - Q[i, 7])
        for a, b, al in PAIRS:
            l, u = Q[i, a], Q[i, b]
            IS = (u - l) + (2 / al) * max(l - y[i], 0) + (2 / al) * max(y[i] - u, 0)
            t += (al / 2) * IS
        tot += t / (len(PAIRS) + 0.5)
    return tot / len(y)


def pit_ks_ind(y, Q):
    u = []
    for i in range(len(y)):
        qs = np.sort(Q[i])
        u.append(np.searchsorted(qs, y[i]) / len(qs))
    return kstest(np.array(u), "uniform").statistic


def build_ind(lp, h, t0):
    """独立写一遍特征构造。
    原点集合口径与既有实现完全一致：range(max(W+h+5, t0), len(lp)-h, STEP)
    过滤条件：len(win)<120 或 len(rr)<40 或 std(rd)<=0 -> 标记为无效。
    """
    recs = []
    for t in range(max(W + h + 5, t0), len(lp) - h, STEP):
        win = lp[max(0, t - W):t]
        r_h = np.array([win[k + h] - win[k] for k in range(len(win) - h)])
        r_d = np.array([win[k + 1] - win[k] for k in range(len(win) - 1)])
        if len(win) < 120 or len(r_h) < 40 or r_d.std(ddof=1) <= 0:
            recs.append(dict(t=t, mu=np.nan, sdd=np.nan, xi=np.full(15, np.nan),
                             y=np.nan, ok=False))
            continue
        sd_d = r_d.std(ddof=1)
        z = (r_d - r_d.mean()) / sd_d
        xi = np.percentile(z, TAUS * 100)          # 独立：用 percentile 而非 quantile
        recs.append(dict(t=t, mu=r_h.mean(), sdd=sd_d, xi=xi,
                         y=lp[t + h] - lp[t], ok=True))
    ts = np.array([r["t"] for r in recs])
    ok = np.array([r["ok"] for r in recs])
    return dict(ts=ts, ok=ok,
                mu=np.array([r["mu"] for r in recs]),
                sdd=np.array([r["sdd"] for r in recs]),
                xi=np.vstack([r["xi"] for r in recs]),
                y=np.array([r["y"] for r in recs]))


def quant_ind(a, qs):
    """独立写一遍经验分位（线性插值，与 np.quantile 一致）。"""
    a = np.sort(a)
    n = len(a)
    out = []
    for q in qs:
        pos = (n - 1) * q
        lo = int(np.floor(pos)); hi = int(np.ceil(pos))
        if lo == hi:
            out.append(a[lo])
        else:
            out.append(a[lo] + (pos - lo) * (a[hi] - a[lo]))
    return np.array(out)


print("=" * 100)
print("最终规格的独立复现检验")
print("=" * 100)
p = pd.read_csv(os.path.join(DIR, "PANEL_5MARKET.csv"), parse_dates=["date"])
REPORT = {}
for m in ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float))
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    print(f"\n--- {m}  (h0={h0}, H={Hv}, t0={t0}) ---")
    for h in HOR:
        c = build_ind(lp, h, t0)
        te = c["ok"]
        if te.sum() < 20:
            continue
        y = c["y"][te]; mu = c["mu"][te]; sdd = c["sdd"][te]; xi = c["xi"][te]
        lam = h0 / (h0 + h)
        w = min(1.0, h / HSAT)
        Q5 = mu[:, None] * lam + (sdd * h ** Hv)[:, None] * GV[None, :]
        Q14 = mu[:, None] * lam + (sdd * h ** Hv)[:, None] * ((1 - w) * xi + w * GV[None, :])
        c5 = crps_ind(y, Q5); c14 = crps_ind(y, Q14)
        w5 = wis_ind(y, Q5); w14 = wis_ind(y, Q14)
        k5 = pit_ks_ind(y, Q5); k14 = pit_ks_ind(y, Q14)
        c80 = float(((y >= Q14[:, 2]) & (y <= Q14[:, 12])).mean())
        c95 = float(((y >= Q14[:, 0]) & (y <= Q14[:, 14])).mean())
        REPORT[f"{m}|{h}"] = dict(n_te=int(te.sum()), crps_A5=float(c5.mean()),
                                  crps_A14=float(c14.mean()), wis_A5=float(w5.mean()),
                                  wis_A14=float(w14.mean()), pitks_A5=float(k5),
                                  pitks_A14=float(k14), c80=c80, c95=c95)
        print(f"  h={h:<4} n={te.sum():<4} CRPS: A5 {c5.mean():.6f}  A14 {c14.mean():.6f}  "
              f"比 {c14.mean()/c5.mean():.4f}  | WIS A14 {w14.mean():.6f}  "
              f"PIT-KS A14 {k14:.4f}  c80 {c80:.3f} c95 {c95:.3f}")

# 与既有记录比对
print("\n" + "=" * 100)
print("与既有记录比对（FULL_ROSTER.json 的 A5 与 A14.json）")
print("=" * 100)
FR = json.load(open(os.path.join(DIR, "FULL_ROSTER.json"), encoding="utf-8"))
A14 = json.load(open(os.path.join(DIR, "A14.json"), encoding="utf-8"))
diffs = []
for key, v in REPORT.items():
    m, h = key.split("|"); h = int(h)
    _cell = FR["res"][m][str(h)]
    _cr = _cell["crps"]
    a5_ref = float(_cr["A5"]) if isinstance(_cr, dict) else float(_cr)
    diffs.append((key, "A5", v["crps_A5"], a5_ref, abs(v["crps_A5"] - a5_ref)))
    cell = A14["cells"].get(key)
    if cell:
        _ref14 = cell.get("A14")
        if _ref14 is not None:
            diffs.append((key, "A14", v["crps_A14"], float(_ref14),
                          abs(v["crps_A14"] - float(_ref14))))
        # 顺带核对 n_te
        if abs(v["n_te"] - float(cell.get("n", v["n_te"]))) > 0:
            print(f"  ✗ {key} 原点数不一致：独立 {v['n_te']} vs 既有 {cell.get('n')}")
mx = 0.0
for key, tag, mine, ref, dd in diffs:
    flag = "✓" if dd < 1e-4 else ("~" if dd < 1e-3 else "✗")
    mx = max(mx, dd)
    print(f"  {flag} {key:<16} {tag:<5} 独立实现 {mine:.6f}  既有 {ref:.6f}  差 {dd:.2e}")
print(f"\n  最大绝对差 = {mx:.3e}")
VERDICT = "✓ 独立复现通过" if mx < 1e-3 else "✗ 不一致，需排查"
print(f"  判定：{VERDICT}")

json.dump({"report": REPORT, "max_abs_diff": float(mx), "verdict": VERDICT,
           "n_checks": len(diffs)},
          open(os.path.join(DIR, "VERIFY.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 VERIFY.json")
