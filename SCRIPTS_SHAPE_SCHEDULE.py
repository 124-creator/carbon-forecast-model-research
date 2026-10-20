# -*- coding: utf-8 -*-
"""
最后一步：形状混合比例的两种候选
 A8a: w(h) = min(1, h/50)  —— 0 自由参数，常数 50 来自 oracle 剖面，须披露
 A8b: 全局 h_s 在训练期池化拟合（1 参数，25 格共用），OOS 评估
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
HOR = [5, 20, 40, 60, 120]
STEP, W = 2, 250
HS_GRID = np.array([1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 400, 800, 1600])


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
    t = d.mean() / np.sqrt(v / n)
    hln = np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    return float(2 * (1 - norm.cdf(abs(t * hln))))


def cells(m, h):
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)); dt = pd.to_datetime(d.date).to_numpy()
    cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
    t0 = int(np.searchsorted(dt, np.datetime64(cut)))
    h0, Hv = PAR[m]
    ts = list(range(max(W + h + 5, t0), len(lp) - h, STEP))
    B, X, E, y = [], [], [], []
    for t in ts:
        win = lp[t - W:t]
        rd = np.diff(win); rr = win[h:] - win[:-h]
        mu = rr.mean(); sdh = np.std(rr, ddof=1); sdd = np.std(rd, ddof=1)
        loc = mu * (h0 / (h0 + h)); sc = sdd * h ** Hv
        B.append(loc + sc * GV)
        X.append(loc + sc * np.quantile((rd - rd.mean()) / sdd, TAUS))
        E.append(mu + sdh * np.quantile((rr - rr.mean()) / sdh, TAUS))
        y.append(lp[t + h] - lp[t])
    return np.array(y), np.array(B), np.array(X), np.array(E)


C = {f"{m}|{h}": cells(m, h) for h in HOR for m in MK}

print("=" * 120)
print("A8b：全局 h_s 在训练期（前 50% 原点）池化拟合，25 格共用")
print("=" * 120)
for s in HS_GRID:
    tot, cnt = 0.0, 0
    for h in HOR:
        w = h / (h + s)
        for m in MK:
            y, B, X, E = C[f"{m}|{h}"]
            k = len(y) // 2
            tot += crps_row(y[:k], (1 - w) * X[:k] + w * B[:k]).sum(); cnt += k
    print(f"  h_s={s:>7.0f}  训练期总 CRPS={tot/cnt:.5f}")
best = min(HS_GRID, key=lambda s: sum(
    crps_row(C[f'{m}|{h}'][0][:len(C[f'{m}|{h}'][0])//2],
             (1 - h/(h+s)) * C[f'{m}|{h}'][2][:len(C[f'{m}|{h}'][0])//2]
             + (h/(h+s)) * C[f'{m}|{h}'][1][:len(C[f'{m}|{h}'][0])//2]).sum()
    for h in HOR for m in MK))
print(f"  -> 训练期最优全局 h_s = {best:.0f}")

print("\n" + "=" * 120)
print("OOS 对比（后 50% 原点）：A8a(w=h/50) / A8b(全局 h_s) / A6 / A5 / EMP")
print("=" * 120)
print(f"{'h':>4} {'市场':<11}{'A8a':>9}{'A8b':>9}{'A6':>9}{'A5':>9}{'EMP':>9} | "
      f"{'a/EMP':>7}{'b/EMP':>7}{'6/EMP':>7}{'5/EMP':>7}")
RES = {}
for h in HOR:
    for m in MK:
        y, B, X, E = C[f"{m}|{h}"]
        k = len(y) // 2
        wa = min(1.0, h / 50.0); wb = h / (h + best)
        Qa = (1 - wa) * X[k:] + wa * B[k:]
        Qb = (1 - wb) * X[k:] + wb * B[k:]
        v = {"A8a": Qa, "A8b": Qb, "A6": X[k:], "A5": B[k:], "EMP": E[k:]}
        L = {kk: crps_row(y[k:], v[kk]) for kk in v}
        c = {kk: float(l.mean()) for kk, l in L.items()}
        RES[f"{m}|{h}"] = dict(crps=c,
                               r={kk: c[kk] / c["EMP"] for kk in ("A8a", "A8b", "A6", "A5")},
                               p_a=dm_hln(L["A8a"], L["EMP"]),
                               p_b=dm_hln(L["A8b"], L["EMP"]),
                               p_a5=dm_hln(L["A8a"], L["A5"]),
                               p_a6=dm_hln(L["A8a"], L["A6"]))
        print(f"{h:>4} {m:<11}{c['A8a']:>9.5f}{c['A8b']:>9.5f}{c['A6']:>9.5f}"
              f"{c['A5']:>9.5f}{c['EMP']:>9.5f} | {c['A8a']/c['EMP']:>7.3f}"
              f"{c['A8b']/c['EMP']:>7.3f}{c['A6']/c['EMP']:>7.3f}{c['A5']/c['EMP']:>7.3f}")
    print("-" * 120)

print("\n" + "=" * 108)
print("汇总（5 市场中位，OOS 后 50% 原点）")
print("=" * 108)
print(f"{'h':>5}{'A8a/EMP':>9}{'胜':>6}{'A8b/EMP':>9}{'胜':>6}"
      f"{'A6/EMP':>9}{'胜':>6}{'A5/EMP':>9}{'胜':>6}")
for h in HOR:
    ks = [f"{m}|{h}" for m in MK]
    g = lambda kk: np.median([RES[k]["r"][kk] for k in ks])
    n = lambda kk: int(sum(RES[k]["r"][kk] < 1 for k in ks))
    print(f"{h:>5}{g('A8a'):>9.3f}{n('A8a'):>4}/5{g('A8b'):>9.3f}{n('A8b'):>4}/5"
          f"{g('A6'):>9.3f}{n('A6'):>4}/5{g('A5'):>9.3f}{n('A5'):>4}/5")
for kk in ("A8a", "A8b", "A6", "A5"):
    a = np.array([RES[k]["r"][kk] for k in RES])
    print(f"\n全部 {len(a)} 格 {kk}/EMP：中位 {np.median(a):.3f} 胜 {int((a<1).sum())}/{len(a)}"
          f" 最大 {a.max():.3f} 最小 {a.min():.3f}")

print("\n" + "=" * 108)
print("A8a 的显著性")
print("=" * 108)
for h in HOR:
    ks = [f"{m}|{h}" for m in MK]
    print(f"h={h:>3} 显著优于 EMP: {sum(RES[k]['p_a']<0.05 and RES[k]['r']['A8a']<1 for k in ks)}/5",
          f" 显著优于 A5: {sum(RES[k]['p_a5']<0.05 and RES[k]['r']['A8a']<1 for k in ks)}/5",
          f" 显著优于 A6: {sum(RES[k]['p_a6']<0.05 and RES[k]['r']['A8a']<1 for k in ks)}/5")

json.dump(dict(global_hs=float(best), cells=RES),
          open(os.path.join(OUT, "SHAPE_SCHEDULE.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 SHAPE_SCHEDULE.json")
