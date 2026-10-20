# -*- coding: utf-8 -*-
"""
交叉点定律：EMP 什么时候开始坏？
================================
机制假说：EMP 的宽度来自窗口内的 h 步收益，共 W-h+1 个但高度重叠，
有效独立个数约 (W-h)/h。当这个数掉到 10 以下，窗口宽度估计就不可靠，
A5 的参数化 h^H 标度开始占优。

若成立，交叉点 h* 应满足 (W-h*)/h* = k，即
    h* = W / (k+1)
=> h* 与 W 成正比。取 k=10，W=250 时 h*=22.7，正好在观测到的 h=20 附近。

检验：W in {125,250,500} 三档，求每档的 A5/EMP 曲线与交叉点。
预测：h* = 11.4 / 22.7 / 45.5。
如果三档交叉点按比例移动，这条定律成立；如果不动，机制解释是错的。
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
STEP = 2
WLIST = [125, 250, 500]
HS = {125: [2, 3, 5, 8, 12, 16, 20, 30, 40, 60],
      250: [2, 3, 5, 8, 12, 16, 20, 30, 40, 60, 80, 120],
      500: [2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 60, 80, 120, 160, 200]}


def crps_row(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


print("=" * 118)
print("交叉点定律：h* = W/(k+1)，k≈10  =>  h* = 11.4 / 22.7 / 45.5")
print("=" * 118)
ALL = {}
for W in WLIST:
    print(f"\n{'='*118}\nW = {W}   预测交叉点 h* ≈ {W/11:.1f}\n{'='*118}")
    print(f"{'h':>4}{'(W-h)/h':>9} | {'A5/EMP':>8}{'A5/RW':>7}{'worst':>7} | "
          f"{'CEA':>7}{'GUANGZHOU':>10}{'SHANGHAI':>9}{'HUBEI':>7}{'EU':>7} | {'胜EMP':>6}")
    for h in HS[W]:
        per, win, allr = {}, {}, []
        for m in MK:
            d = p[p.market == m].sort_values("date").reset_index(drop=True)
            lp = np.log(d.price.to_numpy(float))
            dt = pd.to_datetime(d.date).to_numpy()
            cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
            t0 = int(np.searchsorted(dt, np.datetime64(cut)))
            h0, Hv = PAR[m]
            lo = max(W + h + 5, t0)
            ts = list(range(lo, len(lp) - h, STEP))
            if len(ts) < 30:
                continue
            Qa, Qe, Qw, y = [], [], [], []
            for t in ts:
                win_ = lp[t - W:t]
                rr = win_[h:] - win_[:-h]; rd = np.diff(win_)
                mu = rr.mean(); sdh = np.std(rr, ddof=1); sdd = np.std(rd, ddof=1)
                xi = np.quantile((rr - rr.mean()) / sdh, TAUS)
                Qa.append(mu * (h0 / (h0 + h)) + sdd * h ** Hv * GV)
                Qe.append(mu + sdh * xi)
                Qw.append(sdd * np.sqrt(h) * GV)
                y.append(lp[t + h] - lp[t])
            y = np.array(y)
            ca = crps_row(y, np.array(Qa)).mean()
            ce = crps_row(y, np.array(Qe)).mean()
            cw = crps_row(y, np.array(Qw)).mean()
            per[m] = ca / ce; win[m] = ca / cw; allr.append(ca / ce)
            ALL[f"{W}|{m}|{h}"] = dict(neff=len(y), ratio_emp=float(ca / ce),
                                       ratio_rw=float(ca / cw),
                                       crps_a5=float(ca), crps_emp=float(ce), crps_rw=float(cw))
        if not allr:
            continue
        med = float(np.median(list(per.values())))
        print(f"{h:>4}{(W-h)/h:>9.1f} | {med:>8.3f}{np.median(list(win.values())):>7.3f}"
              f"{'':>7} | {per.get('CEA',np.nan):>7.3f}{per.get('GUANGZHOU',np.nan):>10.3f}"
              f"{per.get('SHANGHAI',np.nan):>9.3f}{per.get('HUBEI',np.nan):>7.3f}"
              f"{per.get('EU',np.nan):>7.3f} | {sum(v<1 for v in allr):>3}/{len(allr)}")

print("\n" + "=" * 118)
print("交叉点估计：A5/EMP 中位首次跌破 1 的期限（在网格上插值）")
print("=" * 118)
for W in WLIST:
    hs = sorted(int(k.split("|")[2]) for k in ALL if k.startswith(f"{W}|"))
    hs = sorted(set(hs))
    med = []
    for h in hs:
        v = [ALL[f"{W}|{m}|{h}"]["ratio_emp"] for m in MK if f"{W}|{m}|{h}" in ALL]
        med.append((h, float(np.median(v))))
    cross = None
    for i in range(1, len(med)):
        if med[i][1] < 1 <= med[i - 1][1]:
            x0, y0 = med[i - 1]; x1, y1 = med[i]
            cross = x0 + (y0 - 1) / (y0 - y1) * (x1 - x0)
            break
    print(f"W={W:>4}: 观测交叉点 h* = " + (f"{cross:.1f}" if cross else "未交叉")
          + f"   预测 W/11 = {W/11:.1f}   W/23 = {W/23:.1f}"
          + f"   比值 观测/预测(W/11) = " + (f"{cross/(W/11):.2f}" if cross else "—"))
    print("        曲线: " + "  ".join(f"h={h}:{v:.3f}" for h, v in med))

print("\n" + "=" * 118)
print("用 (W-h)/h 作为横轴：三档 W 的曲线是否重合？")
print("=" * 118)
print(f"{'x=(W-h)/h':>11}" + "".join(f"{'W='+str(w):>10}" for w in WLIST))
grid = [1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 80, 120]
for g in grid:
    row = f"{g:>11.1f}"
    for W in WLIST:
        cand = [(abs((W - h) / h - g), h) for h in set(HS[W])]
        cand.sort()
        d, h = cand[0]
        if d > 0.45 * g:
            row += f"{'—':>10}"; continue
        v = [ALL[f"{W}|{m}|{h}"]["ratio_emp"] for m in MK if f"{W}|{m}|{h}" in ALL]
        row += f"{np.median(v):>10.3f}" if v else f"{'—':>10}"
    print(row)

json.dump(ALL, open(os.path.join(OUT, "CROSSOVER_LAW.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 CROSSOVER_LAW.json")
