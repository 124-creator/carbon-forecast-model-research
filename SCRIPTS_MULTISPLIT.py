# -*- coding: utf-8 -*-
"""
ROBUST：多切分稳健性 —— 长期限结论是否依赖单一训练/测试切分
=============================================================
动机：CEA h=120 在主切分下只有 51 个训练原点，整个长期限结论挂在一个切分上。
做法：对每个市场用样本长度 55% / 65% / 75% 三个切分点，
      每个切分各自重新拟合 (h0, H)（仍然只用训练段），
      然后评估 A5 vs EMP 与 RW，并报 PICP。
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS); J95 = (0, 14)
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
H0G = [0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 500]
HG = np.round(np.arange(0.15, 0.861, 0.05), 3)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
SPL = [0.55, 0.65, 0.75]


def crps_q(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


def build(lp, h, t0):
    rec = []
    for t in range(W + h + 5, len(lp) - h, STEP):
        win = lp[t - W:t]; rr = win[h:] - win[:-h]; rd = np.diff(win)
        sdw = np.std(rr, ddof=1); sdd = np.std(rd, ddof=1)
        if sdw <= 1e-12 or sdd <= 1e-12:
            continue
        rec.append((t, float(rr.mean()), sdw, sdd,
                    np.quantile((rr - rr.mean()) / sdw, TAUS)))
    if len(rec) < 40:
        return None
    ts = np.array([r[0] for r in rec])
    return dict(h=h, ts=ts, mu=np.array([r[1] for r in rec]),
                sd=np.array([r[2] for r in rec]), sdd=np.array([r[3] for r in rec]),
                xi=np.vstack([r[4] for r in rec]), y=lp[ts + h] - lp[ts],
                tr=(ts <= t0 - h - 2), te=(ts > t0))


def Qof(c, idx, lam, Hval, sh):
    mu = c["mu"][idx]
    s = c["sd"][idx] if Hval is None else c["sdd"][idx] * c["h"] ** Hval
    xi = c["xi"][idx] if sh == "X" else np.tile(GV, (len(mu), 1))
    return mu[:, None] * lam + s[:, None] * xi


def loss(c, idx, lam, Hval, sh):
    return crps_q(c["y"][idx], Qof(c, idx, lam, Hval, sh))


PANEL = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    PANEL[m] = np.log(d.price.to_numpy(float))

print("=" * 150)
print("多切分稳健性：每个切分独立重估 (h0, H)，再在未触碰的测试段评估")
print("=" * 150)
print(f"{'切分':>6}{'市场':>11}{'h':>5}{'n_tr':>6}{'n_te':>6}{'h0*':>7}{'H*':>6}"
      f"{'A5':>10}{'EMP':>10}{'RW':>10}{'vsEMP':>8}{'vsRW':>8}{'P95':>7}{'P95:EMP':>8}")
ALL = []
for frac in SPL:
    for m in MK:
        lp = PANEL[m]
        t0 = int(len(lp) * frac)
        CC = {h: build(lp, h, t0) for h in HOR}
        CC = {h: c for h, c in CC.items() if c is not None and c["tr"].sum() >= 30}
        if not CC:
            continue
        best = (1e9, None, None)
        for Hval in HG:
            for h0 in H0G:
                v = float(np.mean([loss(c, c["tr"], h0 / (h0 + h), Hval, "G").mean()
                                   for h, c in CC.items()]))
                if v < best[0]:
                    best = (v, h0, Hval)
        _, h0, Hv = best
        for h, c in sorted(CC.items()):
            te = c["te"]
            if te.sum() < 20:
                continue
            l5 = loss(c, te, h0 / (h0 + h), Hv, "G")
            le = loss(c, te, 1.0, None, "X")
            lr = loss(c, te, 0.0, None, "X")
            Q = Qof(c, te, h0 / (h0 + h), Hv, "G"); yy = c["y"][te]
            pch = float(((yy >= Q[:, J95[0]]) & (yy <= Q[:, J95[1]])).mean())
            Qe = Qof(c, te, 1.0, None, "X")
            pce = float(((yy >= Qe[:, J95[0]]) & (yy <= Qe[:, J95[1]])).mean())
            ALL.append(dict(frac=frac, m=m, h=h, h0=h0, H=Hv,
                            a5=float(l5.mean()), emp=float(le.mean()), rw=float(lr.mean()),
                            p95=pch, p95e=pce, n_tr=int(c["tr"].sum()), n_te=int(te.sum())))
            r = ALL[-1]
            print(f"{frac:>6.2f}{m:>11}{h:>5}{r['n_tr']:>6}{r['n_te']:>6}{h0:>7}{Hv:>6.2f}"
                  f"{r['a5']:>10.5f}{r['emp']:>10.5f}{r['rw']:>10.5f}"
                  f"{(r['a5']/r['emp']-1)*100:>7.1f}%{(r['a5']/r['rw']-1)*100:>7.1f}%"
                  f"{pch:>7.3f}{pce:>8.3f}")
    print("-" * 150)

print("\n" + "=" * 150)
print("汇总")
print("=" * 150)
for frac in SPL:
    s = [r for r in ALL if r["frac"] == frac]
    de = [r["a5"] / r["emp"] - 1 for r in s]; dr = [r["a5"] / r["rw"] - 1 for r in s]
    print(f"  切分 {frac:.2f}: 共 {len(s)} 格 | vsEMP 胜 {sum(1 for x in de if x<0)}/{len(s)} "
          f"中位 {np.median(de)*100:+.1f}% | vsRW 胜 {sum(1 for x in dr if x<0)}/{len(s)} "
          f"中位 {np.median(dr)*100:+.1f}% | P95 达标 {sum(1 for r in s if r['p95']>=0.90)}/{len(s)}")
print("\n  按期限（跨切分合并）：")
for h in HOR:
    s = [r for r in ALL if r["h"] == h]
    if not s:
        continue
    de = [r["a5"] / r["emp"] - 1 for r in s]; dr = [r["a5"] / r["rw"] - 1 for r in s]
    print(f"    h={h:>3}: {len(s):>2} 格 | vsEMP 胜 {sum(1 for x in de if x<0)}/{len(s)} "
          f"中位 {np.median(de)*100:+6.1f}% | vsRW 胜 {sum(1 for x in dr if x<0)}/{len(s)} "
          f"中位 {np.median(dr)*100:+6.1f}% | P95 中位 {np.median([r['p95'] for r in s]):.3f}")
print("\n  按市场（跨切分合并）：")
for m in MK:
    s = [r for r in ALL if r["m"] == m]
    if not s:
        continue
    de = [r["a5"] / r["emp"] - 1 for r in s]; dr = [r["a5"] / r["rw"] - 1 for r in s]
    print(f"    {m:<11}: {len(s):>2} 格 | vsEMP 胜 {sum(1 for x in de if x<0)}/{len(s)} "
          f"中位 {np.median(de)*100:+6.1f}% | vsRW 胜 {sum(1 for x in dr if x<0)}/{len(s)} "
          f"中位 {np.median(dr)*100:+6.1f}%")
print("\n  参数稳定性（H* 随切分变化）：")
for m in MK:
    for h in [None]:
        vs = [r["H"] for r in ALL if r["m"] == m and r["h"] == HOR[0]]
        print(f"    {m:<11} H* = " + ", ".join(f"{v:.2f}" for v in vs))
json.dump(ALL, open(os.path.join(OUT, "ROBUST_SPLITS.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 ROBUST_SPLITS.json")
