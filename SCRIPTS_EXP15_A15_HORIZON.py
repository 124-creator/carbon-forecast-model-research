# -*- coding: utf-8 -*-
"""
实验体系 15/6+：A15 最终规格的完整密期限网格（替换 §24.31.2 的 A14 版）
=====================================================================
§24.31.2 的密期限网格用的是 A14（h_sat=50，oracle）。现在最终规格是 A15（h_sat=20），
必须重算，否则图 1 与论文的模型不一致。
同时输出 **Table 2（论文主表）** 的候选结构。

设计：
 * 期限 16 个（5→200），5 个市场，主切分，参数逐市场在训练段重估。
 * 同时算 A15(h_sat=20)、A14(h_sat=50)、A5、SD250、RW、EMP、VR 的 CRPS 与覆盖。
 * 报 n_eff（诚实标注长端无统计效力）。
输出 A15_HORIZON.json
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(DIR, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
W = 250; STEP = 2
HGRID = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200]
H0G = [0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 500]
HG = np.round(np.arange(0.15, 0.861, 0.05), 3)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
HSAT15 = 20.0


def crps_q(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


def build(lp, h, t0):
    rec = []
    for t in range(W + h + 5, len(lp) - h, STEP):
        win = lp[max(0, t - W):t]
        rr = win[h:] - win[:-h]
        rd = np.diff(win)
        sdw = np.std(rr, ddof=1); sdd = np.std(rd, ddof=1)
        if sdw <= 1e-12 or sdd <= 1e-12:
            continue
        rec.append((t, float(rr.mean()), sdw, sdd,
                    np.quantile((rd - rd.mean()) / sdd, TAUS)))
    if len(rec) < 40:
        return None
    ts = np.array([r[0] for r in rec])
    return dict(h=h, ts=ts, mu=np.array([r[1] for r in rec]), sd=np.array([r[2] for r in rec]),
                sdd=np.array([r[3] for r in rec]), xi=np.vstack([r[4] for r in rec]),
                y=lp[ts + h] - lp[ts], tr=(ts <= t0 - h - 2), te=(ts > t0))


ALL = []
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    CC = {h: c for h, c in ((h, build(lp, h, t0)) for h in HGRID)
          if c is not None and c["tr"].sum() >= 30}
    best = (1e9, None, None)
    for Hval in HG:
        for h0 in H0G:
            v = float(np.mean([crps_q(c["y"][c["tr"]],
                                     (c["mu"][c["tr"]] * (h0 / (h0 + h)) +
                                      (c["sdd"][c["tr"]] * h ** Hval))[:, None] * GV).mean()
                                for h, c in CC.items()]))
            if v < best[0]:
                best = (v, h0, Hval)
    _, h0, Hv = best
    print(f"\n=== {MN[m]} ===  (h0,H) = ({h0:g}, {Hv:.2f})  期限 {sorted(CC)}")
    for h, c in sorted(CC.items()):
        te = c["te"]
        if te.sum() < 20:
            continue
        y = c["y"][te]; mu = c["mu"][te]; sdd = c["sdd"][te]; sdw = c["sd"][te]; xi = c["xi"][te]
        G = np.tile(GV, (len(y), 1))
        S = sdd * h ** Hv
        lam = h0 / (h0 + h)
        w15 = min(1.0, h / HSAT15); w14 = min(1.0, h / 50.0)
        Q_A5 = mu[:, None] * lam + S[:, None] * G
        Q_A15 = mu[:, None] * lam + S[:, None] * ((1 - w15) * xi + w15 * G)
        Q_A14 = mu[:, None] * lam + S[:, None] * ((1 - w14) * xi + w14 * G)
        Q_EMP = np.vstack([np.quantile(lp[max(0, t - W):t][h:] - lp[max(0, t - W):t][:-h], TAUS)
                           for t in c["ts"][te]])
        Q_RW = (sdd * np.sqrt(h))[:, None] * G
        Q_SD = mu[:, None] * lam + (sdd * np.sqrt(h))[:, None] * G
        Q_VR = (sdw)[:, None] * G
        row = dict(m=m, h=h, h0=h0, H=Hv, n_te=int(te.sum()),
                   neff=float(len(y) * STEP / h),
                   A5=float(crps_q(y, Q_A5).mean()), A14=float(crps_q(y, Q_A14).mean()),
                   A15=float(crps_q(y, Q_A15).mean()), EMP=float(crps_q(y, Q_EMP).mean()),
                   RW=float(crps_q(y, Q_RW).mean()), SD250=float(crps_q(y, Q_SD).mean()),
                   VR=float(crps_q(y, Q_VR).mean()),
                   c80=float(((y >= Q_A15[:, 2]) & (y <= Q_A15[:, 12])).mean()),
                   c95=float(((y >= Q_A15[:, 0]) & (y <= Q_A15[:, 14])).mean()),
                   c80_A5=float(((y >= Q_A5[:, 2]) & (y <= Q_A5[:, 12])).mean()),
                   c95_A5=float(((y >= Q_A5[:, 0]) & (y <= Q_A5[:, 14])).mean()))
        ALL.append(row)
        r = row
        print(f"  h={h:<4} n={r['n_te']:<4} neff={r['neff']:>5.1f}  A5 {r['A5']:.5f}  "
              f"A15 {r['A15']:.5f} ({r['A15']/r['A5']:.3f})  A14 {r['A14']:.5f} ({r['A14']/r['A5']:.3f})  "
              f"/EMP {r['A15']/r['EMP']:.3f}  /RW {r['A15']/r['RW']:.3f}  /SD250 {r['A15']/r['SD250']:.3f}  "
              f"c80 {r['c80']:.3f} c95 {r['c95']:.3f}")

print("\n" + "=" * 122)
print("Table 2 候选：A15（h_sat=20）在密期限网格上的表现")
print("=" * 122)
print(f"{'h':>5}{'n_eff中位':>10}{'A15/A5':>9}{'A15/A14':>10}{'A15/EMP':>10}{'A15/RW':>9}"
      f"{'A15/SD250':>12}{'A15/VR':>9}{'c80':>8}{'c95':>8}")
for h in HGRID:
    s = [r for r in ALL if r["h"] == h]
    if not s:
        continue
    f = lambda k: np.median([r["A15"] / r[k] for r in s])
    print(f"{h:>5}{np.median([r['neff'] for r in s]):>10.1f}{f('A5'):>9.4f}{f('A14'):>10.4f}"
          f"{f('EMP'):>10.4f}{f('RW'):>9.4f}{f('SD250'):>12.4f}{f('VR'):>9.4f}"
          f"{np.median([r['c80'] for r in s]):>8.3f}{np.median([r['c95'] for r in s]):>8.3f}")

print("\n逐市场（合并全部期限）：")
print(f"{'市场':<8}{'A15/A5':>9}{'A15/A14':>10}{'A15/EMP':>10}{'A15/RW':>9}{'A15/SD250':>12}{'A15/VR':>9}")
for m in MK:
    s = [r for r in ALL if r["m"] == m]
    if not s:
        continue
    f = lambda k: np.median([r["A15"] / r[k] for r in s])
    print(f"{MN[m]:<8}{f('A5'):>9.4f}{f('A14'):>10.4f}{f('EMP'):>10.4f}{f('RW'):>9.4f}"
          f"{f('SD250'):>12.4f}{f('VR'):>9.4f}")

json.dump(ALL, open(os.path.join(DIR, "A15_HORIZON.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 A15_HORIZON.json")
