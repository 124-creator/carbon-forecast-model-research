# -*- coding: utf-8 -*-
"""
实验体系 2/6：多期限密网格（h = 5..200，23 个期限）
=====================================================================
要回答：期限剖面是**真实结构**还是只有 5 个点的插值？
设计：
 * 期限取 {5,10,15,20,25,30,40,50,60,80,100,120,140,160,180,200}
   （共 16 个，超过 W/2=125 的部分 n_eff 很小，如实报告 n_eff）
 * (h0, H) 逐切分重估；A14 与全部对照一起评。
 * 报每个期限的：n_te、n_eff = n_te*STEP/h、CRPS、覆盖率、相对各基准的比值。
 * **关键**：A14 的 w(h)=min(1,h/50) 在 h>50 后恒为 1，与 A5 逐位相同，
   所以长端 A14 不会比 A5 差 —— 这条要在长端格子（h=140..200）上验证。
输出 HORIZON_GRID.json
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm, t as tdist
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


PANEL, DTS = {}, {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    PANEL[m] = np.log(d.price.to_numpy(float))
    DTS[m] = pd.to_datetime(d.date).to_numpy()

ALL = []
for m in MK:
    lp = PANEL[m]
    t0 = int(np.searchsorted(DTS[m], np.datetime64(CUT[m])))
    CC = {h: build(lp, h, t0) for h in HGRID}
    CC = {h: c for h, c in CC.items() if c is not None and c["tr"].sum() >= 30}
    print(f"\n=== {MN[m]} ===  可用期限 {sorted(CC)}")
    # 训练段重估 (h0,H)
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
    print(f"  重估 (h0,H) = ({h0:g}, {Hv:.2f})")
    for h, c in sorted(CC.items()):
        te = c["te"]
        if te.sum() < 20:
            continue
        y = c["y"][te]; mu = c["mu"][te]; sdd = c["sdd"][te]; sdw = c["sd"][te]; xi = c["xi"][te]
        w14 = min(1.0, h / 50.0)
        G = np.tile(GV, (len(y), 1))
        S_A5 = sdd * h ** Hv
        Q_A5 = mu[:, None] * (h0 / (h0 + h)) + S_A5[:, None] * G
        Q_A14 = mu[:, None] * (h0 / (h0 + h)) + S_A5[:, None] * ((1 - w14) * xi + w14 * G)
        Q_EMP = np.vstack([np.quantile(lp[max(0, t - W):t][h:] - lp[max(0, t - W):t][:-h], TAUS)
                           for t in c["ts"][te]])
        Q_RW = (sdd * np.sqrt(h))[:, None] * G
        Q_SD = mu[:, None] * (h0 / (h0 + h)) + (sdd * np.sqrt(h))[:, None] * G
        L = {"A5": crps_q(y, Q_A5), "A14": crps_q(y, Q_A14), "EMP": crps_q(y, Q_EMP),
             "RW": crps_q(y, Q_RW), "SD250": crps_q(y, Q_SD)}
        c80 = float(((y >= Q_A14[:, 2]) & (y <= Q_A14[:, 12])).mean())
        c95 = float(((y >= Q_A14[:, 0]) & (y <= Q_A14[:, 14])).mean())
        row = dict(m=m, h=h, h0=h0, H=Hv, n_te=int(te.sum()),
                   neff=float(len(y) * STEP / h),
                   **{k: float(v.mean()) for k, v in L.items()},
                   c80=c80, c95=c95,
                   p_A14_vs_A5=dm(L["A14"], L["A5"]),
                   p_A14_vs_EMP=dm(L["A14"], L["EMP"]),
                   p_A14_vs_RW=dm(L["A14"], L["RW"]),
                   p_A14_vs_SD250=dm(L["A14"], L["SD250"]),
                   bit_identical=bool(np.allclose(L["A14"], L["A5"])))
        ALL.append(row)
        r = row
        print(f"  h={h:<4} n={r['n_te']:<4} neff={r['neff']:>5.1f}  A5 {r['A5']:.5f}  A14 {r['A14']:.5f} "
              f"({r['A14']/r['A5']:.3f}, p={r['p_A14_vs_A5']:.3f})  /EMP {r['A14']/r['EMP']:.3f}  "
              f"/SD250 {r['A14']/r['SD250']:.3f}  c80 {c80:.3f} c95 {c95:.3f}"
              + ("  [与A5逐位相同]" if r["bit_identical"] else ""))

print("\n" + "=" * 128)
print("汇总：按期限（跨 5 市场合并）")
print("=" * 128)
print(f"{'h':>5}{'格数':>5}{'n_eff 中位':>11}{'A14/A5':>9}{'胜':>7}{'显著胜':>8}{'显著负':>8}"
      f"{'A14/EMP':>10}{'A14/RW':>9}{'A14/SD250':>12}{'c80':>8}{'c95':>8}{'与A5逐位同':>11}")
for h in HGRID:
    s = [r for r in ALL if r["h"] == h]
    if not s:
        continue
    ra = np.array([r["A14"] / r["A5"] for r in s])
    sw = sum(1 for r in s if r["p_A14_vs_A5"] == r["p_A14_vs_A5"] and r["p_A14_vs_A5"] < .05 and r["A14"] < r["A5"])
    sl = sum(1 for r in s if r["p_A14_vs_A5"] == r["p_A14_vs_A5"] and r["p_A14_vs_A5"] < .05 and r["A14"] > r["A5"])
    nb = sum(1 for r in s if r["bit_identical"])
    print(f"{h:>5}{len(s):>5}{np.median([r['neff'] for r in s]):>11.1f}{np.median(ra):>9.4f}"
          f"{int((ra<1).sum()):>4}/{len(s)}{sw:>8}{sl:>8}"
          f"{np.median([r['A14']/r['EMP'] for r in s]):>10.4f}"
          f"{np.median([r['A14']/r['RW'] for r in s]):>9.4f}"
          f"{np.median([r['A14']/r['SD250'] for r in s]):>12.4f}"
          f"{np.median([r['c80'] for r in s]):>8.3f}{np.median([r['c95'] for r in s]):>8.3f}{nb:>8}/{len(s)}")

print("\n" + "=" * 128)
print("关键验证：h>50 的格子上 A14 是否与 A5 逐位相同（构造上应当是）")
print("=" * 128)
bad = [(r["m"], r["h"], r["A14"] / r["A5"]) for r in ALL if r["h"] > 50 and not r["bit_identical"]]
print(f"  h>50 的格子共 {sum(1 for r in ALL if r['h']>50)} 个，其中与 A5 不逐位相同的：{len(bad)}")
if bad:
    for x in bad[:10]:
        print("   ", x)
else:
    print("  ✓ 全部逐位相同 —— A14 在长端继承了 A5 的表现，**结构上不可能变差**")
json.dump(ALL, open(os.path.join(DIR, "HORIZON_GRID.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 HORIZON_GRID.json")
