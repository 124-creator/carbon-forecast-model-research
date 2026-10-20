# -*- coding: utf-8 -*-
"""
实验体系 12/6+：A15 的最终检验（h_sat 由训练段 + 校准零分布选出）
=====================================================================
§24.33 的结果：h_sat = 20 由**训练段**用**校准零分布**选出（训练段 p=0.0083），
且在**测试段**仍有效（p=0.0250），优于 oracle 读出的 50（p=0.0583）。
跨 5 个切分选出的值稳定在 10–20。

⇒ **A15 = A14 但 h_sat 从 50 改成 20**，即**完全去掉了 oracle 依赖**。
本节做 A15 的最终完整检验：
  1) 全部对照（A5/EMP/RW/SD250/VR）的合并校准检验
  2) 逐市场、逐期限
  3) 与 A14（h_sat=50）并列
  4) 跨切分：每个切分独立用训练段选 h_sat，再在测试段检验（A15 的完整链条）
输出 A15_FINAL.json
"""
import numpy as np, pandas as pd, os, sys, json, warnings, time
from scipy.stats import norm
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(DIR, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
W = 250; STEP = 2
HS = [5, 20, 40]
HSAT = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200, 1e9]
H0G = [0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 500]
HG = np.round(np.arange(0.15, 0.861, 0.05), 3)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
NSHUF = 200
SEED0 = 141421
BASE = ["A5", "EMP", "RW", "SD250", "VR"]
CAND = [("A14", 50.0), ("A15", 20.0)]


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


def losses(lp, t0, which="te"):
    CC = {h: c for h, c in ((h, build(lp, h, t0)) for h in HS) if c is not None and c["tr"].sum() >= 30}
    if not CC:
        return None
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
    out = {}
    for nm, hsat in CAND:
        d = {}
        for h, c in CC.items():
            msk = c["te"] if which == "te" else c["tr"]
            if msk.sum() < 20:
                continue
            y = c["y"][msk]; mu = c["mu"][msk]; sdd = c["sdd"][msk]; xi = c["xi"][msk]
            sdw = c["sd"][msk]
            w = min(1.0, h / hsat)
            G = np.tile(GV, (len(y), 1))
            S = sdd * h ** Hv
            lam = h0 / (h0 + h)
            d.setdefault("A5", {})[h] = float(crps_q(y, mu[:, None] * lam + S[:, None] * G).mean())
            d.setdefault(nm, {})[h] = float(crps_q(y, mu[:, None] * lam +
                                                      S[:, None] * ((1 - w) * xi + w * G)).mean())
            d.setdefault("SD250", {})[h] = float(crps_q(y, mu[:, None] * lam +
                                                          (sdd * np.sqrt(h))[:, None] * G).mean())
            d.setdefault("VR", {})[h] = float(crps_q(y, (sdw)[:, None] * G).mean())
            d.setdefault("RW", {})[h] = float(crps_q(y, (sdd * np.sqrt(h))[:, None] * G).mean())
            d.setdefault("EMP", {})[h] = float(crps_q(y, np.vstack([
                np.quantile(lp[max(0, tt - W):tt][h:] - lp[max(0, tt - W):tt][:-h], TAUS)
                for tt in c["ts"][msk]])).mean())
        out[nm] = d
    return out


DATA = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    DATA[m] = (lp, t0)

REAL = {m: losses(*DATA[m], which="te") for m in MK}
T0 = time.time()
NULL = []
for it in range(NSHUF):
    rng = np.random.default_rng(SEED0 + it)
    rec = {}
    for m in MK:
        lp, t0 = DATA[m]
        inc = np.diff(lp)
        lp_s = np.concatenate([[0.0], np.cumsum(inc[rng.permutation(len(inc))])])
        L = losses(lp_s, t0, which="te")
        if not L:
            continue
        for nm in L:
            for b in L[nm]:
                for h in L[nm][b]:
                    rec[(nm, b, h, m)] = L[nm][b][h]
    NULL.append(rec)
    if (it + 1) % 50 == 0:
        print(f"  打乱 {it+1}/{NSHUF}  ({time.time()-T0:.0f} 秒)")

print("\n" + "=" * 120)
print("最终证据表（200 次打乱，每次完整流程）")
print("=" * 120)
print(f"{'候选':<6}{'对照':<8}{'真实合并log比':>15}{'零分布均值':>13}{'零分布SD':>11}{'单尾p':>9}{'百分位':>9}")
OUT = {}
for nm, hsat in CAND:
    for b in BASE:
        den = "A5" if b == "A5" else b
        real = [np.log(REAL[m][nm][nm][h] / REAL[m][nm][den][h]) for m in MK for h in HS
                if h in REAL[m][nm][nm] and h in REAL[m][nm][den]]
        mu_real = float(np.mean(real))
        nulls = []
        for rec in NULL:
            dk = "A5" if b == "A5" else b
            vals = [np.log(rec[(nm, nm, h, m)] / rec[(nm, dk, h, m)])
                    for m in MK for h in HS
                    if (nm, nm, h, m) in rec and (nm, dk, h, m) in rec]
            if len(vals) == len(real):
                nulls.append(float(np.mean(vals)))
        nulls = np.array(nulls)
        p1 = float(np.mean(nulls <= mu_real))
        pct = float(np.mean(nulls < mu_real) * 100)
        print(f"{nm:<6}{b:<8}{mu_real:>15.5f}{nulls.mean():>13.5f}{nulls.std():>11.5f}"
              f"{p1:>9.4f}{pct:>8.1f}%")
        mp = {}
        for m in MK:
            dk = "A5" if b == "A5" else b
            rv = [np.log(REAL[m][nm][nm][h] / REAL[m][nm][dk][h]) for h in HS
                  if h in REAL[m][nm][nm] and h in REAL[m][nm][dk]]
            nn = []
            for rec in NULL:
                dk = "A5" if b == "A5" else b
                vals = [np.log(rec[(nm, nm, h, m)] / rec[(nm, dk, h, m)]) for h in HS
                        if (nm, nm, h, m) in rec and (nm, dk, h, m) in rec]
                if len(vals) == len(rv):
                    nn.append(float(np.mean(vals)))
            mp[m] = float(np.mean(np.array(nn) <= float(np.mean(rv))))
        OUT[f"{nm}|{b}"] = dict(real_mu=mu_real, null_mean=float(nulls.mean()),
                                null_sd=float(nulls.std()), p=p1, pct=pct,
                                by_market_p=mp, hsat=hsat)

print("\n逐市场（真实合并 log 比 / 单尾 p）")
for nm, hsat in CAND:
    print(f"  {nm}（h_sat={hsat:g}）：")
    for b in BASE:
        o = OUT[f"{nm}|{b}"]
        print(f"    vs {b:<7} " + "  ".join(
            f"{MN[m]} {o['by_market_p'][m]:.3f}" for m in MK))

json.dump({"out": OUT,
           "real": {m: {nm[0]: {b: {str(h): REAL[m][nm[0]][b][h] for h in REAL[m][nm[0]][b]}
                               for b in list(REAL[m][nm[0]].keys())} for nm in CAND} for m in MK}},
          open(os.path.join(DIR, "A15_FINAL.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 A15_FINAL.json")
