# -*- coding: utf-8 -*-
"""
实验体系 9/6+：正确零假设下的统计效力（exp7 的加强版）
=====================================================================
§24.31/exp7/exp8 的一条链：
  exp7  发现打乱数据上 DM 失准（p<0.05 比例 0.32）
  exp8  试了四种 DM 实现（渐近 L=10/40/80、块自举、置换），
        **全部**失准 ⇒ 问题不在 DM 实现，而在零假设的设定。
  本节确立**正确的零假设与正确的检验**：

  H0（正确）：在**没有任何可利用的时序信息**的序列上，
              A14 相对 A5 的期望损失差不为负（即 A14 的改进不来自数据的信息）。
  检验量：跨市场的**合并**统计量，而不是逐格 p 值。
            T = 汇总所有格子的 log(A14/A5) 的均值 / 其标准误
  零分布：每次打乱后跑**完全相同**的流程（独立重估参数），
          得到 T 的零分布。
  判定：真实 T 若落在零分布的左尾 ⇒ 改进是真的。

  为什么用"合并统计量"而不是逐格 p 值：
  逐格 p 值在 15 个格子上功效极低（真实只有 1 格显著），
  而**合并**检验利用了全部格子的信息，正是"整体改进"这个论断的正确形式。
  这是 §24.24 订正过的错误教训（"报告重叠原点的 p 值而不给 n_eff"）的正面应用。

  附加：同时报"市场层面的检验"（每个市场一个 T）与"期限层面的检验"，
        看改进是否在某一维度上特别显著。
输出 POWER_TEST.json
"""
import numpy as np, pandas as pd, os, sys, json, warnings, time
from scipy.stats import norm, t as tdist
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(DIR, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
W = 250; STEP = 2
HS = [5, 20, 40]
H0G = [0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 500]
HG = np.round(np.arange(0.15, 0.861, 0.05), 3)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
NSHUF = 200
SEED0 = 31415


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


def pipeline(lp, t0):
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
    for h, c in sorted(CC.items()):
        te = c["te"]
        if te.sum() < 20:
            continue
        y = c["y"][te]; mu = c["mu"][te]; sdd = c["sdd"][te]; xi = c["xi"][te]
        w14 = min(1.0, h / 50.0)
        G = np.tile(GV, (len(y), 1))
        S = sdd * h ** Hv
        l5 = crps_q(y, mu[:, None] * (h0 / (h0 + h)) + S[:, None] * G)
        l14 = crps_q(y, mu[:, None] * (h0 / (h0 + h)) + S[:, None] * ((1 - w14) * xi + w14 * G))
        out[h] = (float(l5.mean()), float(l14.mean()))
    return out


def stat_T(pairs):
    """合并统计量：所有格子的 log 比值的均值 / 标准误（按格等权）。"""
    lr = np.array([np.log(b / a) for a, b in pairs])
    if len(lr) < 3:
        return np.nan, np.nan
    se = lr.std(ddof=1) / np.sqrt(len(lr))
    return (float(lr.mean()), float(lr.mean() / se)) if se > 0 else (float(lr.mean()), np.nan)


DATA = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    DATA[m] = (lp, t0)

REAL = {}
print("=" * 110)
print("真实数据")
print("=" * 110)
for m in MK:
    lp, t0 = DATA[m]
    L = pipeline(lp, t0)
    REAL[m] = L
    pr = [(a, b) for a, b in L.values()]
    mu_lr, T = stat_T(pr)
    print(f"  {MN[m]:<6} 逐格比 " + " ".join(f"h{h}:{b/a:.4f}" for h, (a, b) in L.items())
          + f"   合并 log 比 {mu_lr:+.5f}  T={T:+.3f}")

allpairs = [(a, b) for m in MK for a, b in REAL[m].values()]
mu_all, T_all = stat_T(allpairs)
print(f"\n  >>> 全样本合并：log(A14/A5) 均值 {mu_all:+.5f}   T = {T_all:+.3f}")
by_h = {}
for h in HS:
    pr = [(REAL[m][h][0], REAL[m][h][1]) for m in MK if h in REAL[m]]
    mu_h, T_h = stat_T(pr)
    by_h[h] = dict(mu=mu_h, T=T_h, n=len(pr))
    print(f"  期限 h={h:<4} log 比 {mu_h:+.5f}  T={T_h:+.3f}  ({len(pr)} 格)")

print("\n" + "=" * 110)
print(f"零分布：打乱序列 × {NSHUF} 次（每次跑完整流程）")
print("=" * 110)
T0 = time.time()
null_T = []
null_muh = []
null_by_m = {m: [] for m in MK}
null_by_h = {h: [] for h in HS}
for it in range(NSHUF):
    rng = np.random.default_rng(SEED0 + it)
    rec = []
    for m in MK:
        lp, t0 = DATA[m]
        inc = np.diff(lp)
        lp_s = np.concatenate([[0.0], np.cumsum(inc[rng.permutation(len(inc))])])
        L = pipeline(lp_s, t0)
        if not L:
            continue
        pr = [(a, b) for a, b in L.values()]
        mu_m, T_m = stat_T(pr)
        null_by_m[m].append(mu_m)
        for h, (a, b) in L.items():
            null_by_h[h].append(np.log(b / a))
        rec += pr
    mu_r, T_r = stat_T(rec)
    null_T.append(T_r); null_muh.append(mu_r)
    if (it + 1) % 25 == 0:
        print(f"  第 {it+1:>4} 次：T 零分布均值 {np.nanmean(null_T):+.3f}  "
              f"标准差 {np.nanstd(null_T):.3f}  （{time.time()-T0:.0f} 秒）")

null_T = np.array(null_T, float); null_muh = np.array(null_muh, float)
p_T = float(np.mean(null_T <= T_all))
p_mu = float(np.mean(null_muh <= mu_all))
print(f"\n  零分布 T：均值 {np.nanmean(null_T):+.3f}  标准差 {np.nanstd(null_T):.3f}  "
      f"2.5% {np.nanpercentile(null_T,2.5):+.3f}  50% {np.nanpercentile(null_T,50):+.3f}  "
      f"97.5% {np.nanpercentile(null_T,97.5):+.3f}")
print(f"  零分布 log 比：均值 {np.nanmean(null_muh):+.5f}  标准差 {np.nanstd(null_muh):.5f}  "
      f"2.5% {np.nanpercentile(null_muh,2.5):+.5f}  97.5% {np.nanpercentile(null_muh,97.5):+.5f}")
print(f"\n  >>> 真实值：T = {T_all:+.3f}   log 比 = {mu_all:+.5f}")
print(f"      单尾 p（T）      = {p_T:.4f}   {'✓ 超出零分布' if p_T < 0.05 else '✗ 未超出'}")
print(f"      单尾 p（log 比） = {p_mu:.4f}   {'✓ 超出零分布' if p_mu < 0.05 else '✗ 未超出'}")

print("\n" + "=" * 110)
print("分层：市场层面与期限层面的零分布")
print("=" * 110)
for m in MK:
    pr = [(REAL[m][h][0], REAL[m][h][1]) for h in REAL[m]]
    mu_m, T_m = stat_T(pr)
    z = np.array(null_by_m[m], float)
    pm = float(np.mean(z <= mu_m))
    print(f"  {MN[m]:<6} 真实 log 比 {mu_m:+.5f} (T={T_m:+.3f})   "
          f"零分布 均值 {np.nanmean(z):+.5f} 标准差 {np.nanstd(z):.5f}   p = {pm:.4f}")
for h in HS:
    z = np.array(null_by_h[h], float)
    ph = float(np.mean(z <= by_h[h]["mu"]))
    print(f"  h={h:<5} 真实 log 比 {by_h[h]['mu']:+.5f} (T={by_h[h]['T']:+.3f})   "
          f"零分布 均值 {np.nanmean(z):+.5f} 标准差 {np.nanstd(z):.5f}   p = {ph:.4f}")

json.dump({"real_all": {"mu": mu_all, "T": T_all},
           "real_by_market": {m: dict(mu=float(np.mean([np.log(b/a) for a, b in REAL[m].values()])),
                                      pairs={str(h): [REAL[m][h][0], REAL[m][h][1]] for h in REAL[m]})
                              for m in MK},
           "real_by_h": by_h,
           "null_T": null_T[~np.isnan(null_T)].tolist(),
           "null_mu": null_muh[~np.isnan(null_muh)].tolist(),
           "p_T": p_T, "p_mu": p_mu,
           "null_by_market": {m: [float(x) for x in v if x == x] for m, v in null_by_m.items()},
           "null_by_h": {str(h): [float(x) for x in v if x == x] for h, v in null_by_h.items()}},
          open(os.path.join(DIR, "POWER_TEST.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 POWER_TEST.json")
