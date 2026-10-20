# -*- coding: utf-8 -*-
"""
实验体系 7/6+：蒙特卡洛零假设分布（把"打乱序列"升级为可判定检验）
=====================================================================
§24.31.4 的 N2 留下唯一弱点：CEA 在**打乱**序列上仍有 2/5 显著胜。
单次打乱无法判定这是"流程产物"还是"偶然"。本节做**真正的检验**：

  H0：A14 相对 A5 的改进来自**流程**（拟合、检验、选择），不来自数据的时序结构。
  做法：把每个市场的对数价格增量**随机置换**后重建价格路径，
        在打乱数据上跑**完全相同**的流程（独立重估 (h0,H) → 测试段评估 → DM 检验），
        重复 N 次，得到"零假设下显著胜格数"的分布。
  判定：真实数据的显著胜格数若落在该分布的 95% 分位以上 ⇒ 改进是真的。

另外三件事：
  M2 **DM 检验的校准性**：零假设下 p 值应服从 Uniform(0,1)。用 KS 检验。
  M3 **零假设下的效应量分布**：零假设下 A14/A5 中位比的分布，
      真实值若远低于其 95% 分位 ⇒ 效应量显著非零。
  M4 **逐市场、逐期限的零分布**：定位哪些格子的"显著"最可能是流程产物。

输出 MC_PLACEBO.json
"""
import numpy as np, pandas as pd, os, sys, json, warnings, time
from scipy.stats import norm, t as tdist, kstest
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(DIR, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
W = 250; STEP = 2
HS = [5, 20, 40]                    # A14 的增益所在的期限
H0G = [0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 500]
HG = np.round(np.arange(0.15, 0.861, 0.05), 3)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
NSHUF = 60
SEED0 = 20261021


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
    return float(2 * (1 - tdist.cdf(abs(s), df = n - 1)))


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
    """完整流程：独立重估 (h0,H) → 测试段评估 A5 与 A14 → DM。返回逐格结果。"""
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
    out = []
    for h, c in sorted(CC.items()):
        te = c["te"]
        if te.sum() < 20:
            continue
        y = c["y"][te]; mu = c["mu"][te]; sdd = c["sdd"][te]; xi = c["xi"][te]
        w14 = min(1.0, h / 50.0)
        G = np.tile(GV, (len(y), 1))
        S = sdd * h ** Hv
        Q_A5 = mu[:, None] * (h0 / (h0 + h)) + S[:, None] * G
        Q_A14 = mu[:, None] * (h0 / (h0 + h)) + S[:, None] * ((1 - w14) * xi + w14 * G)
        l5 = crps_q(y, Q_A5); l14 = crps_q(y, Q_A14)
        out.append(dict(h=h, n_te=int(te.sum()), h0=h0, H=Hv,
                        a5=float(l5.mean()), a14=float(l14.mean()),
                        ratio=float(l14.mean() / l5.mean()),
                        p=dm(l14, l5), win=bool(dm(l14, l5) == dm(l14, l5) and dm(l14, l5) < .05 and l14.mean() < l5.mean()),
                        loss=bool(dm(l14, l5) == dm(l14, l5) and dm(l14, l5) < .05 and l14.mean() > l5.mean())))
    return out


DATA = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    DATA[m] = (lp, t0)

print("=" * 112)
print("真实数据（基准）")
print("=" * 112)
REAL = {}
for m in MK:
    lp, t0 = DATA[m]
    r = pipeline(lp, t0)
    REAL[m] = r
    nw = sum(x["win"] for x in r); nl = sum(x["loss"] for x in r)
    print(f"  {MN[m]:<6} 中位比 {np.median([x['ratio'] for x in r]):.4f}  显著胜 {nw}/{len(r)}  显著负 {nl}/{len(r)}  "
          f"逐格 p {[round(x['p'],3) for x in r]}")

T0 = time.time()
NULL = []          # 每次打乱的全部 (m,h) 结果
print("\n" + "=" * 112)
print(f"零假设：打乱序列 × {NSHUF} 次，每次跑完整流程（独立重估参数）")
print("=" * 112)
for it in range(NSHUF):
    rng = np.random.default_rng(SEED0 + it)
    rec = []
    for m in MK:
        lp, t0 = DATA[m]
        inc = np.diff(lp)
        lp_s = np.concatenate([[0.0], np.cumsum(inc[rng.permutation(len(inc))])])
        r = pipeline(lp_s, t0)
        if r:
            for x in r:
                x = dict(x); x["m"] = m; x["iter"] = it
                rec.append(x)
    NULL.append(rec)
    if (it + 1) % 10 == 0:
        allr = [x for r in NULL for x in r]
        print(f"  第 {it+1:>3} 次：累计中位比 {np.median([x['ratio'] for x in allr]):.4f}  "
              f"显著胜率 {np.mean([x['win'] for x in allr]):.4f}  "
              f"（{time.time()-T0:.0f} 秒）")

flat = [x for r in NULL for x in r]
null_ratios = np.array([x["ratio"] for x in flat])
null_p = np.array([x["p"] for x in flat])
null_winfreq = np.mean([x["win"] for x in flat])

print("\n" + "=" * 112)
print("M2  DM 检验的校准性：零假设下 p 值应服从 Uniform(0,1)")
print("=" * 112)
ok = null_p[~np.isnan(null_p)]
ks = kstest(ok, "uniform")
print(f"  n = {len(ok)}   KS 统计量 {ks.statistic:.4f}   p = {ks.pvalue:.4f}")
print(f"  p<0.05 的比例 {np.mean(ok<0.05):.4f}（名义 0.05）   p<0.01 的比例 {np.mean(ok<0.01):.4f}（名义 0.01）")
print(f"  零假设下\"显著胜\"的发生频率 {null_winfreq:.4f}   \"显著负\" {np.mean([x['loss'] for x in flat]):.4f}")

print("\n" + "=" * 112)
print("M3  效应量：真实数据 vs 零分布")
print("=" * 112)
# 每次打乱给出 15 格的中位比与显著胜数
per_iter_ratio = np.array([np.median([x["ratio"] for x in r]) for r in NULL])
per_iter_wins = np.array([sum(x["win"] for x in r) for r in NULL])
per_iter_losses = np.array([sum(x["loss"] for x in r) for r in NULL])
real_all = [x for m in MK for x in REAL[m]]
real_ratio = float(np.median([x["ratio"] for x in real_all]))
real_wins = int(sum(x["win"] for x in real_all))
real_losses = int(sum(x["loss"] for x in real_all))
print(f"  零分布（{NSHUF} 次打乱，每次 15 格）：")
print(f"    中位比：均值 {per_iter_ratio.mean():.4f}  标准差 {per_iter_ratio.std():.4f}  "
      f"2.5% {np.percentile(per_iter_ratio,2.5):.4f}  50% {np.percentile(per_iter_ratio,50):.4f}  "
      f"97.5% {np.percentile(per_iter_ratio,97.5):.4f}")
print(f"    显著胜数：均值 {per_iter_wins.mean():.2f}  95% 分位 {np.percentile(per_iter_wins,95):.1f}  "
      f"最大值 {per_iter_wins.max():.0f}")
print(f"    显著负数：均值 {per_iter_losses.mean():.2f}  95% 分位 {np.percentile(per_iter_losses,95):.1f}  "
      f"最大值 {per_iter_losses.max():.0f}")
print(f"  真实数据：")
print(f"    中位比 {real_ratio:.4f}   显著胜 {real_wins}/15   显著负 {real_losses}/15")
p_wins = float(np.mean(per_iter_wins >= real_wins))
p_ratio = float(np.mean(per_iter_ratio <= real_ratio))
print(f"\n  >>> 单尾 p 值（打乱分布的经验上尾）：")
print(f"      显著胜数 >= 真实值：p = {p_wins:.4f}   {'✓ 超出零分布' if p_wins < 0.05 else '✗ 未超出'}")
print(f"      中位比  <= 真实值：p = {p_ratio:.4f}   {'✓ 超出零分布' if p_ratio < 0.05 else '✗ 未超出'}")

print("\n" + "=" * 112)
print("M4  逐 (市场, 期限) 的零分布：哪些格子的'显著'最可能是流程产物")
print("=" * 112)
print(f"{'市场':<8}{'h':>5}{'真实比':>9}{'真实p':>8}{'零分布中位':>12}{'零分布p的5%分位':>15}{'判定':>22}")
CELL = {}
for m in MK:
    for x in REAL[m]:
        sub = np.array([y["p"] for y in flat if y["m"] == m and y["h"] == x["h"]])
        sub = sub[~np.isnan(sub)]
        rr = np.array([y["ratio"] for y in flat if y["m"] == m and y["h"] == x["h"]])
        q05 = float(np.percentile(sub, 5))
        ptail = float(np.mean(sub <= x["p"])) if x["p"] == x["p"] else np.nan
        verdict = ("真实值比零分布 5% 分位还小 ✓" if x["p"] < q05 else
                   ("真实值优于零分布中位" if x["p"] < np.median(sub) else "**与零分布不可区分**"))
        print(f"{MN[m]:<8}{x['h']:>5}{x['ratio']:>9.4f}{x['p']:>8.3f}{np.median(rr):>12.4f}"
              f"{q05:>15.3f}{verdict:>22}")
        CELL[f"{m}|{x['h']}"] = dict(real_ratio=x["ratio"], real_p=x["p"],
                                    null_med=float(np.median(rr)), null_p05=q05,
                                    p_tail=ptail, real_win=x["win"], real_loss=x["loss"])

print("\n" + "=" * 112)
print("M5  逐市场：CEA 的 2/5 显著胜在零分布里常见吗？")
print("=" * 112)
for m in MK:
    sub = np.array([sum(x["win"] for x in r if x["m"] == m) for r in NULL])
    obs = sum(x["win"] for x in REAL[m])
    print(f"  {MN[m]:<6} 真实显著胜 {obs}/{len(REAL[m])}   零分布：均值 {sub.mean():.2f}  "
          f"95% 分位 {np.percentile(sub,95):.1f}  最大 {sub.max():.0f}   "
          f"p(零分布 >= 真实) = {np.mean(sub>=obs):.3f}")

json.dump({"real": {m: REAL[m] for m in MK},
           "null_per_iter_ratio": per_iter_ratio.tolist(),
           "null_per_iter_wins": per_iter_wins.tolist(),
           "null_per_iter_losses": per_iter_losses.tolist(),
           "null_ratio_dist": {"mean": float(null_ratios.mean()), "sd": float(null_ratios.std()),
                               "p2.5": float(np.percentile(null_ratios, 2.5)),
                               "p50": float(np.percentile(null_ratios, 50)),
                               "p97.5": float(np.percentile(null_ratios, 97.5))},
           "calibration": {"n": int(len(ok)), "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
                           "frac_p05": float(np.mean(ok < .05)), "frac_p01": float(np.mean(ok < .01)),
                           "null_win_freq": float(null_winfreq)},
           "verdict": {"real_ratio": real_ratio, "real_wins": real_wins, "real_losses": real_losses,
                       "p_wins": p_wins, "p_ratio": p_ratio},
           "cells": CELL},
          open(os.path.join(DIR, "MC_PLACEBO.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 MC_PLACEBO.json")
