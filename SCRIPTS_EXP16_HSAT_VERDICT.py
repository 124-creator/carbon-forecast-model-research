# -*- coding: utf-8 -*-
"""
实验体系 16/6+：A15 vs A14 的正面对决（h_sat 的选择检验）
=====================================================================
§24.35 发现的关键事实：
  A15 (h_sat=20) 的形状订正只作用于 h<20；h≥20 时与 A5 **逐位相同**。
  A14 (h_sat=50) 作用于 h<50。
  ⇒ h_sat **不是无关紧要的小参数**，它决定形状订正在哪些期限生效。
  ⇒ A15 vs A14 的比较必须覆盖**全部期限**，不能只挑有利的 h。

本节做决定性比较（正确零假设，200 次打乱）：
  1) A15 vs A14 的合并校准检验（全部 16 个期限）
  2) 分期限的 A15 vs A14
  3) 逐市场
  4) **同时检验 h_sat 本身**：把 h_sat 当成一个超参数，
     看"在训练段用校准零分布选"这个程序 vs "oracle 读数"哪个更好。
  5) 最重要的：**在训练段选出 h_sat 之后，在测试段的整体表现**，
     与"在测试段选最优 h_sat"（oracle 上界）比较 —— 量化选择代价。
输出 HSAT_VERDICT.json
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
HS = [5, 10, 15, 20, 30, 40, 60, 120]
HSAT = [10, 15, 20, 30, 40, 50, 60, 80]
H0G = [0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 500]
HG = np.round(np.arange(0.15, 0.861, 0.05), 3)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
NSHUF = 120
SEED0 = 424242


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
    out = {"A5": {}}
    for hsat in HSAT:
        out[f"HS{hsat}"] = {}
    for h, c in CC.items():
        msk = c["te"] if which == "te" else c["tr"]
        if msk.sum() < 20:
            continue
        y = c["y"][msk]; mu = c["mu"][msk]; sdd = c["sdd"][msk]; xi = c["xi"][msk]
        G = np.tile(GV, (len(y), 1))
        S = sdd * h ** Hv
        lam = h0 / (h0 + h)
        out["A5"][h] = float(crps_q(y, mu[:, None] * lam + S[:, None] * G).mean())
        for hsat in HSAT:
            w = min(1.0, h / hsat)
            out[f"HS{hsat}"][h] = float(crps_q(y, mu[:, None] * lam +
                                                  S[:, None] * ((1 - w) * xi + w * G)).mean())
    return out


PANEL = {}
T0S = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    PANEL[m] = np.log(d.price.to_numpy(float)).astype(float)
    T0S[m] = int(np.searchsorted(pd.to_datetime(d.date).to_numpy(), np.datetime64(CUT[m])))

REAL = {m: losses(PANEL[m], T0S[m], "te") for m in MK}
TRR = {m: losses(PANEL[m], T0S[m], "tr") for m in MK}
print("=" * 112)
print("真实数据：各 h_sat 的合并 log 比（vs A5），测试段 / 训练段")
print("=" * 112)
print(f"{'h_sat':>8}{'测试段合并':>12}{'训练段合并':>12}")
for hsat in HSAT:
    te = np.mean([np.log(REAL[m][f"HS{hsat}"][h] / REAL[m]["A5"][h]) for m in MK for h in HS])
    tr = np.mean([np.log(TRR[m][f"HS{hsat}"][h] / TRR[m]["A5"][h]) for m in MK for h in HS])
    print(f"{hsat:>8}{te:>12.5f}{tr:>12.5f}")

print("\n" + "=" * 112)
print(f"打乱零分布（{NSHUF} 次，完整流程）")
print("=" * 112)
T0 = time.time()
NULL = []
for it in range(NSHUF):
    rng = np.random.default_rng(SEED0 + it)
    rec = {}
    for m in MK:
        lp = PANEL[m]; inc = np.diff(lp)
        lp_s = np.concatenate([[0.0], np.cumsum(inc[rng.permutation(len(inc))])])
        L = losses(lp_s, T0S[m], "te")
        if L:
            for nm in L:
                for h in L[nm]:
                    rec[(nm, h, m)] = L[nm][h]
    NULL.append(rec)
    if (it + 1) % 40 == 0:
        print(f"  第 {it+1:>4} 次  ({time.time()-T0:.0f} 秒)")


def pooled_test(cand, base, hset=HS):
    real = [np.log(REAL[m][cand][h] / REAL[m][base][h]) for m in MK for h in hset]
    mu_real = float(np.mean(real))
    nulls = []
    for rec in NULL:
        vals = [np.log(rec[(cand, h, m)] / rec[(base, h, m)]) for m in MK for h in hset
                if (cand, h, m) in rec and (base, h, m) in rec]
        if len(vals) == len(real):
            nulls.append(float(np.mean(vals)))
    nulls = np.array(nulls)
    return mu_real, float(np.mean(nulls <= mu_real)), float(np.mean(nulls < mu_real) * 100), \
        float(nulls.mean()), float(nulls.std())


print("\n" + "=" * 112)
print("一、A15 (h_sat=20) vs A14 (h_sat=50)：正面对决")
print("=" * 112)
mu, p, pct, nm_, ns = pooled_test("HS20", "HS50")
print(f"  全期限合并：真实 log 比 {mu:+.5f}   零分布 {nm_:+.5f}±{ns:.5f}   p = {p:.4f}   百分位 {pct:.1f}%")
print("  分期限：")
for h in HS:
    real = [np.log(REAL[m]["HS20"][h] / REAL[m]["HS50"][h]) for m in MK]
    mu_r = float(np.mean(real))
    nulls = []
    for rec in NULL:
        vals = [np.log(rec[("HS20", h, m)] / rec[("HS50", h, m)]) for m in MK
                if ("HS20", h, m) in rec and ("HS50", h, m) in rec]
        if len(vals) == len(real):
            nulls.append(float(np.mean(vals)))
    nulls = np.array(nulls)
    print(f"    h={h:<5} 真实 {mu_r:+.5f}   p = {float(np.mean(nulls<=mu_r)):.4f}")
print("  逐市场：")
for m in MK:
    real = [np.log(REAL[m]["HS20"][h] / REAL[m]["HS50"][h]) for h in HS]
    mu_r = float(np.mean(real))
    nulls = []
    for rec in NULL:
        vals = [np.log(rec[("HS20", h, m)] / rec[("HS50", h, m)]) for h in HS
                if ("HS20", h, m) in rec and ("HS50", h, m) in rec]
        if len(vals) == len(real):
            nulls.append(float(np.mean(vals)))
    nulls = np.array(nulls)
    print(f"    {MN[m]:<6} 真实 {mu_r:+.5f}   p = {float(np.mean(nulls<=mu_r)):.4f}")

print("\n" + "=" * 112)
print("二、各 h_sat 相对 A5 的合并检验（全部 8 个期限）")
print("=" * 112)
print(f"{'h_sat':>8}{'真实log比':>12}{'零分布均值':>13}{'单尾p':>9}{'百分位':>9}")
for hsat in HSAT:
    mu, p, pct, nm_, ns = pooled_test(f"HS{hsat}", "A5")
    print(f"{hsat:>8}{mu:>12.5f}{nm_:>13.5f}{p:>9.4f}{pct:>8.1f}%")

print("\n" + "=" * 112)
print("三、选择代价：训练段选出的 h_sat vs 测试段 oracle 最优 h_sat")
print("=" * 112)
# 训练段选：训练段校准 p 最小
null_by = {}
for hsat in HSAT:
    nn = []
    for rec in NULL:
        vals = [np.log(rec[(f"HS{hsat}", h, m)] / rec[("A5", h, m)]) for m in MK for h in HS
                if (f"HS{hsat}", h, m) in rec and ("A5", h, m) in rec]
        if vals:
            nn.append(float(np.mean(vals)))
    null_by[hsat] = np.array(nn)
tr_mu = {hsat: float(np.mean([np.log(TRR[m][f"HS{hsat}"][h] / TRR[m]["A5"][h]) for m in MK for h in HS]))
         for hsat in HSAT}
tr_p = {hsat: float(np.mean(null_by[hsat] <= tr_mu[hsat])) for hsat in HSAT}
sel = min(HSAT, key=lambda k: tr_p[k])
te_mu = {hsat: float(np.mean([np.log(REAL[m][f"HS{hsat}"][h] / REAL[m]["A5"][h]) for m in MK for h in HS]))
         for hsat in HSAT}
oracle = min(HSAT, key=lambda k: te_mu[k])
print(f"  训练段校准 p: " + "  ".join(f"{k}:{tr_p[k]:.4f}" for k in HSAT))
print(f"  >>> 训练段选出 h_sat = {sel}")
print(f"  >>> 测试段 oracle 最优 h_sat = {oracle}（测试段 log 比 {te_mu[oracle]:+.5f}）")
print(f"  >>> 训练段选出的 {sel} 在测试段的 log 比 = {te_mu[sel]:+.5f}")
print(f"  >>> 选择代价 = {te_mu[sel]-te_mu[oracle]:+.5f}（即 {(te_mu[sel]/te_mu[oracle]-1)*100:+.2f}% 的 CRPS 差距）")

print("\n" + "=" * 112)
print("四、A15 (h_sat=20) 与 A14 (h_sat=50) 相对 A5 的对比")
print("=" * 112)
for nm_, cand in (("A14 (h_sat=50)", "HS50"), ("A15 (h_sat=20)", "HS20")):
    mu, p, pct, nmm, ns = pooled_test(cand, "A5")
    print(f"  {nm_:<18} 真实 {mu:+.5f}  零分布 {nmm:+.5f}±{ns:.5f}  p = {p:.4f}  百分位 {pct:.1f}%")

json.dump({"sel_hsat": sel, "oracle_hsat": oracle,
           "tr_p": {str(k): v for k, v in tr_p.items()},
           "te_mu": {str(k): v for k, v in te_mu.items()},
           "tr_mu": {str(k): v for k, v in tr_mu.items()},
           "real": {m: {nm2: {str(h): REAL[m][nm2][h] for h in REAL[m][nm2]} for nm2 in REAL[m]} for m in MK}},
          open(os.path.join(DIR, "HSAT_VERDICT.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 HSAT_VERDICT.json")
