# -*- coding: utf-8 -*-
"""
实验体系 11/6+：用**校准后的零假设**重新推导形状调度常数 h_sat
=====================================================================
§24.27 的 A14 用 w(h)=min(1, h/50)，其中 **50 来自 oracle 剖面**（看测试段的最优 w
曲线读出来的），这是 A14 唯一的"非纯观测量"成分，也是它最容易被攻击的地方。

§24.32 建立了正确的零假设框架。现在可以用它做一件之前做不了的事：
**在训练段上、用"打乱零分布"为判据，重新推导 h_sat**，
使这个常数**不再是 oracle 读数**，而是一个有明确判据的估计量。

设计：
  对每个候选 h_sat ∈ {5,10,15,20,30,40,50,60,80,100,150,200,∞}：
    1) 在**训练段**上评估各 h_sat 的合并 log 比（vs A5）
    2) 在**打乱序列**上评估同一 h_sat 的合并 log 比，得到零分布
    3) 计算"校准 p 值" = P(零分布 ≤ 训练段观测值)
  选**训练段 p 值最小**的 h_sat，然后到**测试段**检验它是否仍然有效。
  这就是"用训练段选、超出参数、测试段验证"的完整链条。
  另做：**跨切分选 h_sat**（每个切分独立选，检验选出的值是否稳定）。

关键：**如果训练段选出的 h_sat 与 oracle 的 50 接近且在测试段有效，
那么 A14 的最后一个弱点就消除了。**
输出 HSAT_SELECT.json
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
NSHUF = 120
SEED0 = 161803


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


def fit_par(CC):
    """在给定格集合上拟合 (h0,H)（用 A5 形状）。"""
    best = (1e9, None, None)
    for Hval in HG:
        for h0 in H0G:
            v = float(np.mean([crps_q(c["y"][c["tr"]],
                                     (c["mu"][c["tr"]] * (h0 / (h0 + h)) +
                                      (c["sdd"][c["tr"]] * h ** Hval))[:, None] * GV).mean()
                                for h, c in CC.items()]))
            if v < best[0]:
                best = (v, h0, Hval)
    return best[1], best[2]


def losses(lp, t0, which="te"):
    """返回 {hsat: {h: (l5, l14)}}；which='tr' 用训练段。"""
    CC = {h: c for h, c in ((h, build(lp, h, t0)) for h in HS) if c is not None and c["tr"].sum() >= 30}
    if not CC:
        return None, None
    h0, Hv = fit_par(CC)
    out = {}
    for hsat in HSAT:
        d = {}
        for h, c in CC.items():
            msk = c["te"] if which == "te" else c["tr"]
            if msk.sum() < 20:
                continue
            y = c["y"][msk]; mu = c["mu"][msk]; sdd = c["sdd"][msk]; xi = c["xi"][msk]
            w = 1.0 if hsat > 1e8 else min(1.0, h / hsat)
            G = np.tile(GV, (len(y), 1))
            S = sdd * h ** Hv
            l5 = crps_q(y, mu[:, None] * (h0 / (h0 + h)) + S[:, None] * G)
            l14 = crps_q(y, mu[:, None] * (h0 / (h0 + h)) + S[:, None] * ((1 - w) * xi + w * G))
            d[h] = (float(l5.mean()), float(l14.mean()))
        out[hsat] = d
    return out, (h0, Hv)


def pooled(out):
    """合并 log 比（跨 h 跨市场等权）。"""
    lr = [np.log(b / a) for hs in out for h, (a, b) in out[hs].items()]
    return float(np.mean(lr)) if lr else np.nan


DATA = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    DATA[m] = (lp, t0)

print("=" * 112)
print("一、测试段：各 h_sat 的合并 log 比（vs A5）")
print("=" * 112)
TE = {}
for m in MK:
    TE[m] = losses(*DATA[m], which="te")[0]
print(f"{'h_sat':>8}" + "".join(f"{MN[m]:>10}" for m in MK) + f"{'合并':>12}")
for hsat in HSAT:
    row = f"{('inf' if hsat>1e8 else int(hsat)):>8}"
    for m in MK:
        lr = [np.log(b / a) for h, (a, b) in TE[m][hsat].items()]
        row += f"{np.mean(lr):>10.5f}" if lr else f"{'-':>10}"
    allv = [np.log(b / a) for m in MK for h, (a, b) in TE[m][hsat].items()]
    row += f"{np.mean(allv):>12.5f}"
    print(row)

print("\n" + "=" * 112)
print("二、训练段：各 h_sat 的合并 log 比")
print("=" * 112)
TR = {}
for m in MK:
    TR[m] = losses(*DATA[m], which="tr")[0]
print(f"{'h_sat':>8}" + "".join(f"{MN[m]:>10}" for m in MK) + f"{'合并':>12}")
for hsat in HSAT:
    row = f"{('inf' if hsat>1e8 else int(hsat)):>8}"
    for m in MK:
        lr = [np.log(b / a) for h, (a, b) in TR[m][hsat].items()]
        row += f"{np.mean(lr):>10.5f}" if lr else f"{'-':>10}"
    allv = [np.log(b / a) for m in MK for h, (a, b) in TR[m][hsat].items()]
    row += f"{np.mean(allv):>12.5f}"
    print(row)

print("\n" + "=" * 112)
print(f"三、打乱零分布（{NSHUF} 次，每次跑完整流程，含全部候选 h_sat）")
print("=" * 112)
T0 = time.time()
NULL = {hsat: [] for hsat in HSAT}
for it in range(NSHUF):
    rng = np.random.default_rng(SEED0 + it)
    acc = {hsat: [] for hsat in HSAT}
    for m in MK:
        lp, t0 = DATA[m]
        inc = np.diff(lp)
        lp_s = np.concatenate([[0.0], np.cumsum(inc[rng.permutation(len(inc))])])
        L, _ = losses(lp_s, t0, which="te")
        if not L:
            continue
        for hsat in HSAT:
            for h, (a, b) in L[hsat].items():
                acc[hsat].append(np.log(b / a))
    for hsat in HSAT:
        if acc[hsat]:
            NULL[hsat].append(float(np.mean(acc[hsat])))
    if (it + 1) % 30 == 0:
        print(f"  第 {it+1:>4} 次  ({time.time()-T0:.0f} 秒)")

print("\n" + "=" * 112)
print("四、训练段选 h_sat（判据：训练段观测值在校准零分布中的左尾概率最小）")
print("=" * 112)
print(f"{'h_sat':>8}{'训练段合并':>12}{'零分布均值':>12}{'零分布SD':>11}{'训练段p':>10}"
      f"{'测试段合并':>12}{'测试段p':>10}")
SEL = {}
for hsat in HSAT:
    tr_all = [np.log(b / a) for m in MK for h, (a, b) in TR[m][hsat].items()]
    te_all = [np.log(b / a) for m in MK for h, (a, b) in TE[m][hsat].items()]
    tr_mu = float(np.mean(tr_all)); te_mu = float(np.mean(te_all))
    nz = np.array(NULL[hsat])
    p_tr = float(np.mean(nz <= tr_mu)); p_te = float(np.mean(nz <= te_mu))
    SEL[hsat] = dict(tr_mu=tr_mu, te_mu=te_mu, p_tr=p_tr, p_te=p_te,
                     null_mean=float(nz.mean()), null_sd=float(nz.std()))
    print(f"{('inf' if hsat>1e8 else int(hsat)):>8}{tr_mu:>12.5f}{nz.mean():>12.5f}"
          f"{nz.std():>11.5f}{p_tr:>10.4f}{te_mu:>12.5f}{p_te:>10.4f}")

best = min(SEL, key=lambda k: SEL[k]["p_tr"])
print(f"\n  >>> 训练段选出的 h_sat = {'inf' if best>1e8 else int(best)}"
      f"（训练段校准 p = {SEL[best]['p_tr']:.4f}）")
print(f"      该 h_sat 在**测试段**的校准 p = {SEL[best]['p_te']:.4f}，"
      f"合并 log 比 {SEL[best]['te_mu']:+.5f}")
print(f"      对照：oracle 读出的 h_sat=50 的测试段校准 p = {SEL[50]['p_te']:.4f}"
      f"（合并 log 比 {SEL[50]['te_mu']:+.5f}）")

print("\n" + "=" * 112)
print("五、跨切分选 h_sat 的稳定性（每个切分独立选）")
print("=" * 116)
SPL = [0.45, 0.55, 0.65, 0.75, 0.85]
SEL_SPL = {}
for frac in SPL:
    picks = []
    for m in MK:
        lp, _ = DATA[m]
        t0s = int(len(lp) * frac)
        tr_m = losses(lp, t0s, which="tr")[0]
        te_m = losses(lp, t0s, which="te")[0]
        if tr_m is None:
            continue
        # 用本切分训练段的合并 log 比（跨市场）在**全局零分布**中的左尾概率
        bestk, bestp = None, 1.0
        for hsat in HSAT:
            vals = [np.log(b / a) for mm in MK for h, (a, b) in tr_m[hsat].items()
                    if mm == m] if False else [np.log(b / a) for h, (a, b) in tr_m[hsat].items()]
            if not vals:
                continue
            mu = float(np.mean(vals))
            nz = np.array(NULL[hsat])
            p = float(np.mean(nz <= mu))
            if p < bestp:
                bestp, bestk = p, hsat
        picks.append((m, bestk, bestp))
    SEL_SPL[frac] = [(m, (None if k is None else ('inf' if k > 1e8 else int(k))), p)
                     for m, k, p in picks]
    print(f"  切分 {frac:.2f}: " + "  ".join(f"{MN[m]}→{k}(p={p:.3f})" for m, k, p in SEL_SPL[frac]))

json.dump({"sel": {str(k): v for k, v in SEL.items()},
           "best_hsat": (None if best > 1e8 else int(best)),
           "best_p_tr": SEL[best]["p_tr"], "best_p_te": SEL[best]["p_te"],
           "null": {str(k): v for k, v in NULL.items()},
           "split_sel": {str(k): v for k, v in SEL_SPL.items()},
           "te_by_hsat": {str(hsat): {m: {str(h): TE[m][hsat][h] for h in TE[m][hsat]} for m in MK}
                          for hsat in HSAT},
           "tr_by_hsat": {str(hsat): {m: {str(h): TR[m][hsat][h] for h in TR[m][hsat]} for m in MK}
                          for hsat in HSAT}},
          open(os.path.join(DIR, "HSAT_SELECT.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 HSAT_SELECT.json")
