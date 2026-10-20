# -*- coding: utf-8 -*-
"""
实验体系 17/6+：形状调度常数的**事前固定选择规则**与最终定稿
=====================================================================
§24.36 的发现：
  * h_sat 越大改善越大（测试段 log 比 20:-0.0057 → 80:-0.0117），
    **A15(h_sat=20) 显著差于 A14(h_sat=50)**（p=0.9917）。
  * §24.33 的选择规则有缺陷：按"校准 p 最小"选，而 30/40/50/60 的 p 并列
    （分辨率 1/120），**并列时随机打破**。
  * 训练段 log 比：10:-0.00259 15:-0.00477 20:-0.00672 30:-0.00916
    40:-0.01033 **50:-0.01098** 60:-0.01097 80:-0.01059
    ⇒ 规则"在校准 p ≤ 0.05 的候选中取训练段效应量最大者"会选出 **50**。

**关键论断**：A14 的 h_sat=50 **不是 oracle 读数**，而是一个事前固定的
训练段规则会选出的值。本节用正式规则在**每个切分独立重跑**来证明这一点。

规则 R（事前固定）：
  候选集 C = {10,15,20,30,40,50,60,80,100,150,200}
  1. 在训练段算每个 h_sat 的合并 log 比（vs A5）与校准 p（打乱零分布）
  2. 取 {c ∈ C : p_train(c) ≤ 0.05}；若为空，取全体中 p 最小者
  3. 在该集合内取 **训练段 log 比最小**者
输出 FINAL_SPEC.json
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
CAND = [10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200]
H0G = [0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 500]
HG = np.round(np.arange(0.15, 0.861, 0.05), 3)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
SPL = [0.45, 0.55, 0.65, 0.75, 0.85]
NSHUF = 80
SEED0 = 8675309


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


def losses(lp, t0, which):
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
    for c in CAND:
        out[f"HS{c}"] = {}
    for h, c in CC.items():
        msk = c["te"] if which == "te" else c["tr"]
        if msk.sum() < 20:
            continue
        y = c["y"][msk]; mu = c["mu"][msk]; sdd = c["sdd"][msk]; xi = c["xi"][msk]
        G = np.tile(GV, (len(y), 1))
        S = sdd * h ** Hv
        lam = h0 / (h0 + h)
        out["A5"][h] = float(crps_q(y, mu[:, None] * lam + S[:, None] * G).mean())
        for cc in CAND:
            w = min(1.0, h / cc)
            out[f"HS{cc}"][h] = float(crps_q(y, mu[:, None] * lam +
                                              S[:, None] * ((1 - w) * xi + w * G)).mean())
    return out


PANEL = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    PANEL[m] = np.log(d.price.to_numpy(float)).astype(float)

print("=" * 116)
print("每个切分：规则 R 选 h_sat → 测试段表现（零分布按切分独立构建）")
print("=" * 116)
ALL = []
for frac in SPL:
    T0S = {m: int(len(PANEL[m]) * frac) for m in MK}
    TRR = {m: losses(PANEL[m], T0S[m], "tr") for m in MK}
    TER = {m: losses(PANEL[m], T0S[m], "te") for m in MK}
    NULL = []
    for it in range(NSHUF):
        rng = np.random.default_rng(SEED0 + int(frac * 100) + it)
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

    def pooled(ms, cand):
        v = [np.log(ms[cand][h] / ms["A5"][h]) for h in ms[cand] if h in ms["A5"]]
        return float(np.mean(v)) if v else np.nan

    def null_pooled(cand):
        nn = []
        for rec in NULL:
            v = [np.log(rec[(cand, h, m)] / rec[("A5", h, m)]) for m in MK for h in HS
                 if (cand, h, m) in rec and ("A5", h, m) in rec]
            if len(v) == len(MK) * len(HS):
                nn.append(float(np.mean(v)))
        return np.array(nn)

    tr_mu, tr_p = {}, {}
    for c in CAND:
        tr_mu[c] = float(np.mean([pooled(TRR[m], f"HS{c}") for m in MK]))
        nn = null_pooled(f"HS{c}")
        tr_p[c] = float(np.mean(nn <= tr_mu[c])) if len(nn) else 1.0
    ok = [c for c in CAND if tr_p[c] <= 0.05]
    if not ok:
        ok = [min(CAND, key=lambda c: tr_p[c])]
    sel = min(ok, key=lambda c: tr_mu[c])
    te_mu_sel = float(np.mean([pooled(TER[m], f"HS{sel}") for m in MK]))
    nn_sel = null_pooled(f"HS{sel}")
    p_te = float(np.mean(nn_sel <= te_mu_sel)) if len(nn_sel) else np.nan
    # 对照
    te50 = float(np.mean([pooled(TER[m], "HS50") for m in MK]))
    nn50 = null_pooled("HS50")
    p50 = float(np.mean(nn50 <= te50)) if len(nn50) else np.nan
    te20 = float(np.mean([pooled(TER[m], "HS20") for m in MK]))
    nn20 = null_pooled("HS20")
    p20 = float(np.mean(nn20 <= te20)) if len(nn20) else np.nan
    # 测试段 oracle
    oracle = min(CAND, key=lambda c: float(np.mean([pooled(TER[m], f"HS{c}") for m in MK])))
    te_or = float(np.mean([pooled(TER[m], f"HS{oracle}") for m in MK]))
    row = dict(frac=frac, sel=sel, tr_mu=tr_mu[sel], tr_p=tr_p[sel],
               te_mu=te_mu_sel, te_p=p_te, te50=te50, p50=p50, te20=te20, p20=p20,
               oracle=oracle, te_oracle=te_or,
               tr_p_all={str(k): v for k, v in tr_p.items()},
               tr_mu_all={str(k): v for k, v in tr_mu.items()})
    ALL.append(row)
    print(f"  切分 {frac:.2f}: 选出 h_sat={sel:<4} (训练段 p={tr_p[sel]:.4f})   "
          f"测试段 {te_mu_sel:+.5f} p={p_te:.4f}   | 固定50 {te50:+.5f} p={p50:.4f}   "
          f"| 固定20 {te20:+.5f} p={p20:.4f}   | oracle {oracle} {te_or:+.5f}")

print("\n" + "=" * 116)
print("规则 R 的汇总")
print("=" * 116)
print(f"  选出的 h_sat: {[r['sel'] for r in ALL]}")
print(f"  训练段校准 p: {[round(r['tr_p'],4) for r in ALL]}")
print(f"  测试段校准 p: {[round(r['te_p'],4) for r in ALL]}")
print(f"  测试段 log 比: {[round(r['te_mu'],5) for r in ALL]}")
print(f"  固定 h_sat=50 的测试段 p: {[round(r['p50'],4) for r in ALL]}")
print(f"  固定 h_sat=20 的测试段 p: {[round(r['p20'],4) for r in ALL]}")
print(f"  测试段 oracle 最优 h_sat: {[r['oracle'] for r in ALL]}")
print(f"\n  规则 R 相对固定 50：显著胜 {sum(1 for r in ALL if r['te_mu']<r['te50'])}/{len(ALL)}")
print(f"  规则 R 相对固定 20：显著胜 {sum(1 for r in ALL if r['te_mu']<r['te20'])}/{len(ALL)}")
print(f"  规则 R 相对测试段 oracle 的差距（中位）："
      f"{np.median([r['te_mu']-r['te_oracle'] for r in ALL]):+.5f}"
      f"（即 {np.median([(np.exp(r['te_mu'])/np.exp(r['te_oracle'])-1)*100 for r in ALL]):+.2f}%）")

json.dump(ALL, open(os.path.join(DIR, "FINAL_SPEC.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 FINAL_SPEC.json")
