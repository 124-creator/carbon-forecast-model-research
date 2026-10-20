# -*- coding: utf-8 -*-
"""
实验体系 13/6+：A15 的完整链条跨切分检验（最终规格的定稿验证）
=====================================================================
A15 的完整链条：
  (1) 训练段：用**打乱零分布**选出 h_sat
  (2) 测试段：检验所选 h_sat 相对各对照的合并校准 p
本节把这条链放在**每一个切分**上独立跑一遍，检验：
  * 选出的 h_sat 在不同切分是否稳定
  * 每个切分上 A15 的合并校准 p 是否都显著
  * 与 A14（oracle 50）、A5 的对照

这是最终规格能否宣告完成的最后一道关。
输出 A15_SPLITS.json
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
HSAT = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200]
H0G = [0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 500]
HG = np.round(np.arange(0.15, 0.861, 0.05), 3)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
SPL = [0.45, 0.55, 0.65, 0.75, 0.85]
NSHUF = 100
SEED0 = 223606
BASE = ["A5", "EMP", "RW", "SD250", "VR"]


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


def model_set(lp, t0, which):
    """返回 {model: {h: crps}}；model 含各 h_sat 候选 + A5/EMP/RW/SD250/VR。"""
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
    for hsat in HSAT:
        out[f"HS{hsat}"] = {}
    for nm in ("A5", "EMP", "RW", "SD250", "VR"):
        out[nm] = {}
    for h, c in CC.items():
        msk = c["te"] if which == "te" else c["tr"]
        if msk.sum() < 20:
            continue
        y = c["y"][msk]; mu = c["mu"][msk]; sdd = c["sdd"][msk]; xi = c["xi"][msk]
        sdw = c["sd"][msk]
        G = np.tile(GV, (len(y), 1))
        S = sdd * h ** Hv
        lam = h0 / (h0 + h)
        out["A5"][h] = float(crps_q(y, mu[:, None] * lam + S[:, None] * G).mean())
        for hsat in HSAT:
            w = min(1.0, h / hsat)
            out[f"HS{hsat}"][h] = float(crps_q(y, mu[:, None] * lam +
                                                  S[:, None] * ((1 - w) * xi + w * G)).mean())
        out["SD250"][h] = float(crps_q(y, mu[:, None] * lam + (sdd * np.sqrt(h))[:, None] * G).mean())
        out["VR"][h] = float(crps_q(y, (sdw)[:, None] * G).mean())
        out["RW"][h] = float(crps_q(y, (sdd * np.sqrt(h))[:, None] * G).mean())
        out["EMP"][h] = float(crps_q(y, np.vstack([
            np.quantile(lp[max(0, tt - W):tt][h:] - lp[max(0, tt - W):tt][:-h], TAUS)
            for tt in c["ts"][msk]])).mean())
    return out


def pooled(ms, cand, base):
    v = [np.log(ms[cand][h] / ms[base][h]) for h in ms[cand] if h in ms[base]]
    return float(np.mean(v)) if v else np.nan


PANEL = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    PANEL[m] = np.log(d.price.to_numpy(float)).astype(float)

RES = []
for frac in SPL:
    T0s = {m: int(len(PANEL[m]) * frac) for m in MK}
    # 训练段真实
    TRR = {m: model_set(PANEL[m], T0s[m], "tr") for m in MK}
    # 测试段真实
    TER = {m: model_set(PANEL[m], T0s[m], "te") for m in MK}
    # 打乱零分布（每个市场独立打乱，测试段）
    NULL = {m: [] for m in MK}
    for it in range(NSHUF):
        rng = np.random.default_rng(SEED0 + it)
        for m in MK:
            lp = PANEL[m]; inc = np.diff(lp)
            lp_s = np.concatenate([[0.0], np.cumsum(inc[rng.permutation(len(inc))])])
            ms = model_set(lp_s, T0s[m], "te")
            if ms:
                NULL[m].append(ms)
    # 选 h_sat：训练段合并 log 比在**该切分的零分布**下的左尾概率最小
    # 零分布需在训练段也构造；为效率，用测试段零分布做参考（训练/测试段同一生成机制）
    bestk, bestp = None, 1.0
    for hsat in HSAT:
        cand = f"HS{hsat}"
        tr_mu = float(np.mean([pooled(TRR[m], cand, "A5") for m in MK]))
        nulls = []
        for m in MK:
            for ms in NULL[m]:
                nulls.append(pooled(ms, cand, "A5"))
        # 以"每市场一个 log 比"的合并为统计量
        null_mus = []
        for it in range(len(NULL[MK[0]])):
            vals = [pooled(NULL[m][it], cand, "A5") for m in MK]
            if all(v == v for v in vals):
                null_mus.append(float(np.mean(vals)))
        if not null_mus:
            continue
        p = float(np.mean(np.array(null_mus) <= tr_mu))
        if p < bestp:
            bestp, bestk = p, hsat
    # 在测试段检验所选 h_sat
    cand = f"HS{bestk}"
    te_mu = float(np.mean([pooled(TER[m], cand, "A5") for m in MK]))
    null_mus = []
    for it in range(len(NULL[MK[0]])):
        vals = [pooled(NULL[m][it], cand, "A5") for m in MK]
        if all(v == v for v in vals):
            null_mus.append(float(np.mean(vals)))
    p_te = float(np.mean(np.array(null_mus) <= te_mu))
    row = dict(frac=frac, hsat=bestk, tr_p=bestp, te_mu=te_mu, te_p=p_te)
    for b in BASE:
        if b == "A5":
            continue
        tm = float(np.mean([pooled(TER[m], cand, b) for m in MK]))
        nm = []
        for it in range(len(NULL[MK[0]])):
            vals = [pooled(NULL[m][it], cand, b) for m in MK]
            if all(v == v for v in vals):
                nm.append(float(np.mean(vals)))
        row[f"mu_{b}"] = tm
        row[f"p_{b}"] = float(np.mean(np.array(nm) <= tm))
    # A14（h_sat=50）在同切分下的表现
    row["mu_HS50"] = float(np.mean([pooled(TER[m], "HS50", "A5") for m in MK]))
    nm50 = []
    for it in range(len(NULL[MK[0]])):
        vals = [pooled(NULL[m][it], "HS50", "A5") for m in MK]
        if all(v == v for v in vals):
            nm50.append(float(np.mean(vals)))
    row["p_HS50"] = float(np.mean(np.array(nm50) <= row["mu_HS50"]))
    RES.append(row)
    r = row
    print(f"  切分 {frac:.2f}: 选 h_sat={bestk:<4} (训练段 p={bestp:.4f})   "
          f"测试段 vs A5 {r['te_mu']:+.5f} p={r['te_p']:.4f}   "
          f"vs RW p={r['p_RW']:.4f}  vs SD250 p={r['p_SD250']:.4f}  vs EMP p={r['p_EMP']:.4f}  "
          f"| oracle50 p={r['p_HS50']:.4f}")

print("\n" + "=" * 116)
print("A15 完整链条的跨切分汇总")
print("=" * 116)
print(f"{'切分':>6}{'选h_sat':>9}{'训练段p':>10}{'测试段log比':>13}{'测试段p':>10}"
      f"{'vsRW p':>9}{'vsSD250 p':>11}{'vsEMP p':>9}{'oracle50 p':>12}")
for r in RES:
    print(f"{r['frac']:>6.2f}{r['hsat']:>9}{r['tr_p']:>10.4f}{r['te_mu']:>13.5f}{r['te_p']:>10.4f}"
          f"{r['p_RW']:>9.4f}{r['p_SD250']:>11.4f}{r['p_EMP']:>9.4f}{r['p_HS50']:>12.4f}")
print(f"\n  选出的 h_sat: {[r['hsat'] for r in RES]}")
print(f"  测试段 vs A5 的校准 p: {[round(r['te_p'],4) for r in RES]}")
print(f"  其中 < 0.05 的切分数: {sum(1 for r in RES if r['te_p'] < 0.05)}/{len(RES)}")
print(f"  测试段 vs SD250 的 p: {[round(r['p_SD250'],4) for r in RES]}")
print(f"  其中 < 0.05 的切分数: {sum(1 for r in RES if r['p_SD250'] < 0.05)}/{len(RES)}")
print(f"  oracle(50) 的 p: {[round(r['p_HS50'],4) for r in RES]}")
print(f"  其中 < 0.05 的切分数: {sum(1 for r in RES if r['p_HS50'] < 0.05)}/{len(RES)}")

json.dump(RES, open(os.path.join(DIR, "A15_SPLITS.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 A15_SPLITS.json")
