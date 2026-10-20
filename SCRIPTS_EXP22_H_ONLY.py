# -*- coding: utf-8 -*-
"""
实验体系 22/6+：把 H 机制单独拎出来，做**最小可辩护模型**的定稿检验
=====================================================================
§24.42 的 GARCH_H 对照说明：位置块与形状块的增量无法与噪声区分。
⇒ 必须检验：**只保留 H 机制**（去掉位置块、去掉形状块）的模型表现如何？
   * 若"只含 H"的模型本身就优于固定 √h 标度并显著 ⇒ H 是可独立发表的贡献。
   * 若"只含 H"（= A5）也显著 ⇒ 那 H 就是 A5，故事不成立。
   （注意：A5 = 位置块 + H 标度 + 高斯形状；"只含 H"应指"位置块 + H 标度"，
     即 A5 本身；真正的问题是：A5 相对 SD250 的显著性如何？）

本节做三件事：
  1) **A5 相对 SD250 的校准检验**（这才是 H 机制的直接证据，而不是 A14 vs SD250）
  2) **分解**：A14 = A5 + 形状块。分别检验 A5 vs SD250 与 A14 vs A5，
     看形状块的增量是否为零
  3) **最小模型族**：SD250 / A5 / A14 / GARCH_H 四者两两比较，全部用校准零假设
输出 H_ONLY.json
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
H0G = [0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 500]
HG = np.round(np.arange(0.15, 0.861, 0.05), 3)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
NSHUF = 120
SEED0 = 20260921
HSAT = 50.0
MODELS = ["SD250", "A5", "A14"]


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
    out = {m: {} for m in MODELS}
    for h, c in CC.items():
        msk = c["te"] if which == "te" else c["tr"]
        if msk.sum() < 20:
            continue
        y = c["y"][msk]; mu = c["mu"][msk]; sdd = c["sdd"][msk]; xi = c["xi"][msk]
        G = np.tile(GV, (len(y), 1))
        S = sdd * h ** Hv
        lam = h0 / (h0 + h)
        w = min(1.0, h / HSAT)
        out["SD250"][h] = float(crps_q(y, mu[:, None] * lam + (sdd * np.sqrt(h))[:, None] * G).mean())
        out["A5"][h] = float(crps_q(y, mu[:, None] * lam + S[:, None] * G).mean())
        out["A14"][h] = float(crps_q(y, mu[:, None] * lam +
                                      S[:, None] * ((1 - w) * xi + w * G)).mean())
    return out


PANEL, T0S = {}, {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    PANEL[m] = np.log(d.price.to_numpy(float)).astype(float)
    T0S[m] = int(np.searchsorted(pd.to_datetime(d.date).to_numpy(), np.datetime64(CUT[m])))

REAL = {m: losses(PANEL[m], T0S[m], "te") for m in MK}
print("=" * 112)
print("真实数据：三个模型的合并 log 比")
print("=" * 112)
for m in MK:
    row = f"  {MN[m]:<6}"
    for k in MODELS:
        v = np.mean([np.log(REAL[m][k][h] / REAL[m]["A5"][h]) for h in HS]) if k != "A5" else 0.0
        row += f"  {k} {v:+.5f}" if k != "A5" else "  (基准)"
    print(row)

print("\n" + "=" * 112)
print(f"零分布：打乱序列 × {NSHUF}")
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
            for k in MODELS:
                for h in L[k]:
                    rec[(k, h, m)] = L[k][h]
    NULL.append(rec)
    if (it + 1) % 40 == 0:
        print(f"  第 {it+1:>4} 次  ({time.time()-T0:.0f} 秒)")

print("\n" + "=" * 112)
print("两两校准比较（全部 3 个模型）")
print("=" * 112)
PAIRS = [("A5", "SD250"), ("A14", "A5"), ("A14", "SD250")]
OUT = {}
print(f"{'比较':<20}{'真实合并log比':>15}{'零分布均值':>13}{'零分布SD':>11}{'单尾p':>9}{'百分位':>9}")
for cand, base in PAIRS:
    real = [np.log(REAL[m][cand][h] / REAL[m][base][h]) for m in MK for h in HS]
    mu_real = float(np.mean(real))
    nulls = []
    for rec in NULL:
        vals = [np.log(rec[(cand, h, m)] / rec[(base, h, m)]) for m in MK for h in HS
                if (cand, h, m) in rec and (base, h, m) in rec]
        if len(vals) == len(real):
            nulls.append(float(np.mean(vals)))
    nulls = np.array(nulls)
    p1 = float(np.mean(nulls <= mu_real))
    pct = float(np.mean(nulls < mu_real) * 100)
    print(f"{cand + ' vs ' + base:<20}{mu_real:>15.5f}{nulls.mean():>13.5f}{nulls.std():>11.5f}"
          f"{p1:>9.4f}{pct:>8.1f}%")
    OUT[f"{cand}|{base}"] = dict(real_mu=mu_real, p=p1, pct=pct,
                                 null_mean=float(nulls.mean()), null_sd=float(nulls.std()))

print("\n逐市场（vs A5）：")
for m in MK:
    row = f"  {MN[m]:<6}"
    for cand in ("A5", "A14"):
        real = [np.log(REAL[m][cand][h] / REAL[m]["SD250"][h]) for m in [m] for h in HS]
        mu_real = float(np.mean(real))
        nulls = []
        for rec in NULL:
            vals = [np.log(rec[(cand, h, m)] / rec[("SD250", h, m)]) for h in HS
                    if (cand, h, m) in rec and ("SD250", h, m) in rec]
            if len(vals) == len(real):
                nulls.append(float(np.mean(vals)))
        nulls = np.array(nulls)
        row += f"  {cand}/SD250 {mu_real:+.5f} p={float(np.mean(nulls<=mu_real)):.3f}"
    print(row)

print("\n" + "=" * 112)
print("判定")
print("=" * 112)
p_h = OUT["A5|SD250"]["p"]
p_s = OUT["A14|A5"]["p"]
print(f"  H 机制（A5 vs 固定 √h 标度）：合并 log 比 {OUT['A5|SD250']['real_mu']:+.5f}，"
      f"校准 p = {p_h:.4f}  → {'H 机制稳健' if p_h < 0.05 else 'H 机制不稳健'}")
print(f"  形状块增量（A14 vs A5）：合并 log 比 {OUT['A14|A5']['real_mu']:+.5f}，"
      f"校准 p = {p_s:.4f}  → {'形状块有可检测增量' if p_s < 0.05 else '形状块增量不可检测'}")
VERDICT = ("H_ONLY" if p_h < 0.05 and p_s >= 0.05 else
           "BOTH" if p_h < 0.05 and p_s < 0.05 else
           "NEITHER" if p_h >= 0.05 else "SHAPE_ONLY")
print(f"  >>> 判定：{VERDICT}")
print(f"      （H_ONLY = 论文的可发表贡献是 H 机制本身，形状块应从主张中删除）")

json.dump({"out": OUT, "verdict": VERDICT, "real": REAL,
           "n_shuffle": NSHUF},
          open(os.path.join(DIR, "H_ONLY.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 H_ONLY.json")
