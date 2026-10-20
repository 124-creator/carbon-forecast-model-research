# -*- coding: utf-8 -*-
"""
实验体系 18/6+：**用校准零假设重做安慰剂与数据口径结论**
=====================================================================
审计发现的问题：§24.31.4/§24.31.5 的"显著胜/负"计数全部来自**失准的渐近 DM**
（打乱数据上 p<0.05 比例 0.288）。25 格的期望假阳性是 **7.2 格**，
而"真实 ξ_d"只报 5 格显著胜——**低于零假设期望，因此该结论不成立**。

本节把安慰剂与数据口径全部改用**校准零假设**重做：
  每个变体有自己的零分布（变体改变了模型，零分布也必须相应改变）。

变体：
  V1 真实 ξ_d                （本文模型）
  V2 随机分位 ξ_d            （形状输入被随机化）
  V3 常数高斯                （无形状订正 ≡ A5）
  V4 滞后一期 ξ_d
  V5-V7 三个随机形状调度 w'(h)

每个变体报告：真实合并 log 比、零分布均值±SD、**校准 p**、零分布百分位。
**判据**：真实 ξ_d 的校准 p 应显著优于 V2/V3/V5-V7。
输出 PLACEBO_CALIB.json
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
H0G = [0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 500]
HG = np.round(np.arange(0.15, 0.861, 0.05), 3)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
NSHUF = 60
SEED0 = 97531
HSAT = 50.0
VARIANTS = ["real", "rand", "const", "lag", "rnd1", "rnd2", "rnd3"]
VMAP = {"real": "V1 真实 ξ_d", "rand": "V2 随机分位", "const": "V3 常数高斯",
        "lag": "V4 滞后一期 ξ_d", "rnd1": "V5 随机调度①", "rnd2": "V6 随机调度②",
        "rnd3": "V7 随机调度③"}


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


def eval_variants(lp, t0, seed):
    """在给定（可能打乱的）数据上评估全部变体，返回 {variant: {h: crps}}。"""
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
    rng = np.random.default_rng(seed)
    # 三个随机调度（固定随机数，真实与打乱用同一组）
    rnd_thr = {v: rng.random() * 300 + 5 for v in ("rnd1", "rnd2", "rnd3")}
    out = {v: {} for v in VARIANTS}
    for h, c in sorted(CC.items()):
        te = c["te"]
        if te.sum() < 20:
            continue
        y = c["y"][te]; mu = c["mu"][te]; sdd = c["sdd"][te]; xi = c["xi"][te]
        G = np.tile(GV, (len(y), 1))
        S = sdd * h ** Hv
        lam = h0 / (h0 + h)
        out["const"][h] = float(crps_q(y, mu[:, None] * lam + S[:, None] * G).mean())
        XI = {"real": xi,
              "rand": np.vstack([rng.permutation(GV) for _ in range(len(y))]),
              "const": G,
              "lag": np.vstack([xi[(i - 1) % len(xi)] for i in range(len(xi))])}
        for v in ("real", "rand", "const", "lag"):
            if v == "const":
                continue
            w = min(1.0, h / HSAT)
            out[v][h] = float(crps_q(y, mu[:, None] * lam +
                                    S[:, None] * ((1 - w) * XI[v] + w * G)).mean())
        for v in ("rnd1", "rnd2", "rnd3"):
            w = min(1.0, h / rnd_thr[v])
            out[v][h] = float(crps_q(y, mu[:, None] * lam +
                                    S[:, None] * ((1 - w) * xi + w * G)).mean())
    return out


PANEL = {}
T0S = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    PANEL[m] = np.log(d.price.to_numpy(float)).astype(float)
    T0S[m] = int(np.searchsorted(pd.to_datetime(d.date).to_numpy(), np.datetime64(CUT[m])))

print("=" * 116)
print("真实数据：各变体的合并 log 比（vs A5，即 V3）")
print("=" * 116)
REAL = {m: eval_variants(PANEL[m], T0S[m], 12345) for m in MK}
for v in VARIANTS:
    lr = [np.log(REAL[m][v][h] / REAL[m]["const"][h]) for m in MK for h in HS]
    print(f"  {VMAP[v]:<18} 合并 log 比 {np.mean(lr):+.5f}")

print("\n" + "=" * 116)
print(f"零分布：打乱序列 × {NSHUF}（每个变体各自建零分布）")
print("=" * 116)
T0 = time.time()
NULL = {v: [] for v in VARIANTS}
for it in range(NSHUF):
    rng = np.random.default_rng(SEED0 + it)
    acc = {v: [] for v in VARIANTS}
    for m in MK:
        lp = PANEL[m]; inc = np.diff(lp)
        lp_s = np.concatenate([[0.0], np.cumsum(inc[rng.permutation(len(inc))])])
        r = eval_variants(lp_s, T0S[m], 777 + it)
        if not r:
            continue
        for v in VARIANTS:
            for h in HS:
                if v in r and h in r[v] and h in r["const"]:
                    acc[v].append(np.log(r[v][h] / r["const"][h]))
    for v in VARIANTS:
        if acc[v]:
            NULL[v].append(float(np.mean(acc[v])))
    if (it + 1) % 15 == 0:
        print(f"  第 {it+1:>3} 次  ({time.time()-T0:.0f} 秒)")

print("\n" + "=" * 116)
print("校准后的安慰剂检验")
print("=" * 116)
print(f"{'变体':<20}{'真实log比':>12}{'零分布均值':>13}{'零分布SD':>11}{'校准p':>9}{'百分位':>9}")
OUT = {}
for v in VARIANTS:
    lr = [np.log(REAL[m][v][h] / REAL[m]["const"][h]) for m in MK for h in HS]
    mu_real = float(np.mean(lr))
    nz = np.array(NULL[v])
    p1 = float(np.mean(nz <= mu_real))
    pct = float(np.mean(nz < mu_real) * 100)
    OUT[v] = dict(real_mu=mu_real, null_mean=float(nz.mean()), null_sd=float(nz.std()),
                  p=p1, pct=pct)
    print(f"{VMAP[v]:<20}{mu_real:>12.5f}{nz.mean():>13.5f}{nz.std():>11.5f}{p1:>9.4f}{pct:>8.1f}%")

print("\n" + "=" * 116)
print("判据检验")
print("=" * 116)
p_real = OUT["real"]["p"]
print(f"  V1 真实 ξ_d 的校准 p = {p_real:.4f}")
for v in ("rand", "const", "lag", "rnd1", "rnd2", "rnd3"):
    pv = OUT[v]["p"]
    if v in ("const",):
        verdict = "无形状订正（≡A5），应为 1.000" if abs(pv - 1) < 0.02 else "**异常**"
    elif v in ("rand", "rnd1", "rnd2", "rnd3"):
        verdict = "✓ 安慰剂通过（不显著）" if pv > 0.05 else "**安慰剂失败**"
    else:
        verdict = "（滞后一期，信息类似真实形状）"
    print(f"  {VMAP[v]:<20} 校准 p = {pv:.4f}   {verdict}")
print(f"\n  >>> 关键判据：真实 ξ_d 的校准 p ({p_real:.4f}) 是否显著优于"
      f"随机分位 ({OUT['rand']['p']:.4f}) 与随机调度。")
OK = (p_real < 0.05) and (OUT["rand"]["p"] > 0.05) and \
     all(OUT[v]["p"] > 0.05 for v in ("rnd1", "rnd2", "rnd3"))
print(f"  >>> 判定：{'✓ 安慰剂通过' if OK else '✗ 安慰剂未通过'}")

json.dump({"out": OUT, "verdict": bool(OK), "n_shuffle": NSHUF},
          open(os.path.join(DIR, "PLACEBO_CALIB.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 PLACEBO_CALIB.json")
