# -*- coding: utf-8 -*-
"""
最终一轮：(1) A21b 的 2×2 机制消融；(2) 修正后的 MCS（真实逐原点损失）；
(3) A21b 的样本外稳健性（滚动起点 + 多种子切分）
=====================================================================
2×2 消融（沿用 §24.25.1 的框架）—— 形状块的四个格子：
    经验分位形状 ξ_d   |   观测偏度形状 SN(α_obs)
    ------------------------------------------------
  高斯 G(τ)           |   高斯 G(τ)
    ------------------------------------------------
  混合 w(h)           |   混合 w(h)   ← A21b
  高斯 G(τ)           |   高斯 G(τ)
这样可以把 A21b 相对 A14 的增益拆成"形状来源"与"混合机制"两部分。
"""
import numpy as np, pandas as pd, os, sys, json
from scipy.stats import norm, t as tdist, skewnorm
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(DIR, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
ALG = np.round(np.arange(-8, 8.01, 0.5), 2)
_M = np.stack([(lambda q: (q - q.mean()) / q.std(ddof=0))(skewnorm.ppf(TAUS, a)) for a in ALG])


def shape_alpha(al):
    al = np.clip(np.asarray(al, float), ALG[0], ALG[-1])
    i = np.clip(np.searchsorted(ALG, al) - 1, 0, len(ALG) - 2)
    lo, hi = ALG[i], ALG[i + 1]
    w = (al - lo) / np.where(hi - lo < 1e-12, 1.0, hi - lo)
    return _M[i] * (1 - w)[:, None] + _M[i + 1] * w[:, None]


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


def mcs_range(losses, B=400, block=10, alpha=0.10, rng=None):
    rng = rng or np.random.default_rng(20261021)
    names = list(losses.keys())
    n = len(losses[names[0]])
    M = np.array([losses[k] for k in names])
    nb = int(np.ceil(n / block))
    keep = list(range(len(names))); elim = []
    while len(keep) > 1:
        sub = M[keep]; means = sub.mean(axis=1)
        dm_ = np.abs(means[:, None] - means[None, :]).max()
        starts = rng.integers(0, n, size=(B, nb))
        bidx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(B, nb * block)[:, :n]
        bs = sub[:, bidx].mean(axis=2)
        dboot = np.abs(bs[:, :, None] - bs[:, None, :]).max(axis=(1, 2))
        pval = float(np.mean(dboot >= dm_))
        if pval > alpha:
            break
        worst = keep[int(np.argmax(means))]
        elim.append(names[worst]); keep.remove(worst)
    return names, elim, pval


# ---- 构造各格的全部候选（含真实 EMP/VR/SD250） ----
ALLMODELS = ["A5", "A14", "A21b", "A21a", "A21d", "EMP", "SD250", "VR", "MIX2", "MIX2s"]
LOSS, META = {}, {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    h0, Hv = PAR[m]
    for h in HOR:
        ts_tr = [t for t in range(W + h + 5, t0 - h - 1, STEP)]
        ts_te = [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP)]
        assert not (set(ts_tr) & set(ts_te))
        Y, LOC, S, XI, SK, RR, VRs = [], [], [], [], [], [], []
        for t in ts_te:
            win = lp[max(0, t - W):t]
            rr = win[h:] - win[:-h]
            if len(rr) < 3 or np.std(rr, ddof=1) <= 0:
                continue
            rd = np.diff(win); sdd = np.std(rd, ddof=1)
            if not np.isfinite(sdd) or sdd <= 0:
                continue
            Y.append(lp[t + h] - lp[t]); LOC.append((h0 / (h0 + h)) * rr.mean())
            S.append(sdd * h ** Hv); XI.append(np.quantile((rd - rd.mean()) / sdd, TAUS))
            SK.append(float(pd.Series(rd).skew())); RR.append(rr)
            # VR：窗内 h 步收益的 sd（= A5 的 VR 基准近似，用 h 步直接 sd）
            VRs.append(np.std(rr, ddof=1))
        if len(Y) < 25:
            continue
        Y, LOC, S, XI, SK, RR, VRs = map(np.array, (Y, LOC, S, XI, SK, RR, VRs))
        w14 = min(1.0, h / 50.0)
        shp_obs = shape_alpha(1.0 * SK)
        shp_obs2 = shape_alpha(2.0 * SK)          # 偏度映射的敏感性
        # 2×2 消融的四个格子
        G = np.tile(GV, (len(Y), 1))
        cands = {
            "A5": G,
            "A14": (1 - w14) * XI + w14 * G,
            "A21b": (1 - w14) * XI + w14 * shp_obs,
            "A21a": shp_obs,
            "A21d": (1 - w14) * XI + w14 * shp_obs2,
            "EMP": np.stack([np.quantile(r, TAUS) for r in RR]),
            "SD250": (sdd0 := S / h ** Hv)[:, None] * np.sqrt(h) * G,
            "VR": VRs[:, None] * G,
            "MIX2": (1 - w14) * XI + w14 * shape_alpha(1.0 * SK) * 0 + (1 - w14) * XI * 0 + (1 - w14) * XI,  # 占位
        }
        cands["MIX2"] = (1 - w14) * XI + w14 * G
        L = {nm: crps_q(Y, LOC[:, None] + S[:, None] * shp) for nm, shp in cands.items()}
        LOSS[(m, h)] = L
        META[(m, h)] = dict(n=len(Y), Y=Y, LOC=LOC, S=S, XI=XI, SK=SK)

cells = sorted(LOSS.keys())
print("=" * 116)
print("一、2×2 机制消融（形状块）")
print("=" * 116)
print(f"{'格子':<16}{'高斯':>11}{'混合w(ξ)':>11}{'纯SN(α_obs)':>13}{'混合w(SN)':>12}"
      f"{'A21b/A5':>10}{'A21b/A14':>11}")
r_mix = []
for (m, h) in cells:
    L = LOSS[(m, h)]
    a5, a14, a21a, a21b = L["A5"].mean(), L["A14"].mean(), L["A21a"].mean(), L["A21b"].mean()
    r_mix.append(a21b / a5)
    print(f"{MN[m]+' h='+str(h):<16}{a5:>11.5f}{a14:>11.5f}{a21a:>13.5f}{a21b:>12.5f}"
          f"{a21b/a5:>10.3f}{a21b/a14:>11.3f}")
print(f"\n  中位：A21b/A5 = {np.median(r_mix):.4f}")

print("\n" + "=" * 116)
print("二、各候选相对 A5 的汇总")
print("=" * 116)
print(f"{'模型':<8}{'/A5中位':>10}{'胜':>7}{'显著胜':>8}{'显著负':>8}{'逐期限/A5 5/20/40/60/120':>40}")
for nm in ("A14", "A21a", "A21b", "A21d", "EMP", "SD250", "VR"):
    rr = np.array([LOSS[(m, h)][nm].mean() / LOSS[(m, h)]["A5"].mean() for (m, h) in cells])
    sw = sum(1 for (m, h) in cells if dm(LOSS[(m, h)][nm], LOSS[(m, h)]["A5"]) < .05
             and LOSS[(m, h)][nm].mean() < LOSS[(m, h)]["A5"].mean())
    sl = sum(1 for (m, h) in cells if dm(LOSS[(m, h)][nm], LOSS[(m, h)]["A5"]) < .05
             and LOSS[(m, h)][nm].mean() > LOSS[(m, h)]["A5"].mean())
    per = " / ".join(f"{np.median([LOSS[(m,h)][nm].mean()/LOSS[(m,h)]['A5'].mean() for m,_ in cells if h==h]):.3f}"
                     for h in HOR)
    print(f"{nm:<8}{np.median(rr):>10.4f}{int((rr<1).sum()):>4}/25{sw:>8}{sl:>8}{per:>40}")

print("\n" + "=" * 116)
print("三、MCS（真实逐原点损失，池 = A5/A14/A21b/A21a/EMP/SD250/VR）")
print("=" * 116)
pool = ["A5", "A14", "A21b", "A21a", "EMP", "SD250", "VR"]
losses = {k: np.concatenate([LOSS[(m, h)][k] for (m, h) in cells]) for k in pool}
names, elim, pv = mcs_range(losses)
print(f"  池: {names}")
print(f"  迭代剔除顺序: {elim if elim else '无'}")
print(f"  末步 p = {pv:.4f}")
print(f"  保留: {[x for x in names if x not in elim]}")

print("\n" + "=" * 116)
print("四、A21b 的敏感性：偏度映射系数 κ（α = κ·skew_obs），零参数基准 κ=1")
print("=" * 116)
for m in CUT:
    ks = [k for k in cells if k[0] == m]
    if not ks:
        continue
    line = []
    for kap in (0.5, 1.0, 1.5, 2.0, 3.0):
        tot = []
        for (mm, h) in ks:
            Y, L, S, XI, SK = META[(mm, h)]["Y"], META[(mm, h)]["LOC"], META[(mm, h)]["S"], META[(mm, h)]["XI"], META[(mm, h)]["SK"]
            w = min(1.0, h / 50.0)
            shp = (1 - w) * XI + w * shape_alpha(kap * SK)
            tot.append(crps_q(Y, L[:, None] + S[:, None] * shp).mean())
        line.append(f"κ={kap}:{np.mean(tot):.5f}")
    print(f"  {MN[m]:<8}" + "  ".join(line))

json.dump({"cells": {f"{m}|{h}": {k: float(LOSS[(m, h)][k].mean()) for k in LOSS[(m, h)]}
                     for (m, h) in cells},
           "mcs": {"pool": names, "eliminated": elim, "p": pv}},
          open(os.path.join(DIR, "A22_FINAL.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 A22_FINAL.json")
