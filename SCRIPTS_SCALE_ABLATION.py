# -*- coding: utf-8 -*-
"""
尺度消融：长期限离散度该怎么估？
================================
jats3.py 分解出的核心事实：把 250 日重叠窗口的经验 sigma_h 换成日频锚定的尺度，
中位改善 4.1%，16/20 格更好，最大 -35.9%（CEA h=120）。
机制：h=120 的重叠窗口只有 250/120 ≈ 2 个独立观测；日频尺度有 249 个准独立观测。

但没人测过 BV * h^H（日频锚定 + 拟合期限剖面）。本脚本只改尺度、形状一律 Gaussian，
把尺度问题彻底隔离：
  SIG_EMP    250 日重叠窗口的经验 h 期标准差        ← 现状基准
  SIG_BV_SQ  BV * sqrt(h)                          ← 纯随机游走缩放
  SIG_BV_H   BV * h^H                              ← 候选：日频水平 + 拟合剖面
  SIG_SD_H   sd(日收益) * h^H                      ← 原 VR 模型的尺度（含跳跃方差）
  SIG_SD_SQ  sd(日收益) * sqrt(h)                  ← 全波动 * 随机游走缩放（对照）
  其中 H 由窗口内 VR(20) 反解（price level 口径，非 return 差分）
校准：k 由训练段定，测试段固定
"""
import numpy as np, pandas as pd, os, sys, json, warnings
from scipy.stats import norm
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS); JM = 7; J05 = 0; J95 = 14
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
KS = np.arange(0.30, 2.01, 0.01)
MODELS = ["EMP", "BV_SQ", "BV_H", "SD_H", "SD_SQ"]


def crps_q(yt, Q):
    Qs = np.sort(Q, axis=1)
    ind = (yt[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (yt[:, None] - Qs)).mean(axis=1)


def pick_k(y, Q):
    med = Q[:, JM][:, None]
    cov = np.array([((y >= med[:, 0] + k * (Q[:, J05] - med[:, 0])) &
                     (y <= med[:, 0] + k * (Q[:, J95] - med[:, 0]))).mean() for k in KS])
    return float(KS[int(np.argmin(np.abs(cov - 0.95)))])


def apply_k(Q, k):
    med = Q[:, JM][:, None]
    return med + k * (Q - med)


def scale_parts(win):
    """返回窗口内各种日频尺度与拟合的 H"""
    r = np.diff(win)
    if len(r) < 60:
        return None
    v1 = np.var(r, ddof=1)
    if v1 <= 0:
        return None
    bv = (np.pi / 2.0) * np.mean(np.abs(r[1:]) * np.abs(r[:-1]))
    sig_bv = np.sqrt(bv) if bv > 0 else np.sqrt(v1)
    kk = min(20, len(win) // 3)
    vr = np.var(win[kk:] - win[:-kk], ddof=1) / (kk * v1)
    H = 0.5 + np.log(vr) / (2 * np.log(kk)) if vr > 0 else 0.5
    H = float(np.clip(H, 0.05, 0.95))
    return dict(sig_bv=float(sig_bv), sig_sd=float(np.sqrt(v1)), H=H)


RES = {}
for m in ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    dt = pd.to_datetime(d.date).to_numpy()
    cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
    t0 = int(np.searchsorted(dt, np.datetime64(cut)))
    cache = {t: scale_parts(lp[max(0, t - W):t]) for t in range(W + 5, len(lp))}
    cache = {k: v for k, v in cache.items() if v is not None}
    print(f"\n{'='*130}\n{m}   切分 {cut}  n_tr={t0} n_te={len(lp)-t0}   有效起点 {len(cache)}")
    print(f"{'h':>4}{'n_te':>6}{'H中位':>7}" + "".join(f"{n:>10}" for n in MODELS)
          + f"{'最优':>8}{'BV_H差':>9}{'BV_H/EMP':>10}{'BV_H/SD_H':>11}")
    RES[m] = {}
    for h in HOR:
        ts_tr = [t for t in range(W + h + 5, t0 - h - 1, STEP) if t in cache]
        ts_te = [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP) if t in cache]
        if len(ts_tr) < 30 or len(ts_te) < 30:
            print(f"{h:>4}  SKIP {len(ts_tr)}/{len(ts_te)}")
            continue

        def mk(ts):
            out = {n: np.full((len(ts), 15), np.nan) for n in MODELS}
            for i, t in enumerate(ts):
                sc = cache[t]
                win = lp[max(0, t - W):t]
                rr = win[h:] - win[:-h]
                sd = np.std(rr, ddof=1)
                if sd <= 0:
                    continue
                base = rr.mean()
                out["EMP"][i] = base + sd * GV
                out["BV_SQ"][i] = base + sc["sig_bv"] * np.sqrt(h) * GV
                out["BV_H"][i] = base + sc["sig_bv"] * (h ** sc["H"]) * GV
                out["SD_H"][i] = base + sc["sig_sd"] * (h ** sc["H"]) * GV
                out["SD_SQ"][i] = base + sc["sig_sd"] * np.sqrt(h) * GV
            return out

        A_tr = mk(ts_tr); A_te = mk(ts_te)
        y_tr = lp[np.array(ts_tr) + h] - lp[ts_tr]
        y_te = lp[np.array(ts_te) + h] - lp[ts_te]
        gtr = np.isfinite(A_tr["EMP"]).all(1)
        gte = np.isfinite(A_te["EMP"]).all(1)
        if gtr.sum() < 25 or gte.sum() < 25:
            print(f"{h:>4}  SKIPmask {gtr.sum()}/{gte.sum()}")
            continue
        ks = {n: pick_k(y_tr[gtr], A_tr[n][gtr]) for n in MODELS}
        y = y_te[gte]
        c = {n: float(crps_q(y, apply_k(A_te[n][gte], ks[n])).mean()) for n in MODELS}
        best = min(c, key=c.get)
        Hmed = float(np.median([cache[t]["H"] for t in ts_te]))
        print(f"{h:>4}{int(gte.sum()):>6}{Hmed:>7.3f}" + "".join(f"{c[n]:>10.5f}" for n in MODELS)
              + f"{best:>8}{(c['BV_H']/c[best]-1)*100:>+8.2f}%"
              f"{(c['BV_H']/c['EMP']-1)*100:>+9.2f}%{(c['BV_H']/c['SD_H']-1)*100:>+10.2f}%")
        RES[m][h] = dict(n=int(gte.sum()), H=Hmed, best=best, **c)

print(f"\n{'='*130}")
print("汇总：BV_H 相对各基准的百分比变化（负=更好）")
print(f"{'市场':<12}{'h':>5}{'H':>7}{'BV_H':>10}{'最优':>9}{'vs EMP':>9}{'vs BV_SQ':>10}{'vs SD_H':>10}{'vs SD_SQ':>10}")
agg = {k: [] for k in ["EMP", "BV_SQ", "SD_H", "SD_SQ"]}
Wc = {}
for m in RES:
    for h in sorted(RES[m]):
        r = RES[m][h]
        Wc[r["best"]] = Wc.get(r["best"], 0) + 1
        for k in agg:
            agg[k].append(r["BV_H"] / r[k] - 1)
        print(f"{m:<12}{h:>5}{r['H']:>7.3f}{r['BV_H']:>10.5f}{r['best']:>9}"
              f"{(r['BV_H']/r['EMP']-1)*100:>+8.2f}%{(r['BV_H']/r['BV_SQ']-1)*100:>+9.2f}%"
              f"{(r['BV_H']/r['SD_H']-1)*100:>+9.2f}%{(r['BV_H']/r['SD_SQ']-1)*100:>+9.2f}%")
print(f"\n  获胜计数：{Wc}")
print("\n  BV_H 相对各基准：中位变化 / 更好格数 / 总格数")
for k in ["EMP", "BV_SQ", "SD_H", "SD_SQ"]:
    v = np.array(agg[k]) * 100
    print(f"    vs {k:<7} 中位 {np.median(v):+7.2f}%   更好 {(v<0).sum():>2}/{len(v)}   最差 {v.max():+7.2f}%   最好 {v.min():+7.2f}%")
json.dump(RES, open(os.path.join(OUT, "SCALE_ABLATION.json"), "w"), indent=1,
          ensure_ascii=False, default=float)
print("\n已写出 SCALE_ABLATION.json")
