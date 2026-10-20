# -*- coding: utf-8 -*-
"""
尺度终局：条件波动率 + 显著性
============================
scale1.py 结论：SD_SQ = sd(日收益) * sqrt(h) 在 17/25 格最优，相对 250 日重叠窗口
经验 sigma_h 有 21/25 格改善、中位 -5.85%、最大 -33.72%；BV 与 RV 无差别 → 关键是 n_eff
不是跳跃稳健性。

本脚本回答最后两个问题：
 Q1 日频波动率需要"条件化"吗？  SD_SQ(250日均值) vs SD_SQ_EWMA(lambda=0.94)
    vs SD_SQ_EWMA2(lambda=0.97) —— 若条件化无增益，模型就是纯无条件 sqrt(h) 缩放
 Q2 优势显著吗？逐格 Diebold-Mariano（Harvey-Leybourne-Newbold 修正 + Bartlett HAC）
    对 SD_SQ vs EMP / vs SD_H / vs BV_SQ
另报 PICP95（k 已由训练段定），确认不是靠收窄区间取胜。
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
MODELS = ["EMP", "SD_SQ", "EW94", "EW97", "SD_H", "BV_SQ"]


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


def dm_test(d, L=10):
    """Harvey-Leybourne-Newbold 修正 + Bartlett HAC"""
    n = len(d)
    if n < 20:
        return np.nan, np.nan
    m = d.mean()
    dc = d - m
    lrv = dc @ dc / n
    for l in range(1, L + 1):
        g = dc[l:] @ dc[:-l] / n
        lrv += 2 * (1 - l / (L + 1)) * g
    if lrv <= 0:
        return np.nan, np.nan
    stat = m / np.sqrt(lrv / n)
    corr = np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    stat *= corr
    from scipy.stats import t as tdist
    return float(stat), float(2 * (1 - tdist.cdf(abs(stat), df=n - 1)))


def ewma_var(r, lam):
    v = np.empty(len(r)); v[0] = r[0] ** 2
    for i in range(1, len(r)):
        v[i] = lam * v[i - 1] + (1 - lam) * r[i - 1] ** 2
    return v[-1]


RES = {}
for m in ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    r_all = np.diff(lp)
    dt = pd.to_datetime(d.date).to_numpy()
    cut = "2024-01-01" if m in ("CEA", "GUANGZHOU", "SHANGHAI") else "2023-01-01"
    t0 = int(np.searchsorted(dt, np.datetime64(cut)))
    cache = {}
    for t in range(W + 5, len(lp)):
        win = lp[max(0, t - W):t]
        r = np.diff(win)
        if len(r) < 60:
            continue
        v1 = np.var(r, ddof=1)
        if v1 <= 0:
            continue
        bv = (np.pi / 2.0) * np.mean(np.abs(r[1:]) * np.abs(r[:-1]))
        kk = min(20, len(win) // 3)
        vr = np.var(win[kk:] - win[:-kk], ddof=1) / (kk * v1)
        H = float(np.clip(0.5 + np.log(vr) / (2 * np.log(kk)), 0.05, 0.95)) if vr > 0 else 0.5
        cache[t] = dict(sd=np.sqrt(v1), bv=np.sqrt(bv) if bv > 0 else np.sqrt(v1), H=H,
                        e94=np.sqrt(max(ewma_var(r, 0.94), 1e-14)),
                        e97=np.sqrt(max(ewma_var(r, 0.97), 1e-14)))
    print(f"\n{'='*136}\n{m}   切分 {cut}  n_tr={t0} n_te={len(lp)-t0}   有效起点 {len(cache)}")
    print(f"{'h':>4}{'n_te':>6}" + "".join(f"{n:>10}" for n in MODELS)
          + f"{'最优':>7}{'PICP_SD_SQ':>11}   DM: SD_SQ vs EMP / SD_H / BV_SQ")
    RES[m] = {}
    for h in HOR:
        ts_tr = [t for t in range(W + h + 5, t0 - h - 1, STEP) if t in cache]
        ts_te = [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP) if t in cache]
        if len(ts_tr) < 30 or len(ts_te) < 30:
            print(f"{h:>4}  SKIP {len(ts_tr)}/{len(ts_te)}"); continue

        def mk(ts):
            out = {n: np.full((len(ts), 15), np.nan) for n in MODELS}
            for i, t in enumerate(ts):
                sc = cache[t]
                win = lp[max(0, t - W):t]
                rr = win[h:] - win[:-h]
                sd = np.std(rr, ddof=1)
                if sd <= 0:
                    continue
                b = rr.mean(); sh = np.sqrt(h)
                out["EMP"][i] = b + sd * GV
                out["SD_SQ"][i] = b + sc["sd"] * sh * GV
                out["EW94"][i] = b + sc["e94"] * sh * GV
                out["EW97"][i] = b + sc["e97"] * sh * GV
                out["SD_H"][i] = b + sc["sd"] * (h ** sc["H"]) * GV
                out["BV_SQ"][i] = b + sc["bv"] * sh * GV
            return out

        A_tr = mk(ts_tr); A_te = mk(ts_te)
        y_tr = lp[np.array(ts_tr) + h] - lp[ts_tr]
        y_te = lp[np.array(ts_te) + h] - lp[ts_te]
        gtr = np.isfinite(A_tr["EMP"]).all(1); gte = np.isfinite(A_te["EMP"]).all(1)
        if gtr.sum() < 25 or gte.sum() < 25:
            print(f"{h:>4}  SKIPmask {gtr.sum()}/{gte.sum()}"); continue
        ks = {n: pick_k(y_tr[gtr], A_tr[n][gtr]) for n in MODELS}
        y = y_te[gte]
        L = {n: crps_q(y, apply_k(A_te[n][gte], ks[n])) for n in MODELS}
        c = {n: float(L[n].mean()) for n in MODELS}
        best = min(c, key=c.get)
        Q = apply_k(A_te["SD_SQ"][gte], ks["SD_SQ"])
        pic = float(((y >= Q[:, J05]) & (y <= Q[:, J95])).mean())
        d1 = L["SD_SQ"] - L["EMP"]; d2 = L["SD_SQ"] - L["SD_H"]; d3 = L["SD_SQ"] - L["BV_SQ"]
        s1, p1 = dm_test(d1); s2, p2 = dm_test(d2); s3, p3 = dm_test(d3)
        print(f"{h:>4}{int(gte.sum()):>6}" + "".join(f"{c[n]:>10.5f}" for n in MODELS)
              + f"{best:>7}{pic:>11.3f}   SB={s1:+6.2f}(p{p1:.3f}) "
                f"SH={s2:+6.2f}(p{p2:.3f}) B={s3:+6.2f}(p{p3:.3f})")
        RES[m][h] = dict(n=int(gte.sum()), best=best, picc=pic, **c,
                         dm_emp=[s1, p1], dm_sdh=[s2, p2], dm_bv=[s3, p3])

print(f"\n{'='*136}")
print("汇总：SD_SQ 相对各基准的百分比变化（负=更好）+ DM 显著性")
print(f"{'市场':<12}{'h':>5}{'SD_SQ':>10}{'最优':>8}{'vs EMP':>9}{'p':>7}"
      f"{'vs EW94':>9}{'vs EW97':>9}{'vs SD_H':>9}{'p':>7}{'vs BV_SQ':>9}")
rec = {"EMP": [], "EW94": [], "EW97": [], "SD_H": [], "BV_SQ": []}
sig = {"EMP": 0, "SD_H": 0, "BV_SQ": 0}; Wc = {}; ntot = 0
for m in RES:
    for h in sorted(RES[m]):
        r = RES[m][h]; ntot += 1
        Wc[r["best"]] = Wc.get(r["best"], 0) + 1
        for k in rec:
            rec[k].append(r["SD_SQ"] / r[k] - 1)
        for k in sig:
            if r["SD_SQ"] < r[k] and {"EMP": r["dm_emp"], "SD_H": r["dm_sdh"],
                                      "BV_SQ": r["dm_bv"]}[k][1] < 0.05:
                sig[k] += 1
        print(f"{m:<12}{h:>5}{r['SD_SQ']:>10.5f}{r['best']:>8}"
              f"{(r['SD_SQ']/r['EMP']-1)*100:>+8.2f}%{r['dm_emp'][1]:>7.3f}"
              f"{(r['SD_SQ']/r['EW94']-1)*100:>+8.2f}%{(r['SD_SQ']/r['EW97']-1)*100:>+8.2f}%"
              f"{(r['SD_SQ']/r['SD_H']-1)*100:>+8.2f}%{r['dm_sdh'][1]:>7.3f}"
              f"{(r['SD_SQ']/r['BV_SQ']-1)*100:>+8.2f}%")
print(f"\n  获胜计数：{Wc}")
for k in rec:
    v = np.array(rec[k]) * 100
    print(f"  vs {k:<7} 中位 {np.median(v):+7.2f}%  更好 {(v<0).sum():>2}/{len(v)}"
          f"  显著更好 {sig.get(k,0):>2}/{len(v)}")
json.dump(RES, open(os.path.join(OUT, "SCALE_FINAL.json"), "w"), indent=1,
          ensure_ascii=False, default=float)
print("\n已写出 SCALE_FINAL.json")
