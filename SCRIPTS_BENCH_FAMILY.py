# -*- coding: utf-8 -*-
"""
基准层 v2：把"日频锚定"拆成一个可分离的族
==========================================
v1 暴露了设计缺陷：HAR/ML 直接预测 h 期累计方差，与"日频方差 × h"混在一起，
无法判断增益来自"日频频率"还是"条件化"。本节把族写清楚：

    族 F:  sigma_h = sqrt( h * sigma2_daily )
    sigma2_daily 由不同的日频估计量给出 —— 这是唯一的自由度。

  成员：SD250（=DAT0，常数）、EW97、GARCH(1,1) 滤波、HAR(1)、MSM(4) 滤波、ML(1)
  对照：EMP / EMP0（重叠窗口经验分位）、VR（sd·h^H）、HARh / MLh（直接预测 h 期累计）

若 F 的每个成员都胜 EMP，则"日频锚定"本身是主效应；
若成员之间差异显著，则"条件化"是第二维。

修 v1 的两个 bug：
  (a) GARCH 用 s2full[t+1]（v1 误用训练段数组，越界）；失败时打印原因
  (b) MSM 的 m0 加识别约束 m0 ∈ (0.30, 0.99)；v1 在 CEA 顶到 1e-4 边界导致崩溃

数据：PANEL_5MARKET.csv   输出：BENCH_FAMILY.json
"""
import numpy as np, pandas as pd, os, sys, json, warnings, traceback
from scipy.stats import norm, t as tdist
from scipy.optimize import minimize
from functools import reduce
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(OUT, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS); JM = 7; J05 = 0; J95 = 14
W = 250; HOR = [5, 20, 60, 120]; STEP = 2
KS = np.arange(0.30, 2.01, 0.01)
MODELS = ["EMP", "EMP0", "VR", "SD250", "EW97", "GARCH",
          "HAR1", "MSM1", "ML1", "HARh", "MLh"]
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
K_MSM = 4; B_MSM = 2.0; M0_LO, M0_HI = 0.30, 0.99


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


def dm(d, L=10):
    n = len(d)
    if n < 20:
        return np.nan, np.nan
    m = d.mean(); dc = d - m
    lrv = dc @ dc / n
    for l in range(1, L + 1):
        lrv += 2 * (1 - l / (L + 1)) * (dc[l:] @ dc[:-l] / n)
    if lrv <= 0:
        return np.nan, np.nan
    s = m / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    return float(s), float(2 * (1 - tdist.cdf(abs(s), df=n - 1)))


def msm_build(m0):
    n = 2 ** K_MSM
    bits = np.array([[(s >> j) & 1 for j in range(K_MSM)] for s in range(n)])
    return np.prod(np.where(bits == 0, m0, B_MSM - m0), axis=1)


def msm_P(g):
    return reduce(np.kron, [np.array([[gj, 1 - gj], [1 - gj, gj]]) for gj in g])


def msm_ll(theta, r):
    m0 = M0_LO + (M0_HI - M0_LO) / (1.0 + np.exp(-theta[0]))
    sig = np.exp(theta[1])
    g1 = 1.0 / (1.0 + np.exp(-theta[2]))
    g = [1.0 - (1.0 - g1) ** (B_MSM ** j) for j in range(K_MSM)]
    if sig <= 1e-8 or g1 <= 1e-6:
        return 1e10
    Mv = msm_build(m0); P = msm_P(g); n = 2 ** K_MSM
    sM = sig * np.sqrt(Mv)
    pi = np.full(n, 1.0 / n); ll = 0.0
    for t in range(len(r)):
        f = pi * (norm.pdf(r[t], 0.0, sM) + 1e-300)
        s = f.sum()
        if not np.isfinite(s) or s <= 0:
            return 1e10
        ll += np.log(s)
        pi = P.T @ (f / s)
    return -ll


def msm_fit(r):
    best = None
    for x0 in ([-6.0, np.log(np.std(r)), 0.0], [0.0, np.log(np.std(r)), 1.5],
               [6.0, np.log(np.std(r)), -1.0]):
        try:
            res = minimize(msm_ll, np.array(x0), args=(r,), method="Nelder-Mead",
                           options=dict(maxiter=900, xatol=1e-3, fatol=1e-3))
        except Exception:
            continue
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        return None
    m0 = M0_LO + (M0_HI - M0_LO) / (1.0 + np.exp(-best.x[0]))
    g1 = 1.0 / (1.0 + np.exp(-best.x[2]))
    g = [1.0 - (1.0 - g1) ** (B_MSM ** j) for j in range(K_MSM)]
    return dict(m0=m0, sig=np.exp(best.x[1]), g=g, P=msm_P(g), Mv=msm_build(m0),
                ll=-best.fun)


def msm_filter_path(par, r):
    n = 2 ** K_MSM; P = par["P"]; sM = par["sig"] * np.sqrt(par["Mv"])
    pi = np.full(n, 1.0 / n); out = np.zeros((len(r) + 1, n)); out[0] = pi
    for t in range(len(r)):
        f = pi * (norm.pdf(r[t], 0.0, sM) + 1e-300)
        s = f.sum()
        pi = P.T @ (f / max(s, 1e-300) if s > 0 else pi)
        out[t + 1] = pi
    return out


def daily_feats(lp, t):
    r = np.diff(lp[max(0, t - 60):t + 1])
    if len(r) < 25:
        return None
    v = np.maximum(r, 0) ** 2 + np.maximum(-r, 0) ** 2
    return [np.log(v[-1] + 1e-12), np.log(v[-5:].mean() + 1e-12),
            np.log(v[-22:].mean() + 1e-12), np.log(v[-60:].mean() + 1e-12),
            lp[t] - lp[t - 5], lp[t] - lp[t - 22],
            float((np.abs(r) < 1e-12).mean()), np.std(r[-22:], ddof=1)]


RES = {}
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    ra = np.diff(lp); dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    r_tr = ra[:t0]
    print(f"\n{'='*156}\n{m}   切分 {CUT[m]}  n_tr={t0} n_te={len(lp)-t0}")
    # --- MSM
    msm_par = msm_fit(r_tr)
    piv = msm_filter_path(msm_par, ra) if msm_par else None
    if msm_par:
        print(f"  MSM(4) : m0={msm_par['m0']:.4f}  sigma={msm_par['sig']:.6f}  "
              f"g=[{', '.join(f'{x:.4f}' for x in msm_par['g'])}]  ll={msm_par['ll']:.1f}")
    # --- GARCH
    gp = None
    try:
        from arch import arch_model
        am = arch_model(r_tr * 100.0, mean="Zero", vol="Garch", p=1, q=1, dist="normal")
        rr_ = am.fit(disp="off", show_warning=False)
        om = float(rr_.params["omega"]) / 1e4
        al = float(rr_.params["alpha[1]"]); be = float(rr_.params["beta[1]"])
        if al + be >= 1:
            print(f"  GARCH  : 拒绝（persist={al+be:.4f} >= 1，非平稳）")
        else:
            s2 = np.zeros(len(ra) + 1); s2[0] = om / (1 - al - be)
            for t in range(len(ra)):
                s2[t + 1] = om + al * ra[t] ** 2 + be * s2[t]
            gp = dict(om=om, al=al, be=be, s2=s2)
            print(f"  GARCH  : omega={om:.3e} alpha={al:.3f} beta={be:.3f} persist={al+be:.3f}")
    except Exception as e:
        print(f"  GARCH  : 失败 {type(e).__name__}: {e}")
        traceback.print_exc(limit=1)
    cache = {}
    for t in range(W + 5, len(lp)):
        win = lp[max(0, t - W):t]
        r = np.diff(win)
        if len(r) < 60:
            continue
        v1 = np.var(r, ddof=1)
        if v1 <= 0:
            continue
        kk = min(20, len(win) // 3)
        vr = np.var(win[kk:] - win[:-kk], ddof=1) / (kk * v1)
        H = float(np.clip(0.5 + np.log(vr) / (2 * np.log(kk)), 0.05, 0.95)) if vr > 0 else 0.5
        v = r[0] ** 2
        for i in range(1, len(r)):
            v = 0.97 * v + 0.03 * r[i - 1] ** 2
        cache[t] = dict(sd=float(np.sqrt(v1)), H=H, e97=float(max(v, 1e-14)))
    print(f"{'h':>4}{'n_te':>6}" + "".join(f"{n:>9}" for n in MODELS) + f"{'最优':>7}")
    RES[m] = {}
    for h in HOR:
        ts_tr = [t for t in range(W + h + 5, t0 - h - 1, STEP) if t in cache]
        ts_te = [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP) if t in cache]
        if len(ts_tr) < 30 or len(ts_te) < 30:
            print(f"{h:>4}  SKIP {len(ts_tr)}/{len(ts_te)}"); continue

        # 1 步日频方差模型：HAR(1) 与 ML(1)，训练段拟合
        X1, Y1 = [], []
        for t in ts_tr:
            f = daily_feats(lp, t)
            if f is None:
                continue
            r1 = lp[t + 1] - lp[t]
            X1.append(f); Y1.append(np.log(max(r1 * r1, 1e-14)))
        X1 = np.array(X1); Y1 = np.array(Y1)
        har1 = None; ml1 = None
        if len(Y1) > 40:
            har1 = np.linalg.lstsq(np.column_stack([np.ones(len(X1))] + [X1[:, i] for i in (0, 1, 2)]),
                                   Y1, rcond=None)[0]
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor
                ml1 = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05,
                                                    max_depth=3, min_samples_leaf=15,
                                                    l2_regularization=1.0, random_state=7)
                ml1.fit(X1, Y1)
            except Exception as e:
                print("  ML1 失败:", e)
        # 直接 h 期累计模型
        Xh, Yh = [], []
        for t in ts_tr:
            f = daily_feats(lp, t)
            if f is None:
                continue
            s = float(np.sum(np.diff(lp[t:t + h + 1]) ** 2))
            Xh.append(f); Yh.append(np.log(max(s, 1e-12)))
        Xh = np.array(Xh); Yh = np.array(Yh)
        harh = None; mlh = None
        if len(Yh) > 40:
            harh = np.linalg.lstsq(np.column_stack([np.ones(len(Xh))] + [Xh[:, i] for i in (0, 1, 2)]),
                                   Yh, rcond=None)[0]
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor
                mlh = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05,
                                                    max_depth=3, min_samples_leaf=15,
                                                    l2_regularization=1.0, random_state=7)
                mlh.fit(Xh, Yh)
            except Exception:
                pass

        def mk(ts):
            out = {n: np.full((len(ts), 15), np.nan) for n in MODELS}
            for i, t in enumerate(ts):
                sc = cache[t]; sh = np.sqrt(h)
                win = lp[max(0, t - W):t]; rr = win[h:] - win[:-h]
                sd = np.std(rr, ddof=1)
                if sd <= 0:
                    continue
                out["EMP"][i] = np.quantile(rr, TAUS)
                out["EMP0"][i] = np.quantile(rr - rr.mean(), TAUS)
                out["VR"][i] = sc["sd"] * (h ** sc["H"]) * GV
                out["SD250"][i] = sc["sd"] * sh * GV
                out["EW97"][i] = np.sqrt(h * sc["e97"]) * GV
                if gp is not None:
                    out["GARCH"][i] = np.sqrt(h * gp["s2"][t + 1]) * GV
                if piv is not None:
                    out["MSM1"][i] = np.sqrt(h * msm_par["sig"] ** 2 * (piv[t + 1] @ msm_par["Mv"])) * GV
                f = daily_feats(lp, t)
                if f is not None:
                    if har1 is not None:
                        lv = har1 @ np.array([1.0, f[0], f[1], f[2]])
                        out["HAR1"][i] = np.sqrt(h * np.exp(np.clip(lv, -30, 30))) * GV
                    if ml1 is not None:
                        lv = float(ml1.predict(np.array([f]))[0])
                        out["ML1"][i] = np.sqrt(h * np.exp(np.clip(lv, -30, 30))) * GV
                    if harh is not None:
                        lv = harh @ np.array([1.0, f[0], f[1], f[2]])
                        out["HARh"][i] = np.sqrt(np.exp(np.clip(lv, -30, 30))) * GV
                    if mlh is not None:
                        lv = float(mlh.predict(np.array([f]))[0])
                        out["MLh"][i] = np.sqrt(np.exp(np.clip(lv, -30, 30))) * GV
            return out

        A_tr = mk(ts_tr); A_te = mk(ts_te)
        y_tr = lp[np.array(ts_tr) + h] - lp[ts_tr]
        y_te = lp[np.array(ts_te) + h] - lp[ts_te]
        use = [n for n in MODELS if np.isfinite(A_tr[n]).all(1).mean() > 0.95
               and np.isfinite(A_te[n]).all(1).mean() > 0.95]
        gtr = np.all([np.isfinite(A_tr[n]).all(1) for n in use], axis=0)
        gte = np.all([np.isfinite(A_te[n]).all(1) for n in use], axis=0)
        if gtr.sum() < 25 or gte.sum() < 25:
            print(f"{h:>4}  SKIPmask {gtr.sum()}/{gte.sum()}"); continue
        ks = {n: pick_k(y_tr[gtr], A_tr[n][gtr]) for n in use}
        y = y_te[gte]
        L = {n: crps_q(y, apply_k(A_te[n][gte], ks[n])) for n in use}
        c = {n: float(L[n].mean()) for n in use}
        best = min(c, key=c.get)
        print(f"{h:>4}{int(gte.sum()):>6}" + "".join(
            f"{c[n]:>9.5f}" if n in c else f"{'-':>9}" for n in MODELS) + f"{best:>7}")
        RES[m][h] = dict(n=int(gte.sum()), best=best, **c)

print(f"\n{'='*156}")
print("一、全表 CRPS（realistic 校准，位置=0）")
print(f"{'市场':<12}{'h':>5}" + "".join(f"{n:>9}" for n in MODELS) + f"{'最优':>7}")
Wc = {}
for m in RES:
    for h in sorted(RES[m]):
        r = RES[m][h]; Wc[r["best"]] = Wc.get(r["best"], 0) + 1
        print(f"{m:<12}{h:>5}" + "".join(
            f"{r[n]:>9.5f}" if n in r else f"{'-':>9}" for n in MODELS) + f"{r['best']:>7}")
print(f"\n  获胜计数：{Wc}")

print(f"\n{'='*156}")
print("二、族 F 检验：sigma_h = sqrt(h * sigma2_daily)，相对 EMP（重叠窗口经验分位）")
FAM = ["SD250", "EW97", "GARCH", "HAR1", "MSM1", "ML1"]
print(f"{'成员':<8}{'中位':>9}{'更好':>7}{'更差':>7}" + "".join(f"{m[:6]:>8}" for m in RES))
for n in FAM:
    allr = []
    line = f"{n:<8}"
    for m in RES:
        v = [(RES[m][h][n] / RES[m][h]["EMP"] - 1) * 100 for h in sorted(RES[m]) if n in RES[m][h]]
        allr += v
        line += f"{np.median(v):>+7.2f}%" if v else f"{'-':>8}"
    allr = np.array(allr)
    print(line + f"  中位 {np.median(allr):+.2f}%  更好 {(allr<0).sum()}/{len(allr)}")

print(f"\n{'='*156}")
print("三、条件化增益：族内各成员相对 SD250（常数日频方差 = DAT0）")
print(f"{'成员':<8}{'中位':>9}{'更好':>7}" + "".join(f"{m[:6]:>8}" for m in RES))
for n in FAM[1:] + ["HARh", "MLh"]:
    allr = []
    line = f"{n:<8}"
    for m in RES:
        v = [(RES[m][h][n] / RES[m][h]["SD250"] - 1) * 100 for h in sorted(RES[m]) if n in RES[m][h]]
        allr += v
        line += f"{np.median(v):>+7.2f}%" if v else f"{'-':>8}"
    allr = np.array(allr)
    print(line + f"  中位 {np.median(allr):+.2f}%  更好 {(allr<0).sum()}/{len(allr)}")

print(f"\n{'='*156}")
print("四、两个维度分解：频率（窗口 vs 日频） × 条件化（常数 vs 条件）")
for m in RES:
    for h in sorted(RES[m]):
        r = RES[m][h]
        if "SD250" in r and "ML1" in r:
            freq = (r["SD250"] / r["EMP"] - 1) * 100
            cond = (r["ML1"] / r["SD250"] - 1) * 100
            print(f"  {m:<11}h={h:<4} 频率 {freq:>+7.2f}%   条件化 {cond:>+7.2f}%   "
                  f"合计 {(r['ML1']/r['EMP']-1)*100:>+7.2f}%")
json.dump(RES, open(os.path.join(OUT, "BENCH_FAMILY.json"), "w"), indent=1,
          ensure_ascii=False, default=float)
print("\n已写出 BENCH_FAMILY.json")
