# -*- coding: utf-8 -*-
"""
论文表一：A5 vs 完整文献基准名册，未校准 CRPS 口径
================================================
相对 SCRIPTS_BENCH_FAMILY.py 的两处订正：
  1. 去掉 pick_k / apply_k（§协议订正：per-model k 校准已废弃，CRPS 是 proper score）
  2. 加入 A5（最终规格）作为被检验模型，并输出 WIS 与 DM
名册：EMP EMP0 VR SD250 EW97 GARCH HAR1 MSM1 ML1 HARh MLh  （11 个基准）
输出：FULL_ROSTER.json
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
GV = norm.ppf(TAUS)
W = 250; HOR = [5, 20, 40, 60, 120]; STEP = 2
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
PAR = {"CEA": (0.1, 0.60), "GUANGZHOU": (30, 0.35), "SHANGHAI": (0.1, 0.30),
       "HUBEI": (3, 0.35), "EU": (80, 0.45)}
MODELS = ["A5", "EMP", "EMP0", "VR", "SD250", "EW97", "GARCH",
          "HAR1", "MSM1", "ML1", "HARh", "MLh"]
K_MSM, B_MSM, M0_LO, M0_HI = 4, 2.0, 0.30, 0.99
# WIS 用的区间对：(下, 上, alpha)
PAIRS = [(0, 14, 0.05), (1, 13, 0.10), (2, 12, 0.20), (3, 11, 0.30), (4, 10, 0.40)]
JM = 7


def crps_q(yt, Q):
    Qs = np.sort(Q, axis=1)
    ind = (yt[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (yt[:, None] - Qs)).mean(axis=1)


def wis(yt, Q):
    tot = 0.5 * np.abs(yt - Q[:, JM])
    for lo, hi, a in PAIRS:
        l = Q[:, lo]; u = Q[:, hi]
        IS = (u - l) + (2.0 / a) * np.maximum(l - yt, 0) + (2.0 / a) * np.maximum(yt - u, 0)
        tot = tot + (a / 2.0) * IS
    return tot / (len(PAIRS) + 0.5)


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
AUD = []
for m in CUT:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    lp = np.log(d.price.to_numpy(float)).astype(float)
    ra = np.diff(lp); dt = pd.to_datetime(d.date).to_numpy()
    t0 = int(np.searchsorted(dt, np.datetime64(CUT[m])))
    print(f"\n{'='*168}\n{m}   切分 {CUT[m]}  n_tr={t0} n_te={len(lp)-t0}")
    msm_par = msm_fit(ra[:t0])
    piv = msm_filter_path(msm_par, ra) if msm_par else None
    if msm_par:
        print(f"  MSM(4): m0={msm_par['m0']:.4f} sig={msm_par['sig']:.6f} ll={msm_par['ll']:.1f}")
    gp = None
    try:
        from arch import arch_model
        am = arch_model(ra[:t0] * 100.0, mean="Zero", vol="Garch", p=1, q=1, dist="normal")
        rr_ = am.fit(disp="off", show_warning=False)
        om = float(rr_.params["omega"]) / 1e4
        al = float(rr_.params["alpha[1]"]); be = float(rr_.params["beta[1]"])
        if al + be >= 1:
            print(f"  GARCH: 拒绝 persist={al+be:.4f}")
        else:
            s2 = np.zeros(len(ra) + 1); s2[0] = om / (1 - al - be)
            for t in range(len(ra)):
                s2[t + 1] = om + al * ra[t] ** 2 + be * s2[t]
            gp = dict(s2=s2)
            print(f"  GARCH: alpha={al:.3f} beta={be:.3f} persist={al+be:.3f}")
    except Exception as e:
        print(f"  GARCH: 失败 {type(e).__name__}: {e}")

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
    RES[m] = {}
    for h in HOR:
        ts_tr = [t for t in range(W + h + 5, t0 - h - 1, STEP) if t in cache]
        ts_te = [t for t in range(max(W + h + 5, t0), len(lp) - h, STEP) if t in cache]
        ov = len(set(ts_tr) & set(ts_te))
        AUD.append((m, h, len(ts_tr), len(ts_te), ov))
        if ov != 0:
            print(f"  !! 泄漏 {m} h={h} 重叠 {ov}")
        if len(ts_tr) < 30 or len(ts_te) < 30:
            print(f"{h:>4}  SKIP {len(ts_tr)}/{len(ts_te)}"); continue

        X1, Y1 = [], []
        for t in ts_tr:
            f = daily_feats(lp, t)
            if f is None:
                continue
            X1.append(f); Y1.append(np.log(max((lp[t + 1] - lp[t]) ** 2, 1e-14)))
        X1 = np.array(X1); Y1 = np.array(Y1)
        har1 = ml1 = None
        if len(Y1) > 40:
            har1 = np.linalg.lstsq(np.column_stack([np.ones(len(X1))] + [X1[:, i] for i in (0, 1, 2)]),
                                   Y1, rcond=None)[0]
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor
                ml1 = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_depth=3,
                                                    min_samples_leaf=15, l2_regularization=1.0,
                                                    random_state=7).fit(X1, Y1)
            except Exception as e:
                print("  ML1 失败:", e)
        Xh, Yh = [], []
        for t in ts_tr:
            f = daily_feats(lp, t)
            if f is None:
                continue
            Xh.append(f); Yh.append(np.log(max(float(np.sum(np.diff(lp[t:t + h + 1]) ** 2)), 1e-12)))
        Xh = np.array(Xh); Yh = np.array(Yh)
        harh = mlh = None
        if len(Yh) > 40:
            harh = np.linalg.lstsq(np.column_stack([np.ones(len(Xh))] + [Xh[:, i] for i in (0, 1, 2)]),
                                   Yh, rcond=None)[0]
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor
                mlh = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_depth=3,
                                                    min_samples_leaf=15, l2_regularization=1.0,
                                                    random_state=7).fit(Xh, Yh)
            except Exception:
                pass

        h0, Hv = PAR[m]

        def mk(ts):
            out = {n: np.full((len(ts), 15), np.nan) for n in MODELS}
            for i, t in enumerate(ts):
                sc = cache[t]; sh = np.sqrt(h)
                win = lp[max(0, t - W):t]; rr = win[h:] - win[:-h]
                sd = np.std(rr, ddof=1)
                if sd <= 0:
                    continue
                rd = np.diff(win)
                loc = rr.mean() * (h0 / (h0 + h))
                out["A5"][i] = loc + sc["sd"] * (h ** Hv) * GV
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
                        out["HAR1"][i] = np.sqrt(h * np.exp(np.clip(har1 @ np.array([1.0, f[0], f[1], f[2]]), -30, 30))) * GV
                    if ml1 is not None:
                        out["ML1"][i] = np.sqrt(h * np.exp(np.clip(float(ml1.predict(np.array([f]))[0]), -30, 30))) * GV
                    if harh is not None:
                        out["HARh"][i] = np.sqrt(np.exp(np.clip(harh @ np.array([1.0, f[0], f[1], f[2]]), -30, 30))) * GV
                    if mlh is not None:
                        out["MLh"][i] = np.sqrt(np.exp(np.clip(float(mlh.predict(np.array([f]))[0]), -30, 30))) * GV
            return out

        A_tr = mk(ts_tr); A_te = mk(ts_te)
        y_te = lp[np.array(ts_te) + h] - lp[ts_te]
        y_tr = lp[np.array(ts_tr) + h] - lp[ts_tr]
        use = [n for n in MODELS if np.isfinite(A_tr[n]).all(1).mean() > 0.95
               and np.isfinite(A_te[n]).all(1).mean() > 0.95]
        if "A5" not in use:
            use.append("A5")
        gte = np.all([np.isfinite(A_te[n]).all(1) for n in use], axis=0)
        if gte.sum() < 25:
            print(f"{h:>4}  SKIPmask {gte.sum()}"); continue
        y = y_te[gte]
        L = {n: crps_q(y, A_te[n][gte]) for n in use}
        Wv = {n: float(wis(y, A_te[n][gte]).mean()) for n in use}
        c = {n: float(L[n].mean()) for n in use}
        pv = {n: dm(L["A5"] - L[n])[1] for n in use if n != "A5"}
        RES[m][h] = dict(n=int(gte.sum()), crps=c, wis=Wv, p_vs_A5=pv,
                         c80=float(np.mean((y >= A_te["A5"][gte][:, 2]) & (y <= A_te["A5"][gte][:, 12]))),
                         c95=float(np.mean((y >= A_te["A5"][gte][:, 0]) & (y <= A_te["A5"][gte][:, 14]))))
        rk = sorted(c, key=c.get)
        print(f"{h:>4}{int(gte.sum()):>6}" + "".join(f"{c[n]:>9.5f}" if n in c else f"{'-':>9}"
                                                     for n in MODELS) + f"   A5排名 {rk.index('A5')+1}/{len(rk)}")

print("\n" + "=" * 168)
print("泄漏审计（第三条必须为 0）")
for a in AUD:
    print(f"{a[0]:<12}{a[1]:>5}{a[2]:>10}{a[3]:>10}{a[4]:>7}")
print("重叠总数 =", sum(a[4] for a in AUD))
json.dump(dict(res={m: {str(h): RES[m][h] for h in RES[m]} for m in RES},
               audit=[list(a) for a in AUD], models=MODELS),
          open(os.path.join(OUT, "FULL_ROSTER.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("已写出 FULL_ROSTER.json（汇总见 roster2.py）")
