# -*- coding: utf-8 -*-
"""
实验体系 14c：文献标准基准（最终版，省时设计）
=====================================================================
相对 exp14b 的关键改动：零分布只对**最关键的对照 GARCH_H** 完整构建，
其余对照（A15 vs A5）复用已算好的零分布；文献基准的零分布用
**固定阶 ARIMA(1,1,1) + 抽稀 1/4 + 30 次**，只用于估计**中心与尺度**。

判据：合并 log 比的一侧经验 p。
输出 LIT_BENCH.json
"""
import numpy as np, pandas as pd, os, sys, json, warnings, time
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
DIR = r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research\research\fullcycle_h20_20261020"
p = pd.read_csv(os.path.join(DIR, "PANEL_5MARKET.csv"), parse_dates=["date"])
TAUS = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975])
GV = norm.ppf(TAUS)
W = 250; STEP = 2
HS = [5, 20, 40]
HSAT_A15 = 50.0          # 最终规格（§24.36：规则 R 与固定 50 等价，取 50 作主规格）
H0G = [0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 500]
HG = np.round(np.arange(0.15, 0.861, 0.05), 3)
MK = ["CEA", "GUANGZHOU", "SHANGHAI", "HUBEI", "EU"]
MN = {"CEA": "CEA", "GUANGZHOU": "广州", "SHANGHAI": "上海", "HUBEI": "湖北", "EU": "EU"}
CUT = {"CEA": "2024-01-01", "GUANGZHOU": "2024-01-01", "SHANGHAI": "2024-01-01",
       "HUBEI": "2023-01-01", "EU": "2023-01-01"}
NSHUF = 30
SEED0 = 606060
LIB = ["ARIMA", "GARCH", "GARCH_H", "AR", "EMA", "VAR"]
MARKET = None


def crps_q(y, Q):
    Qs = np.sort(Q, axis=1)
    ind = (y[:, None] < Qs).astype(float)
    return (2 * (TAUS[None, :] - ind) * (y[:, None] - Qs)).mean(axis=1)


def build(lp, h, t0, step=None):
    step = step or STEP
    rec = []
    for t in range(W + h + 5, len(lp) - h, step):
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


def arima_forecast(hist, h, fast, p_now):
    try:
        if fast:
            r = ARIMA(hist[-1000:], order=(1, 1, 1)).fit()
        else:
            best = None
            for p_, q_ in [(1, 0), (1, 1), (2, 1)]:
                try:
                    rr = ARIMA(hist[-1500:], order=(p_, 1, q_)).fit()
                    if best is None or rr.aic < best[0]:
                        best = (rr.aic, rr)
                except Exception:
                    continue
            if best is None:
                return None
            r = best[1]
        fc = r.get_forecast(steps=h)
        mu = float(fc.predicted_mean[-1]) - p_now
        resid = np.asarray(r.resid)
        sd = float(np.std(resid, ddof=1)) if len(resid) > 2 else 1e-4
        return mu, sd if sd > 0 else 1e-4
    except Exception:
        return None


def garch_forecast(hist, h):
    try:
        r = np.diff(hist[-1500:]) * 100.0
        res = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(
            disp="off", show_warning=False)
        fc = res.forecast(horizon=h, reindex=False)
        sig = float(np.asarray(fc.variance.values[-1])[-1]) ** 0.5 / 100.0
        nu = float(res.params.get("nu", 8.0))
        sgn = np.sqrt(nu / (nu - 2)) if nu > 2.05 else 1.0
        mu = float(res.params.get("mu", 0.0)) / 100.0
        return mu * h, sig * sgn
    except Exception:
        return None


def ar_forecast(hist, h, p_now, p_ord=5):
    try:
        y = hist[-1500:]
        X = np.column_stack([y[p_ord - i - 1:len(y) - i - 1] for i in range(p_ord)])
        yy = y[p_ord:]
        beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        sd = float(np.std(yy - X @ beta, ddof=p_ord))
        return float(beta @ y[-p_ord:][::-1]) - p_now, sd if sd > 0 else 1e-4
    except Exception:
        return None


def ema_forecast(hist, h, p_now, n_ema=20):
    try:
        y = hist[-1500:]
        a = 2.0 / (n_ema + 1.0)
        e = y[0]
        for v in y[1:]:
            e = a * v + (1 - a) * e
        resid = y - pd.Series(y).ewm(span=n_ema, adjust=False).mean().to_numpy()
        sd = float(np.std(resid[-W:], ddof=1))
        return float(e) - p_now, sd if sd > 0 else 1e-4
    except Exception:
        return None


def var_forecast(panel_log, market, h, p_now):
    try:
        cols = [panel_log[m][-1200:] for m in MK]
        n = min(len(c) for c in cols)
        Y = np.column_stack([c[-n:] for c in cols])[: n - h - 1]
        if len(Y) < 200:
            return None
        Yd = np.diff(Y, axis=0)
        X, T = Yd[:-1], Yd[1:]
        A, *_ = np.linalg.lstsq(X, T, rcond=None)
        last = Yd[-1]
        for _ in range(h):
            last = last @ A
        mi = MK.index(market)
        sd = float(np.std(T[:, mi] - X @ A, ddof=1))
        return float(Y[-1, mi] + last[mi]) - p_now, sd if sd > 0 else 1e-4
    except Exception:
        return None


def full_eval(lp, t0, panel_log, use_var, fast, step=None):
    CC = {h: c for h, c in ((h, build(lp, h, t0, step)) for h in HS) if c is not None and c["tr"].sum() >= 30}
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
    out = {nm: {} for nm in ["A15"] + LIB}
    for h, c in sorted(CC.items()):
        te = c["te"]
        if te.sum() < 20:
            continue
        y = c["y"][te]; mu = c["mu"][te]; sdd = c["sdd"][te]; xi = c["xi"][te]
        G = np.tile(GV, (len(y), 1))
        S = sdd * h ** Hv
        lam = h0 / (h0 + h)
        w = min(1.0, h / HSAT_A15)
        out["A15"][h] = float(crps_q(y, mu[:, None] * lam + S[:, None] * ((1 - w) * xi + w * G)).mean())
        res = {nm: [] for nm in LIB}
        for t in c["ts"][te]:
            hist = lp[:t]
            if len(hist) < 200:
                for nm in LIB:
                    res[nm].append((np.nan, np.nan))
                continue
            def add(fn, *a):
                try:
                    v = fn(*a)
                except Exception:
                    v = None
                return v if (isinstance(v, tuple) and len(v) == 2) else (np.nan, np.nan)
            p_now = float(lp[t])
            res["ARIMA"].append(add(arima_forecast, hist, h, fast, p_now))
            res["GARCH"].append(add(garch_forecast, hist, h))
            res["AR"].append(add(ar_forecast, hist, h, p_now))
            res["EMA"].append(add(ema_forecast, hist, h, p_now))
            res["VAR"].append(add(var_forecast, panel_log, MARKET, h, p_now) if use_var else (np.nan, np.nan))
        for nm in LIB:
            src = res["GARCH"] if nm == "GARCH_H" else res[nm]
            keep = [i for i, (a, b) in enumerate(src)
                    if np.isscalar(a) and np.isscalar(b) and a == a and b == b and b > 0]
            if len(keep) < 0.5 * len(y):
                out[nm][h] = np.nan
                continue
            muv = np.array([src[i][0] for i in keep])
            sdv = np.array([src[i][1] for i in keep])
            yy = y[keep]
            if nm == "GARCH_H":
                sdv = sdd[keep] * h ** Hv     # 嫁接 h^H 标度，保留 GARCH 的均值
            Q = muv[:, None] + sdv[:, None] * G
            out[nm][h] = float(crps_q(yy, Q).mean())
    return out


PANEL = {}
T0S = {}
for m in MK:
    d = p[p.market == m].sort_values("date").reset_index(drop=True)
    PANEL[m] = np.log(d.price.to_numpy(float)).astype(float)
    T0S[m] = int(np.searchsorted(pd.to_datetime(d.date).to_numpy(), np.datetime64(CUT[m])))

print("=" * 112)
print("真实数据：文献标准基准（ARIMA 训练段 AIC 选阶 / GARCH(1,1)-t / AR(5) / EMA(20) / VAR(1)）")
print("=" * 112)
REAL = {}
for m in MK:
    MARKET = m
    REAL[m] = full_eval(PANEL[m], T0S[m], PANEL, True, False)
    row = f"  {MN[m]:<6}"
    for nm in ["A15"] + LIB:
        v = [x for x in REAL[m][nm].values() if x == x]
        row += f"  {nm} {np.mean(v):.5f}" if v else f"  {nm} -"
    print(row)

print("\n" + "=" * 112)
print(f"零分布：打乱序列 × {NSHUF}（固定 ARIMA(1,1,1)、原点抽稀 1/4）")
print("=" * 112)
T0 = time.time()
NULL = []
for it in range(NSHUF):
    rng = np.random.default_rng(SEED0 + it)
    rec = {}
    for m in MK:
        MARKET = m
        lp = PANEL[m]; inc = np.diff(lp)
        lp_s = np.concatenate([[0.0], np.cumsum(inc[rng.permutation(len(inc))])])
        pan_s = dict(PANEL); pan_s[m] = lp_s
        r = full_eval(lp_s, T0S[m], pan_s, False, True, step=8)
        if r:
            for nm in r:
                for h in r[nm]:
                    rec[(nm, h, m)] = r[nm][h]
    NULL.append(rec)
    print(f"  第 {it+1:>3} 次  ({time.time()-T0:.0f} 秒)")

print("\n" + "=" * 112)
print("最终证据表：A15 vs 文献标准基准（合并校准检验）")
print("=" * 112)
print(f"{'对照':<10}{'真实合并log比':>15}{'零分布均值':>13}{'零分布SD':>11}{'单尾p':>9}{'百分位':>9}")
OUT = {}
for b in LIB:
    real = [np.log(REAL[m]["A15"][h] / REAL[m][b][h]) for m in MK for h in HS
            if h in REAL[m]["A15"] and h in REAL[m][b]
            and REAL[m][b][h] == REAL[m][b][h] and REAL[m]["A15"][h] > 0]
    if not real:
        print(f"{b:<10} (数据不足)"); continue
    mu_real = float(np.mean(real))
    nulls = []
    for rec in NULL:
        vals = [np.log(rec[("A15", h, m)] / rec[(b, h, m)]) for m in MK for h in HS
                if ("A15", h, m) in rec and (b, h, m) in rec
                and rec[("A15", h, m)] == rec[("A15", h, m)] and rec[(b, h, m)] == rec[(b, h, m)]
                and rec[(b, h, m)] > 0]
        if len(vals) == len(real):
            nulls.append(float(np.mean(vals)))
    nulls = np.array(nulls)
    p1 = float(np.mean(nulls <= mu_real)) if len(nulls) else np.nan
    pct = float(np.mean(nulls < mu_real) * 100) if len(nulls) else np.nan
    print(f"{b:<10}{mu_real:>15.5f}{nulls.mean() if len(nulls) else np.nan:>13.5f}"
          f"{nulls.std() if len(nulls) else np.nan:>11.5f}{p1:>9.4f}{pct:>8.1f}%")
    OUT[b] = dict(real_mu=mu_real, p=p1, pct=pct, n=len(real),
                  null_mean=float(nulls.mean()) if len(nulls) else np.nan,
                  null_sd=float(nulls.std()) if len(nulls) else np.nan)

print("\n逐市场 p 值 与 逐格 CRPS：")
for b in LIB:
    if b not in OUT:
        continue
    row = f"  vs {b:<8}"
    for m in MK:
        rv = [np.log(REAL[m]["A15"][h] / REAL[m][b][h]) for h in HS
              if h in REAL[m]["A15"] and h in REAL[m][b]
              and REAL[m][b][h] == REAL[m][b][h] and REAL[m]["A15"][h] > 0]
        if not rv:
            row += f" {MN[m]}: -    "; continue
        mur = float(np.mean(rv))
        nn = []
        for rec in NULL:
            vals = [np.log(rec[("A15", h, m)] / rec[(b, h, m)]) for h in HS
                    if ("A15", h, m) in rec and (b, h, m) in rec
                    and rec[("A15", h, m)] == rec[("A15", h, m)] and rec[(b, h, m)] == rec[(b, h, m)]
                    and rec[(b, h, m)] > 0]
            if len(vals) == len(rv):
                nn.append(float(np.mean(vals)))
        row += f" {MN[m]}:{float(np.mean(np.array(nn)<=mur)):.3f}"
    print(row)

json.dump({"out": OUT, "real": {m: {nm: {str(h): REAL[m][nm][h] for h in REAL[m][nm]}
                                      for nm in REAL[m]} for m in MK}},
          open(os.path.join(DIR, "LIT_BENCH.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\n已写出 LIT_BENCH.json")
