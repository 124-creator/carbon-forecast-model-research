"""Step 13: parsimonious exogenous variants, so the negative is not an artefact of
having thrown 15 features at a near-random-walk.

Step 12 stacked 15 standardised regressors and lost significantly on 9/9 cells.
The obvious objection is over-parameterisation. Three narrow models answer it:

  P2     only hdd30 and cdd30 - two numbers, the whole demand mechanism
  P1     the single exogenous feature with the largest in-sample correlation with
         the target, chosen at each refit using ONLY training rows
  SEAS   11 month dummies. Compliance-cycle seasonality is the one piece of
         genuinely pre-known information: it is on the calendar at forecast time,
         unlike realised weather, so it is the fairest possible test of "is there
         any exploitable structure that persistence cannot see".

Same rolling-origin protocol, same reference.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
PANEL = (r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research"
         r"\research\fullcycle_h20_20261020\PANEL_4MARKET.csv")
EXOG = os.path.join(BASE, "exog")
START = "2014-04-04"
ORIGIN_FRAC = 0.60
HORIZONS = (1, 5, 22)
REFIT = 22
CITY = {"HUBEI": "wuhan", "SHANGHAI": "shanghai", "GUANGZHOU": "guangzhou"}


def load_meteo(name):
    d = pd.read_csv(os.path.join(EXOG, "meteo_%s.csv" % name), skiprows=3)
    d = d.rename(columns={d.columns[0]: "date", d.columns[1]: "tmean",
                          d.columns[2]: "tmax", d.columns[3]: "tmin",
                          d.columns[4]: "precip"})
    d["date"] = pd.to_datetime(d["date"])
    for c in ("tmean", "tmax", "tmin", "precip"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.set_index("date").sort_index()


def nw_t(d, lag):
    d = np.asarray(d, float)
    n = d.size
    e = d - d.mean()
    v = np.dot(e, e) / n
    for L in range(1, lag + 1):
        v += 2.0 * (1.0 - L / (lag + 1.0)) * np.dot(e[L:], e[:-L]) / n
    se = np.sqrt(v / n)
    return d.mean(), se, (d.mean() / se if se > 0 else np.nan)


def ols(A, y, lam=1.0):
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)


panel = pd.read_csv(PANEL, parse_dates=["date"])
wide = (panel.pivot_table(index="date", columns="market", values="price")
        .sort_index())
wide = wide[wide.index >= START]

rows = []
for mkt in ("HUBEI", "SHANGHAI", "GUANGZHOU"):
    sub = wide[[mkt, "EU"]].dropna(subset=[mkt])
    idx = sub.index
    y = sub[mkt].to_numpy(float)
    z = np.log(y)
    n = z.size

    w = load_meteo(CITY[mkt]).reindex(idx, method="ffill")
    t = w["tmean"]
    hdd30 = np.maximum(18.0 - t, 0.0).rolling(30, min_periods=30).sum().to_numpy()
    cdd30 = np.maximum(t - 22.0, 0.0).rolling(30, min_periods=30).sum().to_numpy()

    eu = np.log(sub["EU"].astype(float))
    zs = pd.Series(z, index=idx)
    exog = pd.DataFrame(index=idx)
    exog["hdd30"] = hdd30
    exog["cdd30"] = cdd30
    exog["t_d30"] = (t - t.shift(30)).to_numpy()
    exog["eu_d22"] = (eu - eu.shift(22)).to_numpy()
    exog["own_d22"] = (zs - zs.shift(22)).to_numpy()
    months = pd.get_dummies(idx.month, prefix="m", drop_first=True).to_numpy(float)

    EV = exog.to_numpy(float)
    t0 = int(n * ORIGIN_FRAC)

    for h in HORIZONS:
        nres = n - h
        tgt = z[h:] - z[:nres]
        out = {k: [] for k in ("P2", "P1", "SEAS", "PERS")}
        for tt in range(t0, n - h):
            fit = np.arange(60, tt - h + 1)
            if fit.size < 150:
                for k in out:
                    out[k].append(abs(y[tt + h] - y[tt]))
                continue
            ys = tgt[fit]
            ok = np.isfinite(ys)
            fit, ys = fit[ok], ys[ok]

            def score(cols_list, current, lam=1.0):
                Xs = cols_list[fit]
                okx = np.isfinite(Xs).all(1)
                Xf, yf = Xs[okx], ys[okx]
                if Xf.shape[0] < 120:
                    return 0.0
                mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
                beta = ols(np.c_[np.ones(len(Xf)), (Xf - mu) / sd], yf, lam)
                xt = np.asarray(current, float).ravel()
                if not np.isfinite(xt).all():
                    return 0.0
                return float(np.r_[1.0, (xt - mu) / sd] @ beta)

            # P2: the two demand numbers only
            d2 = score(EV[:, :2], EV[tt, :2])
            # P1: best single exogenous feature by in-sample |corr|
            best, bestd = -1.0, 0.0
            for j in range(EV.shape[1]):
                Xs = EV[fit, j]
                okj = np.isfinite(Xs)
                if okj.sum() < 120:
                    continue
                r = np.corrcoef(Xs[okj], ys[okj])[0, 1]
                if np.isfinite(r) and abs(r) > best:
                    best = abs(r)
                    cand = score(EV[:, [j]], EV[tt, j:j + 1])
                    bestd = cand
            # SEAS: calendar-only
            dS = score(months, months[tt:tt + 1])

            for k, dv in (("P2", d2), ("P1", bestd), ("SEAS", dS)):
                out[k].append(abs(y[tt + h] - np.exp(z[tt] + dv)))
            out["PERS"].append(abs(y[tt + h] - y[tt]))

        ap = np.asarray(out["PERS"])
        for k in ("P2", "P1", "SEAS"):
            ae = np.asarray(out[k])
            m, se, tv = nw_t(ae - ap, lag=max(h - 1, 0))
            rows.append(dict(market=mkt, h=h, model=k, n=ae.size,
                             MAE=ae.mean(), MAE_pers=ap.mean(),
                             dMAE=(ae.mean() / ap.mean() - 1) * 100, t=tv))
    print("done %s" % mkt)

res = pd.DataFrame(rows)
print("\n--- parsimonious exogenous variants vs persistence ---")
print("%-10s %3s %-5s %5s %12s %9s %8s"
      % ("market", "h", "model", "n", "MAE", "dMAE%", "NW_t"))
for _, r in res.iterrows():
    print("%-10s %3d %-5s %5d %12.4f %+9.2f %+8.2f"
          % (r["market"], r["h"], r["model"], r["n"], r["MAE"], r["dMAE"], r["t"]))

print("\nbeats persistence on %d/%d cells; significant at 5%% on %d"
      % (int((res["dMAE"] < 0).sum()), len(res),
         int(((res["dMAE"] < 0) & (res["t"] < -1.96)).sum())))
print("best cell: %s h=%d %s %+.2f%% (t=%+.2f)"
      % (res.loc[res["dMAE"].idxmin(), "market"], res.loc[res["dMAE"].idxmin(), "h"],
         res.loc[res["dMAE"].idxmin(), "model"], res["dMAE"].min(),
         res.loc[res["dMAE"].idxmin(), "t"]))

res.to_csv(os.path.join(BASE, "STEP13_PARSIMONY.csv"), index=False, encoding="utf-8-sig")
print("\nwrote STEP13_PARSIMONY.csv")
