"""Step 12: can exogenous information beat persistence at the compliance horizon?

Every price-history model has lost to persistence (steps 2, 4, 6, 8, 11). The only
remaining source of gain is information persistence cannot see. The strongest
available daily exogenous variable with a real mechanism is local temperature:

    heating / cooling demand -> power load -> coal burn -> emissions
        -> allowance demand -> permit price

That chain operates over weeks, not days, so the honest place to test it is the
multi-month horizon where persistence is weak in absolute terms (Hubei MAE rises
from 0.63 at h=1 to 2.49 at h=22) and where this literature reports nothing,
because it only ever forecasts one step.

Design:
  target    z_{t+h} - z_t          (log price change over h days)
  features  causal temperature summaries for the pilot's own city,
            EUA momentum from the aligned panel, and own-price momentum
  model     ridge on an expanding window, refit every REFIT origins
  protocol  rolling origin, no look-ahead at any point
  reference last-value persistence

If the exogenous correction does not beat persistence here, then the exogenous
route is closed for this data too and that is worth knowing with a number.
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
    p = os.path.join(EXOG, "meteo_%s.csv" % name)
    d = pd.read_csv(p, skiprows=3)
    d = d.rename(columns={d.columns[0]: "date",
                          d.columns[1]: "tmean",
                          d.columns[2]: "tmax",
                          d.columns[3]: "tmin",
                          d.columns[4]: "precip"})
    d["date"] = pd.to_datetime(d["date"])
    for c in ("tmean", "tmax", "tmin", "precip"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.set_index("date").sort_index()


def weather_features(w, idx):
    """Causal summaries of the weather series, evaluated on the price index."""
    ww = w.reindex(idx, method="ffill")
    out = pd.DataFrame(index=idx)
    t = ww["tmean"]
    out["t_ma7"] = t.rolling(7, min_periods=7).mean()
    out["t_ma30"] = t.rolling(30, min_periods=30).mean()
    out["t_d7"] = t - t.shift(7)
    out["t_d30"] = t - t.shift(30)
    hdd = np.maximum(18.0 - t, 0.0)
    cdd = np.maximum(t - 22.0, 0.0)
    out["hdd30"] = hdd.rolling(30, min_periods=30).sum()
    out["cdd30"] = cdd.rolling(30, min_periods=30).sum()
    out["hdd30_d"] = out["hdd30"] - out["hdd30"].shift(30)
    out["cdd30_d"] = out["cdd30"] - out["cdd30"].shift(30)
    out["t_rng"] = (ww["tmax"] - ww["tmin"]).rolling(7, min_periods=7).mean()
    out["precip30"] = ww["precip"].rolling(30, min_periods=30).sum()
    return out


def nw_t(d, lag):
    d = np.asarray(d, float)
    n = d.size
    m = d.mean()
    e = d - m
    v = np.dot(e, e) / n
    for L in range(1, lag + 1):
        v += 2.0 * (1.0 - L / (lag + 1.0)) * np.dot(e[L:], e[:-L]) / n
    se = np.sqrt(v / n)
    return m, se, (m / se if se > 0 else np.nan)


panel = pd.read_csv(PANEL, parse_dates=["date"])
wide = (panel.pivot_table(index="date", columns="market", values="price")
        .sort_index())
wide = wide[wide.index >= START]

rows = []
for mkt in ("HUBEI", "SHANGHAI", "GUANGZHOU"):
    sub = wide[[mkt, "EU"]].copy()
    sub = sub.dropna(subset=[mkt])
    idx = sub.index
    y = sub[mkt].to_numpy(float)
    z = np.log(y)
    n = z.size

    F = pd.DataFrame(index=idx)
    w = load_meteo(CITY[mkt])
    W = weather_features(w, idx)
    for c in W.columns:
        F["%s_%s" % (CITY[mkt], c)] = W[c].to_numpy()

    eu = np.log(sub["EU"].astype(float))
    F["eu_d1"] = (eu - eu.shift(1)).to_numpy()
    F["eu_d5"] = (eu - eu.shift(5)).to_numpy()
    F["eu_d22"] = (eu - eu.shift(22)).to_numpy()
    zs = pd.Series(z, index=idx)
    F["own_d5"] = (zs - zs.shift(5)).to_numpy()
    F["own_d22"] = (zs - zs.shift(22)).to_numpy()

    Fv = F.to_numpy(float)
    t0 = int(n * ORIGIN_FRAC)

    for h in HORIZONS:
        nres = n - h
        tgt = z[h:] - z[:nres]                 # delta target per origin
        ae_ex, ae_p, betas = [], [], []
        preds, actuals = [], []
        use = np.arange(h, max(t0, 60), 1)     # rows usable for fitting
        for t in range(t0, n - h):
            # refit periodically on everything known at the last refit point
            if (t - t0) % REFIT == 0:
                fit_rows = np.arange(60, t - h + 1) if t - h + 1 > 60 else None
                if fit_rows is None or fit_rows.size < 120:
                    pred = 0.0
                else:
                    Xs = Fv[fit_rows]
                    ys = tgt[fit_rows]
                    ok = np.isfinite(Xs).all(1) & np.isfinite(ys)
                    Xs, ys = Xs[ok], ys[ok]
                    if Xs.shape[0] < 120:
                        pred = 0.0
                    else:
                        mu, sd = Xs.mean(0), Xs.std(0) + 1e-9
                        Xn = (Xs - mu) / sd
                        A = np.c_[np.ones(len(Xn)), Xn]
                        lam = 1.0
                        beta = np.linalg.solve(
                            A.T @ A + lam * np.eye(A.shape[1]), A.T @ ys)
                        cur = (mu, sd, beta)
                        pred = None
            if pred is None or not isinstance(pred, float):
                mu, sd, beta = cur
                xt = Fv[t]
                if not np.isfinite(xt).all():
                    d_hat = 0.0
                else:
                    d_hat = float(np.r_[1.0, (xt - mu) / sd] @ beta)
            else:
                d_hat = 0.0
            yh = np.exp(z[t] + d_hat)
            ae_ex.append(abs(y[t + h] - yh))
            ae_p.append(abs(y[t + h] - y[t]))
            preds.append(yh)
            actuals.append(y[t + h])

        ae_ex, ae_p = np.asarray(ae_ex), np.asarray(ae_p)
        d = ae_ex - ae_p
        m, se, tt = nw_t(d, lag=max(h - 1, 0))
        rows.append(dict(market=mkt, h=h, n=d.size,
                         MAE_exog=ae_ex.mean(), MAE_pers=ae_p.mean(),
                         dMAE=(ae_ex.mean() / ae_p.mean() - 1) * 100,
                         diff=m, se=se, t=tt))
    print("done %s" % mkt)

res = pd.DataFrame(rows)
print("\n--- exogenous (weather + EUA + own momentum) vs persistence ---")
print("%-10s %3s %5s %12s %12s %9s %11s %8s"
      % ("market", "h", "n", "MAE_exog", "MAE_pers", "dMAE%", "diff", "NW_t"))
for _, r in res.iterrows():
    print("%-10s %3d %5d %12.4f %12.4f %+9.2f %+11.5f %+8.2f"
          % (r["market"], r["h"], r["n"], r["MAE_exog"], r["MAE_pers"],
             r["dMAE"], r["diff"], r["t"]))

print("\nbeats persistence on %d/%d (market,h) cells; significant at 5%% on %d"
      % (int((res["dMAE"] < 0).sum()), len(res),
         int(((res["dMAE"] < 0) & (res["t"] < -1.96)).sum())))

res.to_csv(os.path.join(BASE, "STEP12_EXOGENOUS.csv"), index=False, encoding="utf-8-sig")
print("\nwrote STEP12_EXOGENOUS.csv")
