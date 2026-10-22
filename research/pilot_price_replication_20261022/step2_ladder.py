"""Step 2: the benchmark ladder on the pilot-price protocol.

Question this answers: is there any headroom above last-value persistence using
the price history ALONE (no exogenous inputs)? If ARIMA / ridge-on-lags cannot
beat persistence, then the only route to a better forecast is information that
persistence does not use, and the model contribution has to live there.

Protocol (Yi 2025 shape): daily close, first 85% train / last 15% test,
metrics MSE / RMSE / MAE / MAPE / R2, one-step ahead.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PANEL = (r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research"
         r"\research\fullcycle_h20_20261020\PANEL_4MARKET.csv")
START = "2014-04-04"
TRAIN_FRAC = 0.85


def metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    res = y - yhat
    return {
        "n": int(y.size),
        "MSE": float(np.mean(res ** 2)),
        "RMSE": float(np.sqrt(np.mean(res ** 2))),
        "MAE": float(np.mean(np.abs(res))),
        "MAPE_pct": float(np.mean(np.abs(res / y)) * 100.0),
        "R2": float(1.0 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2)),
    }


def fit_ridge_lags(y, cut, lags):
    """Predict the CHANGE, then add it back to the last observed level."""
    dy = np.diff(y)
    X, t = [], []
    for i in range(lags, len(dy)):
        X.append(dy[i - lags:i][::-1])
        t.append(dy[i])
    X, t = np.asarray(X), np.asarray(t)
    # row i of X/t corresponds to dy index (lags + i), i.e. level index (lags + i + 1)
    first_level = lags + 1
    tr = np.arange(X.shape[0]) + first_level < cut
    Xtr, ttr = X[tr], t[tr]
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-12
    Xtr = (Xtr - xm) / xs
    A = np.c_[np.ones(len(Xtr)), Xtr]
    beta = np.linalg.lstsq(A, ttr, rcond=None)[0]
    Xa = (X - xm) / xs
    dyhat = np.c_[np.ones(len(Xa)), Xa] @ beta
    yhat = np.full(len(y), np.nan)
    lvl_idx = np.arange(X.shape[0]) + first_level
    yhat[lvl_idx] = y[lvl_idx - 1] + dyhat
    return yhat


df = pd.read_csv(PANEL, parse_dates=["date"])
rows = []
for mkt, g in df.groupby("market"):
    g = g.sort_values("date")
    g = g[g["date"] >= START].reset_index(drop=True)
    y = g["price"].to_numpy(float)
    if y.size < 200:
        continue
    cut = int(len(y) * TRAIN_FRAC)
    te = np.arange(cut, len(y))

    cands = {}
    cands["persistence"] = np.r_[np.nan, y[:-1]]

    # random walk with drift, drift from train only
    dy_tr = np.diff(y[:cut])
    drift = dy_tr.mean()
    yhat = np.r_[np.nan, y[:-1]]
    off = np.arange(len(y)) - cut
    yhat = np.where(np.arange(len(y)) >= cut, yhat + drift * np.maximum(off, 1), np.nan)
    cands["rw_drift"] = yhat

    for p in (1, 5, 22):
        cands["ridge_lag%d" % p] = fit_ridge_lags(y, cut, p)

    # ARIMA on the level, order (1,1,1) with drift; refit once on train only
    try:
        from statsmodels.tsa.arima.model import ARIMA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ARIMA(y[:cut], order=(1, 1, 1), trend="t").fit()
            fc = fit.forecast(steps=len(te))
        yh = np.full(len(y), np.nan)
        yh[te] = fc
        cands["arima111"] = yh
    except Exception as e:                       # noqa: BLE001
        print("ARIMA failed on %s: %s" % (mkt, e))

    for nm, yh in cands.items():
        m = metrics(y[te], yh[te])
        rows.append(dict(market=mkt, model=nm, **m))

res = pd.DataFrame(rows)
for mkt in res["market"].unique():
    s = res[res["market"] == mkt].copy()
    base = s[s["model"] == "persistence"].iloc[0]
    s["MAE_vs_persist_pct"] = (s["MAE"] / base["MAE"] - 1.0) * 100.0
    s["RMSE_vs_persist_pct"] = (s["RMSE"] / base["RMSE"] - 1.0) * 100.0
    print("\n=== %s  (n_test=%d) ===" % (mkt, int(base["n"])))
    print("%-14s %10s %10s %8s %9s %10s %12s %12s"
          % ("model", "MSE", "RMSE", "MAE", "MAPE%", "R2", "dMAE%", "dRMSE%"))
    for _, r in s.sort_values("MAE").iterrows():
        print("%-14s %10.4f %10.4f %8.4f %9.4f %10.6f %+12.2f %+12.2f"
              % (r["model"], r["MSE"], r["RMSE"], r["MAE"], r["MAPE_pct"],
                 r["R2"], r["MAE_vs_persist_pct"], r["RMSE_vs_persist_pct"]))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "STEP2_LADDER.csv")
res.to_csv(out, index=False, encoding="utf-8-sig")
print("\nwrote %s" % out)
