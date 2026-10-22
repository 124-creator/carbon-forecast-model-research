"""Step 4: does persistence stop being optimal as the horizon grows?

Step 3 found first-order autocorrelation of daily log changes around -0.13 to
-0.19 in all three pilots: mean reversion. If that is real, an AR(p) on the log
level should beat last-value persistence increasingly with horizon, and the
one-step-only design used throughout this literature would be hiding it.

Metric set is the field's (MSE/RMSE/MAE/MAPE/R2). Fit on train only, forecast
h steps ahead by iterating the fitted recursion, score on the held-out 15%.
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
HORIZONS = (1, 5, 10, 22)


def met(y, yhat):
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    if y.size < 20:
        return None
    r = y - yhat
    return dict(n=int(y.size), MSE=float(np.mean(r ** 2)),
                RMSE=float(np.sqrt(np.mean(r ** 2))),
                MAE=float(np.mean(np.abs(r))),
                MAPE=float(np.mean(np.abs(r / y)) * 100.0),
                R2=float(1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)))


def fit_ar(z, p):
    """OLS AR(p) on a level series z, with intercept."""
    X, t = [], []
    for i in range(p, len(z)):
        X.append(z[i - p:i][::-1])
        t.append(z[i])
    X, t = np.asarray(X), np.asarray(t)
    A = np.c_[np.ones(len(X)), X]
    beta = np.linalg.lstsq(A, t, rcond=None)[0]
    return beta


def forecast_ar(z_hist, beta, h):
    p = len(beta) - 1
    buf = list(z_hist[-p:])
    for _ in range(h):
        nxt = beta[0] + np.dot(beta[1:], np.asarray(buf[-p:])[::-1])
        buf.append(nxt)
    return buf[-1]


df = pd.read_csv(PANEL, parse_dates=["date"])
rows = []
for mkt, g in df.groupby("market"):
    g = g.sort_values("date")
    g = g[g["date"] >= START].reset_index(drop=True)
    y = g["price"].to_numpy(float)
    if y.size < 300:
        continue
    z = np.log(y)
    cut = int(len(z) * TRAIN_FRAC)

    # implied persistence / half-life from the train-period AR(1)
    b1 = fit_ar(z[:cut], 1)
    phi = b1[1]
    hl = np.log(0.5) / np.log(phi) if 0 < phi < 1 else np.nan
    print("%-10s train AR(1) phi=%.6f  half_life=%.1f days" % (mkt, phi, hl))

    for p in (1, 2):
        beta = fit_ar(z[:cut], p)
        for h in HORIZONS:
            yh_ar, yh_p = [], []
            for t in range(cut, len(z) - h):
                yh_ar.append(np.exp(forecast_ar(z[:t + 1], beta, h)))
                yh_p.append(y[t])                      # persistence: last value
            tgt = y[cut + h: len(z)]
            ma = met(tgt, np.asarray(yh_ar))
            mp = met(tgt, np.asarray(yh_p))
            if ma is None:
                continue
            rows.append(dict(market=mkt, model="AR%d" % p, h=h,
                             MAE=ma["MAE"], RMSE=ma["RMSE"], MAPE=ma["MAPE"],
                             R2=ma["R2"], n=ma["n"],
                             MAE_pers=mp["MAE"], RMSE_pers=mp["RMSE"],
                             R2_pers=mp["R2"],
                             dMAE=(ma["MAE"] / mp["MAE"] - 1) * 100,
                             dRMSE=(ma["RMSE"] / mp["RMSE"] - 1) * 100))

res = pd.DataFrame(rows)
print("\n--- AR(p) vs persistence, h-step ahead, held-out 15% ---")
print("%-10s %-5s %4s %6s %10s %10s %9s %10s %10s %9s"
      % ("market", "model", "h", "n", "MAE_AR", "MAE_pers", "dMAE%", "RMSE_AR",
         "RMSE_pers", "dRMSE%"))
for mkt in res["market"].unique():
    for _, r in res[res["market"] == mkt].sort_values(["h", "model"]).iterrows():
        print("%-10s %-5s %4d %6d %10.4f %10.4f %+9.2f %10.4f %10.4f %+9.2f"
              % (mkt, r["model"], r["h"], r["n"], r["MAE"], r["MAE_pers"],
                 r["dMAE"], r["RMSE"], r["RMSE_pers"], r["dRMSE"]))
    print()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "STEP4_MULTISTEP.csv")
res.to_csv(out, index=False, encoding="utf-8-sig")
print("wrote %s" % out)
