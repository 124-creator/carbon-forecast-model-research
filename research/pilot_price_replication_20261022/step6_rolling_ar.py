"""Step 6: rolling-window AR, i.e. an adaptive reversion target with a proper
estimator instead of an EWMA anchor.

Step 5 failed because EWMA(z) is a noisy, lagging estimate of the local level, so
reverting toward it throws away real level information. The alternative is to
keep the AR form but re-estimate its coefficients on a trailing window, which
lets the implied long-run mean a/(1-phi) drift with the data.

Windows are fixed a priori (no selection on the test window). Expanding window is
included as the "no adaptation" control.
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
HORIZONS = (1, 5, 22)
WINDOWS = (250, 500, 0)          # 0 = expanding


def met(y, yhat):
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    if y.size < 20:
        return None
    r = y - yhat
    return dict(n=int(y.size), MAE=float(np.mean(np.abs(r))),
                RMSE=float(np.sqrt(np.mean(r ** 2))),
                MAPE=float(np.mean(np.abs(r / y)) * 100.0),
                R2=float(1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)))


def ar_fit(zw, p):
    X, t = [], []
    for i in range(p, len(zw)):
        X.append(zw[i - p:i][::-1])
        t.append(zw[i])
    X, t = np.asarray(X), np.asarray(t)
    A = np.c_[np.ones(len(X)), X]
    return np.linalg.lstsq(A, t, rcond=None)[0]


def ar_fc(zlast, beta, h):
    p = len(beta) - 1
    buf = list(zlast[-p:])
    for _ in range(h):
        buf.append(beta[0] + np.dot(beta[1:], np.asarray(buf[-p:])[::-1]))
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

    for W in WINDOWS:
        for p in (1, 2):
            for h in HORIZONS:
                yh, yp, ok_n = [], [], 0
                for tt in range(cut, len(z) - h):
                    lo = 0 if W == 0 else max(0, tt + 1 - W)
                    zw = z[lo:tt + 1]
                    if zw.size < p + 30:
                        yh.append(np.nan)
                        continue
                    beta = ar_fit(zw, p)
                    yh.append(np.exp(ar_fc(zw, beta, h)))
                    yp.append(y[tt])
                tgt = y[cut + h: len(z)][:len(yh)]
                ma, mp = met(tgt, np.asarray(yh)), met(tgt, np.asarray(yp))
                if ma is None:
                    continue
                rows.append(dict(market=mkt, window=W, order=p, h=h, n=ma["n"],
                                 MAE=ma["MAE"], RMSE=ma["RMSE"], R2=ma["R2"],
                                 MAE_pers=mp["MAE"], RMSE_pers=mp["RMSE"],
                                 dMAE=(ma["MAE"] / mp["MAE"] - 1) * 100,
                                 dRMSE=(ma["RMSE"] / mp["RMSE"] - 1) * 100))
        print("done %s window=%s" % (mkt, W if W else "expanding"))

res = pd.DataFrame(rows)
print("\n--- rolling AR vs persistence ---")
print("%-10s %8s %2s %3s %6s %10s %10s %9s %9s"
      % ("market", "window", "p", "h", "n", "MAE", "MAE_pers", "dMAE%", "dRMSE%"))
for mkt in res["market"].unique():
    for _, r in res[res["market"] == mkt].sort_values(["h", "window", "order"]).iterrows():
        print("%-10s %8s %2d %3d %6d %10.4f %10.4f %+9.2f %+9.2f"
              % (mkt, r["window"] if r["window"] else "expand", r["order"], r["h"],
                 r["n"], r["MAE"], r["MAE_pers"], r["dMAE"], r["dRMSE"]))
    print()

print("--- does any single fixed configuration win on ALL markets? ---")
print("%-8s %2s %3s %s" % ("window", "p", "h", "  ".join("%-10s" % m for m in res["market"].unique())))
for W in WINDOWS:
    for p in (1, 2):
        for h in HORIZONS:
            s = res[(res["window"] == W) & (res["order"] == p) & (res["h"] == h)]
            if s.empty:
                continue
            vals = []
            for m in res["market"].unique():
                v = s[s["market"] == m]["dMAE"]
                vals.append("%+9.2f%%" % v.iloc[0] if len(v) else "     n/a ")
            tag = "" if all(float(v.replace("%", "")) < 0 for v in vals) else "  <-not all"
            print("%-8s %2d %3d %s%s"
                  % (W if W else "expand", p, h, "  ".join(vals), tag))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "STEP6_ROLLING_AR.csv")
res.to_csv(out, index=False, encoding="utf-8-sig")
print("\nwrote %s" % out)
