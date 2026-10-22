"""Step 5: local-anchor mean reversion.

Step 4 showed static AR(p) helps EU and Hubei but destroys Shanghai, because a
static AR reverts toward the TRAINING-period mean. Shanghai's test window sits
at a different level, so the model fights the data.

The fix is structural, not a knob: revert toward a causally estimated LOCAL
level instead of a global constant.

    z_hat(t+h) = anchor_t + phi^h * (z_t - anchor_t)
    anchor_t   = EWMA_span(z up to t)        (slow, causal)
    phi        = AR(1) coefficient from train deviations

phi = 1 reproduces persistence exactly, phi = 0 reproduces an anchor forecast,
so this nests both. Nothing is selected on the test window: span and phi are
fixed a priori or estimated on train.
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
SPANS = (120, 250)


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


def ewma_vec(x, span):
    a = 2.0 / (span + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


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

    for span in SPANS:
        anchor = ewma_vec(z, span)
        dev = z - anchor
        # phi from train deviations only
        x, t = dev[:-1][:cut - 1], dev[1:][:cut - 1]
        A = np.c_[np.ones(x.size), x]
        beta = np.linalg.lstsq(A, t, rcond=None)[0]
        phi = float(beta[1])
        print("%-10s span=%3d  phi=%.6f  (phi=1 -> persistence)" % (mkt, span, phi))
        for h in HORIZONS:
            yh, yp = [], []
            for tt in range(cut, len(z) - h):
                yh.append(np.exp(anchor[tt] + (phi ** h) * (z[tt] - anchor[tt])))
                yp.append(y[tt])
            tgt = y[cut + h: len(z)]
            ma, mp = met(tgt, np.asarray(yh)), met(tgt, np.asarray(yp))
            if ma is None:
                continue
            rows.append(dict(market=mkt, span=span, phi=phi, h=h, n=ma["n"],
                             MAE=ma["MAE"], RMSE=ma["RMSE"], MAPE=ma["MAPE"],
                             R2=ma["R2"], MAE_pers=mp["MAE"], RMSE_pers=mp["RMSE"],
                             dMAE=(ma["MAE"] / mp["MAE"] - 1) * 100,
                             dRMSE=(ma["RMSE"] / mp["RMSE"] - 1) * 100))

res = pd.DataFrame(rows)
print("\n--- local-anchor mean reversion vs persistence (held-out 15%) ---")
print("%-10s %5s %4s %6s %10s %10s %9s %10s %10s %9s %8s"
      % ("market", "span", "h", "n", "MAE", "MAE_pers", "dMAE%", "RMSE",
         "RMSE_pers", "dRMSE%", "R2"))
for mkt in res["market"].unique():
    for _, r in res[res["market"] == mkt].sort_values(["h", "span"]).iterrows():
        print("%-10s %5d %4d %6d %10.4f %10.4f %+9.2f %10.4f %10.4f %+9.2f %8.4f"
              % (mkt, r["span"], r["h"], r["n"], r["MAE"], r["MAE_pers"], r["dMAE"],
                 r["RMSE"], r["RMSE_pers"], r["dRMSE"], r["R2"]))
    print()

print("--- summary: mean dMAE% across markets, by span and horizon ---")
for span in SPANS:
    s = res[res["span"] == span]
    line = "  span=%3d " % span
    for h in HORIZONS:
        v = s[s["h"] == h]["dMAE"]
        line += "  h%-2d %+7.2f%%" % (h, v.mean())
    print(line)
print("\n--- worst market per (span,h), MAE degradation ---")
for span in SPANS:
    for h in HORIZONS:
        v = res[(res["span"] == span) & (res["h"] == h)]
        w = v.loc[v["dMAE"].idxmax()]
        print("  span=%3d h%-2d  worst=%s %+.2f%%" % (span, h, w["market"], w["dMAE"]))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "STEP5_LOCAL_ANCHOR.csv")
res.to_csv(out, index=False, encoding="utf-8-sig")
print("\nwrote %s" % out)
