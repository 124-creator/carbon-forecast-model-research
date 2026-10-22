"""Step 3: is there usable lead-lag information from the EU carbon market into
the Chinese pilots?

The mechanism is clock ordering, not statistics: Chinese pilots close in the
Asian afternoon; EUA closes later the same calendar day. So EUA's move on day t
is fully known before a Chinese pilot trades on day t+1. If the carbon markets
are linked, that is a genuine, exploitable input that last-value persistence
cannot use.

Test is deliberately narrow: does EUA's day-t change predict a Chinese pilot's
day-t+1 change, out of sample, and does using it reduce MAE below persistence?
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

wide = (pd.read_csv(PANEL, parse_dates=["date"])
        .pivot_table(index="date", columns="market", values="price")
        .sort_index())
wide = wide[wide.index >= START]
print("aligned rows: %d   %s -> %s" % (len(wide), wide.index.min().date(), wide.index.max().date()))
print("non-null per market:\n%s\n" % wide.notna().sum().to_string())

lg = np.log(wide)
d = lg.diff()


def met(y, yhat):
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    r = y - yhat
    return (float(np.mean(np.abs(r))), float(np.sqrt(np.mean(r ** 2))),
            1.0 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2), int(y.size))


print("--- correlation of daily log change ---")
print("%-10s %14s %16s %16s" % ("china", "same-day EU", "EU lag1->china", "own lag1"))
for c in ("HUBEI", "SHANGHAI", "GUANGZHOU"):
    if c not in d:
        continue
    x = d[[c, "EU"]].dropna()
    same = x[c].corr(x["EU"])
    lead = x[c].corr(x["EU"].shift(1))
    own = x[c].corr(x[c].shift(1))
    print("%-10s %14.4f %16.4f %16.4f" % (c, same, lead, own))

print("\n--- out-of-sample: persistence vs persistence + EUA lead ---")
print("%-10s %8s %10s %10s %8s %10s %10s %8s %10s"
      % ("china", "n_test", "MAE_pers", "MAE_eu", "dMAE%", "RMSE_pers", "RMSE_eu",
         "dRMSE%", "beta"))
for c in ("HUBEI", "SHANGHAI", "GUANGZHOU"):
    if c not in d:
        continue
    sub = wide[[c, "EU"]].copy()
    sub["dy"] = lg[c].diff()
    sub["eu_lag"] = lg["EU"].diff().shift(1)   # EU change from the previous close
    sub = sub.dropna()
    n = len(sub)
    cut = int(n * TRAIN_FRAC)
    # fit beta on train only
    x = sub["eu_lag"].to_numpy()[:cut]
    yy = sub["dy"].to_numpy()[:cut]
    A = np.c_[np.ones(cut), x]
    beta = np.linalg.lstsq(A, yy, rcond=None)[0]
    # predict level: yhat_t = y_{t-1} * exp(a + b*EU_{t-1})
    y_lvl = sub[c].to_numpy()
    y_prev = np.r_[np.nan, y_lvl[:-1]]
    eu = sub["eu_lag"].to_numpy()
    yh_p = y_prev
    yh_e = y_prev * np.exp(beta[0] + beta[1] * eu)
    sl = slice(cut, n)
    ma_p, mr_p, r2_p, nn = met(y_lvl[sl], yh_p[sl])
    ma_e, mr_e, r2_e, _ = met(y_lvl[sl], yh_e[sl])
    print("%-10s %8d %10.4f %10.4f %+8.2f %10.4f %10.4f %+8.2f %10.5f"
          % (c, nn, ma_p, ma_e, (ma_e / ma_p - 1) * 100, mr_p, mr_e,
             (mr_e / mr_p - 1) * 100, beta[1]))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "STEP3_WIDE.csv")
wide.to_csv(out, encoding="utf-8-sig")
print("\nwrote %s" % out)
