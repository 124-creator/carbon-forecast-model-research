"""Step 7: level-shift-gated mean reversion.

Diagnosis from steps 4-6: reversion toward a past level helps EU / Hubei /
Guangzhou but destroys Shanghai, because Shanghai's held-out window sits at a
different level from its history and every AR variant reverts the wrong way.

So the model gates the reversion: revert only while the current level is still
consistent with its own recent history, otherwise fall back to persistence.

    mu_t   = level reference (train mean, or causal expanding mean)
    phi    = AR(1) coefficient from the training period
    gate   = |z_t - median(z_{t-W+1..t})| / (1.4826 * MAD) > THR
    z_hat  = mu_t + (phi_eff ** h) * (z_t - mu_t),  phi_eff = 1 if gate else phi

THR is fixed a priori by the standard robust-outlier convention, NOT tuned on the
held-out window; a sensitivity sweep is reported instead, and THR = inf is the
ungated control so the gate's own contribution is visible.

Shares the field's protocol: daily close, 85/15, MSE/RMSE/MAE/MAPE/R2.
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
W = 250
THRS = (2.0, 2.5, 3.0, 4.0, np.inf)


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


def robust_stat(z, t, w):
    lo = max(0, t + 1 - w)
    seg = z[lo:t + 1]
    if seg.size < 30:
        return 0.0
    med = np.median(seg)
    mad = np.median(np.abs(seg - med))
    scale = 1.4826 * mad
    if scale <= 0:
        return 0.0
    return (z[t] - med) / scale


df = pd.read_csv(PANEL, parse_dates=["date"])
rows = []
diag = []
for mkt, g in df.groupby("market"):
    g = g.sort_values("date")
    g = g[g["date"] >= START].reset_index(drop=True)
    y = g["price"].to_numpy(float)
    if y.size < 300:
        continue
    z = np.log(y)
    dates = g["date"].to_numpy()
    cut = int(len(z) * TRAIN_FRAC)

    # AR(1) on the level, train only
    x, t_ = z[:-1][:cut - 1], z[1:][:cut - 1]
    A = np.c_[np.ones(x.size), x]
    beta = np.linalg.lstsq(A, t_, rcond=None)[0]
    phi = float(beta[1])
    mu_train = float(z[:cut].mean())

    # gate statistic across the whole sample (causal)
    stat = np.array([robust_stat(z, i, W) for i in range(len(z))])
    diag.append(dict(market=mkt, phi=phi,
                     stat_train_max=float(np.abs(stat[:cut]).max()),
                     stat_test_max=float(np.abs(stat[cut:]).max()),
                     stat_test_med=float(np.median(np.abs(stat[cut:]))),
                     mu_train=mu_train, z_test_last=float(z[-1])))

    for mu_name in ("train_mean", "expanding"):
        for thr in THRS:
            for h in HORIZONS:
                yh, yp = [], []
                for tt in range(cut, len(z) - h):
                    mu = mu_train if mu_name == "train_mean" else float(z[:tt + 1].mean())
                    phi_eff = 1.0 if abs(stat[tt]) > thr else phi
                    yh.append(np.exp(mu + (phi_eff ** h) * (z[tt] - mu)))
                    yp.append(y[tt])
                tgt = y[cut + h: len(z)][:len(yh)]
                ma, mp = met(tgt, np.asarray(yh)), met(tgt, np.asarray(yp))
                if ma is None:
                    continue
                trig = float(np.mean([abs(stat[tt]) > thr
                                      for tt in range(cut, len(z) - h)]))
                rows.append(dict(market=mkt, mu=mu_name, thr=thr, h=h, n=ma["n"],
                                 trigger_rate=trig,
                                 MAE=ma["MAE"], RMSE=ma["RMSE"], MAPE=ma["MAPE"],
                                 R2=ma["R2"], MAE_pers=mp["MAE"],
                                 dMAE=(ma["MAE"] / mp["MAE"] - 1) * 100,
                                 dRMSE=(ma["RMSE"] / mp["RMSE"] - 1) * 100))

print("--- level-shift diagnostic ---")
print("%-10s %10s %14s %14s %14s" % ("market", "phi_AR1", "|stat|max_tr",
                                     "|stat|max_te", "|stat|med_te"))
for d in diag:
    print("%-10s %10.6f %14.2f %14.2f %14.2f"
          % (d["market"], d["phi"], d["stat_train_max"], d["stat_test_max"],
             d["stat_test_med"]))

res = pd.DataFrame(rows)
print("\n--- dMAE% vs persistence (negative = better) ---")
for mu_name in ("train_mean", "expanding"):
    print("\nmu = %s" % mu_name)
    hdr = "%-10s %6s" % ("market", "thr")
    for h in HORIZONS:
        hdr += " %9s" % ("h=%d" % h)
    print(hdr + "   trigger@h=22")
    for mkt in res["market"].unique():
        for thr in THRS:
            line = "%-10s %6s" % (mkt, "inf" if not np.isfinite(thr) else "%.1f" % thr)
            trg = np.nan
            for h in HORIZONS:
                v = res[(res["market"] == mkt) & (res["mu"] == mu_name) &
                        (res["thr"] == thr) & (res["h"] == h)]["dMAE"]
                line += " %+9.2f" % v.iloc[0] if len(v) else "       n/a"
                if h == 22 and len(v):
                    trg = res[(res["market"] == mkt) & (res["mu"] == mu_name) &
                              (res["thr"] == thr) & (res["h"] == 22)]["trigger_rate"].iloc[0]
            line += "   %5.1f%%" % (trg * 100) if np.isfinite(trg) else "     n/a"
            print(line)

print("\n--- does the gate beat the ungated control on EVERY market? ---")
for mu_name in ("train_mean", "expanding"):
    for h in HORIZONS:
        base = {}
        for mkt in res["market"].unique():
            v = res[(res["market"] == mkt) & (res["mu"] == mu_name) &
                    (res["thr"] == np.inf) & (res["h"] == h)]["dMAE"]
            if len(v):
                base[mkt] = v.iloc[0]
        for thr in THRS[:-1]:
            vals = {}
            for mkt in res["market"].unique():
                v = res[(res["market"] == mkt) & (res["mu"] == mu_name) &
                        (res["thr"] == thr) & (res["h"] == h)]["dMAE"]
                if len(v):
                    vals[mkt] = v.iloc[0]
            allneg = all(x < 0 for x in vals.values())
            better = all(vals[m] < base[m] for m in vals if m in base)
            tag = "ALL<0" if allneg else ("improves-vs-ungated" if better else "")
            if allneg or better:
                print("  mu=%-11s h=%-2d thr=%.1f  %s  %s"
                      % (mu_name, h, thr,
                         "  ".join("%s %+.1f" % (m, v) for m, v in vals.items()), tag))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "STEP7_GATED.csv")
res.to_csv(out, index=False, encoding="utf-8-sig")
print("\nwrote %s" % out)
