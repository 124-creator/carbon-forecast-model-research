"""Step 9: h-step predictive distributions for the pilot prices, rolling origin.

The point-forecast line is closed (step 8: nothing beats persistence). But a
compliance decision needs an interval, not a number, and a multi-month interval
is where the construction actually matters.

Three interval constructions, all centred on persistence so the point forecast is
never the variable under test:

  A_raw    pool of 1-step residuals, used UNSCALED for horizon h
           (what you get if you forget that the target is h steps away)
  A_sqrt   pool of 1-step residuals scaled by sqrt(h)
           (the textbook iid-increment construction)
  B        pool of realised h-step residuals, horizon-matched
  C        B restricted to the K origins whose log level is closest to today
           (state conditioning - an ablation, not the primary)

Everything is rolling-origin: at origin t only information up to t is used, and
the residual pool only contains residuals whose target date is <= t. Scored with
pinball loss and central-interval coverage at the field's usual nominal levels.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PANEL = (r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research"
         r"\research\fullcycle_h20_20261020\PANEL_4MARKET.csv")
START = "2014-04-04"
ORIGIN_FRAC = 0.60
HORIZONS = (1, 5, 22)
TAUS = np.array([0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975])
KNN = 100


def pinball(tgt, q, tau):
    d = tgt - q
    return np.maximum(tau * d, (tau - 1.0) * d)


df = pd.read_csv(PANEL, parse_dates=["date"])
rows = []
for mkt, g in df.groupby("market"):
    g = g.sort_values("date")
    g = g[g["date"] >= START].reset_index(drop=True)
    y = g["price"].to_numpy(float)
    if y.size < 400:
        continue
    z = np.log(y)
    n = z.size
    t0 = int(n * ORIGIN_FRAC)

    for h in HORIZONS:
        # realised h-step log residuals, indexed by origin date s
        nres = n - h
        res_h = z[h:] - z[:nres]                    # res_h[s] = z[s+h]-z[s]
        res_1 = z[1:] - z[:-1]                      # 1-step residuals

        acc = {k: dict(cov95=[], cov90=[], pin=[], wid=[]) for k in
               ("A_raw", "A_sqrt", "A_sqrt2", "B", "C")}
        for t in range(t0, n - h):
            tgt = z[t + h]
            centre = z[t]
            # ---- pool A: 1-step residuals, all realised by t (s <= t-1) ----
            p1 = res_1[:max(t, 1)]
            pools = {
                "A_raw": centre + p1,
                "A_sqrt": centre + np.sqrt(h) * p1,
                "A_sqrt2": centre + np.sqrt(h) * (p1 - p1.mean()),
            }
            # ---- pool B: h-step residuals fully realised by t (s+h <= t) ----
            hi = t - h + 1
            if hi < 20:
                continue
            pb = res_h[:hi]
            pools["B"] = centre + pb
            # ---- pool C: nearest neighbours in level ----
            dist = np.abs(z[:hi] - centre)
            k = min(KNN, hi)
            idx = np.argpartition(dist, k - 1)[:k]
            pools["C"] = centre + pb[idx]

            for key, sample in pools.items():
                q = np.quantile(sample, TAUS)
                d = tgt - q
                acc[key]["pin"].append(np.mean(np.maximum(TAUS * d, (TAUS - 1) * d)))
                acc[key]["cov95"].append(float(q[0] <= tgt <= q[-1]))
                acc[key]["cov90"].append(float(q[1] <= tgt <= q[-2]))
                acc[key]["wid"].append(float(q[-1] - q[0]))

        for key, a in acc.items():
            if not a["pin"]:
                continue
            rows.append(dict(market=mkt, h=h, method=key, n=len(a["pin"]),
                             pinball=float(np.mean(a["pin"])),
                             cov95=float(np.mean(a["cov95"])),
                             cov90=float(np.mean(a["cov90"])),
                             width95=float(np.mean(a["wid"]))))
    print("done %s" % mkt)

res = pd.DataFrame(rows)
res["pinball_x1e3"] = res["pinball"] * 1e3

print("\n--- rolling-origin interval calibration (nominal 95%% / 90%%) ---")
print("%-10s %3s %-8s %6s %10s %8s %8s %10s"
      % ("market", "h", "method", "n", "pinball", "cov95", "cov90", "width95"))
for mkt in res["market"].unique():
    for h in HORIZONS:
        s = res[(res["market"] == mkt) & (res["h"] == h)]
        for _, r in s.iterrows():
            print("%-10s %3d %-8s %6d %10.6f %8.4f %8.4f %10.5f"
                  % (mkt, h, r["method"], r["n"], r["pinball"], r["cov95"],
                     r["cov90"], r["width95"]))
    print()

print("--- headline: |cov95 - 0.95| by construction ---")
print("%-10s %3s %10s %10s %10s %10s"
      % ("market", "h", "A_raw", "A_sqrt", "B", "C"))
for mkt in res["market"].unique():
    for h in HORIZONS:
        s = res[(res["market"] == mkt) & (res["h"] == h)].set_index("method")
        def g(k):
            return abs(s.loc[k, "cov95"] - 0.95) if k in s.index else np.nan
        print("%-10s %3d %10.4f %10.4f %10.4f %10.4f"
              % (mkt, h, g("A_raw"), g("A_sqrt"), g("B"), g("C")))

print("\n--- pinball relative to B (lower is better; B is the horizon-matched pool) ---")
print("%-10s %3s %10s %10s %10s" % ("market", "h", "A_raw", "A_sqrt", "C"))
for mkt in res["market"].unique():
    for h in HORIZONS:
        s = res[(res["market"] == mkt) & (res["h"] == h)].set_index("method")
        if "B" not in s.index:
            continue
        b = s.loc["B", "pinball"]
        print("%-10s %3d %+10.2f%% %+10.2f%% %+10.2f%%"
              % (mkt, h, (s.loc["A_raw", "pinball"] / b - 1) * 100,
                 (s.loc["A_sqrt", "pinball"] / b - 1) * 100,
                 (s.loc["C", "pinball"] / b - 1) * 100))

base = os.path.dirname(os.path.abspath(__file__))
res.to_csv(os.path.join(base, "STEP9_DENSITY.csv"), index=False, encoding="utf-8-sig")
print("\nwrote STEP9_DENSITY.csv")
