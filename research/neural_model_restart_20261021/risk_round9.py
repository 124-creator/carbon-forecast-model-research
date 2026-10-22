"""
risk_round9.py (v2) — QLIKE and VaR, done correctly.

v1 was wrong in two ways and its numbers must not be used:
  1. It exponentiated the forecast of log variance without the lognormal bias
     correction, so E[log v] was used as if it were log E[v]. QLIKE then blew up
     to O(1e5) and the VaR scale was inflated about tenfold.
  2. It scored every horizon against the one-day proxy rv1, although the ly5 and
     ly22 targets forecast the 5- and 22-day average variance.

v2 fixes both. Forecasts are retransformed with Duan's smearing estimator,
f = mean(exp(residual)) estimated inside each training window, so v_hat =
exp(pred) * f. Each horizon is scored against the realised variance it actually
forecasts (y1, y5, y22). VaR is built only from the one-day model, which is the
only one that maps to a next-day loss.
"""

import math

import numpy as np

import har_x_round9 as H

LEV = ["neg1", "neg5", "semivar5"]
REFIT = 22


def walk_point(d, cols, target, start):
    """Forecast of log variance, retransformed by smearing to a variance level."""
    y = d[target].to_numpy(float)
    X = d[cols].to_numpy(float)
    v = np.full(len(d), np.nan)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    i = int((d["date"] < start).sum())
    while i < len(d):
        m = ok.copy()
        m[i:] = False
        if m.sum() < 150:
            break
        D = np.column_stack([np.ones(m.sum()), X[m]])
        b, *_ = np.linalg.lstsq(D, y[m], rcond=None)
        f = float(np.mean(np.exp(y[m] - D @ b)))          # Duan smearing factor
        j = min(i + REFIT, len(d))
        v[i:j] = np.exp(np.column_stack([np.ones(j - i), X[i:j]]) @ b) * f
        i = j
    return v


def qlike(v_hat, y):
    r = np.clip(v_hat, 1e-14, None) / np.clip(y, 1e-14, None)
    return float(np.mean(r - np.log(r) - 1.0))


def kupiec(x, n, p):
    if x == 0 or x == n:
        return np.nan, np.nan
    ph = x / n
    lr = -2.0 * ((n - x) * math.log(1 - p) + x * math.log(p)
                 - (n - x) * math.log(1 - ph) - x * math.log(ph))
    return lr, math.erfc(math.sqrt(max(lr, 0.0) / 2.0))


def main():
    d = H.load()
    models = {"EWMA22": ["lrv22"], "HAR": H.HAR, "HAR_L": H.HAR + LEV}
    for target, proxy in (("ly1", "y1"), ("ly5", "y5"), ("ly22", "y22")):
        print(f"\n############ target = {target}   scored against {proxy} ############")
        for start in ("2019-01-01", "2022-01-01"):
            fc, base = {}, None
            for name, cols in models.items():
                fc[name] = walk_point(d, cols, target, start)
            m = (np.isfinite(d[proxy]).to_numpy() & (d["date"] >= start).to_numpy())
            for v in fc.values():
                m &= np.isfinite(v)
            y = d[proxy].to_numpy()[m]
            print(f"  --- test from {start}   n={int(m.sum())} ---")
            for name, v in fc.items():
                q = qlike(v[m], y)
                mse = float(np.mean((v[m] - y) ** 2))
                if name == "EWMA22":
                    base = q
                    print(f"    {name:<8} QLIKE={q:.5f}  MSE_var={mse:.3e}   (benchmark)")
                else:
                    print(f"    {name:<8} QLIKE={q:.5f}  MSE_var={mse:.3e}"
                          f"   QLIKE vs EWMA {100*(1-q/base):+.2f}%")
            # ---- VaR from the one-day model, filtered historical simulation ----
            if target == "ly1":
                print("    VaR, one-day, filtered historical simulation:")
                for name in models:
                    sigma = np.sqrt(fc[name])
                    idx = np.where(m)[0]
                    cut = int(0.5 * len(idx))
                    fit_i, test_i = idx[:cut], idx[cut:]
                    z = d["r"].to_numpy()[fit_i] / sigma[fit_i]
                    z = z[np.isfinite(z)]
                    logr = d["r"].to_numpy()[test_i]
                    px = d["eua_close_t"].to_numpy()[test_i]
                    pl = -(np.exp(logr) - 1.0) * px
                    for lvl in (0.05, 0.01):
                        qz = np.quantile(z, lvl)
                        var = -qz * sigma[test_i] * px
                        viol = float(np.mean(pl > var))
                        _, pv = kupiec(int(np.sum(pl > var)), len(pl), lvl)
                        flag = "OK" if (pv == pv and pv > 0.05) else "reject"
                        print(f"      {name:<8} {int(round((1-lvl)*100))}%  "
                              f"violations={viol:.4f} (nominal {lvl:.2f})  "
                              f"Kupiec p={pv:.3f}  {flag}")


if __name__ == "__main__":
    main()
