"""
signed_har_xmarket.py — external validity: does the signed decomposition also work
outside the EU market?

risk_round9g established, on EUA, that adding the sign indicator, the log absolute
return, and the five- and twenty-two-day upside and downside realised variances to
HAR improves QLIKE by 7-10% in both test windows, under both losses, in every full
year from 2019 to 2025, and against all 20 parameter-count placebos.

A result that holds only on one market is an EU fact, not a property of carbon
volatility. The same protocol is therefore applied to the four Chinese pilot
markets in the panel. Each market is treated separately, with an expanding window,
a minimum of 500 training observations, refit every 22 days, and a 10-draw placebo
on its own test window.

The Chinese pilots have shorter series and thin trading, so a market whose series
is too short or too flat is reported as unusable rather than forced.
"""

import numpy as np
import pandas as pd

PANEL = (r"E:\Agent员工\碳价格预测研究\独立数据归档\carbon-forecast-model-research"
         r"\research\fullcycle_h20_20261020\PANEL_5MARKET.csv")

REFIT = 22
REPS = 10
MIN_TRAIN = 500
HAR = ["lrv1", "lrv5", "lrv22"]
CLEAN = ["sign1", "logabs1", "dn5", "up5", "dn22", "up22"]


def prep(d):
    d = d.sort_values("date").reset_index(drop=True)
    d["r"] = np.log(d["price"].astype(float)).diff()
    r2 = d["r"] ** 2
    d["rv1"], d["rv5"], d["rv22"] = r2, r2.rolling(5).mean(), r2.rolling(22).mean()
    for h in (5, 22):
        d[f"y{h}"] = r2.rolling(h).mean().shift(-h)
        d[f"ly{h}"] = np.log(d[f"y{h}"].clip(lower=1e-14))
    for c, s in (("lrv1", "rv1"), ("lrv5", "rv5"), ("lrv22", "rv22")):
        d[c] = np.log(d[s].clip(lower=1e-14))
    pos = d["r"].clip(lower=0.0) ** 2
    neg = d["r"].clip(upper=0.0) ** 2
    for w in (5, 22):
        d[f"dn{w}"] = np.log(neg.rolling(w).mean().clip(lower=1e-14))
        d[f"up{w}"] = np.log(pos.rolling(w).mean().clip(lower=1e-14))
    d["sign1"] = np.sign(d["r"]).fillna(0.0)
    d["logabs1"] = np.log(d["r"].abs().clip(lower=1e-14))
    return d


def walk(d, cols, target, refit=REFIT):
    y = d[target].to_numpy(float)
    X = d[cols].to_numpy(float)
    v = np.full(len(d), np.nan)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    cum = np.cumsum(ok)
    if cum[-1] < MIN_TRAIN + 200:
        return v
    # first row whose preceding window already holds MIN_TRAIN valid observations
    i = int(np.argmax(cum >= MIN_TRAIN)) + 1
    while i < len(d):
        m = ok.copy()
        m[i:] = False
        if m.sum() < MIN_TRAIN:
            break
        D = np.column_stack([np.ones(m.sum()), X[m]])
        b, *_ = np.linalg.lstsq(D, y[m], rcond=None)
        f = float(np.mean(np.exp(y[m] - D @ b)))
        j = min(i + refit, len(d))
        v[i:j] = np.exp(np.column_stack([np.ones(j - i), X[i:j]]) @ b) * f
        i = j
    return v


def ql(v, y):
    r = np.clip(v, 1e-14, None) / np.clip(y, 1e-14, None)
    return r - np.log(r) - 1.0


def block_ci(diff, block=10, reps=2000, seed=20261021):
    rng = np.random.default_rng(seed)
    n = len(diff)
    nb = int(np.ceil(n / block))
    out = np.empty(reps)
    for r in range(reps):
        st = rng.integers(0, max(1, n - block), nb)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in st])[:n]
        out[r] = diff[idx].mean()
    return out


def main():
    raw = pd.read_csv(PANEL)
    raw["date"] = pd.to_datetime(raw["date"])
    rng = np.random.default_rng(20261021)
    for mkt in ["EU", "CEA", "GUANGZHOU", "SHANGHAI", "HUBEI"]:
        g = raw[raw["market"] == mkt][["date", "price"]].dropna()
        d = prep(g)
        nz = (d["r"].abs() > 0).sum()
        print(f"\n########## {mkt}   rows={len(d)}  "
              f"{d['date'].min().date()} -> {d['date'].max().date()}  nonzero_ret={nz}")
        if len(d) < MIN_TRAIN + 300 or nz < 400:
            print("  unusable: series too short or too flat")
            continue
        for target, proxy in (("ly5", "y5"), ("ly22", "y22")):
            v_har = walk(d, HAR, target)
            v_sd = walk(d, HAR + CLEAN, target)
            m = (np.isfinite(d[proxy]).to_numpy() & np.isfinite(v_har)
                 & np.isfinite(v_sd))
            if m.sum() < 200:
                print(f"  {target}: too few test observations ({int(m.sum())})")
                continue
            y = d[proxy].to_numpy()[m]
            a, b = ql(v_har[m], y), ql(v_sd[m], y)
            diff = b - a
            lo, hi = np.percentile(block_ci(diff), [2.5, 97.5])
            gain = 100.0 * (1.0 - b.mean() / a.mean())
            star = " *" if hi < 0 else ("  (worse)" if lo > 0 else "")
            print(f"  {target:<5} n={int(m.sum()):>4}  HAR={a.mean():.5f}  "
                  f"HAR_SD={b.mean():.5f}  gain={gain:+.2f}%  "
                  f"CI[{lo:+.6f},{hi:+.6f}]{star}")
            block = d[CLEAN].to_numpy(float)
            gains = []
            for _ in range(REPS):
                perm = rng.permutation(len(d))
                dd = d.copy()
                for k, c in enumerate(CLEAN):
                    dd[c + "_pl"] = block[perm, k]
                vp = walk(dd, HAR + [c + "_pl" for c in CLEAN], target)
                mm = m & np.isfinite(vp)
                gains.append(100.0 * (1.0 - np.mean(ql(vp[mm], d[proxy].to_numpy()[mm]))
                                      / np.mean(ql(v_har[mm], d[proxy].to_numpy()[mm]))))
            gains = np.array(gains)
            print(f"        placebo mean={gains.mean():+.2f}% max={gains.max():+.2f}  "
                  f"real={gain:+.2f}%")


if __name__ == "__main__":
    main()
