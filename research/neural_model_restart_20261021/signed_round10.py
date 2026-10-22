"""
signed_round10.py — the mechanism behind the signed-decomposition gain.

The signed specification works (7-10% QLIKE on EUA, 3 of 4 Chinese pilots at h=22)
but "signed semivariance HAR" is not new: Patton and Sheppard (2015) introduced the
signed decomposition for equity volatility, so importing it to carbon is a transfer.

The open question is why the ratio constraint fails while the free two-state version
succeeds. A single asymmetry ratio forces upside and downside variation to enter with
equal and opposite coefficients, i.e. it assumes both states decay at the same rate.
If the two states instead have different persistence, the ratio constraint is
misspecified and only the free specification can capture the effect.

This script measures that directly: the first-order autocorrelation and half-life of
log upside variance and log downside variance at each scale, with block-bootstrap
confidence intervals on the difference. If the two persistences differ significantly,
the mechanism is a persistence divergence between signed states, which is a testable
claim about carbon that the equity literature did not need to make.
"""

import numpy as np
import pandas as pd

EUA = (r"E:\Agent员工\碳价格预测研究\独立数据归档\_legacy_202605_archive"
       r"\主要实验推进_审计加固版_CLEAN_20260528\08-新实验\01_数据"
       r"\EUA_原始数据_完整版.csv")

SCALES = (1, 5, 22)


def build():
    raw = pd.read_csv(EUA)
    c = [x for x in raw.columns if "close" in x.lower()][0]
    d = pd.DataFrame({"date": pd.to_datetime(raw["date"]), "p": raw[c].astype(float)})
    d = d.dropna().sort_values("date").reset_index(drop=True)
    d["r"] = np.log(d["p"]).diff()
    pos = d["r"].clip(lower=0.0) ** 2
    neg = d["r"].clip(upper=0.0) ** 2
    for w in SCALES:
        d[f"up{w}"] = np.log(pos.rolling(w).mean().clip(lower=1e-14))
        d[f"dn{w}"] = np.log(neg.rolling(w).mean().clip(lower=1e-14))
    return d


def ar1(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 100:
        return np.nan, np.nan
    a, b = x[:-1], x[1:]
    D = np.column_stack([np.ones(len(a)), a])
    coef, *_ = np.linalg.lstsq(D, b, rcond=None)
    rho = float(coef[1])
    hl = float(np.log(0.5) / np.log(rho)) if 0 < rho < 1 else np.inf
    return rho, hl


def boot_diff(x, y, block=20, reps=1000, seed=20261021):
    """CI for acf1(x) - acf1(y), resampling the two series on a common index."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    out = np.empty(reps)
    for r in range(reps):
        st = rng.integers(0, max(1, n - block), nb)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in st])[:n]
        a1, _ = ar1(x[idx])
        a2, _ = ar1(y[idx])
        out[r] = a1 - a2
    return out


def main():
    d = build()
    print(f"rows={len(d)}  {d['date'].min().date()} -> {d['date'].max().date()}")
    print("\npersistence of the signed states (AR(1) on log variance)\n")
    print(f"{'scale':>6} {'rho_up':>8} {'hl_up':>8} {'rho_dn':>8} {'hl_dn':>8} "
          f"{'diff':>8} {'CI':>26}")
    for w in SCALES:
        ru, hu = ar1(d[f"up{w}"])
        rd, hd = ar1(d[f"dn{w}"])
        bs = boot_diff(d[f"up{w}"], d[f"dn{w}"])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        star = " *" if (lo > 0 or hi < 0) else ""
        print(f"{w:>6} {ru:>8.3f} {hu:>8.1f} {rd:>8.3f} {hd:>8.1f} "
              f"{ru-rd:>+8.3f}   [{lo:+.3f},{hi:+.3f}]{star}")
    print("\nsame test on the symmetric aggregate, as a control:")
    rs, hs = ar1(d["up5"] + d["dn5"])
    print(f"  log(up5+dn5) rho={rs:.3f} half-life={hs:.1f}")


if __name__ == "__main__":
    main()
