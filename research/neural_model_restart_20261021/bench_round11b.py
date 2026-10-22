"""
bench_round11b.py — is the QLIKE gap a calibration problem or an information problem?

bench_round11 produced the uncomfortable result that a parameter-free twenty-two day
moving average of squared returns scores far better in QLIKE than either HAR or the
signed extension. Before concluding that the log-HAR family carries no usable
information on this proxy, one cheap diagnostic separates two very different
explanations.

QLIKE has a closed-form optimal scale. Because

    QLIKE(c v, y) = mean( c v / y - log(c v / y) - 1 )

is minimised at c* = 1 / mean(v / y), a single multiplicative recalibration removes
any level bias. If a scalar fitted only on data before the test point closes the gap
to EWMA, the problem is calibration, and the paper survives. If it does not, the
information genuinely is not there, and the route is finished.

Two things are also tested here that should have been in the original design: HAR
fitted in levels rather than logs, which is Corsi's original specification and does
not need the smearing correction, and a level-space dispersion term

    gm22 = sqrt( rv_pos22 * rv_neg22 )

which is the geometric mean of the two signed monthly components, the level analogue
of the log specification used in the paper draft.

Recalibration is honest: c is estimated at each refit from the training window only.
"""

import numpy as np
import pandas as pd

import har_x_round9 as H

REFIT = 22


def build():
    d = H.load().reset_index(drop=True)
    d = d[np.isfinite(d["r"].to_numpy(float))].reset_index(drop=True)
    pos = d["r"].clip(lower=0.0) ** 2
    neg = d["r"].clip(upper=0.0) ** 2
    d["p22"] = pos.rolling(22).mean()
    d["n22"] = neg.rolling(22).mean()
    d["dn22"] = np.log(d["n22"].clip(lower=1e-14))
    d["up22"] = np.log(d["p22"].clip(lower=1e-14))
    d["g22"] = d["dn22"] + d["up22"]
    d["gm22"] = np.sqrt(d["p22"] * d["n22"])
    for w in (1, 5, 22):
        d[f"v{w}"] = (d["r"] ** 2).rolling(w).mean()
    d["y5"] = (d["r"] ** 2).rolling(5).mean().shift(-5)
    d["y22"] = (d["r"] ** 2).rolling(22).mean().shift(-22)
    return d


def ql(v, y):
    r = np.clip(v, 1e-14, None) / np.clip(y, 1e-14, None)
    return r - np.log(r) - 1.0


def dm(diff, lag=44):
    from math import erfc, sqrt
    n = len(diff)
    mu = diff.mean()
    e = diff - mu
    s = float(e @ e) / n
    for L in range(1, lag + 1):
        w = 1.0 - L / (lag + 1.0)
        s += 2.0 * w * float(e[L:] @ e[:-L]) / n
    if s <= 0:
        return np.nan, np.nan
    stat = mu / np.sqrt(s / n)
    return stat, erfc(abs(stat) / sqrt(2.0))


def walk(d, cols, target, start, logspace, level=None, refit=REFIT):
    """Returns forecast level, plus the training-only QLIKE scale c*."""
    y = d[target].to_numpy(float)
    yy = y  # ly* targets are already in logs; levels pass through unchanged
    ylev = d[(level or target)].to_numpy(float)
    X = d[cols].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(ylev) & np.isfinite(X).all(1)
    v = np.full(len(d), np.nan)
    c = np.full(len(d), np.nan)
    i = int((d["date"] < start).sum())
    while i < len(d):
        m = ok.copy()
        m[i:] = False
        if m.sum() < 150:
            break
        D = np.column_stack([np.ones(m.sum()), X[m]])
        b, *_ = np.linalg.lstsq(D, yy[m], rcond=None)
        if logspace:
            f = float(np.mean(np.exp(yy[m] - D @ b)))
            fit = np.exp(D @ b) * f
        else:
            fit = D @ b
        fit = np.clip(fit, 1e-12, None)
        ratio = float(np.mean(fit / np.clip(ylev[m], 1e-14, None)))
        ctr = 1.0 / ratio if ratio > 0 else 1.0
        j = min(i + refit, len(d))
        Xj = np.column_stack([np.ones(j - i), X[i:j]])
        v[i:j] = (np.exp(Xj @ b) * f) if logspace else (Xj @ b)
        v[i:j] = np.clip(v[i:j], 1e-14, None)
        c[i:j] = ctr
        i = j
    return v, c


def walk_naive(v_series, target, start, d, refit=REFIT):
    y = d[target].to_numpy(float)
    v = v_series.to_numpy(float).copy()
    ok = np.isfinite(y) & np.isfinite(v)
    c = np.full(len(d), np.nan)
    i = int((d["date"] < start).sum())
    while i < len(d):
        m = ok.copy()
        m[i:] = False
        if m.sum() < 150:
            break
        ratio = float(np.mean(np.clip(v[m], 1e-14, None) / np.clip(y[m], 1e-14, None)))
        j = min(i + refit, len(d))
        c[i:j] = 1.0 / ratio if ratio > 0 else 1.0
        i = j
    return v, c


def main():
    d = build()
    lam = 0.94
    r2 = (d["r"] ** 2).to_numpy(float)
    e = np.full(len(d), np.nan)
    s = float(np.nanvar(r2[:200]))
    for t in range(len(d)):
        e[t] = s
        if np.isfinite(r2[t]):
            s = lam * s + (1 - lam) * r2[t]
    d["ewma"] = e
    d["hist22"] = (d["r"] ** 2).rolling(22).mean()

    for start in ("2019-01-01", "2022-01-01"):
        print("\n" + "=" * 84)
        print(f"test window from {start}")
        print("=" * 84)
        for tgt, proxy in (("ly5", "y5"), ("ly22", "y22")):
            print(f"\n  --- horizon {proxy}   n within window ---")
            arms = {}
            arms["HAR_log"] = walk(d, H.HAR, tgt, start, True, proxy)
            arms["HAR-D_log"] = walk(d, H.HAR + ["g22"], tgt, start, True, proxy)
            arms["HAR_lvl"] = walk(d, ["v1", "v5", "v22"], proxy, start, False)
            arms["HAR-D_lvl"] = walk(d, ["v1", "v5", "v22", "gm22"], proxy, start, False)
            arms["EWMA"] = walk_naive(d["ewma"], proxy, start, d)
            arms["HIST22"] = walk_naive(d["hist22"], proxy, start, d)
            m = np.isfinite(d[proxy]).to_numpy() & (d["date"] >= start).to_numpy()
            for v, _ in arms.values():
                m &= np.isfinite(v)
            y = d[proxy].to_numpy()[m]
            print(f"    {'model':<12} {'raw':>9} {'recal':>9} {'gain%':>8}")
            for nm, (v, c) in arms.items():
                a = ql(v[m], y).mean()
                b = ql(v[m] * c[m], y).mean()
                print(f"    {nm:<12} {a:9.5f} {b:9.5f} {100*(1-b/a):+8.2f}")
            v0 = arms["HAR_log"][0][m]
            v1 = arms["HAR-D_log"][0][m]
            v2 = arms["HAR-D_lvl"][0][m]
            v3 = arms["HAR-D_lvl"][1][m]
            for tag, vv in (("HAR-D_log recal", v1 * arms["HAR-D_log"][1][m]),
                            ("HAR-D_lvl recal", v2 * v3)):
                st, pv = dm(ql(vv, y) - ql(arms["EWMA"][0][m] * arms["EWMA"][1][m], y))
                g = 100 * (1 - ql(vv, y).mean()
                           / ql(arms["EWMA"][0][m] * arms["EWMA"][1][m], y).mean())
                print(f"    {tag} vs EWMA(recal): {g:+.2f}%  DM={st:+.2f} p={pv:.4f}")


if __name__ == "__main__":
    main()
