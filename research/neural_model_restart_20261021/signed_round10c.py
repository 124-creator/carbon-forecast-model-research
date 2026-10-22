"""
signed_round10c.py — the minimal specification, its coefficients, and its placebo.

The ablation in signed_round10b is unambiguous. Of the six signed features, only the
twenty-two-day pair matters: adding log upside and log downside monthly variance
alone reproduces 7.1 of the 7.9 percentage points at the five-day horizon and 9.8 of
10.0 at the monthly horizon. The daily sign and magnitude contribute nothing, and
the five-day signed pair contributes about one point.

That explains why the ratio version failed. HAR already contains lrv22, the log of
the monthly mean squared return, which is the log of the SUM of the upside and
downside parts. Writing the signed pair as (level, ratio) and estimating one
asymmetry coefficient forces the two components to enter with equal and opposite
loadings. If both load positively, with the downside loading larger, that constraint
is badly wrong, and the small gain from asym22 was the constrained fit fighting the
data.

So the minimal model is HAR plus two regressors:

    S-HAR:  lrv1 lrv5 lrv22  +  log downside variance(22)  +  log upside variance(22)

This script checks it four ways: QLIKE with bootstrap CIs, the loadings with HAC
standard errors and a test that they differ, the per-year pattern, and the
parameter-count placebo, which now adds only two columns.
"""

import numpy as np

import har_x_round9 as H

REFIT = 22
REPS = 20
SIGNED22 = ["dn22", "up22"]


def add_signed(d):
    d = d.copy()
    pos = d["r"].clip(lower=0.0) ** 2
    neg = d["r"].clip(upper=0.0) ** 2
    d["dn22"] = np.log(neg.rolling(22).mean().clip(lower=1e-14))
    d["up22"] = np.log(pos.rolling(22).mean().clip(lower=1e-14))
    return d


def walk(d, cols, target, start, refit=REFIT):
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


def hac(D, y, lag=22):
    b, *_ = np.linalg.lstsq(D, y, rcond=None)
    e = y - D @ b
    XtX = np.linalg.pinv(D.T @ D)
    S = (D * e[:, None]).T @ (D * e[:, None])
    for L in range(1, lag + 1):
        w = 1.0 - L / (lag + 1.0)
        G = (D[L:] * e[L:, None]).T @ (D[:-L] * e[:-L, None])
        S += w * (G + G.T)
    V = XtX @ S @ XtX
    return b, np.sqrt(np.clip(np.diag(V), 0, None))


def main():
    d = add_signed(H.load())
    rng = np.random.default_rng(20261021)
    for target, proxy in (("ly5", "y5"), ("ly22", "y22")):
        print(f"\n############ target = {target}  scored against {proxy} ############")
        for start in ("2019-01-01", "2022-01-01"):
            v_har = walk(d, H.HAR, target, start)
            v_sh = walk(d, H.HAR + SIGNED22, target, start)
            m = np.isfinite(d[proxy]).to_numpy() & (d["date"] >= start).to_numpy()
            m &= np.isfinite(v_har) & np.isfinite(v_sh)
            y = d[proxy].to_numpy()[m]
            a, b = ql(v_har[m], y), ql(v_sh[m], y)
            lo, hi = np.percentile(block_ci(b - a), [2.5, 97.5])
            print(f"  --- test from {start}   n={int(m.sum())} ---")
            print(f"    HAR    QLIKE={a.mean():.5f}")
            print(f"    S-HAR  QLIKE={b.mean():.5f}   gain={100*(1-b.mean()/a.mean()):+.2f}%"
                  f"   CI[{lo:+.6f},{hi:+.6f}]")
            yrs = d["date"].dt.year.to_numpy()[m]
            row = []
            for yr in np.unique(yrs):
                k = yrs == yr
                if k.sum() < 40:
                    continue
                row.append(f"{yr}:{100*(1-b[k].mean()/a[k].mean()):+.1f}%")
            print("    per-year: " + "  ".join(row))
            block = d[SIGNED22].to_numpy(float)
            gains = []
            for _ in range(REPS):
                perm = rng.permutation(len(d))
                dd = d.copy()
                for k, c in enumerate(SIGNED22):
                    dd[c + "_pl"] = block[perm, k]
                vp = walk(dd, H.HAR + [c + "_pl" for c in SIGNED22], target, start)
                mm = m & np.isfinite(vp)
                gains.append(100.0 * (1.0 - np.mean(ql(vp[mm], d[proxy].to_numpy()[mm]))
                                      / np.mean(ql(v_har[mm], d[proxy].to_numpy()[mm]))))
            gains = np.array(gains)
            real = 100.0 * (1.0 - b.mean() / a.mean())
            print(f"    placebo[2 cols] mean={gains.mean():+.2f}% sd={gains.std():.2f} "
                  f"max={gains.max():+.2f}  real={real:+.2f}%  "
                  f"p={np.mean(gains >= real):.2f}")

    print("\nloadings on the signed pair, full sample, HAC(22) errors:")
    for target in ("ly5", "ly22"):
        X = d[H.HAR + SIGNED22].to_numpy(float)
        ok = np.isfinite(d[target]).to_numpy() & np.isfinite(X).all(1)
        D = np.column_stack([np.ones(ok.sum()), X[ok]])
        y = d[target].to_numpy(float)[ok]
        b, se = hac(D, y)
        names = ["const"] + H.HAR + SIGNED22
        print(f"  --- {target}  n={int(ok.sum())} ---")
        for nm, bi, si in zip(names, b, se):
            t = bi / si if si > 0 else np.nan
            print(f"    {nm:<8} beta={bi:+.5f}  se={si:.5f}  t={t:+.2f}")
        bd, bu = b[-2], b[-1]
        sd, su = se[-2], se[-1]
        z = (bd - bu) / np.sqrt(sd ** 2 + su ** 2)
        print(f"    test down == up:  diff={bd-bu:+.5f}  z={z:+.2f}"
              f"   {'reject equality' if abs(z) > 1.96 else 'cannot reject equality'}")


if __name__ == "__main__":
    main()
