"""
bench_round11.py — the comparison set a referee will demand, plus the two tests
that were missing.

Three gaps stood between the HAR-D result and a submittable paper.

  1. The benchmark set was HAR and EWMA only. A volatility forecasting paper needs
     the GARCH family. Added here: GARCH(1,1), GJR-GARCH(1,1), GARCH(1,1) with
     Student-t innovations, plus the naive random walk and a 22-day historical
     variance, plus HAR-J (HAR with a jump component built from bipower variation).

  2. The claim that the two signed loadings are equal was tested with a z statistic
     that ignored their covariance. Replaced by a proper Wald test using the full
     HAC covariance matrix.

  3. Running many specifications over one test window invites a specification-search
     objection. White's Reality Check is applied across the whole candidate set, so
     the surviving model has to beat HAR after accounting for the search.

A subperiod cut is also reported: excluding 2020 and excluding 2021-2022, to show
the effect is not carried by the pandemic or the energy crisis alone.

GARCH forecasts are iterated analytically from the filtered variance at each origin,
with parameters re-estimated every 22 days and the filter updated daily. HAR-type
models are direct projections, which is the standard asymmetry between the two
families and is stated as such.
"""

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import chi2

import har_x_round9 as H

REFIT = 22
H_MAP = {"y5": 5, "y22": 22}


# ----------------------------------------------------------------------------- data
def build():
    d = H.load().reset_index(drop=True)
    d = d[np.isfinite(d["r"].to_numpy(float))].reset_index(drop=True)
    pos = d["r"].clip(lower=0.0) ** 2
    neg = d["r"].clip(upper=0.0) ** 2
    d["dn22"] = np.log(neg.rolling(22).mean().clip(lower=1e-14))
    d["up22"] = np.log(pos.rolling(22).mean().clip(lower=1e-14))
    d["g22"] = d["dn22"] + d["up22"]
    # jump component from bipower variation, 22-day windows
    absr = d["r"].abs()
    bv = (np.pi / 2.0) * (absr * absr.shift(1)).rolling(22).mean()
    rv = (d["r"] ** 2).rolling(22).mean()
    d["jump"] = (rv - bv).clip(lower=0.0)
    d["lj"] = np.log(d["jump"] + 1e-12)
    return d


# ------------------------------------------------------------- direct HAR-type fit
def walk_direct(d, cols, target, start, refit=REFIT):
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


# ---------------------------------------------------------------- GARCH family
def garch_forecasts(d, start, kind="GARCH", dist="normal", refit=REFIT):
    """Iterated mean-variance forecast over the next five and twenty-two days."""
    r = (d["r"].to_numpy(float) * 100.0)
    n = len(d)
    v5 = np.full(n, np.nan)
    v22 = np.full(n, np.nan)
    ok = np.isfinite(r)
    i0 = int((d["date"] < start).sum())
    gjr = kind == "GJR"
    o = 1 if gjr else 0
    b = i0
    while b < n:
        if ok[:b].sum() < 500:
            b += refit
            continue
        am = arch_model(r[:b], mean="Zero", vol="Garch", p=1, o=o, q=1, dist=dist)
        try:
            res = am.fit(disp="off", show_warning=False)
        except Exception:
            b += refit
            continue
        p = res.params
        om, al, be = float(p["omega"]), float(p["alpha[1]"]), float(p["beta[1]"])
        ga = float(p["gamma[1]"]) if gjr else 0.0
        phi = al + 0.5 * ga + be
        if not (0.0 < phi < 0.999) or om <= 0:
            b += refit
            continue
        jend = min(b + refit, n)
        for j in range(b, jend):
            if not ok[: j + 1].all():
                continue
            aj = arch_model(r[: j + 1], mean="Zero", vol="Garch", p=1, o=o, q=1,
                            dist=dist)
            try:
                rj = aj.fix(p)
            except Exception:
                continue
            s2 = float(rj.conditional_volatility[-1]) ** 2
            neg = 1.0 if r[j] < 0 else 0.0
            e1 = om + (al + ga * neg) * r[j] ** 2 + be * s2
            path = [e1]
            for _ in range(21):
                path.append(om + phi * path[-1])
            path = np.array(path)
            v5[j] = path[:5].mean() / 1e4
            v22[j] = path.mean() / 1e4
        b = jend
    return {"y5": v5, "y22": v22}


# ------------------------------------------------------------- naive benchmarks
def naive_forecasts(d):
    r2 = d["r"] ** 2
    out = {}
    rw = r2.to_numpy(float)
    out["RW"] = {"y5": rw, "y22": rw}
    h = r2.rolling(22).mean().to_numpy(float)
    out["HIST22"] = {"y5": h, "y22": h}
    lam = 0.94
    e = np.full(len(d), np.nan)
    x = r2.to_numpy(float)
    s = np.nanvar(x[:200])
    for t in range(len(d)):
        e[t] = s
        if np.isfinite(x[t]):
            s = lam * s + (1 - lam) * x[t]
    out["EWMA"] = {"y5": e, "y22": e}
    return out


# ----------------------------------------------------------------------- tests
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


def reality_check(L, block=10, reps=2000, seed=20261021):
    """White's RC: does any candidate beat the benchmark (column 0)?"""
    rng = np.random.default_rng(seed)
    D = L - L[:, [0]]
    n, k = D.shape
    stat = float(np.max(-np.sqrt(n) * D.mean(0)))
    nb = int(np.ceil(n / block))
    null = np.empty(reps)
    for b in range(reps):
        st = rng.integers(0, max(1, n - block), nb)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in st])[:n]
        Db = D[idx]
        null[b] = np.max(-np.sqrt(n) * (Db.mean(0) - D.mean(0)))
    return stat, float(np.mean(null >= stat)), np.argmax(-D.mean(0))


def hac_cov(D, y, lag=22):
    b, *_ = np.linalg.lstsq(D, y, rcond=None)
    e = y - D @ b
    XtX = np.linalg.pinv(D.T @ D)
    S = (D * e[:, None]).T @ (D * e[:, None])
    for L in range(1, lag + 1):
        w = 1.0 - L / (lag + 1.0)
        G = (D[L:] * e[L:, None]).T @ (D[:-L] * e[:-L, None])
        S += w * (G + G.T)
    return b, XtX @ S @ XtX


def main():
    d = build()
    marks = {"all": None, "ex2020": 2020, "ex2122": (2021, 2022)}
    for start in ("2019-01-01", "2022-01-01"):
        print("\n" + "=" * 80)
        print(f"test window from {start}")
        print("=" * 80)
        fc = {}
        fc["HAR"] = {"y5": walk_direct(d, H.HAR, "ly5", start),
                     "y22": walk_direct(d, H.HAR, "ly22", start)}
        fc["HAR-D"] = {"y5": walk_direct(d, H.HAR + ["g22"], "ly5", start),
                       "y22": walk_direct(d, H.HAR + ["g22"], "ly22", start)}
        fc["HAR-J"] = {"y5": walk_direct(d, H.HAR + ["lj"], "ly5", start),
                       "y22": walk_direct(d, H.HAR + ["lj"], "ly22", start)}
        for nm in ("RW", "HIST22", "EWMA"):
            nv = naive_forecasts(d)[nm]
            fc[nm] = nv
        for nm, kind, dist in (("GARCH", "GARCH", "normal"),
                               ("GJR", "GJR", "normal"),
                               ("GARCH-t", "GARCH", "t")):
            fc[nm] = garch_forecasts(d, start, kind=kind, dist=dist)
        order = ["HAR", "HAR-D", "HAR-J", "GARCH", "GJR", "GARCH-t",
                 "EWMA", "HIST22", "RW"]
        for tgt, proxy in (("ly5", "y5"), ("ly22", "y22")):
            hkey = proxy
            m = np.isfinite(d[proxy]).to_numpy() & (d["date"] >= start).to_numpy()
            for nm in order:
                m &= np.isfinite(fc[nm][hkey])
            y = d[proxy].to_numpy()[m]
            print(f"\n  --- {tgt} vs {proxy}   n={int(m.sum())} ---")
            per = {}
            for nm in order:
                per[nm] = ql(fc[nm][hkey][m], y)
                print(f"    {nm:<8} QLIKE={per[nm].mean():.5f}")
            base = per["HAR"].mean()
            print("    vs HAR:")
            for nm in order[1:]:
                g = 100.0 * (1.0 - per[nm].mean() / base)
                stat, pv = dm(per[nm] - per["HAR"])
                print(f"      {nm:<8} {g:+7.2f}%   DM={stat:+6.2f}  p={pv:.4f}")
            L = np.column_stack([per[nm] for nm in order])
            stat, pv, k = reality_check(L)
            print(f"    White RC across {len(order)-1} candidates: stat={stat:.2f} "
                  f"p={pv:.4f}  best={order[k]}")
            # subperiods
            yr = d["date"].dt.year.to_numpy()[m]
            for lbl, drop in (("ex2020", 2020), ("ex2122", (2021, 2022))):
                keep = ~np.isin(yr, drop)
                a = per["HAR"][keep].mean()
                b = per["HAR-D"][keep].mean()
                print(f"    {lbl:<8} n={int(keep.sum()):>4}  HAR={a:.5f}  "
                      f"HAR-D={b:.5f}  gain={100*(1-b/a):+.2f}%")
    # ---- Wald test for equal signed loadings, full HAC covariance
    print("\n" + "=" * 80)
    print("Wald test: are the two signed loadings equal?")
    print("=" * 80)
    for target in ("ly5", "ly22"):
        cols = H.HAR + ["dn22", "up22"]
        X = d[cols].to_numpy(float)
        ok = np.isfinite(d[target]).to_numpy() & np.isfinite(X).all(1)
        D = np.column_stack([np.ones(ok.sum()), X[ok]])
        y = d[target].to_numpy(float)[ok]
        b, V = hac_cov(D, y)
        R = np.zeros(len(b))
        R[-2], R[-1] = 1.0, -1.0
        diff = float(R @ b)
        var = float(R @ V @ R)
        stat = diff ** 2 / var
        print(f"  {target}: dn22={b[-2]:+.4f}  up22={b[-1]:+.4f}  diff={diff:+.4f}  "
              f"Wald={stat:.2f}  p={1-chi2.cdf(stat,1):.4f}")


if __name__ == "__main__":
    main()
