"""
risk_round9g.py — verification of the signed multi-scale result, and a clean
specification that removes the degenerate clip.

risk_round9f found that HAR plus signed multi-scale components (downside and
upside realised variance at 1, 5 and 22 days) beats HAR by 7-10% QLIKE in all four
cells, and beats all 20 parameter-count placebos in all four.

One feature in that design is degenerate by construction. dn1 is the log of the
squared return when the return is negative, clipped at 1e-14, so on a positive day
it collapses to about -32. That is not wrong -- the regression can read it as a
near-binary sign indicator plus a magnitude -- but the result should not depend on
an arbitrary clip level.

This script therefore checks four things.

  1. Same model under MSE on log variance and under R^2, not only QLIKE. A gain
     that appears under one loss only is a loss artefact.
  2. Per-year QLIKE gains, to see whether the effect is one regime rather than a
     property of the process.
  3. A clean specification with no clip: sign indicator, log absolute return, and
     the five- and twenty-two-day down/up scales. If the clean version holds, the
     clip was not doing the work.
  4. Placebo for the clean version, same standard as before.
"""

import numpy as np

import har_x_round9 as H

REFIT = 22
REPS = 20
SIGNED = ["dn5", "up5", "dn22", "up22"]
CLEAN = ["sign1", "logabs1", "dn5", "up5", "dn22", "up22"]


def add_signed(d):
    d = d.copy()
    pos = d["r"].clip(lower=0.0) ** 2
    neg = d["r"].clip(upper=0.0) ** 2
    for w in (5, 22):
        d[f"dn{w}"] = np.log(neg.rolling(w).mean().clip(lower=1e-14))
        d[f"up{w}"] = np.log(pos.rolling(w).mean().clip(lower=1e-14))
    d["dn1"] = np.log(neg.clip(lower=1e-14))
    d["up1"] = np.log(pos.clip(lower=1e-14))
    d["sign1"] = np.sign(d["r"]).fillna(0.0)
    d["logabs1"] = np.log(d["r"].abs().clip(lower=1e-14))
    return d


def walk(d, cols, target, start, refit=REFIT):
    """Returns the variance level (with smearing) and the log-scale forecast."""
    y = d[target].to_numpy(float)
    X = d[cols].to_numpy(float)
    v = np.full(len(d), np.nan)
    p = np.full(len(d), np.nan)
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
        p[i:j] = np.column_stack([np.ones(j - i), X[i:j]]) @ b
        v[i:j] = np.exp(p[i:j]) * f
        i = j
    return v, p


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
    d = add_signed(H.load())
    arms = {
        "HAR": H.HAR,
        "HAR_SVU": H.HAR + ["dn1", "dn5", "dn22", "up1", "up5", "up22"],
        "HAR_SD": H.HAR + CLEAN,
    }
    rng = np.random.default_rng(20261021)
    for target, proxy in (("ly5", "y5"), ("ly22", "y22")):
        print(f"\n############ target = {target}  scored against {proxy} ############")
        for start in ("2019-01-01", "2022-01-01"):
            out = {n: walk(d, c, target, start) for n, c in arms.items()}
            m = np.isfinite(d[proxy]).to_numpy() & np.isfinite(d[target]).to_numpy() \
                & (d["date"] >= start).to_numpy()
            for v, p in out.values():
                m &= np.isfinite(v) & np.isfinite(p)
            y = d[proxy].to_numpy()[m]
            tgt = d[target].to_numpy()[m]
            per = {n: ql(out[n][0][m], y) for n in arms}
            print(f"  --- test from {start}   n={int(m.sum())} ---")
            for n in arms:
                v, p = out[n]
                q = per[n].mean()
                mse_log = float(np.mean((p[m] - tgt) ** 2))
                r2 = 1.0 - mse_log / np.var(tgt)
                print(f"    {n:<8} QLIKE={q:.5f}  MSE_log={mse_log:.5f}  R2_log={r2:+.4f}")
            for a in ("HAR_SVU", "HAR_SD"):
                diff = per[a] - per["HAR"]
                lo, hi = np.percentile(block_ci(diff), [2.5, 97.5])
                gain = 100.0 * (1.0 - per[a].mean() / per["HAR"].mean())
                star = " *" if hi < 0 else ("  (worse)" if lo > 0 else "")
                print(f"    {a} vs HAR  QLIKE gain={gain:+.2f}%  CI[{lo:+.6f},{hi:+.6f}]{star}")
            # per-year gains for the clean specification
            yrs = d["date"].dt.year.to_numpy()[m]
            print("    per-year QLIKE gain of HAR_SD over HAR:")
            row = []
            for yr in np.unique(yrs):
                k = yrs == yr
                if k.sum() < 40:
                    continue
                g = 100.0 * (1.0 - per["HAR_SD"][k].mean() / per["HAR"][k].mean())
                row.append(f"{yr}:{g:+.1f}%({int(k.sum())})")
            print("      " + "  ".join(row))
            # placebo for the clean specification
            block = d[CLEAN].to_numpy(float)
            gains = []
            for rep in range(REPS):
                perm = rng.permutation(len(d))
                dd = d.copy()
                for k, c in enumerate(CLEAN):
                    dd[c + "_pl"] = block[perm, k]
                v_pl, _ = walk(dd, H.HAR + [c + "_pl" for c in CLEAN], target, start)
                mm = m & np.isfinite(v_pl)
                gains.append(100.0 * (1.0 - np.mean(ql(v_pl[mm], d[proxy].to_numpy()[mm]))
                                      / np.mean(ql(out["HAR"][0][mm], d[proxy].to_numpy()[mm]))))
            gains = np.array(gains)
            real = 100.0 * (1.0 - per["HAR_SD"].mean() / per["HAR"].mean())
            p = 100.0 * float(np.mean(gains >= real))
            print(f"    placebo[HAR_SD] mean={gains.mean():+.2f}% sd={gains.std():.2f} "
                  f"max={gains.max():+.2f}  real={real:+.2f}%  "
                  f"beats {100-p:.0f}%  p={p/100:.2f}")


if __name__ == "__main__":
    main()
