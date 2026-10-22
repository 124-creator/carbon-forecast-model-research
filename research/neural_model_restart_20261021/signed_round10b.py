"""
signed_round10b.py — where does the 7-10% actually come from?

The persistence of the two signed states is identical (AR(1) 0.979 for both at the
22-day scale, 0.508 against 0.597 at five days with a confidence interval spanning
zero), so the gain is not a decay-rate difference.

That leaves a puzzle. The pair (log sum, log ratio) is a bijection of the pair
(log upside, log downside), and lrv5 is already in HAR, so HAR plus asym5 ought to
span roughly the same space as HAR plus up5 and dn5. Yet the ratio version gains
0.25% and the free version gains 7.9%. Something else in the six-feature block is
doing the work, and the only features that are not near-equivalent to the HAR
aggregates are sign1 and logabs1, which together describe TODAY's return rather than
an average.

This is an ablation: each group of signed features is added on its own, then in
combination. Whichever group reproduces the gain is the mechanism, and the write-up
must name that rather than the whole block.
"""

import numpy as np

import har_x_round9 as H

REFIT = 22
GROUPS = {
    "sign1": ["sign1"],
    "logabs1": ["logabs1"],
    "sign+magn": ["sign1", "logabs1"],
    "dn/up 5": ["dn5", "up5"],
    "dn/up 22": ["dn22", "up22"],
    "dn/up 5+22": ["dn5", "up5", "dn22", "up22"],
    "sign+magn+5": ["sign1", "logabs1", "dn5", "up5"],
    "full": ["sign1", "logabs1", "dn5", "up5", "dn22", "up22"],
}


def extra(d):
    d = d.copy()
    pos = d["r"].clip(lower=0.0) ** 2
    neg = d["r"].clip(upper=0.0) ** 2
    for w in (5, 22):
        d[f"dn{w}"] = np.log(neg.rolling(w).mean().clip(lower=1e-14))
        d[f"up{w}"] = np.log(pos.rolling(w).mean().clip(lower=1e-14))
    d["sign1"] = np.sign(d["r"]).fillna(0.0)
    d["logabs1"] = np.log(d["r"].abs().clip(lower=1e-14))
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


def main():
    d = extra(H.load())
    for target, proxy in (("ly5", "y5"), ("ly22", "y22")):
        print(f"\n############ target = {target}  scored against {proxy} ############")
        for start in ("2019-01-01", "2022-01-01"):
            fc = {n: walk(d, H.HAR + g, target, start) for n, g in GROUPS.items()}
            fc["HAR"] = walk(d, H.HAR, target, start)
            m = np.isfinite(d[proxy]).to_numpy() & (d["date"] >= start).to_numpy()
            for v in fc.values():
                m &= np.isfinite(v)
            y = d[proxy].to_numpy()[m]
            base = ql(fc["HAR"][m], y).mean()
            print(f"  --- test from {start}   n={int(m.sum())}   HAR={base:.5f} ---")
            for n in GROUPS:
                q = ql(fc[n][m], y).mean()
                print(f"    {n:<14} QLIKE={q:.5f}   gain={100*(1-q/base):+.2f}%")


if __name__ == "__main__":
    main()
