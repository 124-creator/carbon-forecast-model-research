"""EXPLORATORY follow-up to the preregistered GRN interval experiment.

The preregistered run (interval_forecast_grn.py) was contaminated: it built
multiplicative residuals y_h / L_s, and in 2020 the EWMA22 level collapses to a
median of 2.9e-06, so a 15% share of the pool carried residuals up to 75x the
level. Global resampling therefore produced 637%-of-level predictive sd, and the
"calibrated" baseline was calibrated only by being uninformative.

This run repeats the same three candidates in LOG space, e_h = log(y_h) - log(L_s),
with a hurdle probability for the zero targets that only binds at h=1. It is
labelled exploratory because the design differs from the frozen preregistration;
nothing here can be reported as a preregistered confirmation.

Candidate A/B/C definitions, the tau grid, the fixed EWMA22 location, the 400
draws and the paired C-vs-B test are unchanged from the preregistration.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DRAWS = 400
SEED = 20261022
TAUS = np.array([0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975])
FINE = np.round(np.arange(0.005, 1.0, 0.005), 6)
HORIZONS = (1, 5, 22)
NEIGHBOUR_GRID = (50, 100, 200)
PRIMARY_K = 100


def pinball(y, q, tau):
    diff = y[:, None] - q
    return np.maximum(tau[None, :] * diff, (tau[None, :] - 1.0) * diff)


def bound_ratio(ratio):
    """Map a residual ratio to a sample, without flooring the evaluation target."""
    return np.maximum(ratio, 0.0)


def main():
    g = dict(np.load(os.path.join(HERE, "grn_range_development.npz"), allow_pickle=False))
    origin = g["GRN_origin"]
    level = g["GRN_ewma22_all"][origin].astype(np.float64)
    y = g["GRN_y"].astype(np.float64)
    tr = g["GRN_tr"].astype(np.int64)
    va = g["GRN_va"].astype(np.int64)
    assert np.isfinite(level).all() and (level > 0).all()

    L_tr, L_va = level[tr], level[va]
    y_tr, y_va = y[tr], y[va]
    logL_tr, logL_va = np.log(L_tr), np.log(L_va)
    year = g["GRN_date"][origin].astype("datetime64[Y]").astype(int) + 1970

    # hurdle probability per horizon, estimated on train only
    q_tr = (y_tr == 0).mean(axis=0)                            # (3,)
    q_va = (y_va == 0).mean(axis=0)

    # log residuals, nan where the target is zero
    pos = y_tr > 0
    e = np.full_like(y_tr, np.nan)
    e[pos] = np.log(y_tr[pos]) - np.repeat(logL_tr[:, None], 3, axis=1)[pos]

    rng = np.random.default_rng(SEED)
    n = va.size
    dist = np.abs(logL_va[:, None] - logL_tr[None, :])

    print("train hurdle rate", q_tr.round(4), " valid", q_va.round(4))
    print("pool share with logL within +-1 of the 2025 median: %.3f"
          % float((np.abs(logL_tr - np.median(logL_va)) < 1.0).mean()))

    out = {"scope": "EXPLORATORY log-space interval experiment on GRN 2025 development",
           "preregistered": False,
           "why_exploratory": "multiplicative ratio residuals of the frozen run were "
                              "contaminated by the 2020 near-zero EWMA level",
           "n_origins": int(n), "draws": DRAWS, "seed": SEED,
           "primary_k": PRIMARY_K, "taus": TAUS.tolist(), "models": {},
           "sensitivity_k": {}}

    def build(h, k):
        j = HORIZONS.index(h)
        pool = e[:, j]
        keep = np.isfinite(pool)
        pool = pool[keep]
        src = np.where(keep)[0]

        qh = q_tr[j]

        # A: h=1 shape, sd shrunk by 1/sqrt(h)
        e1 = e[:, 0]
        e1 = e1[np.isfinite(e1)]
        m, s = e1.mean(), e1.std(ddof=1)
        z = (e1 - m) / s
        ea = m + z[rng.integers(0, e1.size, size=(n, DRAWS))] * s / np.sqrt(h)

        # B: global pool of e_h
        eb = pool[rng.integers(0, pool.size, size=(n, DRAWS))]

        # B2: control -- global pool with 2019-2020 origins dropped, no state
        # conditioning at all. Separates "state conditioning" from "exclude the
        # outlier regime".
        keep2 = keep.copy()
        keep2[src] = year[src] >= 2021
        pool2 = e[keep2, j]
        eb2 = pool2[rng.integers(0, pool2.size, size=(n, DRAWS))]

        # C: pool restricted to the k nearest origins in log level
        nn = np.argsort(dist[:, src], axis=1)[:, :k]
        pick = rng.integers(0, k, size=(n, DRAWS))
        idx = nn[np.arange(n)[:, None], pick]
        ec = pool[idx]

        zero = rng.random((n, DRAWS)) < qh
        return {nm: np.where(zero, 0.0, L_va[:, None] * np.exp(v))
                for nm, v in (("A", ea), ("B", eb), ("B2", eb2), ("C", ec))}

    def score(samples, h):
        yh = y_va[:, HORIZONS.index(h)]
        rec = {}
        for nm, v in samples.items():
            q = np.quantile(v, TAUS, axis=1).T
            pb = pinball(yh, q, TAUS)
            qf = np.quantile(v, FINE, axis=1).T
            rec[nm] = {
                "pinball": float(pb.mean()),
                "pinball_over_mean_y": float(pb.mean() / yh.mean()),
                "crps_over_mean_y": float(2.0 * pinball(yh, qf, FINE).mean() / yh.mean()),
                "coverage": {
                    "50": float(((yh >= q[:, 3]) & (yh <= q[:, 5])).mean()),
                    "80": float(((yh >= q[:, 1]) & (yh <= q[:, 7])).mean()),
                    "95": float(((yh >= q[:, 0]) & (yh <= q[:, 8])).mean()),
                },
                "median_rel_sd": float(np.median(v.std(axis=1, ddof=1) / L_va)),
                "per_origin_pinball": pb.mean(axis=1),
            }
        return rec

    per_origin = {}
    for h in HORIZONS:
        rec = score(build(h, PRIMARY_K), h)
        per_origin[h] = {nm: rec[nm].pop("per_origin_pinball") for nm in ("A", "B", "B2", "C")}
        out["models"][str(h)] = rec

    for k in NEIGHBOUR_GRID:
        s = score(build(22, k), 22)["C"]
        out["sensitivity_k"][str(k)] = {
            "pinball_over_mean_y": s["pinball_over_mean_y"],
            "coverage95": s["coverage"]["95"],
            "median_rel_sd": s["median_rel_sd"]}

    # paired C vs B, per horizon
    out["C_vs_B"] = {}
    out["C_vs_B2"] = {}
    for h in HORIZONS:
        cov = out["models"][str(h)]
        for ref, tag in (("B", "C_vs_B"), ("B2", "C_vs_B2")):
            d = per_origin[h]["C"] - per_origin[h][ref]
            out[tag][str(h)] = {
                "paired_diff": float(d.mean()),
                "paired_t": float(d.mean() / (d.std(ddof=1) / np.sqrt(n))),
                "C_calibrated_95": bool(0.90 <= cov["C"]["coverage"]["95"] <= 0.99),
                "ref_calibrated_95": bool(0.90 <= cov[ref]["coverage"]["95"] <= 0.99),
            }

    with open(os.path.join(HERE, "INTERVAL_FORECAST_GRN_LOGSPACE.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print("\npinball / mean(y_h)   [lower is better]")
    print("%-4s %9s %9s %9s %9s" % ("h", "A", "B", "B2", "C"))
    for h in HORIZONS:
        r = out["models"][str(h)]
        print("%-4d %9.4f %9.4f %9.4f %9.4f"
              % (h, r["A"]["pinball_over_mean_y"], r["B"]["pinball_over_mean_y"],
                 r["B2"]["pinball_over_mean_y"], r["C"]["pinball_over_mean_y"]))
    print("\n95%% coverage")
    for h in HORIZONS:
        r = out["models"][str(h)]
        print("  h%-2d  A=%.3f  B=%.3f  B2=%.3f  C=%.3f"
              % (h, r["A"]["coverage"]["95"], r["B"]["coverage"]["95"],
                 r["B2"]["coverage"]["95"], r["C"]["coverage"]["95"]))
    print("\nmedian relative predictive sd")
    for h in HORIZONS:
        r = out["models"][str(h)]
        print("  h%-2d  A=%.3f  B=%.3f  B2=%.3f  C=%.3f"
              % (h, r["A"]["median_rel_sd"], r["B"]["median_rel_sd"],
                 r["B2"]["median_rel_sd"], r["C"]["median_rel_sd"]))
    print("\nC vs B   and   C vs B2 (exclude 2019-2020) paired:")
    for h in HORIZONS:
        d1 = out["C_vs_B"][str(h)]
        d2 = out["C_vs_B2"][str(h)]
        print("  h%-2d vsB diff=%+.8f t=%+.2f | vsB2 diff=%+.8f t=%+.2f  C_cal=%s B2_cal=%s"
              % (h, d1["paired_diff"], d1["paired_t"], d2["paired_diff"], d2["paired_t"],
                 d2["C_calibrated_95"], d2["ref_calibrated_95"]))
    print("\nk sensitivity at h=22:")
    for k, v in out["sensitivity_k"].items():
        print("  k=%-4s pinball/y=%.4f  cov95=%.3f  rel_sd=%.3f"
              % (k, v["pinball_over_mean_y"], v["coverage95"], v["median_rel_sd"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
