"""Interval (distributional) forecasting experiment on GRN 2025 development.

Design is frozen in INTERVAL_FORECAST_PREREGISTRATION_ZH.md.

Common conditional location L_s = EWMA22 (the strongest point baseline) for all
three candidates, so every pinball difference is attributable to the predictive
distribution alone:

  A  iid           r_1 = y_1 / L_s, standardised, sd shrunk by 1/sqrt(h)
  B  global        resample the h-day residual ratio r_h = y_h / L_s
  C  state-cond.   resample r_h from the 100 training origins nearest in log L_s

Read-only over the archived development arrays. GRN 2026 is never touched.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DRAWS = 400
NEIGHBOURS = 100
SEED = 20261022
TAUS = np.array([0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975])
FINE = np.round(np.arange(0.005, 1.0, 0.005), 6)
HORIZONS = (1, 5, 22)


def pinball(y, q, tau):
    """y: (n,), q: (n,K), tau: (K,) -> (n,K) losses."""
    diff = y[:, None] - q
    return np.maximum(tau[None, :] * diff, (tau[None, :] - 1.0) * diff)


def main():
    g = dict(np.load(os.path.join(HERE, "grn_range_development.npz"), allow_pickle=False))
    origin = g["GRN_origin"]
    level = g["GRN_ewma22_all"][origin].astype(np.float64)   # (1386,) causal EWMA22
    y = g["GRN_y"].astype(np.float64)                        # (1386,3)
    tr = g["GRN_tr"].astype(np.int64)
    va = g["GRN_va"].astype(np.int64)

    ok = np.isfinite(level) & (level > 0)
    print("level>0: %d/%d ; train %d ; validation %d"
          % (ok.sum(), level.size, tr.size, va.size))
    assert ok[tr].all() and ok[va].all(), "non-positive EWMA22 level at an origin"

    L_tr, y_tr = level[tr], y[tr]
    L_va, y_va = level[va], y[va]
    r = y_tr / L_tr[:, None]                                 # (1158,3) residual ratios

    rng = np.random.default_rng(SEED)
    n = va.size

    # state-conditional neighbour index sets
    dist = np.abs(np.log(L_va)[:, None] - np.log(L_tr)[None, :])
    nn = np.argsort(dist, axis=1)[:, :NEIGHBOURS]

    samples, width = {}, {}
    for j, h in enumerate(HORIZONS):
        r_h = r[:, j]
        # A: iid -- h=1 residual shape, sd scaled by 1/sqrt(h)
        r1 = r[:, 0]
        m, s = r1.mean(), r1.std(ddof=1)
        z = (r1 - m) / s
        a = m + z[rng.integers(0, r1.size, size=(n, DRAWS))] * s / np.sqrt(h)

        # B: global h-day residual pool
        b = r_h[rng.integers(0, r_h.size, size=(n, DRAWS))]

        # C: same pool, but restricted to nearest neighbours in log level
        pick = rng.integers(0, NEIGHBOURS, size=(n, DRAWS))
        idx = nn[np.arange(n)[:, None], pick]
        c = r_h[idx]

        samples[h] = {}
        for name, ratio in (("A", a), ("B", b), ("C", c)):
            v = np.maximum(ratio, 0.0) * L_va[:, None]
            samples[h][name] = v
            width.setdefault(name, {})[h] = float(
                np.median(v.std(axis=1, ddof=1) / L_va))

    # ---- score ----
    out = {"scope": "GRN 2025 development interval experiment; preregistered",
           "n_origins": int(n), "draws": DRAWS, "neighbours": NEIGHBOURS, "seed": SEED,
           "taus": TAUS.tolist(), "models": {}}

    qA = {}
    for name in ("A", "B", "C"):
        qA[name] = {}
        for h in HORIZONS:
            qA[name][h] = np.quantile(samples[h][name], TAUS, axis=1).T   # (n,9)

    for name in ("A", "B", "C"):
        rec = {"pinball_by_h": {}, "coverage_by_h": {}, "crps_by_h": {},
               "median_relative_width": width[name]}
        all_pb = []
        for j, h in enumerate(HORIZONS):
            yh = y_va[:, j]
            q = qA[name][h]
            pb = pinball(yh, q, TAUS)
            rec["pinball_by_h"][str(h)] = float(pb.mean())
            all_pb.append(pb.mean(axis=1))
            rec["coverage_by_h"][str(h)] = {
                "50": float(((yh >= q[:, 3]) & (yh <= q[:, 5])).mean()),
                "80": float(((yh >= q[:, 1]) & (yh <= q[:, 7])).mean()),
                "95": float(((yh >= q[:, 0]) & (yh <= q[:, 8])).mean()),
            }
            qf = np.quantile(samples[h][name], FINE, axis=1).T
            rec["crps_by_h"][str(h)] = float(2.0 * pinball(yh, qf, FINE).mean())
        rec["pinball_pooled"] = float(np.mean(all_pb))
        out["models"][name] = rec

    # ---- preregistered paired test: C vs B at h=22 ----
    j22 = HORIZONS.index(22)
    y22 = y_va[:, j22]
    pbB = pinball(y22, qA["B"][22], TAUS).mean(axis=1)
    pbC = pinball(y22, qA["C"][22], TAUS).mean(axis=1)
    diff = pbC - pbB
    tstat = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(n)))
    out["C_vs_B_h22"] = {
        "mean_pinball_B": float(pbB.mean()), "mean_pinball_C": float(pbC.mean()),
        "paired_diff_C_minus_B": float(diff.mean()), "paired_t": tstat,
        "C_better": bool(diff.mean() < 0),
        "significant": bool(abs(tstat) > 2.0),
        "passes_prereg": bool(diff.mean() < 0 and abs(tstat) > 2.0),
    }
    # same test at the other horizons, reported for completeness
    out["C_vs_B_other_h"] = {}
    for j, h in enumerate(HORIZONS):
        yh = y_va[:, j]
        d = (pinball(yh, qA["C"][h], TAUS).mean(axis=1)
             - pinball(yh, qA["B"][h], TAUS).mean(axis=1))
        out["C_vs_B_other_h"][str(h)] = {
            "paired_diff_C_minus_B": float(d.mean()),
            "paired_t": float(d.mean() / (d.std(ddof=1) / np.sqrt(n))),
        }
    # preregistered demonstration: A must be badly overconfident at h=22
    out["H1_demo_A_h22_coverage95"] = out["models"]["A"]["coverage_by_h"]["22"]["95"]

    with open(os.path.join(HERE, "INTERVAL_FORECAST_GRN_RESULT.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print("\n%-4s %10s %10s %10s" % ("h", "A pinball", "B pinball", "C pinball"))
    for h in HORIZONS:
        print("%-4d %10.6f %10.6f %10.6f"
              % (h, out["models"]["A"]["pinball_by_h"][str(h)],
                 out["models"]["B"]["pinball_by_h"][str(h)],
                 out["models"]["C"]["pinball_by_h"][str(h)]))
    print("\n95%% coverage by horizon")
    for name in ("A", "B", "C"):
        print("  %s  " % name + "  ".join(
            "h%-2d=%.3f" % (h, out["models"][name]["coverage_by_h"][str(h)]["95"])
            for h in HORIZONS))
    print("\nmedian relative predictive sd")
    for name in ("A", "B", "C"):
        print("  %s  " % name + "  ".join(
            "h%-2d=%.4f" % (h, width[name][h]) for h in HORIZONS))
    print("\nC vs B at h=22: diff=%+.6f  t=%+.2f  -> %s"
          % (out["C_vs_B_h22"]["paired_diff_C_minus_B"], tstat,
             "PASS" if out["C_vs_B_h22"]["passes_prereg"] else "FAIL"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
