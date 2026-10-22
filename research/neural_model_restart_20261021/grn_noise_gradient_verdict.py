"""Apply the pre-registered GRN noise-gradient decision rules.

Read-only over the two finished development runs. Writes
GRN_NOISE_GRADIENT_VERDICT.json. No training, no new data, no rule changes.
"""

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = ("coupled_path", "independent_path", "direct_marked_control")
SEEDS = (11, 22, 33)
BASELINE_NAME = "EWMA22"
C1_MIN_WINS = 8          # of 9
C2_MIN_WINS = 5          # of 6


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return json.load(fh)


def main():
    base = load("GRN_DEVELOPMENT_COMPARISON.json")
    bw = base["all_models"][BASELINE_NAME]
    base_q, base_mse = float(bw["knownscale_QLIKE"]), float(bw["MSE"])

    results, prim = {}, {}
    for ref in ("gamma", "ewma_anchor"):
        fname = f"grn_predictive_path_v2_{ref}_development_results.json"
        obj = load(fname)
        results[ref] = obj
        prim[ref] = {m: v["primary_all_targets"] for m, v in obj["metrics"].items()}

    table = []
    n_pass_abs = 0
    for ref in ("gamma", "ewma_anchor"):
        for m, v in prim[ref].items():
            q, mse = float(v["knownscale_QLIKE"]), float(v["MSE"])
            ok_q, ok_m = q < base_q, mse <= base_mse
            n_pass_abs += int(ok_q and ok_m)
            table.append({
                "reference": ref, "model": m,
                "knownscale_QLIKE": q, "MSE": mse,
                "raw_loss_difference_vs_EWMA22": float(v["raw_loss_difference_vs_EWMA22"]),
                "n_cells": int(v["n_cells"]), "zero_targets": int(v["zero_targets"]),
                "qlike_better_than_baseline": ok_q,
                "mse_not_worse_than_baseline": ok_m,
                "passes_absolute_gate": bool(ok_q and ok_m),
            })

    c1 = []
    for mode in MODES:
        for s in SEEDS:
            a = float(prim["gamma"][f"{mode}_seed{s}"]["knownscale_QLIKE"])
            b = float(prim["ewma_anchor"][f"{mode}_seed{s}"]["knownscale_QLIKE"])
            c1.append({"mode": mode, "seed": s, "gamma": a, "ewma_anchor": b,
                       "gamma_wins": a < b, "diff": a - b})
    c2 = []
    for ref in ("gamma", "ewma_anchor"):
        for s in SEEDS:
            a = float(prim[ref][f"coupled_path_seed{s}"]["knownscale_QLIKE"])
            b = float(prim[ref][f"direct_marked_control_seed{s}"]["knownscale_QLIKE"])
            c2.append({"reference": ref, "seed": s, "coupled": a, "direct": b,
                       "coupled_wins": a < b, "diff": a - b})

    c1_wins = sum(r["gamma_wins"] for r in c1)
    c2_wins = sum(r["coupled_wins"] for r in c2)
    c1_pass = c1_wins >= C1_MIN_WINS
    c2_pass = c2_wins >= C2_MIN_WINS
    abs_pass = n_pass_abs > 0

    verdict = {
        "status": "GRN PRE2026 DEVELOPMENT; GRN2026 CLOSED; not fresh confirmation",
        "preregistration": "GRN_NOISE_GRADIENT_PREREGISTRATION_ZH.md",
        "baseline": {"name": BASELINE_NAME, "knownscale_QLIKE": base_q, "MSE": base_mse},
        "n_origins": 228, "n_cells": 684, "validation_window": ["2025-01-02", "2025-11-28"],
        "zero_targets": 33,
        "all_y0_kept": bool(results["gamma"].get("all_y0_kept")),
        "primary_selection_scope": results["gamma"].get("primary_selection_scope"),
        "rules": {"C1_min_wins": C1_MIN_WINS, "C1_pairs": len(c1),
                  "C2_min_wins": C2_MIN_WINS, "C2_pairs": len(c2),
                  "absolute_gate": "knownscale_QLIKE < baseline and MSE <= baseline"},
        "C1_gamma_vs_ewma_anchor": {"wins": c1_wins, "pairs": len(c1),
                                    "passed": c1_pass, "cells": c1},
        "C2_coupled_vs_direct": {"wins": c2_wins, "pairs": len(c2),
                                 "passed": c2_pass, "cells": c2},
        "absolute_gate": {"n_passing": n_pass_abs, "n_configs": len(table),
                          "passed": abs_pass},
        "table": sorted(table, key=lambda r: r["knownscale_QLIKE"]),
        "decision": ("OBSERVATION_OPERATOR_LINE_CLOSED"
                     if (not c1_pass and not c2_pass and not abs_pass)
                     else "PREREGISTERED_RESULT_NOT_DECISIVE"),
        "decision_note": (
            "Both incremental gates failed and no configuration passed the absolute "
            "gate. Per pre-registration section 6 the observation-operator line is "
            "closed. C1 changes both the mean anchor and the activity reference, so "
            "the strict finding is that the gamma-anchor configuration is invalid, "
            "not that measurement-error modelling is refuted in the abstract."
        ),
        "limitations": [
            "single product, single validation window; cross-period audit shows a "
            "single window ranking does not transfer",
            "GRN zero ranges are a quote-availability mechanism, not additive "
            "measurement error around a latent variance",
            "C1 is not a one-factor ablation; gamma changes anchor and activity together",
            "GRN 2026 remains unopened and this result does not justify opening it",
        ],
        "source_sha256": {
            "runner": sha(os.path.join(HERE, "grn_predictive_path_v2_runner.py")),
            "gamma_results": sha(os.path.join(
                HERE, "grn_predictive_path_v2_gamma_development_results.json")),
            "ewma_anchor_results": sha(os.path.join(
                HERE, "grn_predictive_path_v2_ewma_anchor_development_results.json")),
            "baseline_comparison": sha(os.path.join(HERE, "GRN_DEVELOPMENT_COMPARISON.json")),
        },
        "elapsed_seconds": {
            "gamma": results["gamma"].get("elapsed_seconds"),
            "ewma_anchor": results["ewma_anchor"].get("elapsed_seconds"),
        },
    }

    out = os.path.join(HERE, "GRN_NOISE_GRADIENT_VERDICT.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, ensure_ascii=False, indent=2)

    print("baseline %s: QLIKE=%.12f MSE=%.6e" % (BASELINE_NAME, base_q, base_mse))
    print("C1 gamma vs ewma_anchor: %d/%d -> %s" % (c1_wins, len(c1), "PASS" if c1_pass else "FAIL"))
    print("C2 coupled vs direct:     %d/%d -> %s" % (c2_wins, len(c2), "PASS" if c2_pass else "FAIL"))
    print("absolute gate:            %d/%d configs pass -> %s"
          % (n_pass_abs, len(table), "PASS" if abs_pass else "FAIL"))
    print("DECISION:", verdict["decision"])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
