"""Write an explicit post-fit price-CRPS correction companion.

The frozen forecast files, model checkpoints, production scorer, original
tables, and primary normalized-CRPS results are never overwritten. All 624
continuous forecast artifacts are recomputed with stable_crps from the already
reviewed cancellation audit. The 66 empirical-atom controls are unchanged.

This is reuse of a qualified formula, not a new independent derivation and not
a new experiment. Output vectors are private; public output contains only cell
aggregates and provenance. The stable vectors remain binary64, with selected
80/120-digit and defining-CDF checks documented by the separate audit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

import lognormal_crps_cancellation_audit as numerical

STUDY = Path(__file__).resolve().parents[1]
SEEDS = (17, 41, 89)
PUBLIC_TABLE = STUDY / "audits/STABLE_PRICE_CRPS_CELLS.csv"
PUBLIC_RECEIPT = STUDY / "audits/STABLE_PRICE_CRPS_RECOMPUTATION.json"


def labels(path, kind):
    if kind == "persistence":
        pieces = path.stem.split("__")
        if len(pieces) == 4:
            panel, backbone, arm, seed = pieces
            return panel, backbone, arm, int(seed[1:])
        assert len(pieces) == 2 and pieces[1].startswith("statistical_")
        return pieces[0], "STATISTICAL", pieces[1], 0
    panel = path.parent.parent.name.rsplit("_h", 1)[0]
    family = path.parent.name
    seed = int(path.stem.split("_")[1])
    return panel, "GAS_BENCHMARK", family, seed


def centered_mean(values):
    values = np.asarray(values)
    return values[0] + np.mean(values - values[0], axis=0)


def write_npz_new(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def output_relative(path, kind):
    if kind == "persistence":
        return Path("persistence") / path.name
    return Path("gas") / path.parent.parent.name / path.parent.name / path.name


def self_test():
    cases = [
        ([.2, .5, .3], [-1., .2, 1.], [.3, .8, 2.5], 1.),
        ([.1, .3, .6], [-1., 0., 2.], [2., 8., 10.], .8),
        ([1.], [.3], [.7], 2.),
    ]
    rows = []
    for weights, means, sigmas, target in cases:
        w, m, s = [np.array(a, dtype=float) for a in (weights, means, sigmas)]
        value = numerical.stable_crps(np.array([target]), w[None], m[None], s[None])[0][0]
        reference = float(numerical.mp_crps(target, w, m, s, 80))
        integration, estimated = numerical.cdf_integral(target, w, m, s)
        assert not numerical.mismatch(np.array([value]), np.array([reference]))[0]
        assert not numerical.mismatch(np.array([integration]), np.array([reference]))[0]
        rows.append({"components": len(w), "stable_relative_error": abs(value - reference) / abs(reference),
                     "cdf_integral_relative_error": abs(integration - reference) / abs(reference)})
    original = np.array([.1, .2, .3, .7])
    assert np.array_equal(centered_mean(np.stack([original] * 3)), original)
    return {"passed": True, "market_files_read": 0, "model_fits": 0,
            "cases": rows, "identical_seed_mean_exact": True,
            "scope": "Synthetic unequal-log-means, wide-tail and single-component identities; no model selection"}


def recompute(candidate, gas, destination):
    started = time.perf_counter()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Correction archive must be new; no existing artifact is overwritten")
    if PUBLIC_TABLE.exists() or PUBLIC_RECEIPT.exists():
        raise FileExistsError("A public correction companion already exists")
    qualification_path = STUDY / "audits/LOGNORMAL_CRPS_CANCELLATION_AUDIT.json"
    qualification = json.loads(qualification_path.read_text())
    formula_path = Path(numerical.__file__)
    formula_hash = numerical.sha(formula_path)
    assert qualification["audit_sha256"] == formula_hash
    assert all(p["stable_error"]["within_original_tolerance"] for p in qualification["high_precision_points"])
    protected_sources = [STUDY / name for name in (
        "source/persistence_market.py", "source/run_persistence_market.py", "source/mixture_metrics.py",
        "source/aggregate_persistence_market.py", "provenance/MARKET_PROTOCOL.json")]
    protected_hashes = {str(p.relative_to(STUDY)): numerical.sha(p) for p in protected_sources}
    tasks = numerical.inventory(candidate, gas)
    groups = {}
    artifact_rows, skipped = [], []
    count_distributions = count_scores = point_violations = 0
    max_absolute = max_relative = 0.
    for ordinal, (path, kind, original_hash) in enumerate(tasks, 1):
        assert numerical.sha(path) == original_hash
        data = numerical.distributions(path, kind)
        input_locator = str(output_relative(path, kind))
        if not data:
            skipped.append({"artifact": input_locator, "original_sha256": original_hash,
                            "reason": "Empirical atoms; no lognormal formula or vector changed"})
            continue
        panel, backbone, arm, seed = labels(path, kind)
        private = {"original_artifact_sha256": np.array(original_hash),
                   "stable_formula_sha256": np.array(formula_hash),
                   "status": np.array("POST_FIT_NUMERICAL_CORRECTION_COMPANION"),
                   "original_artifact_locator": np.array(input_locator)}
        distributions = []
        with np.load(path, allow_pickle=False) as archive:
            for item in data:
                role, h = item["role"], item["horizon"]
                if kind == "persistence":
                    ids = archive[role + "_indices"].copy()
                    dates = archive[role + "_dates"].copy()
                    score_key = f"{role}_score_h{h}_price_CRPS"
                    private[role + "_indices"], private[role + "_dates"] = ids, dates
                else:
                    ids, dates = archive["origin"].copy(), archive["date"].copy()
                    score_key = "score_price_CRPS"
                    private["origin"], private["date"] = ids, dates
                corrected = numerical.stable_crps(item["target"], item["weights"], item["ell"], item["sd"])[0]
                assert np.isfinite(corrected).all() and np.all(corrected >= -numerical.ATOL)
                original = item["saved"]
                error = np.abs(corrected - original)
                bad = numerical.mismatch(original, corrected)
                max_absolute = max(max_absolute, float(error.max()))
                max_relative = max(max_relative, float(np.max(error / np.maximum(np.abs(corrected), 1e-300))))
                point_violations += int(bad.sum())
                count_distributions += 1
                count_scores += len(corrected)
                private[score_key] = corrected
                distributions.append({"role": role, "horizon": h, "n_origins": len(ids),
                                      "private_score_key": score_key})
                key = (kind, panel, backbone, arm, role, h)
                shared = groups.setdefault(key, {"ids": ids, "dates": dates, "seeds": [],
                                                 "original": [], "corrected": []})
                assert np.array_equal(ids, shared["ids"]) and np.array_equal(dates, shared["dates"])
                assert seed not in shared["seeds"]
                shared["seeds"].append(seed)
                shared["original"].append(original)
                shared["corrected"].append(corrected)
        relative = output_relative(path, kind)
        target = destination / relative
        write_npz_new(target, private)
        artifact_rows.append({"artifact": input_locator, "kind": kind, "panel": panel,
            "backbone_or_family_group": backbone, "arm_or_family": arm, "seed": seed,
            "original_sha256": original_hash, "correction_relative_path": str(relative),
            "correction_sha256": numerical.sha(target), "distributions": distributions})
        if ordinal % 50 == 0:
            print(json.dumps({"stage": "stable_price_companion", "inputs_considered": ordinal,
                              "continuous_archives_written": len(artifact_rows)}), flush=True)
    assert len(artifact_rows) == 624 and len(skipped) == 66
    assert count_distributions == 2314 and count_scores == 407134
    assert point_violations == qualification["counts"]["saved_vs_stable_violations"]
    rows = []
    for key, values in sorted(groups.items()):
        kind, panel, backbone, arm, role, h = key
        expected = SEEDS if (kind == "persistence" and backbone != "STATISTICAL") or arm.endswith("gru_gmm") else (0,)
        assert tuple(sorted(values["seeds"])) == expected
        original = centered_mean(np.stack(values["original"]))
        corrected = centered_mean(np.stack(values["corrected"]))
        old_mean, new_mean = float(original.mean()), float(corrected.mean())
        delta = new_mean - old_mean
        stratum = "additional_shenzhen" if panel.startswith("SHENZHEN_PLOS_") else "development"
        rows.append({"kind": kind, "panel": panel, "stratum": stratum, "backbone": backbone,
            "arm_or_family": arm, "role": role, "horizon": h, "seed_count": len(values["seeds"]),
            "n_origins": len(original), "frozen_price_CRPS_mean": old_mean,
            "stable_price_CRPS_mean": new_mean, "stable_minus_frozen_mean": delta,
            "stable_over_frozen_minus_one": new_mean / old_mean - 1 if old_mean > 0 else None,
            "mean_change_exceeds_original_numeric_tolerance": abs(delta) > numerical.ATOL + numerical.RTOL * abs(new_mean),
            "status": "POST_FIT_NUMERICAL_CORRECTION_COMPANION"})
    assert len(rows) == 934
    grouped = []
    for key in sorted({(r["kind"], r["stratum"], r["role"], r["horizon"]) for r in rows}):
        block = [r for r in rows if (r["kind"], r["stratum"], r["role"], r["horizon"]) == key]
        relative = [r["stable_over_frozen_minus_one"] for r in block if r["stable_over_frozen_minus_one"] is not None]
        grouped.append({"kind": key[0], "stratum": key[1], "role": key[2], "horizon": key[3],
            "model_panel_horizon_rows": len(block),
            "minimum_relative_mean_change": min(relative) if relative else None,
            "maximum_relative_mean_change": max(relative) if relative else None,
            "maximum_absolute_mean_change": max(abs(r["stable_minus_frozen_mean"]) for r in block),
            "mean_changes_exceeding_original_numeric_tolerance": sum(r["mean_change_exceeds_original_numeric_tolerance"] for r in block)})
    # End-of-run evidence that this correction did not alter original experiments.
    assert all(numerical.sha(path) == expected for path, _, expected in tasks)
    assert all(numerical.sha(STUDY / name) == expected for name, expected in protected_hashes.items())
    assert numerical.sha(formula_path) == formula_hash
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    with PUBLIC_TABLE.open("x", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())
    receipt = {"status": "POST_FIT_NUMERICAL_CORRECTION_COMPANION", "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": numerical.sha(__file__), "stable_formula_sha256": formula_hash,
        "numerical_qualification_sha256": numerical.sha(qualification_path),
        "protected_source_sha256": protected_hashes, "cell_table_sha256": numerical.sha(PUBLIC_TABLE),
        "all_690_original_prediction_hashes_rechecked_unchanged": True,
        "all_protected_source_hashes_rechecked_unchanged": True,
        "continuous_prediction_artifacts_corrected": 624, "atomic_artifacts_unchanged": 66,
        "origin_horizon_distributions": count_distributions, "price_CRPS_vector_values": count_scores,
        "public_cell_rows": len(rows), "original_numeric_tolerance": {"rtol": numerical.RTOL, "atol": numerical.ATOL},
        "frozen_values_outside_tolerance_against_stable": point_violations,
        "max_point_absolute_difference": max_absolute, "max_point_relative_difference": max_relative,
        "seed_aggregation": "First seed + mean(seed - first seed) per origin, then mean origins; scores are not CDF-ensembled",
        "aggregate_change_ranges": grouped,
        "model_fits_and_gradient_steps": 0, "test_parameter_selection_or_retuning": False,
        "original_score_vectors_and_tables_replaced": False,
        "primary_normalized_scores_checkpoints_and_decisions_modified": False,
        "numerical_scope": "Stable binary64 price CRPS only. Eight difficult cases were separately checked at 80/120 digits and by the CDF definition; these full vectors are not all recomputed at arbitrary precision",
        "author_role": "Numerical correction implementation reusing the reviewed audit formula; not a new independent derivation",
        "readme": "Use these separately labeled price-CRPS vectors/cell values as the numerical correction companion; retain the frozen original artifacts and qualification evidence. No new primary comparison or model claim is made",
        "continuous_artifacts": artifact_rows, "unchanged_atomic_artifacts": skipped,
        "elapsed_seconds": time.perf_counter() - started}
    numerical.save_new(PUBLIC_RECEIPT, receipt)
    numerical.save_new(destination / "MANIFEST.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistence-private", type=Path, default=Path("/tmp/carbon-persistence-market-runs"))
    parser.add_argument("--gas-private", type=Path, default=Path("/tmp/carbon-gas-normalization-run"))
    parser.add_argument("--private-output", type=Path, default=Path("/tmp/carbon-stable-price-crps-20261019b"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2, allow_nan=False))
        return
    result = recompute(args.persistence_private, args.gas_private, args.private_output)
    print(json.dumps({key: result[key] for key in ("status", "continuous_prediction_artifacts_corrected",
        "price_CRPS_vector_values", "public_cell_rows", "model_fits_and_gradient_steps")}), flush=True)


if __name__ == "__main__":
    main()
