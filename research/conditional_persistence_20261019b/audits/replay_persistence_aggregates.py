"""Internal independent statistical replay of the completed candidate study.

The reviewer did not write the candidate model, its market runner, or its
production aggregator. The reviewer did implement the separate GAS benchmark
and the shared mixture-score primitive; this replay therefore verifies stored
candidate scores and statistics, not an independently rederived density score.
No production aggregation functions are imported. No model is fitted.

All 312 neural fits, 13 references, COMPLETE manifests, checkpoints, forecasts,
and the complete public summary/table set must exist before any score is read.
Bootstrap means use origin multiplicity matrices rather than expanded sampled
score vectors. --self-test accesses no market artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
DEVELOPMENT = tuple(sorted(
    f"{m}_{y}" for m in ("EUA_LONG", "GUANGZHOU", "HUBEI", "SHANGHAI")
    for y in (2021, 2022, 2023) if (m, y) != ("SHANGHAI", 2021)))
ADDITIONAL = ("SHENZHEN_PLOS_2020", "SHENZHEN_PLOS_2021")
PANELS = tuple(sorted(DEVELOPMENT + ADDITIONAL))
STRATA = {"development": DEVELOPMENT, "additional_shenzhen": ADDITIONAL}
SEEDS = (17, 41, 89)
HORIZONS = (1, 5, 20)
NEURAL_ARMS = {
    "GRU16": ("conditional_joint005", "constant_joint005", "conditional_endpoint",
              "constant_endpoint", "scale_capacity_joint005", "independent_horizon_endpoint"),
    "MLP16": ("conditional_joint005", "constant_joint005"),
}
STATISTICAL_ARMS = ("statistical_endpoint", "statistical_joint005")
METRICS = (
    "normalized_CRPS", "normalized_NLL", "price_CRPS", "price_NLL",
    "mean_price_squared_error", "rw_mean_price_squared_error", "mean_price",
    "coverage90", "width90", "lower90", "upper90", "normalized_width90",
)
EXPECTED_AGGREGATOR_SHA = "7e8fd1865ffc3acebb5f174d9d088110307522f93c0cde62c79c3db0f7d05d87"
COMPLETE_OUTPUTS = ("SUMMARY.json", "CELL_METRICS.csv", "CELL_COMPARISONS.csv",
                    "PARAMETER_SUMMARIES.csv", "MODEL_FIT_ACCOUNTING.json", "INPUT_INTEGRITY.json")
ERRORS = Counter()
MAX_ERRORS = {}


def digest(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected a regular file: {path.name}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def expect(condition, label):
    if not condition:
        raise AssertionError(label)


def close(actual, expected, label, *, rtol=2e-12, atol=2e-12):
    actual, expected = float(actual), float(expected)
    expect(math.isfinite(actual) and math.isfinite(expected), "Nonfinite " + label)
    error = abs(actual - expected)
    MAX_ERRORS[label] = max(MAX_ERRORS.get(label, 0.0), error)
    expect(np.isclose(actual, expected, rtol=rtol, atol=atol),
           f"{label} differs: independent={actual!r}, published={expected!r}")
    ERRORS[label] += 1


def expected_neural_jobs():
    return {
        f"{panel}__{backbone}__{arm}__s{seed}": (panel, backbone, arm, seed)
        for panel in PANELS for backbone, arms in NEURAL_ARMS.items()
        for arm in arms for seed in SEEDS
    }


def inventory_before_scores(public, private, expected_aggregator_sha):
    """Reject incomplete experiments without looking at any performance field."""
    jobs = expected_neural_jobs()
    forecast_names = set(jobs) | {
        f"{p}__{arm}" for p in PANELS for arm in STATISTICAL_ARMS}
    inventories = (
        (public / "fits", "*.json", set(jobs)),
        (public / "references", "*.json", set(PANELS)),
        (private / "checkpoints", "*.pt", set(jobs)),
        (private / "forecasts", "*.npz", forecast_names),
        (private / "references", "*.json", set(PANELS)),
    )
    for directory, glob, expected in inventories:
        actual = {p.stem for p in directory.glob(glob)}
        expect(actual == expected,
               f"Incomplete inventory in {directory.name}: actual {len(actual)}, expected {len(expected)}")
    expect(len(jobs) == 312 and len(forecast_names) == 338, "Fixed experiment count mismatch")
    expect(all((public / name).is_file() for name in COMPLETE_OUTPUTS), "Complete public aggregation absent")
    starts = sorted(public.glob("START_*.json"))
    completes = sorted(public.glob("COMPLETE_*.json"))
    expect(len(completes) > 0 and {p.name.replace("COMPLETE_", "START_") for p in completes}
           == {p.name for p in starts}, "Incomplete shard manifests")
    manifests = [read_json(path) for path in completes]
    all_panels = [p for manifest in manifests for p in manifest["panels"]]
    expect(Counter(all_panels) == Counter(PANELS), "Shard panels are duplicated or missing")
    expect(sum(m["neural_fits_completed"] for m in manifests) == 312, "Incomplete neural count")
    expect(sum(m["static_fits_completed"] for m in manifests) == 39, "Incomplete static count")
    identity = manifests[0]["identity"]
    for manifest in manifests:
        expect(manifest["identity"] == identity, "Shard identity differs")
        expect(manifest["no_test_parameter_updates"] is True, "Unexpected test parameter updates")
    for path in completes:
        m = read_json(path)
        start = read_json(public / path.name.replace("COMPLETE_", "START_"))
        expect(start["identity"] == identity and start["panels"] == m["panels"], "START/COMPLETE mismatch")
    for name, sha in identity["source_sha256"].items():
        expect(digest(STUDY / name) == sha, "Frozen training source changed")
    expect(digest(STUDY / "provenance" / "MARKET_PROTOCOL.json") == identity["protocol_sha256"],
           "Frozen market protocol changed")
    aggregator = STUDY / "source" / "aggregate_persistence_market.py"
    expect(digest(aggregator) == expected_aggregator_sha, "Aggregator differs from reviewed final version")
    listed = {r["job"]: r["receipt_sha256"] for m in manifests for r in m["jobs"]}
    expect(set(listed) == set(jobs), "COMPLETE job coverage mismatch")
    expected_job_count = sum(len(m["jobs"]) for m in manifests)
    expect(expected_job_count == 312, "Duplicate COMPLETE jobs")
    fits, references, artifact_hashes = {}, {}, {}
    prepared_hashes = {}
    for key, labels in sorted(jobs.items()):
        path = public / "fits" / f"{key}.json"
        expect(digest(path) == listed[key], "Fit receipt changed since COMPLETE")
        r = read_json(path)
        expect(r["identity"] == identity and r["job"] == key, "Fit receipt identity mismatch")
        expect((r["panel"], r["fit"]["backbone"], r["fit"]["arm"], r["fit"]["seed"]) == labels,
               "Fit labels mismatch")
        expect(r["fit"]["epochs_run"] == 40 and len(r["fit"]["history"]) == 41, "Incomplete training")
        expect(r["fit"]["optimizer_steps"] == 40 * math.ceil(r["fit"]["train_origins"] / 64),
               "Incomplete optimizer updates")
        expect(r["test_parameter_updates"] == 0, "Unexpected test update")
        fits[key] = r
    for panel in PANELS:
        r = read_json(public / "references" / f"{panel}.json")
        expect(r["identity"] == identity and r["statistical_fits"] == 3, "Reference mismatch")
        expect(read_json(private / "references" / f"{panel}.json") == r["reference"], "Private reference mismatch")
        references[panel] = r
    for r in list(fits.values()) + list(references.values()):
        panel = r["prepared"]["panel"]
        ph = r["prepared"]["prepared_sha256"]
        expect(prepared_hashes.setdefault(panel, ph) == ph, "Models used different input bytes")
        for relative, sha in r["private_artifacts"].items():
            expect(digest(private / relative) == sha, "Saved artifact bytes changed")
            expect(artifact_hashes.setdefault(relative, sha) == sha, "Inconsistent artifact hashes")
    # Performance fields are accessed only after this complete inventory guard.
    summary = read_json(public / "SUMMARY.json")
    expect(summary["all_prespecified_fits_complete_before_analysis"] is True, "Premature production summary")
    expect(summary["identity"] == identity and summary["aggregator_sha256"] == expected_aggregator_sha,
           "Summary identity mismatch")
    b = summary["bootstrap"]
    expect((b["replicates"], b["block_length"], b["seed"]) == (1999, 40, 20261019),
           "Unexpected bootstrap specification")
    expect(b["panel_order"] == list(PANELS) and b["quantile_method"] == "linear",
           "Bootstrap ordering/quantile convention mismatch")
    return fits, references, summary, {
        "neural_receipts": len(fits), "reference_receipts": len(references),
        "checkpoints": 312, "forecast_files": 338, "static_fits": 39,
        "complete_manifests": len(completes), "private_artifacts_hash_checked": len(artifact_hashes),
        "identity": identity, "aggregator_sha256": expected_aggregator_sha,
        "summary_sha256": digest(public / "SUMMARY.json"),
    }


def stable_seed_average(vectors):
    arrays = np.asarray(vectors, dtype=np.float64)
    if len(arrays) == 1:
        return arrays[0].copy()
    expect(len(arrays) == 3, "Unexpected neural seed count")
    return arrays[0] + ((arrays[1] - arrays[0]) + (arrays[2] - arrays[0])) / 3.0


def load_test_scores(fits, references, private):
    pools, metadata = {}, {}
    file_jobs = []
    for key, r in sorted(fits.items()):
        f = r["fit"]
        file_jobs.append((key, r["panel"], f["backbone"], f["arm"], f["seed"],
                          r["metrics"]["test"], r["prepared"]["role_counts"]["test"]))
    for panel, r in sorted(references.items()):
        for mode in ("endpoint", "joint005"):
            arm = "statistical_" + mode
            file_jobs.append((f"{panel}__{arm}", panel, "STATISTICAL", arm, 0,
                              r["metrics"][mode]["test"], r["prepared"]["role_counts"]["test"]))
    for filename, panel, backbone, arm, seed, public_metrics, count in file_jobs:
        with np.load(private / "forecasts" / f"{filename}.npz", allow_pickle=False) as f:
            # Date and index arrays are loaded before any saved metric value.
            dates = f["test_dates"].copy()
            targets = f["test_target_dates"].copy()
            parsed, target_parsed = dates.astype("datetime64[D]"), targets.astype("datetime64[D]")
            cutoff = np.datetime64("2022-01-01" if panel.startswith("SHENZHEN_PLOS_") else "2024-01-01")
            expect(dates.shape == (count,) and targets.shape == (count, 3), "Date shape mismatch")
            expect(not np.isnat(parsed).any() and not np.isnat(target_parsed).any(), "Invalid dates")
            expect(np.all(parsed < cutoff) and np.all(target_parsed < cutoff), "Reserved date rejected")
            expect(np.all(np.diff(parsed).astype(int) > 0), "Dates duplicate or unordered")
            year = panel.rsplit("_", 1)[1]
            expect(np.all(np.char.startswith(dates.astype("U10"), year)) and
                   np.all(np.char.startswith(targets[:, 2].astype("U10"), year)),
                   "Test origin/maximum endpoint leaves the fixed annual block")
            origin, target_indices = f["test_indices"].copy(), f["test_target_indices"].copy()
            expect(np.issubdtype(origin.dtype, np.integer) and np.all(np.diff(origin) > 0), "Bad origin index")
            np.testing.assert_array_equal(target_indices, origin[:, None] + np.array(HORIZONS))
            current = (dates, targets, origin, target_indices)
            if panel not in metadata:
                metadata[panel] = current
            else:
                for a, b in zip(current, metadata[panel]):
                    np.testing.assert_array_equal(a, b)
            group = (panel, backbone, arm)
            storage = pools.setdefault(group, {})
            expect(seed not in storage, "Duplicate seed in score pool")
            storage[seed] = {}
            for h in HORIZONS:
                storage[seed][h] = {}
                expect(public_metrics[str(h)]["n"] == count, "Public origin count mismatch")
                for metric in METRICS:
                    values = np.asarray(f[f"test_score_h{h}_{metric}"], dtype=float)
                    expect(values.shape == (count,) and np.isfinite(values).all(), "Invalid saved score")
                    close(values.mean(), public_metrics[str(h)][metric], "per_fit_saved_score_means")
                    storage[seed][h][metric] = values.copy()
    averaged = {}
    for group, seed_scores in sorted(pools.items()):
        seeds = (0,) if group[1] == "STATISTICAL" else SEEDS
        expect(tuple(sorted(seed_scores)) == seeds, "Incomplete seed pool")
        averaged[group] = {
            h: {m: stable_seed_average([seed_scores[s][h][m] for s in seeds]) for m in METRICS}
            for h in HORIZONS
        }
    expect(len(averaged) == 130, "Expected ten model groups in every panel")
    return averaged, metadata


def multiplicity_bootstrap(scores):
    rng = np.random.default_rng(20261019)
    result = {}
    for panel in PANELS:
        groups = sorted(g for g in scores if g[0] == panel)
        columns = [(g, h) for g in groups for h in HORIZONS]
        matrix = np.column_stack([scores[g][h]["normalized_CRPS"] for g, h in columns])
        n = len(matrix)
        starts = rng.integers(0, n, size=(1999, (n + 39) // 40))
        selected = ((starts[:, :, None] + np.arange(40)) % n).reshape(1999, -1)[:, :n]
        multiplicities = np.vstack([np.bincount(row, minlength=n) for row in selected])
        expect(np.all(multiplicities.sum(axis=1) == n), "Bootstrap retained size changed")
        means = multiplicities @ matrix / n
        for j, key in enumerate(columns):
            result[key] = means[:, j]
    return result


def contrast_list():
    definitions = []
    for backbone, arms in NEURAL_ARMS.items():
        candidate = (backbone, "conditional_joint005")
        controls = [(backbone, a) for a in arms if a != candidate[1]]
        controls += [("STATISTICAL", "statistical_joint005"), ("STATISTICAL", "statistical_endpoint")]
        for control in controls:
            definitions.append((candidate, control))
    definitions += [
        (("GRU16", "conditional_endpoint"), ("GRU16", "constant_endpoint")),
        (("GRU16", "conditional_endpoint"), ("GRU16", "independent_horizon_endpoint")),
        (("GRU16", "constant_joint005"), ("GRU16", "constant_endpoint")),
    ]
    return [(f"{a[0]}__{a[1]}__vs__{b[0]}__{b[1]}", a, b) for a, b in definitions]


def replay_comparisons(scores, boot, summary):
    contrasts, cell_rows = {}, []
    expect(set(summary["comparisons"]) == {x[0] for x in contrast_list()}, "Unexpected comparison family")
    for name, candidate, control in contrast_list():
        for stratum, panels in STRATA.items():
            for label, horizons in (("h5_h20", (5, 20)), ("h1_secondary", (1,))):
                values, differences, draws, individual = [], [], [], {}
                secondary = {m: [] for m in ("price_CRPS_relative", "normalized_NLL_difference",
                                             "price_NLL_difference", "coverage90_difference",
                                             "normalized_width90_difference")}
                for panel in panels:
                    for h in horizons:
                        ag, bg = (panel, *candidate), (panel, *control)
                        a, b = scores[ag][h], scores[bg][h]
                        am, bm = a["normalized_CRPS"].mean(), b["normalized_CRPS"].mean()
                        expect(am > 0 and bm > 0 and np.all(boot[(bg, h)] > 0), "Nonpositive CRPS denominator")
                        relative = float(am / bm - 1)
                        values.append(relative)
                        differences.append(float(am - bm))
                        draws.append(boot[(ag, h)] / boot[(bg, h)] - 1)
                        individual[f"{panel}_h{h}"] = relative
                        secondary["price_CRPS_relative"].append(float(a["price_CRPS"].mean() / b["price_CRPS"].mean() - 1))
                        for m in ("normalized_NLL", "price_NLL", "coverage90", "normalized_width90"):
                            secondary[m + "_difference"].append(float(a[m].mean() - b[m].mean()))
                        cell_rows.append({
                            "comparison": name, "stratum": stratum, "panel": panel, "horizon": h,
                            "candidate_CRPS": float(am), "control_CRPS": float(bm),
                            "relative_CRPS_minus_one": relative, "CRPS_difference": float(am-bm),
                            "strictly_better": relative < 0, "exact_tie": relative == 0,
                            "n_origins": len(a["normalized_CRPS"]),
                        })
                v = np.asarray(values)
                distribution = np.stack(draws).mean(axis=0)
                ci = np.quantile(distribution, [.025, .975], method="linear")
                result = {
                    "comparison": name, "stratum": stratum, "horizon_group": label,
                    "cells": len(v), "mean_relative_CRPS_minus_one": float(v.mean()),
                    "mean_CRPS_difference": float(np.mean(differences)),
                    "better_cells": int(np.sum(v < 0)), "tie_cells": int(np.sum(v == 0)),
                    "worse_cells": int(np.sum(v > 0)), "strict_win_fraction": float(np.mean(v < 0)),
                    "bootstrap_nominal95_relative_CRPS": ci.tolist(),
                    "tiny_absolute_relative_lt_1e_minus12_including_zero": int(np.sum(np.abs(v) < 1e-12)),
                    "tiny_nonzero_absolute_relative_lt_1e_minus12": int(np.sum((v != 0) & (np.abs(v) < 1e-12))),
                }
                published = summary["comparisons"][name][stratum][label]
                for key in ("cells", "better_cells", "tie_cells", "worse_cells"):
                    expect(result[key] == published[key], "Strict count mismatch: " + name)
                for key in ("mean_relative_CRPS_minus_one", "mean_CRPS_difference", "strict_win_fraction"):
                    close(result[key], published[key], "comparison_point_statistics")
                expect(set(individual) == set(published["cell_relative_values"]), "Cell inventory mismatch")
                for key, value in individual.items():
                    close(value, published["cell_relative_values"][key], "per_cell_relative_CRPS")
                    expect((value < 0) == (published["cell_relative_values"][key] < 0),
                           "Per-cell strict winner differs")
                    expect((value == 0) == (published["cell_relative_values"][key] == 0),
                           "Per-cell exact tie differs")
                for value, target in zip(ci, published["bootstrap_nominal95_relative_CRPS"]):
                    close(value, target, "bootstrap_interval_endpoints", rtol=2e-11, atol=2e-13)
                for key, value in secondary.items():
                    close(np.mean(value), published["secondary_equal_cell_means"][key], "secondary_metric_differences")
                tiny = published["tiny_effect_description_only"]
                expect(result["tiny_absolute_relative_lt_1e_minus12_including_zero"]
                       == tiny["cells_below_threshold_including_exact_zero"], "Tiny-effect descriptive count mismatch")
                expect(result["tiny_nonzero_absolute_relative_lt_1e_minus12"]
                       == tiny["nonzero_cells_below_threshold"], "Tiny nonzero count mismatch")
                contrasts[(name, stratum, label)] = result
    expect(len(contrasts) == 52 and len(cell_rows) == 507, "Incomplete prespecified comparison replay")
    return contrasts, cell_rows


def audit_public_tables(public, scores, comparison_rows):
    with (public / "CELL_METRICS.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(g[0], g[1], g[2], h) for g in scores for h in HORIZONS}
    keys = [(r["panel"], r["backbone"], r["arm"], int(r["horizon"])) for r in rows]
    expect(set(keys) == expected and len(keys) == len(expected) == 390, "CELL_METRICS inventory mismatch")
    for row, key in zip(rows, keys):
        g, h = key[:3], key[3]
        expect(int(row["n_origins"]) == len(scores[g][h]["normalized_CRPS"]), "Cell count mismatch")
        expect(int(row["seed_count"]) == (1 if g[1] == "STATISTICAL" else 3), "Cell seed count mismatch")
        for metric in METRICS:
            close(scores[g][h][metric].mean(), row[metric], "cell_metric_means")
        denominator = scores[g][h]["rw_mean_price_squared_error"].mean()
        if denominator > 0:
            ratio = scores[g][h]["mean_price_squared_error"].mean() / denominator
            close(ratio, row["mean_price_MSE_over_RW"], "cell_price_MSE_ratios")
            close(ratio-1, row["mean_price_MSE_relative_RW_minus_one"], "cell_price_MSE_ratios")
        else:
            expect(row["mean_price_MSE_over_RW"] == "" and row["mean_price_MSE_relative_RW_minus_one"] == "",
                   "Zero-RW denominator must have no finite relative MSE")
    with (public / "CELL_COMPARISONS.csv").open(newline="") as handle:
        published = list(csv.DictReader(handle))
    by_key = {(r["comparison"], r["stratum"], r["panel"], int(r["horizon"])): r for r in published}
    expect(len(by_key) == len(published) == len(comparison_rows) == 507, "CELL_COMPARISONS inventory mismatch")
    for r in comparison_rows:
        p = by_key[(r["comparison"], r["stratum"], r["panel"], r["horizon"])]
        for key in ("candidate_CRPS", "control_CRPS", "relative_CRPS_minus_one", "CRPS_difference"):
            close(r[key], p[key], "cell_comparison_table")
        for key in ("strictly_better", "exact_tie"):
            expect(str(r[key]) == p[key], "CSV strict win/tie differs")


def replay_gates(contrasts, summary):
    def result(backbone, control, stratum="development"):
        name = f"{backbone}__conditional_joint005__vs__{backbone}__{control}"
        return contrasts[(name, stratum, "h5_h20")]
    main = result("GRU16", "constant_joint005")
    capacity = result("GRU16", "scale_capacity_joint005")
    mlp = result("MLP16", "constant_joint005")
    additional = result("GRU16", "constant_joint005", "additional_shenzhen")
    checks = {
        "GRU_development_primary_mean_negative": main["mean_relative_CRPS_minus_one"] < 0,
        "GRU_at_least_60_percent_cells_strictly_better": 5*main["better_cells"] >= 3*main["cells"],
        "GRU_development_bootstrap_upper_negative": main["bootstrap_nominal95_relative_CRPS"][1] < 0,
        "GRU_main_lower_than_scale_capacity_on_average": capacity["mean_relative_CRPS_minus_one"] < 0,
        "MLP_development_primary_mean_negative": mlp["mean_relative_CRPS_minus_one"] < 0,
        "Shenzhen_additional_primary_mean_nonpositive": additional["mean_relative_CRPS_minus_one"] <= 0,
    }
    original = summary["internal_continuation"]
    expect(set(checks) == set(original["checks"]), "Gate definitions differ")
    for key, value in checks.items():
        expect(bool(value) == original["checks"][key]["passed"], "Gate result differs: " + key)
    passed = all(checks.values())
    decision = "PASS_INTERNAL_CONTINUATION_CHECKS" if passed else "STOP_FIXED_CANDIDATE"
    expect(passed == original["all_passed"] and decision == original["decision"], "Continuation decision differs")
    expect([key for key, value in checks.items() if not value] == original["failed_checks"], "Failed gate list differs")
    return {"all_passed": passed, "decision": decision, "checks": checks,
            "passed_count": sum(checks.values()), "gate_count": 6,
            "rounding_or_tolerance_in_gate_decisions": False}, {
        "primary_GRU_development": main, "capacity_GRU_development": capacity,
        "replication_MLP_development": mlp, "additional_GRU_shenzhen": additional,
    }


def self_test():
    rng = np.random.default_rng(20261019)
    n = 113
    starts = rng.integers(0, n, size=(1999, (n+39)//40))
    selected = ((starts[:, :, None] + np.arange(40)) % n).reshape(1999, -1)[:, :n]
    counts = np.vstack([np.bincount(row, minlength=n) for row in selected])
    x = np.column_stack((np.linspace(.1, 2, n), np.cos(np.arange(n)) + 2))
    error = float(np.max(np.abs(counts @ x / n - x[selected].mean(axis=1))))
    expect(error < 2e-14, "Multiplicity reference self-test failed")
    base = np.array([.1, .3, 1/7, 1.0])
    np.testing.assert_array_equal(stable_seed_average([base, base, base]), base)
    print(json.dumps({"synthetic_only": True, "new_fits": 0,
                      "count_vs_expanded_mean_max_abs": error,
                      "identical_seeds_exact_tie": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", type=Path, default=Path("/tmp/carbon-persistence-market-runs"))
    parser.add_argument("--public-dir", type=Path, default=STUDY/"results/market/persistence")
    parser.add_argument("--output", type=Path, default=STUDY/"audits/PERSISTENCE_AGGREGATE_REPLAY.json")
    parser.add_argument("--expected-aggregator-sha", default=EXPECTED_AGGREGATOR_SHA)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    fits, references, summary, inventory = inventory_before_scores(
        args.public_dir, args.private_dir, args.expected_aggregator_sha)
    scores, metadata = load_test_scores(fits, references, args.private_dir)
    bootstrap = multiplicity_bootstrap(scores)
    contrasts, cell_rows = replay_comparisons(scores, bootstrap, summary)
    audit_public_tables(args.public_dir, scores, cell_rows)
    gates, key_results = replay_gates(contrasts, summary)
    expect(digest(STUDY/"source/aggregate_persistence_market.py") == args.expected_aggregator_sha,
           "Production source changed during replay")
    expect(digest(args.public_dir/"SUMMARY.json") == inventory["summary_sha256"],
           "Production summary changed during replay")
    report = {
        "status": "PASS_INTERNAL_INDEPENDENT_STATISTICAL_REPLAY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_role": "Did not author candidate model/market runner/production aggregator.",
        "other_roles_disclosed": ["Implemented the separate GAS benchmark",
                                 "Implemented the shared mixture-score primitive"],
        "independence_scope": "Internal independent aggregation of saved candidate scores, not an external replication or independent density-score derivation.",
        "production_statistical_functions_imported": False,
        "new_model_fits": 0,
        "private_numeric_inputs": "Saved test scores only; dates and origin/target indices checked before score access. No feature, target-price, distribution-parameter or checkpoint tensor arrays loaded.",
        "inventory": inventory, "auditor_script_sha256": digest(__file__),
        "all_312_fits_and_complete_summary_required_before_score_reads": True,
        "all_panels_have_common_origins_across_models_seeds_horizons": True,
        "test_origins_per_panel": {p: len(metadata[p][0]) for p in PANELS},
        "score_seed_average": "First seed plus average deviations; preserves exact identical-seed ties; no CDF ensemble.",
        "bootstrap": {"replicates": 1999, "block": 40, "seed": 20261019,
                      "panel_order": list(PANELS), "method": "origin-count matrix multiplication",
                      "all_horizons_and_models_paired_within_panel": True,
                      "quantile": "linear .025/.975", "nominal_only": True},
        "comparison_groups_verified": len(contrasts), "comparison_cells_verified": len(cell_rows),
        "cell_metric_rows_verified": 390,
        "numeric_checks_by_category": dict(ERRORS),
        "max_absolute_difference_by_category": MAX_ERRORS,
        "all_strict_wins_exact_ties_and_tiny_descriptive_counts_verified": True,
        "internal_continuation_recomputed": gates, **key_results,
        "all_recomputed_comparison_summaries": list(contrasts.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in (
        "status", "comparison_groups_verified", "comparison_cells_verified",
        "numeric_checks_by_category", "max_absolute_difference_by_category",
        "internal_continuation_recomputed")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
