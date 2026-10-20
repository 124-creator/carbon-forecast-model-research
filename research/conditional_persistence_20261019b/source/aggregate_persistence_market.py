"""Aggregate the complete frozen persistence market experiment.

Implementation role, not independent audit. This module neither trains models
nor recomputes predictive-density scores. It verifies the complete receipt and
artifact set, checks stored score means against the fit receipts, and aggregates
the stored per-origin scores. Importing this module has no filesystem side
effects. --self-test uses only fabricated arrays and performs no market reads.

Formal analysis requires all 312 neural receipts, 13 reference receipts, 312
checkpoints, 338 forecast files, and COMPLETE manifests covering every fixed
panel exactly once. No partial-result mode or test-driven selection is provided.

Bootstrap convention: one default_rng(20261019), panels in lexicographic order,
1999 circular resamples of length n, block 40. Every horizon, model, and score
within a panel uses the same origin indices. Seeds are averaged as SCORES at
each origin before resampling; their distributions are not combined into a CDF.
Intervals are nominal, conditional on the fixed development/additional panels,
and do not establish transportability to a population of independent markets.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
DEVELOPMENT_PANELS = tuple(sorted(
    f"{market}_{year}" for market in ("EUA_LONG", "GUANGZHOU", "HUBEI", "SHANGHAI")
    for year in (2021, 2022, 2023) if (market, year) != ("SHANGHAI", 2021)))
ADDITIONAL_PANELS = ("SHENZHEN_PLOS_2020", "SHENZHEN_PLOS_2021")
PANELS = tuple(sorted(DEVELOPMENT_PANELS + ADDITIONAL_PANELS))
STRATA = {"development": DEVELOPMENT_PANELS, "additional_shenzhen": ADDITIONAL_PANELS}
ARMS = {
    "GRU16": ("conditional_joint005", "constant_joint005", "conditional_endpoint",
              "constant_endpoint", "scale_capacity_joint005", "independent_horizon_endpoint"),
    "MLP16": ("conditional_joint005", "constant_joint005"),
    "STATISTICAL": ("statistical_endpoint", "statistical_joint005"),
}
SEEDS = (17, 41, 89)
HORIZONS = (1, 5, 20)
ROLES = ("train", "validation", "test")
SCORE_NAMES = (
    "normalized_CRPS", "normalized_NLL", "price_CRPS", "price_NLL",
    "mean_price_squared_error", "rw_mean_price_squared_error", "mean_price",
    "coverage90", "width90", "lower90", "upper90", "normalized_width90",
)
PARAMETERS = ("f", "rho_h5", "rho_h20")
PARAMETER_COUNTS = {
    ("GRU16", "conditional_joint005"): 1042, ("GRU16", "constant_joint005"): 1025,
    ("GRU16", "conditional_endpoint"): 1042, ("GRU16", "constant_endpoint"): 1025,
    ("GRU16", "scale_capacity_joint005"): 1042, ("GRU16", "independent_horizon_endpoint"): 1059,
    ("MLP16", "conditional_joint005"): 1282, ("MLP16", "constant_joint005"): 1265,
}
BOOTSTRAP_REPLICATES = 1999
BOOTSTRAP_BLOCK = 40
BOOTSTRAP_SEED = 20261019
OUTPUT_NAMES = (
    "SUMMARY.json", "CELL_METRICS.csv", "CELL_COMPARISONS.csv",
    "PARAMETER_SUMMARIES.csv", "MODEL_FIT_ACCOUNTING.json", "INPUT_INTEGRITY.json",
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_array(value, label, shape=None):
    array = np.asarray(value, dtype=np.float64)
    require(np.all(np.isfinite(array)), f"Nonfinite {label}")
    if shape is not None:
        require(array.shape == shape, f"Incorrect shape for {label}")
    return array


def same_array(actual, expected, label):
    require(np.array_equal(actual, expected), f"Pairing mismatch: {label}")


def same_float(actual, expected, label):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)),
            f"Nonfinite scalar: {label}")
    require(np.isclose(actual, expected, rtol=2e-12, atol=2e-11),
            f"Stored aggregate mismatch: {label}")


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Expected regular JSON file: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def job_key(panel, backbone, arm, seed):
    return f"{panel}__{backbone}__{arm}__s{seed}"


def expected_jobs():
    return {job_key(panel, backbone, arm, seed): (panel, backbone, arm, seed)
            for panel in PANELS for backbone in ("GRU16", "MLP16")
            for arm in ARMS[backbone] for seed in SEEDS}


def stratum_for(panel):
    return "development" if panel in DEVELOPMENT_PANELS else "additional_shenzhen"


def validate_protocol(path):
    protocol = read_json(path)
    require(protocol["synthetic_decision"] == "PASS_FINITE_SYNTHETIC_RESOURCE_GATE",
            "Synthetic gate did not authorize the market study")
    require(protocol["training"]["planned_gradient_fits"] == 312 and
            protocol["training"]["planned_static_fits"] == 39,
            "This aggregator requires the completed 13-panel, 312-fit experiment")
    require(protocol["training"]["seeds"] == list(SEEDS) and
            protocol["training"]["epochs"] == 40, "Training protocol mismatch")
    require(protocol["model"]["horizons_evaluated"] == list(HORIZONS), "Horizon protocol mismatch")
    for backbone in ("GRU16", "MLP16"):
        require(protocol["model"]["arms_" + backbone] == list(ARMS[backbone]),
                "Arm protocol mismatch")
    require(protocol["data"]["additional_panels"] == list(ADDITIONAL_PANELS),
            "Additional-stratum protocol mismatch")
    return protocol


def complete_inventory(public_dir, private_dir, protocol_path):
    """Fail on incompleteness before loading any forecast score arrays."""
    jobs = expected_jobs()
    require(len(jobs) == 312, "Incorrect internal experiment inventory")
    fit_paths = {p.stem: p for p in (public_dir / "fits").glob("*.json")}
    reference_paths = {p.stem: p for p in (public_dir / "references").glob("*.json")}
    checkpoint_paths = {p.stem: p for p in (private_dir / "checkpoints").glob("*.pt")}
    forecast_paths = {p.stem: p for p in (private_dir / "forecasts").glob("*.npz")}
    private_reference_paths = {p.stem: p for p in (private_dir / "references").glob("*.json")}
    forecast_names = set(jobs) | {f"{panel}__statistical_{mode}"
                                  for panel in PANELS for mode in ("endpoint", "joint005")}
    inventories = ((set(fit_paths), set(jobs), "neural receipts"),
                   (set(reference_paths), set(PANELS), "reference receipts"),
                   (set(checkpoint_paths), set(jobs), "checkpoints"),
                   (set(forecast_paths), forecast_names, "forecast files"),
                   (set(private_reference_paths), set(PANELS), "private references"))
    for actual, expected, label in inventories:
        require(actual == expected,
                f"Incomplete or unexpected {label}: actual={len(actual)}, expected={len(expected)}, "
                f"missing={len(expected - actual)}, extra={len(actual - expected)}; no analysis performed")
    complete_paths = sorted(public_dir.glob("COMPLETE_*.json"))
    start_paths = sorted(public_dir.glob("START_*.json"))
    require(complete_paths and {p.name.replace("COMPLETE_", "START_") for p in complete_paths}
            == {p.name for p in start_paths}, "Every started shard must have a COMPLETE manifest")
    manifests = [read_json(p) for p in complete_paths]
    starts = [read_json(p) for p in start_paths]
    identity = manifests[0]["identity"]
    require(identity["protocol_sha256"] == sha(protocol_path), "Protocol hash changed")
    expected_sources = {"source/run_persistence_market.py", "source/persistence_market.py",
                        "source/persistence.py", "source/mixture_metrics.py",
                        "data_sources/prepare_data.py"}
    require(set(identity["source_sha256"]) == expected_sources, "Unexpected frozen source identity")
    for name, digest in identity["source_sha256"].items():
        require(sha(STUDY / name) == digest, f"Frozen source bytes changed: {name}")
    manifest_panels, manifest_jobs = [], []
    manifest_receipt_hashes = {}
    for path, manifest in zip(complete_paths, manifests):
        require(manifest["identity"] == identity, "Inconsistent completed-shard identity")
        panels = manifest["panels"]
        require(len(panels) == len(set(panels)), "Duplicate panel within a shard")
        require(manifest["neural_fits_completed"] == 24 * len(panels) and
                manifest["static_fits_completed"] == 3 * len(panels), "Incomplete shard fit counts")
        require(manifest["no_test_parameter_updates"] is True, "Unexpected test updates")
        shard_jobs = {key for key, values in jobs.items() if values[0] in panels}
        listed = [row["job"] for row in manifest["jobs"]]
        require(set(listed) == shard_jobs and len(listed) == len(shard_jobs),
                "Shard job manifest does not match its panel set")
        manifest_panels.extend(panels)
        manifest_jobs.extend(listed)
        manifest_receipt_hashes.update({row["job"]: row["receipt_sha256"] for row in manifest["jobs"]})
        start = read_json(public_dir / path.name.replace("COMPLETE_", "START_"))
        require(start["identity"] == identity and start["panels"] == panels,
                "START and COMPLETE disagree")
    require(Counter(manifest_panels) == Counter(PANELS) and Counter(manifest_jobs) == Counter(jobs.keys()),
            "Completed shards must cover each fixed panel and job exactly once")
    fits = {key: read_json(path) for key, path in sorted(fit_paths.items())}
    references = {panel: read_json(path) for panel, path in sorted(reference_paths.items())}
    artifact_hashes, prepared_hashes = {}, {}
    for key, receipt in list(fits.items()) + list(references.items()):
        require(receipt["identity"] == identity, "Receipt source identity mismatch")
        panel = receipt["prepared"]["panel"]
        require(panel in PANELS, "Receipt names an unregistered panel")
        prepared = receipt["prepared"]["prepared_sha256"]
        require(prepared_hashes.setdefault(panel, prepared) == prepared,
                "Matched models used different prepared arrays")
        if key in jobs:
            panel_expected, backbone, arm, seed = jobs[key]
            require(receipt["job"] == key and receipt["panel"] == panel_expected and panel == panel_expected,
                    "Neural receipt job mismatch")
            require(receipt["market_neural_fits"] == 1 and receipt["test_parameter_updates"] == 0,
                    "Unexpected neural-fit or test-update count")
            require(receipt["shared_path_distribution"] == (arm != "independent_horizon_endpoint"),
                    "Incorrect shared-path claim in receipt")
            require((receipt["fit"]["backbone"], receipt["fit"]["arm"], receipt["fit"]["seed"])
                    == (backbone, arm, seed), "Receipt fit labels mismatch")
            expected_artifacts = {f"checkpoints/{key}.pt", f"forecasts/{key}.npz"}
            require(sha(fit_paths[key]) == manifest_receipt_hashes[key], "Fit receipt changed after COMPLETE")
        else:
            require(key == panel and receipt["statistical_fits"] == 3 and
                    receipt["fitting_scope"] == "train only", "Invalid reference receipt")
            require(read_json(private_dir / "references" / f"{panel}.json") == receipt["reference"],
                    "Private reference parameters and public receipt disagree")
            expected_artifacts = {f"references/{panel}.json", f"forecasts/{panel}__statistical_endpoint.npz",
                                  f"forecasts/{panel}__statistical_joint005.npz"}
        require(set(receipt["private_artifacts"]) == expected_artifacts, "Incorrect artifact inventory")
        for relative, digest in receipt["private_artifacts"].items():
            artifact = private_dir / relative
            require(artifact.is_file() and not artifact.is_symlink(), "Private artifact must be a regular file")
            require(sha(artifact) == digest, f"Private artifact hash mismatch: {relative}")
            artifact_hashes[relative] = digest
    integrity = {
        "all_expected_inputs_present_before_score_loading": True,
        "neural_receipts": 312, "reference_receipts": 13, "statistical_optimizer_fits": 39,
        "checkpoints": 312, "forecast_files": 338, "private_reference_files": 13,
        "completed_shards": len(manifests), "identity": identity,
        "prepared_sha256_by_panel": prepared_hashes,
        "public_input_sha256": {str(p.relative_to(public_dir)): sha(p)
            for p in sorted(list(fit_paths.values()) + list(reference_paths.values()) + complete_paths + start_paths)},
        "private_artifact_sha256": artifact_hashes,
        "score_recomputation": "None; saved per-origin scores are aggregated and checked against public means",
        "implementation_is_not_independent_audit": True,
    }
    return fits, references, manifests, starts, integrity


def validate_fit_history(fit, key):
    n_train, n_validation = fit["train_origins"], fit["validation_origins"]
    require(n_train >= 128 and n_validation == 126, "Incorrect training/validation sample size")
    expected_steps = 40 * math.ceil(n_train / 64)
    require(fit["epochs_run"] == 40 and fit["batch_size"] == 64 and
            fit["optimizer_steps"] == expected_steps, "Incomplete fixed training budget")
    history = fit["history"]
    require(len(history) == 41 and [r["epoch"] for r in history] == list(range(41)),
            "Training receipt must contain epochs 0 through 40")
    for epoch, row in enumerate(history):
        require(row["optimizer_steps"] == epoch * math.ceil(n_train / 64),
                "Incorrect epoch optimizer-step history")
        for role in ("training", "validation"):
            value = row[role]
            require(set(value["CRPS_h"]) == {"1", "5", "20"}, "Incomplete per-horizon epoch history")
            finite_array([value["objective"], value["endpoint_CRPS"], *value["CRPS_h"].values()],
                         f"{key} epoch {epoch} {role} history")
            same_float(value["endpoint_CRPS"], np.mean(list(value["CRPS_h"].values())),
                       "equal-horizon validation/training score")
            if fit["auxiliary_weight"]:
                finite_array([value["path_NLL"]], "joint path-likelihood history")
                same_float(value["objective"], value["endpoint_CRPS"] + fit["auxiliary_weight"] * value["path_NLL"],
                           "joint objective accounting")
            else:
                require(value["path_NLL"] is None, "Endpoint-only history should not claim path NLL")
                same_float(value["objective"], value["endpoint_CRPS"], "endpoint objective accounting")
    validation_scores = [float(r["validation"]["endpoint_CRPS"]) for r in history]
    selected = min(range(41), key=lambda epoch: validation_scores[epoch])
    require(fit["selected_epoch"] == selected and fit["epoch_zero_selected"] == (selected == 0),
            "Selected checkpoint is not the earliest minimum validation CRPS")
    require(fit["selected_validation_CRPS"] == validation_scores[selected],
            "Selected validation score does not match history")
    require(fit["training_complete_before_validation_selected_state_reload"] is True and
            fit["test_used_for_selection"] is False, "Invalid training/selection scope")
    require(fit["initial_f_absolute_error"] <= 1e-13 and
            fit["initial_rho_absolute_error"] <= 1e-13, "Incorrect zero-head reference initialization")
    require(math.isfinite(fit["elapsed_seconds"]) and fit["elapsed_seconds"] >= 0,
            "Invalid training runtime")


def fit_accounting(fits, references, manifests, starts):
    rows, static_rows = [], []
    for key, receipt in sorted(fits.items()):
        fit = receipt["fit"]
        n_train, n_validation = fit["train_origins"], fit["validation_origins"]
        require(receipt["prepared"]["role_counts"]["train"] == n_train and
                receipt["prepared"]["role_counts"]["validation"] == n_validation,
                "Fit and prepared role counts disagree")
        validate_fit_history(fit, key)
        require(fit["parameter_count"] == PARAMETER_COUNTS[(fit["backbone"], fit["arm"])],
                "Unexpected model capacity in fitted receipt")
        rows.append({"job": key, "panel": receipt["panel"], "stratum": stratum_for(receipt["panel"]),
            **{name: fit[name] for name in ("backbone", "arm", "seed", "train_origins", "validation_origins",
                "epochs_run", "optimizer_steps", "selected_epoch", "epoch_zero_selected", "parameter_count",
                "auxiliary_weight", "elapsed_seconds")}})
    for panel, receipt in sorted(references.items()):
        reference = receipt["reference"]
        require(reference["statistical_fits"] == 3 and len(reference["receipts"]) == 3,
                "Each panel requires exactly three shared reference optimizers")
        for optimization in reference["receipts"]:
            require(optimization["restarts"] == 0, "Unplanned static-optimizer restart")
            finite_array([optimization["fun"], optimization["initial_fun"], optimization["elapsed_seconds"]],
                         "static optimizer diagnostics")
            require(optimization["training_origins"] == receipt["prepared"]["role_counts"]["train"],
                    "Reference fitting scope mismatch")
            static_rows.append({"panel": panel, "stratum": stratum_for(panel),
                **{name: optimization[name] for name in ("estimator", "success", "status", "message", "nit",
                     "nfev", "initial_fun", "fun", "restarts", "elapsed_seconds")}})
    require(sum(m["optimizer_steps"] for m in manifests) == sum(r["optimizer_steps"] for r in rows),
            "COMPLETE and fit receipts disagree on optimizer steps")
    require(sum(m["epochs_completed"] for m in manifests) == 312 * 40, "Incomplete epoch accounting")
    require(sum(m["epoch_zero_selected"] for m in manifests) == sum(r["epoch_zero_selected"] for r in rows),
            "COMPLETE and fit receipts disagree on epoch-zero selections")
    groups = []
    for group_key in sorted({(r["stratum"], r["backbone"], r["arm"]) for r in rows}):
        selected = [r for r in rows if (r["stratum"], r["backbone"], r["arm"]) == group_key]
        groups.append({"stratum": group_key[0], "backbone": group_key[1], "arm": group_key[2],
            "fits": len(selected), "epoch_zero_selected": sum(r["epoch_zero_selected"] for r in selected),
            "selected_epoch_counts": dict(sorted(Counter(r["selected_epoch"] for r in selected).items())),
            "optimizer_steps": sum(r["optimizer_steps"] for r in selected),
            "summed_fit_seconds": sum(r["elapsed_seconds"] for r in selected)})
    first_start = min(datetime.fromisoformat(s["started_utc"]) for s in starts)
    last_complete = max(datetime.fromisoformat(m["completed_utc"]) for m in manifests)
    return {"neural_fits": len(rows), "statistical_optimizer_fits": len(static_rows),
        "shared_reference_panels": len(references), "epochs_per_neural_fit": 40,
        "neural_epochs_completed": sum(r["epochs_run"] for r in rows),
        "optimizer_steps": sum(r["optimizer_steps"] for r in rows),
        "epoch_zero_selected": sum(r["epoch_zero_selected"] for r in rows),
        "epoch_zero_interpretation": "Validation selected the initialized reference after all 40 epochs were actually fitted; this is not zero training",
        "summed_neural_fit_seconds": sum(r["elapsed_seconds"] for r in rows),
        "summed_statistical_optimizer_seconds": sum(r["elapsed_seconds"] for r in static_rows),
        "summed_shard_elapsed_seconds": sum(m["elapsed_seconds"] for m in manifests),
        "calendar_span_first_start_to_last_complete_seconds": (last_complete - first_start).total_seconds(),
        "runtime_note": "Summed fit/shard times overlap across parallel shards; calendar span includes prediction, I/O and gaps",
        "static_success_false_count": sum(not r["success"] for r in static_rows),
        "static_status_policy": "Fixed-budget finite valid solutions are retained with actual status; success=False does not imply a retry or a global optimum",
        "groups": groups, "neural_jobs": rows, "static_optimizers": static_rows}


def load_scores(fits, references, private_dir, panels=PANELS):
    """Read only dates, indices, parameters and saved scores, never prices/targets."""
    totals, deviations, sample_counts = {}, {}, {}
    parameter_values = defaultdict(list)
    origins = {}
    score_means_verified = 0
    jobs = []
    for key, receipt in sorted(fits.items()):
        fit = receipt["fit"]
        jobs.append((receipt["panel"], fit["backbone"], fit["arm"], key, receipt["metrics"], receipt["prepared"]))
    for panel, receipt in sorted(references.items()):
        for mode in ("endpoint", "joint005"):
            jobs.append((panel, "STATISTICAL", "statistical_" + mode,
                         f"{panel}__statistical_{mode}", receipt["metrics"][mode], receipt["prepared"]))
    for panel, backbone, arm, key, public_metrics, prepared in jobs:
        group = (panel, backbone, arm)
        if group not in totals:
            totals[group] = {h: {name: None for name in SCORE_NAMES} for h in HORIZONS}
            deviations[group] = {h: {name: None for name in SCORE_NAMES} for h in HORIZONS}
            sample_counts[group] = 0
        with np.load(private_dir / "forecasts" / f"{key}.npz", allow_pickle=False) as archive:
            # Check every stored origin/target date before loading any scores.
            cutoff = np.datetime64("2022-01-01" if panel.startswith("SHENZHEN_PLOS_") else "2024-01-01")
            role_metadata = {}
            for role in ROLES:
                dates = archive[f"{role}_dates"].copy()
                parsed = np.asarray(dates, dtype="datetime64[D]")
                require(parsed.ndim == 1 and len(parsed) and not np.any(np.isnat(parsed)) and
                        np.all(parsed < cutoff) and np.all(parsed[1:] > parsed[:-1]),
                        "Invalid or reserved origin dates in forecast archive")
                role_metadata[role] = {"dates": dates}
                if role != "train":
                    targets = archive[f"{role}_target_dates"].copy()
                    target_parsed = np.asarray(targets, dtype="datetime64[D]")
                    require(targets.shape == (len(dates), 3) and not np.any(np.isnat(target_parsed)) and
                            np.all(target_parsed < cutoff), "Invalid or reserved target dates")
                    role_metadata[role]["target_dates"] = targets
            for role in ROLES:
                metadata = role_metadata[role]
                metadata["indices"] = archive[f"{role}_indices"].copy()
                n = len(metadata["indices"])
                require(n == prepared["role_counts"][role] and len(metadata["dates"]) == n,
                        "Forecast/prepared role-count mismatch")
                require(np.issubdtype(metadata["indices"].dtype, np.integer) and
                        np.all(np.diff(metadata["indices"]) > 0), "Invalid origin index sequence")
                if role != "train":
                    metadata["target_indices"] = archive[f"{role}_target_indices"].copy()
                    same_array(metadata["target_indices"], metadata["indices"][:, None] + np.array(HORIZONS),
                               "horizon maturity indices")
                shared = origins.setdefault((panel, role), metadata)
                for name, value in metadata.items():
                    same_array(value, shared[name], f"{panel}/{role}/{name}")
                for name in PARAMETERS:
                    value = finite_array(archive[f"{role}_{name}"], f"{key}/{role}/{name}", (n,))
                    require(np.all(value > 0) if name == "f" else np.all((value >= 0) & (value <= 1)),
                            "Invalid forecast parameter domain")
                    parameter_values[(panel, backbone, arm, role, name)].append(value.copy())
                    descriptive = public_metrics["train_parameters" if role == "train" else role + "_parameters"][name]
                    for label, actual in (("mean", value.mean()), ("std", value.std()),
                                          ("minimum", value.min()), ("maximum", value.max())):
                        same_float(actual, descriptive[label], f"{key}/{role}/{name}/{label}")
                if role == "train":
                    require(not any(name.startswith("train_score_") for name in archive.files),
                            "Unexpected training-side market scoring archive")
                    continue
                for h in HORIZONS:
                    for part in ("weights", "means", "sigmas"):
                        require(f"{role}_{part}_h{h}" in archive.files, "Incomplete saved predictive distribution")
                    require(public_metrics[role][str(h)]["n"] == n, "Public metric origin-count mismatch")
                    for name in SCORE_NAMES:
                        value = finite_array(archive[f"{role}_score_h{h}_{name}"],
                                             f"{key}/{role}/h{h}/{name}", (n,))
                        same_float(value.mean(), public_metrics[role][str(h)][name],
                                   f"{key}/{role}/h{h}/{name}")
                        if name == "coverage90":
                            require(np.all((value == 0) | (value == 1)), "Coverage indicator must be binary")
                        score_means_verified += 1
                        if role == "test":
                            first = totals[group][h][name]
                            if first is None:
                                totals[group][h][name] = value.copy()
                                deviations[group][h][name] = np.zeros_like(value)
                            else:
                                deviations[group][h][name] += value - first
            sample_counts[group] += 1
    for group, horizons in totals.items():
        expected_count = 1 if group[1] == "STATISTICAL" else len(SEEDS)
        require(sample_counts[group] == expected_count, "Incomplete score seed pool")
        for h, metrics in horizons.items():
            for name in SCORE_NAMES:
                # Same arithmetic mean, centered on the first seed so three
                # identical predictions remain exactly the original scores.
                metrics[name] += deviations[group][h][name] / expected_count
    # RW is deterministic for an origin/horizon and must agree across all arms.
    for panel in panels:
        for h in HORIZONS:
            baseline = totals[(panel, "STATISTICAL", "statistical_endpoint")][h]["rw_mean_price_squared_error"]
            for group in totals:
                if group[0] == panel:
                    require(np.allclose(totals[group][h]["rw_mean_price_squared_error"], baseline,
                                        rtol=2e-15, atol=1e-14), "Random-walk comparator differs between models")
    parameter_rows = []
    for (panel, backbone, arm, role, name), arrays in sorted(parameter_values.items()):
        values = np.stack(arrays)
        parameter_rows.append({"panel": panel, "stratum": stratum_for(panel), "backbone": backbone,
            "arm": arm, "role": role, "parameter": name, "seed_count": len(arrays),
            "origins_per_seed": values.shape[1], "pooled_origin_seed_mean": float(values.mean()),
            "pooled_origin_seed_std": float(values.std()), "mean_within_seed_std": float(values.std(axis=1).mean()),
            "minimum": float(values.min()), "maximum": float(values.max()),
            "scope": "Descriptive pooled origin-by-seed parameters, not a distribution ensemble"})
    return totals, origins, parameter_rows, score_means_verified


def circular_indices(rng, n, replicates=BOOTSTRAP_REPLICATES, block=BOOTSTRAP_BLOCK):
    require(n > 0 and replicates > 0 and block > 0, "Invalid bootstrap dimensions")
    starts = rng.integers(n, size=(replicates, math.ceil(n / block)))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(replicates, -1)[:, :n]


def bootstrap_means(scores, panels=PANELS, replicates=BOOTSTRAP_REPLICATES,
                    block=BOOTSTRAP_BLOCK, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    result = {}
    for panel in sorted(panels):
        groups = sorted(key for key in scores if key[0] == panel)
        require(groups, "Missing panel in bootstrap")
        n = len(scores[groups[0]][HORIZONS[0]]["normalized_CRPS"])
        indices = circular_indices(rng, n, replicates, block)
        for group in groups:
            result[group] = {}
            for h, metrics in scores[group].items():
                require(len(metrics["normalized_CRPS"]) == n, "Horizon/model bootstrap origins differ")
                result[group][h] = metrics["normalized_CRPS"][indices].mean(axis=1)
    return result


def cell_metric_rows(scores):
    rows = []
    for (panel, backbone, arm), horizons in sorted(scores.items()):
        market, year = panel.rsplit("_", 1)
        for h, metrics in sorted(horizons.items()):
            row = {"panel": panel, "market": market, "year": int(year), "stratum": stratum_for(panel),
                "backbone": backbone, "arm": arm, "horizon": h, "n_origins": len(metrics["normalized_CRPS"]),
                "seed_count": 1 if backbone == "STATISTICAL" else len(SEEDS),
                **{name: float(value.mean()) for name, value in metrics.items()}}
            denominator = row["rw_mean_price_squared_error"]
            row["mean_price_MSE_over_RW"] = row["mean_price_squared_error"] / denominator if denominator > 0 else None
            row["mean_price_MSE_relative_RW_minus_one"] = row["mean_price_MSE_over_RW"] - 1 if denominator > 0 else None
            rows.append(row)
    return rows


def comparison(scores, boot, candidate, control, panels, horizons, name, stratum):
    """Equal-weight cell ratios of mean scores, with paired origin bootstrap."""
    ratios, differences, boot_ratios, cell_rows = [], [], [], []
    secondary = defaultdict(list)
    for panel in panels:
        for h in horizons:
            a = scores[(panel, *candidate)][h]
            b = scores[(panel, *control)][h]
            ma, mb = float(a["normalized_CRPS"].mean()), float(b["normalized_CRPS"].mean())
            require(ma > 0 and mb > 0, "CRPS comparison needs positive mean scores")
            ratio = ma / mb - 1
            ratios.append(ratio)
            differences.append(ma - mb)
            denominator = boot[(panel, *control)][h]
            require(np.all(denominator > 0), "Bootstrap baseline CRPS must stay positive")
            boot_ratios.append(boot[(panel, *candidate)][h] / denominator - 1)
            price_denominator = float(b["price_CRPS"].mean())
            require(price_denominator > 0, "Price CRPS comparison needs a positive mean denominator")
            secondary["price_CRPS_relative"].append(float(a["price_CRPS"].mean()) / price_denominator - 1)
            for metric in ("normalized_NLL", "price_NLL", "coverage90", "normalized_width90"):
                secondary[metric + "_difference"].append(float(a[metric].mean() - b[metric].mean()))
            cell_rows.append({"comparison": name, "stratum": stratum, "panel": panel, "horizon": h,
                "candidate_backbone": candidate[0], "candidate_arm": candidate[1],
                "control_backbone": control[0], "control_arm": control[1],
                "n_origins": len(a["normalized_CRPS"]), "candidate_CRPS": ma, "control_CRPS": mb,
                "relative_CRPS_minus_one": ratio, "CRPS_difference": ma - mb,
                "strictly_better": ratio < 0, "exact_tie": ratio == 0})
    distribution = np.mean(np.stack(boot_ratios), axis=0)
    interval = np.quantile(distribution, [.025, .975], method="linear")
    values = np.asarray(ratios)
    return {"comparison": name, "stratum": stratum, "horizons": list(horizons),
        "candidate": {"backbone": candidate[0], "arm": candidate[1]},
        "control": {"backbone": control[0], "arm": control[1]},
        "cells": len(values), "mean_relative_CRPS_minus_one": float(values.mean()),
        "mean_CRPS_difference": float(np.mean(differences)),
        "better_cells": int(np.sum(values < 0)), "tie_cells": int(np.sum(values == 0)),
        "worse_cells": int(np.sum(values > 0)), "strict_win_fraction": float(np.mean(values < 0)),
        "tiny_effect_description_only": {
            "absolute_relative_threshold": 1e-12,
            "cells_below_threshold_including_exact_zero": int(np.sum(np.abs(values) < 1e-12)),
            "nonzero_cells_below_threshold": int(np.sum((values != 0) & (np.abs(values) < 1e-12))),
            "does_not_change_strict_wins_ties_or_gates": True,
            "note": "Near-machine-precision effects may include arithmetic differences such as sigmoid(logit(rho_base)) versus stored rho_base; no model or threshold is retroactively altered"},
        "bootstrap_nominal95_relative_CRPS": interval.tolist(),
        "secondary_equal_cell_means": {key: float(np.mean(value)) for key, value in secondary.items()},
        "cell_relative_values": {f"{row['panel']}_h{row['horizon']}": row["relative_CRPS_minus_one"]
                                 for row in cell_rows}}, cell_rows


def contrast_design(candidate, control):
    value = {"candidate_neural_parameters": PARAMETER_COUNTS.get(candidate),
             "control_neural_parameters": PARAMETER_COUNTS.get(control)}
    if candidate in PARAMETER_COUNTS and control in PARAMETER_COUNTS:
        value["control_minus_candidate_parameters"] = PARAMETER_COUNTS[control] - PARAMETER_COUNTS[candidate]
    pair = (candidate[1], control[1])
    if pair == ("conditional_joint005", "conditional_endpoint"):
        value["interpretation"] = (
            "Same conditional-persistence architecture and capacity; compare the fixed .05 path-NLL auxiliary "
            "objective with endpoint-only training. Each starts at its corresponding train-fitted constant-rho reference.")
    elif pair == ("conditional_endpoint", "independent_horizon_endpoint"):
        value["interpretation"] = (
            "Same endpoint-only objective, common scale, features and training budget. The control replaces one shared "
            "rho head with separate h5/h20 heads and has 17 additional parameters (1059 versus 1042). This is the "
            "planned comparison of shared within-origin future-kernel restriction with independent horizon marginals; "
            "it is not exactly capacity matched.")
    elif pair == ("conditional_joint005", "independent_horizon_endpoint"):
        value["interpretation"] = (
            "Overall performance comparison changes both path auxiliary objective and horizon restriction, "
            "and the control has 17 extra parameters. Do not attribute this contrast solely to cross-horizon consistency.")
    elif pair == ("conditional_joint005", "scale_capacity_joint005"):
        value["interpretation"] = (
            "Same joint objective and 1042-parameter GRU architecture. The extra head changes persistence in the "
            "candidate and scale in the control, whose persistence remains at its train-fitted constant.")
    else:
        value["interpretation"] = "Prespecified matched or contextual contrast; inspect the named arms and fit capacities"
    return value


def all_comparisons(scores, boot):
    contrasts = []
    for backbone in ("GRU16", "MLP16"):
        main = (backbone, "conditional_joint005")
        controls = [(backbone, arm) for arm in ARMS[backbone] if arm != main[1]]
        controls += [("STATISTICAL", "statistical_joint005"), ("STATISTICAL", "statistical_endpoint")]
        for control in controls:
            name = f"{backbone}__conditional_joint005__vs__{control[0]}__{control[1]}"
            contrasts.append((name, main, control))
    contrasts += [
        ("GRU16__conditional_endpoint__vs__GRU16__constant_endpoint",
         ("GRU16", "conditional_endpoint"), ("GRU16", "constant_endpoint")),
        ("GRU16__conditional_endpoint__vs__GRU16__independent_horizon_endpoint",
         ("GRU16", "conditional_endpoint"), ("GRU16", "independent_horizon_endpoint")),
        ("GRU16__constant_joint005__vs__GRU16__constant_endpoint",
         ("GRU16", "constant_joint005"), ("GRU16", "constant_endpoint")),
    ]
    results, rows = {}, []
    for name, candidate, control in contrasts:
        results[name] = {}
        for stratum, panels in STRATA.items():
            results[name][stratum] = {}
            for horizon_label, horizons in (("h5_h20", (5, 20)), ("h1_secondary", (1,))):
                value, current = comparison(scores, boot, candidate, control, panels, horizons, name, stratum)
                value["design"] = contrast_design(candidate, control)
                results[name][stratum][horizon_label] = value
                rows.extend(current)
    return results, rows


def method_summaries(cell_rows):
    results = []
    for stratum, panels in STRATA.items():
        for backbone, arms in ARMS.items():
            for arm in arms:
                for label, horizons in (("h5_h20", (5, 20)), ("h1_secondary", (1,))):
                    rows = [row for row in cell_rows if row["stratum"] == stratum and
                            row["backbone"] == backbone and row["arm"] == arm and row["horizon"] in horizons]
                    require(len(rows) == len(panels) * len(horizons), "Incomplete method summary cells")
                    rw_ratios = [row["mean_price_MSE_over_RW"] for row in rows]
                    results.append({"stratum": stratum, "backbone": backbone, "arm": arm,
                        "horizon_group": label, "cells": len(rows),
                        "mean_normalized_CRPS": float(np.mean([r["normalized_CRPS"] for r in rows])),
                        "mean_normalized_NLL": float(np.mean([r["normalized_NLL"] for r in rows])),
                        "mean_price_MSE_over_RW": float(np.mean(rw_ratios)) if all(x is not None for x in rw_ratios) else None,
                        "cells_with_zero_RW_MSE": sum(x is None for x in rw_ratios),
                        "mean_coverage90": float(np.mean([r["coverage90"] for r in rows])),
                        "mean_absolute_cell_coverage_error_from_0_9": float(np.mean([abs(r["coverage90"] - .9) for r in rows])),
                        "mean_normalized_width90": float(np.mean([r["normalized_width90"] for r in rows])),
                        "raw_currency_metrics": "Reported per cell in CELL_METRICS.csv; price CRPS, price widths and raw MSE are not averaged across currencies"})
    return results


def continuation_checks(primary_gru, capacity, primary_mlp, additional):
    """Strict gates operate on original floats; no rounded values/tolerance."""
    checks = {
        "GRU_development_primary_mean_negative": {
            "value": primary_gru["mean_relative_CRPS_minus_one"], "rule": "value < 0",
            "passed": primary_gru["mean_relative_CRPS_minus_one"] < 0},
        "GRU_at_least_60_percent_cells_strictly_better": {
            "better_cells": primary_gru["better_cells"], "cells": primary_gru["cells"],
            "rule": "5 * better_cells >= 3 * cells; strict cell improvement means relative < 0",
            "passed": 5 * primary_gru["better_cells"] >= 3 * primary_gru["cells"]},
        "GRU_development_bootstrap_upper_negative": {
            "value": primary_gru["bootstrap_nominal95_relative_CRPS"][1], "rule": "value < 0",
            "passed": primary_gru["bootstrap_nominal95_relative_CRPS"][1] < 0},
        "GRU_main_lower_than_scale_capacity_on_average": {
            "value": capacity["mean_relative_CRPS_minus_one"], "rule": "equal-cell mean(main/capacity - 1) < 0",
            "passed": capacity["mean_relative_CRPS_minus_one"] < 0},
        "MLP_development_primary_mean_negative": {
            "value": primary_mlp["mean_relative_CRPS_minus_one"], "rule": "value < 0",
            "passed": primary_mlp["mean_relative_CRPS_minus_one"] < 0},
        "Shenzhen_additional_primary_mean_nonpositive": {
            "value": additional["mean_relative_CRPS_minus_one"], "rule": "value <= 0",
            "passed": additional["mean_relative_CRPS_minus_one"] <= 0},
    }
    passed = all(row["passed"] for row in checks.values())
    return {"all_passed": passed, "decision": "PASS_INTERNAL_CONTINUATION_CHECKS" if passed else "STOP_FIXED_CANDIDATE",
        "checks": checks, "failed_checks": [key for key, row in checks.items() if not row["passed"]],
        "rounding_or_tie_tolerance_used": False,
        "interpretation": "Internal research decision only; even PASS does not establish SOTA, independent-market transportability or publication readiness",
        "if_failed": "Stop this fixed candidate. Complete the already specified baselines; do not tune q, losses, horizons, seeds or market years, and do not open reserved 2024+ data to rescue it"}


def csv_text(rows):
    require(bool(rows), "Cannot serialize an empty table")
    fields = list(rows[0])
    require(all(list(row) == fields for row in rows), "CSV row schema mismatch")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_outputs(output_dir, outputs):
    require(set(outputs) == set(OUTPUT_NAMES), "Incorrect public output inventory")
    require(not any((output_dir / name).exists() for name in outputs), "Refuse to overwrite an existing aggregation")
    encoded = {name: json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
               if name.endswith(".json") else csv_text(value) for name, value in outputs.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, contents in encoded.items():
        with (output_dir / name).open("x", encoding="utf-8", newline="") as handle:
            handle.write(contents)


def aggregate(public_dir, private_dir, output_dir, protocol_path):
    started = time.perf_counter()
    validate_protocol(protocol_path)
    require(not any((output_dir / name).exists() for name in OUTPUT_NAMES),
            "Existing analysis outputs found; no overwrite or partial reanalysis")
    fits, references, manifests, starts, integrity = complete_inventory(public_dir, private_dir, protocol_path)
    accounting = fit_accounting(fits, references, manifests, starts)
    scores, origins, parameter_rows, verified_means = load_scores(fits, references, private_dir)
    boot = bootstrap_means(scores)
    cell_rows = cell_metric_rows(scores)
    comparisons, comparison_rows = all_comparisons(scores, boot)
    gru_name = "GRU16__conditional_joint005__vs__GRU16__constant_joint005"
    mlp_name = "MLP16__conditional_joint005__vs__MLP16__constant_joint005"
    capacity_name = "GRU16__conditional_joint005__vs__GRU16__scale_capacity_joint005"
    primary_gru = comparisons[gru_name]["development"]["h5_h20"]
    primary_mlp = comparisons[mlp_name]["development"]["h5_h20"]
    additional = comparisons[gru_name]["additional_shenzhen"]["h5_h20"]
    capacity = comparisons[capacity_name]["development"]["h5_h20"]
    gates = continuation_checks(primary_gru, capacity, primary_mlp, additional)
    integrity.update({"saved_score_means_verified": verified_means,
        "common_origins_across_seeds_models_and_horizons": True,
        "reserved_dates_rejected_before_scores": True,
        "loaded_private_numeric_fields": "Only scores, forecast parameters and origin/target indices; no raw/derived prices, targets, features or checkpoint tensor values",
        "aggregator_sha256": sha(__file__)})
    summary = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "author_role": "Implementation; requires a different researcher for independent audit",
        "all_prespecified_fits_complete_before_analysis": True,
        "identity": integrity["identity"], "aggregator_sha256": sha(__file__),
        "statistic": "Average the three seed scores at each origin; take the origin mean for each fixed panel/horizon; compute candidate/control - 1; average cells equally",
        "seed_mean_numerics": "first seed score + mean(seed score - first seed score); identical seed scores stay exactly identical to a single reference",
        "distribution_ensemble": False,
        "strata": {name: {"panels": list(panels), "primary_h5_h20_cells": 2 * len(panels),
                           "test_origins_by_panel": {panel: len(origins[(panel, "test")]["indices"]) for panel in panels}}
                    for name, panels in STRATA.items()},
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "block_length": BOOTSTRAP_BLOCK,
            "seed": BOOTSTRAP_SEED, "generator": "numpy.random.default_rng", "panel_order": list(PANELS),
            "resampling": "Circular contiguous origin blocks; ceil(n/40) starts followed by truncation to n",
            "pairing": "Common resampled origin indices across all methods, seeds-averaged scores, and horizons within each panel",
            "quantiles": [.025, .975], "quantile_method": "linear", "nominal_only": True,
            "boundary": "Within fixed annual panels only; approximate dependent-data intervals. Cross-market and cross-year dependence between separately resampled panels is not modeled. These intervals do not establish a population-of-markets or unseen-market effect; ablation intervals are not multiplicity-adjusted"},
        "primary_GRU_development": primary_gru, "replication_MLP_development": primary_mlp,
        "additional_GRU_shenzhen": additional, "capacity_GRU_development": capacity,
        "internal_continuation": gates, "comparisons": comparisons,
        "method_summaries": method_summaries(cell_rows),
        "fit_accounting": {key: value for key, value in accounting.items()
                           if key not in ("groups", "neural_jobs", "static_optimizers")},
        "tables": {"cell_metrics": "CELL_METRICS.csv", "cell_comparisons": "CELL_COMPARISONS.csv",
                   "parameters": "PARAMETER_SUMMARIES.csv", "fit_accounting": "MODEL_FIT_ACCOUNTING.json",
                   "integrity": "INPUT_INTEGRITY.json"},
        "public_output_scope": "Cell and fit aggregates only; no per-origin prices, returns, scores, parameters or forecasts",
        "aggregation_elapsed_seconds_before_serialization": time.perf_counter() - started,
    }
    write_outputs(output_dir, {"SUMMARY.json": summary, "CELL_METRICS.csv": cell_rows,
        "CELL_COMPARISONS.csv": comparison_rows, "PARAMETER_SUMMARIES.csv": parameter_rows,
        "MODEL_FIT_ACCOUNTING.json": accounting, "INPUT_INTEGRITY.json": integrity})
    return summary


def self_test_saved_scores():
    """Temporary fabricated NPZ schema, exercising the actual score loader."""
    import tempfile
    panel = "SYNTHETIC_2001"
    counts = {"train": 7, "validation": 3, "test": 5}
    prepared = {"role_counts": counts}

    def fixture(multiplier):
        archive, public = {}, {}
        for role, n in counts.items():
            start = {"train": 39, "validation": 100, "test": 200}[role]
            ids = np.arange(start, start + n)
            dates = np.datetime64("2001-01-01") + ids.astype("timedelta64[D]")
            archive[role + "_indices"] = ids
            archive[role + "_dates"] = dates.astype(str)
            for name in PARAMETERS:
                archive[f"{role}_{name}"] = np.full(n, 1. if name == "f" else .5)
            public["train_parameters" if role == "train" else role + "_parameters"] = {
                name: {"mean": float(archive[f"{role}_{name}"].mean()), "std": 0.,
                       "minimum": float(archive[f"{role}_{name}"][0]),
                       "maximum": float(archive[f"{role}_{name}"][0])} for name in PARAMETERS}
            if role == "train":
                continue
            archive[role + "_target_indices"] = ids[:, None] + np.array(HORIZONS)
            archive[role + "_target_dates"] = (dates[:, None] + np.array(HORIZONS).astype("timedelta64[D]")).astype(str)
            public[role] = {}
            for h in HORIZONS:
                for part, value in (("weights", 1.), ("means", 0.), ("sigmas", 1.)):
                    archive[f"{role}_{part}_h{h}"] = np.full((n, 1), value)
                row = {"n": n}
                for name in SCORE_NAMES:
                    base = np.arange(1, n + 1, dtype=float) * .1
                    value = ((np.arange(n) % 2).astype(float) if name == "coverage90" else
                             base if name == "rw_mean_price_squared_error" else multiplier * base)
                    archive[f"{role}_score_h{h}_{name}"] = value
                    row[name] = float(value.mean())
                public[role][str(h)] = row
        return archive, public

    with tempfile.TemporaryDirectory(prefix="persistence-aggregate-fabricated-") as directory:
        private_dir = Path(directory)
        (private_dir / "forecasts").mkdir()
        fits = {}
        for seed, multiplier in zip(SEEDS, (.7, .8, .9)):
            key = job_key(panel, "GRU16", "conditional_joint005", seed)
            arrays, public = fixture(multiplier)
            np.savez(private_dir / "forecasts" / f"{key}.npz", **arrays)
            fits[key] = {"panel": panel, "fit": {"backbone": "GRU16", "arm": "conditional_joint005", "seed": seed},
                         "metrics": public, "prepared": prepared}
        references = {panel: {"metrics": {}, "prepared": prepared}}
        for mode in ("endpoint", "joint005"):
            arrays, public = fixture(1.)
            np.savez(private_dir / "forecasts" / f"{panel}__statistical_{mode}.npz", **arrays)
            references[panel]["metrics"][mode] = public
        scores, origins, parameters, checked = load_scores(fits, references, private_dir, panels=(panel,))
        actual = scores[(panel, "GRU16", "conditional_joint005")][20]["normalized_CRPS"]
        require(np.allclose(actual, .8 * np.arange(1, 6) * .1, rtol=1e-14, atol=1e-14),
                "Saved seed scores were not averaged per origin")
        require(checked == 5 * 2 * 3 * len(SCORE_NAMES) and len(origins) == 3 and len(parameters) == 27,
                "Incomplete fabricated score/parameter verification")
        first = next(iter(fits.values()))
        first["metrics"]["test"]["5"]["normalized_CRPS"] += .1
        try:
            load_scores(fits, references, private_dir, panels=(panel,))
        except ValueError:
            pass
        else:
            raise AssertionError("A corrupted public score mean must be rejected")
        for key in fits:
            arrays, public = fixture(1.)
            np.savez(private_dir / "forecasts" / f"{key}.npz", **arrays)
            fits[key]["metrics"] = public
        identical, _, _, _ = load_scores(fits, references, private_dir, panels=(panel,))
        for h in HORIZONS:
            for metric in SCORE_NAMES:
                same_array(identical[(panel, "GRU16", "conditional_joint005")][h][metric],
                           identical[(panel, "STATISTICAL", "statistical_endpoint")][h][metric],
                           "three identical seeds versus one reference")
        paired_boot = bootstrap_means(identical, panels=(panel,), replicates=19, block=3, seed=17)
        exact, _ = comparison(identical, paired_boot, ("GRU16", "conditional_joint005"),
                              ("STATISTICAL", "statistical_endpoint"), (panel,), (5, 20),
                              "fabricated_identical", "fabricated")
        require(exact["mean_relative_CRPS_minus_one"] == 0 and exact["tie_cells"] == 2 and
                exact["bootstrap_nominal95_relative_CRPS"] == [0., 0.],
                "Identical seeds/reference must give exact cell and bootstrap ties")


def self_test():
    """Small analytic aggregation examples only; no files or market data read."""
    # A ratio of means differs from a mean of origin ratios and from pooling cells.
    panel_names = ("A", "B")
    data = {}
    for panel, denominator, factor in (("A", np.array([1., 3., 2., 4., 5.]), .8),
                                       ("B", np.array([10., 30., 20., 40., 50.]), 1.1)):
        for arm, multiplier in (("control", 1.), ("candidate", factor)):
            values = {}
            for h in HORIZONS:
                x = denominator * multiplier * h
                values[h] = {name: (x.copy() if name not in ("coverage90",) else np.full(len(x), .9))
                             for name in SCORE_NAMES}
            data[(panel, "FAKE", arm)] = values
    boot = bootstrap_means(data, panels=panel_names, replicates=19, block=3, seed=17)
    result, _ = comparison(data, boot, ("FAKE", "candidate"), ("FAKE", "control"),
                           panel_names, (5, 20), "fabricated", "fabricated")
    same_float(result["mean_relative_CRPS_minus_one"], -.05, "equal cell ratios")
    require(result["better_cells"] == 2 and result["worse_cells"] == 2, "Incorrect strict win count")
    require(np.max(np.abs(np.array(result["bootstrap_nominal95_relative_CRPS"]) + .05)) < 1e-14,
            "Paired proportional-score bootstrap must have a constant relative effect")
    for key in data:
        require(np.max(np.abs(boot[key][20] / 4 - boot[key][5])) < 1e-12,
                "All horizons must use identical origin draws")
    rng = np.random.default_rng(5)
    ids = circular_indices(rng, 7, 11, 4)
    require(ids.shape == (11, 7) and np.all((ids >= 0) & (ids < 7)), "Circular block shape/range")
    require(np.all((ids[:, 1:4] - ids[:, :3]) % 7 == 1), "First block is not contiguous")
    require(np.all((ids[:, 5:] - ids[:, 4:6]) % 7 == 1), "Truncated final block is not contiguous")
    seed_scores = np.array([[1., 5.], [5., 1.], [3., 3.]])
    same_array(seed_scores.mean(axis=0), np.array([3., 3.]), "seed-score origin averaging")
    base = {"mean_relative_CRPS_minus_one": -1e-16, "better_cells": 14, "cells": 22,
            "bootstrap_nominal95_relative_CRPS": [-2e-16, -1e-17]}
    gates = continuation_checks(base, base, base, {**base, "mean_relative_CRPS_minus_one": 0.})
    require(gates["all_passed"], "Unrounded tiny gains and zero Shenzhen margin should pass their exact gates")
    bad = continuation_checks({**base, "better_cells": 13}, base, base, base)
    require(not bad["all_passed"] and len(bad["failed_checks"]) == 1, "60-percent gate must reject 13/22")
    zero = continuation_checks({**base, "mean_relative_CRPS_minus_one": 0.}, base, base, base)
    require(not zero["all_passed"], "Strict development mean gate must reject zero")
    for weight in (0., .05):
        history = []
        for epoch in range(41):
            value = .5 if epoch < 3 else .4
            entry = {"objective": value + weight * 1.2, "endpoint_CRPS": value,
                     "path_NLL": 1.2 if weight else None,
                     "CRPS_h": {"1": value, "5": value, "20": value}}
            history.append({"epoch": epoch, "optimizer_steps": epoch * 2,
                            "training": entry.copy(), "validation": entry.copy()})
        fit = {"train_origins": 128, "validation_origins": 126, "epochs_run": 40,
               "batch_size": 64, "optimizer_steps": 80, "history": history,
               "auxiliary_weight": weight, "selected_epoch": 3, "epoch_zero_selected": False,
               "selected_validation_CRPS": .4, "initial_f_absolute_error": 0.,
               "initial_rho_absolute_error": 0., "elapsed_seconds": 1.,
               "training_complete_before_validation_selected_state_reload": True,
               "test_used_for_selection": False}
        validate_fit_history(fit, "fabricated_history")
        wrong = {**fit, "selected_epoch": 4}
        try:
            validate_fit_history(wrong, "fabricated_wrong_history")
        except ValueError:
            pass
        else:
            raise AssertionError("A later tied checkpoint must not pass history validation")
    self_test_saved_scores()
    return {"passed": True, "scope": "Fabricated score arrays only", "market_files_read": 0,
        "model_fits": 0, "checks": ["equal cell ratio mean", "strict wins", "paired bootstrap",
            "common horizon draws", "circular truncation", "seed-score mean", "unrounded gate boundaries",
            "nested endpoint/joint learning histories", "earliest tied checkpoint",
            "temporary fabricated saved-score schema", "corrupt score receipt rejection",
            "three identical seeds versus single reference retain exact score and bootstrap ties"],
        "formal_bootstrap_configuration": {"replicates": BOOTSTRAP_REPLICATES, "block": BOOTSTRAP_BLOCK,
                                           "seed": BOOTSTRAP_SEED}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", type=Path, default=STUDY / "results" / "market" / "persistence")
    parser.add_argument("--private-dir", type=Path, default=Path("/tmp/carbon-persistence-market-runs"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--protocol", type=Path, default=STUDY / "provenance" / "MARKET_PROTOCOL.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2, allow_nan=False))
        return
    output_dir = args.output_dir or args.public_dir
    summary = aggregate(args.public_dir, args.private_dir, output_dir, args.protocol)
    print(json.dumps({"stage": "complete_aggregation", "neural_fits": 312, "static_fits": 39,
                      "decision": summary["internal_continuation"]["decision"],
                      "summary": str(output_dir / "SUMMARY.json")}, allow_nan=False))


if __name__ == "__main__":
    main()
