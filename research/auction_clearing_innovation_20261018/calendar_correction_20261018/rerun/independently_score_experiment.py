#!/usr/bin/env python3
"""Independent frozen-development forecast audit. No model/runner imports.

Prices and labels come from the frozen float64 examples, never reconstructed
from float32 training tensors. Primary CRPS integrates the squared finite CDF;
the separate quadratic expectation identity is used only for cross-checks.
The three neural seeds form one forecast: average points and concatenate
distributions with one third of each seed's weights. Seeds are not market N.

Run after the root's completion signal:
  python independently_score_experiment.py --mode full
For an independently useful completed-classical audit: --mode classical-only.
All writes are restricted to results/independent_metrics/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform as runtime_platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

BOOTSTRAP_SEED = 20261018
BOOTSTRAP_DRAWS = 2000
BLOCK_LENGTH = 20
WEIGHT_TOLERANCE = 1e-10
MOMENT_ABS_TOLERANCE = 1e-8
MOMENT_REL_TOLERANCE = 1e-10
CRPS_ABS_TOLERANCE = 1e-9
CRPS_REL_TOLERANCE = 1e-10
MATCHED = ("DirectMT", "GRU_full", "MLP_full", "Ridge", "HistGB")
COMPARATORS = ("RW", "DirectMT", "GRU_full", "MLP_full", "Ridge",
               "HistGB", "SDMC_plugin", "SDMC_no_supply")
NON_GAMMA_NEURAL_MODELS = ("GRU_full", "GRU_price", "MLP_full")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(data), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def finite_max(values):
    values = np.asarray(values)
    return float(np.max(values)) if values.size else 0.0


def exact_source_match(observed, expected, name):
    observed, expected = np.asarray(observed), np.asarray(expected)
    require(observed.shape == expected.shape, f"{name}: shape mismatch")
    exact = np.array_equal(observed, expected, equal_nan=True)
    finite = np.isfinite(observed) & np.isfinite(expected)
    difference = finite_max(np.abs(observed[finite].astype(np.float64) - expected[finite].astype(np.float64)))
    require(exact, f"{name}: not identical to frozen source; max diff {difference}")
    return {"exact_match": True, "max_absolute_difference": difference,
            "saved_dtype": str(observed.dtype), "source_dtype": str(expected.dtype)}


def crps_cdf_integral(atoms, weights, observed):
    """Integrate F(t)^2 below y and (1-F(t))^2 above y, interval by interval."""
    atoms = np.asarray(atoms, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    order = np.argsort(atoms, axis=1, kind="stable")
    x = np.take_along_axis(atoms, order, axis=1)
    w = np.take_along_axis(weights, order, axis=1)
    cdf = np.cumsum(w, axis=1)
    widths = np.diff(x, axis=1)
    before_y = np.clip(observed[:, None] - x[:, :-1], 0.0, widths)
    after_y = widths - before_y
    interior = np.sum(cdf[:, :-1] ** 2 * before_y
                      + (1.0 - cdf[:, :-1]) ** 2 * after_y, axis=1)
    score = interior + np.maximum(x[:, 0] - observed, 0.0) \
        + np.maximum(observed - x[:, -1], 0.0)
    return score, x, w, cdf


def crps_quadratic(atoms, weights, observed):
    """Independent E|X-y| - E|X-X'|/2 definition, O(K^2) per row."""
    answer = []
    for x, w, y in zip(atoms, weights, observed):
        first = float(np.dot(w, np.abs(x - y)))
        second = float(np.sum(w[:, None] * w[None, :] * np.abs(x[:, None] - x[None, :])))
        answer.append(first - second / 2.0)
    return np.asarray(answer)


def check_crps_synthetic():
    rng = np.random.default_rng(70931)
    atoms = rng.normal(size=(31, 13))
    atoms[0] = 2.0
    atoms[1, :5] = atoms[1, 5]
    weights = rng.dirichlet(np.ones(13), size=31)
    weights[2, :4] = 0.0
    weights /= weights.sum(axis=1, keepdims=True)
    y = rng.normal(size=31)
    y[3], y[4], y[5] = -40.0, 40.0, atoms[5, 5]
    integral = crps_cdf_integral(atoms, weights, y)[0]
    quadratic = crps_quadratic(atoms, weights, y)
    require(np.allclose(integral, quadratic, atol=CRPS_ABS_TOLERANCE,
                        rtol=CRPS_REL_TOLERANCE), "Synthetic CRPS identities disagree")
    affine = crps_cdf_integral(81.0 + 2.7 * atoms, weights, 81.0 + 2.7 * y)[0]
    require(np.allclose(affine, 2.7 * integral, atol=CRPS_ABS_TOLERANCE,
                        rtol=CRPS_REL_TOLERANCE), "CRPS price-unit affine identity")
    return {"status": "PASS", "artificial_rows": len(y),
            "max_integral_vs_quadratic_absolute_difference": finite_max(np.abs(integral - quadratic)),
            "max_affine_scaling_difference": finite_max(np.abs(affine - 2.7 * integral)),
            "cases": ["unequal weights", "ties", "zero weights", "outside support", "at atom"]}


def read_frozen_inputs(root):
    freeze_path = root / "PROTOCOL_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_checks = []
    for relative, expected_hash in freeze["frozen_sha256"].items():
        observed_hash = sha256(root / relative)
        require(observed_hash == expected_hash, f"Frozen file changed: {relative}")
        freeze_checks.append({"path": relative, "sha256": observed_hash, "matches_freeze": True})
    config = json.loads((root / "experiment_config.json").read_text(encoding="utf-8"))
    require(config["evaluation_issue_years"] == [2021, 2022, 2023], "Unexpected evaluation years")
    require(config["horizons"] == [1, 5, 20], "Unexpected horizons")
    require(config["refit_seeds"] == [17, 29, 43], "Unexpected refit seeds")
    meta = pd.read_csv(root / "local_only/example_metadata.csv", keep_default_na=False)
    with np.load(root / "local_only/examples_development.npz", allow_pickle=False) as z:
        needed = ("price", "anchor", "scale", "y", "completed", "bid_ratio",
                  "has_bid", "known_status", "cancelled")
        data = {name: z[name].copy() for name in needed}
    for name in ("price", "anchor", "scale", "y", "bid_ratio"):
        require(data[name].dtype == np.float64, f"Source precision is not float64: {name}")
    n = len(meta)
    require(all(len(v) == n for v in data.values()), "Example metadata/arrays length mismatch")
    completed = np.isfinite(data["price"]) & (data["price"] > 0)
    require(np.array_equal(completed, data["completed"].astype(bool)), "Frozen completion mask mismatch")
    require(np.isfinite(data["anchor"]).all() and (data["anchor"] > 0).all(), "Invalid historical anchor")
    require(np.isfinite(data["scale"]).all() and (data["scale"] > 0).all(), "Invalid forecast scale")
    require(meta.issue_year.max() <= 2023 and meta.target_date.max() < "2024-01-01", "Confirmation date encountered")
    require((meta.history_last_date < meta.issue_date).all(), "History reaches issue time")
    require((meta.issue_date <= meta.target_date).all(), "Target precedes issue time")
    reconstructed = data["anchor"][completed] + data["scale"][completed] * data["y"][completed]
    precision_error = finite_max(np.abs(reconstructed - data["price"][completed]))
    require(precision_error <= 1e-10, "Frozen float64 normalized target does not reconstruct price")
    data["completed"] = completed
    for name in ("has_bid", "known_status", "cancelled"):
        data[name] = data[name].astype(bool)
    data["horizon"] = meta.horizon.to_numpy(dtype=np.int64)
    data["issue_year"] = meta.issue_year.to_numpy(dtype=np.int64)
    return config, freeze, meta, data, {
        "freeze_id": freeze["freeze_id"], "protocol_freeze_sha256": sha256(freeze_path),
        "all_frozen_files_unchanged": True, "files": freeze_checks,
        "source_examples": n, "float64_target_reconstruction_max_absolute_error": precision_error}


def require_run_completion(root, mode, freeze_audit):
    receipt = json.loads((root / "results/classical/CLASSICAL_RUN_RECEIPT.json").read_text())
    require(receipt["status"].lower() == "completed", "Classical run is not complete")
    require(len(receipt["completed_fold_models"]) == 12, "Classical run count is not 12")
    audit = {"classical_status": receipt["status"], "classical_fold_runs": 12,
             "classical_receipt_sha256": sha256(root / "results/classical/CLASSICAL_RUN_RECEIPT.json")}
    if mode == "full":
        path = root / "results/neural/RUN_SUMMARY.json"
        require(path.exists(), "Wait for the root's neural completion signal and RUN_SUMMARY")
        summary = json.loads(path.read_text())
        require(summary["status"] == "COMPLETE", "Neural run is not COMPLETE")
        require(not summary["failures"], "Neural run reports failures")
        require(summary["protocol_freeze_sha256"] == freeze_audit["protocol_freeze_sha256"],
                "Neural run used another protocol freeze")
        audit.update({"neural_status": summary["status"], "neural_fits": summary["fits"],
                      "neural_refits": summary["refits"], "neural_failures": summary["failures"],
                      "neural_summary_sha256": sha256(path)})
    return audit


def read_prediction(root, model, year, seed, meta, data):
    family = "classical" if seed is None else "neural"
    suffix = "" if seed is None else f"_s{seed}"
    relative = f"local_only/{family}/{model}_{year}{suffix}.npz"
    path = root / relative
    require(path.exists(), f"Missing required prediction: {relative}")
    before = sha256(path)
    with np.load(path, allow_pickle=False) as saved:
        wanted = ("example_indices", "mean_z", "atoms_z", "weights", "anchor", "scale", "y_z",
                  "completed", "horizon", "mechanical_failure", "mean_bid_ratio", "bid_mixture", "bid_concentration")
        prediction = {name: saved[name].copy() for name in wanted if name in saved.files}
        schema = {name: {"shape": list(saved[name].shape), "dtype": str(saved[name].dtype)}
                  for name in prediction}
    require(sha256(path) == before, f"Prediction changed while reading: {relative}")
    indices = prediction["example_indices"]
    expected = np.flatnonzero(data["issue_year"] == year)
    require(np.array_equal(indices, expected), f"Prediction index identity/order mismatch: {relative}")
    require(len(np.unique(indices)) == len(indices), f"Duplicate prediction indices: {relative}")
    n = len(indices)
    z, atoms, weights = (prediction[name].astype(np.float64) for name in ("mean_z", "atoms_z", "weights"))
    require(z.shape == (n,) and atoms.shape == weights.shape and atoms.shape[0] == n,
            f"Forecast dimensions invalid: {relative}")
    require(np.isfinite(z).all() and np.isfinite(atoms).all() and np.isfinite(weights).all(),
            f"Non-finite forecast: {relative}")
    if model == "RW":
        require(np.array_equal(z, np.zeros(n, dtype=np.float64)),
                f"RW point forecast is not the same-origin historical anchor: {relative}")
    require((weights >= 0).all(), f"Negative forecast weights: {relative}")
    total = weights.sum(axis=1)
    weight_error = finite_max(np.abs(total - 1.0))
    require(weight_error <= WEIGHT_TOLERANCE, f"Weights not normalized: {relative}")
    weighted_mean = np.sum(atoms * weights, axis=1)
    mean_error = finite_max(np.abs(weighted_mean - z))
    require(np.allclose(weighted_mean, z, atol=MOMENT_ABS_TOLERANCE, rtol=MOMENT_REL_TOLERANCE),
            f"Stored point differs from distribution mean: {relative}")
    matches = {name: exact_source_match(prediction[name], data[source][indices], f"{relative}:{name}")
               for name, source in (("anchor", "anchor"), ("scale", "scale"), ("y_z", "y"))}
    if "completed" in prediction:
        matches["completed"] = exact_source_match(prediction["completed"], data["completed"][indices], relative)
    if "horizon" in prediction:
        matches["horizon"] = exact_source_match(prediction["horizon"], data["horizon"][indices], relative)
    if seed is not None and model in NON_GAMMA_NEURAL_MODELS:
        # Frozen non-Gamma baselines explicitly save NaN for inapplicable
        # volume/failure diagnostics. This is not a missing price forecast.
        for name in ("mean_bid_ratio", "bid_concentration", "mechanical_failure"):
            require(name in prediction and np.isnan(prediction[name]).all(),
                    f"Unexpected non-Gamma diagnostic placeholder: {relative}:{name}")
        for name in ("mean_bid_ratio", "bid_mixture", "bid_concentration", "mechanical_failure"):
            prediction.pop(name, None)
    if seed is not None and model not in NON_GAMMA_NEURAL_MODELS:
        for name in ("mean_bid_ratio", "bid_mixture", "bid_concentration", "mechanical_failure"):
            require(name in prediction, f"Neural diagnostic missing {name}: {relative}")
            require(np.isfinite(prediction[name]).all(), f"Neural diagnostic not finite: {relative}:{name}")
        require((prediction["mean_bid_ratio"] > 0).all() and (prediction["bid_concentration"] > 0).all(),
                f"Invalid Gamma diagnostic parameters: {relative}")
        require((prediction["bid_mixture"] >= 0).all(), f"Invalid Gamma mixture: {relative}")
        require(np.allclose(prediction["bid_mixture"].sum(axis=1), 1, atol=2e-6, rtol=0),
                f"Gamma mixture not normalized: {relative}")
        require(((prediction["mechanical_failure"] >= 0) & (prediction["mechanical_failure"] <= 1)).all(),
                f"Mechanical failure outside [0,1]: {relative}")
    prediction.update({"model": model, "year": year, "seed": seed,
                       "mean_z": z, "atoms_z": atoms, "weights": weights})
    return prediction, {
        "path": relative, "sha256": before, "bytes": path.stat().st_size,
        "model": model, "issue_year": year, "seed": seed, "schema": schema,
        "n_issued": n, "n_price_scored": int(data["completed"][indices].sum()),
        "n_price_unscored": int((~data["completed"][indices]).sum()),
        "n_cancelled_records": int(data["cancelled"][indices].sum()),
        "n_atoms": atoms.shape[1], "max_weight_sum_error": weight_error,
        "gamma_diagnostics_applicable": seed is not None and model not in NON_GAMMA_NEURAL_MODELS,
        "non_gamma_nan_placeholders_scored": False,
        "RW_point_exactly_equals_frozen_origin_anchor": True if model == "RW" else None,
        "minimum_weight_sum": float(total.min()), "maximum_weight_sum": float(total.max()),
        "max_stored_mean_vs_weighted_atoms_error_z": mean_error,
        "max_stored_mean_vs_weighted_atoms_error_price": finite_max(
            np.abs(weighted_mean - z) * data["scale"][indices]),
        "source_precision_matches": matches, "index_identity_exact": True,
        "all_issued_forecasts_retained": True}


def ensemble_prediction(predictions, data):
    n_seeds = len(predictions)
    first = predictions[0]
    require(n_seeds == 3, "Primary neural estimator must use all three seeds")
    for other in predictions[1:]:
        require(np.array_equal(first["example_indices"], other["example_indices"]), "Ensemble seed identity mismatch")
    output = {"model": first["model"], "year": first["year"], "seed": "ensemble",
              "example_indices": first["example_indices"],
              "mean_z": np.mean(np.stack([p["mean_z"] for p in predictions]), axis=0),
              "atoms_z": np.concatenate([p["atoms_z"] for p in predictions], axis=1),
              "weights": np.concatenate([p["weights"] / n_seeds for p in predictions], axis=1)}
    if "mechanical_failure" in first:
        output["mechanical_failure"] = np.mean(np.stack([p["mechanical_failure"] for p in predictions]), axis=0)
        for name in ("mean_bid_ratio", "bid_concentration"):
            output[name] = np.concatenate([p[name].astype(np.float64) for p in predictions], axis=1)
        output["bid_mixture"] = np.concatenate([p["bid_mixture"].astype(np.float64) / n_seeds
                                                  for p in predictions], axis=1)
    weights = output["weights"]
    require(np.max(np.abs(weights.sum(axis=1) - 1.0)) <= WEIGHT_TOLERANCE, "Ensemble weight normalization")
    residual = np.sum(output["atoms_z"] * weights, axis=1) - output["mean_z"]
    require(np.allclose(residual, 0.0, atol=MOMENT_ABS_TOLERANCE, rtol=0), "Ensemble mean identity")
    output["ensemble_audit"] = {
        "model": first["model"], "issue_year": first["year"], "seed_count": n_seeds,
        "n_atoms": output["atoms_z"].shape[1], "max_weight_sum_error": finite_max(np.abs(weights.sum(axis=1) - 1)),
        "max_mean_identity_error_z": finite_max(np.abs(residual)),
        "point_rule": "arithmetic mean of three seed point forecasts",
        "distribution_rule": "concatenate all seed atoms, divide each seed weight by three"}
    return output


def forecast_rows(prediction, meta, data, crosschecks, crosscheck_seen):
    indices = prediction["example_indices"]
    anchor, scale, truth = (data[name][indices] for name in ("anchor", "scale", "price"))
    completed = data["completed"][indices]
    point = anchor + scale * prediction["mean_z"]
    atoms = anchor[:, None] + scale[:, None] * prediction["atoms_z"]
    weights = prediction["weights"]
    require(np.isfinite(point).all() and np.isfinite(atoms).all(), "Non-finite original-unit forecast")
    squared_error = np.full(len(indices), np.nan)
    absolute_error = np.full(len(indices), np.nan)
    crps = np.full(len(indices), np.nan)
    coverage = np.full(len(indices), np.nan)
    width = np.full(len(indices), np.nan)
    error = point[completed] - truth[completed]
    squared_error[completed], absolute_error[completed] = error ** 2, np.abs(error)
    scores, ordered, _, cdf = crps_cdf_integral(atoms[completed], weights[completed], truth[completed])
    require(np.isfinite(scores).all() and (scores >= -CRPS_ABS_TOLERANCE).all(), "Invalid price-unit CRPS")
    crps[completed] = scores
    lower = ordered[np.arange(len(scores)), np.argmax(cdf >= .1, axis=1)]
    upper = ordered[np.arange(len(scores)), np.argmax(cdf >= .9, axis=1)]
    coverage[completed] = (truth[completed] >= lower) & (truth[completed] <= upper)
    width[completed] = upper - lower
    kind = (prediction["seed"] == "ensemble", atoms.shape[1])
    if kind not in crosscheck_seen:
        use = np.flatnonzero(completed)
        selected = use[np.unique([0, len(use) // 2, len(use) - 1])]
        quadratic = crps_quadratic(atoms[selected], weights[selected], truth[selected])
        require(np.allclose(crps[selected], quadratic, atol=CRPS_ABS_TOLERANCE,
                            rtol=CRPS_REL_TOLERANCE), "Actual-forecast CRPS identities disagree")
        crosschecks.append({"model": prediction["model"], "year": prediction["year"],
                            "seed": prediction["seed"], "atom_count": atoms.shape[1], "n_rows": len(selected),
                            "max_absolute_difference": finite_max(np.abs(crps[selected] - quadratic))})
        crosscheck_seen.add(kind)
    result = {"indices": indices, "point": point, "squared_error": squared_error,
              "absolute_error": absolute_error, "crps": crps, "coverage": coverage, "width": width,
              "nonpositive_point": point <= 0, "nonpositive_atom_mass": np.sum(weights * (atoms <= 0), axis=1)}
    if "mechanical_failure" in prediction:
        result["mechanical_failure"] = prediction["mechanical_failure"]
        has_bid = data["has_bid"][indices]
        marginal_nll = np.full(len(indices), np.nan)
        mu = prediction["mean_bid_ratio"][has_bid].astype(np.float64)
        a = prediction["bid_concentration"][has_bid].astype(np.float64)
        mixture = prediction["bid_mixture"][has_bid].astype(np.float64)
        y = data["bid_ratio"][indices][has_bid, None]
        require((y > 0).all() and np.isfinite(y).all(), "Invalid observed B ratio")
        log_density = a * np.log(a / mu) - gammaln(a) + (a - 1) * np.log(y) - a * y / mu
        with np.errstate(divide="ignore"):
            marginal_nll[has_bid] = -logsumexp(np.log(mixture) + log_density, axis=1)
        require(np.isfinite(marginal_nll[has_bid]).all(), "Non-finite independent marginal B NLL")
        result["bid_marginal_nll"] = marginal_nll
    return result


def cells_from_rows(model, estimator, rows, data, horizons):
    indices = rows["indices"]
    cells = []
    year = int(data["issue_year"][indices[0]])
    for h in horizons:
        issued = data["horizon"][indices] == h
        use = issued & data["completed"][indices]
        require(use.any(), f"Empty completed cell {model}/{year}/{h}")
        cell = {"model": model, "estimator": estimator, "issue_year": year, "horizon": h,
                "n_issued": int(issued.sum()), "n_price_scored": int(use.sum()),
                "n_price_unscored": int((issued & ~data["completed"][indices]).sum()),
                "n_cancelled_records": int((issued & data["cancelled"][indices]).sum()),
                "price_mse": float(rows["squared_error"][use].mean()),
                "price_mae": float(rows["absolute_error"][use].mean()),
                "price_crps": float(rows["crps"][use].mean()),
                "interval80_coverage": float(rows["coverage"][use].mean()),
                "interval80_mean_width": float(rows["width"][use].mean()),
                "nonpositive_mean_forecasts_issued": int(rows["nonpositive_point"][issued].sum()),
                "mean_nonpositive_atom_mass_issued": float(rows["nonpositive_atom_mass"][issued].mean())}
        if "mechanical_failure" in rows:
            failure = rows["mechanical_failure"]
            cancels = issued & data["cancelled"][indices]
            bid_use = issued & data["has_bid"][indices]
            cell.update({"n_bid_scored": int(bid_use.sum()),
                         "marginal_bid_ratio_nll": float(rows["bid_marginal_nll"][bid_use].mean()) if bid_use.any() else None,
                         "mechanical_failure_mean_issued": float(failure[issued].mean()),
                         "mechanical_failure_mean_completed": float(failure[use].mean()),
                         "mechanical_failure_mean_cancelled": float(failure[cancels].mean()) if cancels.any() else None,
                         "mechanical_failure_rounded_one_count": int((failure[issued] == 1).sum())})
        cells.append(cell)
    return cells


def attach_rw_and_aggregate(cells, rw_cells, years, horizons):
    rw = {(c["issue_year"], c["horizon"]): c for c in rw_cells}
    for c in cells:
        reference = rw[(c["issue_year"], c["horizon"])]
        require(c["n_issued"] == reference["n_issued"] and c["n_price_scored"] == reference["n_price_scored"],
                "Comparator population differs from same-origin RW")
        require(reference["price_mse"] > 0 and reference["price_crps"] > 0, "Nonpositive RW denominator")
        c.update({"rw_price_mse": reference["price_mse"], "mse_ratio_to_rw": c["price_mse"] / reference["price_mse"],
                  "rw_price_crps": reference["price_crps"], "crps_ratio_to_rw": c["price_crps"] / reference["price_crps"]})
    groups = {}
    for c in cells:
        groups.setdefault((c["model"], c["estimator"]), []).append(c)
    aggregate = []
    for (model, estimator), group in groups.items():
        require(len(group) == len(years) * len(horizons), f"Incomplete aggregate cells for {model}/{estimator}")
        row = {"model": model, "estimator": estimator, "n_cells": len(group),
               "n_issued": sum(c["n_issued"] for c in group),
               "n_price_scored": sum(c["n_price_scored"] for c in group),
               "n_price_unscored": sum(c["n_price_unscored"] for c in group),
               "primary_equal_cell_mse_ratio_to_rw": float(np.mean([c["mse_ratio_to_rw"] for c in group])),
               "secondary_equal_cell_crps_ratio_to_rw": float(np.mean([c["crps_ratio_to_rw"] for c in group])),
               "equal_cell_price_mse": float(np.mean([c["price_mse"] for c in group])),
               "equal_cell_price_mae": float(np.mean([c["price_mae"] for c in group])),
               "equal_cell_price_crps": float(np.mean([c["price_crps"] for c in group])),
               "equal_cell_interval80_coverage": float(np.mean([c["interval80_coverage"] for c in group])),
               "equal_cell_interval80_width": float(np.mean([c["interval80_mean_width"] for c in group])),
               "cells_beating_rw_mse": sum(c["mse_ratio_to_rw"] < 1 for c in group),
               "nonpositive_mean_forecasts_issued": sum(c["nonpositive_mean_forecasts_issued"] for c in group)}
        for year in years:
            row[f"year_{year}_mse_ratio_to_rw"] = float(np.mean([c["mse_ratio_to_rw"] for c in group if c["issue_year"] == year]))
        if "marginal_bid_ratio_nll" in group[0]:
            row["equal_cell_marginal_bid_ratio_nll"] = float(np.mean([c["marginal_bid_ratio_nll"] for c in group]))
            row["equal_cell_mechanical_failure_mean_issued"] = float(np.mean([c["mechanical_failure_mean_issued"] for c in group]))
        aggregate.append(row)
    return aggregate


def compare_classical_published(root, cells, aggregate, classical_models):
    path = root / "results/classical/CLASSICAL_METRICS.json"
    published = json.loads(path.read_text())
    differences = []
    for model in classical_models:
        independent = [c for c in cells if c["model"] == model]
        saved = published["models"][model]
        old = {(c["issue_year"], c["horizon"]): c for c in saved["cells"]}
        for c in independent:
            reference = old[(c["issue_year"], c["horizon"])]
            for field in ("price_mse", "price_mae", "price_crps", "mse_ratio_to_rw", "interval80_coverage", "interval80_mean_width"):
                delta = abs(c[field] - reference[field])
                require(np.isclose(c[field], reference[field], rtol=1e-9, atol=1e-8),
                        f"Independent classical metric mismatch: {model}/{c['issue_year']}/{c['horizon']}/{field}")
                differences.append({"model": model, "year": c["issue_year"], "horizon": c["horizon"],
                                    "metric": field, "absolute_difference": delta})
        independent_aggregate = next(a for a in aggregate if a["model"] == model)
        metric = "primary_equal_cell_mse_ratio_to_rw"
        require(np.isclose(independent_aggregate[metric], saved[metric], atol=1e-10, rtol=1e-10),
                f"Classical aggregate mismatch: {model}")
    return {"status": "PASS", "published_metrics_sha256": sha256(path), "comparisons": len(differences),
            "max_absolute_difference": max(d["absolute_difference"] for d in differences),
            "by_metric_max_absolute_difference": {field: max(d["absolute_difference"] for d in differences if d["metric"] == field)
                                                  for field in sorted({d["metric"] for d in differences})},
            "all_differences": differences,
            "independence": "Published values read only after the independent price/CDF-integration metrics were computed."}


def block_bootstrap(meta, data, primary_rows, models, config, aggregate, cells, output):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    years, horizons = config["evaluation_issue_years"], config["horizons"]
    sampled_cells, original_cells, strata = [], [], []
    for year in years:
        ids = primary_rows[(models[0], year)]["indices"]
        dates, inverse = np.unique(meta.iloc[ids].issue_date.to_numpy(), return_inverse=True)
        n = len(dates)
        require(n >= BLOCK_LENGTH, "Not enough unique origins for the fixed moving block")
        count_cube = np.zeros((n, len(horizons)), dtype=np.float64)
        loss_cube = np.zeros((n, len(horizons), len(models)), dtype=np.float64)
        for j, horizon in enumerate(horizons):
            valid = (data["horizon"][ids] == horizon) & data["completed"][ids]
            np.add.at(count_cube[:, j], inverse[valid], 1)
            for k, model in enumerate(models):
                rows = primary_rows[(model, year)]
                require(np.array_equal(rows["indices"], ids), "Bootstrap pairing indices differ")
                np.add.at(loss_cube[:, j, k], inverse[valid], rows["squared_error"][valid])
        # Noncircular moving blocks: starts are 0,...,n-L; truncate only the
        # concatenated final block so every draw has exactly n origin positions.
        starts = rng.integers(0, n - BLOCK_LENGTH + 1,
                              size=(BOOTSTRAP_DRAWS, math.ceil(n / BLOCK_LENGTH)))
        sampled_positions = (starts[:, :, None] + np.arange(BLOCK_LENGTH)).reshape(BOOTSTRAP_DRAWS, -1)[:, :n]
        multiplicity = np.zeros((BOOTSTRAP_DRAWS, n), dtype=np.int32)
        np.add.at(multiplicity, (np.arange(BOOTSTRAP_DRAWS)[:, None], sampled_positions), 1)
        require((multiplicity.sum(axis=1) == n).all(), "Bootstrap origin count changed")
        draw_counts = multiplicity @ count_cube
        require((draw_counts > 0).all(), "Bootstrap draw has empty scored horizon cell")
        draw_losses = (multiplicity @ loss_cube.reshape(n, -1)).reshape(BOOTSTRAP_DRAWS, len(horizons), len(models))
        sampled_cells.append(draw_losses / draw_counts[:, :, None])
        original_cells.append(loss_cube.sum(axis=0) / count_cube.sum(axis=0)[:, None])
        strata.append({"issue_year": year, "n_unique_issue_dates": n, "block_length": BLOCK_LENGTH,
                       "possible_block_starts": n - BLOCK_LENGTH + 1,
                       "blocks_per_draw_before_final_truncation": math.ceil(n / BLOCK_LENGTH),
                       "first_issue_date": dates[0], "last_issue_date": dates[-1],
                       "n_forecast_records": len(ids), "n_price_scored": int(count_cube.sum()),
                       "n_platforms_preserved": int(meta.iloc[ids].platform.nunique()),
                       "sampled_position_matrix_sha256": hashlib.sha256(sampled_positions.astype("<i8").tobytes()).hexdigest(),
                       "all_model_indices_identical": True})
    mse_draws = np.concatenate(sampled_cells, axis=1)
    original = np.concatenate(original_cells, axis=0)
    rw_index, sdmc_index = models.index("RW"), models.index("SDMC")
    require((mse_draws[:, :, rw_index] > 0).all(), "Bootstrap RW cell denominator nonpositive")
    r_draws = (mse_draws / mse_draws[:, :, rw_index, None]).mean(axis=1)
    point_r = (original / original[:, rw_index, None]).mean(axis=0)
    lookup = {a["model"]: a for a in aggregate}
    for k, model in enumerate(models):
        require(np.isclose(point_r[k], lookup[model]["primary_equal_cell_mse_ratio_to_rw"], rtol=1e-12, atol=1e-12),
                "Bootstrap original statistic differs from main table")
    ci_rows, secondary_rows, ratio_draws, direct_draws = [], [], [], []
    for comparator in COMPARATORS:
        k = models.index(comparator)
        ratio = r_draws[:, sdmc_index] / r_draws[:, k]
        estimate = point_r[sdmc_index] / point_r[k]
        interval = np.quantile(ratio, [.025, .975], method="linear")
        ci_rows.append({"model": "SDMC", "fixed_comparator": comparator,
                        "primary_R_ratio": float(estimate), "lower_95": float(interval[0]), "upper_95": float(interval[1]),
                        "bootstrap_proportion_ratio_below_one": float(np.mean(ratio < 1)),
                        "n_draws": BOOTSTRAP_DRAWS, "block_length_origins": BLOCK_LENGTH,
                        "comparator_reselected_in_bootstrap": False})
        direct = (mse_draws[:, :, sdmc_index] / mse_draws[:, :, k]).mean(axis=1)
        direct_interval = np.quantile(direct, [.025, .975], method="linear")
        secondary_rows.append({"model": "SDMC", "fixed_comparator": comparator,
                               "secondary_equal_cell_direct_mse_ratio": float((original[:, sdmc_index] / original[:, k]).mean()),
                               "lower_95": float(direct_interval[0]), "upper_95": float(direct_interval[1]),
                               "used_for_advancement_gate": False})
        ratio_draws.append(ratio)
        direct_draws.append(direct)
    np.savez_compressed(output / "BOOTSTRAP_REPLICATES.npz",
                        models=np.asarray(models), comparators=np.asarray(COMPARATORS),
                        aggregate_R=r_draws, primary_R_ratios=np.stack(ratio_draws, axis=1),
                        secondary_direct_cell_ratios=np.stack(direct_draws, axis=1),
                        seed=np.asarray(BOOTSTRAP_SEED), block_length=np.asarray(BLOCK_LENGTH))
    return {"status": "PASS", "random_generator": "NumPy default_rng / PCG64",
            "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "block_length_origins": BLOCK_LENGTH,
            "method": "Year-stratified paired noncircular moving-block percentile bootstrap; all horizons/platforms at one issue_date share multiplicity.",
            "primary_statistic": "R_SDMC / R_fixed_comparator; R_m = mean over nine cells of MSE_m_cell / MSE_RW_cell, recomputed within each bootstrap draw.",
            "secondary_statistic": "mean over nine cells of MSE_SDMC_cell / MSE_fixed_comparator_cell; descriptive, not the gate statistic.",
            "interval_quantiles": [.025, .975], "quantile_method": "linear", "strata": strata,
            "primary_comparisons": ci_rows, "secondary_comparisons": secondary_rows,
            "limitations": ["Seeds are averaged before resampling and never count as independent market observations.",
                            "Comparator identities are fixed at their original-sample identities, including the selected best matched comparator.",
                            "No model-selection adjustment, simultaneous-coverage guarantee, or bootstrap model refitting is claimed.",
                            "Platforms receive their original within-cell record weights; the bootstrap does not impose equal platform weighting.",
                            "Block length and stratification are fixed protocol choices; there is no score-driven block-length search."]}


def advancement_gates(aggregate, bootstrap):
    lookup = {a["model"]: a for a in aggregate}
    r = {model: a["primary_equal_cell_mse_ratio_to_rw"] for model, a in lookup.items()}
    best = min(MATCHED, key=lambda name: (r[name], MATCHED.index(name)))
    sdmc = lookup["SDMC"]
    ci = next(c for c in bootstrap["primary_comparisons"] if c["fixed_comparator"] == best)
    worst_year = max(sdmc[f"year_{year}_mse_ratio_to_rw"] for year in (2021, 2022, 2023))
    checks = {
        "MSE_vs_RW": {"value": r["SDMC"], "criterion": "R_SDMC <= 0.98", "pass": r["SDMC"] <= .98},
        "MSE_vs_best_matched_predictor": {"value": r["SDMC"] / r[best], "comparator": best,
                                           "criterion": "R_SDMC / R_best_matched <= 0.98", "pass": r["SDMC"] / r[best] <= .98},
        "mean_demand_integration": {"value": r["SDMC"] / r["SDMC_plugin"], "criterion": "R_SDMC / R_plugin <= 0.99",
                                    "pass": r["SDMC"] / r["SDMC_plugin"] <= .99},
        "supply_operator": {"value": r["SDMC"] / r["SDMC_no_supply"], "criterion": "R_SDMC / R_no_supply <= 0.99",
                            "pass": r["SDMC"] / r["SDMC_no_supply"] <= .99},
        "stability": {"cells_beating_RW": sdmc["cells_beating_rw_mse"], "worst_year_R": worst_year,
                      "criterion": "At least 6/9 cells strictly beat RW and every year R <= 1.10",
                      "pass": sdmc["cells_beating_rw_mse"] >= 6 and worst_year <= 1.10},
        "uncertainty": {"comparator": best, "lower_95": ci["lower_95"], "upper_95": ci["upper_95"],
                        "criterion": "Paired percentile 95% upper bound of R_SDMC/R_fixed_best < 1",
                        "pass": ci["upper_95"] < 1}}
    all_pass = all(c["pass"] for c in checks.values())
    return {"all_development_gates_pass": all_pass, "decision": "ELIGIBLE_FOR_FROZEN_ONCE_ONLY_CONFIRMATION" if all_pass else "STOP_LEAVE_CONFIRMATION_UNSCORED",
            "best_matched_comparator_fixed_from_original_sample": best,
            "matched_candidates_in_fixed_tie_order": list(MATCHED), "checks": checks,
            "confirmation_access_performed_by_this_script": False,
            "no_score_driven_model_selection_or_training_change": True}


def write_review(output, aggregate, inventory, checks, gates, bootstrap, mode, population):
    lines = ["# 独立评分审计", "", "## 结论", "",
             "评分实现及预测身份核查通过。" + (f"冻结开发门槛结果：**{gates['decision']}**。" if gates else "本次仅复算已完成经典基线。"),
             "", "本脚本没有导入生产模型、训练 runner 或其评分函数；没有训练或修改模型，没有读取 2024—2025 确认数据。", "",
             "## 主指标与估计对象", "",
             "主表使用 3 个 issue year × 3 个 horizon 共 9 格等权的 MSE/RW MSE。神经点预测先平均 3 个 seed，概率分布将其全部原子按 seed 权重 1/3 拼接。真实价格直接取冻结 examples 中的 float64 price。", "",
             "| 模型 | R：9 格等权 MSE/RW | CRPS/RW | 赢 RW 格数 | 原价 MAE | 80%覆盖率 |", "|---|---:|---:|---:|---:|---:|"]
    for row in sorted(aggregate, key=lambda a: a["primary_equal_cell_mse_ratio_to_rw"]):
        lines.append(f"| {row['model']} | {row['primary_equal_cell_mse_ratio_to_rw']:.9f} | {row['secondary_equal_cell_crps_ratio_to_rw']:.9f} | {row['cells_beating_rw_mse']}/9 | {row['equal_cell_price_mae']:.6f} | {row['equal_cell_interval80_coverage']:.6f} |")
    lines += ["", "## 样本与数值审计", "",
              f"共核验 {len(inventory)} 个正式预测 NPZ；预测覆盖 {population['n_unique_issue_dates']} 个独立日期起点和 {population['n_issued_records']} 条 horizon/platform 记录，其中 {population['n_price_scored_records']} 条有真实价格可评分。seed 数不作为市场样本量。",
              f"无价格的 {population['n_price_unscored_records']} 条记录保留预测并排除价格损失；取消记录按目标事件识别，跨 horizon 的重复预测记录不冒充多个独立取消事件。",
              "所有 NPZ 的 indices 与冻结 metadata 精确匹配，anchor/scale/y_z 与冻结 float64 源逐元素一致；审计报告逐文件最大权重和误差、点预测与原子均值差异。", "",
              "CRPS 主实现按有限 CDF 的平方逐区间积分；另用 O(K²) 的期望距离定义互证。经典公开评分只在本脚本完成独立复算后用于核对。"]
    if bootstrap:
        lines += ["", "## 固定对照配对区间", "",
                  "每年分别按 unique issue_date 排序，以 20 个连续日期位置组成不环绕 moving block，有放回抽块并截到该年原日期数；同日全部 horizon/platform 跟随同一 multiplicity，所有模型及 RW 使用相同抽样。2000 次，固定 seed=20261018。",
                  "", "主区间与 gate 均针对 R_SDMC/R_固定对照；每次 bootstrap 在格内重算 MSE 与 RW 分母，然后重算 9 格等权 R。另一种 mean(格内 SDMC/对照 MSE) 仅在 SECONDARY_DIRECT_CELL_CI.csv 中报告，不能混称主区间。", "",
                  "| 固定对照 | R 比值 | 95% 下限 | 95% 上限 |", "|---|---:|---:|---:|"]
        for row in bootstrap["primary_comparisons"]:
            lines.append(f"| {row['fixed_comparator']} | {row['primary_R_ratio']:.9f} | {row['lower_95']:.9f} | {row['upper_95']:.9f} |")
        lines += ["", "## 冻结门槛", "", "| 门槛 | 结果 | 预先规定条件 |", "|---|---|---|"]
        for name, gate in gates["checks"].items():
            lines.append(f"| {name} | {'PASS' if gate['pass'] else 'FAIL'} | {gate['criterion']} |")
    lines += ["", "## 必须保留的限定", "",
              "这是一轮所有开发年份已在历史项目中曝光后的前瞻冻结开发实验；不能称开发集从未看过，也不能将当前结果写成封存确认成绩。",
              "", "最佳 matched 对照按原样本冻结候选的 R 确定，bootstrap 内不重选。区间不包含对照选择修正，不提供全部比较同时成立的覆盖保证，也不估计重新训练的不确定性。平台仍按原来的格内记录权重计入。",
              "", "机械 failure 表示模型下 B<S 的概率，不能称其已校准为真实取消概率。边际 B NLL 在 B/B₀ 单位下计算；没有使用它改写主价格 gate。",
              "", "拍卖结果次日 00:00 UTC 可用是已披露的历史数据协议假设，本评分审计不认证字段实际历史发布时间及后续修订。", ""]
    (output / "INDEPENDENT_REVIEW_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--mode", choices=("full", "classical-only"), default="full")
    args = parser.parse_args()
    root = args.study_dir.resolve()
    output = root / "results/independent_metrics"
    if args.mode == "classical-only":
        output /= "classical_validation"
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    checks = {"synthetic_CRPS": check_crps_synthetic(), "criteria": {
        "weight_sum_absolute_tolerance": WEIGHT_TOLERANCE,
        "distribution_mean_absolute_tolerance_z": MOMENT_ABS_TOLERANCE,
        "distribution_mean_relative_tolerance_z": MOMENT_REL_TOLERANCE,
        "CRPS_absolute_tolerance_price": CRPS_ABS_TOLERANCE,
        "CRPS_relative_tolerance": CRPS_REL_TOLERANCE,
        "stored_anchor_scale_y_identity": "exact elementwise equality including missing values"}}
    try:
        config, freeze, meta, data, freeze_audit = read_frozen_inputs(root)
        completion = require_run_completion(root, args.mode, freeze_audit)
        years, horizons = config["evaluation_issue_years"], config["horizons"]
        models = config["classical_models"] + (config["neural_models"] if args.mode == "full" else [])
        inventory, ensemble_audits, cells, seed_cells, crosschecks = [], [], [], [], []
        seen, primary_rows = set(), {}
        for model in models:
            neural = model in config["neural_models"]
            for year in years:
                predictions = []
                for seed in config["refit_seeds"] if neural else [None]:
                    prediction, audit = read_prediction(root, model, year, seed, meta, data)
                    inventory.append(audit)
                    predictions.append(prediction)
                    if neural:
                        rows = forecast_rows(prediction, meta, data, crosschecks, seen)
                        seed_cells.extend(cells_from_rows(model, f"seed_{seed}", rows, data, horizons))
                primary = ensemble_prediction(predictions, data) if neural else predictions[0]
                if neural:
                    ensemble_audits.append(primary["ensemble_audit"])
                rows = forecast_rows(primary, meta, data, crosschecks, seen)
                primary_rows[(model, year)] = rows
                cells.extend(cells_from_rows(model, "three_seed_ensemble" if neural else "deterministic", rows, data, horizons))
        expected_files = len(config["classical_models"]) * len(years)
        if args.mode == "full":
            expected_files += len(config["neural_models"]) * len(years) * len(config["refit_seeds"])
        require(len(inventory) == expected_files, "Formal prediction file count mismatch")
        rw_cells = [c for c in cells if c["model"] == "RW"]
        aggregate = attach_rw_and_aggregate(cells, rw_cells, years, horizons)
        seed_aggregate = attach_rw_and_aggregate(seed_cells, rw_cells, years, horizons) if seed_cells else []
        checks["actual_forecast_CRPS_crosschecks"] = crosschecks
        checks["classical_published_score_comparison"] = compare_classical_published(root, cells, aggregate, config["classical_models"])
        checks["all_prediction_identity_precision_and_finiteness_checks"] = "PASS"
        checks["scorer_scope_amendment"] = {
            "initial_failed_scorer_sha256": "aa5879e1ab5bb986550893bd1976f8e3f71e59e37d6d9b214619a4f3161d9104",
            "initial_failure": "The independent scorer incorrectly required finite Gamma diagnostics for GRU_full, whose frozen non-Gamma model saves explicit NaN placeholders.",
            "correction": "For the fixed non-Gamma GRU_full/GRU_price/MLP_full identities, verify NaN placeholders and mark Gamma diagnostics not applicable; retain all strict price/atom/weight checks.",
            "prediction_or_training_files_changed": False,
            "first_failure_receipt_retained": True,
            "main_statistic_and_bootstrap_definition_changed": False}
        checks["ensemble_vs_mean_seed_loss"] = []
        for a in aggregate:
            group = [s for s in seed_aggregate if s["model"] == a["model"]]
            if group:
                average_seed_r = float(np.mean([s["primary_equal_cell_mse_ratio_to_rw"] for s in group]))
                require(a["primary_equal_cell_mse_ratio_to_rw"] <= average_seed_r + 1e-10, "Point ensemble MSE violates convexity")
                checks["ensemble_vs_mean_seed_loss"].append({"model": a["model"],
                    "ensemble_R": a["primary_equal_cell_mse_ratio_to_rw"], "mean_individual_seed_R_not_primary": average_seed_r,
                    "ensemble_CRPS_ratio": a["secondary_equal_cell_crps_ratio_to_rw"],
                    "mean_individual_seed_CRPS_ratio_not_primary": float(np.mean([s["secondary_equal_cell_crps_ratio_to_rw"] for s in group]))})
        bootstrap = block_bootstrap(meta, data, primary_rows, models, config, aggregate, cells, output) if args.mode == "full" else None
        gates = advancement_gates(aggregate, bootstrap) if bootstrap else None
        evaluation = np.isin(data["issue_year"], years)
        scored = evaluation & data["completed"]
        cancelled = evaluation & data["cancelled"]
        target_columns = ["target_date", "platform", "result_event_id"]
        population = {"n_unique_issue_dates": int(meta.loc[evaluation, "issue_date"].nunique()),
                      "n_issued_records": int(evaluation.sum()), "n_price_scored_records": int(scored.sum()),
                      "n_price_unscored_records": int((evaluation & ~data["completed"]).sum()),
                      "n_cancelled_forecast_records": int(cancelled.sum()),
                      "n_unique_cancelled_target_events": int(len(meta.loc[cancelled, target_columns].drop_duplicates())),
                      "market_N_is_not_seed_count": True,
                      "all_unscored_forecasts_retained": True,
                      "unknown_or_cancelled_price_imputed_zero": False}
        audit = {"overall_status": "PASS", "mode": args.mode, "expected_formal_prediction_npz": expected_files,
                 "observed_formal_prediction_npz": len(inventory),
                 "formal_classical_npz": sum(a["seed"] is None for a in inventory),
                 "formal_neural_npz": sum(a["seed"] is not None for a in inventory),
                 "validation_npz_excluded_from_test_flows": len(list((root / "local_only/neural").glob("*_validation.npz"))),
                 "max_weight_sum_error": max(a["max_weight_sum_error"] for a in inventory),
                 "max_stored_mean_vs_atoms_error_z": max(a["max_stored_mean_vs_weighted_atoms_error_z"] for a in inventory),
                 "max_stored_mean_vs_atoms_error_price": max(a["max_stored_mean_vs_weighted_atoms_error_price"] for a in inventory),
                 "max_saved_source_numeric_difference": max(v["max_absolute_difference"] for a in inventory for v in a["source_precision_matches"].values()),
                 "files": inventory, "ensembles": ensemble_audits, "population": population}
        summary = {"overall_status": "PASS", "mode": args.mode, "completed_utc": datetime.now(timezone.utc).isoformat(),
                   "script_sha256": sha256(__file__), "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
                   "elapsed_seconds": time.perf_counter() - started,
                   "freeze_audit": freeze_audit, "run_completion": completion, "population": population,
                   "primary_definition": "R_m=arithmetic mean over nine year/horizon cells of raw-price MSE_m / same-record RW MSE",
                   "neural_ensemble_definition": "Equal-weight point mean and equally weighted concatenation of three seed distributions, scored once",
                   "actual_price_source": "local_only/examples_development.npz:price, float64",
                   "CRPS_implementation": "Direct piecewise integration of squared empirical CDF; independent quadratic identity cross-check",
                   "primary_aggregate": aggregate, "primary_cells": cells,
                   "per_seed_aggregate_descriptive_only": seed_aggregate,
                   "bootstrap": bootstrap, "gates": gates,
                   "confirmation_data_read": False, "training_runs_by_this_script": 0,
                   "production_modules_imported": [], "production_score_functions_used": False}
        for name, records in (("FULL_CELLS.csv", cells), ("AGGREGATE.csv", aggregate),
                              ("PER_SEED_CELLS.csv", seed_cells), ("PER_SEED_AGGREGATE.csv", seed_aggregate)):
            pd.DataFrame(records).to_csv(output / name, index=False, float_format="%.17g")
        if bootstrap:
            pd.DataFrame(bootstrap["primary_comparisons"]).to_csv(output / "PAIRED_CI.csv", index=False, float_format="%.17g")
            pd.DataFrame(bootstrap["secondary_comparisons"]).to_csv(output / "SECONDARY_DIRECT_CELL_CI.csv", index=False, float_format="%.17g")
            write_json(output / "PAIRED_CI.json", bootstrap)
            write_json(output / "GATES.json", gates)
        write_json(output / "SCORING_CHECKS.json", checks)
        write_json(output / "PREDICTION_AUDIT.json", audit)
        write_json(output / "INDEPENDENT_METRICS.json", summary)
        write_review(output, aggregate, inventory, checks, gates, bootstrap, args.mode, population)
        manifest = {"status": "COMPLETE", "script_sha256": sha256(__file__), "outputs": []}
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name != "OUTPUT_MANIFEST.json" and not path.name.endswith(".tmp"):
                manifest["outputs"].append({"path": str(path.relative_to(root)), "sha256": sha256(path), "bytes": path.stat().st_size})
        write_json(output / "OUTPUT_MANIFEST.json", manifest)
        print(json.dumps(plain({"status": "PASS", "mode": args.mode, "prediction_npz": len(inventory),
                                "population": population, "aggregate_R": {a["model"]: a["primary_equal_cell_mse_ratio_to_rw"] for a in aggregate},
                                "gates": gates, "output": str(output), "elapsed_seconds": time.perf_counter() - started}), ensure_ascii=False))
    except Exception as error:
        failure = {"status": "FAIL", "mode": args.mode, "error": repr(error), "traceback": traceback.format_exc(),
                   "script_sha256": sha256(__file__), "elapsed_seconds": time.perf_counter() - started,
                   "no_training_or_confirmation_access": True}
        name = "SCORING_FAILURE_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
        write_json(output / name, failure)
        print(json.dumps(failure, ensure_ascii=False))
        raise


if __name__ == "__main__":
    main()
