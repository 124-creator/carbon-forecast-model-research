"""Frozen finite synthetic qualification of forecast-CDF filter geometry.

Synthetic observations only. No neural fitting, market inputs or test tuning.
All eligible filters use the same nominal future density family. The DGP oracle
is used only after selection to calculate conditional expected forecast scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
from scipy.special import logsumexp, ndtr
from scipy.stats import binom

import geometry_filter as gf


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "provenance" / "SYNTHETIC_PROTOCOL.json"
SQRT2PI = np.sqrt(2.0 / np.pi)
HORIZONS = (1, 5, 20)
PRIMARY = (5, 20)
COMPETITORS = ("current_cdf", "future_variance", "probability_euclidean",
               "rotated_future_cdf", "chi_square_only", "crps_kl",
               "tempered_bayes", "contaminated_bayes")
PROPAGATION_ACCOUNTING = {}


def reset_propagation_accounting():
    PROPAGATION_ACCOUNTING.clear()
    PROPAGATION_ACCOUNTING.update(
        calls=0, rows=0, rows_with_nonunit_sum_before_normalization=0,
        rows_with_any_coordinate_changed=0, maximum_absolute_mass_error=0.0,
        maximum_absolute_coordinate_change=0.0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n")


def normal_abs(y, sd):
    z = np.asarray(y) / np.asarray(sd)
    return np.asarray(sd) * SQRT2PI * np.exp(-.5 * z * z) + np.asarray(y) * (2 * ndtr(z) - 1)


def transition(slow: float, fast: float) -> np.ndarray:
    eye = np.eye(2)
    uniform = np.full((2, 2), .5)
    return np.kron(slow * eye + (1 - slow) * uniform,
                   fast * eye + (1 - fast) * uniform)


def simulate(seed: int, scenario: dict, n: int, variances: np.ndarray):
    """Reuse the same four draw streams across scenarios within a seed."""
    rng = np.random.default_rng(seed)
    uniforms = rng.random(n)
    innovations = rng.standard_normal(n)
    shock_flags = rng.random(n)
    shocks = rng.standard_normal(n)
    p = transition(scenario["slow_persistence"], scenario["fast_persistence"])
    cdf = np.cumsum(p, axis=1)
    states = np.empty(n, dtype=np.int64)
    states[0] = min(3, int(uniforms[0] * 4))
    for t in range(1, n):
        states[t] = np.searchsorted(cdf[states[t - 1]], uniforms[t], side="right")
    clean = np.sqrt(variances[states]) * innovations
    contaminated = shock_flags < scenario["contamination_probability"]
    observations = clean + contaminated * scenario["additive_shock_sd"] * shocks
    return states, observations, clean, contaminated


def oracle_states(obs: np.ndarray, p: np.ndarray, variances: np.ndarray,
                  epsilon: float, shock_sd: float) -> np.ndarray:
    """True Bayes posterior from each method-independent observed history."""
    n_seeds, n_times = obs.shape
    result = np.empty((n_seeds, n_times, 4), dtype=np.float64)
    prior = np.full((n_seeds, 4), .25)
    for t in range(n_times):
        y2 = obs[:, t, None] ** 2
        ll = -.5 * (np.log(2 * np.pi * variances) + y2 / variances)
        if epsilon:
            wide = variances + shock_sd ** 2
            ll_wide = -.5 * (np.log(2 * np.pi * wide) + y2 / wide)
            ll = np.logaddexp(np.log1p(-epsilon) + ll,
                              np.log(epsilon) + ll_wide)
        lpost = np.log(prior) + ll
        posterior = np.exp(lpost - logsumexp(lpost, axis=1, keepdims=True))
        result[:, t] = posterior
        prior = posterior @ p
    return result


def oracle_kernels(reference: dict, scenario: dict):
    """Exact distance matrices; future contamination is never discarded.

    Pair distances need only the binomial count of independent added shocks.
    They do not require enumerating all future state or contamination paths.
    """
    true_p = transition(scenario["slow_persistence"], scenario["fast_persistence"])
    result = {}
    for h in HORIZONS:
        model = reference["conditional"][h]
        wt, st = gf.conditional_mixture(true_p, reference["v"], h)
        wm, sm = model["weights"], model["sds"]
        eps, tau = scenario["contamination_probability"], scenario["additive_shock_sd"]
        cross_distance = np.zeros((len(sm), len(st)))
        true_distance = np.zeros((len(st), len(st)))
        for count in range(h + 1):
            mass = float(binom.pmf(count, h, eps))
            if mass:
                cross_distance += mass * SQRT2PI * np.sqrt(
                    sm[:, None] ** 2 + st[None, :] ** 2 + count * tau ** 2 / h)
        for count in range(2 * h + 1):
            mass = float(binom.pmf(count, 2 * h, eps))
            if mass:
                true_distance += mass * SQRT2PI * np.sqrt(
                    st[:, None] ** 2 + st[None, :] ** 2 + count * tau ** 2 / h)
        result[h] = {"cross": wm @ cross_distance @ wt.T,
                     "true_pair": wt @ true_distance @ wt.T,
                     "true_weights": wt, "true_sds": st}
    return result


def target_features(obs: np.ndarray, reference: dict, origins: np.ndarray):
    cumulative = np.concatenate([np.zeros((len(obs), 1)), np.cumsum(obs, axis=1)], axis=1)
    ys, expected_absolute = {}, {}
    for h in HORIZONS:
        y = (cumulative[:, origins + h + 1] - cumulative[:, origins + 1]) / np.sqrt(h)
        comp = reference["conditional"][h]
        a = normal_abs(y[:, :, None], comp["sds"][None, None, :]) @ comp["weights"].T
        ys[h], expected_absolute[h] = y, a
    return ys, expected_absolute


def empirical_score(q: np.ndarray, a: np.ndarray, pair: np.ndarray):
    return np.sum(q * a, axis=-1) - .5 * np.einsum("...i,ij,...j->...", q, pair, q)


def expected_score(q: np.ndarray, truth: np.ndarray, pair: np.ndarray, cross: np.ndarray):
    return np.einsum("...i,ij,...j->...", q, cross, truth) - .5 * np.einsum(
        "...i,ij,...j->...", q, pair, q)


def apply_update(family: str, prior: np.ndarray, y: np.ndarray,
                 parameter: np.ndarray, ref: dict):
    if family == "tempered_bayes":
        return gf.tempered_bayes(prior, y, parameter)
    if family == "contaminated_bayes":
        return gf.contaminated_bayes(prior, y, parameter)
    if family == "crps_kl":
        return gf.update_crps_kl(prior, y, parameter)
    key = {"future_cdf": "future", "current_cdf": "current",
           "future_variance": "variance", "probability_euclidean": "euclid",
           "rotated_future_cdf": "rotated",
           "future_cdf_ratio_preserving": "future"}.get(family)
    g = np.zeros((4, 4)) if family == "chi_square_only" else ref["metrics"][key]
    return gf.update_qp(prior, y, parameter, g,
                        kappa=1.0 if family == "chi_square_only" else .01,
                        ratio_preserving=family == "future_cdf_ratio_preserving")


def assert_probabilities(q: np.ndarray):
    if not np.isfinite(q).all() or np.min(q) < -1e-12 or np.max(np.abs(q.sum(axis=-1) - 1)) > 2e-10:
        raise AssertionError("Invalid probability mass beyond the separately recorded QP roundoff repair")


def advance(q: np.ndarray, p: np.ndarray):
    prior = q @ p
    assert_probabilities(prior)
    # Analytically this denominator is one. Apply the same roundoff convention
    # to all recursions, including the zero-update configurations.
    denominator = prior.sum(axis=-1, keepdims=True)
    normalized = prior / denominator
    if not PROPAGATION_ACCOUNTING:
        reset_propagation_accounting()
    changes = np.abs(normalized - prior)
    c = PROPAGATION_ACCOUNTING
    c["calls"] += 1
    c["rows"] += int(prior.size // prior.shape[-1])
    c["rows_with_nonunit_sum_before_normalization"] += int(np.count_nonzero(denominator != 1))
    c["rows_with_any_coordinate_changed"] += int(np.count_nonzero(np.any(changes != 0, axis=-1)))
    c["maximum_absolute_mass_error"] = max(c["maximum_absolute_mass_error"], float(np.max(np.abs(denominator - 1))))
    c["maximum_absolute_coordinate_change"] = max(c["maximum_absolute_coordinate_change"], float(np.max(changes)))
    return normalized


def oracle_empirical_features(y: np.ndarray, kernel: dict, scenario: dict, h: int):
    wt, st = kernel["true_weights"], kernel["true_sds"]
    result = np.zeros(y.shape + (4,))
    eps, tau = scenario["contamination_probability"], scenario["additive_shock_sd"]
    for count in range(h + 1):
        mass = float(binom.pmf(count, h, eps))
        if mass:
            sd = np.sqrt(st * st + count * tau * tau / h)
            result += mass * (normal_abs(y[..., None], sd) @ wt.T)
    return result


def family_grid(family: str, protocol: dict):
    key = "beta" if family == "tempered_bayes" else "epsilon" if family == "contaminated_bayes" else "eta"
    return np.asarray(protocol["grids"][key], dtype=np.float64)


def run_scenario(scenario: dict, protocol: dict, reference: dict, out: Path, public: Path):
    started = time.monotonic()
    seeds = protocol["simulation"]["seeds"]
    families = protocol["families"]
    sim = protocol["simulation"]
    n = sim["total_observations_per_scenario_seed"]
    items = [simulate(seed, scenario, n, reference["v"]) for seed in seeds]
    states, obs, clean, flags = [np.stack([item[j] for item in items]) for j in range(4)]
    validation_origins = np.arange(sim["validation_start_index"], sim["validation_last_index"] + 1)
    test_origins = np.arange(sim["test_start_index"], sim["test_last_index"] + 1)
    _, val_a = target_features(obs, reference, validation_origins)
    n_seeds, n_grid = len(seeds), len(protocol["grids"]["eta"])
    val_scores, chosen_index, chosen_parameter, chosen_prior = {}, {}, {}, {}
    validation_trials = []
    nominal_bayes_prior = None
    phase_receipts = []

    for family in families:
        family_start = time.monotonic()
        grid = family_grid(family, protocol)
        assert len(grid) == n_grid
        parameters = np.broadcast_to(grid, (n_seeds, n_grid)).reshape(-1)
        prior = np.full((n_seeds * n_grid, 4), .25)
        totals = np.zeros((n_seeds, n_grid, len(HORIZONS)))
        for t in range(sim["test_start_index"]):
            y = np.repeat(obs[:, t], n_grid)
            q = apply_update(family, prior, y, parameters, reference)
            assert_probabilities(q)
            if sim["validation_start_index"] <= t <= sim["validation_last_index"]:
                j = t - sim["validation_start_index"]
                grouped_q = q.reshape(n_seeds, n_grid, 4)
                for hi, h in enumerate(HORIZONS):
                    totals[:, :, hi] += empirical_score(
                        grouped_q, val_a[h][:, j, None, :], reference["conditional"][h]["pair"])
            prior = advance(q, reference["P"])
        scores = totals / len(validation_origins)
        mean_scores = scores.mean(axis=-1)
        index = np.argmin(mean_scores, axis=1)
        val_scores[family] = scores
        chosen_index[family] = index
        chosen_parameter[family] = grid[index]
        chosen_prior[family] = prior.reshape(n_seeds, n_grid, 4)[np.arange(n_seeds), index].copy()
        if family == "tempered_bayes":
            nominal_bayes_prior = prior.reshape(n_seeds, n_grid, 4)[:, np.flatnonzero(grid == 1)[0]].copy()
        for si, seed in enumerate(seeds):
            for gi, parameter in enumerate(grid):
                validation_trials.append({
                    "scenario": scenario["name"], "seed": seed, "family": family,
                    "grid_index": gi, "parameter": float(parameter),
                    "validation_crps_h1": float(scores[si, gi, 0]),
                    "validation_crps_h5": float(scores[si, gi, 1]),
                    "validation_crps_h20": float(scores[si, gi, 2]),
                    "validation_crps_mean": float(mean_scores[si, gi]),
                    "selected_within_family": bool(index[si] == gi)})
        elapsed = time.monotonic() - family_start
        phase_receipts.append({"phase": "validation", "family": family,
                               "trials": n_seeds * n_grid,
                               "recursive_updates": n_seeds * n_grid * sim["test_start_index"],
                               "elapsed_seconds": elapsed})
        print(json.dumps({"scenario": scenario["name"], "phase": "validation_complete",
                          "family": family, "seconds": round(elapsed, 3)}), flush=True)

    # Freeze every hyperparameter and the composite competitor before computing
    # any test metric or obtaining the DGP oracle's filtered state probabilities.
    selected_val = {f: val_scores[f][np.arange(n_seeds), chosen_index[f]].mean(axis=1) for f in families}
    competitor_val = np.stack([selected_val[f] for f in COMPETITORS], axis=1)
    selected_competitors = [COMPETITORS[i] for i in np.argmin(competitor_val, axis=1)]
    selection = {"scenario": scenario["name"], "selection_observation_index": int(test_origins[0]),
                 "last_validation_label_index": sim["last_validation_label_index"],
                 "no_test_metric_computed": True,
                 "seeds": [{"seed": seed, "validation_selected_competitor": selected_competitors[si],
                            "families": {f: {"grid_index": int(chosen_index[f][si]),
                                               "parameter": float(chosen_parameter[f][si]),
                                               "validation_crps": float(selected_val[f][si])}
                                         for f in families}} for si, seed in enumerate(seeds)]}
    selection_path = public / scenario["name"] / "SELECTION_BEFORE_TEST.json"
    dump(selection_path, selection)
    selection_sha = sha(selection_path)

    true_transition = transition(scenario["slow_persistence"], scenario["fast_persistence"])
    truth_all = oracle_states(obs, true_transition, reference["v"],
                              scenario["contamination_probability"], scenario["additive_shock_sd"])
    truth = truth_all[:, test_origins]
    kernels = oracle_kernels(reference, scenario)
    test_y, test_a = target_features(obs, reference, test_origins)
    predictions, odds_diagnostics = {}, {}

    for family in families + ["nominal_bayes", "fixed_prior"]:
        family_start = time.monotonic()
        if family == "fixed_prior":
            predictions[family] = np.full((n_seeds, len(test_origins), 4), .25)
            continue
        prior = nominal_bayes_prior.copy() if family == "nominal_bayes" else chosen_prior[family].copy()
        parameters = np.ones(n_seeds) if family == "nominal_bayes" else chosen_parameter[family]
        update_family = "tempered_bayes" if family == "nominal_bayes" else family
        qs = np.empty((n_seeds, len(test_origins), 4))
        cross_values = np.empty((n_seeds, len(test_origins)))
        log_deltas = np.full((n_seeds, len(test_origins)), np.nan)
        boundary_counts = np.zeros(n_seeds, dtype=np.int64)
        for j, t in enumerate(test_origins):
            q = apply_update(update_family, prior, obs[:, t], parameters, reference)
            assert_probabilities(q)
            qs[:, j] = q
            cross_values[:, j] = q[:, 1] * prior[:, 2] - q[:, 2] * prior[:, 1]
            interior = (q[:, 1] > 0) & (q[:, 2] > 0)
            log_deltas[interior, j] = (np.log(q[interior, 1]) - np.log(q[interior, 2])
                                      - np.log(prior[interior, 1]) + np.log(prior[interior, 2]))
            boundary_counts += ~interior
            prior = advance(q, reference["P"])
        predictions[family] = qs
        odds_diagnostics[family] = {"cross_product": cross_values, "finite_log_odds_change": log_deltas,
                                    "boundary_counts": boundary_counts}
        elapsed = time.monotonic() - family_start
        phase_receipts.append({"phase": "selected_test", "family": family,
                               "streams": n_seeds, "recursive_updates": n_seeds * len(test_origins),
                               "elapsed_seconds": elapsed})
        print(json.dumps({"scenario": scenario["name"], "phase": "selected_test_complete",
                          "family": family, "seconds": round(elapsed, 3)}), flush=True)

    expected, empirical, excess, oracle_exp, oracle_emp = {}, {}, {}, {}, {}
    for h in HORIZONS:
        pair = reference["conditional"][h]["pair"]
        oracle_exp[h] = .5 * np.einsum("...i,ij,...j->...", truth, kernels[h]["true_pair"], truth)
        true_a = oracle_empirical_features(test_y[h], kernels[h], scenario, h)
        oracle_emp[h] = empirical_score(truth, true_a, kernels[h]["true_pair"])
        for family, q in predictions.items():
            exp = expected_score(q, truth, pair, kernels[h]["cross"])
            raw = empirical_score(q, test_a[h], pair)
            gap = exp - oracle_exp[h]
            if float(np.min(gap)) < -1e-10:
                raise AssertionError(f"Negative oracle excess risk {scenario['name']} {family} h{h}: {gap.min()}")
            expected[family, h], empirical[family, h], excess[family, h] = exp, raw, gap
    if scenario["name"] == "clean_correct":
        max_oracle_gap = max(float(np.max(np.abs(excess["nominal_bayes", h]))) for h in HORIZONS)
        if max_oracle_gap > 2e-10:
            raise AssertionError("Correct nominal Bayes must equal true oracle")
    else:
        max_oracle_gap = None

    slow_change = np.zeros_like(states, dtype=bool)
    slow_change[:, 1:] = states[:, 1:] // 2 != states[:, :-1] // 2
    recent_change = np.stack([slow_change[:, max(1, t - 4):t + 1].any(axis=1) for t in test_origins], axis=1)
    rows = []
    for family in list(predictions) + ["true_oracle"]:
        for si, seed in enumerate(seeds):
            row = {"scenario": scenario["name"], "seed": seed, "family": family,
                   "test_origins": len(test_origins),
                   "validation_selected_competitor": selected_competitors[si]}
            for h in HORIZONS:
                ex = oracle_exp[h][si] if family == "true_oracle" else expected[family, h][si]
                raw = oracle_emp[h][si] if family == "true_oracle" else empirical[family, h][si]
                row[f"expected_crps_h{h}"] = float(ex.mean())
                row[f"empirical_crps_h{h}"] = float(raw.mean())
                row[f"excess_crps_h{h}"] = 0.0 if family == "true_oracle" else float(excess[family, h][si].mean())
            row["expected_primary"] = .5 * (row["expected_crps_h5"] + row["expected_crps_h20"])
            row["empirical_primary"] = .5 * (row["empirical_crps_h5"] + row["empirical_crps_h20"])
            row["recent_true_slow_change_origins"] = int(recent_change[si].sum())
            row["recent_change_expected_primary"] = None
            if scenario["name"] == "clean_correct" and recent_change[si].any():
                mask = recent_change[si]
                vec = .5 * (oracle_exp[5][si] + oracle_exp[20][si]) if family == "true_oracle" else .5 * (
                    expected[family, 5][si] + expected[family, 20][si])
                row["recent_change_expected_primary"] = float(vec[mask].mean())
            if family in odds_diagnostics:
                diagnostic = odds_diagnostics[family]
                logs = diagnostic["finite_log_odds_change"][si]
                finite = np.isfinite(logs)
                row["max_abs_odds_cross_product"] = float(np.max(np.abs(diagnostic["cross_product"][si])))
                row["mean_abs_odds_cross_product"] = float(np.mean(np.abs(diagnostic["cross_product"][si])))
                row["odds_boundary_origins"] = int(diagnostic["boundary_counts"][si])
                row["max_abs_log_odds_change_when_finite"] = float(np.max(np.abs(logs[finite]))) if finite.any() else None
            rows.append(row)

    arrays = {"observed_returns": obs, "latent_clean_returns": clean, "latent_states": states,
              "contamination_flags": flags, "test_origins": test_origins,
              "validation_origins": validation_origins, "oracle_posterior": truth,
              "recent_true_slow_change": recent_change, "seeds": np.asarray(seeds)}
    for family, q in predictions.items():
        arrays[f"q__{family}"] = q
        if family in families:
            arrays[f"initial_test_prior__{family}"] = chosen_prior[family]
            arrays[f"selected_parameter__{family}"] = chosen_parameter[family]
            arrays[f"validation_scores__{family}"] = val_scores[family]
        elif family == "nominal_bayes":
            arrays[f"initial_test_prior__{family}"] = nominal_bayes_prior
        for h in HORIZONS:
            arrays[f"expected__{family}__h{h}"] = expected[family, h]
            arrays[f"empirical__{family}__h{h}"] = empirical[family, h]
    for h in HORIZONS:
        arrays[f"oracle_expected__h{h}"] = oracle_exp[h]
        arrays[f"oracle_empirical__h{h}"] = oracle_emp[h]
    trace = out / f"{scenario['name']}.npz"
    np.savez_compressed(trace, **arrays)
    receipt = {"scenario": scenario, "completed": True, "independent_seeds": n_seeds,
               "synthetic_observations": int(obs.size), "validation_filter_trials": n_seeds * n_grid * len(families),
               "selected_family_streams": n_seeds * len(families),
               "extra_nominal_bayes_streams": n_seeds, "fixed_prior_streams": n_seeds,
               "true_oracle_streams": n_seeds, "neural_fits": 0, "neural_training_gradient_optimizer_steps": 0,
               "selection_file_sha256": selection_sha,
               "trace_file": trace.name, "trace_bytes": trace.stat().st_size, "trace_sha256": sha(trace),
               "minimum_oracle_excess_crps": min(float(np.min(x)) for x in excess.values()),
               "clean_nominal_bayes_max_abs_oracle_gap": max_oracle_gap,
               "phases": phase_receipts, "elapsed_seconds": time.monotonic() - started}
    dump(public / scenario["name"] / "EXECUTION_RECEIPT.json", receipt)
    return rows, validation_trials, receipt


def csv_rows(path: Path, rows: list[dict]):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], protocol: dict, receipts: list[dict]):
    indexed = {(r["scenario"], r["seed"], r["family"]): r for r in rows}
    seeds = protocol["simulation"]["seeds"]
    positives = [s["name"] for s in protocol["scenarios"] if s["role"].startswith("Prespecified positive")]
    strata = {s["name"]: [s["name"]] for s in protocol["scenarios"]}
    strata["positive_scenarios_equal_weight"] = positives
    controls = [f for f in protocol["families"] if f != "future_cdf"] + [
        "nominal_bayes", "fixed_prior", "true_oracle", "validation_selected_competitor"]
    rng = np.random.default_rng(20261019)
    draws = rng.integers(0, len(seeds), size=(9999, len(seeds)))
    comparisons = []
    for stratum, scenarios in strata.items():
        for control in controls:
            for metric in ("expected_primary", "empirical_primary"):
                cell_relative = np.empty((len(scenarios), len(seeds)))
                for sj, scenario in enumerate(scenarios):
                    for si, seed in enumerate(seeds):
                        candidate = indexed[scenario, seed, "future_cdf"]
                        name = candidate["validation_selected_competitor"] if control == "validation_selected_competitor" else control
                        denominator = indexed[scenario, seed, name][metric]
                        cell_relative[sj, si] = candidate[metric] / denominator - 1
                seed_relative = cell_relative.mean(axis=0)
                interval = np.quantile(seed_relative[draws].mean(axis=1), [.025, .975])
                comparisons.append({"stratum": stratum, "candidate": "future_cdf", "control": control,
                                    "metric": metric, "scenario_seed_cells": int(cell_relative.size),
                                    "independent_seeds": len(seeds),
                                    "mean_relative_minus_one": float(seed_relative.mean()),
                                    "bootstrap95_lower": float(interval[0]), "bootstrap95_upper": float(interval[1]),
                                    "strict_better_cells": int((cell_relative < 0).sum()),
                                    "exact_tie_cells": int((cell_relative == 0).sum()),
                                    "strict_worse_cells": int((cell_relative > 0).sum()),
                                    "seed_relative_values": seed_relative.tolist()})
    comparison_index = {(c["stratum"], c["control"], c["metric"]): c for c in comparisons}
    pos = "positive_scenarios_equal_weight"
    key = comparison_index[pos, "validation_selected_competitor", "expected_primary"]
    clean = comparison_index["clean_correct", "validation_selected_competitor", "expected_primary"]
    current = {s: comparison_index[s, "current_cdf", "expected_primary"]["mean_relative_minus_one"] for s in positives}
    rotated = comparison_index[pos, "rotated_future_cdf", "expected_primary"]["mean_relative_minus_one"]
    empirical = comparison_index[pos, "validation_selected_competitor", "empirical_primary"]["mean_relative_minus_one"]
    checks = {
        "positive_expected_gain_at_least_one_percent": {"value": key["mean_relative_minus_one"], "rule": "value <= -0.01", "passed": key["mean_relative_minus_one"] <= -.01},
        "positive_expected_seed_interval_upper_negative": {"value": key["bootstrap95_upper"], "rule": "value < 0", "passed": key["bootstrap95_upper"] < 0},
        "positive_each_beats_current_cdf": {"values": current, "rule": "each value < 0", "passed": all(v < 0 for v in current.values())},
        "positive_beats_rotated_metric": {"value": rotated, "rule": "value < 0", "passed": rotated < 0},
        "positive_empirical_gain": {"value": empirical, "rule": "value < 0", "passed": empirical < 0},
        "clean_noninferiority_half_percent": {"value": clean["mean_relative_minus_one"], "rule": "value <= 0.005", "passed": clean["mean_relative_minus_one"] <= .005},
        "numerical_and_causal_checks": {"value": "Required before and during execution; final independent replay remains separately required", "passed": all(r["completed"] for r in receipts)}
    }
    passed = all(c["passed"] for c in checks.values())
    decision = {"decision": "PASS_FINITE_SYNTHETIC_RESOURCE_GATE" if passed else "STOP_FIXED_GEOMETRY_CANDIDATE",
                "all_passed": passed, "checks": checks,
                "failed_checks": [key for key, value in checks.items() if not value["passed"]],
                "publication_ready": False,
                "interpretation": "Finite synthetic method qualification only; no market validity, novelty clearance or universal robustness follows."}
    return comparisons, decision


def run(protocol: dict, out: Path, public: Path):
    out.mkdir(parents=True, exist_ok=False)
    public.mkdir(parents=True, exist_ok=True)
    if (public / "START.json").exists():
        raise FileExistsError("This frozen run has already started; use an explicitly reviewed new run record")
    freeze_path = ROOT / "provenance" / "SOURCE_FREEZE.json"
    freeze = json.loads(freeze_path.read_text())
    for relative, expected_hash in freeze["sha256"].items():
        if sha(ROOT / relative) != expected_hash:
            raise AssertionError(f"Frozen input changed: {relative}")
    for relative in freeze["required_prefit_audits"]:
        audit = json.loads((ROOT / relative).read_text())
        if audit.get("status") != "PASS":
            raise AssertionError(f"Pre-fit audit not passed: {relative}")
        for source, reviewed_hash in audit["source_hashes"].items():
            if source not in freeze["sha256"] or freeze["sha256"][source] != reviewed_hash:
                raise AssertionError(f"Stale pre-fit audit for source: {source}")
    import scipy
    start = {"study": protocol["study"], "started_utc": datetime.now(timezone.utc).isoformat(),
             "protocol_sha256": sha(PROTOCOL), "source_freeze_sha256": sha(freeze_path),
             "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
             "thread_environment": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
             "source_hashes": freeze["sha256"], "synthetic_only": True,
             "no_neural_or_emission_parameter_fitting": True}
    dump(public / "START.json", start)
    started = time.monotonic()
    gf.reset_numerical_counters()
    reset_propagation_accounting()
    reference = gf.build_reference(horizons=HORIZONS)
    u = np.array([1., 2., 4., -7.]) / np.sqrt(70.)
    reflection = np.eye(4) - 2 * np.outer(u, u)
    reference["metrics"]["rotated"] = reflection @ reference["metrics"]["future"] @ reflection
    if not np.allclose(np.linalg.eigvalsh(reference["metrics"]["rotated"]),
                       np.linalg.eigvalsh(reference["metrics"]["future"]), rtol=1e-11, atol=1e-12):
        raise AssertionError("Rotated control must preserve the metric spectrum")
    rows, trials, receipts = [], [], []
    for scenario in protocol["scenarios"]:
        sr, st, receipt = run_scenario(scenario, protocol, reference, out, public)
        rows.extend(sr); trials.extend(st); receipts.append(receipt)
        csv_rows(public / "SELECTED_RESULTS.csv", rows)
        csv_rows(public / "TRIAL_VALIDATION.csv", trials)
        dump(public / "PARTIAL_ACCOUNTING.json", {"completed_scenarios": len(receipts), "receipts": receipts})
    comparisons, decision = aggregate(rows, protocol, receipts)
    assert len(trials) == protocol["simulation"]["evaluated_validation_filter_trials"]
    assert sum(r["selected_family_streams"] for r in receipts) == protocol["simulation"]["selected_family_streams"]
    csv_rows(public / "PAIRED_COMPARISONS.csv", [{k: v for k, v in c.items() if k != "seed_relative_values"} for c in comparisons])
    summary = {"study": protocol["study"], "completed_utc": datetime.now(timezone.utc).isoformat(),
               "decision": decision, "validation_filter_trials": len(trials),
               "selected_family_streams": sum(r["selected_family_streams"] for r in receipts),
               "actual_neural_fits": 0, "actual_emission_or_transition_parameter_fits": 0,
               "neural_training_gradient_optimizer_steps": 0,
               "online_convex_solver_accounting": gf.get_numerical_counters(),
               "common_nominal_markov_propagation_normalization": dict(PROPAGATION_ACCOUNTING),
               "unique_simulated_scenario_seed_sequences": len(receipts) * len(protocol["simulation"]["seeds"]),
               "independent_random_seeds": len(protocol["simulation"]["seeds"]),
               "synthetic_observations": sum(r["synthetic_observations"] for r in receipts),
               "validation_recursive_analysis_updates": sum(p["recursive_updates"] for r in receipts for p in r["phases"] if p["phase"] == "validation"),
               "selected_test_recursive_analysis_updates_including_nominal_bayes": sum(p["recursive_updates"] for r in receipts for p in r["phases"] if p["phase"] == "selected_test"),
               "elapsed_seconds": time.monotonic() - started, "scenario_receipts": receipts,
               "comparisons": comparisons, "no_market_prices_or_confirmation_data_opened": True,
               "limits": ["Known nominal HMM parameters; only filtering hyperparameters are selected.",
                          "Future observed-return targets include shocks; eligible filters use a common nominal future family.",
                          "The moderate-shock scenario also misspecifies the fixed variance-16 contamination-likelihood control.",
                          "Conditional expected scores use the known DGP for evaluation only.",
                          "Metric selection is within established variable-metric proper-score filtering.",
                          "Internal local source freeze, not external preregistration."]}
    for relative, expected_hash in freeze["sha256"].items():
        if sha(ROOT / relative) != expected_hash:
            raise AssertionError(f"Frozen source changed during execution: {relative}")
    dump(public / "SUMMARY.json", summary)
    dump(public / "COMPLETE.json", {"complete": True, "summary_sha256": sha(public / "SUMMARY.json"),
                                   "trace_files": [{k: r[k] for k in ("trace_file", "trace_bytes", "trace_sha256")} for r in receipts],
                                   "source_hashes_unchanged": True})
    print(json.dumps({"complete": True, "decision": decision["decision"],
                      "validation_trials": len(trials), "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--public", type=Path, default=ROOT / "results" / "synthetic")
    args = parser.parse_args()
    run(json.loads(PROTOCOL.read_text()), args.out, args.public)


