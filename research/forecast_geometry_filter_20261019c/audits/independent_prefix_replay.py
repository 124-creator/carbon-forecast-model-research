"""Post-freeze validation-prefix audit, designed during formal execution.

Replays the frozen production state updates and production arithmetic for
exact tie resolution. Separately scores these replayed states using Fourier
reward laws, erf absolute moments, and direct future slices from the independent
trace audit. This is partial implementation reuse, not an independent optimizer.
No formal run/run_scenario or test state update is called.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve()
DESIGN = ROOT / "audits" / "PREFIX_REPLAY_DESIGN_FREEZE.json"
DEPENDENCIES = [
    "source/geometry_filter.py", "source/run_synthetic.py",
    "provenance/SYNTHETIC_PROTOCOL.json", "audits/independent_trace_replay.py",
    "audits/independent_prefix_replay.py",
]
TOLERANCES = {"independent_score": 2e-9, "exact_replay_score": 0.,
              "incoming_prior": 0., "simulated_observation": 0.,
              "parameter_and_selection": 0., "independent_selection_gap": 2e-9}
COUNTS, MAXIMA = {}, {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")


def check(name, actual, expected, tolerance):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    error = float(np.max(np.abs(actual - expected), initial=0.))
    assert np.isfinite(error) and error <= tolerance, (name, error, tolerance)
    MAXIMA[name] = max(MAXIMA.get(name, 0.), error)
    COUNTS[name] = COUNTS.get(name, 0) + actual.size


def source_hashes():
    return {relative: digest(ROOT / relative) for relative in DEPENDENCIES}


def seal():
    if DESIGN.exists():
        raise FileExistsError("The pre-read design record is immutable")
    value = {
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "audit": "Validation-prefix deterministic replay and independent distribution scoring",
        "source_hashes": source_hashes(), "tolerances": TOLERANCES,
        "design_started_during_formal_execution": True,
        "formal_result_files_opened_before_design_seal": False,
        "disclosure": "The parent relayed COMPLETE and the aggregate STOP decision while this audit script was being finished. No NPZ, score value, selected parameter or final result file was opened before this seal; tolerances and coverage were fixed in the preceding audit design and are unrelated to the relayed decision.",
        "coverage": "All four scenarios, all 20 fixed seeds, all 10 families, all seven grid entries; t=0..658 updates, t=128..639 validation origins, labels up to659.",
        "reuse": "Frozen production core, apply_update, advance and original empirical-score arithmetic are replayed; none is described as a separately implemented optimizer.",
        "independent_parts": "Fixed-seed observations reconstructed separately; Fourier polynomial laws, erf absolute moments and direct future-return sums score every validation configuration.",
        "exact_ties": "Use the replayed frozen arithmetic to verify exact argmin choices. Independent Fourier arithmetic validates losses and optimality within its fixed score tolerance; any different raw argmin is reported, not silently substituted.",
        "no_formal_run_invocation": True, "no_test_state_updates": True,
        "scientific_protocol_changed": False,
    }
    write(DESIGN, value)
    print(json.dumps({"design_sealed": str(DESIGN), "source_hashes": value["source_hashes"]}))


def reconstruct_prefix(protocol, scenario, transition):
    sim = protocol["simulation"]
    n, last = sim["total_observations_per_scenario_seed"], sim["last_validation_label_index"]
    ptrue = transition(scenario["slow_persistence"], scenario["fast_persistence"])
    all_obs = []
    for seed in sim["seeds"]:
        rng = np.random.default_rng(seed)
        uniforms = rng.random(n)
        innovations = rng.standard_normal(n)
        flags = rng.random(n)
        shocks = rng.standard_normal(n)
        states = np.empty(last + 1, dtype=int)
        states[0] = int(uniforms[0] * 4)
        for t in range(1, last + 1):
            states[t] = int(np.count_nonzero(uniforms[t] >= np.cumsum(ptrue[states[t - 1]])))
        variances = np.array([.16, .64, .64, 2.56])
        clean = np.sqrt(variances[states]) * innovations[:last + 1]
        observed = clean + (flags[:last + 1] < scenario["contamination_probability"]) * scenario["additive_shock_sd"] * shocks[:last + 1]
        all_obs.append(observed)
    return np.asarray(all_obs)


def scenario_replay(scenario, protocol, trace_dir, public, production, independent, ref, independent_laws, independent_pairs, trials):
    name = scenario["name"]
    sim = protocol["simulation"]
    seeds, families = sim["seeds"], protocol["families"]
    ns, ng = len(seeds), len(protocol["grids"]["eta"])
    path = trace_dir / f"{name}.npz"
    receipt = read(public / name / "EXECUTION_RECEIPT.json")
    assert digest(path) == receipt["trace_sha256"]
    selection_path = public / name / "SELECTION_BEFORE_TEST.json"
    assert digest(selection_path) == receipt["selection_file_sha256"]
    selection = read(selection_path)
    assert selection["no_test_metric_computed"] is True
    assert selection["selection_observation_index"] == selection["last_validation_label_index"] == 659
    z = np.load(path, allow_pickle=False)
    observations = reconstruct_prefix(protocol, scenario, independent.transition)
    check("fixed_seed_prefix_observations", observations,
          z["observed_returns"][:, :observations.shape[1]], TOLERANCES["simulated_observation"])
    origins = np.arange(sim["validation_start_index"], sim["validation_last_index"] + 1)
    check("validation_origins", origins, z["validation_origins"], 0.)
    _, original_features = production.target_features(observations, ref, origins)
    independent_features = {}
    for h in (1, 5, 20):
        target = np.stack([observations[:, t + 1:t + h + 1].sum(axis=1)
                           for t in origins], axis=1) / np.sqrt(h)
        w, variance = independent_laws[h]
        independent_features[h] = independent.abs_moment(target[:, :, None], variance) @ w.T
    selected_values, independent_values = {}, {}
    family_receipts = []
    selected_checks = 0
    raw_independent_argmin_differences = []
    for family in families:
        started = time.monotonic()
        grid = production.family_grid(family, protocol)
        assert len(grid) == ng == 7
        parameters = np.broadcast_to(grid, (ns, ng)).reshape(-1)
        prior = np.full((ns * ng, 4), .25)
        total_original = np.zeros((ns, ng, 3))
        total_independent = np.zeros_like(total_original)
        for t in range(sim["test_start_index"]):
            q = production.apply_update(family, prior, np.repeat(observations[:, t], ng), parameters, ref)
            production.assert_probabilities(q)
            if sim["validation_start_index"] <= t <= sim["validation_last_index"]:
                j = t - sim["validation_start_index"]
                grouped = q.reshape(ns, ng, 4)
                for hi, h in enumerate((1, 5, 20)):
                    total_original[:, :, hi] += production.empirical_score(
                        grouped, original_features[h][:, j, None, :], ref["conditional"][h]["pair"])
                    independent_score = np.sum(grouped * independent_features[h][:, j, None, :], axis=-1)
                    independent_score -= .5 * np.sum((grouped @ independent_pairs[h]) * grouped, axis=-1)
                    total_independent[:, :, hi] += independent_score
            prior = production.advance(q, ref["P"])
        original_scores = total_original / len(origins)
        independently_scored = total_independent / len(origins)
        saved_scores = z[f"validation_scores__{family}"]
        check("exact_validation_score_replay", original_scores, saved_scores, TOLERANCES["exact_replay_score"])
        check("independent_fourier_validation_scores", independently_scored, saved_scores, TOLERANCES["independent_score"])
        original_means = original_scores.mean(axis=-1)
        independent_means = independently_scored.mean(axis=-1)
        selected = np.argmin(original_means, axis=1)
        selected_parameters = grid[selected]
        selected_priors = prior.reshape(ns, ng, 4)[np.arange(ns), selected]
        check("selected_incoming_test_priors", selected_priors,
              z[f"initial_test_prior__{family}"], TOLERANCES["incoming_prior"])
        check("selected_parameters", selected_parameters, z[f"selected_parameter__{family}"], 0.)
        selected_values[family] = original_means[np.arange(ns), selected]
        independent_values[family] = independent_means[np.arange(ns), selected]
        for si, seed in enumerate(seeds):
            entry = selection["seeds"][si]
            assert entry["seed"] == seed
            chosen = entry["families"][family]
            assert chosen["grid_index"] == int(selected[si])
            assert chosen["parameter"] == float(selected_parameters[si])
            check("selection_validation_means", np.array(chosen["validation_crps"]),
                  np.array(selected_values[family][si]), 0.)
            independently_selected = int(np.argmin(independent_means[si]))
            gap = float(independent_means[si, selected[si]] - independent_means[si, independently_selected])
            assert gap <= TOLERANCES["independent_selection_gap"]
            if independently_selected != selected[si]:
                raw_independent_argmin_differences.append({
                    "scenario": name, "seed": seed, "family": family,
                    "frozen_arithmetic_grid_index": int(selected[si]),
                    "independent_arithmetic_grid_index": independently_selected,
                    "independent_loss_gap": gap})
            for gi, parameter in enumerate(grid):
                row = trials[name, seed, family, gi]
                assert float(row["parameter"]) == float(parameter)
                assert (row["selected_within_family"] == "True") == (gi == selected[si])
                check("public_trial_horizon_scores", original_scores[si, gi],
                      np.array([float(row[f"validation_crps_h{h}"]) for h in (1, 5, 20)]), 0.)
                check("public_trial_mean_scores", np.array(original_means[si, gi]),
                      np.array(float(row["validation_crps_mean"])), 0.)
            selected_checks += 1
        if family == "tempered_bayes":
            beta_one = int(np.flatnonzero(grid == 1.)[0])
            bayes_prior = prior.reshape(ns, ng, 4)[:, beta_one]
            check("nominal_bayes_incoming_test_prior", bayes_prior,
                  z["initial_test_prior__nominal_bayes"], 0.)
        family_receipts.append({
            "family": family, "configuration_streams": ns * ng,
            "recursive_updates": ns * ng * sim["test_start_index"],
            "validation_forecast_scores": ns * ng * len(origins) * 3,
            "selected_family_streams": ns,
            "maximum_independent_score_difference": float(np.max(np.abs(independently_scored - saved_scores))),
            "exact_production_arithmetic_replay": bool(np.array_equal(original_scores, saved_scores)),
            "elapsed_seconds": time.monotonic() - started,
        })
    composite = np.stack([selected_values[f] for f in production.COMPETITORS], axis=1)
    composite_indices = np.argmin(composite, axis=1)
    for si, entry in enumerate(selection["seeds"]):
        name_selected = production.COMPETITORS[int(composite_indices[si])]
        assert entry["validation_selected_competitor"] == name_selected
        COUNTS["composite_competitor_choices"] = COUNTS.get("composite_competitor_choices", 0) + 1
    assert selected_checks == ns * len(families)
    z.close()
    print(json.dumps({"prefix_scenario_passed": name, "configuration_streams": ns * ng * len(families)}), flush=True)
    return {"scenario": name, "status": "PASS", "trace_sha256": digest(path),
            "selection_sha256": digest(selection_path),
            "configuration_streams": ns * ng * len(families),
            "selected_family_checks": selected_checks,
            "family_receipts": family_receipts,
            "independent_raw_argmin_disagreements_within_tolerance": raw_independent_argmin_differences}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--traces", type=Path)
    parser.add_argument("--public", type=Path, default=ROOT / "results" / "synthetic")
    parser.add_argument("--out", type=Path, default=ROOT / "audits" / "INDEPENDENT_PREFIX_REPLAY.json")
    args = parser.parse_args()
    if args.seal:
        seal()
        return
    assert args.traces is not None
    sealed = read(DESIGN)
    hashes = source_hashes()
    assert hashes == sealed["source_hashes"] and TOLERANCES == sealed["tolerances"]
    protocol = read(ROOT / "provenance/SYNTHETIC_PROTOCOL.json")
    assert (args.public / "COMPLETE.json").exists()
    for scenario in protocol["scenarios"]:
        assert (args.traces / f"{scenario['name']}.npz").exists()
    completion = read(args.public / "COMPLETE.json")
    assert completion["complete"] is True and completion["source_hashes_unchanged"] is True
    assert digest(args.public / "SUMMARY.json") == completion["summary_sha256"]
    started = time.monotonic()
    sys.path.insert(0, str(ROOT / "source"))
    import geometry_filter as gf
    import run_synthetic as production
    spec = importlib.util.spec_from_file_location("independent_score_for_prefix", ROOT / "audits/independent_trace_replay.py")
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)
    _, laws, pairs, _, _ = independent.reference()
    ref = gf.build_reference(horizons=(1, 5, 20))
    u = np.array([1., 2., 4., -7.]) / np.sqrt(70.)
    reflection = np.eye(4) - 2 * np.outer(u, u)
    ref["metrics"]["rotated"] = reflection @ ref["metrics"]["future"] @ reflection
    gf.reset_numerical_counters()
    production.reset_propagation_accounting()
    rows = list(csv.DictReader((args.public / "TRIAL_VALIDATION.csv").open()))
    trials = {(row["scenario"], int(row["seed"]), row["family"], int(row["grid_index"])): row for row in rows}
    assert len(rows) == len(trials) == 5600
    scenario_results = [scenario_replay(scenario, protocol, args.traces, args.public,
                                       production, independent, ref, laws, pairs, trials)
                        for scenario in protocol["scenarios"]]
    assert sum(row["configuration_streams"] for row in scenario_results) == 5600
    assert sum(row["selected_family_checks"] for row in scenario_results) == 800
    assert COUNTS["composite_competitor_choices"] == 80
    assert source_hashes() == hashes
    result = {
        "status": "PASS", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "audit": "Complete validation-prefix replay with independent distribution scoring",
        "source_hashes": hashes, "design_freeze_sha256": digest(DESIGN),
        "formal_complete_sha256": digest(args.public / "COMPLETE.json"),
        "formal_source_commit": read(args.public / "START.json")["git_commit"],
        "tolerances": TOLERANCES, "coordinate_counts": COUNTS, "maximum_errors": MAXIMA,
        "scenarios": scenario_results, "validation_configuration_streams_replayed": 5600,
        "recursive_updates_replayed": 3690400, "selected_family_choices_checked": 800,
        "composite_choices_checked": 80,
        "production_solver_counters_during_replay": gf.get_numerical_counters(),
        "independent_fourier_diagnostics": independent.MAXIMA,
        "production_update_implementation_reused": True,
        "production_score_arithmetic_reused_for_exact_selection": True,
        "independent_score_method": "Fourier matrix-polynomial reward laws, direct future-return slices, erf absolute moments, independently combined mixture CRPS",
        "benchmark_run_or_run_scenario_called": False, "test_state_updates_called": False,
        "market_data_read": False, "elapsed_seconds": time.monotonic() - started,
        "limits": [
            "Production state updates are deterministically re-executed, not solved using another optimizer; the separate core and test-KKT audits address mathematical correctness.",
            "Original scoring arithmetic is additionally reused to reproduce strict frozen argmin/tie choices exactly; independent Fourier scoring validates every loss within the fixed threshold.",
            "This audit was designed during formal execution and sealed before opening result files; the parent relayed only the aggregate STOP decision before the script was finished. No score-adaptive changes or tolerance changes were made.",
            "These are audit replays, not additional fitted models, new simulation seeds, new test streams or new scientific confirmation data.",
        ]}
    write(args.out, result)
    print(json.dumps({"status": "PASS", "output": str(args.out),
                      "validation_configurations": 5600, "maximum_errors": MAXIMA,
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
