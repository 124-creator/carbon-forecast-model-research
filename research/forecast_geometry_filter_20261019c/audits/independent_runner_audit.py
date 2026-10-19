"""Independent pre-fit runner review with static checks and finite fixtures.

Does not import geometry_filter, call run/run_scenario, or read market data.
Only named numerical/bookkeeping helpers are extracted from the runner AST. Their outputs are
checked against independently constructed indexing, enumeration and integration
fixtures. Core filtering/DP/QP correctness belongs to the separate core audit.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp, ndtr
from scipy.stats import binom, norm


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "source" / "run_synthetic.py"
PROTOCOL = ROOT / "provenance" / "SYNTHETIC_PROTOCOL.json"
OUTPUT = ROOT / "audits" / "RUNNER_PREFIT_AUDIT.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_close(actual, expected, atol=1e-11, rtol=1e-11):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def main():
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in (RUNNER, PROTOCOL)}
    source = RUNNER.read_text()
    protocol = json.loads(PROTOCOL.read_text())
    tree = ast.parse(source)
    nodes = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    helpers = ["normal_abs", "transition", "simulate", "oracle_states", "oracle_kernels",
               "target_features", "empirical_score", "expected_score", "assert_probabilities",
               "reset_propagation_accounting", "advance", "oracle_empirical_features",
               "family_grid", "aggregate"]
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("HORIZONS", "PRIMARY", "COMPETITORS", "PROPAGATION_ACCOUNTING"):
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    env = {"np": np, "ndtr": ndtr, "logsumexp": logsumexp, "binom": binom,
           "SQRT2PI": np.sqrt(2 / np.pi), **constants}
    module = ast.Module(body=[nodes[name] for name in helpers], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(RUNNER), "exec"), env)
    checks = []

    def check(name, function):
        try:
            details = function()
            checks.append({"name": name, "status": "PASS", "details": details})
        except Exception as error:
            checks.append({"name": name, "status": "FAIL", "error": f"{type(error).__name__}: {error}"})

    def indexing_and_noninterference():
        sim = protocol["simulation"]
        horizons = constants["HORIZONS"]
        assert list(horizons) == protocol["reference_model"]["horizons"]
        assert list(constants["PRIMARY"]) == protocol["evaluation"]["primary_horizons"]
        va = np.arange(sim["validation_start_index"], sim["validation_last_index"] + 1)
        te = np.arange(sim["test_start_index"], sim["test_last_index"] + 1)
        assert len(va) == sim["validation_origins"] == 512
        assert len(te) == sim["test_origins"] == 1024
        assert va[0] == sim["burn_observations"]
        assert va[-1] + max(horizons) == sim["last_validation_label_index"] == te[0]
        assert te[-1] + max(horizons) == sim["last_test_label_index"]
        assert sim["total_observations_per_scenario_seed"] == sim["last_test_label_index"] + 1
        n = sim["total_observations_per_scenario_seed"]
        obs = np.stack((np.arange(n, dtype=float), 1 - .5 * np.arange(n, dtype=float)))
        ref = {"conditional": {h: {"weights": np.ones((4, 1)), "sds": np.array([1.])} for h in horizons}}
        val_y, val_a = env["target_features"](obs, ref, va)
        test_y, _ = env["target_features"](obs, ref, te)
        for origin_set, ys in ((va, val_y), (te, test_y)):
            for h in horizons:
                expected = np.stack([obs[:, int(t) + 1:int(t) + h + 1].sum(axis=1) / np.sqrt(h) for t in origin_set], axis=1)
                assert_close(ys[h], expected, atol=2e-10)
        changed = obs.copy()
        changed[:, te[0] + 1:] += 1e6
        changed_y, changed_a = env["target_features"](changed, ref, va)
        for h in horizons:
            assert np.array_equal(val_y[h], changed_y[h])
            assert np.array_equal(val_a[h], changed_a[h])
        return {"validation_origins": [int(va[0]), int(va[-1])],
                "validation_label_max": int(va[-1] + max(horizons)),
                "test_origin_min": int(te[0]), "test_future_label_min": int(te[0] + 1),
                "test_label_max": int(te[-1] + max(horizons)),
                "future_suffix_perturbation_leaves_validation_identical": True}

    def source_selection_order():
        run_scenario = nodes["run_scenario"]
        calls = [n for n in ast.walk(run_scenario) if isinstance(n, ast.Call)]
        target_calls = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "target_features"]
        assert len(target_calls) == 2
        for call in target_calls:
            assert isinstance(call.args[0], ast.Name) and call.args[0].id == "obs"
        val_call = next(n for n in target_calls if ast.unparse(n.args[2]) == "validation_origins")
        test_call = next(n for n in target_calls if ast.unparse(n.args[2]) == "test_origins")
        selection_dump = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "dump" and ast.unparse(n.args[0]) == "selection_path")
        oracle_calls = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id in ("oracle_states", "oracle_kernels", "expected_score")]
        assert val_call.lineno < selection_dump.lineno < test_call.lineno
        assert all(n.lineno > selection_dump.lineno for n in oracle_calls)
        argmins = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "argmin"]
        assert len(argmins) == 2 and all(n.lineno < selection_dump.lineno for n in argmins)
        loops = [n for n in ast.walk(run_scenario) if isinstance(n, ast.For)]
        assert any(ast.unparse(n.iter) == "range(sim['test_start_index'])" for n in loops)
        assert any(ast.unparse(n.iter) == "enumerate(test_origins)" for n in loops)
        return {"selection_dump_line": selection_dump.lineno,
                "first_test_target_call_line": test_call.lineno,
                "oracle_and_test_scores_follow_selection": True,
                "interpretation": "y659 is a matured validation label and current information, while test future labels start at660."}

    def selection_and_counting():
        families = protocol["families"]
        competitors = constants["COMPETITORS"]
        assert set(competitors) == set(families) - {"future_cdf", "future_cdf_ratio_preserving"}
        assert len(competitors) == 8 and "true_oracle" not in competitors
        grids = {f: env["family_grid"](f, protocol) for f in families}
        assert all(len(g) == 7 for g in grids.values())
        ns, nc, nf, ng = len(protocol["simulation"]["seeds"]), len(protocol["scenarios"]), len(families), 7
        assert ns == protocol["simulation"]["independent_sequence_seeds"] == 20
        assert len(set(protocol["simulation"]["seeds"])) == ns
        assert nc * ns * nf * ng == protocol["simulation"]["evaluated_validation_filter_trials"] == 5600
        assert nc * ns * nf == protocol["simulation"]["selected_family_streams"] == 800
        indices = [n for n in ast.walk(nodes["run_scenario"]) if isinstance(n, ast.Assign) and len(n.targets) == 1 and ast.unparse(n.targets[0]) == "index"]
        assert len(indices) == 1
        local = {"np": np, "mean_scores": np.array([[3., 1., 1., 2., 5., 6., 7.], [1., 1., 1., 1., 1., 1., 1.]])}
        exec(compile(ast.Module(body=indices, type_ignores=[]), str(RUNNER), "exec"), local)
        assert np.array_equal(local["index"], [1, 0])
        fixture_scores = {f: np.full((2, 7, 3), 10.) for f in families}
        fixture_chosen = {f: np.zeros(2, dtype=int) for f in families}
        fixture_scores["current_cdf"][0, 0] = [1, 1, 1]
        fixture_scores["future_variance"][0, 0] = [1, 1, 1]
        fixture_scores["contaminated_bayes"][1, 0] = [.7, .7, .7]
        fixture_scores["future_cdf"][:] = 0
        fixture_scores["future_cdf_ratio_preserving"][:] = 0
        wanted = {"selected_val", "competitor_val", "selected_competitors"}
        assignments = [n for n in nodes["run_scenario"].body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in wanted]
        assert len(assignments) == 3
        fixture_env = {"np": np, "val_scores": fixture_scores, "chosen_index": fixture_chosen,
                       "n_seeds": 2, "families": families, "COMPETITORS": competitors}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), str(RUNNER), "exec"), fixture_env)
        assert fixture_env["selected_competitors"] == ["current_cdf", "contaminated_bayes"]
        return {"validation_trials": nc * ns * nf * ng, "selected_family_streams": nc * ns * nf,
                "composite_controls": list(competitors), "exact_ties_choose_first": True,
                "candidate_and_ratio_ablation_excluded_from_composite": True}

    def crps_numerical_integrals():
        q, truth = np.array([.1, .2, .3, .4]), np.array([.4, .3, .2, .1])
        wm = np.array([[.7, .2, .1], [.2, .6, .2], [.1, .2, .7], [.3, .3, .4]])
        wt = np.array([[.8, .2], [.6, .4], [.4, .6], [.2, .8]])
        sm, st, y = np.array([.3, .8, 1.6]), np.array([.5, 1.2]), .7
        pair = wm @ (np.sqrt(2 / np.pi) * np.sqrt(sm[:, None] ** 2 + sm[None, :] ** 2)) @ wm.T
        cross = wm @ (np.sqrt(2 / np.pi) * np.sqrt(sm[:, None] ** 2 + st[None, :] ** 2)) @ wt.T
        a = env["normal_abs"](y, sm) @ wm.T
        f = lambda z: float((q @ wm) @ norm.cdf(z / sm))
        g = lambda z: float((truth @ wt) @ norm.cdf(z / st))
        raw_reference = quad(lambda z: f(z) ** 2, -np.inf, y, epsabs=1e-11)[0] + quad(lambda z: (1 - f(z)) ** 2, y, np.inf, epsabs=1e-11)[0]
        expected_reference = quad(lambda z: (f(z) - g(z)) ** 2 + g(z) * (1 - g(z)), -np.inf, np.inf, epsabs=1e-11)[0]
        oracle_reference = quad(lambda z: g(z) * (1 - g(z)), -np.inf, np.inf, epsabs=1e-11)[0]
        raw = float(env["empirical_score"](q, a, pair))
        expected = float(env["expected_score"](q, truth, pair, cross))
        assert_close(raw, raw_reference, atol=2e-10)
        assert_close(expected, expected_reference, atol=2e-10)
        assert expected >= oracle_reference - 1e-10
        return {"empirical_crps_error": abs(raw - raw_reference),
                "conditional_expected_crps_error": abs(expected - expected_reference),
                "expected_crps_excess_is_nonnegative": True}

    def oracle_history_and_common_randomness():
        v = np.asarray(protocol["reference_model"]["emission_variances"])
        p = env["transition"](.8, .2)
        observations = np.array([[.1, -.3, 2.3, .8], [-.6, .4, -.2, .5]])
        epsilon, tau = .17, 1.4
        actual = env["oracle_states"](observations, p, v, epsilon, tau)
        maximum_error = 0.
        for si, series in enumerate(observations):
            for t in range(len(series)):
                masses = np.zeros(4)
                for path in itertools.product(range(4), repeat=t + 1):
                    mass = .25
                    for j, state in enumerate(path):
                        if j:
                            mass *= p[path[j - 1], state]
                        mass *= (1 - epsilon) * norm.pdf(series[j], scale=np.sqrt(v[state])) + epsilon * norm.pdf(series[j], scale=np.sqrt(v[state] + tau * tau))
                    masses[path[-1]] += mass
                expected = masses / masses.sum()
                maximum_error = max(maximum_error, float(np.max(np.abs(actual[si, t] - expected))))
                assert_close(actual[si, t], expected)
        changed = observations.copy()
        changed[:, 3] += 10
        assert_close(env["oracle_states"](changed, p, v, epsilon, tau)[:, :3], actual[:, :3], atol=0, rtol=0)
        scen0, scen1 = protocol["scenarios"][:2]
        out0 = env["simulate"](314, scen0, 32, v)
        out1 = env["simulate"](314, scen1, 32, v)
        assert np.array_equal(out0[0], out1[0]) and np.array_equal(out0[2], out1[2])
        assert_close(env["advance"](np.full((3, 4), .25), p), np.full((3, 4), .25))
        return {"enumerated_history_posterior_max_error": maximum_error,
                "future_observation_does_not_change_past_oracle": True,
                "same_seed_equal_dynamics_share_state_and_clean_draws": True}

    def contamination_kernels():
        wm = np.array([[.8, .2], [.6, .4], [.4, .6], [.2, .8]])
        wt = np.array([[.9, .1], [.7, .3], [.3, .7], [.1, .9]])
        sm, st = np.array([.6, 1.2]), np.array([.7, 1.5])
        reference = {"v": np.array([.16, .64, .64, 2.56]),
                     "conditional": {h: {"weights": wm, "sds": sm} for h in (1, 2, 3)}}
        scenario = {"slow_persistence": .6, "fast_persistence": .1,
                    "contamination_probability": .23, "additive_shock_sd": 1.7}
        old_horizons = env["HORIZONS"]
        env["HORIZONS"] = (1, 2, 3)
        env["gf"] = SimpleNamespace(conditional_mixture=lambda _p, _v, _h: (wt, st))
        try:
            kernels = env["oracle_kernels"](reference, scenario)
            max_error = 0.
            for h in (1, 2, 3):
                epsilon, tau = scenario["contamination_probability"], scenario["additive_shock_sd"]
                cross, true_pair = np.zeros((2, 2)), np.zeros((2, 2))
                for length, destination, left in ((h, cross, sm), (2 * h, true_pair, st)):
                    for flags in itertools.product((0, 1), repeat=length):
                        probability = float(np.prod([epsilon if flag else 1 - epsilon for flag in flags]))
                        destination += probability * np.sqrt(2 / np.pi) * np.sqrt(left[:, None] ** 2 + st[None, :] ** 2 + sum(flags) * tau * tau / h)
                expected_cross, expected_pair = wm @ cross @ wt.T, wt @ true_pair @ wt.T
                max_error = max(max_error, float(np.max(np.abs(kernels[h]["cross"] - expected_cross))), float(np.max(np.abs(kernels[h]["true_pair"] - expected_pair))))
                assert_close(kernels[h]["cross"], expected_cross)
                assert_close(kernels[h]["true_pair"], expected_pair)
                y = np.array([[.2, -1.1], [.6, 2.0]])
                a = np.zeros(y.shape + (4,))
                for flags in itertools.product((0, 1), repeat=h):
                    probability = float(np.prod([epsilon if flag else 1 - epsilon for flag in flags]))
                    sd = np.sqrt(st * st + sum(flags) * tau * tau / h)
                    a += probability * (env["normal_abs"](y[..., None], sd) @ wt.T)
                assert_close(env["oracle_empirical_features"](y, kernels[h], scenario, h), a)
            return {"horizons": [1, 2, 3], "maximum_enumeration_error": max_error,
                    "cross_uses_h_shocks_true_pair_uses_2h_shocks": True,
                    "core_DP_stubbed_with_declared_finite_fixture": True}
        finally:
            env["HORIZONS"] = old_horizons
            del env["gf"]

    def paired_aggregation():
        fixture_rows = []
        seeds = protocol["simulation"]["seeds"]
        all_families = protocol["families"] + ["nominal_bayes", "fixed_prior", "true_oracle"]
        for sj, scenario in enumerate(protocol["scenarios"]):
            for si, seed in enumerate(seeds):
                selected = constants["COMPETITORS"][si % len(constants["COMPETITORS"])]
                for fi, family in enumerate(all_families):
                    base = (1 + si) * (1 + sj) * (1 + fi * .03)
                    if family == "future_cdf":
                        base *= .93 + .001 * si - .01 * sj
                    fixture_rows.append({"scenario": scenario["name"], "seed": seed, "family": family,
                                         "validation_selected_competitor": selected,
                                         "expected_primary": base, "empirical_primary": base * (1 + .002 * (fi - si))})
        comparisons, decision = env["aggregate"](fixture_rows, protocol, [{"completed": True} for _ in protocol["scenarios"]])
        index = {(r["scenario"], r["seed"], r["family"]): r for r in fixture_rows}
        positive_names = [s["name"] for s in protocol["scenarios"] if s["role"].startswith("Prespecified positive")]
        main = next(c for c in comparisons if c["stratum"] == "positive_scenarios_equal_weight" and c["control"] == "validation_selected_competitor" and c["metric"] == "expected_primary")
        manual = []
        for seed in seeds:
            values = []
            for scenario in positive_names:
                row = index[scenario, seed, "future_cdf"]
                control = index[scenario, seed, row["validation_selected_competitor"]]
                values.append(row["expected_primary"] / control["expected_primary"] - 1)
            manual.append(sum(values) / len(values))
        draws = np.random.default_rng(20261019).integers(0, len(seeds), size=(9999, len(seeds)))
        bootstrap_means = np.array([sum(manual[int(i)] for i in sample) / len(seeds) for sample in draws])
        interval = np.quantile(bootstrap_means, [.025, .975])
        assert_close(main["seed_relative_values"], manual)
        assert_close(main["mean_relative_minus_one"], sum(manual) / len(manual))
        assert_close([main["bootstrap95_lower"], main["bootstrap95_upper"]], interval)
        assert main["scenario_seed_cells"] == 40 and main["independent_seeds"] == 20
        assert main["strict_better_cells"] + main["exact_tie_cells"] + main["strict_worse_cells"] == 40
        assert decision["publication_ready"] is False
        assert len(comparisons) == 5 * (len(all_families)) * 2
        return {"bootstrap_draws": len(draws), "bound_scenarios_per_seed": 2,
                "mean_of_within_cell_ratios_verified": True,
                "composite_control_uses_recorded_validation_choice": True,
                "comparison_rows": len(comparisons)}

    def propagation_normalization_accounting():
        bookkeeping = env["PROPAGATION_ACCOUNTING"]
        bookkeeping.clear()
        matrix = env["transition"](.97, .1)
        vector = np.array([.07, .13, .31, .49])
        batch = np.array([[.1, .2, .3, .4], [.25, .25, .25, .25]])
        tensor = np.full((2, 2, 4), .25)
        tensor[0, 0, 0] += 5e-12
        tensor[0, 1, 3] -= 2e-12
        fixtures = [(vector, matrix), (batch, matrix), (tensor, np.eye(4))]
        raw_rows, returned_rows = [], []
        for q, transition_matrix in fixtures:
            prior = q @ transition_matrix
            # This is the complete previous numerical output before bookkeeping.
            previous_output = prior / prior.sum(axis=-1, keepdims=True)
            actual = env["advance"](q, transition_matrix)
            assert np.array_equal(actual, previous_output)
            raw_rows.extend(prior.reshape(-1, 4))
            returned_rows.extend(actual.reshape(-1, 4))
        expected = {"calls": 3, "rows": 7,
                    "rows_with_nonunit_sum_before_normalization": 0,
                    "rows_with_any_coordinate_changed": 0,
                    "maximum_absolute_mass_error": 0.,
                    "maximum_absolute_coordinate_change": 0.}
        for raw_row, returned_row in zip(raw_rows, returned_rows):
            mass = float(np.sum(raw_row))
            changed = any(float(before) != float(after) for before, after in zip(raw_row, returned_row))
            expected["rows_with_nonunit_sum_before_normalization"] += int(mass != 1.)
            expected["rows_with_any_coordinate_changed"] += int(changed)
            expected["maximum_absolute_mass_error"] = max(expected["maximum_absolute_mass_error"], abs(mass - 1.))
            for before, after in zip(raw_row, returned_row):
                expected["maximum_absolute_coordinate_change"] = max(expected["maximum_absolute_coordinate_change"], abs(float(after) - float(before)))
        assert expected["rows_with_nonunit_sum_before_normalization"] >= 2
        assert expected["rows_with_any_coordinate_changed"] >= 2
        assert bookkeeping == expected
        recorded = dict(bookkeeping)
        env["reset_propagation_accounting"]()
        assert all(value == 0 for value in bookkeeping.values())
        return {"fixture_shapes": [list(q.shape) for q, _ in fixtures],
                "outputs_bitwise_identical_to_previous_normalization": True,
                "independent_rowwise_counts": recorded,
                "lazy_initialization_and_explicit_reset_verified": True}

    def source_freeze_and_accounting():
        run_text = ast.unparse(nodes["run"])
        scenario_text = ast.unparse(nodes["run_scenario"])
        assert "actual_gradient_optimizer_steps" not in source
        assert "'gradient_optimizer_steps'" not in run_text + scenario_text
        assert "neural_training_gradient_optimizer_steps" in source
        assert "get_numerical_counters" in run_text and "reset_numerical_counters" in run_text
        assert "reset_propagation_accounting()" in run_text
        assert "common_nominal_markov_propagation_normalization" in run_text
        assert "source_hashes" in run_text and "required_prefit_audits" in run_text
        assert "SOURCE_FREEZE.json" in run_text and "Pre-fit audit" in source
        assert "initial_test_prior" in source
        assert "moderate-shock scenario" in source
        assert "geometry_filter" not in sys.modules
        for relative, value in hashes.items():
            assert digest(ROOT / relative) == value, f"Source changed during audit: {relative}"
        return {"neural_optimizer_counts_distinguished_from_online_solver_counts": True,
                "source_and_prefit_audit_hash_binding_present": True,
                "common_markov_normalization_separately_recorded": True,
                "initial_test_priors_retained_for_replay": True,
                "core_module_not_imported": True}

    check("validation_test_indices_and_label_noninterference", indexing_and_noninterference)
    check("selection_precedes_oracle_and_test_scoring_static", source_selection_order)
    check("hyperparameter_ties_composite_selection_and_trial_counts", selection_and_counting)
    check("empirical_and_expected_CRPS_vs_CDF_integrals", crps_numerical_integrals)
    check("oracle_history_causality_and_common_randomness", oracle_history_and_common_randomness)
    check("future_contamination_count_kernels_vs_enumeration", contamination_kernels)
    check("ratio_aggregation_and_paired_seed_bootstrap", paired_aggregation)
    check("common_markov_propagation_output_and_accounting", propagation_normalization_accounting)
    check("source_freeze_and_numerical_solver_accounting", source_freeze_and_accounting)
    result = {"status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
              "audit": "independent_runner_prefit",
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "source_hashes": hashes,
              "audit_script_sha256": digest(Path(__file__)),
              "protocol_version": protocol["version"],
              "scope": "Read-only runner/protocol review and independent finite fixtures; no formal run or market data.",
              "core_imported": "geometry_filter" in sys.modules,
              "formal_run_called": False, "run_scenario_called": False,
              "checks": checks,
              "failed_checks": [c["name"] for c in checks if c["status"] != "PASS"],
              "manual_review_limits": [
                  "Core DP, QP, KL convergence and filtering math require the separate independent core audit.",
                  "AST order and suffix perturbation establish the reviewed implementation's intended selection boundary, not a general static information-flow theorem.",
                  "Validation weights h1/h5/h20 differ from primary test weights h5/h20 by frozen design.",
                  "The moderate-shock control has a misspecified fixed contamination scale; this is explicitly disclosed.",
                  "Twenty simulation seeds, with paired scenarios, are the independent units; origins and horizons are not counted as independent replications."
              ]}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "checks": len(checks), "failed_checks": result["failed_checks"], "output": str(OUTPUT)}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
