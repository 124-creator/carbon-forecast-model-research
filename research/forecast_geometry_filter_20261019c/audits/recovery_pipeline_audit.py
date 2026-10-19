"""Independent recovery audit: provenance, solver failure fixture, and gates.

Uses only archived synthetic numerical inputs and artificial score tables.
It does not call the formal runner, simulate a benchmark, or open market data.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import root
from scipy.special import logsumexp, ndtr


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "attempts" / "01_kl_dual_stagnation"
sys.path.insert(0, str(ROOT / "source"))
import geometry_filter as gf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def provenance_check():
    archived_freeze = read(ARCHIVE / "provenance" / "SOURCE_FREEZE.json")
    archived_hashes = {relative: sha(ARCHIVE / relative)
                       for relative in archived_freeze["sha256"]}
    assert archived_hashes == archived_freeze["sha256"]
    unchanged = ["source/run_synthetic.py", "provenance/SYNTHETIC_PROTOCOL.json",
                 "methods/GEOMETRY_DERIVATION_ZH.md",
                 "audits/independent_core_audit.py", "audits/independent_runner_audit.py",
                 "audits/INDEPENDENT_CORE_AUDIT.json", "audits/RUNNER_PREFIT_AUDIT.json",
                 "literature/DIRECT_NEIGHBORS_ZH.md", "literature/PRIOR_MECHANISM_MAP_ZH.md"]
    for relative in unchanged:
        assert sha(ROOT / relative) == archived_hashes[relative], relative
    old_tree = ast.parse((ARCHIVE / "source/geometry_filter.py").read_text())
    new_tree = ast.parse((ROOT / "source/geometry_filter.py").read_text())
    def definitions(tree):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    old_def, new_def = definitions(old_tree), definitions(new_tree)
    assert set(old_def) == set(new_def)
    changed = [name for name in old_def if old_def[name] != new_def[name]]
    assert changed == ["update_crps_kl"]
    old_other = [ast.dump(node, include_attributes=False) for node in old_tree.body
                 if not isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    new_other = [ast.dump(node, include_attributes=False) for node in new_tree.body
                 if not isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    assert old_other == new_other
    sig = inspect.signature(gf.update_crps_kl)
    assert sig.parameters["tolerance"].default == 1e-12
    assert sig.parameters["max_iter"].default == 100
    failure = read(ARCHIVE / "FAILURE_RECEIPT.json")
    log = (ARCHIVE / "EXECUTION_FAILURE.log").read_text()
    passes = [json.loads(line) for line in log.splitlines() if line.startswith("{")]
    assert len(passes) == failure["completed_validation_family_passes"] == 6
    assert all(row["phase"] == "validation_complete" for row in passes)
    assert failure["test_streams"] == failure["completed_scenarios"] == 0
    assert failure["completed_configuration_validation_streams"] == 6 * 20 * 7 == 840
    assert not list((ARCHIVE / "results").rglob("SELECTION_BEFORE_TEST.json"))
    return {"status": "PASS", "archived_freeze_sha256": sha(ARCHIVE / "provenance/SOURCE_FREEZE.json"),
            "archived_frozen_files_verified": len(archived_hashes),
            "unchanged_current_scientific_inputs": unchanged,
            "only_changed_function": changed[0],
            "module_constants_and_other_definitions_unchanged": True,
            "solver_tolerance": 1e-12, "newton_iteration_budget": 100,
            "failed_attempt_completed_validation_configuration_streams": 840,
            "failed_attempt_completed_scenarios": 0,
            "failed_attempt_test_streams": 0,
            "no_archived_selection_or_test_score_file": True}


def solver_failure_fixture():
    diagnostic_path = ROOT / "audits" / "kl_roundoff_diagnostic.json"
    diagnostic = read(diagnostic_path)
    fixture = diagnostic["batch_input_for_exact_reproduction"]
    p = np.asarray(fixture["p"], dtype=float)
    y = np.asarray(fixture["y"], dtype=float)
    eta_key = "eta" if "eta" in fixture else "parameter"
    eta = np.asarray(fixture[eta_key], dtype=float)
    assert len(p) == 140
    old_spec = importlib.util.spec_from_file_location(
        "archived_geometry_core_for_recovery", ARCHIVE / "source/geometry_filter.py")
    old = importlib.util.module_from_spec(old_spec)
    old_spec.loader.exec_module(old)
    old_exception = None
    try:
        old.update_crps_kl(p, y, eta, return_receipt=True)
    except FloatingPointError as exc:
        old_exception = str(exc)
    gf.reset_numerical_counters()
    q, numerical = gf.update_crps_kl(p, y, eta, return_receipt=True)
    assert numerical["tolerance"] == 1e-12 and numerical["max_iter_budget"] == 100
    assert numerical["max_dual_gradient_residual"] <= 1e-12
    assert numerical["max_iterations"] <= 100
    assert np.array_equal(q[eta == 0], p[eta == 0])
    v = np.array([.16, .64, .64, 2.56])
    sd = np.sqrt(v)
    pair = np.sqrt(2 / np.pi) * np.sqrt(v[:, None] + v[None, :])
    maximum_probability_error = 0.
    maximum_objective_error = 0.
    maximum_kkt = 0.
    maximum_root_residual = 0.
    root_cases = 0
    failed_row_checks = []
    for row in np.flatnonzero(eta > 0):
        pp, yy, ee, actual = p[row], y[row], eta[row], q[row]
        z = yy / sd
        absolute = sd * np.sqrt(2 / np.pi) * np.exp(-z * z / 2) + yy * (2 * ndtr(z) - 1)
        prior_logits = np.log(pp[:3]) - np.log(pp[3])
        row_difference = pair[:3] - pair[3]
        def probability(x):
            z = np.r_[x, 0.]
            return np.exp(z - logsumexp(z))
        def equation(x):
            return x - prior_logits + ee * (absolute[:3] - absolute[3]
                                             - row_difference @ probability(x))
        def jacobian(x):
            qq = probability(x)
            return np.eye(3) - ee * row_difference @ (np.diag(qq) - np.outer(qq, qq))[:, :3]
        solved = root(equation, prior_logits, jac=jacobian,
                      method="hybr", options={"xtol": 1e-11, "maxfev": 1000})
        independent = probability(solved.x)
        residual = float(np.max(np.abs(equation(solved.x))))
        assert residual <= 2e-9
        gradient = absolute - pair @ actual + (np.log(actual) - np.log(pp)) / ee
        kkt = float(np.max(np.abs(gradient - gradient.mean())))
        def objective(xx):
            return float(xx @ absolute - .5 * xx @ pair @ xx
                         + np.sum(xx * (np.log(xx) - np.log(pp))) / ee)
        probability_error = float(np.max(np.abs(actual - independent)))
        objective_error = abs(objective(actual) - objective(independent))
        maximum_probability_error = max(maximum_probability_error, probability_error)
        maximum_objective_error = max(maximum_objective_error, objective_error)
        maximum_kkt = max(maximum_kkt, kkt)
        maximum_root_residual = max(maximum_root_residual, residual)
        root_cases += 1
        if int(row) in [item["batch_row"] for item in diagnostic["failing_rows"]]:
            failed_row_checks.append({"row": int(row), "original_primal_kkt": kkt,
                                      "independent_probability_error": probability_error,
                                      "independent_objective_error": objective_error})
    assert maximum_probability_error <= 2e-8
    assert maximum_objective_error <= 2e-9
    assert maximum_kkt <= 2e-8
    return {"status": "PASS", "diagnostic_sha256": sha(diagnostic_path),
            "fixture_rows": len(p), "zero_step_rows": int((eta == 0).sum()),
            "independent_primal_root_cases": root_cases,
            "original_failure_reproduced": old_exception is not None,
            "original_exception": old_exception,
            "new_core_numerical_receipt": numerical,
            "independent_equations": "Three original primal log-odds KKT equations, no core dual factors",
            "maximum_probability_error": maximum_probability_error,
            "maximum_original_objective_error": maximum_objective_error,
            "maximum_original_primal_kkt": maximum_kkt,
            "maximum_independent_root_residual": maximum_root_residual,
            "original_failed_rows": failed_row_checks,
            "formal_validation_scores_computed": 0,
            "formal_test_streams": 0}


def gate_check(protocol):
    tree = ast.parse((ROOT / "source/run_synthetic.py").read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "aggregate")
    env = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "aggregate_fixture", "exec"), env)
    aggregate = env["aggregate"]
    family_names = protocol["families"] + ["nominal_bayes", "fixed_prior", "true_oracle"]
    positive_names = [s["name"] for s in protocol["scenarios"]
                      if s["role"].startswith("Prespecified positive")]
    def build(case):
        rows = []
        for scenario in protocol["scenarios"]:
            positive = scenario["name"] in positive_names
            clean = scenario["name"] == "clean_correct"
            for si, seed in enumerate(protocol["simulation"]["seeds"]):
                for family in family_names:
                    expected, empirical = 2., 2.
                    if family == "future_cdf":
                        expected = 1.96 if positive else 2.001 if clean else 2.
                        empirical = expected
                        if case == "gain" and positive:
                            expected = empirical = 1.99
                        if case == "interval" and positive:
                            expected = 2 * (.68 if si < 10 else 1.28)
                        if case == "empirical" and positive:
                            empirical = 2.01
                        if case == "clean" and clean:
                            expected = 2.012
                    if case == "current" and family == "current_cdf" and scenario["name"] == positive_names[0]:
                        expected = 1.96
                    if case == "rotated" and family == "rotated_future_cdf" and positive:
                        expected = 1.96
                    rows.append({"scenario": scenario["name"], "seed": seed, "family": family,
                                 "validation_selected_competitor": "tempered_bayes",
                                 "expected_primary": expected,
                                 "empirical_primary": empirical})
        receipts = [{"completed": case != "numerical"} for _ in protocol["scenarios"]]
        return aggregate(rows, protocol, receipts)
    _, baseline = build("pass")
    assert baseline["all_passed"]
    failures = {
        "gain": "positive_expected_gain_at_least_one_percent",
        "interval": "positive_expected_seed_interval_upper_negative",
        "current": "positive_each_beats_current_cdf",
        "rotated": "positive_beats_rotated_metric",
        "empirical": "positive_empirical_gain",
        "clean": "clean_noninferiority_half_percent",
        "numerical": "numerical_and_causal_checks",
    }
    results = []
    for fixture, target in failures.items():
        _, decision = build(fixture)
        assert not decision["checks"][target]["passed"], fixture
        assert decision["decision"] == "STOP_FIXED_GEOMETRY_CANDIDATE"
        assert not decision["publication_ready"]
        results.append({"fixture": fixture, "required_failure": target,
                        "observed_failed_checks": decision["failed_checks"]})
    sim = protocol["simulation"]
    assert sim["validation_last_index"] + 20 == sim["last_validation_label_index"] == sim["test_start_index"] == 659
    assert sim["test_last_index"] + 20 == sim["last_test_label_index"] == 1702
    assert sim["total_observations_per_scenario_seed"] == 1703
    return {"status": "PASS", "fixture_types": "Artificial score tables only",
            "passing_fixture": True, "failing_gate_fixtures": results,
            "strict_equality_rejected_for_current_and_rotated_controls": True,
            "formal_runner_called": False, "market_data_reads": 0,
            "maturity": {"last_validation_label": 659, "first_test_origin_after_observation": 659,
                         "first_test_future_label": 660, "last_test_label": 1702},
            "declared_full_run_counts_not_completed_by_this_audit": {
                "validation_configuration_streams": 4 * 20 * 10 * 7,
                "selected_family_streams": 4 * 20 * 10,
                "selected_test_streams_including_nominal_bayes": 4 * 20 * 11,
                "unique_scenario_seed_sequences": 80, "independent_seeds": 20,
                "synthetic_observations": 4 * 20 * 1703,
                "validation_recursive_updates": 4 * 20 * 10 * 7 * 659,
                "selected_test_recursive_updates_including_nominal_bayes": 4 * 20 * 11 * 1024,
                "neural_fits": 0, "emission_or_transition_parameter_fits": 0}}


def main():
    protocol = read(ROOT / "provenance/SYNTHETIC_PROTOCOL.json")
    bound = ["source/geometry_filter.py", "source/run_synthetic.py",
             "provenance/SYNTHETIC_PROTOCOL.json", "methods/GEOMETRY_DERIVATION_ZH.md"]
    hashes = {relative: sha(ROOT / relative) for relative in bound}
    results = {"provenance": provenance_check(),
               "original_failure_fixture": solver_failure_fixture(),
               "gate_and_counts": gate_check(protocol)}
    original_reruns = {}
    for name in ("INDEPENDENT_CORE_AUDIT_RECOVERY.json", "RUNNER_PREFIT_AUDIT_RECOVERY.json"):
        path = ROOT / "audits" / name
        data = read(path)
        assert data["status"] == "PASS"
        for relative, expected in data["source_hashes"].items():
            assert hashes[relative] == expected
        original_reruns[name] = {"sha256": sha(path), "status": data["status"]}
    assert hashes == {relative: sha(ROOT / relative) for relative in bound}
    result = {
        "status": "PASS",
        "decision": "PASS_RECOVERY_QUALIFICATION",
        "audit": "Independent recovery pipeline and original-failure-fixture audit",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": hashes, "audit_script_sha256": sha(Path(__file__)),
        "scope": "Frozen design review, unchanged original numerical audits, archived numerical failure, artificial gates",
        "original_audit_reruns": original_reruns, "checks": results,
        "formal_validation_scores_computed": 0, "formal_test_streams": 0,
        "market_data_reads": 0, "publication_ready": False,
        "blocking_scientific_protocol_errors_found": [],
        "required_before_formal_restart": [
            "Create a new source freeze referencing the recovered core and new audit receipts; preserve prior freeze and failed attempt.",
            "Use a new run record and execute the unchanged finite protocol from the start."],
        "required_before_any_market_continuation": [
            "Complete all four scenarios and retain every planned validation trial and selected stream.",
            "Independently replay all saved test updates, original-objective KKT, scores, choices, counts and gate.",
            "Treat runner numerical_and_causal_checks as execution-stage status only; independent audit completion is a separate required release receipt.",
            "Continue only if every originally frozen scientific gate and final independent audit passes."],
        "interpretive_limits": [
            "No test- or oracle-based selection was found in the reviewed source; this is code/provenance evidence, not proof of all unrecorded historical actions.",
            "The first forecast is issued after y659, so it is valid to select with a validation label maturing at y659.",
            "The current learned object is a filtering update, with known transition/emission parameters; no neural fitting is performed.",
            "Both positive scenarios share random draws within each seed; only 20 seeds are independent.",
            "The fixed contamination-scale Bayes control is intentionally misspecified in the moderate-shock scenario.",
            "Passing the finite gate does not establish model originality, market accuracy or publication readiness."]}
    output = ROOT / "audits" / "RECOVERY_PIPELINE_AUDIT.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output),
                      "failure_fixture": results["original_failure_fixture"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
