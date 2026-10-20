"""Reproduce the archived KL failure and audit its arithmetic-only repair.

No validation score, hyperparameter selection, test forecast, or market input
is computed. Only the previously fixed synthetic validation observations are
passed through the fixed filters. The original primal KL objective is checked
without using the production dual factorization.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import mpmath as mp
import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
import geometry_filter as gf
import run_synthetic as runner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, threshold, name):
    if not np.isfinite(value) or value > threshold:
        raise AssertionError(f"{name}: {value:.17g} > {threshold:.17g}")


def independent_primal_kkt(p, y, eta, q):
    p, y, eta, q = map(np.asarray, (p, y, eta, q))
    used = eta > 0
    yy = y[used, None]
    sd = np.sqrt(np.array([.16, .64, .64, 2.56]))
    z = yy / sd
    a = np.sqrt(2 / np.pi) * sd * np.exp(-.5 * z * z) + yy * (2 * ndtr(z) - 1)
    pair = np.sqrt(2 / np.pi) * np.sqrt(sd[:, None] ** 2 + sd[None, :] ** 2)
    g = a - q[used] @ pair.T + (np.log(q[used]) - np.log(p[used])) / eta[used, None]
    return float(np.max(np.abs(g - g.mean(axis=1, keepdims=True))))


def high_precision_primal(p, y, eta):
    """80-digit root of the original three primal log-odds KKT equations."""
    pp = [mp.mpf(float(x)) for x in p]
    yy, ee = mp.mpf(float(y)), mp.mpf(float(eta))
    variances = [mp.mpf(float(v)) for v in (.16, .64, .64, 2.56)]
    sd = [mp.sqrt(v) for v in variances]
    a = [mp.sqrt(2 / mp.pi) * s * mp.exp(-yy * yy / (2 * s * s))
         + yy * mp.erf(yy / (mp.sqrt(2) * s)) for s in sd]
    pair = mp.matrix([[mp.sqrt(2 / mp.pi) * mp.sqrt(vi + vj)
                       for vj in variances] for vi in variances])
    lp = [mp.log(pp[i] / pp[3]) for i in range(3)]

    def probability(x):
        z = list(x) + [mp.mpf(0)]
        maximum = max(z)
        masses = [mp.exp(v - maximum) for v in z]
        total = mp.fsum(masses)
        return [v / total for v in masses]

    def residual(*x):
        q = probability(x)
        return tuple(x[i] - lp[i] + ee * (a[i] - a[3]
                     - mp.fsum((pair[i, j] - pair[3, j]) * q[j] for j in range(4)))
                     for i in range(3))

    def jacobian(*x):
        q = probability(x)
        return mp.matrix([[int(i == j) - ee * mp.fsum(
            (pair[i, k] - pair[3, k]) * q[k] * (int(k == j) - q[j])
            for k in range(4)) for j in range(3)] for i in range(3)])

    solved = mp.findroot(residual, lp, J=jacobian, solver="mdnewton",
                         tol=mp.mpf("1e-65"), maxsteps=100)
    q = probability(solved)
    kkt = max(abs(v) for v in residual(*solved))

    def objective(qvalue):
        return (mp.fsum(qvalue[i] * a[i] for i in range(4))
                - mp.fsum(qvalue[i] * pair[i, j] * qvalue[j]
                          for i in range(4) for j in range(4)) / 2
                + mp.fsum(qvalue[i] * mp.log(qvalue[i] / pp[i]) for i in range(4)) / ee)

    return q, kkt, objective


def inspect_archived_roundoff(failure, old):
    """Compare the same old final point's full Newton step at 80 digits."""
    p = np.array(failure["p"])[None, :]
    y = np.array([failure["y"]])
    eta = np.array([failure["eta"]])
    u = np.array(failure["u_after_budget"])[None, :]
    logp = np.log(p)
    c = old.normal_abs(y, old.SDS) - (old.CURRENT_PAIR @ old.PI)[None, :]
    c -= c.mean(axis=1, keepdims=True)
    old_value, q = old._dual_value(u, logp, c, eta)
    mean = q @ old._L2
    g = u - mean
    covariance = np.einsum("bi,ij,ik->bjk", q, old._L2, old._L2)
    covariance -= mean[:, :, None] * mean[:, None, :]
    hessian = np.eye(old._L2.shape[1])[None] + eta[:, None, None] * covariance
    direction = -np.linalg.solve(hessian, g[..., None])[..., 0]
    trial_u = u + direction
    trial_value, trial_q = old._dual_value(trial_u, logp, c, eta)
    allowance = 8 * np.finfo(float).eps * (1 + np.abs(old_value))
    armijo = old_value + 1e-4 * np.sum(g * direction, axis=1) + allowance

    def exact_phi(x):
        logits = [mp.mpf(float(logp[0, i])) - mp.mpf(float(eta[0]))
                  * (mp.mpf(float(c[0, i])) + mp.fsum(
                      mp.mpf(float(x[0, j])) * mp.mpf(float(old._L2[i, j]))
                      for j in range(x.shape[1]))) for i in range(4)]
        maximum = max(logits)
        return (mp.fsum(mp.mpf(float(v)) ** 2 for v in x[0]) / 2
                + (maximum + mp.log(mp.fsum(mp.exp(z - maximum) for z in logits)))
                / mp.mpf(float(eta[0])))

    exact_difference = exact_phi(trial_u) - exact_phi(u)
    return {
        "scope": "Archived final point; same mathematical old dual, binary64 inputs lifted exactly to 80 digits",
        "exact_full_step_objective_difference_80_digits": mp.nstr(exact_difference, 60),
        "binary64_full_step_objective_difference": float(trial_value[0] - old_value[0]),
        "old_roundoff_allowance": float(allowance[0]),
        "old_full_step_accepted_in_this_one_row_recalculation": bool(trial_value[0] <= armijo[0]),
        "old_point_gradient_inf_norm": float(np.max(np.abs(g))),
        "full_step_gradient_inf_norm": float(np.max(np.abs(trial_u - trial_q @ old._L2))),
        "roundoff_to_true_decrease_magnitude": float(allowance[0] / abs(exact_difference)),
        "note": "Scalar BLAS reevaluation can round differently from the original 140-row batch; archived exception is separately reproduced with exact full batch."
    }


def exact_failure_fixture(fixture):
    old_path = ROOT / "attempts/01_kl_dual_stagnation/source/geometry_filter.py"
    spec = importlib.util.spec_from_file_location("archived_geometry_filter", old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    args = fixture["batch_input_for_exact_reproduction"]
    old_exception = None
    try:
        old.update_crps_kl(**args, return_receipt=True)
    except FloatingPointError as exc:
        old_exception = str(exc)
    if old_exception != fixture["exception"]:
        raise AssertionError(f"Archived exact failure reproduction differs: {old_exception!r}")
    q, receipt = gf.update_crps_kl(**args, return_receipt=True)
    p, y, eta = (np.asarray(args[k]) for k in ("p", "y", "eta"))
    independent_kkt = independent_primal_kkt(p, y, eta, q)
    require(independent_kkt, 2e-8, "fixed fixture original KKT")
    if not np.array_equal(q[eta == 0], p[eta == 0]):
        raise AssertionError("eta=0 identity changed")

    mp.mp.dps = 80
    max_probability_gap, max_objective_gap, max_mp_kkt = mp.mpf(0), mp.mpf(0), mp.mpf(0)
    per_row = []
    for i in np.flatnonzero(eta > 0):
        q_mp, kkt_mp, objective = high_precision_primal(p[i], y[i], eta[i])
        candidate = [mp.mpf(float(v)) for v in q[i]]
        q_gap = max(abs(candidate[j] - q_mp[j]) for j in range(4))
        objective_gap = abs(objective(candidate) - objective(q_mp))
        max_probability_gap = max(max_probability_gap, q_gap)
        max_objective_gap = max(max_objective_gap, objective_gap)
        max_mp_kkt = max(max_mp_kkt, kkt_mp)
        per_row.append({"batch_row": int(i), "eta": float(eta[i]),
                        "probability_gap": float(q_gap), "objective_gap": float(objective_gap),
                        "independent_kkt_80_digits": mp.nstr(kkt_mp, 8)})
    require(float(max_probability_gap), 2e-8, "high precision probability")
    require(float(max_objective_gap), 2e-9, "high precision original objective")
    require(float(max_mp_kkt), 1e-50, "high precision independent stationary equations")
    result = {
        "status": "PASS", "archived_exception_reproduced": old_exception,
        "new_solver_receipt": receipt,
        "independent_unfactored_primal_kkt_max": independent_kkt,
        "high_precision": {
            "precision_decimal_digits": 80, "positive_eta_cases": len(per_row),
            "method": "Original three primal log-odds stationary equations; independent Gaussian absolute moments; prior-log-odds initialization",
            "max_probability_gap": float(max_probability_gap),
            "max_original_objective_gap": float(max_objective_gap),
            "max_independent_equation_residual": mp.nstr(max_mp_kkt, 30),
            "cases": per_row,
        },
        "roundoff_evidence": inspect_archived_roundoff(fixture["failing_rows"][0], old),
    }
    print(json.dumps({"phase": "fixed_failure_and_80_digit_primal_check", "status": "PASS",
                      "positive_eta_rows": len(per_row),
                      "max_probability_gap": float(max_probability_gap)}), flush=True)
    return result


def replay_validation_without_scoring(protocol):
    ref = gf.build_reference()
    u = np.array([1., 2., 4., -7.]) / np.sqrt(70)
    rotation = np.eye(4) - 2 * np.outer(u, u)
    ref["metrics"]["rotated"] = rotation @ ref["metrics"]["future"] @ rotation
    sim = protocol["simulation"]
    seeds = sim["seeds"]
    gf.reset_numerical_counters()
    runner.reset_propagation_accounting()
    checks = []
    for scenario in protocol["scenarios"]:
        # Original full length is required to preserve the frozen RNG stream.
        # Only t<test_start_index observations are consumed below. No target
        # sums, feature matrices, scores, oracle filters or test forecasts exist.
        observations = np.stack([
            runner.simulate(seed, scenario, sim["total_observations_per_scenario_seed"],
                            ref["v"])[1][:sim["test_start_index"]] for seed in seeds])
        for family in protocol["families"]:
            started = time.monotonic()
            grid = runner.family_grid(family, protocol)
            parameters = np.broadcast_to(grid, (len(seeds), len(grid))).reshape(-1)
            prior = np.full((len(parameters), 4), .25)
            max_dual, max_kkt, max_sum, min_probability, max_iterations = 0., 0., 0., 1., 0
            for t in range(sim["test_start_index"]):
                y = np.repeat(observations[:, t], len(grid))
                if family == "crps_kl":
                    q, r = gf.update_crps_kl(prior, y, parameters, return_receipt=True)
                    if r["tolerance"] != 1e-12 or r["max_iter_budget"] != 100:
                        raise AssertionError("Original numerical settings changed")
                    max_dual = max(max_dual, r["max_dual_gradient_residual"])
                    max_kkt = max(max_kkt, independent_primal_kkt(prior, y, parameters, q))
                    max_iterations = max(max_iterations, r["max_iterations"])
                    require(max_dual, 1e-12, "full validation KL dual residual")
                    require(max_kkt, 2e-8, "full validation unfactored KL KKT")
                else:
                    q = runner.apply_update(family, prior, y, parameters, ref)
                runner.assert_probabilities(q)
                if family != "contaminated_bayes":
                    if not np.array_equal(q[parameters == 0], prior[parameters == 0]):
                        raise AssertionError("Zero update identity changed")
                max_sum = max(max_sum, float(np.max(np.abs(q.sum(axis=1) - 1))))
                min_probability = min(min_probability, float(q.min()))
                prior = runner.advance(q, ref["P"])
            row = {"scenario": scenario["name"], "family": family, "status": "PASS",
                   "seeds": len(seeds), "grid_entries": len(grid),
                   "observation_indices": [0, sim["test_start_index"] - 1],
                   "configuration_streams": len(parameters),
                   "update_rows": len(parameters) * sim["test_start_index"],
                   "max_simplex_mass_error": max_sum, "min_probability": min_probability,
                   "elapsed_seconds": time.monotonic() - started}
            if family == "crps_kl":
                row.update(max_dual_gradient_residual=max_dual,
                           max_independent_unfactored_kkt_residual=max_kkt,
                           max_newton_iterations=max_iterations)
            checks.append(row)
            print(json.dumps({k: row[k] for k in ("scenario", "family", "status", "elapsed_seconds")}), flush=True)
    return {"status": "PASS", "checks": checks,
            "configuration_streams": sum(x["configuration_streams"] for x in checks),
            "update_rows": sum(x["update_rows"] for x in checks),
            "kl_configuration_streams": sum(x["configuration_streams"] for x in checks if x["family"] == "crps_kl"),
            "validation_scores_computed": False, "hyperparameters_selected": False,
            "test_forecasts_computed": False, "test_scores_computed": False,
            "synthetic_generation_note": "Frozen length 1703 preserves random draw positions; only origins 0..658 are consumed for filters, no targets or scores constructed.",
            "numerical_counters": gf.get_numerical_counters(),
            "propagation_roundoff_accounting": runner.PROPAGATION_ACCOUNTING.copy()}


def main():
    started = time.monotonic()
    paths = ["source/geometry_filter.py", "source/run_synthetic.py", "provenance/SYNTHETIC_PROTOCOL.json",
             "attempts/01_kl_dual_stagnation/source/geometry_filter.py"]
    before = {p: sha(ROOT / p) for p in paths}
    protocol = json.loads((ROOT / "provenance/SYNTHETIC_PROTOCOL.json").read_text())
    fixture = json.loads((ROOT / "audits/kl_roundoff_diagnostic.json").read_text())
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "audit": "Arithmetic repair qualification; no performance or market evaluation",
              "source_hashes": before, "script_sha256": sha(__file__),
              "tolerance_preserved": 1e-12, "max_iter_preserved": 100,
              "line_search_budget_preserved": 50, "model_or_protocol_changes": False,
              "market_reads": 0}
    report["fixed_failure"] = exact_failure_fixture(fixture)
    report["unscored_validation_replay"] = replay_validation_without_scoring(protocol)
    after = {p: sha(ROOT / p) for p in paths}
    if before != after:
        raise AssertionError("Sources changed during numerical diagnosis")
    report.update(status="PASS", elapsed_seconds=time.monotonic() - started)
    target = ROOT / "audits/SOLVER_REPAIR_DIAGNOSTIC.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(target),
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
