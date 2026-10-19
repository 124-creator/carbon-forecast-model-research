"""Independent deterministic numerical qualifications; no experiment fitting.

The production core is checked against CDF quadrature, future-path enumeration,
SLSQP, and a primal log-odds KKT solver. This script reads no market data. Its
fixed fixtures and numerical tolerances are unrelated to benchmark outcomes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, root
from scipy.special import ndtr, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
import geometry_filter as gf
import run_synthetic as runner

RNG = np.random.default_rng(20261019)
CHECKS = {}
START = time.monotonic()
SOURCE_PATHS = ["source/geometry_filter.py", "source/run_synthetic.py",
                "provenance/SYNTHETIC_PROTOCOL.json"]
HASHES = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
          for p in SOURCE_PATHS}


def record(name, data):
    CHECKS[name] = {"status": "PASS", **data}
    print(json.dumps({"check": name, **CHECKS[name]}), flush=True)


def require(error, bound, name):
    if not np.isfinite(error) or error > bound:
        raise AssertionError(f"{name}: {error:.17g} > {bound:.17g}")


def independent_a(y, sd):
    sd = np.asarray(sd)
    z = y / sd
    return sd * np.sqrt(2 / np.pi) * np.exp(-.5 * z * z) + y * (2 * ndtr(z) - 1)


def enumerate_state_paths(p, h):
    """Direct 4**h paths, with integer variance ticks; no lattice recursion."""
    ticks = np.array([1, 4, 4, 16], dtype=int)
    result = np.zeros((4, 16 * h + 1))
    for origin in range(4):
        for path in product(range(4), repeat=h):
            previous = origin
            probability = 1.
            variance_ticks = 0
            for state in path:
                probability *= p[previous, state]
                variance_ticks += int(ticks[state])
                previous = state
            result[origin, variance_ticks] += probability
    return result


def enumerate_observed_law(p, h, eps, tau):
    """State paths times explicit binary shock paths; no binomial function."""
    states = enumerate_state_paths(p, h)
    flag_mass = np.zeros(h + 1)
    for flags in product((0, 1), repeat=h):
        probability = 1.
        for flag in flags:
            probability *= eps if flag else 1 - eps
        flag_mass[sum(flags)] += probability
    columns, scales = [], []
    for ticks in range(states.shape[1]):
        if states[:, ticks].sum() == 0:
            continue
        for count, probability in enumerate(flag_mass):
            if probability:
                columns.append(states[:, ticks] * probability)
                scales.append(np.sqrt((.16 * ticks + tau * tau * count) / h))
    return np.column_stack(columns), np.array(scales)


def integrated_crps(weights, sd, y):
    left = quad(lambda x: float(weights @ ndtr(x / sd)) ** 2,
                -np.inf, y, epsabs=2e-11, epsrel=2e-11, limit=200)[0]
    right = quad(lambda x: float(weights @ ndtr(-x / sd)) ** 2,
                 y, np.inf, epsabs=2e-11, epsrel=2e-11, limit=200)[0]
    return left + right


def integrated_expected_crps(fw, fs, tw, ts):
    # For T=P(Y<=x), E[(F(x)-1{Y<=x})**2]=(F-T)**2+T*(1-T).
    # All fixture laws are symmetric, so stable survival CDFs suffice on x>=0.
    def integrand(x):
        f = float(fw @ ndtr(-x / fs))
        t = float(tw @ ndtr(-x / ts))
        return 2 * ((f - t) ** 2 + t * (1 - t))
    return quad(integrand, 0, np.inf, epsabs=2e-11, epsrel=2e-11,
                limit=200)[0]


def check_dp(ref):
    max_mass, max_scale, count = 0., 0., 0
    wrong = runner.transition(.6, .1)
    general = np.array([[.4, .1, .2, .3], [.1, .7, .1, .1],
                        [.05, .15, .7, .1], [.2, .3, .4, .1]])
    for p in (ref["P"], wrong, general):
        for h in range(1, 6):
            w, sd = gf.conditional_mixture(p, gf.V, h)
            exact = enumerate_state_paths(p, h)
            locations = h + 3 * np.arange(w.shape[1])
            max_mass = max(max_mass, float(np.max(np.abs(w - exact[:, locations]))))
            max_scale = max(max_scale, float(np.max(np.abs(sd ** 2 - .16 * locations / h))))
            require(np.max(np.abs(exact.sum(1) - 1)), 2e-13, "enumerated mass")
            count += 1
    require(max_mass, 2e-13, "DP path mass")
    require(max_scale, 3e-14, "DP normalized variance")
    record("future_dp_against_path_enumeration", {
        "transition_fixtures": 3, "horizons": [1, 2, 3, 4, 5],
        "cases": count, "max_mass_error": max_mass,
        "max_normalized_variance_error": max_scale,
        "mass_tolerance": 2e-13, "variance_tolerance": 3e-14})


def check_crps_and_gram(ref):
    distributions = {"current": {"weights": np.eye(4), "sds": gf.SDS,
                                   "pair": gf.CURRENT_PAIR, "gram": gf.CURRENT_GRAM}}
    distributions.update({f"h{h}": ref["conditional"][h] for h in (1, 5, 20)})
    max_score, max_gram, cases = 0., 0., 0
    for law in distributions.values():
        w, sd = law["weights"], law["sds"]
        for y in (-2.3, 0., .8, 4.1):
            q, p = RNG.dirichlet(np.ones(4), size=2)
            analytic = float(gf.mixture_crps(q, y, w, sd, law["pair"]))
            max_score = max(max_score, abs(analytic - integrated_crps(q @ w, sd, y)))
            d = q - p
            integral = 2 * quad(lambda x: float((d @ w) @ ndtr(-x / sd)) ** 2,
                                0, np.inf, epsabs=2e-12, epsrel=2e-11, limit=200)[0]
            max_gram = max(max_gram, abs(float(d @ law["gram"] @ d) - integral))
            cases += 1
    require(max_score, 2e-9, "CRPS CDF quadrature")
    require(max_gram, 2e-10, "CDF Gram quadrature")
    record("crps_and_gram_against_cdf_quadrature", {
        "cases": cases, "laws": list(distributions),
        "max_crps_absolute_error": max_score, "crps_tolerance": 2e-9,
        "max_gram_absolute_error": max_gram, "gram_tolerance": 2e-10})


def check_qp(ref, rotated):
    metrics = dict(ref["metrics"])
    metrics["rotated"] = rotated
    metrics["chi_square_only"] = np.zeros((4, 4))
    metrics["future_ratio_preserving"] = ref["metrics"]["future"]
    priors = RNG.dirichlet(np.ones(4), size=8) @ ref["P"]
    priors[0] = .25
    observations = (-8., -2., -.6, 0., .2, .9, 2.5, 10.)
    pair = np.sqrt(2 / np.pi) * np.sqrt(gf.V[:, None] + gf.V[None, :])
    max_obj_gap, max_q_gap, max_kkt, max_repair, max_ratio = 0., 0., 0., 0., 0.
    cases, failures = 0, []
    core_receipts = []
    for name, g in metrics.items():
        is_ratio = name == "future_ratio_preserving"
        kappa = 1. if name == "chi_square_only" else .01
        for p, y in zip(priors, observations):
            a = independent_a(y, np.sqrt(gf.V))
            for eta in (.001, .01, .1, 1., 10., 100.):
                q, receipt = gf.update_qp(p, y, eta, g, kappa,
                                         ratio_preserving=is_ratio, return_receipt=True)
                multiplier = min(eta, 1.)

                def objective(x):
                    d = x - p
                    score = x @ a - .5 * x @ pair @ x
                    penalty = (d @ g @ d + kappa * np.sum(d * d / p)) / eta
                    return multiplier * (score + penalty)

                def gradient(x):
                    d = x - p
                    return multiplier * (a - pair @ x + 2 / eta * (g @ d + kappa * d / p))

                constraints = [{"type": "eq", "fun": lambda x: x.sum() - 1,
                                "jac": lambda x: np.ones(4)}]
                if is_ratio:
                    ratio_vec = np.array([0, p[2], -p[1], 0])
                    constraints.append({"type": "eq", "fun": lambda x: x @ ratio_vec,
                                        "jac": lambda x: ratio_vec})
                independent = minimize(objective, p.copy(), jac=gradient, method="SLSQP",
                                       bounds=[(0, 1)] * 4, constraints=constraints,
                                       options={"ftol": 1e-13, "maxiter": 500})
                if not independent.success:
                    failures.append({"metric": name, "y": y, "eta": eta,
                                     "message": str(independent.message)})
                obj_gap = abs(objective(q) - objective(independent.x))
                q_gap = np.max(np.abs(q - independent.x))
                max_obj_gap = max(max_obj_gap, float(obj_gap))
                max_q_gap = max(max_q_gap, float(q_gap))
                max_kkt = max(max_kkt, receipt["max_kkt_residual"])
                max_repair = max(max_repair, receipt["max_feasibility_repair"])
                max_ratio = max(max_ratio, abs(q[1] * p[2] - q[2] * p[1]) if is_ratio else 0)
                core_receipts.append({"metric": name, "y": y, "eta": eta,
                                      "scaled_objective_gap": float(obj_gap),
                                      "probability_gap": float(q_gap),
                                      "core_kkt": receipt["max_kkt_residual"]})
                cases += 1
    if failures:
        raise AssertionError(f"Independent SLSQP failure(s): {failures}")
    require(max_obj_gap, 2e-9, "QP scaled objective versus SLSQP")
    require(max_q_gap, 3e-5, "QP probabilities versus SLSQP")
    require(max_kkt, 2e-8, "QP KKT")
    require(max_repair, 2e-11, "QP feasibility repair")
    require(max_ratio, 2e-13, "QP exact middle-odds constraint")
    record("qp_against_independent_slsqp", {
        "cases": cases, "metrics": list(metrics),
        "independent_solver": "SLSQP on original simplex and optional linear ratio constraint",
        "independent_solver_failures": failures,
        "max_scaled_objective_gap": max_obj_gap, "objective_tolerance": 2e-9,
        "max_probability_gap": max_q_gap, "probability_tolerance": 3e-5,
        "max_core_kkt_residual": max_kkt, "kkt_tolerance": 2e-8,
        "max_recorded_feasibility_repair": max_repair,
        "max_ratio_constraint_residual": max_ratio,
        "worst_probability_fixture": max(core_receipts, key=lambda x: x["probability_gap"]),
        "worst_objective_fixture": max(core_receipts, key=lambda x: x["scaled_objective_gap"])})


def check_kl(ref):
    priors = RNG.dirichlet(np.ones(4), size=8) @ ref["P"]
    observations = (-8., -2., -.6, 0., .2, .9, 2.5, 10.)
    pair = np.sqrt(2 / np.pi) * np.sqrt(gf.V[:, None] + gf.V[None, :])
    max_q, max_kkt, max_root, max_obj, cases = 0., 0., 0., 0., 0
    for p, y in zip(priors, observations):
        a = independent_a(y, np.sqrt(gf.V))
        ap = a[:3] - a[3]
        row_diff = pair[:3] - pair[3]
        prior_logits = np.log(p[:3]) - np.log(p[3])
        for eta in (.001, .01, .1, 1., 10., 100.):
            q, receipt = gf.update_crps_kl(p, y, eta, return_receipt=True)

            def probability(x):
                z = np.r_[x, 0.]
                return np.exp(z - logsumexp(z))

            def residual(x):
                return x - prior_logits + eta * (ap - row_diff @ probability(x))

            def jacobian(x):
                qq = probability(x)
                dq = (np.diag(qq) - np.outer(qq, qq))[:, :3]
                return np.eye(3) - eta * row_diff @ dq

            solved = root(residual, prior_logits, jac=jacobian, method="hybr",
                          options={"xtol": 1e-11, "maxfev": 1000})
            residual_error = float(np.max(np.abs(residual(solved.x))))
            # HYBR may report floating-point stagnation at an exact root. The
            # predeclared equation residual, not its message, is the criterion.
            require(residual_error, 2e-9, "independent primal log-odds root")
            x = probability(solved.x)
            grad = a - pair @ q + (np.log(q) - np.log(p)) / eta
            kkt = float(np.max(np.abs(grad - grad.mean())))

            def objective(xx):
                return xx @ a - .5 * xx @ pair @ xx + np.sum(xx * (np.log(xx) - np.log(p))) / eta

            max_q = max(max_q, float(np.max(np.abs(q - x))))
            max_obj = max(max_obj, abs(float(objective(q) - objective(x))))
            max_kkt = max(max_kkt, kkt, receipt["max_kkt_residual"])
            max_root = max(max_root, residual_error)
            cases += 1
    require(max_q, 2e-8, "KL independent probability")
    require(max_obj, 2e-9, "KL independent objective")
    require(max_kkt, 2e-8, "KL original KKT")
    record("native_crps_kl_against_primal_log_odds_root", {
        "cases": cases,
        "independent_solver": "HYBR root of three original primal KKT log-odds equations; no core dual/factorization",
        "max_probability_gap": max_q, "probability_tolerance": 2e-8,
        "max_original_objective_gap": max_obj, "objective_tolerance": 2e-9,
        "max_original_kkt_residual": max_kkt, "kkt_tolerance": 2e-8,
        "max_independent_equation_residual": max_root, "equation_tolerance": 2e-9})


def check_geometry_and_invariance(ref, rotated, rotation):
    # Nonuniform pi is intentional: the K=2 statement is not restricted to .5.
    pi2 = np.array([.27, .73])
    b2 = np.eye(2) - np.ones((2, 2)) / 2
    c2 = np.diag(pi2) - np.outer(pi2, pi2)
    normalized = []
    for scale in (.004, .31, 7., 100.):
        g = scale * b2
        normalized.append(g / np.trace(c2 @ g))
    two_state_error = max(float(np.max(np.abs(g - normalized[0]))) for g in normalized)
    require(two_state_error, 2e-13, "K=2 normalized metric equality")

    # Split one genuinely identical component and divide its stationary mass.
    sd = np.array([.3, .8, 1.9])
    pair = np.sqrt(2 / np.pi) * np.sqrt(sd[:, None] ** 2 + sd[None, :] ** 2)
    pi = np.array([.2, .35, .45])
    b = np.eye(3) - np.ones((3, 3)) / 3
    g = -.5 * b @ pair @ b
    duplicate = np.array([0, 1, 1, 2])
    pi_split = np.array([.2, .35 * .4, .35 * .6, .45])
    b4 = np.eye(4) - np.ones((4, 4)) / 4
    g_split = -.5 * b4 @ pair[duplicate[:, None], duplicate[None, :]] @ b4
    den = np.trace((np.diag(pi) - np.outer(pi, pi)) @ g)
    den_split = np.trace((np.diag(pi_split) - np.outer(pi_split, pi_split)) @ g_split)
    pair_cdf_dist = pair - .5 * (np.diag(pair)[:, None] + np.diag(pair)[None, :])
    pair_formula = .5 * pi @ pair_cdf_dist @ pi
    split_error = float(abs(den - den_split))
    pair_error = float(abs(den - pair_formula))
    require(split_error, 2e-13, "duplicate-state normalization")
    require(pair_error, 2e-13, "pairwise interpretation of normalization")
    p = np.array([.3, .25, .45])
    q = np.array([.21, .34, .45])
    ps = np.array([p[0], p[1] * .4, p[1] * .6, p[2]])
    qs = np.array([q[0], q[1] * .4, q[1] * .6, q[2]])
    chi_error = abs(np.sum((q - p) ** 2 / p) - np.sum((qs - ps) ** 2 / ps))
    geom_error = abs((q - p) @ g @ (q - p) / den - (qs - ps) @ g_split @ (qs - ps) / den_split)
    require(float(chi_error), 2e-13, "split chi-square penalty")
    require(float(geom_error), 2e-13, "split normalized forecast penalty")

    eigen_error = float(np.max(np.abs(np.linalg.eigvalsh(rotated) - np.linalg.eigvalsh(ref["metrics"]["future"]))))
    require(eigen_error, 2e-12, "rotated metric same spectrum")
    require(float(np.max(np.abs(rotation @ np.ones(4) - 1))), 2e-13, "rotation preserves tangent")
    require(float(np.max(np.abs(rotation @ rotation - np.eye(4)))), 2e-13, "rotation orthogonal")
    covariance = np.diag(gf.PI) - np.outer(gf.PI, gf.PI)
    norm_error = abs(float(np.trace(covariance @ rotated)) - 1)
    require(norm_error, 2e-13, "rotation normalization")
    if np.linalg.norm(rotated - ref["metrics"]["future"]) < .1:
        raise AssertionError("Rotation fixture failed to change geometry orientation")
    uniform = gf.build_reference(np.full((4, 4), .25))
    degenerate_error = max(float(np.max(np.abs(uniform["metrics"][k])))
                           for k in ("future", "variance"))
    require(degenerate_error, 0., "IID future metric exact degeneracy")
    # The tempting slow/fast swap must be recognized as an observable relabeling.
    j = np.eye(4)[[0, 2, 1, 3]]
    swapped = gf.build_reference(runner.transition(.1, .97))
    swap_error = max(float(np.max(np.abs(swapped["P"] - j @ ref["P"] @ j))),
                     float(np.max(np.abs(swapped["v"] - j @ ref["v"]))))
    for h in (1, 5, 20):
        swap_error = max(swap_error, float(np.max(np.abs(swapped["conditional"][h]["weights"] - j @ ref["conditional"][h]["weights"]))))
    require(swap_error, 2e-12, "factor swap relabeling")
    record("geometry_degeneracy_and_invariance", {
        "k2_max_normalized_matrix_error": two_state_error,
        "duplicate_state_normalization_error": split_error,
        "normalization_pair_distance_identity_error": pair_error,
        "duplicate_state_chi_square_error": float(chi_error),
        "duplicate_state_forecast_penalty_error": float(geom_error),
        "rotation_spectrum_error": eigen_error, "rotation_normalization_error": norm_error,
        "iid_future_metric_max_abs": degenerate_error,
        "factor_swap_relabeling_error": swap_error})


def check_odds_and_zero(ref):
    p = RNG.dirichlet(np.ones(4), size=17) @ ref["P"]
    y = np.linspace(-4, 4, len(p))
    odds_error = 0.
    for beta in (0., .1, 1., 2.):
        q = gf.tempered_bayes(p, y, beta)
        odds_error = max(odds_error, float(np.max(np.abs(q[:, 1] * p[:, 2] - q[:, 2] * p[:, 1]))))
    for eps in (0., .05, .2):
        q = gf.contaminated_bayes(p, y, eps)
        odds_error = max(odds_error, float(np.max(np.abs(q[:, 1] * p[:, 2] - q[:, 2] * p[:, 1]))))
    for eta in (.001, 1., 100.):
        q = gf.update_crps_kl(p, y, eta)
        odds_error = max(odds_error, float(np.max(np.abs(q[:, 1] * p[:, 2] - q[:, 2] * p[:, 1]))))
        q = gf.update_qp(p, y, eta, ref["metrics"]["future"], ratio_preserving=True)
        odds_error = max(odds_error, float(np.max(np.abs(q[:, 1] * p[:, 2] - q[:, 2] * p[:, 1]))))
    require(odds_error, 2e-13, "identical-emission odds")
    identities = [gf.update_qp(p, y, 0., ref["metrics"]["future"]),
                  gf.update_qp(p, y, 0., ref["metrics"]["future"], ratio_preserving=True),
                  gf.update_crps_kl(p, y, 0.), gf.tempered_bayes(p, y, 0.)]
    if not all(np.array_equal(q, p) for q in identities):
        raise AssertionError("Zero update failed exact binary64 identity")
    positive_prior = np.eye(4) @ ref["P"]
    if np.min(positive_prior) <= 0:
        raise AssertionError("Nominal dynamics does not restore strictly positive priors")
    record("identical_emission_odds_and_zero_step", {
        "max_cross_product_odds_residual": odds_error, "odds_tolerance": 2e-13,
        "zero_step_exact_identity_families": 4,
        "minimum_next_prior_over_simplex_vertices": float(np.min(positive_prior))})


def check_oracle_and_indexing(ref, protocol):
    max_expected, max_entropy, max_identity, cases = 0., 0., 0., 0
    q, truth = np.array([.12, .2, .33, .35]), np.array([.39, .15, .18, .28])
    for scenario in protocol["scenarios"]:
        kernels = runner.oracle_kernels(ref, scenario)
        ptrue = runner.transition(scenario["slow_persistence"], scenario["fast_persistence"])
        for h in (1, 5):
            wt, st = enumerate_observed_law(ptrue, h, scenario["contamination_probability"], scenario["additive_shock_sd"])
            model = ref["conditional"][h]
            numeric = integrated_expected_crps(q @ model["weights"], model["sds"], truth @ wt, st)
            analytic = float(runner.expected_score(q, truth, model["pair"], kernels[h]["cross"]))
            max_expected = max(max_expected, abs(numeric - analytic))
            numeric_entropy = integrated_expected_crps(truth @ wt, st, truth @ wt, st)
            analytic_entropy = .5 * truth @ kernels[h]["true_pair"] @ truth
            max_entropy = max(max_entropy, abs(numeric_entropy - analytic_entropy))
            if scenario["name"] == "clean_correct":
                regret = (q - truth) @ model["gram"] @ (q - truth)
                max_identity = max(max_identity, abs(analytic - analytic_entropy - regret))
            if analytic < analytic_entropy - 2e-10:
                raise AssertionError("Oracle expected-CRPS lower bound violated")
            cases += 1
    require(max_expected, 2e-9, "oracle expected CRPS independent CDF integration")
    require(max_entropy, 2e-9, "oracle entropy independent CDF integration")
    require(max_identity, 2e-12, "Bayes CRPS regret identity")
    # The h20 exact Bayes identity uses separately checked finite-mixture algebra.
    for h in (1, 5, 20):
        law = ref["conditional"][h]
        expected = runner.expected_score(q, truth, law["pair"], law["pair"])
        entropy = .5 * truth @ law["pair"] @ truth
        regret = (q - truth) @ law["gram"] @ (q - truth)
        require(abs(float(expected - entropy - regret)), 2e-12, f"h{h} Bayes regret")
    # Strong indexed fixture: perturb current/past and remote future separately.
    obs = np.arange(100, dtype=float)[None, :]
    origins = np.array([5, 20, 50])
    targets, _ = runner.target_features(obs, ref, origins)
    indexing_error = 0.
    for h in (1, 5, 20):
        direct = np.array([obs[0, t + 1:t + h + 1].sum() / np.sqrt(h) for t in origins])
        indexing_error = max(indexing_error, float(np.max(np.abs(targets[h][0] - direct))))
    require(indexing_error, 2e-12, "future target off-by-one")
    sim = protocol["simulation"]
    if not (sim["validation_last_index"] + 20 == sim["last_validation_label_index"] == sim["test_start_index"]):
        raise AssertionError("Validation labels have not matured by first test forecast")
    if sim["test_last_index"] + 20 != sim["total_observations_per_scenario_seed"] - 1:
        raise AssertionError("Last test label or total observation count mismatch")
    ntrial = len(protocol["scenarios"]) * len(sim["seeds"]) * len(protocol["families"]) * 7
    nselected = len(protocol["scenarios"]) * len(sim["seeds"]) * len(protocol["families"])
    if ntrial != sim["evaluated_validation_filter_trials"] or nselected != sim["selected_family_streams"]:
        raise AssertionError("Protocol search counts mismatch")
    if not (ntrial == 5600 and nselected == 800):
        raise AssertionError("Declared finite qualification changed")
    record("oracle_expected_scores_and_causal_indexing", {
        "path_enumerated_oracle_cases": cases, "horizons": [1, 5],
        "max_expected_crps_error": max_expected, "max_true_entropy_error": max_entropy,
        "cdf_integration_tolerance": 2e-9, "max_bayes_regret_identity_error": max_identity,
        "max_explicit_future_slice_error": indexing_error,
        "validation_label_maturity_index": sim["last_validation_label_index"],
        "first_test_forecast_origin": sim["test_start_index"],
        "declared_validation_trials": ntrial, "declared_selected_streams": nselected,
        "selection_scope_review": "Source-reviewed: within-family and composite selectors use only empirical validation scores; their frozen JSON is written before oracle states or test scores are calculated. This numeric script does not run any validation or test filter stream."})


def run():
    protocol = json.loads((ROOT / "provenance/SYNTHETIC_PROTOCOL.json").read_text())
    ref = gf.build_reference()
    u = np.array([1., 2., 4., -7.]) / np.sqrt(70)
    rotation = np.eye(4) - 2 * np.outer(u, u)
    rotated = rotation @ ref["metrics"]["future"] @ rotation
    check_dp(ref)
    check_crps_and_gram(ref)
    check_geometry_and_invariance(ref, rotated, rotation)
    check_qp(ref, rotated)
    check_kl(ref)
    check_odds_and_zero(ref)
    check_oracle_and_indexing(ref, protocol)


if __name__ == "__main__":
    status, failure = "PASS", None
    try:
        run()
        if any(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != value for p, value in HASHES.items()):
            raise AssertionError("Audited source changed during this audit; rerun on a stable final version")
    except Exception as exc:
        status, failure = "FAIL", {"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()}
    receipt = {"status": status, "created_utc": datetime.now(timezone.utc).isoformat(),
               "audit": "Independent mathematical and numerical core qualification",
               "source_hashes": HASHES,
               "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "synthetic_fixture_seed": 20261019,
               "market_data_reads": 0, "benchmark_validation_trials": 0,
               "benchmark_test_streams": 0, "elapsed_seconds": time.monotonic() - START,
               "checks": CHECKS, "failure": failure,
               "numerical_counters_during_audit": gf.get_numerical_counters()}
    target = ROOT / "audits" / "INDEPENDENT_CORE_AUDIT.json"
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": status, "receipt": str(target), "failure": failure,
                      "elapsed_seconds": receipt["elapsed_seconds"]}), flush=True)
    sys.exit(0 if status == "PASS" else 1)
