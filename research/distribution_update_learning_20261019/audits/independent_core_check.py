#!/usr/bin/env python3
"""Numerical/math audit of source/scores.py and source/learning.py.

Only synthetic constants and generated numerical fixtures are used. No market
file, forecast file, or fitted model is read. External CDF quadrature, direct
atom sums and scipy optimization provide the principal references.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from itertools import product
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "source"
SEED = 20261019


def load_source(name):
    spec = importlib.util.spec_from_file_location("independent_audit_" + name, SOURCE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical_crps(y, weights, sigma):
    w, s = np.asarray(weights), np.asarray(sigma)
    def lower(x):
        return float(np.dot(w, norm.cdf(x / s))) ** 2
    def upper(x):
        return float(np.dot(w, norm.sf(x / s))) ** 2
    lo = quad(lower, -np.inf, y, epsabs=2e-11, epsrel=2e-11, limit=200)
    hi = quad(upper, y, np.inf, epsabs=2e-11, epsrel=2e-11, limit=200)
    return lo[0] + hi[0], lo[1] + hi[1]


def numerical_ab(y, weights, s0, s1):
    w = np.asarray(weights)
    def f0(x): return float(np.dot(w, norm.cdf(x / s0)))
    def d(x): return float(np.dot(w, norm.cdf(x / s1) - norm.cdf(x / s0)))
    a = quad(lambda x: d(x) ** 2, -np.inf, np.inf, epsabs=2e-11, epsrel=2e-11, limit=200)[0]
    bleft = quad(lambda x: -d(x) * f0(x), -np.inf, y, epsabs=2e-11, epsrel=2e-11, limit=200)[0]
    bright = quad(lambda x: d(x) * (1 - f0(x)), y, np.inf, epsabs=2e-11, epsrel=2e-11, limit=200)[0]
    return a, bleft + bright


def lognormal_quadrature(y, locations, sigmas, weights):
    """Integrate the CDF definition in log coordinates; bound omitted tails."""
    mu, sig, w = map(np.asarray, (locations, sigmas, weights))
    ly = math.log(y)
    left = min(float(np.min(mu - 12 * sig)), ly - 2)
    right = max(float(np.max(mu + 12 * sig)), ly + 2)
    def integrand(t, side):
        z = (t - mu) / sig
        probability = float(np.dot(w, norm.cdf(z) if side == "left" else norm.sf(z)))
        return math.exp(t) * probability * probability
    total, quadrature_error = 0.0, 0.0
    for first, last, side in ((left, ly, "left"), (ly, right, "right")):
        cuts = [first] + sorted(float(x) for x in mu if first < x < last) + [last]
        for a, b in zip(cuts, cuts[1:]):
            value, error = quad(lambda t: integrand(t, side), a, b, epsabs=2e-9, epsrel=1e-11, limit=200)
            total += value
            quadrature_error += error
    lower_tail_bound = math.exp(left) * float(np.dot(w, norm.cdf((left - mu) / sig))) ** 2
    means = np.exp(mu + sig ** 2 / 2)
    upper_tail_bound = float(np.dot(w * means, norm.sf((right - mu - sig ** 2) / sig)))
    assert np.all(np.isfinite(means)) and np.all(means > 0)
    return total, quadrature_error, lower_tail_bound + upper_tail_bound


def score_checks(scores):
    fixtures = [
        (np.array([1.]), np.array([0.2]), -1.1, 1.8),
        (np.array([1.]), np.array([2.1]), 0.0, 0.65),
        (np.array([0.2, 0.5, 0.3]), np.array([0.15, 0.7, 2.0]), 0.4, 1.35),
        (np.array([0.7, 0.3]), np.array([0.4, 1.5]), -2.4, 0.75),
        (np.array([0.05, 0.8, 0.15]), np.array([0.3, 1.2, 2.7]), 3.2, 1.0),
        (np.array([0.25, 0.25, 0.5]), np.array([0.1, 0.8, 1.5]), 0.0, 2.0),
    ]
    crps_error = ab_error = mixture_error = w1_error = integration_estimate = 0.0
    for w, s0, y, factor in fixtures:
        s1 = s0 * factor
        external, estimated = numerical_crps(y, w, s0)
        crps_error = max(crps_error, abs(float(scores.crps(y, w, s0)) - external))
        integration_estimate = max(integration_estimate, estimated)
        ae, be = numerical_ab(y, w, s0, s1)
        a, b, score0, _ = scores.pair_terms(np.asarray([y]), w, s0, s1)
        ab_error = max(ab_error, abs(float(a[0]) - ae), abs(float(b[0]) - be))
        d = lambda x: float(np.dot(w, norm.cdf(x / s1) - norm.cdf(x / s0)))
        w1_external = quad(lambda x: abs(d(x)), -np.inf, 0, epsabs=2e-11, limit=200)[0] + quad(lambda x: abs(d(x)), 0, np.inf, epsabs=2e-11, limit=200)[0]
        w1_error = max(w1_error, abs(float(scores.scale_w1(w, s0, factor)) - w1_external))
        for alpha in (0.0, 0.13, 0.5, 0.91, 1.0):
            mixed_w = np.r_[(1 - alpha) * w, alpha * w]
            mixed_s = np.r_[s0, s1]
            exact_score, _ = numerical_crps(y, mixed_w, mixed_s)
            quadratic_score = float(score0[0] + a[0] * alpha ** 2 - 2 * b[0] * alpha)
            mixture_error = max(mixture_error, abs(exact_score - quadratic_score))
    wb = np.array([[.2, .5, .3], [.5, .3, .2]])
    sb = np.array([[.2, .7, 1.8], [.5, 1.1, 2.1]])
    yb = np.array([-.9, 1.7])
    batched = scores.crps(yb, wb, sb)
    batch_error = max(abs(float(batched[i]) - numerical_crps(yb[i], wb[i], sb[i])[0]) for i in range(2))
    rng = np.random.default_rng(SEED)
    atomic_error = 0.0
    for _ in range(12):
        atoms = np.round(rng.normal(size=(3, 9)), 1)  # Include ties and zero atoms.
        weights = rng.dirichlet(np.ones(9), size=3)
        y = rng.normal(size=3)
        expected = np.sum(weights * abs(atoms - y[:, None]), axis=1) - .5 * np.sum(
            weights[:, :, None] * weights[:, None, :] * abs(atoms[:, :, None] - atoms[:, None, :]), axis=(1, 2)
        )
        atomic_error = max(atomic_error, float(np.max(abs(scores.atomic_crps(y, atoms, weights) - expected))))
    qs = scores.quantiles(np.array([.005, .05, .5, .95, .995]), wb, sb)
    recovered = np.sum(wb[:, None, :] * norm.cdf(qs[:, :, None] / sb[:, None, :]), axis=2)
    quantile_cdf_error = float(np.max(abs(recovered - np.array([.005, .05, .5, .95, .995]))))
    assert crps_error < 2e-8 and ab_error < 2e-8 and mixture_error < 3e-8 and batch_error < 2e-8
    assert w1_error < 3e-8 and atomic_error < 1e-12 and quantile_cdf_error < 1e-12
    log_fixtures = [
        (90., np.log([80.]), np.array([.08]), np.array([1.])),
        (110., np.log([75., 105., 140.]), np.array([.1, .3, .55]), np.array([.2, .5, .3])),
        (12., np.log([15., 60.]), np.array([.7, 1.2]), np.array([.65, .35])),
        (250., np.log([25., 80., 160.]), np.array([.02, .2, .8]), np.array([.1, .55, .35])),
    ]
    log_error = log_tail = log_quad_error = 0.0
    for y, mu, sig, w in log_fixtures:
        ext, qerr, tail = lognormal_quadrature(y, mu, sig, w)
        actual = float(scores.lognormal_crps(y, mu, sig, w))
        log_error = max(log_error, abs(actual - ext))
        log_tail, log_quad_error = max(log_tail, tail), max(log_quad_error, qerr)
        assert actual >= 0 and np.isfinite(actual)
    mu = np.log(np.array([[70., 100.], [25., 60.]]))
    sig = np.array([[.1, .4], [.2, .7]])
    w = np.array([[.3, .7], [.6, .4]])
    prices = np.array([90., 80.])
    values = scores.lognormal_crps(prices, mu, sig, w)
    log_batch_error = max(abs(float(values[i]) - lognormal_quadrature(prices[i], mu[i], sig[i], w[i])[0]) for i in range(2))
    invalid_rejected = 0
    for price, sigma in ((0.0, .3), (10.0, 0.0), (10.0, -.3)):
        try:
            scores.lognormal_crps(price, np.array([0.]), np.array([sigma]), np.array([1.]))
        except ValueError:
            invalid_rejected += 1
    assert invalid_rejected == 3 and log_error < 5e-7 and log_batch_error < 5e-7 and log_tail < 1e-12
    return {
        "normal_mixture_fixtures": len(fixtures), "mixed_alpha_checks": 5 * len(fixtures),
        "CRPS_CDF_integral_max_abs_error": crps_error,
        "a_b_direct_CDF_integral_max_abs_error": ab_error,
        "mixture_CRPS_quadratic_max_abs_error": mixture_error,
        "W1_CDF_integral_max_abs_error": w1_error,
        "batched_CRPS_max_abs_error": batch_error,
        "atomic_direct_pair_sum_max_abs_error": atomic_error,
        "quantile_CDF_max_abs_error": quantile_cdf_error,
        "normal_integrator_max_reported_error": integration_estimate,
        "lognormal_fixtures": len(log_fixtures),
        "lognormal_CDF_integral_max_abs_error": log_error,
        "lognormal_batched_max_abs_error": log_batch_error,
        "lognormal_max_omitted_tail_bound": log_tail,
        "lognormal_integrator_max_reported_error": log_quad_error,
        "lognormal_invalid_nonpositive_inputs_rejected": invalid_rejected,
        "domain": "Positive finite scales, normalized nonnegative weights; centered normal mixtures, finite-mean lognormal mixtures. No arbitrary mean normal-mixture API is claimed.",
    }


def reference_crps_erf(y, w, s):
    """Independent scalar closed form, already cross-checked by CDF integration above."""
    def absnormal(z, sigma):
        t = z / sigma
        return sigma * math.sqrt(2 / math.pi) * math.exp(-.5 * t * t) + z * math.erf(t / math.sqrt(2))
    first = sum(weight * absnormal(y, sigma) for weight, sigma in zip(w, s))
    second = sum(wi * wj * math.sqrt(2 / math.pi) * math.hypot(si, sj) for wi, si in zip(w, s) for wj, sj in zip(w, s))
    return first - .5 * second


def control_checks(scores, cap_fraction):
    h = 5
    signs = np.array(list(product((-1, 1), repeat=h)))
    paths = np.zeros_like(signs, dtype=float)
    for i, row in enumerate(signs):
        total = 0.0
        for j, sign in enumerate(row):
            paths[i, j] = sign * (1.0 + .7 * (total < 0)) / math.sqrt(h)
            total += paths[i, j]
    w, s0, factor = np.array([.2, .5, .3]), np.array([.3, .8, 1.4]), 1.45
    external_w1 = quad(lambda x: abs(float(np.dot(w, norm.cdf(x / (s0 * factor)) - norm.cdf(x / s0)))), -np.inf, 0, epsabs=2e-11)[0] * 2
    # GH integrates the continuation over a numerical reference Gaussian; the
    # source instead applies the analytic variance-sum identity.
    nodes, gh_weights = hermgauss(100)
    gh_weights = gh_weights / math.sqrt(math.pi)
    def continuation(s, rem):
        locations = s + math.sqrt(2 * rem) * nodes
        return float(sum(weight * .5 * (reference_crps_erf(x, w, s0) - reference_crps_erf(x, w, s0 * factor)) for x, weight in zip(locations, gh_weights)))
    results, max_reference_error, max_odd_error = [], 0.0, 0.0
    for fraction in sorted(set((cap_fraction, 1.0))):
        expected_controls, expected_spent, clip_count = [], [], 0
        for path in paths:
            prefix, spent, total = 0.0, 0.0, 0.0
            cap = fraction * external_w1
            for j, r in enumerate(path):
                rem = (h - j - 1) / h
                m = .5 * (continuation(prefix + r, rem) - continuation(prefix - r, rem))
                remaining = max(cap - spent, 0)
                mc = min(remaining, max(-remaining, m))
                clip_count += abs(mc - m) > 1e-11
                total += mc
                spent += abs(mc)
                prefix += r
            expected_controls.append(total)
            expected_spent.append(spent)
        actual, caps, spent = scores.reflected_control(paths, w, s0, factor, cap_fraction=fraction)
        max_reference_error = max(max_reference_error, float(np.max(abs(actual - expected_controls))), float(np.max(abs(spent - expected_spent))))
        assert np.max(spent - caps) < 1e-12 and np.max(abs(actual) - caps) < 1e-12
        # Isolate a current step by zeroing all later steps. Prior spent is then
        # identical for +r and -r, so oddness of the clipped increment is tested.
        for j in range(h):
            before = paths.copy()
            before[:, j:] = 0
            plus, minus = before.copy(), before.copy()
            plus[:, j], minus[:, j] = paths[:, j], -paths[:, j]
            c_before = scores.reflected_control(before, w, s0, factor, cap_fraction=fraction)[0]
            c_plus = scores.reflected_control(plus, w, s0, factor, cap_fraction=fraction)[0]
            c_minus = scores.reflected_control(minus, w, s0, factor, cap_fraction=fraction)[0]
            max_odd_error = max(max_odd_error, float(np.max(abs(c_plus + c_minus - 2 * c_before))))
        for delta in (0., .05, .1):
            probabilities = np.prod((1 + delta * signs) / 2, axis=1)
            mean_c = float(np.dot(probabilities, actual))
            mean_d = float(np.dot(probabilities, spent))
            assert abs(mean_c) <= delta * mean_d + 1e-12
            results.append({"cap_fraction": fraction, "sign_imbalance": delta,
                "E_control": mean_c, "delta_E_spent": delta * mean_d,
                "delta_cap": delta * float(caps[0]), "clipped_step_count_reference": int(clip_count),
                "path_count": len(paths)})
    assert max_reference_error < 2e-8 and max_odd_error < 1e-12
    no_prefix = scores.reflected_control(paths, w, s0, factor, no_prefix=True, cap_fraction=cap_fraction)[0]
    h1 = scores.reflected_control(np.array([[-3.], [-.2], [0.], [.7], [2.]]), w, s0, factor, cap_fraction=cap_fraction)[0]
    same = scores.reflected_control(paths, w, s0, 1.0, cap_fraction=cap_fraction)
    # Mathematically zero; ndtr(-x)=1-ndtr(x) only up to float64 roundoff.
    # The first audit invocation used exact equality and failed at ~1e-16.
    assert np.max(abs(no_prefix)) < 1e-14 and np.max(abs(h1)) < 1e-14
    assert all(np.max(abs(x)) == 0 for x in same)
    # Per-origin factors/scales agree with independently calling one origin.
    batch_s0 = s0[None, :] * np.linspace(.75, 1.2, len(paths))[:, None]
    factors = np.linspace(.65, 1.4, len(paths))
    batched = scores.reflected_control(paths, w, batch_s0, factors, cap_fraction=cap_fraction)
    singleton = [scores.reflected_control(paths[i:i+1], w, batch_s0[i], factors[i], cap_fraction=cap_fraction) for i in range(len(paths))]
    batch_error = max(float(np.max(abs(batched[k] - np.array([row[k][0] for row in singleton])))) for k in range(3))
    assert batch_error < 1e-12
    return {"default_checked_cap_fraction": cap_fraction,
        "Gaussian_continuation_quadrature_max_abs_error": max_reference_error,
        "clipped_increment_oddness_max_abs_error": max_odd_error,
        "h1_zero_center_scale_control_max_abs": float(np.max(abs(h1))),
        "no_prefix_control_max_abs": float(np.max(abs(no_prefix))),
        "batched_vs_single_origin_max_abs_error": batch_error,
        "leveraged_path_enumerations": results}


def scipy_box(u, l):
    scale = max(float(np.max(abs(u))), float(np.max(abs(l))), 1e-30)
    objective = lambda w: float((w @ u @ w - 2 * l @ w) / scale)
    derivative = lambda w: (2 * u @ w - 2 * l) / scale
    candidates = []
    for start in (np.zeros(len(l)), np.full(len(l), .5), np.ones(len(l))):
        fit = minimize(objective, start, jac=derivative, bounds=[(0, 1)] * len(l), method="SLSQP", options={"ftol": 1e-13, "maxiter": 1000})
        if fit.success:
            candidates.append(fit)
    assert candidates
    best = min(candidates, key=lambda f: f.fun)
    return best.x, float(best.fun * scale), scale


def qp_checks(learning):
    rng = np.random.default_rng(SEED + 7)
    normalized_error, maximum_outside, minimum_eigenvalue = 0., 0., 0.
    count = 0
    for d in (1, 2, 3):
        for j in range(18):
            k = rng.normal(size=(d, d))
            u = k @ k.T + np.eye(d) * (0 if j % 3 == 0 else .03)
            if j == 0:
                u = np.ones((d, d))  # A genuinely singular face.
            l = rng.normal(size=d)
            scale = (1e-10, 1., 1e6)[j % 3]
            u, l = u * scale, l * scale
            actual, value = learning.box_quadratic(u, l)
            _, reference, denominator = scipy_box(u, l)
            normalized_error = max(normalized_error, abs(value - reference) / denominator)
            maximum_outside = max(maximum_outside, float(np.max(-actual)), float(np.max(actual - 1)))
            minimum_eigenvalue = min(minimum_eigenvalue, float(np.min(np.linalg.eigvalsh(u))))
            assert value <= 1e-12
            count += 1
    assert normalized_error < 2e-8 and maximum_outside <= 1e-12
    return {"quadratic_cases": count, "dimensions": [1, 2, 3],
        "scipy_reference": "SLSQP with analytic derivative and three starts; objective scaled for numeric conditioning",
        "normalized_objective_max_abs_difference": normalized_error,
        "maximum_box_constraint_excess": maximum_outside,
        "minimum_generated_eigenvalue_roundoff": minimum_eigenvalue}


def manual_block_reference(a, b, c, cap, phi, indices, joint, allowed, delta, confidence):
    """Bootstrap through frequency vectors and independently assembled scalar moments."""
    n, d = phi.shape
    frequency = np.zeros((len(indices), n))
    for i, row in enumerate(indices):
        for index in row:
            frequency[i, int(index)] += 1 / n
    b_moments = [(b - lam * c) * phi[:, k] for lam in joint for k in range(d)]
    pairs = [(i, j) for i in range(d) for j in range(i, d)]
    a_moments = [a * phi[:, i] * phi[:, j] for i, j in pairs]
    k_moments = [cap * phi[:, i] for i in range(d)]
    means, errors, ses = [], [], []
    for values in b_moments + a_moments + k_moments:
        mean = float(sum(values) / n)
        replicas = frequency @ values
        se = math.sqrt(float(sum((replicas - float(np.mean(replicas))) ** 2) / (len(replicas) - 1)))
        means.append(mean)
        ses.append(se)
        errors.append((replicas - mean) / se if se > 1e-14 else np.zeros(len(replicas)))
    means, ses, errors = np.array(means), np.array(ses), np.array(errors).T
    nb, na = len(b_moments), len(a_moments)
    relevant = []
    wrong_relevant = []
    for row in errors:
        relevant.append(max(0., max(row[:nb]), max(abs(row[nb:nb+na])), max(-row[nb+na:])))
        wrong_relevant.append(max(0., max(-row[:nb]), max(abs(row[nb:nb+na])), max(row[nb+na:])))
    rank = math.ceil(confidence * (len(indices) - 1))
    q = sorted(relevant)[rank]
    wrong_q = sorted(wrong_relevant)[rank]
    raw_lower = (means[:nb] - q * ses[:nb]).reshape(len(joint), d)
    k_upper = np.maximum(0, means[nb+na:] + q * ses[nb+na:])
    lower_all = raw_lower - abs(np.array(joint))[:, None] * delta * k_upper
    choice = np.array([max(allowed, key=lambda j: (lower_all[j, k], -j)) for k in range(d)])
    lower = lower_all[choice, np.arange(d)]
    center, radius = np.zeros((d, d)), np.zeros((d, d))
    for z, (i, j) in enumerate(pairs):
        center[i, j] = center[j, i] = means[nb+z]
        radius[i, j] = radius[j, i] = q * ses[nb+z]
    upper = center + np.diag(radius.sum(axis=1))
    return q, wrong_q, lower, k_upper, upper, radius, np.array(joint)[choice]


def bootstrap_checks(learning):
    rng = np.random.default_rng(SEED + 11)
    n, d, repeats = 37, 3, 399
    phi = rng.dirichlet([.7, 1.3, 2.0], size=n)
    a = rng.uniform(.01, .2, n)
    b = rng.normal(.02, .1, n)
    cap = rng.uniform(.01, .08, n)
    c = np.tanh(rng.normal(size=n)) * cap
    indices = learning.circular_indices(n, 5, repeats, rng)
    assert indices.shape == (repeats, n)
    for start in range(0, n, 5):
        chunk = indices[:, start:min(start+5, n)]
        assert np.all(np.diff(chunk, axis=1) % n == 1)
    joint = (0., .5, 1.)
    actual_w, receipt = learning.block_gate(a, b, c, cap, phi, indices, sign_budget=.05, confidence=.95)
    ref = manual_block_reference(a, b, c, cap, phi, indices, joint, list(range(3)), .05, .95)
    q, _, lower, k_upper, upper, radius, chosen = ref
    differences = [abs(q - receipt["bootstrap_joint_quantile"]),
        float(np.max(abs(lower - receipt["lower_B"]))), float(np.max(abs(k_upper - receipt["upper_K_basis"]))),
        float(np.max(abs(upper - receipt["upper_A"]))) ]
    assert np.allclose(chosen, receipt["selected_lambdas"], atol=0, rtol=0)
    _, ref_value, objective_scale = scipy_box(upper, lower)
    differences.append(abs(float(actual_w @ upper @ actual_w - 2 * lower @ actual_w) - ref_value) / objective_scale)
    _, baseline_receipt = learning.block_gate(a, b, c, cap, phi, indices, sign_budget=.05, lambdas=(0.,), joint_lambdas=joint, confidence=.95)
    assert baseline_receipt["bootstrap_joint_quantile"] == receipt["bootstrap_joint_quantile"]
    assert baseline_receipt["selected_lambdas"] == [0.] * d
    # Exhaust all symmetric sign corners of the elementwise A confidence box.
    pairs = [(i, j) for i in range(d) for j in range(i, d)]
    minimum_margin = math.inf
    diagonal = np.diag(radius.sum(axis=1))
    for signs in product((-1., 1.), repeat=len(pairs)):
        error = np.zeros_like(radius)
        for sign, (i, j) in zip(signs, pairs):
            error[i, j] = error[j, i] = sign * radius[i, j]
        minimum_margin = min(minimum_margin, float(np.min(np.linalg.eigvalsh(diagonal - error))))
    assert minimum_margin > -1e-12
    # Asymmetric one-outlier fixtures make flipping B/K confidence directions detectable.
    sign_fixtures = []
    for label in ("B_lower", "K_upper"):
        n = 21
        aa, bb, cc, kk = np.ones(n) * .04, np.ones(n) * .02, np.zeros(n), np.ones(n) * .1
        if label == "B_lower":
            bb = np.zeros(n); bb[0] = 2.
        else:
            kk = np.zeros(n); kk[0] = 2.
        ph = np.ones((n, 1))
        ix = learning.circular_indices(n, 1, 1999, rng)
        _, rec = learning.block_gate(aa, bb, cc, kk, ph, ix, confidence=.95)
        right, wrong, *_ = manual_block_reference(aa, bb, cc, kk, ph, ix, joint, list(range(3)), .05, .95)
        assert abs(right - wrong) > .1
        assert abs(rec["bootstrap_joint_quantile"] - right) < 1e-10
        sign_fixtures.append({"fixture": label, "correct_quantile": right, "incorrect_reversed_quantile": wrong})
    assert max(differences) < 2e-8 and receipt["nominal_only"] is True
    return {"manual_frequency_bootstrap_max_abs_difference": max(differences),
        "joint_raw_baseline_uses_identical_quantile": True,
        "Gershgorin_all_symmetric_corner_count": 2 ** len(pairs),
        "Gershgorin_minimum_eigenvalue_margin": minimum_margin,
        "asymmetric_direction_fixtures": sign_fixtures,
        "scope": "Correct signs and deterministic construction checked. Circular-block bootstrap remains approximate; no iid/distribution-free coverage theorem is inferred."}


def bernstein_checks(learning):
    rng = np.random.default_rng(SEED + 13)
    b = rng.uniform(-.08, .12, size=(8, 64))
    c = rng.uniform(-.03, .03, size=(8, 64))
    a, cap, rb, delta, sign = .04, .03, .2, .05, .1
    grid = (0., .5, 1.)
    expected = []
    for lam in grid:
        values = b - lam * c
        means = values.mean(axis=1)
        variance = np.sum((values - means[:, None]) ** 2, axis=1) / 63
        logterm = math.log(2 / (delta / len(grid)))
        bound = np.sqrt(2 * variance * logterm / 64) + 7 * (rb + 2 * abs(lam) * cap) * logterm / (3 * 63)
        expected.append(means - bound - abs(lam) * sign * cap)
    all_lower = np.stack(expected, axis=1)
    weights, selection = learning.bernstein_scalar(a, b, c, cap, rb, sign, lambdas=grid, delta=delta)
    expected_weights = np.clip(all_lower.max(axis=1) / a, 0, 1)
    expected_selection = np.array(grid)[all_lower.argmax(axis=1)]
    error = float(np.max(abs(weights - expected_weights)))
    assert error < 1e-12 and np.array_equal(selection, expected_selection)
    zero_range_bound = float(learning.empirical_bernstein(np.zeros(64), 0, .025))
    assert zero_range_bound == 0
    return {"independent_formula_max_abs_difference": error, "constant_zero_range_bound": zero_range_bound}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap-fraction", type=float, default=.25)
    args = parser.parse_args()
    if not 0 < args.cap_fraction <= 1:
        raise ValueError("Audit cap fraction must be in (0,1]")
    source_hashes = {name: digest(SOURCE / name) for name in ("scores.py", "learning.py")}
    scores, learning = load_source("scores"), load_source("learning")
    tests = {"scores": lambda: score_checks(scores), "controls": lambda: control_checks(scores, args.cap_fraction),
        "box_quadratic": lambda: qp_checks(learning), "block_bootstrap": lambda: bootstrap_checks(learning),
        "empirical_bernstein": lambda: bernstein_checks(learning)}
    results, failures = {}, []
    for name, fn in tests.items():
        try:
            results[name] = {"status": "passed", **fn()}
        except Exception as exc:
            results[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            failures.append(name)
    hashes_after = {name: digest(SOURCE / name) for name in source_hashes}
    if hashes_after != source_hashes:
        failures.append("source_changed_during_audit")
    output = {"metadata": {"date": "2026-10-19", "seed": SEED, "cap_fraction": args.cap_fraction,
        "script_sha256": digest(Path(__file__)), "source_hashes": source_hashes,
        "source_hashes_after": hashes_after, "numpy_version": np.__version__,
        "market_rows_read": 0, "predictive_model_fits": 0},
        "all_passed": not failures, "failures": failures, "checks": results,
        "audit_development_record": {
            "initial_attempt": "Four groups passed; controls stopped at an exact-zero assertion for theoretical h1/no-prefix controls.",
            "diagnosis": "The source normal-CDF arithmetic leaves float64 residuals of order 1e-16; a separate h1 probe peaked at 3.3306690738754695e-17 and no-prefix probe at 1.221245327087672e-16.",
            "audit_only_correction": "Use an explicit 1e-14 absolute tolerance for the theoretical zero-control assertions; keep the external quadrature, oddness, cap and bias tolerances unchanged.",
            "source_modified_by_this_auditor": False,
            "initial_script_sha256": "8764b3838d98a14c930e766a46a80ba11e565b2203ad175ed993aad72e618d36",
        },
        "mathematical_review": {
            "B_lower_bootstrap_direction": "replica_minus_empirical_mean; lower bound empirical_mean - q*SE",
            "K_upper_bootstrap_direction": "empirical_mean_minus_replica; upper bound empirical_mean + q*SE",
            "A_upper": "Two-sided entrywise radii e_ij, symmetric; Ahat+diag(row_sum(e)) dominates the entire entrywise symmetric error box.",
            "cap": "Predictable remaining amplitude and symmetric clipping preserve oddness; sum absolute increments <= cap.",
            "bias": "Requires |E[sign(r_j) | abs(r_j), past]|<=delta. Ordinary reflection-TV alone is insufficient for the delta*cap bound.",
            "range": "For same-center symmetric F0/F1, osc(B)<=W1/2, each reflection amplitude<=W1/4. Choosing total cap W1/4 is valid but can truncate multiple contributions; W1 cap remains a meaningful fixed ablation.",
            "reference": "Code uses per-step reference variance 1/h and terminal reference variance 1; correctness of the reference law is not required for oddness.",
            "scope": "Circular-block bootstrap is only an empirical time-series adaptation. Algebraic signs do not establish finite-sample coverage under dependence or nonstationarity.",
        }}
    json_path = HERE / "INDEPENDENT_CORE_CHECK.json"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = ["# 独立核心数值与数学审计", "", "日期：2026-10-19。仅合成数值；不读市场，不拟合预测模型。", "",
        f"总体状态：{'通过' if output['all_passed'] else '存在失败'}。默认核验cap_fraction={args.cap_fraction}，并比较cap_fraction=1消融。", "",
        "## 检查方法", "",
        "GMM CRPS直接积分CDF平方定义；a与b分别直接积分，不用源代码的自距离恒等式作唯一参考。混合alpha对照组合CDF数值积分。原子CRPS对照完整双重绝对距离和。", "",
        "对数正态CRPS在正实数支持上用log坐标积分CDF定义，报告积分误差及省略尾部的显式上界；另外检查正价格/正尺度和有限均值。", "",
        "反射控制用数值Gaussian continuation积分与独立标量实现核验；枚举32条幅度依赖真实过去符号的路径，逐步隔离clip后的奇性，并检验总幅度、强符号偏差预算、h1与no-prefix零控制。", "",
        "box-QP对照scipy SLSQP，包含PSD奇异矩阵、不同尺度及1/2/3维；只比较应唯一的目标值，不强求非唯一最优权重相同。", "",
        "bootstrap用独立重采样频率矩阵重建统计量；偏斜单异常值fixture使B或K的符号反转产生不同临界值。矩阵上界检查全部64个对称误差盒顶点。", "",
        "## 结果", "", "```json", json.dumps(results, ensure_ascii=False, indent=2), "```", "",
        "## 数学审查", "",
        "审计开发记录：第一轮其余四组通过，控制组使用浮点精确等号检查理论零值，在约1e−16的normal-CDF舍入残差处失败。定位后仅将审计该断言改为1e−14绝对容差，源代码未改，其余容差未放宽。重跑为审计修正，不计新的独立市场实验。", "",
        "B下界的bootstrap方向为bootstrap mean−empirical mean；K上界方向相反。A使用双侧误差半宽e，逐元素误差盒通过行和对角项得到Loewner上界。源码方向正确；该结论不把block bootstrap升级为有限样本分布自由保证。", "",
        "predictable remaining cap加对称clip保持当前增量奇性。δ×cap的偏差界依赖给定当前幅度与过去的条件符号偏差假设；普通反射TV不够。K(X)变化时总体上界涉及E K(X)，源码block_gate对cap×basis另建upper_K，这是正确处理方向。", "",
        "F0/F1同中心对称时，osc(B)≤W1/2，单步reflection绝对值≤W1/4；把整个路径总预算设为W1/4是合法而更紧的设计，但会截断多个贡献，不是无需实验的最优性定理。W1整额消融应保留。", "",
        "参考方差代码为(h−j−1)/h，表示每步方差1/h、终点总参考方差1。若注释仍写unit-variance Gaussian increments，应改成上述精确表述；这不影响奇性。", "",
        "适用域：正有限尺度、非负归一混合权重。当前GMM API是零中心而非任意混合均值。积分/优化核验只覆盖所列数值域，不能代替后续pipeline对非有限值和输入约束的硬检查。", "",
        "## 文件身份", "", "```json", json.dumps(output["metadata"], ensure_ascii=False, indent=2), "```", "",
        "集中界来源：Maurer & Pontil (2009), Theorem4 / Corollary5, https://arxiv.org/pdf/0907.3740 。其他数学恒等式由本次外部积分与代数审计交叉核验。", "",
    ]
    (HERE / "INDEPENDENT_CORE_CHECK_ZH.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"all_passed": output["all_passed"], "failures": failures, "json": str(json_path)}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
