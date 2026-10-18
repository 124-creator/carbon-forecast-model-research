#!/usr/bin/env python3
"""Independent mathematical checks; imports no project model or market data.

Usage: python check_random_clearing_math.py --output-dir NEW_DIRECTORY
The output directory must not exist, so previous receipts are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import betainc, gammainc, gammaincc, logsumexp
from scipy.stats import gamma


def gamma_cdf_positive_s(s, shape, scale):
    shape = np.asarray(shape, dtype=np.float64)
    out = np.ones_like(shape)
    mask = shape > 0
    out[mask] = gammainc(shape[mask], s / scale)
    return out


def clearing_mass(alpha, scale, supply):
    alpha = np.asarray(alpha, dtype=np.float64)
    included = np.cumsum(alpha[::-1])[::-1]
    strictly_above = np.concatenate([included[1:], [0.0]])
    mass = gamma_cdf_positive_s(supply, strictly_above, scale) - gamma_cdf_positive_s(
        supply, included, scale
    )
    fail = float(gammainc(alpha.sum(), supply / scale))
    return mass, fail


def realized_clearing_index(weights, supply):
    weights = np.asarray(weights)
    total = weights.sum(axis=1)
    success = total >= supply
    desc = np.cumsum(weights[:, ::-1], axis=1)
    index = weights.shape[1] - 1 - np.argmax(desc >= supply, axis=1)
    index = np.where(success, index, -1)
    return index, success


def exact_clearing(prices, volumes, supply):
    cumulative = Fraction(0)
    for price, volume in sorted(zip(prices, volumes), reverse=True):
        cumulative += Fraction(volume)
        if cumulative >= Fraction(supply):
            return price
    return None


def conditional_mass_given_b(alpha, supply, observed_b):
    alpha = np.asarray(alpha)
    total = alpha.sum()
    ratio = supply / observed_b
    if not 0 < ratio < 1:
        raise ValueError("This independent Beta check uses observed_b > supply > 0")
    included = np.cumsum(alpha[::-1])[::-1]
    above = np.concatenate([included[1:], [0.0]])

    def tail_cdf(a):
        if a <= 0:
            return 1.0
        if a >= total - 1e-12:
            return 0.0
        return float(betainc(a, total - a, ratio))

    return np.array([tail_cdf(a) - tail_cdf(b) for a, b in zip(above, included)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--mc-samples", type=int, default=400000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {args.output_dir}")
    if args.mc_samples < 10000:
        raise SystemExit("Use at least 10000 Monte Carlo samples.")
    rng = np.random.default_rng(args.seed)
    checks = {}

    def record(name, passed, **evidence):
        checks[name] = {"pass": bool(passed), **evidence}

    # Strict quantile boundary; exact rational arithmetic prevents floating ambiguity.
    prices = [10, 20, 30]
    supply = 1
    u = Fraction(2, 3)
    regular_q = next(p for i, p in enumerate(prices, 1) if Fraction(i, 3) >= u)
    strict_q = next(p for i, p in enumerate(prices, 1) if Fraction(i, 3) > u)
    actual = exact_clearing(prices, [1, 1, 1], supply)
    record("strict_inverse_boundary", actual == strict_q == 30 and regular_q == 20,
           actual=actual, strict_inverse=strict_q, usual_inverse=regular_q)

    book_a = [10, 12, 14, 15, 16, 18, 20]
    book_b = [10, 11, 13, 15, 17, 19, 20]
    stats_a = [min(book_a), max(book_a), sum(book_a)/7, float(np.median(book_a))]
    stats_b = [min(book_b), max(book_b), sum(book_b)/7, float(np.median(book_b))]
    original = [exact_clearing(b, [1000]*7, 3500) for b in (book_a, book_b)]
    changed = [exact_clearing(b, [1000]*7, 1500) for b in (book_a, book_b)]
    record("weighted_summary_nonidentification", stats_a == stats_b and original == [15, 15]
           and changed == [18, 19], summary_min_max_mean_median=stats_a,
           same_supply_clearing=original, changed_supply_clearing=changed)
    unweighted = [exact_clearing([10,20,30], v, 3) for v in ([4,1,1], [1,1,4])]
    record("count_weighted_summary_nonidentification", unweighted == [10,30],
           same_price_summary_and_total_volume=True, clearing=unweighted)

    alpha = np.array([[0.4,1.5,0.8,0.5,0.2], [1.0,2.0,0.7,1.3,0.6],
                      [0.3,0.4,0.5,0.7,1.2]], dtype=np.float64)
    scale = np.array([0.35, 0.9, 2.2])
    pi = np.array([0.2, 0.5, 0.3])
    supply = 3.0
    masses, fails = zip(*(clearing_mass(a, t, supply) for a,t in zip(alpha, scale)))
    masses, fails = np.array(masses), np.array(fails)
    per_component_error = np.abs(masses.sum(axis=1) + fails - 1)
    joint = pi[:,None] * masses
    success_prob = float(joint.sum())
    mixture_fail = float(pi @ fails)
    record("nonnegative_and_normalized", masses.min() >= -1e-14 and
           per_component_error.max() < 1e-12 and abs(success_prob + mixture_fail - 1) < 1e-12,
           minimum_raw_mass=float(masses.min()), max_component_error=float(per_component_error.max()),
           success_probability=success_prob, failure_probability=mixture_fail)

    actual_component_weights = joint.sum(axis=1) / success_prob
    wrong_component_weights = pi
    record("success_reweights_mixture", np.max(np.abs(actual_component_weights-pi)) > 0.05,
           original_pi=pi.tolist(), correct_success_weights=actual_component_weights.tolist(),
           max_difference_if_original_pi_used=float(np.max(np.abs(actual_component_weights-pi))))

    scaled = np.array([clearing_mass(a, t*1000, supply*1000)[0] for a,t in zip(alpha,scale)])
    record("volume_unit_invariance", np.allclose(scaled, masses, atol=2e-14, rtol=0),
           max_absolute_error=float(np.max(np.abs(scaled-masses))))

    n = args.mc_samples
    component = rng.choice(3, size=n, p=pi)
    weights = np.empty((n,5), dtype=np.float64)
    for k in range(3):
        mask = component == k
        weights[mask] = rng.gamma(alpha[k], scale[k], size=(int(mask.sum()),5))
    index, mechanical_success = realized_clearing_index(weights, supply)
    empirical = np.bincount(component[mechanical_success]*5+index[mechanical_success], minlength=15).reshape(3,5)/n
    mc_bound = 6*np.sqrt(joint*(1-joint)/n)+3/n
    mc_error = np.abs(empirical-joint)
    record("gamma_operator_vs_independent_monte_carlo", np.all(mc_error <= mc_bound),
           samples=n, max_absolute_mass_error=float(mc_error.max()),
           max_error_over_six_sigma_plus_three_counts=float(np.max(mc_error/mc_bound)),
           empirical_success=float(mechanical_success.mean()), analytic_success=success_prob)

    index_more, success_more = realized_clearing_index(weights, supply+1.0)
    record("pathwise_supply_monotonicity_on_shared_success", np.all(index_more[success_more] <= index[success_more]),
           shared_success_count=int(success_more.sum()),
           statement="Only the same demand realization and both supplies mechanically successful; no conditional-mean monotonicity claim.")

    # A global admission constant cannot change conditional price probabilities.
    rho = 0.97
    admitted = rho*joint
    record("constant_admission_cancels_from_price_law",
           np.allclose(admitted/admitted.sum(), joint/success_prob, atol=2e-14) and
           abs(admitted.sum() + (1-admitted.sum())-1) < 1e-14,
           max_conditional_mass_error=float(np.max(np.abs(admitted/admitted.sum()-joint/success_prob))),
           working_cancel_probability=float(1-admitted.sum()))

    # History-based residual grids differ by component; all reference means equal lastprice.
    z = np.array([-2.0,-0.8,0.0,0.7,2.3])
    spans = np.array([0.5,1.3,2.0])
    reference_mean = (masses @ z) / masses.sum(axis=1)
    grid = 42.0 + 1.7*spans[:,None]*(z[None,:]-reference_mean[:,None])
    conditional_mean = float(np.sum(joint*grid)/success_prob)
    component_means = np.sum(masses*grid,axis=1)/masses.sum(axis=1)
    wrong_grid = 42.0 + 1.7*(spans[:,None]*z[None,:]-reference_mean[:,None])
    wrong_mean = float(np.sum(joint*wrong_grid)/success_prob)
    record("span_scaled_reference_centering_rw_identity", abs(conditional_mean-42)<1e-12 and
           np.max(np.abs(component_means-42))<1e-12 and abs(wrong_mean-42)>1e-3,
           component_means=component_means.tolist(), correct_mixture_mean=conditional_mean,
           mean_under_unscaled_reference_bug=wrong_mean,
           note="Reference and future distributions here match; no price-dependent screening.")

    # The implemented positive-price version uses exponential MOMENT centering,
    # not centering of log-price means. Exactness is claimed in the unclipped interior.
    r = 0.07
    ref_weights = masses/masses.sum(axis=1,keepdims=True)
    scaled_z = r*spans[:,None]*z[None,:]
    ref_log_moment = logsumexp(np.log(ref_weights)+scaled_z,axis=1)
    log_ratio = scaled_z-ref_log_moment[:,None]
    positive_prices = 42*np.exp(log_ratio)
    positive_means = np.sum(ref_weights*positive_prices,axis=1)
    positive_mix_mean = float(np.sum(joint*positive_prices)/success_prob)
    shifted_prices = 42*np.exp(log_ratio+r*0.12)
    shifted_means = np.sum(ref_weights*shifted_prices,axis=1)
    expected_shifted = float(42*np.exp(r*0.12))
    record("positive_exponential_moment_rw_anchor_interior", np.all(positive_prices>0) and
           np.max(np.abs(log_ratio))<15 and np.max(np.abs(positive_means-42))<1e-12 and
           abs(positive_mix_mean-42)<1e-12 and np.max(np.abs(shifted_means-expected_shifted))<1e-12,
           component_means=positive_means.tolist(), mixture_mean=positive_mix_mean,
           minimum_positive_price=float(positive_prices.min()),
           maximum_absolute_log_ratio=float(np.max(np.abs(log_ratio))),
           expected_mean_with_location_012=expected_shifted,
           clipping_note="No log-ratio clip or weight floor active in this algebraic check; no root-model autograd claim.")

    # Conditional distribution from the independent Dirichlet representation.
    cond_alpha = alpha[1]
    observed_b = 7.0
    beta_mass = conditional_mass_given_b(cond_alpha, supply, observed_b)
    dirichlet_draws = rng.dirichlet(cond_alpha, size=n)
    cond_index, cond_success = realized_clearing_index(observed_b*dirichlet_draws, supply)
    cond_emp = np.bincount(cond_index, minlength=5)/n
    cond_bound = 6*np.sqrt(beta_mass*(1-beta_mass)/n)+3/n
    record("conditional_beta_formula_vs_dirichlet_monte_carlo", np.all(cond_success) and
           np.all(np.abs(beta_mass-cond_emp) <= cond_bound),
           max_absolute_mass_error=float(np.max(np.abs(beta_mass-cond_emp))),
           analytic_mass=beta_mass.tolist(), empirical_mass=cond_emp.tolist())

    # Integrating the conditional Beta likelihood recovers the Gamma clearing mass.
    integrated = []
    for j in range(5):
        def integrand(b):
            return float(gamma.pdf(b, a=cond_alpha.sum(), scale=scale[1]) *
                         conditional_mass_given_b(cond_alpha, supply, b)[j])
        value, _ = quad(integrand, supply, np.inf, epsabs=2e-10, epsrel=2e-9, limit=200)
        integrated.append(value)
    integration_error = float(np.max(np.abs(np.array(integrated)-masses[1])))
    record("joint_beta_gamma_integral_matches_marginal", integration_error < 2e-8,
           max_absolute_mass_error=integration_error)

    # Exact discrete CRPS remains valid outside the support, without clipping observations.
    atoms = grid.reshape(-1)
    probs = (joint/success_prob).reshape(-1)
    y = float(atoms.max()+3)
    crps = float(probs@np.abs(atoms-y)-0.5*np.sum(probs[:,None]*probs[None,:]*np.abs(atoms[:,None]-atoms[None,:])))
    lo, hi = float(min(atoms.min(),y)), float(max(atoms.max(),y))
    breaks = sorted(set([lo,hi,y,*atoms.tolist()]))
    direct_integral = 0.0
    for left,right in zip(breaks[:-1],breaks[1:]):
        midpoint=(left+right)/2
        cdf=float(probs[atoms<=midpoint].sum())
        direct_integral+=(right-left)*(cdf-float(y<=midpoint))**2
    record("crps_energy_equals_full_cdf_integral_outside_support", abs(crps-direct_integral)<1e-11,
           energy_crps=crps, full_integral_crps=direct_integral, observation_outside_support=True)

    # A single fixed-supply Gamma operator is a reparameterization of any positive
    # categorical law; this is an important novelty limitation, not a superiority test.
    target_mass = np.array([0.03, 0.17, 0.37, 0.11, 0.32])
    total_shape, fixed_scale, fixed_supply = 4.2, 1.1, 2.8
    base_cdf = float(gammainc(total_shape, fixed_supply/fixed_scale))
    target_cdf = np.cumsum(target_mass)
    cumulative_shapes = [total_shape]
    for probability in target_cdf[:-1]:
        desired = base_cdf+(1-base_cdf)*probability
        def root(a):
            return (1.0 if a <= 0 else float(gammainc(a, fixed_supply/fixed_scale)))-desired
        cumulative_shapes.append(brentq(root, 0.0, total_shape, xtol=1e-13))
    cumulative_shapes.append(0.0)
    equivalent_alpha = -np.diff(cumulative_shapes)
    recovered, _ = clearing_mass(equivalent_alpha, fixed_scale, fixed_supply)
    recovered /= recovered.sum()
    categorical_error = float(np.max(np.abs(recovered-target_mass)))
    record("fixed_supply_categorical_reparameterization", categorical_error < 1e-11,
           target_mass=target_mass.tolist(), recovered_mass=recovered.tolist(),
           equivalent_gamma_shapes=equivalent_alpha.tolist(), max_absolute_error=categorical_error,
           implication="No new single-supply categorical expressivity; possible differences are shared parameterization, joint laws, capacity and learning.")

    receipt = {
        "status": "PASS" if all(c["pass"] for c in checks.values()) else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent deterministic identities and simulated Gamma/Dirichlet orders only; no market data, no training, no project imports.",
        "not_validated": ["root model implementation", "autograd and custom special-function gradients", "real market fitting", "economic identification", "predictive superiority", "screening-rule identification"],
        "seed": args.seed,
        "mc_samples": n,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": checks,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"MATH_CHECK_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"status":receipt["status"], "checks":len(checks), "receipt":str(args.output_dir/"MATH_CHECK_RECEIPT.json")}, ensure_ascii=False))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
