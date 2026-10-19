#!/usr/bin/env python3
"""Independent finite-state / CDF-integral audit; never fits any parameter.

Production propagation is compared with complete 3**h state enumeration.
The audit does not read market prices or invoke a training/optimizer function.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp, ndtr
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "persistence.py"
PROTOCOL = ROOT / "provenance" / "PROTOCOL.json"
OUT = ROOT / "audits" / "INDEPENDENT_PERSISTENCE_AUDIT.json"
Q = np.array([0.25, 1.0, 4.0])
PI = np.array([0.45, 0.40, 0.15])
SHIFTS = np.array([0, 1, 5])


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_production():
    spec = importlib.util.spec_from_file_location("persistence_under_audit", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def states_enumerated(h, rho, pi=PI):
    states = np.asarray(list(itertools.product(range(3), repeat=h)), dtype=int)
    transition = rho * np.eye(3) + (1 - rho) * pi[None, :]
    masses = pi[states[:, 0]].copy()
    for j in range(1, h):
        masses *= transition[states[:, j-1], states[:, j]]
    # r_j has unit marginal variance. The scored endpoint is D_h / sqrt(h).
    state_variances = Q[states] / np.dot(pi, Q)
    endpoint_variances = state_variances.sum(1) / h
    lattice = SHIFTS[states].sum(1)
    return states, masses, state_variances, endpoint_variances, lattice


def mixture_cdf(value, masses, variances):
    return float(np.dot(masses, ndtr(value / np.sqrt(variances))))


def mixture_sf(value, masses, variances):
    return float(np.dot(masses, ndtr(-value / np.sqrt(variances))))


def score_by_cdf(y, masses, variances):
    lower, lower_err = quad(
        lambda z: mixture_cdf(z, masses, variances) ** 2,
        -np.inf, y, epsabs=1e-11, epsrel=1e-11, limit=160)
    upper, upper_err = quad(
        lambda z: mixture_sf(z, masses, variances) ** 2,
        y, np.inf, epsabs=1e-11, epsrel=1e-11, limit=160)
    return lower + upper, lower_err + upper_err


def expected_score_by_cdf(predictive, truth):
    wp, vp = predictive
    wt, vt = truth

    def integrand(z):
        f = mixture_cdf(z, wp, vp)
        g = mixture_cdf(z, wt, vt)
        sf_g = mixture_sf(z, wt, vt)
        return (f - g) ** 2 + g * sf_g

    answer, err = quad(integrand, -np.inf, np.inf,
                       epsabs=1e-11, epsrel=1e-11, limit=160)
    return answer, err


def enumerated_nll(paths, rho, pi=PI):
    n, h = paths.shape
    states, weights, _, _, _ = states_enumerated(h, rho, pi)
    keep = weights > 0
    states, weights = states[keep], weights[keep]
    emissions_sd = np.sqrt(Q / np.dot(pi, Q))
    answers = []
    for path in paths:
        # Independent complete-path log summation; no forward recursion.
        log_density = np.log(weights)
        for j in range(h):
            log_density += norm.logpdf(path[j], scale=emissions_sd[states[:, j]])
        answers.append(-float(logsumexp(log_density)) / h)
    return np.asarray(answers)


def require_close(actual, expected, tolerance, label):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape:
        raise AssertionError(f"{label}: shape {a.shape} != {b.shape}")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise AssertionError(f"{label}: non-finite quantity")
    error = float(np.max(np.abs(a-b))) if a.size else 0.0
    if error > tolerance:
        raise AssertionError(f"{label}: absolute error {error} > {tolerance}")
    return error


def check_enumeration_and_moments(module):
    pis = [PI, np.array([.2, .5, .3])]
    rhos = [0., .1, .5, .9, 1.]
    worst_weight = worst_variance = worst_kurtosis = worst_covariance = 0.0
    cases = pairs = 0
    for pi in pis:
        raw_mean = float(pi @ Q)
        q_variance = float(pi @ Q**2 - raw_mean**2)
        for rho in rhos:
            predictions = module.propagate(rho, horizons=(1,2,3,4,5,20), pi=pi)
            for h in range(1, 6):
                _, masses, daily_variance, endpoint_variance, lattice = states_enumerated(h, rho, pi)
                exact_w = np.bincount(lattice, weights=masses, minlength=5*h+1)
                w, sd = predictions[h]
                worst_weight = max(worst_weight, require_close(
                    w[0], exact_w, 2e-13, "DP vs all state sequences"))
                expected_variance = np.array([1.0])
                actual_variance = np.array([np.sum(w[0]*sd**2)])
                worst_variance = max(worst_variance, require_close(
                    actual_variance, expected_variance, 2e-13, "unit endpoint variance"))
                exact_kurtosis = 3*float(np.dot(masses, endpoint_variance**2))-3
                analytic_kurtosis = (3*q_variance/(h*h*raw_mean**2)
                    * (h+2*sum((h-k)*rho**k for k in range(1,h))))
                worst_kurtosis = max(worst_kurtosis, require_close(
                    np.array([exact_kurtosis]), np.array([analytic_kurtosis]),
                    4e-13, "enumerated fourth moment"))
                for i in range(h):
                    for j in range(i+1,h):
                        actual_cov = float(np.dot(masses, daily_variance[:,i]*daily_variance[:,j])-1)
                        expected_cov = q_variance/raw_mean**2*rho**(j-i)
                        worst_covariance = max(worst_covariance, require_close(
                            np.array([actual_cov]), np.array([expected_cov]),
                            4e-13, "enumerated squared-return covariance"))
                        pairs += 1
                if np.any(w < -1e-15) or abs(float(w.sum())-1) > 2e-13:
                    raise AssertionError("Invalid probability mass")
                cases += 1
            w, sd = predictions[20]
            h = 20
            if np.any(w < -1e-15) or abs(float(w.sum())-1) > 2e-13:
                raise AssertionError("H20 probability failure")
            expected_kurt = 3*q_variance/(h*h*raw_mean**2)*(h+2*sum((h-k)*rho**k for k in range(1,h)))
            worst_kurtosis = max(worst_kurtosis, require_close(
                np.array([3*np.sum(w[0]*sd**4)-3]), np.array([expected_kurt]),
                4e-13, "H20 DP fourth moment"))
    return dict(enumerated_cases=cases, covariance_pairs=pairs,
                maximum_weight_difference=worst_weight,
                maximum_variance_difference=worst_variance,
                maximum_kurtosis_difference=worst_kurtosis,
                maximum_covariance_difference=worst_covariance,
                h20_moment_cases=len(pis)*len(rhos))


def check_scores(module):
    worst_score = worst_risk = worst_integral_error = 0.0
    score_cases = risk_cases = 0
    for rho in [0., .37, 1.]:
        prediction = module.propagate(rho, horizons=(1,2,5))
        for h in [1,2,5]:
            _, masses, _, variance, _ = states_enumerated(h, rho)
            w, sd = prediction[h]
            for y in [-4., -.7, 0., 1.3]:
                exact, integration_error = score_by_cdf(y, masses, variance)
                actual = module.crps(np.array([y]), w, sd)
                worst_score = max(worst_score, require_close(
                    actual, np.array([exact]), 8e-10, "CRPS CDF integral"))
                worst_integral_error = max(worst_integral_error, integration_error)
                score_cases += 1
    for rho, truth_rho in [(.19,.84), (.8,.1), (.5,.5)]:
        actual = module.population_risk([rho], [truth_rho], horizons=(1,2,5))[0]
        for index,h in enumerate([1,2,5]):
            _, wp, _, vp, _ = states_enumerated(h,rho)
            _, wt, _, vt, _ = states_enumerated(h,truth_rho)
            exact, integration_error = expected_score_by_cdf((wp,vp),(wt,vt))
            worst_risk = max(worst_risk, require_close(
                np.array([actual[index]]), np.array([exact]), 8e-10,
                "expected CRPS external CDF integral"))
            worst_integral_error = max(worst_integral_error, integration_error)
            risk_cases += 1
    return dict(cdf_integral_score_cases=score_cases,
                expected_score_integral_cases=risk_cases,
                maximum_crps_difference=worst_score,
                maximum_expected_crps_difference=worst_risk,
                maximum_quadrature_error_estimate=worst_integral_error)


def check_likelihood(module):
    maximum = 0.0
    stable_maximum = 0.0
    emission_difference = 0.0
    cases = 0
    for h in [1,2,3,4,5]:
        paths = np.stack([np.linspace(-1.7,1.1,h),
                          np.linspace(.2,2.3,h),
                          np.zeros(h),
                          np.linspace(-3.2,.1,h)])
        density = module.path_emissions(paths)
        expected_density = norm.pdf(paths[:,:,None], scale=np.sqrt(Q/(PI@Q)))
        emission_difference = max(emission_difference, require_close(
            density, expected_density, 1e-14, "emission density"))
        for rho in [0.,.17,.6,.94,1.]:
            expected = enumerated_nll(paths,rho)
            actual = module.path_nll_from_emissions(density,rho)
            maximum = max(maximum, require_close(actual,expected,4e-12,
                                                  "forward vs enumerated NLL"))
            stable_actual = module.path_nll_from_emissions(
                module.path_emissions(paths,stable=True),rho)
            stable_maximum = max(stable_maximum, require_close(
                stable_actual,expected,4e-12,"shifted-emission forward vs enumerated NLL"))
            cases += len(paths)
    return dict(path_likelihood_cases=cases,
                maximum_emission_difference=emission_difference,
                maximum_nll_difference=maximum,
                stable_path_likelihood_cases=cases,
                maximum_stable_nll_difference=stable_maximum)


def check_autograd(module):
    import torch

    rho = torch.tensor([.13,.51,.89], dtype=torch.float64, requires_grad=True)
    pred = module.torch_propagate(rho, horizons=(1,2,5,20))
    numpy_pred = module.propagate(rho.detach().numpy(), horizons=(1,2,5,20))
    worst_value = 0.0
    for h in pred:
        w, sd = pred[h]
        nw, nsd = numpy_pred[h]
        worst_value = max(worst_value, require_close(
            w.detach().numpy(), nw, 1e-13, "Torch vs NumPy weights"),
            require_close(sd.detach().numpy(), nsd, 1e-13, "Torch vs NumPy sigma"))

    def torch_crps(weights, sd, y):
        z = y[:,None]/sd
        a = sd*np.sqrt(2/np.pi)*torch.exp(-.5*z*z)
        a = a + y[:,None]*torch.erf(z/np.sqrt(2))
        pair = np.sqrt(2/np.pi)*torch.sqrt(sd[:,None]**2+sd[None,:]**2)
        return (weights*a).sum(-1)-.5*torch.einsum("bi,ij,bj->b",weights,pair,weights)

    y = torch.tensor([-.71,.3,1.45], dtype=torch.float64)
    h1_loss = torch_crps(*pred[1],y).sum()
    h1_gradient = torch.autograd.grad(h1_loss,rho,retain_graph=True)[0]
    if not torch.equal(h1_gradient, torch.zeros_like(h1_gradient)):
        raise AssertionError("h1 rho derivative is not exactly zero")
    worst_gradient = 0.0
    nonzero = []
    for h in [2,5]:
        loss = torch_crps(*pred[h],y).sum()
        gradient = torch.autograd.grad(loss,rho,retain_graph=True)[0].detach().numpy()
        numerical = []
        for r,target in zip(rho.detach().numpy(),y.detach().numpy()):
            eps = 1e-5
            scores = []
            for changed in [r-eps,r+eps]:
                _, masses, _, variance, _ = states_enumerated(h,changed)
                scores.append(score_by_cdf(float(target),masses,variance)[0])
            numerical.append((scores[1]-scores[0])/(2*eps))
        worst_gradient = max(worst_gradient, require_close(
            gradient,np.array(numerical),2e-8,"autograd vs independent enumerated CDF finite difference"))
        nonzero.extend(np.abs(gradient).tolist())
    if min(nonzero) <= 1e-7:
        raise AssertionError("h>=2 positive control has no useful gradient")
    return dict(torch_numpy_maximum_difference=worst_value,
                h1_autograd_derivative=h1_gradient.detach().numpy().tolist(),
                higher_horizon_gradient_cases=len(nonzero),
                higher_horizon_gradient_minimum_abs=min(nonzero),
                maximum_gradient_difference=worst_gradient)


def check_numerical_guards(module):
    rejected = []
    for name,rho,pi in [
        ("rho_nan",np.nan,PI), ("rho_inf",np.inf,PI),
        ("pi_nan",.5,np.array([np.nan,.4,.6])),
        ("rho_negative",-.1,PI), ("rho_above_one",1.1,PI)]:
        try:
            module.propagate(rho,horizons=(1,2),pi=pi)
        except (ValueError,FloatingPointError):
            rejected.append(name)
        else:
            raise AssertionError(f"Invalid input not rejected: {name}")

    # Either a finite accurate likelihood or explicit failure is acceptable.
    # Silent probability flooring / fake posterior is not exact likelihood.
    extreme = np.array([[80.,.2,-.4]])
    expected = enumerated_nll(extreme,.8)
    try:
        actual = module.path_nll_from_emissions(module.path_emissions(extreme),.8)
    except (ValueError,FloatingPointError):
        extreme_status = "explicit numerical-range rejection"
        extreme_error = None
    else:
        extreme_error = require_close(actual,expected,1e-8,
                                       "extreme finite-path NLL, no silent floor")
        extreme_status = "stable exact likelihood"
    # The runner explicitly uses the shifted-emission interface; require that
    # this positive control succeeds, not merely that raw underflow is refused.
    stable_extreme = np.array([[80.,.2,-.4], [-80.,80.,-.1], [160.,-160.,0.]])
    stable_worst = 0.0
    for rho in [0.,.8,.999,1.]:
        expected = enumerated_nll(stable_extreme,rho)
        stable_density = module.path_emissions(stable_extreme,stable=True)
        if stable_density.shape != (3,3,4):
            raise AssertionError("Stable emission channel contract changed")
        actual = module.path_nll_from_emissions(stable_density,rho)
        stable_worst = max(stable_worst, require_close(
            actual,expected,2e-10,"stable extreme finite-path NLL"))
    return dict(rejected_invalid_inputs=rejected,
                extreme_path_handling=extreme_status,
                extreme_path_nll_absolute_difference=extreme_error,
                stable_extreme_path_likelihood_cases=12,
                maximum_stable_extreme_nll_difference=stable_worst,
                extreme_test_is_synthetic=True)


def check_protocol():
    p = json.loads(PROTOCOL.read_text())
    s = p["synthetic"]
    assert s["q"] == Q.tolist()
    assert s["pi"] == PI.tolist()
    assert s["n"] == [256,1024,4096]
    assert s["seeds"] == list(range(1000,1020))
    assert set(s["arms"]) == {"conditional_endpoint","constant_endpoint",
                              "conditional_joint005","constant_joint005"}
    assert s["main"] == "conditional_joint005"
    assert s["main_matched_baseline"] == "constant_joint005"
    gate = s["market_resource_gate"]
    assert "per-seed" in gate and "-.0002" in gate and "14 of 20" in gate
    return dict(synthetic_n=s["n"], seeds=s["seeds"], arms=s["arms"],
                main=s["main"], main_matched=s["main_matched_baseline"],
                endpoint_loss=s["endpoint_loss"], joint_loss=s["joint_loss"],
                gate=gate, scope="protocol identity only; no fitting executed")


def main():
    module = load_production()
    record = dict(
        audit="independent_persistence_model_audit",
        created_utc=datetime.now(timezone.utc).isoformat(),
        production_source_sha256=sha256(SOURCE),
        protocol_sha256=sha256(PROTOCOL),
        audit_script_sha256=sha256(Path(__file__)),
        scope="Pure numerical / mathematical audit. No market reads, no synthetic parameter fits.",
        production_imports=["source/persistence.py"],
        independent_reference="All finite state sequences + direct CDF integrals + complete-path log-sum-exp",
        checks={})
    functions = [
        ("enumeration_and_moments",lambda:check_enumeration_and_moments(module)),
        ("crps_and_expected_score",lambda:check_scores(module)),
        ("likelihood",lambda:check_likelihood(module)),
        ("autograd",lambda:check_autograd(module)),
        ("numerical_guards",lambda:check_numerical_guards(module)),
        ("protocol",check_protocol)]
    for name,operation in functions:
        try:
            record["checks"][name] = dict(passed=True,**operation())
        except Exception as exc:
            record["checks"][name] = dict(passed=False,error=f"{type(exc).__name__}: {exc}")
    record["passed"] = all(c["passed"] for c in record["checks"].values())
    if OUT.exists():
        old = json.loads(OUT.read_text())
        record["prior_attempts"] = old.get("prior_attempts",[]) + [{
            key:old.get(key) for key in ["created_utc","passed","production_source_sha256",
                                        "protocol_sha256","audit_script_sha256","checks"]}]
    OUT.write_text(json.dumps(record,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({"passed":record["passed"],"output":str(OUT),
                      "checks":{k:v["passed"] for k,v in record["checks"].items()}},ensure_ascii=False))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
