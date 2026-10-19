"""Small synthetic-only audit of online_baselines.py; no market inputs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr, ndtri


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from online_baselines import (  # noqa: E402
    AdaptiveBOA,
    BOACDFMixture,
    DelayedCDFBOA,
    EmpiricalCRPSMixture,
    QuantileBOAG,
)


def absolute_normal(mean, sigma):
    standardized = np.asarray(mean) / np.asarray(sigma)
    return sigma * (
        2.0 * np.exp(-0.5 * standardized**2) / math.sqrt(2.0 * math.pi)
        + standardized * (2.0 * ndtr(standardized) - 1.0)
    )


def two_normal_crps(alpha, y, mu=(-0.8, 1.1), sigma=(0.7, 1.6)):
    weights = np.array([1.0 - alpha, alpha])
    mu, sigma = np.asarray(mu), np.asarray(sigma)
    observed = absolute_normal(mu - y, sigma)
    pairwise = absolute_normal(
        mu[:, None] - mu[None, :],
        np.sqrt(sigma[:, None]**2 + sigma[None, :]**2),
    )
    return float(weights @ observed - 0.5 * weights @ pairwise @ weights)


def normal_ab(y, mu=(-0.8, 1.1), sigma=(0.7, 1.6)):
    mu, sigma = np.asarray(mu), np.asarray(sigma)
    t = absolute_normal(mu - y, sigma)
    cross = float(absolute_normal(mu[0] - mu[1], np.hypot(*sigma)))
    same = 2.0 * sigma / math.sqrt(math.pi)
    a = cross - 0.5 * same.sum()
    b = 0.5 * (t[0] - t[1] - same[0] + cross)
    return float(a), float(b)


def require_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid input was not rejected")


def run():
    result = {}

    # A closed-form one-round oracle checks reward direction and the 1/2
    # second-order correction, independently of another recursion copy.
    boa = BOACDFMixture()
    boa.update_ab(2.0, 0.0, 0.5)  # delta_0 vs delta_2, observed y=0
    expected_bad_weight = 1.0 / (1.0 + math.exp(0.5))
    assert abs(boa.alpha - expected_bad_weight) < 2e-15
    result["one_round_closed_form"] = {"passed": True, "alpha": boa.alpha}

    # Verify exact Gaussian-mixture CRPS geometry and its scalar derivative.
    a, b = normal_ab(0.23)
    alphas = np.array([0.0, 0.1, 0.37, 0.8, 1.0])
    base = two_normal_crps(0.0, 0.23)
    errors = [abs(two_normal_crps(x, 0.23) - base - (a*x*x - 2*b*x)) for x in alphas]
    step = 1e-5
    numerical_gradient = (two_normal_crps(0.37 + step, 0.23) - two_normal_crps(0.37 - step, 0.23)) / (2*step)
    gradient_error = abs(numerical_gradient - 2*(a*0.37 - b))
    assert max(errors) < 2e-14 and gradient_error < 1e-9
    result["exact_crps_quadratic_and_gradient"] = {
        "passed": True, "max_quadratic_error": max(errors), "gradient_error": gradient_error
    }

    # Cumulative empirical mixing is independently compared with numerical
    # optimization of actual CRPS, rather than its own AB implementation.
    rng = np.random.default_rng(20261019)
    y = rng.normal(0.1, 1.2, size=160)
    terms = np.array([normal_ab(value) for value in y])
    empirical = EmpiricalCRPSMixture()
    empirical.replay_ab(terms[:, 0], terms[:, 1])
    objective = lambda alpha: np.mean([two_normal_crps(alpha, value) for value in y])
    optimum = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-12})
    optimization_error = abs(empirical.alpha - optimum.x)
    assert optimization_error < 2e-7
    result["empirical_mixture_vs_direct_optimization"] = {
        "passed": True, "alpha_error": optimization_error
    }

    rolling = EmpiricalCRPSMixture(window=13)
    rolling.replay_ab(terms[:, 0], terms[:, 1])
    expected_rolling = np.clip(terms[-13:, 1].sum()/terms[-13:, 0].sum(), 0, 1)
    assert abs(rolling.alpha-expected_rolling) < 1e-14
    result["rolling_window"] = {"passed": True}

    # Unit changes and exchanging expert names must not alter predictions.
    ordinary, scaled, swapped = BOACDFMixture(), BOACDFMixture(), BOACDFMixture()
    max_scale_error = max_swap_error = 0.0
    for a, b in terms:
        ordinary.update_ab(a, b)
        scaled.update_ab(a*10000.0, b*10000.0)
        swapped.update_ab(a, a-b)
        max_scale_error = max(max_scale_error, abs(ordinary.alpha-scaled.alpha))
        max_swap_error = max(max_swap_error, abs(ordinary.alpha+swapped.alpha-1.0))
    assert max_scale_error < 2e-13 and max_swap_error < 2e-13
    result["scale_and_expert_exchange_invariance"] = {
        "passed": True, "max_scale_error": max_scale_error, "max_swap_error": max_swap_error
    }

    # Identical experts yield neither evidence nor a spurious preference.
    identical = BOACDFMixture()
    identical.replay_ab(np.zeros(50), np.zeros(50))
    assert identical.alpha == 0.5
    require_error(lambda: identical.update_ab(0.0, 1.0))
    require_error(lambda: identical.update_ab(-1.0, 0.0))
    require_error(lambda: identical.update_ab(1.0, np.nan))
    result["zero_information_and_input_validation"] = {"passed": True}

    # Explicit delayed feedback: second ticket retains its issued alpha
    # after the first ticket has changed the current weights.
    delayed, manual = DelayedCDFBOA(), BOACDFMixture()
    ticket0 = delayed.issue(0, 3)
    ticket1 = delayed.issue(1, 4)
    require_error(lambda: delayed.observe(ticket0, 2.0, 0.0, available_index=2))
    require_error(lambda: delayed.observe(ticket1, 2.0, 1.5, available_index=4))
    delayed.observe(ticket0, 2.0, 0.0, available_index=3)
    manual.update_ab(2.0, 0.0, ticket0.alpha)
    ticket3 = delayed.issue(3, 6)
    assert ticket3.alpha != ticket1.alpha
    delayed.observe(ticket1, 2.0, 1.5, available_index=4)
    manual.update_ab(2.0, 1.5, ticket1.alpha)
    assert np.array_equal(delayed.weights, manual.weights)
    require_error(lambda: delayed.observe(ticket1, 2.0, 1.5, available_index=5))
    require_error(lambda: delayed.issue(2, 7))
    result["delayed_maturity_and_archived_weights"] = {"passed": True}

    # Replay is deterministic and allows separate frozen/continuing copies.
    replay, manual = BOACDFMixture(), BOACDFMixture()
    trace = replay.replay_ab(terms[:, 0], terms[:, 1])
    for i, (a, b) in enumerate(terms):
        assert trace[i] == manual.alpha
        manual.update_ab(a, b)
    assert np.array_equal(replay.weights, manual.weights)
    result["chronological_replay"] = {"passed": True}

    # Quantile sign/tie convention and post-combination sorting are checked
    # using a constructed crossing forecast, not a market fit.
    quantile = QuantileBOAG([0.05, 0.5, 0.95])
    experts = np.array([[-2.0, -1.0], [0.0, 3.0], [2.0, 4.0]])
    quantile.core._weights = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    issued = quantile.predict(experts)
    assert np.array_equal(issued, [-2.0, 2.0, 3.0])
    subgradient = np.array([-0.05, -0.5, 0.05])  # y=2.5
    expected_variance = (subgradient[:, None]*(issued[:, None]-experts))**2
    quantile.update_quantiles(experts, 2.5, issued)
    assert np.allclose(quantile.core.V, expected_variance, rtol=0, atol=1e-15)
    assert np.allclose(quantile.weights.sum(axis=1), 1.0)
    tie = QuantileBOAG([0.5])
    tie.update_quantiles(np.array([[0.0, 2.0]]), 1.0, np.array([1.0]))
    assert tie.weights[0, 0] > 0.5
    result["quantile_rearrangement_and_subgradient"] = {"passed": True}

    # For an equal-weight finite distribution the midpoint grid, not a
    # casually chosen percent grid, makes quantile loss exactly equal CRPS.
    count = 19
    tau = (np.arange(count)+0.5)/count
    atoms = np.sort(ndtri(tau) * 0.8 + 0.15)
    max_atom_error = 0.0
    for truth in [-3.0, -0.3, 0.0, 0.9, 4.0]:
        exact = np.abs(atoms-truth).mean()-0.5*np.abs(atoms[:, None]-atoms[None, :]).mean()
        pinball = ((truth < atoms)-tau)*(atoms-truth)
        max_atom_error = max(max_atom_error, abs(exact-2*pinball.mean()))
    assert max_atom_error < 2e-15
    result["nineteen_atom_crps_identity"] = {"passed": True, "max_error": max_atom_error}

    result["metadata"] = {
        "scope": "synthetic algebra, numerical invariance, and interface maturity audit",
        "market_data_accessed": False,
        "full_profoc_runtime_reproduced": False,
        "author_commit_inspected": "b5073ca70aea60ce209676faa99e04da725ffe04",
        "source_sha256": hashlib.sha256((ROOT/"source"/"online_baselines.py").read_bytes()).hexdigest(),
        "numpy": np.__version__,
        "all_checks_passed": True,
    }
    output = ROOT / "audits" / "online_baselines_review.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"output": str(output), "checks": len(result)-1, "all_checks_passed": True}))


if __name__ == "__main__":
    run()
