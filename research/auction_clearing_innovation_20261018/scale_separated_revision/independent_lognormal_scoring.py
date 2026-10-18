"""Independent exact lognormal-mixture scoring; no project model imports.

The production score is not called or read here. The closed form is obtained
from E|X-y| and the independently drawn pair E|X-X'|. Numerical verification
instead integrates the squared mixture CDF, on the logarithmic price axis.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri, erf


def expanded_sigma(sigma, shape):
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.ndim == 1:
        sigma = sigma[:, None]
    return np.broadcast_to(sigma, shape)


def validate_parameters(mu, sigma, weights):
    mu, weights = np.asarray(mu, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    sigma = expanded_sigma(sigma, mu.shape)
    if mu.ndim != 2 or mu.shape != weights.shape:
        raise ValueError("Lognormal mixture parameter dimensions differ")
    if not (np.isfinite(mu).all() and np.isfinite(sigma).all() and np.isfinite(weights).all()):
        raise ValueError("Lognormal parameters are not finite")
    if not ((sigma > 0).all() and (weights >= 0).all()):
        raise ValueError("Lognormal sigma must be positive and weights nonnegative")
    if np.max(np.abs(weights.sum(axis=1) - 1.0)) > 1e-10:
        raise ValueError("Lognormal mixture weights do not sum to one")
    return mu, sigma, weights


def mixture_mean(mu, sigma, weights):
    mu, sigma, weights = validate_parameters(mu, sigma, weights)
    with np.errstate(over="raise", invalid="raise"):
        return np.sum(weights * np.exp(mu + 0.5 * sigma ** 2), axis=1)


def exact_crps(mu, sigma, weights, observed, chunk_size=24):
    """Exact CRPS for a positive continuous finite lognormal mixture.

    For m_i=exp(mu_i+s_i^2/2), positive y and z_i=(log y-mu_i)/s_i:
      A_i = y erf(z_i/sqrt(2)) - m_i erf((z_i-s_i)/sqrt(2)).
    Symmetry of the ordered double sum gives half the pair-distance term as
      sum_ij w_i w_j m_i erf((mu_i-mu_j+s_i^2)/sqrt(2(s_i^2+s_j^2))).
    Diagonal i=j remains present: it compares two independent draws from
    the same continuous component, not one deterministic atom with itself.
    """
    mu, sigma, weights = validate_parameters(mu, sigma, weights)
    observed = np.asarray(observed, dtype=np.float64)
    if observed.shape != (len(mu),) or not np.isfinite(observed).all():
        raise ValueError("CRPS requires one finite observation per row")
    result = np.empty(len(mu), dtype=np.float64)
    for start in range(0, len(mu), chunk_size):
        end = min(start + chunk_size, len(mu))
        u, s, w, y = mu[start:end], sigma[start:end], weights[start:end], observed[start:end]
        with np.errstate(over="raise", invalid="raise"):
            means = np.exp(u + .5 * s ** 2)
            positive = y > 0
            first = means - y[:, None]
            if positive.any():
                z = (np.log(y[positive])[:, None] - u[positive]) / s[positive]
                first[positive] = y[positive, None] * erf(z / math.sqrt(2)) \
                    - means[positive] * erf((z - s[positive]) / math.sqrt(2))
            variance_sum = s[:, :, None] ** 2 + s[:, None, :] ** 2
            d = (u[:, :, None] - u[:, None, :] + s[:, :, None] ** 2) / np.sqrt(2 * variance_sum)
            half_pair = np.sum(w[:, :, None] * w[:, None, :] * means[:, :, None] * erf(d), axis=(1, 2))
            result[start:end] = np.sum(w * first, axis=1) - half_pair
    return result


def mixture_cdf_logprice(log_price, mu, sigma, weights):
    mu, sigma, weights = validate_parameters(mu, sigma, weights)
    log_price = np.asarray(log_price, dtype=np.float64)
    return np.sum(weights * ndtr((log_price[:, None] - mu) / sigma), axis=1)


def mixture_quantile(mu, sigma, weights, probability):
    """Solve the actual mixture CDF on log-price, never mix component quantiles."""
    mu, sigma, weights = validate_parameters(mu, sigma, weights)
    if not 0 < probability < 1:
        raise ValueError("Only interior quantiles are finite")
    component_log_quantile = mu + sigma * ndtri(probability)
    lower = component_log_quantile.min(axis=1)
    upper = component_log_quantile.max(axis=1)
    for _ in range(60):
        midpoint = .5 * (lower + upper)
        cdf = np.sum(weights * ndtr((midpoint[:, None] - mu) / sigma), axis=1)
        lower = np.where(cdf < probability, midpoint, lower)
        upper = np.where(cdf >= probability, midpoint, upper)
    log_quantile = .5 * (lower + upper)
    residual = np.sum(weights * ndtr((log_quantile[:, None] - mu) / sigma), axis=1) - probability
    with np.errstate(over="raise", invalid="raise"):
        quantile = np.exp(log_quantile)
    return quantile, residual


def numerical_crps(mu, sigma, weights, observed):
    """Independent squared-CDF integration, using only the mixture CDF/SF.

    The finite integration domain contains y and at least twelve marginal
    standard deviations beyond every component. Analytic upper bounds on
    omitted tails are returned separately; no closed-form CRPS term is used.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=np.float64), mu.shape)
    weights = np.asarray(weights, dtype=np.float64)
    y = float(observed)
    if y <= 0:
        raise ValueError("Numerical verification here is for positive market-price targets")
    log_y = math.log(y)
    lower = float(min(np.min(mu - 12 * sigma), log_y - 2))
    upper = float(max(np.max(mu + sigma ** 2 + 12 * sigma), log_y + 2))
    knots = np.unique(np.concatenate([mu + k * sigma for k in (-7, -3, 0, 3, 7)]))
    # An ensemble can have 153 components and more than 400 explicit break
    # points. QUADPACK needs one initial interval per point plus room to adapt;
    # this is capacity, not a change to the CRPS identity or its error tolerance.
    subinterval_limit = max(400, 2 * len(knots) + 50)

    def below(z):
        f = float(np.dot(weights, ndtr((z - mu) / sigma)))
        return math.exp(z) * f * f

    def above(z):
        # Direct SF evaluation avoids cancelling 1 minus a nearly-one CDF.
        sf = float(np.dot(weights, ndtr((mu - z) / sigma)))
        return math.exp(z) * sf * sf

    left, left_error = quad(below, lower, log_y,
                            points=knots[(knots > lower) & (knots < log_y)],
                            epsabs=1e-9, epsrel=1e-10, limit=subinterval_limit)
    right, right_error = quad(above, log_y, upper,
                              points=knots[(knots > log_y) & (knots < upper)],
                              epsabs=1e-9, epsrel=1e-10, limit=subinterval_limit)
    left_bound = math.exp(lower) * float(np.dot(weights, ndtr((lower - mu) / sigma))) ** 2
    # Integral SF^2 <= integral SF <= E[X 1(X>upper_bound)].
    right_bound = float(np.dot(weights * np.exp(mu + .5 * sigma ** 2),
                               ndtr((mu + sigma ** 2 - upper) / sigma)))
    return {"crps_by_numerical_CDF_integration": left + right,
            "quad_error_estimate": left_error + right_error,
            "omitted_tail_upper_bound": left_bound + right_bound,
            "log_integration_bounds": [lower, upper]}


def preflight():
    rng = np.random.default_rng(630218)
    mu = 4.2 + rng.normal(0, .35, size=(12, 3))
    sigma = rng.uniform(.02, .6, size=(12, 3))
    weights = rng.dirichlet(np.ones(3), size=12)
    y = np.exp(4.2 + rng.normal(0, .7, size=12))
    mu[0] = [4, 4, 4]
    sigma[0] = [.1, .1, .1]
    weights[1] = [1, 0, 0]
    sigma[2] = [1e-5, .0005, .002]
    y[2] = np.exp(mu[2, 0])
    score = exact_crps(mu, sigma, weights, y)
    comparisons = []
    for index in range(len(y)):
        numerical = numerical_crps(mu[index], sigma[index], weights[index], y[index])
        difference = abs(score[index] - numerical["crps_by_numerical_CDF_integration"])
        if not np.isclose(score[index], numerical["crps_by_numerical_CDF_integration"], atol=1e-8, rtol=1e-9):
            raise AssertionError(f"Independent lognormal CRPS verification failed at artificial row {index}: {difference}")
        comparisons.append({"artificial_row": index, "exact_crps": float(score[index]),
                            "absolute_difference": float(difference), **numerical})
    quantiles = []
    for q in (.1, .5, .9):
        value, residual = mixture_quantile(mu, sigma, weights, q)
        if not (np.isfinite(value).all() and (value > 0).all() and np.max(np.abs(residual)) <= 1e-8):
            raise AssertionError("Mixture quantile failed its independent CDF equation")
        quantiles.append({"probability": q, "max_CDF_residual": float(np.max(np.abs(residual)))})
    scaled = exact_crps(mu + math.log(3.7), sigma, weights, y * 3.7)
    if not np.allclose(scaled, score * 3.7, atol=1e-8, rtol=1e-9):
        raise AssertionError("Lognormal CRPS price-unit scaling failed")
    zero = exact_crps(mu, sigma, weights, np.zeros(len(mu)))
    negative = exact_crps(mu, sigma, weights, np.full(len(mu), -2.0))
    if not np.allclose(negative - zero, 2, atol=1e-10, rtol=0):
        raise AssertionError("Nonpositive observation branch failed")
    return {"status": "PASS", "artificial_rows": len(y), "real_market_records_read": 0,
            "exact_formula": "weighted E|X-y| minus half the full independent-pair E|X-X'|, retaining diagonal component pairs",
            "verification_method": "independent numerical integral of squared mixture CDF and SF on log-price",
            "criteria": {"CRPS_absolute": 1e-8, "CRPS_relative": 1e-9, "CDF_quantile_absolute": 1e-8},
            "max_CRPS_absolute_difference": max(c["absolute_difference"] for c in comparisons),
            "max_price_scaling_difference": float(np.max(np.abs(scaled - 3.7 * score))),
            "quantile_checks": quantiles, "numerical_comparisons": comparisons}
