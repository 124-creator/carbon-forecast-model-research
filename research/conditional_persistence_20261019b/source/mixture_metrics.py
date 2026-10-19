"""Shared exact finite-Gaussian-mixture scores for both new subexperiments.

Inputs y[N], weights/means/sigmas[N,K] are in the SAME normalized log-return
units.  ``unit[N]`` converts these into raw log returns.  No Monte Carlo sample
mean replaces an analytic distributional mean.  Pairwise terms are chunked
over origins to limit memory; zero-weight components are permitted.

For real-line distributions, CRPS(F,y)=E|X-y|-.5 E|X-X'|.  Applying normal
absolute-moment identities gives the Gaussian-mixture expression.  Applying
lognormal truncated-moment identities gives price CRPS.  These are independent
mathematical implementations, not vendored author code.

Primary formula reference:
  Gneiting et al. (2005), https://doi.org/10.1175/MWR2904.1
  Jordan, Krueger & Lerch (2019), https://doi.org/10.18637/jss.v090.i12
The repository's older zero-mean implementation is used only as a numerical
special-case cross-check in audits, never as an imported implementation here.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import erf, logsumexp, ndtr, ndtri

SQRT2 = math.sqrt(2.0)
SQRT2PI_INV = math.sqrt(2.0 / math.pi)
LOG2PI_HALF = 0.5 * math.log(2.0 * math.pi)


def _validated(y, weights, means, sigmas):
    y = np.asarray(y, dtype=np.float64)
    w, m, s = [np.asarray(x, dtype=np.float64) for x in (weights, means, sigmas)]
    if y.ndim != 1 or w.ndim != 2 or w.shape != m.shape or w.shape != s.shape or len(y) != len(w):
        raise ValueError("Expected y[N] and matching weights, means, sigmas[N,K]")
    if w.shape[1] < 1 or not all(np.isfinite(x).all() for x in (y, w, m, s)):
        raise ValueError("Mixture arrays must be finite with at least one component")
    totals = w.sum(axis=1)
    if np.any(w < 0) or np.any(s <= 0) or np.any(totals <= 0):
        raise ValueError("Weights must be nonnegative and sigmas strictly positive")
    if np.any(np.abs(totals - 1.0) > 5e-5):
        raise ValueError("Mixture weights do not sum to one within float32 tolerance")
    # Only remove numerical softmax/DP summation error, not invalid weights.
    w = w / totals[:, None]
    return y, w, m, s


def _normal_abs(mean, sigma):
    absolute = np.abs(mean)
    z = absolute / sigma
    return sigma * SQRT2PI_INV * np.exp(-0.5 * z * z) + absolute * erf(z / SQRT2)


def gaussian_mixture_crps(y, weights, means, sigmas, *, chunk_size=64):
    y, w, m, s = _validated(y, weights, means, sigmas)
    result = np.empty(len(y), dtype=np.float64)
    for start in range(0, len(y), chunk_size):
        sl = slice(start, start + chunk_size)
        wc, mc, sc = w[sl], m[sl], s[sl]
        first = np.sum(wc * _normal_abs(mc - y[sl, None], sc), axis=1)
        pair = _normal_abs(mc[:, :, None] - mc[:, None, :],
                           np.hypot(sc[:, :, None], sc[:, None, :]))
        second = 0.5 * np.einsum("ni,nij,nj->n", wc, pair, wc)
        result[sl] = first - second
    return result


def gaussian_mixture_nll(y, weights, means, sigmas):
    y, w, m, s = _validated(y, weights, means, sigmas)
    with np.errstate(divide="ignore"):
        logw = np.log(w)
    z = (y[:, None] - m) / s
    return -logsumexp(logw - np.log(s) - 0.5 * z * z - LOG2PI_HALF, axis=1)


def gaussian_mixture_quantiles(weights, means, sigmas, levels=(0.05, 0.95), *, iterations=64):
    weights = np.asarray(weights)
    _, w, m, s = _validated(np.zeros(len(weights)), weights, means, sigmas)
    q = np.asarray(levels, dtype=np.float64)
    if q.ndim != 1 or not np.isfinite(q).all() or np.any(q <= 0) or np.any(q >= 1):
        raise ValueError("Quantile levels must lie strictly in (0,1)")
    result = np.empty((len(w), len(q)), dtype=np.float64)
    for j, level in enumerate(q):
        component = m + s * ndtri(level)
        # Every component's level-quantile brackets the mixture quantile.
        lo, hi = component.min(axis=1), component.max(axis=1)
        for _ in range(iterations):
            mid = lo + 0.5 * (hi - lo)
            cdf = np.sum(w * ndtr((mid[:, None] - m) / s), axis=1)
            below = cdf < level
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        result[:, j] = lo + 0.5 * (hi - lo)
    return result


def lognormal_mixture_crps(price, weights, log_means, log_sigmas, *, chunk_size=64):
    price, w, m, s = _validated(price, weights, log_means, log_sigmas)
    if np.any(price <= 0):
        raise ValueError("Observed prices must be positive")
    result = np.empty(len(price), dtype=np.float64)
    for start in range(0, len(price), chunk_size):
        sl = slice(start, start + chunk_size)
        wc, mc, sc, yc = w[sl], m[sl], s[sl], price[sl]
        with np.errstate(over="raise", invalid="raise"):
            component_mean = np.exp(mc + 0.5 * sc * sc)
            logy = np.log(yc)[:, None]
            first_abs = component_mean * erf((mc + sc * sc - logy) / (SQRT2 * sc))
            first_abs += yc[:, None] * erf((logy - mc) / (SQRT2 * sc))
            first = np.sum(wc * first_abs, axis=1)
            delta = mc[:, :, None] - mc[:, None, :]
            pair_sd = np.hypot(sc[:, :, None], sc[:, None, :])
            pair_abs = component_mean[:, :, None] * erf(
                (delta + sc[:, :, None] ** 2) / (SQRT2 * pair_sd))
            pair_abs += component_mean[:, None, :] * erf(
                (-delta + sc[:, None, :] ** 2) / (SQRT2 * pair_sd))
            result[sl] = first - 0.5 * np.einsum("ni,nij,nj->n", wc, pair_abs, wc)
    return result


def score_mixture(y, weights, means, sigmas, *, origin_price, unit, target_price=None):
    """Return per-origin arrays in a common schema, never market aggregates.

    Keys: normalized_CRPS, normalized_NLL, price_CRPS, price_NLL,
    mean_price_squared_error, rw_mean_price_squared_error, mean_price,
    coverage90, width90, lower90, upper90, normalized_width90.
    Price metrics use the exact lognormal-mixture transform and mean.
    """
    y, w, m, s = _validated(y, weights, means, sigmas)
    p0, u = np.asarray(origin_price, dtype=float), np.asarray(unit, dtype=float)
    if p0.shape != y.shape or u.shape != y.shape or not np.isfinite(p0).all() or not np.isfinite(u).all():
        raise ValueError("origin_price and unit must be finite N-vectors")
    if np.any(p0 <= 0) or np.any(u <= 0):
        raise ValueError("origin prices and scoring units must be positive")
    with np.errstate(over="raise", invalid="raise"):
        implied = p0 * np.exp(u * y)
        actual = implied if target_price is None else np.asarray(target_price, dtype=float)
        if actual.shape != y.shape or not np.isfinite(actual).all() or np.any(actual <= 0):
            raise ValueError("target_price must be a finite positive N-vector")
        if not np.allclose(actual, implied, rtol=2e-6, atol=1e-9):
            raise ValueError("Target price is inconsistent with the issued log-return target")
        log_mean = np.log(p0)[:, None] + u[:, None] * m
        log_sigma = u[:, None] * s
        mean_price = np.sum(w * np.exp(log_mean + 0.5 * log_sigma ** 2), axis=1)
        quantile = gaussian_mixture_quantiles(w, m, s)
        bounds = p0[:, None] * np.exp(u[:, None] * quantile)
    nll = gaussian_mixture_nll(y, w, m, s)
    result = {
        "normalized_CRPS": gaussian_mixture_crps(y, w, m, s),
        "normalized_NLL": nll,
        "price_CRPS": lognormal_mixture_crps(actual, w, log_mean, log_sigma),
        "price_NLL": nll + np.log(u) + np.log(actual),
        "mean_price_squared_error": (mean_price - actual) ** 2,
        "rw_mean_price_squared_error": (p0 - actual) ** 2,
        "mean_price": mean_price,
        "coverage90": ((actual >= bounds[:, 0]) & (actual <= bounds[:, 1])).astype(float),
        "width90": bounds[:, 1] - bounds[:, 0],
        "lower90": bounds[:, 0], "upper90": bounds[:, 1],
        "normalized_width90": quantile[:, 1] - quantile[:, 0],
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("A forecast metric is nonfinite; no score clipping applied")
    return result


def metrics(y, *, unit, price0, pricetrue, w, means, sigmas):
    """Keyword-compatible wrapper used by the persistence market runner."""
    return score_mixture(y, w, means, sigmas, origin_price=price0,
                         unit=unit, target_price=pricetrue)


def equal_atom_crps(y, atoms):
    """Exact CRPS of the equal-weight finite distribution, including its atoms."""
    y, x = np.asarray(y, dtype=float), np.asarray(atoms, dtype=float)
    if y.ndim != 1 or x.ndim != 2 or len(x) != len(y) or x.shape[1] < 1:
        raise ValueError("Expected y[N], atoms[N,S]")
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("Atomic targets and forecasts must be finite")
    ordered = np.sort(x, axis=1)
    count = x.shape[1]
    coefficient = (2 * np.arange(count) - count + 1) / float(count * count)
    half_pair = ordered @ coefficient
    return np.mean(np.abs(x-y[:, None]), axis=1) - half_pair


def score_atoms(y, atoms, *, origin_price, unit, target_price=None):
    """Scores for the fixed 101-atom empirical-residual control.

    A Lebesgue density/log score is NOT defined for this forecast; no NLL keys
    are returned, rather than adding an unprespecified smoothing bandwidth.
    Quantiles use the left inverse of the equal-mass discrete CDF.
    """
    y, atoms = np.asarray(y, dtype=float), np.asarray(atoms, dtype=float)
    p0, u = np.asarray(origin_price, dtype=float), np.asarray(unit, dtype=float)
    if y.ndim != 1 or atoms.ndim != 2 or len(atoms) != len(y) or p0.shape != y.shape or u.shape != y.shape:
        raise ValueError("Atomic price-score dimensions do not match")
    if not all(np.isfinite(a).all() for a in (y, atoms, p0, u)) or np.any(p0 <= 0) or np.any(u <= 0):
        raise ValueError("Atomic inputs must be finite and prices/units positive")
    with np.errstate(over="raise", invalid="raise"):
        implied = p0 * np.exp(u * y)
        actual = implied if target_price is None else np.asarray(target_price, dtype=float)
        if actual.shape != y.shape or not np.isfinite(actual).all() or np.any(actual <= 0) or not np.allclose(actual, implied, rtol=2e-6, atol=1e-9):
            raise ValueError("Atomic target price is inconsistent with the log-return target")
        price_atoms = p0[:, None] * np.exp(u[:, None] * atoms)
        mean_price = price_atoms.mean(axis=1)
        order = np.sort(atoms, axis=1)
        at = np.ceil(np.array([.05,.95])*atoms.shape[1]).astype(int)-1
        q = order[:, at]
        bounds = p0[:, None] * np.exp(u[:, None] * q)
    return {"normalized_CRPS": equal_atom_crps(y, atoms),
            "price_CRPS": equal_atom_crps(actual, price_atoms),
            "mean_price_squared_error": (mean_price-actual)**2,
            "rw_mean_price_squared_error": (p0-actual)**2,
            "mean_price": mean_price,
            "coverage90": ((actual >= bounds[:,0]) & (actual <= bounds[:,1])).astype(float),
            "width90": bounds[:,1]-bounds[:,0], "lower90": bounds[:,0], "upper90": bounds[:,1],
            "normalized_width90": q[:,1]-q[:,0]}


def torch_gaussian_mixture_crps(y, weights, means, sigmas):
    """Differentiable per-origin training loss; imports torch only when called."""
    import torch

    def normal_abs(a, b):
        z = a / b
        return b * SQRT2PI_INV * torch.exp(-0.5 * z.square()) + a * torch.erf(z / SQRT2)

    first = (weights * normal_abs(means - y[:, None], sigmas)).sum(dim=1)
    distance = means[:, :, None] - means[:, None, :]
    scale = torch.sqrt(sigmas[:, :, None].square() + sigmas[:, None, :].square())
    pair = normal_abs(distance, scale)
    second = 0.5 * (weights[:, :, None] * pair * weights[:, None, :]).sum(dim=(1, 2))
    return first - second

