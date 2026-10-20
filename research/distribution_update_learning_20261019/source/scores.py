"""Analytic scores for centered Gaussian mixtures and their price transforms.

These are standard expected-absolute-distance identities, not new scores.
All market Gaussian mixture components share a zero log-increment mean.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

SQRT_2_PI = np.sqrt(2.0 / np.pi)


def normal_abs(z, sigma):
    z, sigma = np.broadcast_arrays(np.asarray(z, float), np.asarray(sigma, float))
    t = z / np.maximum(sigma, 1e-300)
    value = sigma * SQRT_2_PI * np.exp(-0.5 * t * t) + z * (2 * ndtr(t) - 1)
    return np.where(sigma > 0, value, np.abs(z))


def self_distance(weights, sigma):
    sigma = np.asarray(sigma, float)
    w = np.asarray(weights, float)
    return SQRT_2_PI * np.sum(
        w[..., :, None] * w[..., None, :] *
        np.sqrt(sigma[..., :, None] ** 2 + sigma[..., None, :] ** 2), axis=(-2, -1))


def crps(y, weights, sigma):
    return np.sum(np.asarray(weights) * normal_abs(np.asarray(y)[..., None], sigma), axis=-1) - 0.5 * self_distance(weights, sigma)


def pair_terms(y, weights, sigma0, sigma1):
    """Return exact a, b, score0, score1; score(alpha)=s0+a*alpha^2-2*b*alpha."""
    w = np.asarray(weights)
    s0, s1 = np.asarray(sigma0), np.asarray(sigma1)
    self0, self1 = self_distance(w, s0), self_distance(w, s1)
    cross = SQRT_2_PI * np.sum(w[..., :, None] * w[..., None, :] * np.sqrt(s0[..., :, None] ** 2 + s1[..., None, :] ** 2), axis=(-2, -1))
    a = np.maximum(cross - 0.5 * self0 - 0.5 * self1, 0.0)
    score0 = np.sum(w * normal_abs(np.asarray(y)[..., None], s0), axis=-1) - 0.5 * self0
    score1 = np.sum(w * normal_abs(np.asarray(y)[..., None], s1), axis=-1) - 0.5 * self1
    b = 0.5 * (a + score0 - score1)
    return np.broadcast_to(a, np.shape(y)).copy(), b, score0, score1


def scale_w1(weights, sigma0, factor):
    """Exact W1 under the common positive-scale coupling, zero-centered mixture."""
    return SQRT_2_PI * np.sum(np.asarray(weights) * np.asarray(sigma0), axis=-1) * np.abs(np.asarray(factor) - 1)


def reflected_control(increments, weights, sigma0, factor, capped=True, no_prefix=False, cap_fraction=0.25):
    """Predictably capped odd increments of a reference continuation of b.

    increments are the realized FUTURE path, used ONLY after all labels mature.
    sigma0/factor are the forecasts issued at the original prediction origin.
    The reference has Gaussian increment variance 1/h (unit terminal variance), not an estimated
    true conditional law. Its correctness is unnecessary for the oddness claim.
    """
    r = np.asarray(increments, float)
    n, h = r.shape
    w = np.asarray(weights, float)
    s0 = np.broadcast_to(np.asarray(sigma0, float), (n, len(w)))
    f = np.broadcast_to(np.asarray(factor, float), (n,))
    s1 = s0 * f[:, None]
    cap = cap_fraction * scale_w1(w, s0, f)
    prefix, control, spent = np.zeros(n), np.zeros(n), np.zeros(n)
    for j in range(h):
        remaining_variance = (h - j - 1) / h
        smooth0 = np.sqrt(s0 * s0 + remaining_variance)
        smooth1 = np.sqrt(s1 * s1 + remaining_variance)
        p = np.zeros(n) if no_prefix else prefix
        plus, minus = (p + r[:, j])[:, None], (p - r[:, j])[:, None]
        # The a and self-distance constants in G cancel in the reflection.
        increment = 0.25 * np.sum(w * (
            normal_abs(plus, smooth0) - normal_abs(plus, smooth1)
            - normal_abs(minus, smooth0) + normal_abs(minus, smooth1)), axis=-1)
        if capped:
            remaining = np.maximum(cap - spent, 0)
            increment = np.clip(increment, -remaining, remaining)
        control += increment
        spent += np.abs(increment)
        prefix += r[:, j]
    if capped and np.any(spent > cap + 1e-10):
        raise AssertionError("Predictable control exceeded its known origin budget")
    return control, cap, spent


def cdf(x, weights, sigma):
    return np.sum(np.asarray(weights) * ndtr(np.asarray(x)[..., None] / np.asarray(sigma)), axis=-1)


def quantiles(probabilities, weights, sigma):
    """Deterministic bisection; supports batched mixture weights and scales."""
    p, w, s = np.asarray(probabilities), np.asarray(weights), np.asarray(sigma)
    shape = np.broadcast_shapes(w.shape[:-1], s.shape[:-1])
    s = np.broadcast_to(s, shape + (s.shape[-1],))
    w = np.broadcast_to(w, shape + (w.shape[-1],))
    maximum = np.max(s, axis=-1)[..., None]
    low = np.broadcast_to(-12 * maximum, shape + (len(p),)).copy()
    high = -low
    for _ in range(64):
        mid = (low + high) / 2
        f = np.sum(w[..., None, :] * ndtr(mid[..., :, None] / s[..., None, :]), axis=-1)
        low = np.where(f < p, mid, low)
        high = np.where(f >= p, mid, high)
    return (low + high) / 2


def mixture_parts(weights, sigma0, factor, alpha):
    w = np.asarray(weights)
    factor, alpha = np.broadcast_arrays(factor, alpha)
    sigma0 = np.broadcast_to(sigma0, factor.shape + (len(w),))
    mixed_w = np.concatenate([(1 - alpha[..., None]) * w, alpha[..., None] * w], axis=-1)
    mixed_s = np.concatenate([sigma0, sigma0 * factor[..., None]], axis=-1)
    return mixed_w, mixed_s


def lognormal_crps(price, log_location, log_sigma, weights):
    """Exact CRPS via E|X-y| and E|X-X'| for a finite lognormal mixture."""
    y = np.asarray(price)[..., None]
    mu, sig, w = np.broadcast_arrays(log_location, log_sigma, weights)
    if np.any(y <= 0) or np.any(sig <= 0):
        raise ValueError("Positive realized price and component scales required")
    mean = np.exp(mu + sig * sig / 2)
    first = mean - y + 2 * (y * ndtr((np.log(y) - mu) / sig)
                           - mean * ndtr((np.log(y) - mu - sig * sig) / sig))
    den = np.sqrt(sig[..., :, None] ** 2 + sig[..., None, :] ** 2)
    mi, mj = mu[..., :, None], mu[..., None, :]
    si, sj = sig[..., :, None], sig[..., None, :]
    ei, ej = mean[..., :, None], mean[..., None, :]
    minimum = ei * ndtr((mj - mi - si * si) / den) + ej * ndtr((mi - mj - sj * sj) / den)
    pair = ei + ej - 2 * minimum
    result = np.sum(w * first, axis=-1) - 0.5 * np.sum(w[..., :, None] * w[..., None, :] * pair, axis=(-2, -1))
    return np.maximum(result, 0)


def atomic_crps(y, atoms, weights=None):
    """Exact finite support CRPS with a linear-memory sorted-support identity."""
    x = np.asarray(atoms)
    w = np.ones_like(x) / x.shape[-1] if weights is None else np.broadcast_to(weights, x.shape)
    order = np.argsort(x, axis=-1)
    x, w = np.take_along_axis(x, order, axis=-1), np.take_along_axis(w, order, axis=-1)
    cw, cwx = np.cumsum(w, axis=-1), np.cumsum(w * x, axis=-1)
    half_pair = np.sum(w * (x * (cw - w) - (cwx - w * x)), axis=-1)
    return np.sum(w * np.abs(x - np.asarray(y)[..., None]), axis=-1) - half_pair
