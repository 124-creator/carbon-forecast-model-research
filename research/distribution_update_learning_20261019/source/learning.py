"""Quadratic CDF learning and explicitly approximate time-series calibration.

The empirical-Bernstein helper is valid for independent bounded observations.
The circular-block bootstrap routine is an empirical market adaptation: it is
not covered by the independent-sample theorem and is never labeled a finite
sample distribution-free safety guarantee.
"""
from __future__ import annotations

from itertools import product
import numpy as np


def box_quadratic(upper, lower):
    """Exact active-face enumeration for dimension <= 3, including zero fallback."""
    u, l = np.asarray(upper), np.asarray(lower)
    d = len(l)
    if d > 3:
        raise ValueError("Only the prospectively fixed low-dimensional gate is implemented")
    best, best_value = np.zeros(d), 0.0
    for status in product((-1, 0, 1), repeat=d):
        status = np.asarray(status)
        free, fixed = np.flatnonzero(status == -1), np.flatnonzero(status != -1)
        w = np.maximum(status, 0).astype(float)
        if len(free):
            uf = u[np.ix_(free, free)]
            rhs = l[free] - u[np.ix_(free, fixed)] @ w[fixed]
            w[free] = np.linalg.lstsq(uf, rhs, rcond=1e-12)[0]
            if np.max(np.abs(uf @ w[free] - rhs), initial=0) > 1e-7:
                continue
        if np.any(w < -1e-9) or np.any(w > 1 + 1e-9):
            continue
        w = np.clip(w, 0, 1)
        value = float(w @ u @ w - 2 * l @ w)
        if value < best_value:
            best, best_value = w.copy(), value
    return best, best_value


def empirical_gate(a, b, basis):
    phi = np.asarray(basis)
    matrix = np.mean(np.asarray(a)[:, None, None] * phi[:, :, None] * phi[:, None, :], axis=0)
    linear = np.mean(np.asarray(b)[:, None] * phi, axis=0)
    return box_quadratic(matrix, linear)[0]


def circular_indices(n, block_length, repetitions, rng):
    starts = rng.integers(n, size=(repetitions, (n + block_length - 1) // block_length))
    return ((starts[:, :, None] + np.arange(block_length)) % n).reshape(repetitions, -1)[:, :n]


def block_gate(a, b, c, cap, basis, indices, sign_budget=0.05,
               lambdas=(0., .5, 1.), joint_lambdas=None, confidence=.95):
    """Approximate simultaneous bootstrap bounds; returns learnable gate and receipt.

    joint_lambdas allows a no-control baseline with exactly the main operator's
    joint confidence budget. The forecasts are frozen before these rows.
    """
    a, b, c, cap, phi = map(np.asarray, (a, b, c, cap, basis))
    n, d = phi.shape
    if np.max(a, initial=0) < 1e-16:
        return np.zeros(d), {"status": "identical_experts", "weights": [0.] * d, "selected_lambdas": [0.] * d, "n": n, "nominal_only": True}
    joint = np.asarray(lambdas if joint_lambdas is None else joint_lambdas, float)
    raw = (b[:, None, None] - c[:, None, None] * joint[None, :, None]) * phi[:, None, :]
    pairs = [(i, j) for i in range(d) for j in range(i, d)]
    astat = np.stack([a * phi[:, i] * phi[:, j] for i, j in pairs], axis=1)
    kstat = cap[:, None] * phi
    stats = np.concatenate([raw.reshape(n, -1), astat, kstat], axis=1)
    mean = stats.mean(axis=0)
    replicas = stats[indices].mean(axis=1)
    se = replicas.std(axis=0, ddof=1)
    standardized = (replicas - mean) / np.where(se > 1e-14, se, 1)
    standardized[:, se <= 1e-14] = 0
    nb, na = len(joint) * d, len(pairs)
    # Lower B, two-sided A, upper K; one shared maximum protects selection.
    relevant = np.concatenate([standardized[:, :nb], np.abs(standardized[:, nb:nb+na]), -standardized[:, nb+na:]], axis=1)
    q = max(0., float(np.quantile(np.max(relevant, axis=1), confidence, method="higher")))
    raw_lower = (mean[:nb] - q * se[:nb]).reshape(len(joint), d)
    k_upper = np.maximum(0., mean[nb+na:] + q * se[nb+na:])
    lower_by_lambda = raw_lower - np.abs(joint[:, None]) * sign_budget * k_upper[None, :]
    permitted = [i for i, value in enumerate(joint) if np.any(np.isclose(value, lambdas))]
    permitted = np.asarray(permitted)
    chosen = permitted[np.argmax(lower_by_lambda[permitted], axis=0)]
    lower = lower_by_lambda[chosen, np.arange(d)]
    amean, radius = np.zeros((d, d)), np.zeros((d, d))
    for z, (i, j) in enumerate(pairs):
        amean[i, j] = amean[j, i] = mean[nb + z]
        radius[i, j] = radius[j, i] = q * se[nb + z]
    upper = amean + np.diag(radius.sum(axis=1))
    w, upper_risk = box_quadratic(upper, lower)
    receipt = {
        "status": "fitted", "n": n, "weights": w.tolist(),
        "selected_lambdas": joint[chosen].tolist(), "lower_B": lower.tolist(),
        "upper_A": upper.tolist(), "upper_K_basis": k_upper.tolist(),
        "bootstrap_joint_quantile": q, "upper_risk_difference": upper_risk,
        "sign_budget": sign_budget, "nominal_only": True,
        "control_variance_ratio_lambda1": float(np.var(b-c, ddof=1) / max(np.var(b, ddof=1), 1e-30)),
        "mean_control": float(np.mean(c)), "mean_cap": float(np.mean(cap)),
    }
    return w, receipt


def empirical_bernstein(values, range_length, eta):
    values = np.asarray(values)
    n = values.shape[-1]
    if n < 2:
        return np.full(values.shape[:-1], np.inf)
    log = np.log(2 / eta)
    return np.sqrt(2 * np.var(values, axis=-1, ddof=1) * log / n) + 7 * range_length * log / (3 * (n - 1))


def bernstein_scalar(a_known, b, c, cap_known, b_range, sign_budget,
                     lambdas=(0., .5, 1.), delta=.05):
    """Independent samples, frozen fixed experts, known constant a and cap.

    The generic support of b-lambda*c uses b_range + 2*abs(lambda)*cap.
    This deliberately retains range costs even if an observed variance is zero.
    """
    if a_known <= 1e-16:
        return np.zeros(np.shape(b)[:-1]), np.zeros(np.shape(b)[:-1])
    values = []
    for lam in lambdas:
        corrected = b - lam * c
        radius = empirical_bernstein(corrected, b_range + 2 * abs(lam) * cap_known, delta / len(lambdas))
        values.append(corrected.mean(axis=-1) - radius - abs(lam) * sign_budget * cap_known)
    lower = np.stack(values, axis=-1)
    selection = np.argmax(lower, axis=-1)
    best = np.max(lower, axis=-1)
    return np.clip(best / a_known, 0, 1), np.asarray(lambdas)[selection]
