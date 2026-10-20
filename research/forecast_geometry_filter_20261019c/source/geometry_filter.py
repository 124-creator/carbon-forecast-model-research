"""Price-free finite-state filtering and exact Gaussian-mixture calculations.

The current latent state is S_t.  Every future observation first transitions
with P and then emits a zero-mean Gaussian.  Future targets are cumulative
returns divided by sqrt(h).  No market files, fitted objects, random numbers,
or disk IO are used by this module.

Public interfaces
-----------------
build_reference(transition=None, horizons=(1, 5, 20))
conditional_mixture(transition, variances, horizon) -> weights[K,M], sds[M]
normal_abs(y, sds) -> E|N(0,sds**2)-y|, with outer broadcasting
pair_abs(sds_left, sds_right=None) -> E|Z_left-Z_right|
mixture_crps(q, y, weights, sds, pair=None) -> score per observation
update_qp(p, y, eta, G, kappa=.01, ratio_preserving=False,
          return_receipt=False) -> q, or (q, aggregate numerical receipt)
update_crps_kl(p, y, eta, return_receipt=False) -> q or (q, receipt)
tempered_bayes(p, y, beta=1.) -> q
contaminated_bayes(p, y, epsilon, shock_variance=16.) -> q
get_numerical_counters() / reset_numerical_counters()

Update functions accept p[...,4], one observation per row, and scalar or
rowwise step sizes. A one-dimensional p and scalar observation/step return
a one-dimensional q. eta=0 is the exactly preserved prior, not a limit
implemented using a tiny positive step.  Input priors must already sum to one;
normal floating-point summation error is tolerated but not silently repaired.

All metrics in reference['metrics'] are normalized *after* combining horizons.
reference['current_G'] and each conditional['gram'] are unnormalized matrices.
The optional 'pair' in mixture_crps is the K-by-K expected absolute distance
between the conditional component laws, NOT the M-by-M Gaussian kernel unless
the conditional weights are the identity.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from scipy.special import erf, logsumexp


K = 4
PI = np.full(K, 0.25, dtype=np.float64)
V = np.array([0.25, 1.0, 1.0, 4.0], dtype=np.float64) / 1.5625
SDS = np.sqrt(V)
CENTER = np.eye(K, dtype=np.float64) - np.ones((K, K)) / K
_T_SLOW = 0.97 * np.eye(2) + 0.03 * np.full((2, 2), 0.5)
_T_FAST = 0.10 * np.eye(2) + 0.90 * np.full((2, 2), 0.5)
TRANSITION = np.kron(_T_SLOW, _T_FAST)
_LATTICE_INCREMENT = np.array([0, 1, 1, 5], dtype=np.int64)
_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
_LOG_2PI = np.log(2.0 * np.pi)
CURRENT_PAIR = _SQRT_2_OVER_PI * np.sqrt(V[:, None] + V[None, :])
CURRENT_GRAM = -0.5 * CENTER @ CURRENT_PAIR @ CENTER
CURRENT_GRAM = 0.5 * (CURRENT_GRAM + CURRENT_GRAM.T)

# This PSD factorization is used only for the equivalent CRPS+KL dual.
# Duplicate middle emissions give exact rank two; roundoff eigenvalues are
# excluded. The original, unfactored CRPS gradient is used in the KKT receipt.
_GRAM_EIGENVALUES, _GRAM_EIGENVECTORS = np.linalg.eigh(CURRENT_GRAM)
_POSITIVE_EIGENVALUES = _GRAM_EIGENVALUES > 1e-14
_L2 = _GRAM_EIGENVECTORS[:, _POSITIVE_EIGENVALUES] * np.sqrt(
    2.0 * _GRAM_EIGENVALUES[_POSITIVE_EIGENVALUES]
)[None, :]

_COUNTER_DEFAULTS = {
    "qp_update_calls": 0,
    "qp_update_rows": 0,
    "qp_zero_step_identity_rows": 0,
    "qp_simplex_solve_calls": 0,
    "qp_solved_rows": 0,
    "qp_full_face_fast_rows": 0,
    "qp_enumerated_rows": 0,
    "qp_negative_preclip_coordinates": 0,
    "qp_negative_preclip_rows": 0,
    "qp_renormalized_rows": 0,
    "qp_repaired_coordinates": 0,
    "qp_repaired_rows": 0,
    "qp_max_negative_preclip_abs": 0.0,
    "qp_max_sum_error_after_clip_before_normalization": 0.0,
    "qp_max_abs_repair": 0.0,
    "kl_update_calls": 0,
    "kl_update_rows": 0,
    "kl_zero_step_identity_rows": 0,
    "kl_optimized_rows": 0,
    "kl_converged_rows": 0,
    "kl_newton_iterations": 0,
    "kl_max_iterations_used": 0,
    "kl_line_search_halvings": 0,
}
NUMERICAL_COUNTERS: dict[str, int | float] = _COUNTER_DEFAULTS.copy()


def reset_numerical_counters() -> None:
    """Reset process-local counters before a separately recorded execution.

    Counts concern chosen simplex solutions only, not discarded candidate
    faces. A ratio-preserving solve has three optimization coordinates; a
    regular solve has four. Concurrent processes must each return their own
    counters. Numerical repairs are never omitted merely because the caller
    did not request the per-call receipt.
    """
    NUMERICAL_COUNTERS.clear()
    NUMERICAL_COUNTERS.update(_COUNTER_DEFAULTS)


def get_numerical_counters() -> dict[str, Any]:
    """Return a copy, including the pre-fixed feasibility tolerance."""
    return {
        **NUMERICAL_COUNTERS,
        "qp_feasibility_tolerance": 5e-12,
        "coordinate_scope": "chosen optimization coordinates: 4 ordinary, 3 ratio-preserving",
        "scope": "process-local since the most recent explicit reset",
    }


def _finite_vector(value: Any, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.ndim > 1 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite scalar or nonempty vector")
    return a.reshape(-1)


def _prepare_inputs(
    p: Any, y: Any, step: Any, step_name: str = "eta"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    original_p = np.asarray(p, dtype=np.float64)
    prior = original_p[None, :] if original_p.ndim == 1 else original_p
    if prior.ndim != 2 or prior.shape[1] != K or prior.shape[0] == 0:
        raise ValueError("p must have shape (4,) or (batch,4)")
    if not np.all(np.isfinite(prior)) or np.any(prior <= 0.0):
        raise ValueError("Every input prior probability must be finite and >0")
    if np.max(np.abs(prior.sum(axis=1) - 1.0)) > 1e-10:
        raise ValueError("Input priors must already sum to one")
    observations = _finite_vector(y, "y")
    steps = _finite_vector(step, step_name)
    if np.any(steps < 0.0):
        raise ValueError(f"{step_name} must be nonnegative")
    n = max(prior.shape[0], observations.size, steps.size)
    if any(k not in (1, n) for k in (prior.shape[0], observations.size, steps.size)):
        raise ValueError("Prior, observations and steps have incompatible batches")
    prior = np.broadcast_to(prior, (n, K))
    observations = np.broadcast_to(observations, (n,))
    steps = np.broadcast_to(steps, (n,))
    scalar = original_p.ndim == 1 and np.ndim(y) == 0 and np.ndim(step) == 0
    return prior, observations, steps, scalar


def _restore(q: np.ndarray, scalar: bool) -> np.ndarray:
    return q[0] if scalar else q


def normal_abs(y: Any, sds: Any) -> np.ndarray:
    """Absolute normal moments, outer-broadcasting y[...,None] against sds.

    Arbitrary finite nonnegative standard deviations, including zero, are
    accepted. Zero standard deviation represents an actual point mass at zero.
    """
    observations = np.asarray(y, dtype=np.float64)
    scales = _finite_vector(sds, "sds")
    if not np.all(np.isfinite(observations)) or np.any(scales < 0.0):
        raise ValueError("y must be finite and sds finite/nonnegative")
    yy = observations[..., None]
    positive = scales > 0.0
    safe_sd = np.where(positive, scales, 1.0)
    z = yy / safe_sd
    with np.errstate(over="ignore", under="ignore"):
        result = _SQRT_2_OVER_PI * safe_sd * np.exp(-0.5 * z * z)
        result = result + yy * erf(z / np.sqrt(2.0))
    return np.where(positive, result, np.abs(yy))


def pair_abs(sds_left: Any, sds_right: Any = None) -> np.ndarray:
    """Pairwise E|Z_i-Z'_j| for independent zero-mean Gaussians."""
    left = _finite_vector(sds_left, "sds_left")
    right = left if sds_right is None else _finite_vector(sds_right, "sds_right")
    if np.any(left < 0.0) or np.any(right < 0.0):
        raise ValueError("Standard deviations must be nonnegative")
    return _SQRT_2_OVER_PI * np.hypot(left[:, None], right[None, :])


def mixture_crps(
    q: Any, y: Any, weights: Any, sds: Any, pair: Any = None
) -> np.ndarray:
    """Exact CRPS of the mixture q @ weights, per batch observation.

    Each row of weights is a conditional Gaussian-scale-mixture law. It may
    include an arbitrary number of scales, such as measurement-shock counts.
    The two mixture draws in the absolute-distance term are independent.
    """
    q0 = np.asarray(q, dtype=np.float64)
    probs = q0[None, :] if q0.ndim == 1 else q0
    w = np.asarray(weights, dtype=np.float64)
    scales = _finite_vector(sds, "sds")
    yy = _finite_vector(y, "y")
    if probs.ndim != 2 or w.ndim != 2 or probs.shape[1] != w.shape[0]:
        raise ValueError("q must be (batch,K) and weights (K,M)")
    if scales.size != w.shape[1] or np.any(scales < 0.0):
        raise ValueError("sds must match the Gaussian-mixture support")
    if (not np.all(np.isfinite(probs)) or not np.all(np.isfinite(w))
            or np.any(probs < 0.0) or np.any(w < 0.0)):
        raise ValueError("Mixture probabilities must be finite and nonnegative")
    if (np.max(np.abs(probs.sum(axis=1) - 1.0)) > 1e-10
            or np.max(np.abs(w.sum(axis=1) - 1.0)) > 1e-10):
        raise ValueError("Both levels of mixture weights must sum to one")
    n = max(probs.shape[0], yy.size)
    if probs.shape[0] not in (1, n) or yy.size not in (1, n):
        raise ValueError("q and y have incompatible batches")
    probs = np.broadcast_to(probs, (n, probs.shape[1]))
    yy = np.broadcast_to(yy, (n,))
    a = w @ pair_abs(scales) @ w.T if pair is None else np.asarray(pair, dtype=float)
    if a.shape != (w.shape[0], w.shape[0]) or not np.all(np.isfinite(a)):
        raise ValueError("pair must be the finite K-by-K conditional distance matrix")
    b = normal_abs(yy, scales) @ w.T
    score = np.sum(probs * b, axis=1) - 0.5 * np.einsum(
        "bi,ij,bj->b", probs, a, probs
    )
    return score[0] if q0.ndim == 1 and np.ndim(y) == 0 else score


def _validate_transition(transition: Any) -> np.ndarray:
    p = np.asarray(transition, dtype=np.float64)
    if p.shape != (K, K) or not np.all(np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("transition must be a finite nonnegative 4-by-4 matrix")
    if np.max(np.abs(p.sum(axis=1) - 1.0)) > 1e-12:
        raise ValueError("transition rows must sum to one")
    return p


def _conditional_mixtures(
    transition: Any, variances: Any, horizons: tuple[int, ...]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    p = _validate_transition(transition)
    v = _finite_vector(variances, "variances")
    if v.shape != (K,) or np.any(v <= 0.0):
        raise ValueError("variances must contain four positive values")
    lattice_v = v[0] * (1.0 + 3.0 * _LATTICE_INCREMENT)
    if not np.allclose(v, lattice_v, rtol=1e-13, atol=1e-15):
        raise ValueError("Exact lattice DP requires v proportional to [1,4,4,16]")
    if (not horizons or any(not isinstance(h, (int, np.integer)) or h < 1
                            for h in horizons)):
        raise ValueError("horizons must be positive integers")
    wanted = set(int(h) for h in horizons)
    # Axes: initial CURRENT state, latest future state, cumulative lattice cost.
    state = np.eye(K, dtype=np.float64)[:, :, None]
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for h in range(1, max(wanted) + 1):
        moved = np.einsum("ism,sj->ijm", state, p, optimize=True)
        old_size = moved.shape[-1]
        state = np.zeros((K, K, old_size + 5), dtype=np.float64)
        for j, offset in enumerate(_LATTICE_INCREMENT):
            state[:, j, offset:offset + old_size] = moved[:, j, :]
        if h in wanted:
            w = state.sum(axis=1)
            sds = np.sqrt(v[0] * (h + 3.0 * np.arange(5 * h + 1)) / h)
            result[h] = (w, sds)
    return result


def conditional_mixture(
    transition: Any, variances: Any, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Law of sum_{j=1}^h r_{t+j}/sqrt(h), conditional on CURRENT S_t.

    The transition is taken before each emission, including the first future
    emission. The current observation is not added to the cumulative target.
    Scales must be proportional to the fixed [1,4,4,16] variance lattice.
    """
    return _conditional_mixtures(transition, variances, (horizon,))[int(horizon)]


def _normalize_metric(g: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    symmetric = 0.5 * (g + g.T)
    covariance = np.diag(PI) - np.outer(PI, PI)
    c = float(np.trace(covariance @ symmetric))
    threshold = 1e-14 * max(1.0, float(np.linalg.norm(symmetric, ord=2)))
    if c < -threshold:
        raise ValueError("A metric has materially negative normalization")
    is_zero = c <= threshold
    normalized = np.zeros_like(symmetric) if is_zero else symmetric / c
    receipt = {
        "denominator": c,
        "zero_threshold": threshold,
        "degenerate_returned_zero": bool(is_zero),
        "raw_min_eigenvalue": float(np.linalg.eigvalsh(symmetric)[0]),
        "normalized_expected_quadratic": float(np.trace(covariance @ normalized)),
        "normalized_centering_residual": float(np.max(np.abs(normalized @ np.ones(K)))),
    }
    return normalized, receipt


def build_reference(
    transition: Any = None, horizons: tuple[int, ...] = (1, 5, 20)
) -> dict[str, Any]:
    """Build exact reference distributions and the four normalized metrics.

    Uniform pi remains the declared stationary distribution. A replacement
    transition must preserve it; swapping slow/fast factors is permitted.
    The required metric horizons 1,5,20 are always included.
    """
    p = _validate_transition(TRANSITION if transition is None else transition)
    if np.max(np.abs(PI @ p - PI)) > 1e-12:
        raise ValueError("build_reference requires uniform stationary pi")
    requested = tuple(sorted(set(horizons) | {1, 5, 20}))
    laws = _conditional_mixtures(p, V, requested)
    conditional: dict[int, dict[str, Any]] = {}
    for h, (w, sds) in laws.items():
        a = w @ pair_abs(sds) @ w.T
        a = 0.5 * (a + a.T)
        gram = -0.5 * CENTER @ a @ CENTER
        gram = 0.5 * (gram + gram.T)
        conditional[h] = {
            "weights": w,
            "sds": sds,
            "pair": a,
            "gram": gram,
            "variance": w @ (sds * sds),
            "max_mass_error": float(np.max(np.abs(w.sum(axis=1) - 1.0))),
        }
    u5 = CENTER @ conditional[5]["variance"]
    u20 = CENTER @ conditional[20]["variance"]
    raw = {
        "future": 0.5 * (conditional[5]["gram"] + conditional[20]["gram"]),
        "current": CURRENT_GRAM.copy(),
        "variance": 0.5 * (np.outer(u5, u5) + np.outer(u20, u20)),
        "euclid": CENTER.copy(),
    }
    metrics: dict[str, np.ndarray] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for name, g in raw.items():
        metrics[name], receipts[name] = _normalize_metric(g)
    return {
        "P": p.copy(), "pi": PI.copy(), "v": V.copy(),
        "emission_sd": SDS.copy(), "current_pair": CURRENT_PAIR.copy(),
        "current_G": CURRENT_GRAM.copy(), "conditional": conditional,
        "metrics": metrics, "raw_metrics": raw,
        "normalization_receipts": receipts,
        "future_convention": "transition_then_emit; cumulative_return/sqrt(h)",
    }


def _face_solution(h: np.ndarray, c: np.ndarray, support: tuple[int, ...]) -> np.ndarray:
    """Solve the equality-constrained stationary point on a simplex face.

    The full-space Hessian need not be positive definite. After eliminating
    the last face coordinate its tangent-space Hessian is strictly positive
    definite because kappa>0 and all prior probabilities are positive.
    """
    n, d = c.shape
    result = np.zeros((n, d), dtype=np.float64)
    if len(support) == 1:
        result[:, support[0]] = 1.0
        return result
    idx = np.asarray(support, dtype=int)
    hs = h[:, idx, :][:, :, idx]
    cs = c[:, idx]
    tangent = (hs[:, :-1, :-1] - hs[:, :-1, -1][:, :, None]
               - hs[:, -1, :-1][:, None, :] + hs[:, -1, -1][:, None, None])
    rhs = -(cs[:, :-1] - cs[:, -1, None]
            + hs[:, :-1, -1] - hs[:, -1, -1, None])
    free = np.linalg.solve(tangent, rhs[..., None])[..., 0]
    result[:, idx[:-1]] = free
    result[:, idx[-1]] = 1.0 - free.sum(axis=1)
    return result


def _simplex_qp(
    h: np.ndarray, c: np.ndarray, anchor: np.ndarray
) -> tuple[np.ndarray, dict[str, int | float]]:
    n, d = c.shape
    result = _face_solution(h, c, tuple(range(d)))
    feasible_tol = 5e-12
    full_ok = result.min(axis=1) >= -feasible_tol
    remaining = np.flatnonzero(~full_ok)
    if remaining.size:
        hh, cc, aa = h[remaining], c[remaining], anchor[remaining]
        gradient_at_anchor = np.einsum("bij,bj->bi", hh, aa) + cc
        best = np.full(remaining.size, np.inf)
        best_q = np.zeros_like(aa)
        for size in range(1, d):
            for support in combinations(range(d), size):
                trial = _face_solution(hh, cc, support)
                feasible = trial.min(axis=1) >= -feasible_tol
                delta = trial - aa
                objective = (0.5 * np.einsum("bi,bij,bj->b", delta, hh, delta)
                             + np.sum(gradient_at_anchor * delta, axis=1))
                better = feasible & (objective < best)
                best[better] = objective[better]
                best_q[better] = trial[better]
        if not np.all(np.isfinite(best)):
            raise FloatingPointError("Simplex enumeration found no feasible vertex")
        result[remaining] = best_q
    # Only numerical negatives admitted by feasible_tol are repaired. These
    # corrections and all equality/optimality residuals remain inspectable.
    before_repair = result.copy()
    result = np.maximum(result, 0.0)
    sums_after_clip = result.sum(axis=1, keepdims=True)
    result /= sums_after_clip
    changed = result != before_repair
    negative_before_clip = before_repair < 0.0
    negative_abs = float(np.max(np.maximum(-before_repair, 0.0)))
    sum_error = float(np.max(np.abs(sums_after_clip - 1.0)))
    repair_size = float(np.max(np.abs(result - before_repair)))
    info = {
        "full_face_fast_rows": int(full_ok.sum()),
        "enumerated_rows": int(remaining.size),
        "additional_faces_per_enumerated_row": int(2**d - 2),
        "negative_preclip_coordinates": int(negative_before_clip.sum()),
        "negative_preclip_rows": int(np.any(negative_before_clip, axis=1).sum()),
        "renormalized_rows": int(np.count_nonzero(sums_after_clip[:, 0] != 1.0)),
        "repaired_coordinates": int(changed.sum()),
        "repaired_rows": int(np.any(changed, axis=1).sum()),
        "max_negative_preclip_abs": negative_abs,
        "max_sum_error_after_clip_before_normalization": sum_error,
        "max_feasibility_repair": repair_size,
    }
    NUMERICAL_COUNTERS["qp_simplex_solve_calls"] += 1
    NUMERICAL_COUNTERS["qp_solved_rows"] += n
    for key in (
        "full_face_fast_rows", "enumerated_rows", "negative_preclip_coordinates",
        "negative_preclip_rows", "renormalized_rows", "repaired_coordinates", "repaired_rows",
    ):
        NUMERICAL_COUNTERS["qp_" + key] += info[key]
    for key, value in (
        ("qp_max_negative_preclip_abs", negative_abs),
        ("qp_max_sum_error_after_clip_before_normalization", sum_error),
        ("qp_max_abs_repair", repair_size),
    ):
        NUMERICAL_COUNTERS[key] = max(NUMERICAL_COUNTERS[key], value)
    return result, info


def _qp_kkt(h: np.ndarray, c: np.ndarray, q: np.ndarray) -> dict[str, float]:
    gradient = np.einsum("bij,bj->bi", h, q) + c
    active = q > 0.0
    multiplier = np.sum(np.where(active, gradient, 0.0), axis=1) / active.sum(axis=1)
    active_error = np.max(np.where(active, np.abs(gradient - multiplier[:, None]), 0.0))
    dual_error = np.max(np.where(~active, np.maximum(multiplier[:, None] - gradient, 0.0), 0.0))
    equality_error = np.max(np.abs(q.sum(axis=1) - 1.0))
    negativity = max(0.0, float(-np.min(q)))
    return {
        "max_active_stationarity_residual": float(active_error),
        "max_inactive_dual_violation": float(dual_error),
        "max_simplex_equality_residual": float(equality_error),
        "max_negativity": negativity,
        "max_kkt_residual": float(max(active_error, dual_error, equality_error, negativity)),
    }


def update_qp(
    p: Any, y: Any, eta: Any, G: Any, kappa: float = 0.01,
    ratio_preserving: bool = False, return_receipt: bool = False,
) -> Any:
    """Exact simplex active-set solution of the registered CRPS quadratic.

    min_q CRPS(sum_i q_i N(0,v_i), y)
          + ((q-p)'G(q-p) + kappa*sum_i (q_i-p_i)^2/p_i)/eta.

    G=0,kappa=1 is the chi-square-only control. ratio_preserving enforces
    q[1]/p[1] == q[2]/p[2] by solving the three-coordinate collapsed problem,
    including its transformed loss and penalty, not by post-hoc adjustment.
    """
    prior, yy, steps, scalar = _prepare_inputs(p, y, eta)
    if not np.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("kappa must be finite and strictly positive")
    metric = np.asarray(G, dtype=np.float64)
    if metric.shape not in ((K, K), (len(prior), K, K)):
        raise ValueError("G must have shape (4,4) or (batch,4,4)")
    if not np.all(np.isfinite(metric)):
        raise ValueError("G must be finite")
    if np.max(np.abs(metric - np.swapaxes(metric, -1, -2))) > 1e-10:
        raise ValueError("G must be symmetric")
    if np.min(np.linalg.eigvalsh(metric)) < -1e-10 * max(1.0, float(np.max(np.abs(metric)))):
        raise ValueError("G must be positive semidefinite up to roundoff")
    metric = np.broadcast_to(metric, (len(prior), K, K))
    q = prior.copy()
    used = np.flatnonzero(steps > 0.0)
    NUMERICAL_COUNTERS["qp_update_calls"] += 1
    NUMERICAL_COUNTERS["qp_update_rows"] += len(prior)
    NUMERICAL_COUNTERS["qp_zero_step_identity_rows"] += int((steps == 0.0).sum())
    receipt: dict[str, Any] = {
        "algorithm": "simplex_tangent_spd_active_set",
        "rows": int(len(prior)), "zero_step_identity_rows": int((steps == 0.0).sum()),
        "ratio_preserving": bool(ratio_preserving), "kappa": float(kappa),
        "max_kkt_residual": 0.0, "full_face_fast_rows": 0, "enumerated_rows": 0,
        "max_feasibility_repair": 0.0,
    }
    if used.size:
        pp, ee, gy = prior[used], steps[used], metric[used]
        r = gy.copy()
        r[:, np.arange(K), np.arange(K)] += kappa / pp
        hessian = -CURRENT_PAIR[None, :, :] + 2.0 * r / ee[:, None, None]
        b = normal_abs(yy[used], SDS)
        b -= b.mean(axis=1, keepdims=True)  # an irrelevant common loss constant
        linear = b - 2.0 * np.einsum("bij,bj->bi", r, pp) / ee[:, None]
        if ratio_preserving:
            transform = np.zeros((len(used), K, 3))
            transform[:, 0, 0] = 1.0
            middle = pp[:, 1] + pp[:, 2]
            transform[:, 1, 1] = pp[:, 1] / middle
            transform[:, 2, 1] = pp[:, 2] / middle
            transform[:, 3, 2] = 1.0
            hh = np.einsum("bia,bij,bjc->bac", transform, hessian, transform)
            cc = np.einsum("bia,bi->ba", transform, linear)
            anchor = np.column_stack((pp[:, 0], middle, pp[:, 3]))
            reduced, info = _simplex_qp(hh, cc, anchor)
            solution = np.einsum("bia,ba->bi", transform, reduced)
            if return_receipt:
                receipt.update(_qp_kkt(hh, cc, reduced))
                receipt["max_ratio_constraint_residual"] = float(np.max(np.abs(
                    solution[:, 1] * pp[:, 2] - solution[:, 2] * pp[:, 1]
                )))
        else:
            solution, info = _simplex_qp(hessian, linear, pp)
            if return_receipt:
                receipt.update(_qp_kkt(hessian, linear, solution))
        q[used] = solution
        if return_receipt:
            receipt.update(info)
            delta = solution - pp
            penalty = (np.einsum("bi,bij,bj->b", delta, gy, delta)
                       + kappa * np.sum(delta * delta / pp, axis=1)) / ee
            original_score = mixture_crps(solution, yy[used], np.eye(K), SDS, CURRENT_PAIR)
            receipt["min_original_objective"] = float(np.min(original_score + penalty))
            receipt["max_original_objective"] = float(np.max(original_score + penalty))
    answer = _restore(q, scalar)
    return (answer, receipt) if return_receipt else answer


def _dual_value(
    u: np.ndarray, logp: np.ndarray, c: np.ndarray, eta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    logits = logp - eta[:, None] * (c + u @ _L2.T)
    logz = logsumexp(logits, axis=1)
    q = np.exp(logits - logz[:, None])
    value = 0.5 * np.sum(u * u, axis=1) + logz / eta
    return value, q


def update_crps_kl(
    p: Any, y: Any, eta: Any, return_receipt: bool = False,
    tolerance: float = 1e-12, max_iter: int = 100,
) -> Any:
    """Strictly convex CRPS(q,y)+KL(q||p)/eta via an equivalent dual Newton.

    On the simplex write CRPS=c'q+q'G0 q+constant, G0=L L'. The dual is
        min_u .5||u||^2 + logsumexp(log(p)-eta*(c+sqrt(2)*L*u))/eta.
    Its Hessian is I+2*eta*L'Cov(q)L, always positive definite. This avoids
    ill-conditioning from tiny primal q at large eta. The receipt checks
    the original CRPS+KL KKT equation using the unmodified distance matrix.

    Backtracking uses the stationary-equation merit .5*||gradient||**2.
    For the unchanged Newton direction H*d=-gradient, its directional
    derivative is -||gradient||**2. Because H is positive definite, a
    stationary point of this merit is exactly the original dual minimizer.
    This avoids subtracting rounded objective values when the genuine
    Newton decrease is far below binary64 precision near the solution.
    """
    prior, yy, steps, scalar = _prepare_inputs(p, y, eta)
    if not np.isfinite(tolerance) or tolerance <= 0 or max_iter < 1:
        raise ValueError("Positive tolerance and max_iter required")
    answer = prior.copy()
    used = np.flatnonzero(steps > 0.0)
    NUMERICAL_COUNTERS["kl_update_calls"] += 1
    NUMERICAL_COUNTERS["kl_update_rows"] += len(prior)
    NUMERICAL_COUNTERS["kl_zero_step_identity_rows"] += int((steps == 0.0).sum())
    NUMERICAL_COUNTERS["kl_optimized_rows"] += int(used.size)
    receipt: dict[str, Any] = {
        "algorithm": "equivalent_strongly_convex_dual_newton",
        "line_search_merit": "half_squared_dual_gradient_norm",
        "rows": int(len(prior)), "zero_step_identity_rows": int((steps == 0.0).sum()),
        "tolerance": float(tolerance), "max_iter_budget": int(max_iter),
        "max_iterations": 0, "max_kkt_residual": 0.0,
        "max_dual_gradient_residual": 0.0, "total_backtracking_halvings": 0,
        "converged": True,
        "gram_factorization_max_error": float(np.max(np.abs(
            0.5 * _L2 @ _L2.T - CURRENT_GRAM
        ))),
    }
    if used.size:
        pp, ee = prior[used], steps[used]
        logp = np.log(pp)
        b = normal_abs(yy[used], SDS)
        c = b - (CURRENT_PAIR @ PI)[None, :]
        c -= c.mean(axis=1, keepdims=True)
        u = pp @ _L2
        counts = np.zeros(len(used), dtype=int)
        total_halvings = 0
        for _ in range(max_iter):
            _, qq = _dual_value(u, logp, c, ee)
            means = qq @ _L2
            gradient = u - means
            unfinished = np.flatnonzero(np.max(np.abs(gradient), axis=1) > tolerance)
            if unfinished.size == 0:
                break
            qq_a, mean_a, grad_a = qq[unfinished], means[unfinished], gradient[unfinished]
            covariance = np.einsum("bi,ij,ik->bjk", qq_a, _L2, _L2)
            covariance -= mean_a[:, :, None] * mean_a[:, None, :]
            hessian = (np.eye(_L2.shape[1])[None, :, :]
                       + ee[unfinished, None, None] * covariance)
            direction = -np.linalg.solve(hessian, grad_a[..., None])[..., 0]
            squared_residual = np.sum(grad_a * grad_a, axis=1)
            alpha = np.ones(len(unfinished))
            old_u = u[unfinished].copy()
            for _line in range(50):
                trial_u = old_u + alpha[:, None] * direction
                _, trial_q = _dual_value(
                    trial_u, logp[unfinished], c[unfinished], ee[unfinished]
                )
                trial_gradient = trial_u - trial_q @ _L2
                trial_squared_residual = np.sum(trial_gradient * trial_gradient, axis=1)
                accepted = trial_squared_residual <= (1.0 - 2e-4 * alpha) * squared_residual
                if np.all(accepted):
                    break
                alpha[~accepted] *= 0.5
                halvings = int((~accepted).sum())
                total_halvings += halvings
                NUMERICAL_COUNTERS["kl_line_search_halvings"] += halvings
            else:
                raise FloatingPointError("CRPS+KL dual Newton line search did not converge")
            u[unfinished] = trial_u
            counts[unfinished] += 1
            NUMERICAL_COUNTERS["kl_newton_iterations"] += int(unfinished.size)
            NUMERICAL_COUNTERS["kl_max_iterations_used"] = max(
                NUMERICAL_COUNTERS["kl_max_iterations_used"], int(counts.max())
            )
        _, qq = _dual_value(u, logp, c, ee)
        residual = np.max(np.abs(u - qq @ _L2), axis=1)
        if np.any(residual > tolerance):
            raise FloatingPointError(
                f"CRPS+KL Newton budget exceeded; dual residual={float(residual.max()):.3g}"
            )
        if np.any(qq <= 0.0):
            raise FloatingPointError("CRPS+KL interior probabilities underflowed; no clipping applied")
        NUMERICAL_COUNTERS["kl_converged_rows"] += int(used.size)
        answer[used] = qq
        if return_receipt:
            gradient = b - qq @ CURRENT_PAIR.T + np.log(qq / pp) / ee[:, None]
            centered = gradient - gradient.mean(axis=1, keepdims=True)
            receipt.update({
                "max_iterations": int(counts.max()),
                "max_kkt_residual": float(np.max(np.abs(centered))),
                "max_simplex_equality_residual": float(np.max(np.abs(qq.sum(axis=1) - 1.0))),
                "max_dual_gradient_residual": float(residual.max()),
                "total_backtracking_halvings": int(total_halvings),
                "min_probability": float(qq.min()),
            })
    restored = _restore(answer, scalar)
    return (restored, receipt) if return_receipt else restored


def _normal_logpdf(y: np.ndarray, variances: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        out = -0.5 * (_LOG_2PI + np.log(variances)[None, :]
                      + y[:, None] * y[:, None] / variances[None, :])
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("Normal log density exceeded binary64 range")
    return out


def tempered_bayes(p: Any, y: Any, beta: Any = 1.0) -> np.ndarray:
    """q_i proportional to p_i * normal_pdf(y;0,v_i)**beta."""
    prior, yy, powers, scalar = _prepare_inputs(p, y, beta, "beta")
    q = prior.copy()
    used = np.flatnonzero(powers > 0.0)
    if used.size:
        logw = np.log(prior[used]) + powers[used, None] * _normal_logpdf(yy[used], V)
        q[used] = np.exp(logw - logsumexp(logw, axis=1, keepdims=True))
    return _restore(q, scalar)


def contaminated_bayes(
    p: Any, y: Any, epsilon: Any, shock_variance: float = 16.0
) -> np.ndarray:
    """Bayes update with (1-eps)N(0,v_i)+eps*N(0,v_i+shock_variance)."""
    prior, yy, eps, scalar = _prepare_inputs(p, y, epsilon, "epsilon")
    if np.any(eps > 1.0) or not np.isfinite(shock_variance) or shock_variance < 0.0:
        raise ValueError("epsilon must be in [0,1] and shock variance nonnegative")
    regular = _normal_logpdf(yy, V)
    contaminated = _normal_logpdf(yy, V + shock_variance)
    with np.errstate(divide="ignore"):
        log_likelihood = np.logaddexp(
            np.log1p(-eps)[:, None] + regular,
            np.log(eps)[:, None] + contaminated,
        )
    logw = np.log(prior) + log_likelihood
    q = np.exp(logw - logsumexp(logw, axis=1, keepdims=True))
    return _restore(q, scalar)


# Avoid accidental mutation of the reference constants by a caller.
for _constant in (PI, V, SDS, CENTER, TRANSITION, CURRENT_PAIR, CURRENT_GRAM, _L2):
    _constant.setflags(write=False)


__all__ = [
    "K", "PI", "V", "SDS", "CENTER", "TRANSITION", "CURRENT_PAIR", "CURRENT_GRAM",
    "build_reference", "conditional_mixture", "normal_abs", "pair_abs", "mixture_crps",
    "update_qp", "update_crps_kl", "tempered_bayes", "contaminated_bayes",
    "get_numerical_counters", "reset_numerical_counters", "NUMERICAL_COUNTERS",
]
