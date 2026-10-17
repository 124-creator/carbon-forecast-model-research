"""Standard statistical controls for the structural forecasting study.

No file access, training-data discovery, model imports, random search or tuning.
Both predictors return cumulative changes normalized by the query's fixed scale,
with shape (batch, 20).  The runner converts them to price forecasts.

State encoding is supplied by the caller from the ENTIRE visible prefix:
    (last_nonzero_sign + 1) * 3 + min(consecutive_zero_age, 2).
Event order is zero, negative, positive.  A zero preserves the last nonzero sign.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

HORIZON = 20
N_STATES = 9
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
STD_FLOOR = 1e-6
MAGNITUDE_FLOOR = 1e-4


@dataclass(frozen=True, eq=False)
class RidgeModel:
    """Intercept is coefficient row zero; all arrays are read-only copies."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    coefficients: np.ndarray
    alpha: float
    n_train: int
    channels: int


def _finite_array(value: Any, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _targets(batch: dict[str, Any]) -> np.ndarray:
    y = _finite_array(batch["y"], "y", 2)
    if y.shape[0] == 0 or y.shape[1] != HORIZON:
        raise ValueError("training y must have shape (N>0, 20)")
    return y


def _labels(value: Any, name: str, upper: int, n: int | None = None) -> np.ndarray:
    labels = _finite_array(value, name, 1)
    if n is not None and labels.shape != (n,):
        raise ValueError(f"{name} must have length {n}")
    if np.any(labels < 0) or np.any(labels >= upper):
        raise ValueError(f"{name} outside [0,{upper})")
    if np.any(labels != np.floor(labels)):
        raise ValueError(f"{name} must contain integer labels")
    return labels.astype(np.int64)


def _features(batch: dict[str, Any], channels: int | None = None) -> np.ndarray:
    x = _finite_array(batch["x"], "x", 3)
    if x.shape[1] != HORIZON or x.shape[2] < 1:
        raise ValueError("x must have shape (B, 20, C>=1)")
    if channels is not None and x.shape[2] != channels:
        raise ValueError("prediction channel count differs from training")
    return x


def _normalized_scale_weights(scale: Any, n: int) -> np.ndarray:
    """Compute scale**2 / mean(scale**2) without squaring a huge raw scale."""
    scale = _finite_array(scale, "scale", 1)
    if scale.shape != (n,) or np.any(scale <= 0):
        raise ValueError("scale must have length N and be strictly positive")
    relative_square = np.square(scale / np.max(scale))
    weights = relative_square / np.mean(relative_square)
    if not np.isfinite(weights).all():
        raise FloatingPointError("non-finite normalized scale weights")
    return weights


def _readonly(array: np.ndarray) -> np.ndarray:
    output = np.array(array, dtype=np.float64, copy=True)
    output.setflags(write=False)
    return output


def fit_ridge(train: dict[str, Any], alpha: float) -> RidgeModel:
    """Fit the specified weighted multi-output ridge; never select alpha.

    Raw flattened features use unweighted TRAIN population means and standard
    deviations, floored at 1e-6.  For X=[1, standardized_features] and normalized
    weights w, solve (X.T W X / N + alpha*diag(0,1,...)) beta=X.T W Y/N.
    """
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and strictly positive")
    x = _features(train)
    y = _targets(train)
    n = y.shape[0]
    if x.shape[0] != n:
        raise ValueError("x/y training row count mismatch")
    weights = _normalized_scale_weights(train["scale"], n)
    flat = x.reshape(n, -1)
    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0, ddof=0), STD_FLOOR)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise FloatingPointError("non-finite training feature moments")
    standardized = (flat - mean) / std
    design = np.concatenate((np.ones((n, 1)), standardized), axis=1)
    normal = design.T @ (weights[:, None] * design) / n
    penalty = np.full(design.shape[1], alpha, dtype=np.float64)
    penalty[0] = 0.0
    normal.flat[:: normal.shape[0] + 1] += penalty
    rhs = design.T @ (weights[:, None] * y) / n
    coefficients = np.linalg.solve(normal, rhs)
    if not np.isfinite(coefficients).all():
        raise FloatingPointError("non-finite ridge coefficients")
    return RidgeModel(
        feature_mean=_readonly(mean),
        feature_std=_readonly(std),
        coefficients=_readonly(coefficients),
        alpha=alpha,
        n_train=n,
        channels=x.shape[2],
    )


def predict_ridge(model: RidgeModel, batch: dict[str, Any]) -> np.ndarray:
    """Return normalized cumulative changes; do not rescale by query scale."""
    x = _features(batch, channels=model.channels)
    flat = x.reshape(x.shape[0], HORIZON * model.channels)
    standardized = (flat - model.feature_mean) / model.feature_std
    output = model.coefficients[0] + standardized @ model.coefficients[1:]
    if output.shape != (x.shape[0], HORIZON) or not np.isfinite(output).all():
        raise FloatingPointError("invalid ridge predictions")
    return output


def state_transition_table() -> np.ndarray:
    """Rows are pre-event states; columns are zero, negative, positive."""
    table = np.empty((N_STATES, 3), dtype=np.int64)
    for state in range(N_STATES):
        sign_block, zero_age = divmod(state, 3)
        table[state, 0] = 3 * sign_block + min(zero_age + 1, 2)
        table[state, 1] = 0
        table[state, 2] = 6
    return table


def fit_markov_mle(train: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Empirical Markov counts plus one global pseudo-observation per state.

    q_global=(global_counts+1)/(N+3).
    q[state]=(state_counts+q_global)/(state_count+1).
    b[state,direction] averages |y[:,0]| with one global direction-magnitude
    pseudo-observation. Missing global directions use mean |nonzero y[:,0]|;
    if every first-step change is zero, both global magnitudes default to one.

    This is a smoothed standard baseline, not an unsmoothed MLE or a new model.
    """
    y = _targets(train)
    n = y.shape[0]
    state = _labels(train["state"], "state", N_STATES, n)
    event = _labels(train["event"], "event", 3, n)
    first = y[:, 0]
    semantic_ok = (
        ((event == 0) & (first == 0))
        | ((event == 1) & (first < 0))
        | ((event == 2) & (first > 0))
    )
    if not semantic_ok.all():
        raise ValueError("event must match the sign of normalized first-step y")
    counts = np.zeros((N_STATES, 3), dtype=np.float64)
    np.add.at(counts, (state, event), 1.0)
    global_q = (counts.sum(axis=0) + 1.0) / (n + 3.0)
    q = (counts + global_q[None, :]) / (counts.sum(axis=1, keepdims=True) + 1.0)

    magnitude = np.abs(first)
    nonzero = event != 0
    global_abs = float(magnitude[nonzero].mean()) if nonzero.any() else 1.0
    b = np.empty((N_STATES, 2), dtype=np.float64)
    for direction, event_id in enumerate((1, 2)):
        direction_mask = event == event_id
        global_direction = (
            float(magnitude[direction_mask].mean())
            if direction_mask.any()
            else global_abs
        )
        sums = np.bincount(
            state[direction_mask],
            weights=magnitude[direction_mask],
            minlength=N_STATES,
        )
        b[:, direction] = (
            sums + global_direction
        ) / (counts[:, event_id] + 1.0)
    b = np.maximum(b, np.nextafter(MAGNITUDE_FLOOR, np.inf))
    if not np.isfinite(q).all() or not np.isfinite(b).all():
        raise FloatingPointError("non-finite Markov parameters")
    return _readonly(q), _readonly(b)


def predict_markov_mle(
    model: tuple[np.ndarray, np.ndarray], batch: dict[str, Any]
) -> np.ndarray:
    """Exact cumulative reward means under a fixed query-context Markov law.

    b uses the query's normalized units for every future step.  No future
    rescaling, future context recomputation or simulated/sampled path is used.
    Bellman recursion v_h=r+P v_(h-1) is independent of the neural model code.
    """
    q = _finite_array(model[0], "q", 2)
    b = _finite_array(model[1], "b", 2)
    if q.shape != (N_STATES, 3) or b.shape != (N_STATES, 2):
        raise ValueError("Markov q/b must have shapes (9,3)/(9,2)")
    if np.any(q < 0) or not np.allclose(q.sum(axis=1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("q rows must be nonnegative probabilities summing to one")
    if np.any(b <= 0):
        raise ValueError("b magnitudes must be positive")
    state = _labels(batch["state"], "state", N_STATES)
    transition = state_transition_table()
    matrix = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    for current in range(N_STATES):
        for event in range(3):
            matrix[current, transition[current, event]] += q[current, event]
    reward = -q[:, 1] * b[:, 0] + q[:, 2] * b[:, 1]
    value = np.zeros(N_STATES, dtype=np.float64)
    output = np.empty((state.shape[0], HORIZON), dtype=np.float64)
    for horizon in range(HORIZON):
        value = reward + matrix @ value
        output[:, horizon] = value[state]
    if not np.isfinite(output).all():
        raise FloatingPointError("non-finite Markov predictions")
    return output

