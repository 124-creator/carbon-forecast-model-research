"""Independent mathematical GAS-Norm task adaptation; no author code is copied.

Primary sources (author revisions inspected on 2026-10-19):
  https://arxiv.org/html/2410.03935v1 , Sections 3.2--3.3, Eq. (4).
  https://github.com/edo-urettini/GAS_Norm/blob/
    cc63080958ebc9bfbb4bc95c2333c76a2070e607/normalizers/GAS_norm.py
  https://github.com/edo-urettini/GAS_Norm/blob/
    cc63080958ebc9bfbb4bc95c2333c76a2070e607/models/GAS_LSTM.py

This is not a reproduction of the whole published experiment.  In particular:
* Student-t is a working filter, not the final log-return predictive density.
* Filtering uses prior moments, then consumes the current observation.
* Future statistics follow the paper's pure AR prediction equations.  Passing
  a fabricated observation equal to the current mean to the score update would
  give a different scale recursion and is deliberately not done here.
* The fitted objective uses the author's two-strength likelihood weight and
  adjacent-state squared penalties, on train-only standardized inputs.  Each
  genuine next observation is used once; the author's duplicated final target
  is omitted.  This choice is named ``author_penalty_terminal_fixed_v1``.
* The variance line uses beta_scale, as in the author code; Eq. (4) in the
  preprint prints beta_mean on that line.

High-level interface:
  fit = fit_gas_normalizer(x_train)        # mature training prefix ONLY
  hist = filter_gas(x_observed, fit)       # rows remain causally ordered
  future_mean, future_std = forecast_gas(fit, hist, horizon=h)
  # row t predicts x[t+h] using x[:t+1], never x[t+1:].

For the carbon task x_t^(h) = log(P_t/P_(t-h)).  Forecasting this observed
target process h steps ahead gives the forward h-step return at origin t.
It avoids replacing an h-step return variance by h times a daily variance.
The caller owns calendar validation, train/validation purging, and the common
CRPS scoring unit.  No file access, market fitting, or network happens at import.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln


OBJECTIVE_VERSION = "author_penalty_terminal_fixed_v1"
FIT_VERSION = "gas_norm_independent_task_adaptation_v1"
PARAMETER_ORDER = (
    "alpha_mean", "alpha_scale", "beta_mean", "beta_scale",
    "omega_mean", "omega_scale", "initial_mean", "initial_scale2",
)


@dataclass(frozen=True)
class GASParameters:
    """States and parameters are in the fixed standardized training units.

    ``scale2`` is Student-t scale squared.  Its actual variance is
    ``scale2 * nu / (nu - 2)``.  The distinction is preserved everywhere.
    """

    alpha_mean: float
    alpha_scale: float
    beta_mean: float
    beta_scale: float
    omega_mean: float
    omega_scale: float
    initial_mean: float
    initial_scale2: float
    nu: float = 20.0
    gamma_mean: float = 0.1
    gamma_scale: float = 0.1

    def validate(self) -> None:
        if not all(math.isfinite(float(x)) for x in asdict(self).values()):
            raise ValueError("GAS parameters must be finite")
        if self.nu <= 2:
            raise ValueError("Finite standardization variance requires nu > 2")
        for name in ("alpha_mean", "alpha_scale", "beta_mean", "beta_scale"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("gamma_mean", "gamma_scale"):
            if not 0 < getattr(self, name) <= 0.5:
                raise ValueError(f"{name} must lie in (0, 0.5]")
        if self.omega_scale <= 0 or self.initial_scale2 <= 0:
            raise ValueError("The scale intercept and initial scale2 must be positive")

    @property
    def variance_multiplier(self) -> float:
        return self.nu / (self.nu - 2)

    def vector(self) -> np.ndarray:
        return np.asarray([getattr(self, key) for key in PARAMETER_ORDER], dtype=float)

    @classmethod
    def from_vector(cls, vector, *, nu=20.0, gamma_mean=0.1, gamma_scale=0.1):
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (8,):
            raise ValueError("Expected the eight static parameters in PARAMETER_ORDER")
        result = cls(**dict(zip(PARAMETER_ORDER, vector.tolist())), nu=float(nu),
                     gamma_mean=float(gamma_mean), gamma_scale=float(gamma_scale))
        result.validate()
        return result


@dataclass(frozen=True)
class GASFit:
    params: GASParameters
    input_mean: float
    input_std: float
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"params": asdict(self.params), "input_mean": self.input_mean,
                "input_std": self.input_std, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, value: dict[str, Any]):
        result = cls(GASParameters(**value["params"]), float(value["input_mean"]),
                     float(value["input_std"]), dict(value["metadata"]))
        result.params.validate()
        if not math.isfinite(result.input_mean) or not math.isfinite(result.input_std) or result.input_std <= 0:
            raise ValueError("Invalid fixed training transform")
        return result


@dataclass(frozen=True)
class GASHistory:
    """All arrays align to the input observations; no row is shifted implicitly.

    ``prior_mean`` and ``prior_std`` are in the caller's original input units.
    The two ``next_*_internal`` arrays are explicitly in standardized units and
    represent the next-observation prior after consuming each row.
    """

    normalized: np.ndarray
    prior_mean: np.ndarray
    prior_std: np.ndarray
    next_mean_internal: np.ndarray
    next_scale2_internal: np.ndarray


def _finite_vector(values, *, minimum=1) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or len(result) < minimum:
        raise ValueError(f"Expected a one-dimensional series of length >= {minimum}")
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite observations must not be silently imputed")
    return result


def _update_unchecked(value: float, mean: float, scale2: float,
                      p: GASParameters) -> tuple[float, float]:
    residual = value - mean
    # hypot avoids squaring a very large residual in the Student score.
    reference = math.sqrt(p.nu * scale2)
    length = math.hypot(reference, residual)
    mean_score = residual * (reference / length) ** 2
    scale_score = scale2 * ((p.nu + 1) * (residual / length) ** 2 - 1)
    updated_mean = mean + p.gamma_mean / (1 - p.gamma_mean) * p.alpha_mean * mean_score
    updated_scale2 = scale2 + p.gamma_scale / (1 - p.gamma_scale) * p.alpha_scale * scale_score
    next_mean = p.omega_mean + p.beta_mean * updated_mean
    next_scale2 = p.omega_scale + p.beta_scale * updated_scale2
    if not math.isfinite(next_mean) or not math.isfinite(next_scale2) or next_scale2 <= 0:
        raise FloatingPointError("Invalid GAS state; no state clipping is applied")
    return next_mean, next_scale2


def gas_update(value: float, mean: float, scale2: float,
               params: GASParameters) -> tuple[float, float]:
    """Consume one observed standardized value and return the next prior."""
    params.validate()
    if not all(math.isfinite(float(x)) for x in (value, mean, scale2)) or scale2 <= 0:
        raise ValueError("Input state and observation must be finite; scale2 > 0")
    return _update_unchecked(float(value), float(mean), float(scale2), params)


def penalized_objective(vector, standardized_train, *, nu=20.0,
                        gamma_mean=0.1, gamma_scale=0.1) -> float:
    """Train-prefix objective, T-1 genuine next targets, in fixed input units.

    For every t=0,...,T-2, update with x[t], then score x[t+1] under the
    resulting Student prior.  Penalize the squared change of each state.
    This matches the inspected author objective except that no x[-1] copy is
    appended and the denominator is T-1.  It is not Eq. (2)'s general FIM
    penalty, and metadata always makes that boundary explicit.
    """
    x = _finite_vector(standardized_train, minimum=2)
    try:
        p = GASParameters.from_vector(vector, nu=nu, gamma_mean=gamma_mean,
                                      gamma_scale=gamma_scale)
        mean, scale2 = p.initial_mean, p.initial_scale2
        constant = float(gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(math.pi * nu))
        total = 0.0
        weight = gamma_mean + gamma_scale
        for t in range(len(x) - 1):
            next_mean, next_scale2 = _update_unchecked(float(x[t]), mean, scale2, p)
            residual = (float(x[t + 1]) - next_mean) / math.sqrt(nu * next_scale2)
            # log1p(residual**2), evaluated without overflow from the square.
            log_shape = 2.0 * math.log(math.hypot(1.0, residual))
            log_density = constant - 0.5 * math.log(next_scale2) - 0.5 * (nu + 1) * log_shape
            penalty = 0.5 * (1 - gamma_mean) * (next_mean - mean) ** 2
            penalty += 0.5 * (1 - gamma_scale) * (next_scale2 - scale2) ** 2
            total += -weight * log_density + penalty
            mean, scale2 = next_mean, next_scale2
        result = total / (len(x) - 1)
        return float(result) if math.isfinite(result) else 1e100
    except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
        return 1e100


def fit_gas_normalizer(values, *, nu=20.0, gamma=0.1,
                       maxiter=400, maxfev=4000) -> GASFit:
    """Fit once to a caller-supplied mature training prefix, without validation.

    nu=20 and gamma=.1 are fixed task settings, not selected here.  A train-only
    affine transform fixes the units of the author's state penalty.  There is
    one deterministic Powell start, no search over starts/hyperparameters.
    Optimizer nonconvergence is retained in metadata; an invalid optimization
    result raises rather than silently turning the arm into Global Norm.
    """
    x = _finite_vector(values, minimum=8)
    input_mean = float(x.mean())
    input_std = float(x.std(ddof=0))
    if input_std <= 1e-12:
        raise ValueError("Training target is constant at the declared numerical tolerance")
    standardized = (x - input_mean) / input_std
    initial = np.asarray([0.001, 0.001, 0.5, 0.5, 0.0, 1.0, 0.0, 1.0])
    GASParameters.from_vector(initial, nu=nu, gamma_mean=gamma, gamma_scale=gamma)
    bounds = [(0.0, 1.0)] * 4 + [(None, None), (1e-6, None), (None, None), (1e-6, None)]

    def objective(v):
        return penalized_objective(v, standardized, nu=nu, gamma_mean=gamma, gamma_scale=gamma)

    initial_objective = objective(initial)
    optimized = minimize(objective, initial, method="Powell", bounds=bounds,
                         options={"maxiter": int(maxiter), "maxfev": int(maxfev),
                                  "xtol": 1e-6, "ftol": 1e-7})
    params = GASParameters.from_vector(optimized.x, nu=nu, gamma_mean=gamma, gamma_scale=gamma)
    fitted_objective = objective(params.vector())
    if not math.isfinite(fitted_objective) or fitted_objective >= 1e90:
        raise FloatingPointError("Static GAS fit did not produce a valid finite state path")
    metadata = {
        "fit_version": FIT_VERSION, "objective": OBJECTIVE_VERSION,
        "train_observations": len(x), "genuine_next_targets": len(x) - 1,
        "train_values_sha256": hashlib.sha256(np.ascontiguousarray(x, dtype="<f8").tobytes()).hexdigest(),
        "input_transform": "train_mean_and_population_std",
        "parameter_order": list(PARAMETER_ORDER), "nu": float(nu),
        "gamma_mean": float(gamma), "gamma_scale": float(gamma),
        "optimizer": "Powell", "starts": 1, "maxiter": int(maxiter), "maxfev": int(maxfev),
        "optimizer_success": bool(optimized.success), "optimizer_status": int(optimized.status),
        "optimizer_message": str(optimized.message), "nfev": int(optimized.nfev),
        "nit": int(optimized.nit), "initial_objective": float(initial_objective),
        "fitted_objective": float(fitted_objective),
        "forecast_rule": "paper_pure_AR_no_fabricated_future_observations",
        "student_state": "scale_squared_not_variance",
        "normalization_rule": "prior_before_observation",
        "author_revision": "cc63080958ebc9bfbb4bc95c2333c76a2070e607",
        "no_hyperparameter_search": True,
    }
    return GASFit(params, input_mean, input_std, metadata)


def filter_gas(values, fit: GASFit) -> GASHistory:
    """Causal filter: prefix output is invariant to any suffix replacement."""
    x = _finite_vector(values)
    fit.params.validate()
    if not math.isfinite(fit.input_mean) or not math.isfinite(fit.input_std) or fit.input_std <= 0:
        raise ValueError("Invalid training transform")
    standardized = (x - fit.input_mean) / fit.input_std
    p = fit.params
    prior_mean, prior_std, z, next_mean, next_scale2 = [np.empty(len(x), dtype=float) for _ in range(5)]
    mean, scale2 = p.initial_mean, p.initial_scale2
    for t, value in enumerate(standardized):
        std = math.sqrt(p.variance_multiplier * scale2)
        prior_mean[t] = fit.input_mean + fit.input_std * mean
        prior_std[t] = fit.input_std * std
        z[t] = (float(value) - mean) / std
        mean, scale2 = _update_unchecked(float(value), mean, scale2, p)
        next_mean[t], next_scale2[t] = mean, scale2
    return GASHistory(z, prior_mean, prior_std, next_mean, next_scale2)


def _ar_advance(value: np.ndarray, omega: float, beta: float, steps: int) -> np.ndarray:
    if steps == 0:
        return np.array(value, dtype=float, copy=True)
    if beta == 1:
        return np.asarray(value, dtype=float) + steps * omega
    power = beta ** steps
    # A small finite geometric sum avoids cancellation near beta=1.
    total = math.fsum(beta ** j for j in range(steps))
    return power * np.asarray(value, dtype=float) + omega * total


def forecast_gas(fit: GASFit, history: GASHistory, *, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and actual std of x[t+h], for every observed origin row t.

    history already holds the prior for t+1.  Hence only h-1 additional AR
    steps are taken.  The score term is not evaluated on invented future data.
    No assertion is made that the working Student model is the true process.
    """
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    p = fit.params
    mean = _ar_advance(history.next_mean_internal, p.omega_mean, p.beta_mean, int(horizon) - 1)
    scale2 = _ar_advance(history.next_scale2_internal, p.omega_scale, p.beta_scale, int(horizon) - 1)
    if not np.isfinite(mean).all() or not np.isfinite(scale2).all() or np.any(scale2 <= 0):
        raise FloatingPointError("Invalid un-clipped future GAS statistics")
    return (fit.input_mean + fit.input_std * mean,
            fit.input_std * np.sqrt(p.variance_multiplier * scale2))

