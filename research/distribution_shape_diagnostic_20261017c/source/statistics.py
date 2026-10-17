"""Zero-mean GARCH(1,1) reference, fitted once on a caller-supplied price prefix.

This module never opens market files. It implements the standard conditional
variance recursion, not a new model and not an author benchmark reproduction.
Forecasts are cumulative-change standard deviations in the original price unit.
Using normal/t5 atoms for h>1 is a marginal shape approximation, not the exact
distribution of a recursively simulated GARCH path or of a sum of t variables.
"""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t

HORIZON = 20
STATIONARITY_CAP = 1.0 - 1e-6
BOUNDS = ((-23.0, 5.0), (-14.0, 14.0), (-14.0, 14.0))
OPTIONS = {"maxiter": 500, "maxfun": 2000, "ftol": 1e-12,
           "gtol": 1e-8, "maxls": 30}


def _price_hash(p):
    return hashlib.sha256(np.asarray(p, dtype="<f8").tobytes()).hexdigest()


def _decode(theta):
    """Positive omega and alpha+beta<CAP, with their exact Jacobian."""
    theta = np.asarray(theta, dtype=np.float64)
    logits = np.array([theta[1], theta[2], 0.0])
    weights = np.exp(logits - np.max(logits))
    weights /= weights.sum()
    omega = float(np.exp(theta[0]))
    alpha, beta = STATIONARITY_CAP * weights[:2]
    jac = np.zeros((3, 3), dtype=np.float64)
    jac[0, 0] = omega
    jac[1, 1] = alpha * (1 - alpha / STATIONARITY_CAP)
    jac[1, 2] = -alpha * beta / STATIONARITY_CAP
    jac[2, 1] = jac[1, 2]
    jac[2, 2] = beta * (1 - beta / STATIONARITY_CAP)
    return np.array([omega, alpha, beta]), jac


def _conditional_nll(theta, returns, kind):
    """Mean conditional NLL and analytic gradient; initial variance fixed at 1."""
    (omega, alpha, beta), jac = _decode(theta)
    variance = 1.0
    dv = np.zeros(3, dtype=np.float64)
    nll = 0.0
    gradient = np.zeros(3, dtype=np.float64)
    nu = 5.0
    t_constant = (.5 * math.log(math.pi * (nu - 2))
                  + math.lgamma(nu / 2) - math.lgamma((nu + 1) / 2))
    for innovation in returns:
        squared = innovation * innovation
        if kind == "normal":
            nll += .5 * (math.log(2 * math.pi) + math.log(variance)
                         + squared / variance)
            dnll_dv = .5 / variance * (1 - squared / variance)
        elif kind == "t5":
            # Standardized t5 has unit variance: scale=sqrt((nu-2)/nu).
            nll += (t_constant + .5 * math.log(variance)
                    + .5 * (nu + 1) * math.log1p(squared / ((nu - 2) * variance)))
            dnll_dv = (.5 / variance
                       * (1 - (nu + 1) * squared / ((nu - 2) * variance + squared)))
        else:
            raise ValueError("kind must be 'normal' or 't5'")
        gradient += dnll_dv * dv
        dv = jac[0] + squared * jac[1] + variance * jac[2] + beta * dv
        variance = omega + alpha * squared + beta * variance
    return float(nll / len(returns)), gradient / len(returns)


def fit_garch(price_prefix, kind="normal"):
    """One fixed-start L-BFGS-B fit, no retries or best-of-start selection.

    Parameters are fitted to raw price increments divided by the RMS estimated
    from this prefix alone. No intercept/conditional mean is estimated. The
    returned dictionary contains parameters and a nested receipt. A failed
    optimizer is returned with eligible=False, and forecast_garch refuses it.
    Caller must construct the purged training prefix; this function cannot infer
    the calendar split from an array of prices.
    """
    if kind not in ("normal", "t5"):
        raise ValueError("kind must be 'normal' or 't5'")
    p = np.asarray(price_prefix, dtype=np.float64)
    if p.ndim != 1 or len(p) < 21 or not np.isfinite(p).all():
        raise ValueError("Need at least 21 finite one-dimensional prefix prices")
    started = time.monotonic()
    returns = np.diff(p)
    largest = float(np.max(np.abs(returns)))
    scale = (largest * math.sqrt(float(np.mean((returns / largest) ** 2)))
             if largest else 0.0)
    receipt = {
        "family": "garch_" + kind, "likelihood": kind,
        "zero_conditional_mean": True, "degrees_of_freedom": 5 if kind == "t5" else None,
        "fitted_quantity": "raw_price_increment / training_prefix_increment_RMS",
        "fit_prices": len(p), "fit_returns": len(returns),
        "fit_prefix_sha256_float64_le": _price_hash(p),
        "fit_scale": scale, "initial_normalized_variance": 1.0,
        "stationarity_cap": STATIONARITY_CAP,
        "optimizer_method": "L-BFGS-B", "optimizer_options": dict(OPTIONS),
        "optimizer_bounds": [list(pair) for pair in BOUNDS],
        "planned_start_count": 1, "actual_start_count": 0, "retry_count": 0,
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "trace_kind": "every objective-and-gradient evaluation; not epoch selection",
        "trace": [],
        "forecast_limitation": "h>1 normal/t5 shape is an approximation; variance sum is analytic",
    }
    if not math.isfinite(scale) or scale <= 0:
        receipt.update(status="not_estimable_zero_or_nonfinite_scale", success=False,
                       seconds=time.monotonic() - started)
        return {"kind": kind, "family": "garch_" + kind, "eligible": False,
                "success": False, "receipt": receipt}

    normalized = returns / scale
    remaining = STATIONARITY_CAP - .05 - .90
    initial = np.array([math.log(.05), math.log(.05 / remaining),
                        math.log(.90 / remaining)], dtype=np.float64)
    receipt["fixed_initial_parameters"] = {"omega": .05, "alpha": .05, "beta": .90}
    receipt["fixed_initial_transformed"] = initial.tolist()

    def objective(theta):
        value, gradient = _conditional_nll(theta, normalized, kind)
        receipt["trace"].append({
            "evaluation": len(receipt["trace"]) + 1,
            "mean_normalized_nll": value,
            "gradient_inf_norm": float(np.max(abs(gradient))),
            "transformed_parameters": np.asarray(theta).tolist(),
        })
        return value, gradient

    receipt["actual_start_count"] = 1
    try:
        result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                          bounds=BOUNDS, options=OPTIONS)
        decoded, _ = _decode(result.x)
        omega, alpha, beta = map(float, decoded)
        finite = (np.isfinite(decoded).all() and math.isfinite(float(result.fun))
                  and np.isfinite(result.jac).all())
        eligible = bool(result.success and finite)
        receipt.update(status="converged" if eligible else "optimizer_failed",
                       success=bool(result.success), optimizer_status=int(result.status),
                       optimizer_message=str(result.message), iterations=int(result.nit),
                       function_evaluations=int(result.nfev),
                       gradient_evaluations=int(result.njev),
                       final_normalized_mean_nll=float(result.fun),
                       final_original_unit_mean_nll=float(result.fun + math.log(scale)),
                       final_gradient_inf_norm=float(np.max(abs(result.jac))),
                       final_transformed_parameters=result.x.tolist(),
                       seconds=time.monotonic() - started,
                       selected_final_iterate_only=True)
        return {
            "kind": kind, "family": "garch_" + kind, "eligible": eligible,
            "success": eligible, "omega": omega, "alpha": alpha, "beta": beta,
            "omega_original_unit": omega * scale * scale,
            "fit_scale": scale, "initial_variance": 1.0, "fit_prices": len(p),
            "fit_prefix_sha256_float64_le": receipt["fit_prefix_sha256_float64_le"],
            "receipt": receipt,
        }
    except Exception as exc:
        receipt.update(status="optimizer_exception", success=False,
                       error=type(exc).__name__ + ": " + str(exc),
                       seconds=time.monotonic() - started)
        return {"kind": kind, "family": "garch_" + kind, "eligible": False,
                "success": False, "receipt": receipt}


def forecast_garch(price, origins, params):
    """Return [B,20] cumulative-change sigma using prices through each origin.

    v_(t+1)=omega+alpha*r_t^2+beta*v_t is filtered with observed increments.
    E_t[v_(t+j+1)]=omega+(alpha+beta)*E_t[v_(t+j)]. Zero-mean innovations
    are conditionally uncorrelated, so cumulative-change variances add. Prices
    after max(origins) are not read for filtering or validated for finiteness.
    """
    if not params.get("eligible", False):
        raise ValueError("Refusing a failed/non-eligible GARCH fit; no fallback")
    q = np.asarray(origins)
    if q.ndim != 1 or not np.issubdtype(q.dtype, np.integer):
        raise ValueError("origins must be a one-dimensional integer array")
    if not len(q):
        return np.empty((0, HORIZON), dtype=np.float64)
    p = np.asarray(price)
    if p.ndim != 1 or np.any(q < 0) or int(q.max()) >= len(p):
        raise ValueError("Invalid price dimensions or origin bounds")
    n_fit = int(params["fit_prices"])
    if int(q.min()) < n_fit - 1:
        raise ValueError("A query predates the end of the fitted training prefix")
    past = np.asarray(p[:int(q.max()) + 1], dtype=np.float64)
    if not np.isfinite(past).all():
        raise ValueError("Nonfinite observed past")
    if _price_hash(past[:n_fit]) != params["fit_prefix_sha256_float64_le"]:
        raise ValueError("Forecast series does not share the exact fitted prefix")
    scale = float(params["fit_scale"])
    omega, alpha, beta = (float(params[name]) for name in ("omega", "alpha", "beta"))
    if not (scale > 0 and omega > 0 and alpha >= 0 and beta >= 0 and alpha + beta < 1):
        raise ValueError("Invalid GARCH parameters")
    v_next = np.empty(len(past), dtype=np.float64)
    v_next[0] = float(params["initial_variance"])
    for i in range(1, len(past)):
        innovation = (past[i] - past[i - 1]) / scale
        v_next[i] = omega + alpha * innovation * innovation + beta * v_next[i - 1]
    forward_variance = v_next[q].copy()
    cumulative = np.zeros(len(q), dtype=np.float64)
    sigma = np.empty((len(q), HORIZON), dtype=np.float64)
    for j in range(HORIZON):
        cumulative += forward_variance
        sigma[:, j] = scale * np.sqrt(cumulative)
        forward_variance = omega + (alpha + beta) * forward_variance
    if not np.isfinite(sigma).all():
        raise ValueError("Nonfinite GARCH variance forecast")
    return sigma


def standardized_atoms(kind="normal", n=31):
    """Deterministic midpoint-quantile finite measure, exactly normalized 0/1.

    Equal weights and discrete second-moment normalization describe the stated
    finite measure. Its CRPS must be computed as a discrete-distribution CRPS;
    it is not an iid sample and is not exact continuous normal/t CRPS.
    """
    if kind not in ("normal", "t5") or n != 31:
        raise ValueError("This protocol fixes normal/t5 with 31 atoms")
    tau = (np.arange(n, dtype=np.float64) + .5) / n
    raw = norm.ppf(tau) if kind == "normal" else student_t.ppf(tau, df=5)
    symmetric = (raw - raw[::-1]) / 2
    atoms = symmetric / math.sqrt(float(np.mean(symmetric * symmetric)))
    probabilities = np.full(n, 1.0 / n, dtype=np.float64)
    return atoms, probabilities


def forecast_garch_fhs(price, origins, params, draws=2048, seed=1729):
    """Filtered historical simulation with the already fitted normal GARCH.

    Returns native-price cumulative-change atoms [B,20,31], with equal weights.
    Training-prefix one-step filtered residuals are centered and standardized.
    For each origin, iid residual draws recursively evolve the GARCH variance.
    The 31 midpoint quantiles are processed by the same data.regular_quantiles
    used by the other empirical controls, in horizon-standardized units. Their
    finite-measure mean/variance are then exactly matched to 0/sigma_h**2.

    This is an explicit finite-atom, Monte-Carlo-quantile approximation to FHS.
    It uses no new likelihood fit, no new labels, no test-time parameter update,
    and no test-time residual-pool update. SeedSequence([seed, origin]) makes
    each forecast independent of the order or grouping of requested origins.
    """
    from data import regular_quantiles

    if params.get("kind") != "normal":
        raise ValueError("This protocol reuses only the normal GARCH fit for FHS")
    if not isinstance(draws, (int, np.integer)) or draws < 32:
        raise ValueError("draws must be an integer >=32; the formal protocol uses 2048")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    q = np.asarray(origins)
    sigma = forecast_garch(price, q, params)
    if not len(q):
        return np.empty((0, HORIZON, 31), dtype=np.float64)
    scale = float(params["fit_scale"])
    omega, alpha, beta = (float(params[name]) for name in ("omega", "alpha", "beta"))
    past = np.asarray(price)[:int(q.max())+1]
    v_next = np.empty(len(past), dtype=np.float64)
    v_next[0] = float(params["initial_variance"])
    residuals = []
    for i in range(1, len(past)):
        innovation = (past[i] - past[i-1]) / scale
        if i < int(params["fit_prices"]):
            residuals.append(innovation / math.sqrt(v_next[i-1]))
        v_next[i] = omega + alpha*innovation*innovation + beta*v_next[i-1]
    pool = np.asarray(residuals, dtype=np.float64)
    pool -= pool.mean()
    pool_variance = float(np.mean(pool*pool))
    if not math.isfinite(pool_variance) or pool_variance <= 0:
        raise ValueError("Degenerate fitted residual pool; no manufactured fallback")
    pool /= math.sqrt(pool_variance)

    answer = np.empty((len(q), HORIZON, 31), dtype=np.float64)
    for row, origin in enumerate(q):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(origin)]))
        innovations = pool[rng.integers(len(pool), size=(int(draws), HORIZON))]
        path_variance = np.full(int(draws), v_next[origin], dtype=np.float64)
        cumulative = np.zeros(int(draws), dtype=np.float64)
        normalized_paths = np.empty((int(draws), HORIZON), dtype=np.float64)
        for j in range(HORIZON):
            increment = np.sqrt(path_variance)*innovations[:, j]
            cumulative += increment
            normalized_paths[:, j] = cumulative / (sigma[row, j]/scale)
            path_variance = omega + alpha*increment*increment + beta*path_variance
        answer[row] = regular_quantiles(normalized_paths)*sigma[row, :, None]
    return answer
