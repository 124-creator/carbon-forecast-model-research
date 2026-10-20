"""Independent mathematical implementations of CRPS aggregation baselines.

Sources (no profoc source code is vendored or translated into this module):
  Berrisch & Ziel, CRPS Learning, arXiv:2102.00968v3, sections 4--5.
  Wintenberger, Optimal learning with Bernstein Online Aggregation,
  arXiv:1404.1356v5, Figures 1--2.
  Author implementation inspected at profoc commit
  b5073ca70aea60ce209676faa99e04da725ffe04, src/conline.cpp and src/loss.cpp.

AdaptiveBOA implements the author's equal-prior, unsmoothed, gamma=1 BOA
recursion, using positive regret = aggregate linearized loss - expert loss.
The positive exponent below agrees with that author code. The negative
exponent printed in (33f) of the accepted CRPS Learning manuscript does not
agree with its positive-regret definition (33a); it is NOT used here.

BOACDFMixture is our CRPS-gradient specialization to a *vertical CDF pool*.
It is not a reproduction of CRPS Learning's pointwise quantile pool.
QuantileBOAG implements the unsmoothed pointwise special case, not profoc's
complete spline/hyperparameter-selection procedure. Only NumPy is required.

Time: low-level update/replay methods assume the caller supplies labels
already mature and in their historical order. A historical replay can be
performed backwards *from a current decision into its history for reading*,
but updates MUST then run oldest-to-newest, never in reverse chronology.
DelayedCDFBOA additionally enforces issue/observe and maturity bookkeeping.
For deployment, pass the alpha or quantiles actually ISSUED for the target;
never recompute that old prediction with today's weights.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


def _finite_array(value, *, ndim=None, name="array"):
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _alpha(value):
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    return value


def _ab(a, b):
    a, b = float(a), float(b)
    if not math.isfinite(a) or not math.isfinite(b) or a < 0.0:
        raise ValueError("A must be finite/nonnegative and B finite")
    if a == 0.0 and b != 0.0:
        raise ValueError("identical CDFs (A=0) require B=0")
    return a, b


class AdaptiveBOA:
    """Equal-prior fully adaptive BOA, optionally parallel across tasks.

    ``regret[..., k]`` is positive when expert k improves the linearized
    loss. ``forget`` is the fraction of old R, V and E forgotten per
    observed label (zero by default). Forgetting forfeits the undiscounted
    procedure's theoretical guarantees; it is only a named sensitivity.
    """

    def __init__(self, n_experts=2, *, n_tasks=None, forget=0.0):
        if int(n_experts) != n_experts or n_experts < 2:
            raise ValueError("n_experts must be an integer >= 2")
        if n_tasks is not None and (int(n_tasks) != n_tasks or n_tasks < 1):
            raise ValueError("n_tasks must be a positive integer")
        if not math.isfinite(forget) or not 0.0 <= forget <= 1.0:
            raise ValueError("forget must be in [0, 1]")
        self.n_experts = int(n_experts)
        self.forget = float(forget)
        shape = (self.n_experts,) if n_tasks is None else (int(n_tasks), self.n_experts)
        self._weights = np.full(shape, 1.0 / self.n_experts)
        self.R = np.zeros(shape)
        self.V = np.zeros(shape)
        self.E = np.zeros(shape)
        self.eta = np.zeros(shape)
        self.n_updates = 0

    @property
    def weights(self):
        return self._weights.copy()

    def update_regret(self, regret):
        r = _finite_array(regret, name="regret")
        if r.shape != self.R.shape:
            raise ValueError(f"regret shape must be {self.R.shape}")
        retained = 1.0 - self.forget
        squared = np.square(r)
        if not np.all(np.isfinite(squared)):
            raise ValueError("regret too large: squared regret overflow")
        variance = retained * self.V + squared
        envelope = np.maximum(retained * self.E, np.abs(r))
        # The exponent floor follows the numerical scale of the inspected
        # author implementation. It does not clip any observations or loss.
        floor = math.exp(-350.0)
        envelope = np.maximum(envelope, floor)
        eta = np.minimum(
            0.5 / envelope,
            np.sqrt(math.log(self.n_experts) / np.maximum(variance, floor)),
        )
        eta = np.minimum(eta, math.exp(350.0))
        # With CURRENT E and eta, |eta*r| <= 1/2 in exact arithmetic.
        # Thus the author's exceptional-range indicator is identically
        # zero. Clipping the product prevents a one-ULP phantom exception.
        eta_r = np.clip(eta * r, -0.5, 0.5)
        score = retained * self.R + 0.5 * r * (1.0 - eta_r)
        log_weight = np.log(eta) + eta * score
        log_weight -= np.max(log_weight, axis=-1, keepdims=True)
        weight = np.exp(log_weight)
        weight /= np.sum(weight, axis=-1, keepdims=True)
        if not np.all(np.isfinite(weight)):
            raise FloatingPointError("BOA update produced nonfinite weights")
        self.V, self.E, self.eta, self.R = variance, envelope, eta, score
        self._weights = weight
        self.n_updates += 1
        return self.weights


class BOACDFMixture:
    """BOA with exact gradients for F_alpha=(1-alpha)F0+alpha*F1.

    A = integral (F1-F0)^2 dz.
    B = integral (1{y<=z}-F0(z))*(F1(z)-F0(z)) dz.
    The CRPS difference from F0 is A*alpha^2 - 2*B*alpha.
    ``alpha_issued`` must be the old forecast weight when feedback is delayed.
    Omit it only during an immediate historical replay of mature records.
    """

    def __init__(self, *, forget=0.0):
        self.core = AdaptiveBOA(2, forget=forget)

    @property
    def weights(self):
        return self.core.weights

    @property
    def alpha(self):
        return float(self.core.weights[1])

    def update_ab(self, a, b, alpha_issued=None):
        a, b = _ab(a, b)
        alpha = self.alpha if alpha_issued is None else _alpha(alpha_issued)
        derivative = 2.0 * (a * alpha - b)
        regret = derivative * np.array([alpha, -(1.0 - alpha)])
        self.core.update_regret(regret)
        return self.alpha

    def replay_ab(self, a, b, *, alpha_issued=None):
        """Replay mature AB rows oldest-to-newest; return pre-update alphas.

        If ``alpha_issued`` is supplied, use those archived prediction
        weights for gradients. Otherwise issue each virtual calibration
        prediction at the current weights, then consume its mature label.
        This latter operation is a batch fitting procedure on past data;
        it does not reproduce the historical forecasts of a delayed stream.
        """
        a = _finite_array(a, ndim=1, name="A")
        b = _finite_array(b, ndim=1, name="B")
        if a.shape != b.shape:
            raise ValueError("A and B must have equal shape")
        archived = None if alpha_issued is None else _finite_array(
            alpha_issued, ndim=1, name="alpha_issued"
        )
        if archived is not None and archived.shape != a.shape:
            raise ValueError("alpha_issued shape must match A")
        used = np.empty(a.size)
        for i in range(a.size):
            used[i] = self.alpha if archived is None else _alpha(archived[i])
            self.update_ab(a[i], b[i], used[i])
        return used


class EmpiricalCRPSMixture:
    """Ordinary historical CRPS minimization, not Bernstein aggregation.

    The exact unregularized solution is clip(sum(B)/sum(A), 0, 1).
    ``window=None`` retains all mature rows; an integer uses the last
    ``window`` mature rows. A no-information tie retains ``initial_alpha``.
    The optional ``alpha_issued`` argument is validated for API compatibility
    but is not part of this empirical optimization objective.
    """

    def __init__(self, *, window=None, initial_alpha=0.5):
        if window is not None and (int(window) != window or window < 1):
            raise ValueError("window must be None or a positive integer")
        self.window = None if window is None else int(window)
        self.initial_alpha = _alpha(initial_alpha)
        self._rows = deque(maxlen=self.window)

    @property
    def alpha(self):
        # fsum avoids accumulation/subtraction drift in the rolling window.
        total_a = math.fsum(row[0] for row in self._rows)
        if total_a == 0.0:
            return self.initial_alpha
        total_b = math.fsum(row[1] for row in self._rows)
        return float(np.clip(total_b / total_a, 0.0, 1.0))

    @property
    def weights(self):
        alpha = self.alpha
        return np.array([1.0 - alpha, alpha])

    def update_ab(self, a, b, alpha_issued=None):
        if alpha_issued is not None:
            _alpha(alpha_issued)
        self._rows.append(_ab(a, b))
        return self.alpha

    def replay_ab(self, a, b, *, alpha_issued=None):
        a = _finite_array(a, ndim=1, name="A")
        b = _finite_array(b, ndim=1, name="B")
        if a.shape != b.shape:
            raise ValueError("A and B must have equal shape")
        archived = None if alpha_issued is None else _finite_array(
            alpha_issued, ndim=1, name="alpha_issued"
        )
        if archived is not None and archived.shape != a.shape:
            raise ValueError("alpha_issued shape must match A")
        used = np.empty(a.size)
        for i in range(a.size):
            used[i] = self.alpha if archived is None else _alpha(archived[i])
            self.update_ab(a[i], b[i], used[i])
        return used


@dataclass(frozen=True)
class CDFTicket:
    sequence: int
    origin: int
    target_index: int
    alpha: float


class DelayedCDFBOA(BOACDFMixture):
    """Optional clock/maturity guard around the scalar BOA baseline.

    One instance represents one source/fold/horizon. At calendar index t,
    consume all newly mature tickets in (target_index, origin) order, then
    issue the next forecast. Pending AB quantities must refer to the expert
    CDFs frozen at the ticket's origin. This class never reads market data.
    """

    def __init__(self, *, forget=0.0):
        super().__init__(forget=forget)
        self._clock = -1
        self._last_origin = -1
        self._sequence = 0
        self._last_observed = (-1, -1)
        self._pending = {}

    def issue(self, origin, target_index):
        if int(origin) != origin or int(target_index) != target_index:
            raise ValueError("origin/target_index must be integers")
        origin, target_index = int(origin), int(target_index)
        if origin < self._clock or origin <= self._last_origin or target_index <= origin:
            raise ValueError("issue requires a new origin at/after the clock and a future target")
        ticket = CDFTicket(self._sequence, origin, target_index, self.alpha)
        self._pending[ticket.sequence] = ticket
        self._sequence += 1
        self._clock = self._last_origin = origin
        return ticket

    def observe(self, ticket, a, b, *, available_index):
        if not isinstance(ticket, CDFTicket) or self._pending.get(ticket.sequence) != ticket:
            raise ValueError("unknown, altered, or already consumed ticket")
        if int(available_index) != available_index:
            raise ValueError("available_index must be an integer")
        available_index = int(available_index)
        if available_index < self._clock or ticket.target_index > available_index:
            raise ValueError("label is immature or observation clock moved backwards")
        key = (ticket.target_index, ticket.origin)
        if key <= self._last_observed:
            raise ValueError("mature observations must be consumed chronologically")
        if any((other.target_index, other.origin) < key for other in self._pending.values()):
            raise ValueError("an earlier mature ticket must be consumed first")
        alpha = self.update_ab(a, b, ticket.alpha)
        del self._pending[ticket.sequence]
        self._clock, self._last_observed = available_index, key
        return alpha


class QuantileBOAG:
    """Unsmoothed pointwise CRPS Learning special case on a fixed tau grid.

    Experts supply their quantiles externally (analytic normal quantiles or
    deterministic Gaussian-mixture CDF inversion). Arrays have shape
    (n_quantiles, n_experts). Independent BOA states learn each quantile;
    combined quantiles are sorted as in the paper and author implementation.
    Quantile levels do NOT define the tails of a complete distribution:
    any later CRPS integration/tail extension must be stated separately.
    """

    def __init__(self, tau: Iterable[float], *, n_experts=2, forget=0.0):
        tau = _finite_array(list(tau), ndim=1, name="tau")
        if tau.size == 0 or np.any(tau <= 0.0) or np.any(tau >= 1.0) or np.any(np.diff(tau) <= 0.0):
            raise ValueError("tau must be strictly increasing and inside (0, 1)")
        self.tau = tau.copy()
        self.core = AdaptiveBOA(n_experts, n_tasks=tau.size, forget=forget)

    @property
    def weights(self):
        return self.core.weights

    def _experts(self, quantiles):
        quantiles = _finite_array(quantiles, ndim=2, name="expert quantiles")
        if quantiles.shape != self.core.R.shape:
            raise ValueError(f"expert quantiles must have shape {self.core.R.shape}")
        if np.any(np.diff(quantiles, axis=0) < 0.0):
            raise ValueError("each expert's quantiles must be nondecreasing")
        return quantiles

    def predict(self, quantiles):
        quantiles = self._experts(quantiles)
        combined = np.sum(self.weights * quantiles, axis=1)
        return np.sort(combined)

    def update_quantiles(self, quantiles, y, prediction_issued=None):
        quantiles = self._experts(quantiles)
        y = float(y)
        if not math.isfinite(y):
            raise ValueError("y must be finite")
        prediction = self.predict(quantiles) if prediction_issued is None else _finite_array(
            prediction_issued, ndim=1, name="prediction_issued"
        )
        if prediction.shape != self.tau.shape or np.any(np.diff(prediction) < 0.0):
            raise ValueError("issued quantiles must match tau and be nondecreasing")
        # At ties, use the upper subgradient, as in profoc src/loss.cpp.
        subgradient = (prediction >= y).astype(float) - self.tau
        regret = subgradient[:, None] * (prediction[:, None] - quantiles)
        self.core.update_regret(regret)
        return self.weights

    def replay_quantiles(self, quantiles, y, *, predictions_issued=None):
        quantiles = _finite_array(quantiles, ndim=3, name="expert quantiles")
        y = _finite_array(y, ndim=1, name="y")
        if quantiles.shape[0] != y.size or quantiles.shape[1:] != self.core.R.shape:
            raise ValueError("quantile replay dimensions do not match")
        archived = None if predictions_issued is None else _finite_array(
            predictions_issued, ndim=2, name="predictions_issued"
        )
        if archived is not None and archived.shape != (y.size, self.tau.size):
            raise ValueError("predictions_issued dimensions do not match")
        used = np.empty((y.size, self.tau.size))
        for i in range(y.size):
            used[i] = self.predict(quantiles[i]) if archived is None else archived[i]
            self.update_quantiles(quantiles[i], y[i], used[i])
        return used
