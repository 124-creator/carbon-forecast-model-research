"""Frozen project statistical reference, copied without algorithm changes.

Source repository: 124-creator/carbon-forecast-model-research
Source commit: 1c67e562e721dd82fff4e4143b97a798a1cf44fa
Source path: research/recurrent_response_20261014/
             fair_recurrent_tuning_20261016/fair_core.py
Source Git blob: ab4ded1b48845c2c1bc8d972dd655441a0c6b984
Source file SHA256:
    20359991e4b3b4357466314fbc463f46e0f33dcffaa15dccac064f1a68b2f0b9

`reference` below is an exact source copy of that function. It returns all
20-horizon cumulative CHANGES relative to each current price, plus the original
three-dimensional context. It is an existing project baseline, not a new method
or an author-library benchmark reproduction. Wrappers are new convenience code.

No torch, optimizer, learned market parameters, date imputation, or source-data
access is used by this module. `anchor_arrays` reads only each market's price
prefix ending at the largest requested origin and returns RAW price units.
"""
from __future__ import annotations

import numpy as np

SOURCE_COMMIT = "1c67e562e721dd82fff4e4143b97a798a1cf44fa"
SOURCE_GIT_BLOB = "ab4ded1b48845c2c1bc8d972dd655441a0c6b984"
SOURCE_FILE_SHA256 = "20359991e4b3b4357466314fbc463f46e0f33dcffaa15dccac064f1a68b2f0b9"
SOURCE_URL = (
    "https://github.com/124-creator/carbon-forecast-model-research/blob/"
    + SOURCE_COMMIT
    + "/research/recurrent_response_20261014/fair_recurrent_tuning_20261016/fair_core.py"
)


def reference(price):
    bank = [(0., 1.)] + [(r, q) for r in (0., .25, .5, .75, .9, .97)
                         for q in (.02, .1, .25, .5, .8)]
    rho, permanent = np.array(bank).T
    transitory = 1 - permanent
    n = len(price)
    state = np.zeros(len(bank))
    variance = transitory / (1 - rho ** 2)
    states = np.zeros((n, len(bank)))
    errors = np.zeros_like(states)
    innovations = np.ones_like(states)
    for t in range(1, n):
        error = price[t] - price[t - 1] - (rho - 1) * state
        innovation = (rho - 1) ** 2 * variance + 1
        covariance = rho * (rho - 1) * variance + transitory
        state = rho * state + covariance * error / innovation
        variance = np.maximum(rho ** 2 * variance + transitory - covariance ** 2 / innovation, 0)
        states[t], errors[t], innovations[t] = state, error, innovation
    base = np.zeros((n, 20))
    context = np.zeros((n, 3))
    selected = 0
    for t in range(n):
        if t >= 60 and (t - 60) % 20 == 0:
            start = max(1, t - 240)
            scale = np.maximum(np.mean(errors[start:t] ** 2 / innovations[start:t], axis=0), 1e-12)
            criterion = np.log(innovations[start:t]).sum(0) + (t - start) * np.log(scale)
            selected = int(np.argmin(criterion))
        r, q = bank[selected]
        context[t] = [states[t, selected], r * (1 - q), 1 - q]
        base[t] = (r ** np.arange(1, 21) - 1) * states[t, selected]
    return base, context


def reference_at(price, origin):
    """Scalar walk-forward access: returns (20 raw changes, 3-value context).

    The supplied container may contain later observations; only [:origin+1]
    is indexed. For many origins use `reference(prefix)` once or anchor_arrays.
    """
    if not isinstance(origin, (int, np.integer)) or origin < 0 or origin >= len(price):
        raise ValueError("origin must be a valid observed integer index")
    prefix = np.asarray(price[:int(origin) + 1], dtype=np.float64)
    if prefix.ndim != 1 or not np.isfinite(prefix).all():
        raise ValueError("nonfinite or non-vector observed price prefix")
    base, context = reference(prefix)
    return base[-1].copy(), context[-1].copy()


def anchor_arrays(data, markets):
    """Return [N,20] raw cumulative-change anchors in exactly data row order.

    `data` needs market_names, market integer IDs, and local integer origins.
    Each market object needs .name and .price. Scale/target fields are neither
    read nor used. Normalize the returned changes by scale*sqrt(h) externally
    only when the caller's neural target uses that normalization.
    """
    raw_origins = np.asarray(data["origin"])
    raw_mids = np.asarray(data["market"])
    if raw_origins.ndim != 1 or raw_mids.shape != raw_origins.shape:
        raise ValueError("market and origin must be matching one-dimensional arrays")
    if not np.issubdtype(raw_origins.dtype, np.integer) or not np.issubdtype(raw_mids.dtype, np.integer):
        raise ValueError("market IDs and origins must be integer arrays")
    origins, mids = raw_origins.astype(np.int64), raw_mids.astype(np.int64)
    names = list(data["market_names"])
    if len(set(names)) != len(names):
        raise ValueError("market_names contains duplicates")
    by_name = {market.name: market for market in markets}
    if len(by_name) != len(markets) or any(name not in by_name for name in names):
        raise ValueError("market objects are duplicated or missing")
    if len(mids) and (mids.min() < 0 or mids.max() >= len(names) or origins.min() < 0):
        raise ValueError("invalid market ID or negative origin")
    anchors = np.zeros((len(origins), 20), dtype=np.float64)
    for mid, name in enumerate(names):
        use = np.flatnonzero(mids == mid)
        if not len(use):
            continue
        market = by_name[name]
        last = int(origins[use].max())
        if last >= len(market.price):
            raise ValueError("requested origin is not an observed price")
        prefix = np.asarray(market.price[:last + 1], dtype=np.float64)
        if prefix.ndim != 1 or len(prefix) != last + 1 or not np.isfinite(prefix).all():
            raise ValueError("invalid observed price prefix")
        base, _ = reference(prefix)
        anchors[use] = base[origins[use]]
    return anchors
