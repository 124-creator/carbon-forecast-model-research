"""Causal multi-market price windows. No quote/calendar imputation.

All split decisions precede metric computation. Market IDs stratify sampling and
moment estimation; they are not neural inputs. A data set's quoted price meaning
is supplied by its provenance manifest, never inferred from the column name.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

HORIZON = 20
LENGTH = 20
SCORE_HORIZONS = np.array([0, 4, 19])
VIEW_VERTICES = {
    "full": np.arange(21),
    "short": np.arange(10, 21),
    "sparse": np.arange(0, 21, 2),
}


@dataclass
class Market:
    name: str
    dates: np.ndarray
    day: np.ndarray
    price: np.ndarray
    source_sha256: str
    source_path: str


def load_market(path, name, price_column="price", end_exclusive="2024-01-01"):
    """End-exclusive development loader; rejects duplicate/unordered dates.

    The file hash authenticates the source, not data licensing or vintage.
    The caller must not route a confirmation file to a development run.
    """
    path = Path(path)
    blob = path.read_bytes()
    rows = list(csv.DictReader(blob.decode("utf-8-sig").splitlines()))
    dates = np.array([r["date"][:10] for r in rows])
    if not len(dates) or np.any(dates[1:] <= dates[:-1]):
        raise ValueError(f"{name}: dates must be unique and strictly ascending")
    keep = dates < end_exclusive
    kept = [r for r, good in zip(rows, keep) if good]
    p = np.array([float(r[price_column]) for r in kept], dtype=np.float64)
    d = dates[keep]
    if not len(p) or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError(f"{name}: missing/nonpositive/nonfinite price")
    return Market(name, d, np.array([date.fromisoformat(s).toordinal() for s in d]),
                  p, hashlib.sha256(blob).hexdigest(), str(path.resolve()))


def split_support(market, cutoff, validation_size=80, train_cap=None):
    """Train labels finish before the first validation origin; all val labels
    finish strictly before the calendar cutoff shared by every market.
    """
    end = int(np.searchsorted(market.dates, cutoff, side="left"))
    mature = np.arange(LENGTH, end - HORIZON, dtype=np.int64)
    if len(mature) < validation_size + HORIZON + 120:
        raise ValueError(f"{market.name}/{cutoff}: insufficient mature history")
    val = mature[-validation_size:]
    train = mature[mature + HORIZON < val[0]]
    if train_cap is not None:
        train = train[-int(train_cap):]
    assert train[-1] + HORIZON < val[0]
    assert val[-1] + HORIZON < end
    # Each genuine one-step increment in training support is counted once.
    scale_start = int(train[0] - LENGTH)
    scale_stop = int(train[-1] + 1)
    increments = np.diff(market.price[scale_start:scale_stop])
    scale = max(float(np.sqrt(np.mean(increments ** 2))), 1e-8)
    return dict(end=end, train=train, validation=val, scale=scale,
                scale_start=scale_start, scale_stop=scale_stop,
                cutoff=cutoff, cap=train_cap)


def inputs(market, origins, scale, view="full"):
    origins = np.asarray(origins, dtype=np.int64)
    if not len(origins) or origins.min() < LENGTH or origins.max() >= len(market.price):
        raise ValueError("invalid past input support")
    vertices = VIEW_VERTICES[view]
    ids = origins[:, None] - LENGTH + vertices[None, :]
    observed_price = market.price[ids]
    observed_day = market.day[ids]
    # Recompute differences AFTER coarsening. Dropped prices cannot leak via
    # original increments; this is a genuinely nested information set.
    dp = np.diff(observed_price, axis=1) / scale
    obs_gap = np.broadcast_to(np.diff(vertices), dp.shape)
    day_gap = np.diff(observed_day, axis=1)
    tokens = np.stack([dp, np.log1p(obs_gap), np.log1p(day_gap)], axis=-1)
    n = tokens.shape[1]
    x = np.zeros((len(origins), LENGTH, 3), dtype=np.float64)
    mask = np.zeros((len(origins), LENGTH), dtype=bool)
    x[:, :n] = tokens
    mask[:, :n] = True
    return x, mask


def labels(market, origins):
    origins = np.asarray(origins, dtype=np.int64)
    if origins.max() + HORIZON >= len(market.price):
        raise ValueError("unmature target requested")
    return market.price[origins[:, None] + np.arange(1, HORIZON + 1)] - market.price[origins, None]


def pack(markets, cutoff, role, validation_size=80, train_cap=None):
    chunks = []
    supports = {}
    for m_id, market in enumerate(markets):
        support = split_support(market, cutoff, validation_size, train_cap)
        supports[market.name] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                 for k, v in support.items()}
        origins = support[role]
        scale = support["scale"]
        raw_y = labels(market, origins)
        d = dict(origin=origins, market=np.full(len(origins), m_id, dtype=np.int64),
                 scale=np.full(len(origins), scale), raw_y=raw_y,
                 y=raw_y / scale / np.sqrt(np.arange(1, 21)),
                 dates=market.dates[origins])
        for view in VIEW_VERTICES:
            d[view], d[view + "_mask"] = inputs(market, origins, scale, view)
        chunks.append(d)
    result = {key: np.concatenate([d[key] for d in chunks]) for key in chunks[0]}
    result["supports"] = supports
    result["market_names"] = [m.name for m in markets]
    return result


def evaluation_pack(markets, cutoff, end_exclusive, supports, same_year=True):
    chunks = []
    for m_id, market in enumerate(markets):
        origins = np.arange(LENGTH, len(market.price) - HORIZON)
        use = (market.dates[origins] >= cutoff) & (market.dates[origins + HORIZON] < end_exclusive)
        if same_year:
            years = np.array([d[:4] for d in market.dates])
            use &= years[origins] == years[origins + HORIZON]
        origins = origins[use]
        if not len(origins):
            raise ValueError(f"no evaluation origins: {market.name}, {cutoff}, {end_exclusive}")
        scale = supports[market.name]["scale"]
        raw_y = labels(market, origins)
        d = dict(origin=origins, market=np.full(len(origins), m_id, dtype=np.int64),
                 scale=np.full(len(origins), scale), raw_y=raw_y,
                 y=raw_y / scale / np.sqrt(np.arange(1, 21)), dates=market.dates[origins])
        for view in VIEW_VERTICES:
            d[view], d[view + "_mask"] = inputs(market, origins, scale, view)
        chunks.append(d)
    out = {key: np.concatenate([d[key] for d in chunks]) for key in chunks[0]}
    out["market_names"] = [m.name for m in markets]
    return out


def group_weights(data):
    """Equal market contribution regardless of numbers of overlapping windows."""
    mids = data["market"]
    groups = np.unique(mids)
    return np.array([len(mids) / (len(groups) * (mids == m).sum()) for m in mids])


def metrics(data, forecast, view="full"):
    """Equal mean across market/year/horizon cells; never ratio of pooled MSEs."""
    if forecast.shape != data["raw_y"].shape or not np.isfinite(forecast).all():
        raise ValueError("invalid forecast array")
    years = np.array([d[:4] for d in data["dates"]])
    records = []
    for mid, name in enumerate(data["market_names"]):
        for year in np.unique(years[data["market"] == mid]):
            use = (data["market"] == mid) & (years == year)
            for h in SCORE_HORIZONS:
                truth, pred = data["raw_y"][use, h], forecast[use, h]
                mse = float(np.mean((pred - truth) ** 2))
                rw = float(np.mean(truth ** 2))
                records.append(dict(market=name, year=int(year), horizon=int(h + 1),
                                    origins=int(use.sum()), mse=mse, rw_mse=rw,
                                    relative_mse=mse / rw, mae=float(np.mean(abs(pred - truth))),
                                    view=view))
    return dict(score=float(np.mean([r["relative_mse"] for r in records])), cells=records)


def boundary_checks():
    """Nontrivial information-containment and label-boundary checks."""
    d = np.array([date(2010, 1, 1).toordinal() + i for i in range(500)])
    m = Market("synthetic", np.array([date.fromordinal(int(t)).isoformat() for t in d]),
               d, 100 + np.sin(np.arange(500) / 4) + np.arange(500) / 30, "synthetic", "synthetic")
    origins = np.array([100])
    before = {v: inputs(m, origins, 1, v)[0] for v in VIEW_VERTICES}
    # Odd vertices are not in the sparse view. Altering one must be invisible.
    changed = m.price.copy(); changed[81] += 9
    other = Market(m.name, m.dates, m.day, changed, "", "")
    sparse_error = float(np.max(abs(before["sparse"] - inputs(other, origins, 1, "sparse")[0])))
    fine_difference = float(np.max(abs(before["full"] - inputs(other, origins, 1, "full")[0])))
    assert sparse_error == 0 and fine_difference > 0
    changed2 = m.price.copy(); changed2[101:] *= 7
    future = Market(m.name, m.dates, m.day, changed2, "", "")
    future_errors = {v: float(np.max(abs(before[v] - inputs(future, origins, 1, v)[0]))) for v in VIEW_VERTICES}
    assert all(e == 0 for e in future_errors.values())
    cutoff = m.dates[400]
    s = split_support(m, cutoff)
    changed3 = m.price.copy(); changed3[400:] += 1000
    alt = Market(m.name, m.dates, m.day, changed3, "", "")
    s2 = split_support(alt, cutoff)
    assert s["scale"] == s2["scale"] and np.array_equal(s["train"], s2["train"])
    assert s["train"][-1] + 20 < s["validation"][0]
    return dict(passed=True, coarse_deleted_price_error=sparse_error,
                full_deleted_price_difference=fine_difference,
                future_input_errors=future_errors, future_scale_error=abs(s["scale"] - s2["scale"]),
                train_n=len(s["train"]), train_max_label=int(s["train"][-1] + 20),
                first_validation_origin=int(s["validation"][0]), validation_max_label=int(s["validation"][-1] + 20))


if __name__ == "__main__":
    import json
    print(json.dumps(boundary_checks(), indent=2))
