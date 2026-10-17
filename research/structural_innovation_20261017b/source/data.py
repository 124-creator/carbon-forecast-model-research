"""Strictly causal, annual refit data for the SMP structural experiment.

Only the five hashed development files supplied by a caller are accepted.
Features do not use a future quotation, date, scale, covariate, or label.
Targets are cumulative RAW price changes divided by the origin's past scale;
the runner restores native units before risk weighting and evaluation.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

L = H = 20
TOL = 1e-10
SCORE_H = np.array([0, 4, 19], dtype=int)


@dataclass
class Market:
    name: str
    dates: np.ndarray
    days: np.ndarray
    price: np.ndarray
    covariates: np.ndarray
    covariate_names: tuple
    source_sha256: str


def load_market(spec, end_exclusive="2024-01-01"):
    raw = Path(spec["path"]).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != spec["sha256"]:
        raise ValueError("Source identity changed: " + spec["name"])
    records = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    dates = np.array([r["date"][:10] for r in records])
    if len(records) == 0 or np.any(dates[1:] <= dates[:-1]):
        raise ValueError("Unique increasing observed dates are required")
    if np.any(dates >= end_exclusive):
        raise ValueError("This entry point accepts development-only files; refuse a suffix")
    prices = np.array([float(r["price"]) for r in records], dtype=np.float64)
    names = ("auction_volume", "cover_ratio", "bidder_count") if spec["with_auction"] else ()
    cov = np.array([[float(r[k]) for k in names] for r in records]).reshape(len(records), len(names))
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Invalid price: do not impute or clean quotes")
    if len(names) and (not np.isfinite(cov).all() or np.any(cov <= 0)):
        raise ValueError("Invalid historical auction covariates")
    days = np.array([date.fromisoformat(d).toordinal() for d in dates], dtype=np.int64)
    return Market(spec["name"], dates, days, prices, cov, names, actual)


def state_path(price):
    """Prefix scan; a zero means an observed zero NET change, not no trades."""
    n = len(price)
    sign = np.zeros(n, dtype=np.int64)
    age = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        dp = float(price[i] - price[i - 1])
        if abs(dp) <= TOL:
            sign[i] = sign[i - 1]
            age[i] = min(int(age[i - 1]) + 1, 2)
        else:
            sign[i] = -1 if dp < 0 else 1
            age[i] = 0
    return sign, age, (sign + 1) * 3 + age


class AnnualData:
    def __init__(self, market, year, validation_size=80):
        self.market, self.year = market, int(year)
        self.start = int(np.searchsorted(market.dates, f"{year}-01-01"))
        self.end = int(np.searchsorted(market.dates, f"{year+1}-01-01"))
        mature = np.arange(60, self.start - H, dtype=np.int64)
        if len(mature) < validation_size + H + 80:
            raise ValueError("Insufficient mature training and purged validation history")
        self.validation = mature[-validation_size:]
        self.train = mature[mature + H < self.validation[0]]
        self.test = np.arange(self.start, self.end - H, dtype=np.int64)
        if not len(self.test):
            raise ValueError("No same-year full-H20 evaluation origins")
        self.stats_end = int(self.validation[0])
        self.train_rms = max(float(np.sqrt(np.mean(np.diff(market.price[:self.stats_end]) ** 2))), 1e-8)
        self.floor = max(self.train_rms * .01, 1e-8)
        if len(market.covariate_names):
            values = np.log(market.covariates[:self.stats_end])
            self.cov_mean = values.mean(axis=0)
            self.cov_std = np.maximum(values.std(axis=0), 1e-6)
        else:
            self.cov_mean = self.cov_std = np.empty(0, dtype=np.float64)
        self.sign, self.age, self.states = state_path(market.price)
        self._features = {}

    def at(self, t):
        t = int(t)
        if t < 60 or t >= len(self.market.price):
            raise ValueError("Invalid feature origin")
        if t not in self._features:
            p, days = self.market.price, self.market.days
            s = max(float(np.sqrt(np.mean(np.diff(p[t - 60:t + 1]) ** 2))), self.floor)
            sl = slice(t - L + 1, t + 1)
            columns = [np.diff(p[t - L:t + 1]) / s,
                       np.log1p(np.diff(days[t - L:t + 1])) / np.log(8.),
                       np.full(L, np.log(s / self.train_rms)),
                       self.sign[sl].astype(float), self.age[sl].astype(float) / 2.]
            x = np.stack(columns, axis=1)
            if len(self.market.covariate_names):
                cov = (np.log(self.market.covariates[sl]) - self.cov_mean) / self.cov_std
                x = np.concatenate((x, cov), axis=1)
            self._features[t] = (np.clip(x, -10., 10.), s, int(self.states[t]))
        x, s, state = self._features[t]
        return x.copy(), s, state

    def batch(self, origins, available_through, labels=True):
        ids = np.asarray(origins, dtype=np.int64)
        if ids.ndim != 1 or len(ids) == 0 or ids.max() > available_through:
            raise ValueError("Unknown or empty input support")
        xs, scales, states = zip(*(self.at(t) for t in ids))
        out = dict(origin=ids.copy(), x=np.stack(xs), scale=np.array(scales),
                   state=np.array(states, dtype=np.int64), base=self.market.price[ids].copy())
        if labels:
            if int(ids.max()) + H > available_through:
                raise ValueError("Immature target requested")
            future = self.market.price[ids[:, None] + np.arange(1, H + 1)]
            out["delta"] = future - out["base"][:, None]
            out["y"] = out["delta"] / out["scale"][:, None]
            d1 = out["delta"][:, 0]
            out["event"] = np.where(abs(d1) <= TOL, 0, np.where(d1 < 0, 1, 2)).astype(np.int64)
        return out

    def support_receipt(self):
        return dict(market=self.market.name, year=self.year, source_sha256=self.market.source_sha256,
                    source_rows=len(self.market.price), train_n=len(self.train), validation_n=len(self.validation),
                    evaluation_n=len(self.test), train_first_origin=int(self.train[0]),
                    train_last_origin=int(self.train[-1]), max_train_label=int(self.train[-1] + H),
                    first_validation_origin=int(self.validation[0]),
                    last_validation_origin=int(self.validation[-1]),
                    max_validation_label=int(self.validation[-1] + H), evaluation_first_origin=int(self.test[0]),
                    evaluation_last_origin=int(self.test[-1]), evaluation_last_label=int(self.test[-1] + H),
                    evaluation_first_date=str(self.market.dates[self.test[0]]),
                    evaluation_last_date=str(self.market.dates[self.test[-1]]),
                    statistics_price_prefix_end_exclusive=self.stats_end, training_prefix_rms=self.train_rms,
                    scale_floor=self.floor, covariates=list(self.market.covariate_names),
                    cov_mean=self.cov_mean.tolist(), cov_std=self.cov_std.tolist(),
                    training_state_counts=np.bincount(self.states[self.train], minlength=9).tolist(),
                    current_state_from_full_observed_prefix=True, future_context_frozen=True,
                    quote_imputation=False, feature_clip=[-10, 10],
                    input_channels=["increment/origin_scale", "log1p(past_calendar_gap)/log8",
                                    "log(origin_scale/training_prefix_rms)", "last_nonzero_sign_at_each_past_token",
                                    "capped_zero_run_age_at_each_past_token/2"] + list(self.market.covariate_names),
                    evaluation="same-year complete H20; no training updates during evaluation year")


def risk_denominator(train):
    """Fixed horizon weights from training labels, never validation/test labels."""
    reference = np.mean(np.asarray(train["delta"]) ** 2, axis=0)
    floor = max(float(np.mean(np.asarray(train["scale"]) ** 2)) * 1e-8, 1e-16)
    return np.maximum(reference, floor)


def metrics(predicted_change, batch):
    raw = np.asarray(predicted_change, dtype=np.float64)
    truth = np.asarray(batch["delta"], dtype=np.float64)
    if raw.shape != truth.shape or not np.isfinite(raw).all():
        raise ValueError("Invalid forecast for evaluation")
    mse = np.mean((raw - truth) ** 2, axis=0)
    rw = np.mean(truth ** 2, axis=0)
    if np.any(rw <= 0):
        raise ValueError("Undefined relative risk for constant evaluation target")
    relative = mse / rw
    return dict(n=len(raw), score=float(relative[SCORE_H].mean()), mse=mse.tolist(),
                rw_mse=rw.tolist(), relative_mse=relative.tolist(),
                mae=np.mean(abs(raw - truth), axis=0).tolist(),
                score_horizons=[1, 5, 20], score_denominator="same evaluation origins' RW MSE per horizon")
