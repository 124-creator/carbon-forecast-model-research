"""Development-only, causal data for the shape-necessity diagnostic."""
from __future__ import annotations
import csv
import hashlib
from datetime import date
from pathlib import Path
import numpy as np

H, L, K = 20, 20, 31
HS = np.array([0, 4, 19])
TAU = (np.arange(K) + .5) / K


def normalize_atoms(x):
    """Equal-weight population moments; no variance clipping disguised as exactness."""
    x = np.asarray(x, dtype=np.float64)
    z = x - x.mean(axis=-1, keepdims=True)
    v = np.mean(z*z, axis=-1, keepdims=True)
    if not np.isfinite(v).all() or np.any(v <= 0):
        raise ValueError('Degenerate atom distribution')
    return z / np.sqrt(v)


def regular_quantiles(values):
    q = np.quantile(values, TAU, axis=0).T
    if not np.isfinite(q).all() or np.any(np.var(q, axis=-1) <= 0):
        raise ValueError('Degenerate empirical quantiles; do not manufacture a shape')
    # A fixed gap prevents repeated-quote atoms from giving a singular neural
    # initialization. All empirical controls use exactly the same operation.
    gaps = np.maximum(np.diff(q, axis=-1), 1e-4)
    return normalize_atoms(np.concatenate([np.zeros_like(q[..., :1]), np.cumsum(gaps, axis=-1)], axis=-1))


class Data:
    def __init__(self, spec, year, validation_size=80):
        raw = Path(spec['path']).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if self.sha256 != spec['sha256']:
            raise ValueError('Input hash mismatch')
        rows = list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
        dates = np.array([r['date'][:10] for r in rows])
        prices = np.array([float(r['price']) for r in rows], dtype=np.float64)
        if not len(rows) or np.any(dates >= '2024-01-01'):
            raise ValueError('Only the five explicitly approved development files are accepted')
        if np.any(dates[1:] <= dates[:-1]) or not np.isfinite(prices).all() or np.any(prices <= 0):
            raise ValueError('Invalid quotes: no cleaning or imputation permitted')
        self.initialize(spec['name'], dates, prices, year, validation_size)

    def initialize(self, name, dates, prices, year, validation_size=80):
        self.name, self.dates, self.price, self.year = name, np.asarray(dates), np.asarray(prices), int(year)
        self.day = np.array([date.fromisoformat(d).toordinal() for d in self.dates])
        self.inc = np.r_[0., np.diff(self.price)]
        self.start = int(np.searchsorted(self.dates, f'{year}-01-01'))
        self.end = int(np.searchsorted(self.dates, f'{year+1}-01-01'))
        mature = np.arange(60, self.start-H)
        if len(mature) < validation_size+H+80:
            raise ValueError('Insufficient purged history')
        self.val = mature[-validation_size:]
        self.train = mature[mature+H < self.val[0]]
        self.test = np.arange(self.start, self.end-H)
        if not len(self.test):
            raise ValueError('No full-H same-year evaluation origins')
        self.stats_end = int(self.val[0])
        self.train_rms = float(np.sqrt(np.mean(self.inc[1:self.stats_end]**2)))
        initial = float(np.mean(self.inc[1:61]**2))
        if initial <= 0 or self.train_rms <= 0:
            raise ValueError('A positive observed starting variance is required')
        self.variance_floor = max(initial*1e-8, 1e-16)
        self.variance = np.full(len(self.price), np.nan)
        self.variance[60] = max(initial, self.variance_floor)
        for t in range(61, len(self.price)):
            self.variance[t] = max(.94*self.variance[t-1] + .06*self.inc[t]**2, self.variance_floor)
        self.scales = np.sqrt(self.variance)
        train = self.batch(self.train, self.stats_end-1)
        self.base_atoms = regular_quantiles(train['y'])

    def batch(self, origins, known_through, labels=True):
        ids = np.asarray(origins, dtype=np.int64)
        if ids.ndim != 1 or not len(ids) or ids.min() < 60 or ids.max() > known_through:
            raise ValueError('Unknown feature origin')
        if labels and ids.max()+H > known_through:
            raise ValueError('Immature labels')
        x = []
        for t in ids:
            scale = self.scales[t]
            x.append(np.column_stack([
                self.inc[t-L+1:t+1]/scale,
                np.log1p(np.diff(self.day[t-L:t+1]))/np.log(8.),
                np.full(L, np.log(scale/self.train_rms)),
            ]))
        sigma = self.scales[ids, None]*np.sqrt(np.arange(1, H+1))[None, :]
        out = dict(origin=ids.copy(), date=self.dates[ids].copy(), x=np.clip(np.stack(x), -10., 10.),
                   sigma=sigma, base=self.price[ids].copy())
        if labels:
            out['delta'] = self.price[ids[:, None]+np.arange(1, H+1)]-self.price[ids, None]
            out['y'] = out['delta']/sigma
        return out

    def rolling_atoms(self, origins):
        answer = []
        for t in np.asarray(origins, dtype=int):
            # All pool members must have their complete H20 outcome by origin t.
            ids = np.arange(max(60, t-H-119), t-H+1)
            if len(ids) < 2:
                raise ValueError('Insufficient mature rolling pool')
            answer.append(regular_quantiles(self.batch(ids, t)['y']))
        return np.stack(answer)

    def receipt(self):
        return dict(name=self.name, sha256=getattr(self, 'sha256', 'synthetic'), rows=len(self.price),
                    year=self.year, train_n=len(self.train), validation_n=len(self.val), evaluation_n=len(self.test),
                    last_train_label=int(self.train[-1]+H), first_validation_origin=int(self.val[0]),
                    last_validation_label=int(self.val[-1]+H), first_evaluation_origin=int(self.test[0]),
                    last_evaluation_label=int(self.test[-1]+H),
                    train_first_date=str(self.dates[self.train[0]]),
                    evaluation_first_date=str(self.dates[self.test[0]]),
                    evaluation_last_date=str(self.dates[self.test[-1]]),
                    training_statistics_price_end_exclusive=self.stats_end,
                    scale='causal EWMA lambda=.94, initial past60 squared raw increments; sqrt(h) horizon rule',
                    variance_floor=self.variance_floor, training_rms=self.train_rms,
                    empirical_distribution='31 midpoint quantiles, adjacent gap floor1e-4, exact population mean0/variance1',
                    rolling_pool='last120 full-H20 matured standardized outcome vectors; endpoints <= current origin',
                    quote_imputation=False, future_calendar_used=False, covariates_used=False)
