"""Causal, fixed-source experiments with an explicit target supervision budget.

Only the five previously approved development files are accepted. A supervision
budget is NOT a short-history or new-market cold-start claim: observed histories
remain available as causal features, and prices can algebraically imply labels.
"""
from __future__ import annotations
import csv
import hashlib
from datetime import date
from pathlib import Path
import numpy as np

H, L = 20, 20
HORIZONS = np.array([0, 4, 19])
SPECS = [
    ('EEX_EUA_T3PA', None, '5a0c7580edebc2a7a5e66d7487568f6d99a8e08cca0785676c2aaeccad7db6bb'),
    ('GUANGZHOU', 'GUANGZHOU_development.csv', '8d78a4f13c5a799519546fd3bfa0e0ac5eaa52b0441a983a95ef06f5e5e34051'),
    ('HUBEI', 'HUBEI_development.csv', '04655d59c1b585f6c35c02a43469a965fe3589eb613d7d8d8e46996751b7d797'),
    ('SHANGHAI', 'SHANGHAI_D4_development.csv', '425424353bb3ac09e3d96b291680b4179e86f180da626a5d19e5c1e58ed960a1'),
    ('EU_REPORTED', 'EU_REPORTED_development.csv', 'ca7c192380897076ef57516a11ea2ccefda6ba92ab9823d9aa8f2bb281cefedc'),
]


class Series:
    def __init__(self, name, dates, price, digest='synthetic'):
        self.name, self.sha256 = name, digest
        self.dates = np.asarray(dates, dtype='U10')
        self.price = np.asarray(price, dtype=np.float64)
        if len(self.price) != len(self.dates) or len(self.price) <= 100:
            raise ValueError('Insufficient or misaligned series')
        if np.any(self.dates[1:] <= self.dates[:-1]) or not np.isfinite(self.price).all() or np.any(self.price <= 0):
            raise ValueError('No cleaning, imputation, rounding or reordering is permitted')
        self.days = np.array([date.fromisoformat(x).toordinal() for x in self.dates])
        self.inc = np.r_[0., np.diff(self.price)]
        initial = float(np.mean(self.inc[1:61] ** 2))
        if initial <= 0:
            raise ValueError('Initial causal volatility must be positive')
        self.initial_rms = initial ** .5
        self.floor = max(initial * 1e-8, 1e-16)
        variance = np.full(len(self.price), np.nan)
        variance[60] = max(initial, self.floor)
        for t in range(61, len(self.price)):
            variance[t] = max(.94 * variance[t-1] + .06 * self.inc[t] ** 2, self.floor)
        self.scale = np.sqrt(variance)

    def batch(self, ids, known_through, labels=True):
        ids = np.asarray(ids, dtype=np.int64)
        if not len(ids) or np.any(ids < 60) or np.any(ids > known_through):
            raise ValueError('Unknown feature origin')
        if labels and np.any(ids + H > known_through):
            raise ValueError('Immature complete-H20 labels')
        x = []
        for t in ids:
            x.append(np.column_stack((
                self.inc[t-L+1:t+1] / self.scale[t],
                np.log1p(np.diff(self.days[t-L:t+1])) / np.log(8.),
                np.full(L, np.log(self.scale[t] / self.initial_rms)),
            )))
        sigma = self.scale[ids, None] * np.sqrt(np.arange(1, H+1))[None, :]
        result = {'x': np.clip(np.stack(x), -10., 10.), 'origin': ids.copy(),
                  'date': self.dates[ids].copy(), 'sigma': sigma,
                  'base': self.price[ids].copy()}
        if labels:
            result['delta'] = self.price[ids[:, None] + np.arange(1, H+1)] - self.price[ids, None]
            result['y'] = result['delta'] / sigma
            result['last_label_date'] = self.dates[ids+H].copy()
        return result

    def split(self, budget, year):
        if budget not in (100, 400):
            raise ValueError('Unfrozen target budget')
        start = int(np.searchsorted(self.dates, f'{year}-01-01'))
        end = int(np.searchsorted(self.dates, f'{year+1}-01-01'))
        # Define the full400 calendar spans once. The100 condition takes nested,
        # evenly spaced origins with the same endpoints, avoiding a recency or
        # time-coverage explanation for changes in supervision budget.
        ns, nq, nv = 240, 80, 80
        available = np.arange(60, start-H)
        if len(available) < ns+nq+nv+2*H or end-start <= H:
            raise ValueError('Insufficient history for the fixed400 spans or evaluation year')
        val = available[-nv:]
        before_val = available[available+H < val[0]]
        meta = before_val[-nq:]
        before_meta = available[available+H < meta[0]]
        support = before_meta[-ns:]
        test = np.arange(start, end-H)
        if tuple(map(len, (support, meta, val))) != (ns, nq, nv) or not len(test):
            raise ValueError('Insufficient purged supervision or evaluation origins')
        if budget == 100:
            support = support[np.rint(np.linspace(0, len(support)-1, 60)).astype(int)]
            meta = meta[np.rint(np.linspace(0, len(meta)-1, 20)).astype(int)]
            val = val[np.rint(np.linspace(0, len(val)-1, 20)).astype(int)]
        assert support[-1]+H < meta[0] and meta[-1]+H < val[0]
        assert val[-1]+H < start
        return dict(support=support, meta=meta, val=val, test=test, start=start, end=end)


def load_series(repo_root, eex_path):
    repo_root = Path(repo_root)
    result = {}
    for name, filename, digest in SPECS:
        path = Path(eex_path) if filename is None else repo_root/'research/cross_market_innovation_20261017/data'/filename
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f'Input identity mismatch: {name}')
        rows = list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
        dates = [r['date'][:10] for r in rows]
        if any(d >= '2024-01-01' for d in dates):
            raise ValueError('This runner refuses later development/confirmation prices')
        result[name] = Series(name, dates, [float(r['price']) for r in rows], digest)
    return result


def make_case(series, target, budget, year=2022):
    target_series = series[target]
    split = target_series.split(budget, year)
    # Same rich source history for both target supervision budgets. No source
    # label reaches the earliest target meta origin under either budget.
    cutoff = min(str(target_series.dates[target_series.split(b, year)['meta'][0]]) for b in (100, 400))
    case = {'target': target, 'budget': budget, 'year': year, 'sources': {}, 'source_cutoff': cutoff}
    for role in ('support', 'meta', 'val'):
        case['target_'+role] = target_series.batch(split[role], split['start']-1)
    case['target_test'] = target_series.batch(split['test'], split['end']-1)
    excluded = []
    for name, src in series.items():
        if name == target:
            continue
        if {name, target} == {'EEX_EUA_T3PA', 'EU_REPORTED'}:
            excluded.append({'name': name, 'reason': 'related EU ETS series excluded as mutual transfer sources'})
            continue
        known = int(np.searchsorted(src.dates, cutoff))-1
        ids = np.arange(60, known-H+1)
        if len(ids) < 64+H+32:
            raise ValueError(f'Insufficient fixed rich-source episode support: {name} for {target}')
        batch = src.batch(ids, known)
        assert max(batch['last_label_date']) < cutoff
        case['sources'][name] = batch
    if not case['sources']:
        raise ValueError('No source series')
    def role_info(role):
        batch = case['target_'+role]
        return {'n': len(batch['origin']), 'first_origin': int(batch['origin'][0]),
                'last_origin': int(batch['origin'][-1]), 'first_date': str(batch['date'][0]),
                'last_date': str(batch['date'][-1]),
                'last_label_date': str(batch['last_label_date'][-1])}
    case['receipt'] = {
        'target': target, 'target_sha256': target_series.sha256, 'year': year, 'budget': budget,
        'budget_interpretation': 'retained supervised origin windows;100 is nested and spans the same role endpoints as400; causal full past quote histories remain available; not a cold-start or independent-label count',
        'roles': {role: role_info(role) for role in ('support', 'meta', 'val', 'test')},
        'source_cutoff_exclusive': cutoff,
        'sources': {name: {'n': len(batch['origin']), 'sha256': series[name].sha256,
                           'first_origin': int(batch['origin'][0]), 'last_origin': int(batch['origin'][-1]),
                           'first_date': str(batch['date'][0]), 'last_label_date': str(batch['last_label_date'][-1])}
                    for name, batch in case['sources'].items()},
        'excluded_sources': excluded, 'H': H, 'L': L,
        'initial_volatility': 'first60 raw increments; common causal EWMA .94 thereafter',
        'new_shenzhen_used': False, 'confirmation_used': False,
    }
    return case


def source_episode(batch, rng, support_n=64, query_n=32):
    """A chronological source episode, with a full-H label purge."""
    n = len(batch['origin'])
    q0 = int(rng.integers(support_n+H, n-query_n+1))
    s_ids = np.arange(q0-H-support_n, q0-H)
    q_ids = np.arange(q0, q0+query_n)
    assert batch['origin'][s_ids[-1]]+H < batch['origin'][q_ids[0]]
    keys = ('x', 'y', 'origin', 'date', 'last_label_date')
    return ({k: batch[k][s_ids] for k in keys}, {k: batch[k][q_ids] for k in keys})
