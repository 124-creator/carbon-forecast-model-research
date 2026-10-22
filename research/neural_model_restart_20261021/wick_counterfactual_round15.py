"""Independent round-15 wick counterfactuals; development data only.

2026 is removed before features, donor construction, labels, or model fitting.
Default training and epoch selection use all-bucket allocation cross-entropy,
normalized only by the origin-known past22 risk budget. Short-horizon QLIKE is
an explicitly amplitude-sensitive optional objective, never timing identification.
The imported R14/R15 files are never edited or monkey-patched.  The only temporary
module setting is R15.R.SOURCE, restored immediately after R15.prepare().

Run --check-only for synthetic checks that do not open any market data file.
"""
from __future__ import annotations

import os
for _thread_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_thread_var] = '1'

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import bridge_budget_round15 as B15

HERE = Path(__file__).resolve().parent
EPS = 1e-12
RESERVED_START = np.datetime64('2026-01-01')
VALIDATION_START = np.datetime64('2025-01-01')
SOURCES = {
    'KRBN': HERE / 'data_acquisition_round13' / 'KRBN_ETF_USD_daily_completed_20200731_20261020.csv',
    'KCCA': HERE / 'data_acquisition_round13' / 'KCCA_ETF_USD_daily_completed_20211005_20261020.csv',
}
MODES = [
    'body_matched_transport', 'body_matched_equilibrium',
    'body_only_transport', 'conditional_wick_placebo',
]
SHAPE_NAMES = ['eta', 'xi', 'eta_xi', 'eta_squared', 'xi_squared', 'abs_eta', 'abs_xi']
BODY_COLUMNS = (0, 3, 5)
WICK_COLUMNS = (1, 2, 4, 6)
DONOR_FEATURE_NAMES = ['logrange1', 'abs_eta', 'volume_ratio']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def selected_source(instrument):
    """Use the requested CSV only for prepare; restore the imported setting."""
    original = B15.R.SOURCE
    B15.R.SOURCE = SOURCES[instrument]
    try:
        yield SOURCES[instrument]
    finally:
        B15.R.SOURCE = original


def make_shape(eta, xi):
    eta, xi = np.asarray(eta), np.asarray(xi)
    return np.column_stack([eta, xi, eta * xi, eta ** 2, xi ** 2, np.abs(eta), np.abs(xi)])


def donor_library(source):
    """Construct observed covariates only, after removing the reserved period."""
    raw = pd.read_csv(source, parse_dates=['date'])
    d = raw.loc[raw.date < pd.Timestamp(RESERVED_START)].copy().reset_index(drop=True)
    assert d.date.is_unique and d.date.is_monotonic_increasing
    assert len(d) and d.date.max() < pd.Timestamp(RESERVED_START)
    prices = d[['open', 'high', 'low', 'close']].to_numpy(float)
    assert np.isfinite(prices).all() and (prices > 0).all()
    assert (d.high >= d[['open', 'close']].max(axis=1)).all()
    assert (d.low <= d[['open', 'close']].min(axis=1)).all()
    r = np.log(d.close.to_numpy(float) / d.open.to_numpy(float))
    u = np.log(d.high.to_numpy(float) / d.open.to_numpy(float))
    low = np.log(d.low.to_numpy(float) / d.open.to_numpy(float))
    q = u - low
    eta = np.divide(r, q, out=np.zeros(len(d)), where=q > EPS)
    xi = np.divide(u + low - r, q, out=np.zeros(len(d)), where=q > EPS)
    assert (np.abs(eta) + np.abs(xi) <= 1 + 1e-6).all()
    volume = d.volume.to_numpy(float)
    assert np.isfinite(volume).all() and (volume > 0).all()
    rolling_volume = pd.Series(volume).rolling(22).mean().to_numpy()
    matching = np.column_stack([
        np.log(np.maximum(q ** 2 / (4 * np.log(2)), EPS)),
        np.abs(eta), np.log(volume / rolling_volume),
    ])
    span = np.maximum(1 - np.abs(eta), 0)
    zeta = np.divide(xi, span, out=np.zeros(len(d)), where=span > 1e-10)
    assert (np.abs(zeta) <= 1 + 1e-6).all()
    zero_range = q <= EPS
    return dict(date=d.date.to_numpy(), eta=eta, xi=xi, zero_range=zero_range,
                zeta=np.clip(zeta, -1, 1), matching=matching,
                valid=np.isfinite(matching).all(axis=1) & ~zero_range)


def conditional_placebo(D, library, seed=9031, nearest=32):
    """Past-only matched donors; no target, future shape, or full-sample scale."""
    t = np.asarray(D['t'], dtype=int)
    assert np.all(np.diff(t) > 0)
    np.testing.assert_array_equal(library['date'][t], D['date'])
    np.testing.assert_allclose(D['shape'], make_shape(library['eta'][t], library['xi'][t]),
                               rtol=2e-6, atol=2e-7)
    rng = np.random.default_rng(seed)
    shape = np.empty_like(D['shape'], dtype=np.float32)
    chosen = np.full(len(t), -1, dtype=np.int64)
    pool_size = np.zeros(len(t), dtype=np.int64)
    match_scale = np.full((len(t), 3), np.nan, dtype=float)
    selected_distance = np.full(len(t), np.nan, dtype=float)
    zero_query = library['zero_range'][t]
    all_indices = np.arange(len(library['date']))
    for j, origin in enumerate(t):
        pool = np.flatnonzero(library['valid'] & (all_indices < origin))
        pool_size[j] = len(pool)
        if zero_query[j]:
            np.testing.assert_array_equal(D['shape'][j], np.zeros(7, dtype=np.float32))
            shape[j] = D['shape'][j]
            # No draw and no donor: angular geometry is undefined for q=0.
            continue
        if not len(pool):
            raise ValueError(f'No strictly past finite donor for origin {origin}.')
        assert library['date'][pool].max() < D['date'][j]
        past = library['matching'][pool]
        scale = np.maximum(past.std(axis=0), 1e-5)
        query = library['matching'][origin]
        assert np.isfinite(query).all()
        # Centering by a past-only mean cancels from these pairwise distances.
        distance = np.sum(((past - query) / scale) ** 2, axis=1)
        candidates = np.argsort(distance, kind='stable')[:min(nearest, len(pool))]
        donor_position = int(rng.choice(candidates))
        donor = int(pool[donor_position])
        assert donor < origin
        # Preserve the actual float32 eta, including the body-derived features.
        eta = float(D['shape'][j, 0])
        assert abs(eta) <= 1 + 1e-7
        xi = float(library['zeta'][donor]) * max(1 - abs(eta), 0)
        shape[j] = make_shape(np.array([eta]), np.array([xi]))[0]
        shape[j, list(BODY_COLUMNS)] = D['shape'][j, list(BODY_COLUMNS)]
        chosen[j] = donor
        pool_size[j] = len(pool)
        match_scale[j] = scale
        selected_distance[j] = distance[donor_position]
    has_donor = chosen >= 0
    np.testing.assert_array_equal(has_donor, ~zero_query)
    assert np.all(chosen[has_donor] < t[has_donor])
    assert np.all(library['date'][chosen[has_donor]] < D['date'][has_donor])
    assert (np.abs(shape[:, 0]) + np.abs(shape[:, 1]) <= 1 + 2e-7).all()
    np.testing.assert_array_equal(shape[:, BODY_COLUMNS], D['shape'][:, BODY_COLUMNS])
    donor_dates = np.full(len(t), np.datetime64('NaT'), dtype=library['date'].dtype)
    donor_dates[has_donor] = library['date'][chosen[has_donor]]
    donor_lag = t[has_donor] - chosen[has_donor]
    audit = dict(seed=seed, nearest=nearest, feature_names=DONOR_FEATURE_NAMES,
                 donor_eligibility='finite observed features, nonzero range, donor row < query row; no labels used',
                 matching_scale='per-query population std over strictly past eligible pool; floor 1e-5',
                 draw='uniform among nearest min(32,past_pool_size), stable distance ordering',
                 neutral_geometry='query eta retained; xi=0',
                 zero_range_queries_kept_neutral=int(zero_query.sum()),
                 zero_range_library_rows_excluded=int(library['zero_range'].sum()),
                 zero_range_donor_sentinel='origin=-1,date=NaT; no RNG draw, matching std/distance NaN',
                 all_donors_strictly_past=True, query_count=len(t),
                 min_pool_size=int(pool_size.min()), max_pool_size=int(pool_size.max()),
                 minimum_donor_lag=int(donor_lag.min()) if len(donor_lag) else None,
                 maximum_donor_lag=int(donor_lag.max()) if len(donor_lag) else None)
    details = dict(donor_origin=chosen, donor_date=donor_dates, zero_range_query=zero_query,
                   pool_size=pool_size, matching_std=match_scale,
                   squared_matching_distance=selected_distance)
    return shape, audit, details


def safe_background(D, min_training=100, min_predictions=50, fit_function=None):
    """Match R15 pre-year OOF fitting; skip insufficient early folds explicitly."""
    if fit_function is None:
        fit_function = B15.fit_bucket_model
    background = np.full((len(D['t']), 3), np.nan, dtype=float)
    trace = []
    for year in (2022, 2023, 2024, 2025):
        start = np.datetime64(f'{year}-01-01')
        stop = np.datetime64(f'{year + 1}-01-01')
        train = D['end'] < start
        pred = (D['date'] >= start) & (D['date'] < stop)
        if year < 2025:
            pred &= D['train']
        record = dict(prediction_year=year, training_targets=int(train.sum()),
                      eligible_origins=int(pred.sum()), predicted_origins=0,
                      last_training_label=(str(pd.Timestamp(D['end'][train].max()).date())
                                           if train.any() else None))
        if train.sum() <= min_training or pred.sum() <= min_predictions:
            record.update(status='skipped_insufficient_history_or_origins',
                          required_training=f'>{min_training}', required_origins=f'>{min_predictions}')
            trace.append(record)
            continue
        assert (D['end'][train] < start).all()
        assert (D['date'][pred] >= start).all()
        assert D['end'][train].max() < D['date'][pred].min()
        background[pred] = fit_function(D['x'], D['Y'], train, pred)
        record.update(status='fit_preyear_only', predicted_origins=int(pred.sum()))
        trace.append(record)
    usable = np.isfinite(background).all(axis=1)
    tr = np.flatnonzero(D['train'] & usable)
    va = np.flatnonzero(D['val'] & usable)
    assert len(tr) > 100 and len(va) > 100
    assert not np.intersect1d(tr, va).size
    assert (D['end'][tr] < VALIDATION_START).all()
    assert (D['date'][tr] >= np.datetime64('2022-01-01')).all()
    assert (D['date'][va] >= VALIDATION_START).all()
    assert (D['end'][va] < RESERVED_START).all()
    # Do not silently score a truncated subset if the 2025 background failed.
    np.testing.assert_array_equal(va, np.flatnonzero(D['val']))
    assert (background[usable] > 0).all()
    return background, tr, va, trace


class WickHead(nn.Module):
    """Same 19 -> 12 -> 8 -> 2 architecture for every counterfactual."""
    def __init__(self, nf, mode, mean, sd):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.body = nn.Sequential(nn.Linear(nf, 12), nn.Tanh(), nn.Linear(12, 8), nn.Tanh())
        self.head = nn.Linear(8, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer('raw_zero_shape', torch.tensor(-mean[-7:] / sd[-7:], dtype=torch.float32))

    def actual_and_neutral(self, x):
        actual = x.clone()
        if self.mode == 'body_only_transport':
            for col in WICK_COLUMNS:
                actual[:, -7 + col] = self.raw_zero_shape[col]
        neutral = actual.clone()
        if self.mode == 'body_only_transport':
            neutral[:, -7:] = self.raw_zero_shape
        else:
            # Keep the observed eta, eta^2, |eta| in the normalized coordinates.
            for col in WICK_COLUMNS:
                neutral[:, -7 + col] = self.raw_zero_shape[col]
        return actual, neutral

    def forward(self, x, w):
        actual, neutral = self.actual_and_neutral(x)
        delta = 1.25 * torch.tanh(self.head(self.body(actual)) - self.head(self.body(neutral)))
        if self.mode == 'body_matched_equilibrium':
            d1, d2 = delta.unbind(1)
            adjustment = torch.stack([torch.zeros_like(d1), 2 * d1, 2 * (d1 + d2)], dim=1)
            out = torch.softmax(torch.log(w) + adjustment, dim=1)
        else:
            out = B15.generator_transport(w, delta)
        return out, delta


def mean_risks_np(total, shares):
    """All saved models receive the exact same float64 total/22 column."""
    total = np.asarray(total, dtype=np.float64)
    shares = np.asarray(shares, dtype=np.float64)
    return np.column_stack([total * shares[:, 0],
                            total * (shares[:, 0] + shares[:, 1]) / 5,
                            total / 22])


def known_past_budget(D):
    """R14 raw feature column 2 is log(mean Parkinson risk over past 22 rows)."""
    c = 22 * np.exp(np.asarray(D['x'][:, 2], dtype=np.float64))
    assert np.isfinite(c).all() and (c > 0).all()
    return c


def objective_rows(w, total, bucket_y, past_budget, objective):
    if objective == 'allocation':
        # The normalizer is observed at the origin; never use future Y.sum().
        return -((bucket_y / past_budget[:, None]) * torch.log(w.clamp_min(EPS))).sum(dim=1)
    if objective == 'short_qlike':
        v = B15.mean_risks(total, w)
        target = torch.stack([bucket_y[:, 0], bucket_y[:, :2].sum(dim=1) / 5], dim=1)
        return B15.loss_rows(v[:, :2], target).mean(dim=1)
    raise ValueError(objective)


def allocation_rows_np(w, bucket_y, past_budget):
    return -((bucket_y / past_budget[:, None]) * np.log(np.maximum(w, EPS))).sum(axis=1)


def fit_head(D, background, tr, va, shape, mode, seed, epochs, objective='allocation'):
    # Exclude early rows without OOF forecasts before forming any Torch tensor.
    usable = np.flatnonzero(np.isfinite(background).all(axis=1))
    tr_local = np.searchsorted(usable, tr)
    va_local = np.searchsorted(usable, va)
    np.testing.assert_array_equal(usable[tr_local], tr)
    np.testing.assert_array_equal(usable[va_local], va)
    actual_shape = np.asarray(shape, dtype=np.float32).copy()
    if mode == 'body_only_transport':
        actual_shape[:, WICK_COLUMNS] = 0
    raw_x = np.column_stack([D['x'], actual_shape])[usable]
    mean = raw_x[tr_local].mean(axis=0)
    sd = raw_x[tr_local].std(axis=0) + 1e-5
    x = torch.tensor((raw_x - mean) / sd, dtype=torch.float32)
    total64 = background[usable].sum(axis=1, dtype=np.float64)
    w064 = background[usable] / total64[:, None]
    y64 = np.column_stack([D['Y'][usable, 0], D['Y'][usable, :2].sum(axis=1) / 5,
                           D['Y'][usable].sum(axis=1) / 22])
    total = torch.tensor(total64, dtype=torch.float32)
    w0 = torch.tensor(w064, dtype=torch.float32)
    y = torch.tensor(y64, dtype=torch.float32)
    bucket_y = torch.tensor(D['Y'][usable], dtype=torch.float32)
    past_budget = torch.tensor(known_past_budget(D)[usable], dtype=torch.float32)
    base_v = B15.mean_risks(total, w0)
    base_objective = objective_rows(w0, total, bucket_y, past_budget, objective)
    assert torch.isfinite(x).all() and torch.isfinite(base_v).all()
    assert torch.isfinite(base_objective).all()
    torch.manual_seed(seed)
    net = WickHead(x.shape[1], mode, mean, sd)
    opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.002)
    with torch.no_grad():
        init_w, init_delta = net(x[va_local], w0[va_local])
        assert torch.allclose(init_w, w0[va_local], atol=2e-6, rtol=2e-6)
        assert torch.equal(init_delta, torch.zeros_like(init_delta))
    # Epoch zero is an exact double-precision fallback in the saved predictions.
    best = 0.0
    best_epoch = 0
    state = copy.deepcopy(net.state_dict())
    bad = 0
    history = [[0, 0.0]]
    for epoch in range(1, epochs + 1):
        net.train()
        order = np.random.default_rng(seed + epoch).permutation(tr_local)
        for batch in np.array_split(order, max(1, int(np.ceil(len(order) / 128)))):
            opt.zero_grad()
            w, delta = net(x[batch], w0[batch])
            loss = (objective_rows(w, total[batch], bucket_y[batch], past_budget[batch], objective)
                    - base_objective[batch]).mean()
            loss = loss + .001 * delta.square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2)
            opt.step()
        net.eval()
        with torch.no_grad():
            w, _ = net(x[va_local], w0[va_local])
            score = float((objective_rows(w, total[va_local], bucket_y[va_local],
                                          past_budget[va_local], objective)
                           - base_objective[va_local]).mean())
        history.append([epoch, score])
        if score < best - 1e-5:
            best, best_epoch, bad = score, epoch, 0
            state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
        if bad >= 15 and epoch >= 20:
            break
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        w, delta = net(x[va_local], w0[va_local])
    assert torch.isfinite(w).all() and torch.all(w > 0)
    assert torch.allclose(w.sum(1), torch.ones(len(va)), atol=2e-6, rtol=2e-6)
    shares = w.numpy().astype(np.float64)
    # Correct only checked float32 mass roundoff, never an invalid transport.
    shares /= shares.sum(axis=1, keepdims=True)
    if best_epoch == 0:
        shares = w064[va_local].copy()
    pred = mean_risks_np(total64[va_local], shares)
    assert np.array_equal(pred[:, 2], total64[va_local] / 22)
    return pred, shares, dict(seed=seed, objective=objective, best_epoch=best_epoch,
        epochs_run=len(history) - 1, validation_objective_loss_difference=best,
        parameters=sum(p.numel() for p in net.parameters()),
        mean_abs_edge_tilt=float(delta.abs().mean()),
        epoch_zero_exact_background=True, feature_mean=mean.tolist(), feature_std=sd.tolist(), history=history)


def synthetic_checks():
    """Pure synthetic checks: never call prepare, donor_library, or read a CSV."""
    rng = np.random.default_rng(771)
    eta = np.linspace(-.7, .7, 45)
    xi = .5 * (1 - np.abs(eta)) * np.sin(np.arange(len(eta)))
    dates = np.datetime64('2023-01-01') + np.arange(len(eta)).astype('timedelta64[D]')
    matching = np.column_stack([np.arange(len(eta)) / 30, np.abs(eta), np.cos(np.arange(len(eta)))])
    library = dict(date=dates, eta=eta, xi=xi, zeta=xi / (1 - np.abs(eta)),
                   matching=matching, valid=np.ones(len(eta), dtype=bool),
                   zero_range=np.zeros(len(eta), dtype=bool))
    origins = np.array([10, 20, 30, 40])
    d = dict(t=origins, date=dates[origins], shape=make_shape(eta[origins], xi[origins]).astype(np.float32))
    placebo, _, detail = conditional_placebo(d, library)
    assert np.all(detail['donor_origin'] < origins)
    assert detail['pool_size'].tolist() == origins.tolist()
    altered = {key: value.copy() for key, value in library.items()}
    altered['matching'][20:] += 1000
    altered['xi'][20:] *= -1
    altered['zeta'][20:] *= -1
    # The first query's entire future must be irrelevant to its donor and shape.
    first = {key: value[:1].copy() for key, value in d.items()}
    placebo_a, _, detail_a = conditional_placebo(first, library)
    placebo_b, _, detail_b = conditional_placebo(first, altered)
    np.testing.assert_array_equal(placebo_a, placebo_b)
    np.testing.assert_array_equal(detail_a['matching_std'], detail_b['matching_std'])
    np.testing.assert_array_equal(detail_a['donor_origin'], detail_b['donor_origin'])
    raw = np.column_stack([rng.normal(size=(4, 12)), d['shape']]).astype(np.float32)
    mean, sd = raw.mean(axis=0), raw.std(axis=0) + 1e-5
    normalized = torch.tensor((raw - mean) / sd)
    weights = torch.tensor([[.05, .15, .8]] * 4, dtype=torch.float32)
    capacities = []
    for mode in MODES:
        net = WickHead(raw.shape[1], mode, mean, sd)
        actual, neutral = net.actual_and_neutral(normalized)
        neutral_shape = neutral.detach().numpy()[:, -7:] * sd[-7:] + mean[-7:]
        np.testing.assert_allclose(neutral_shape[:, WICK_COLUMNS], 0, atol=2e-7)
        if mode != 'body_only_transport':
            np.testing.assert_allclose(neutral_shape[:, BODY_COLUMNS], raw[:, -7:][:, BODY_COLUMNS], atol=2e-7)
        else:
            np.testing.assert_allclose(neutral_shape, 0, atol=2e-7)
            actual_shape = actual.detach().numpy()[:, -7:] * sd[-7:] + mean[-7:]
            np.testing.assert_allclose(actual_shape[:, WICK_COLUMNS], 0, atol=2e-7)
        out, delta = net(normalized, weights)
        assert torch.allclose(out, weights, atol=2e-6, rtol=2e-6)
        assert torch.equal(delta, torch.zeros_like(delta))
        capacities.append(sum(p.numel() for p in net.parameters()))
        if mode != 'body_only_transport':
            # With a learned nonzero head, a body-matched zero-wick candle is still nested.
            with torch.no_grad():
                net.head.weight.fill_(.03)
            _, neutral_delta = net(neutral, weights)
            assert torch.equal(neutral_delta, torch.zeros_like(neutral_delta))
    assert len(set(capacities)) == 1
    # Check the nonzero-tilt equilibrium against detailed balance independently.
    w_reference = np.array([[.05, .15, .8], [.3, .3, .4]], dtype=np.float64)
    edge_tilt = np.array([[.7, -.8], [-.5, .3]], dtype=np.float64)
    potential = np.column_stack([np.zeros(2), 2 * edge_tilt[:, 0], 2 * edge_tilt.sum(axis=1)])
    stationary = w_reference * np.exp(potential)
    stationary /= stationary.sum(axis=1, keepdims=True)
    flux12 = stationary[:, 0] * w_reference[:, 1] * np.exp(edge_tilt[:, 0])
    flux21 = stationary[:, 1] * w_reference[:, 0] * np.exp(-edge_tilt[:, 0])
    flux23 = stationary[:, 1] * w_reference[:, 2] * np.exp(edge_tilt[:, 1])
    flux32 = stationary[:, 2] * w_reference[:, 1] * np.exp(-edge_tilt[:, 1])
    np.testing.assert_allclose(flux12, flux21, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(flux23, flux32, rtol=1e-12, atol=1e-15)
    # A KCCA-like synthetic calendar must skip 2022 yet fit only pre-year labels.
    calendar = pd.bdate_range('2021-10-05', '2025-12-31').to_numpy()
    query = np.arange(100, len(calendar) - 22)
    query_date, label_end = calendar[query], calendar[query + 22]
    allowed = (label_end < VALIDATION_START) | (query_date >= VALIDATION_START)
    query, query_date, label_end = query[allowed], query_date[allowed], label_end[allowed]
    fake = dict(t=query, date=query_date, end=label_end,
                train=label_end < VALIDATION_START, val=query_date >= VALIDATION_START,
                x=np.ones((len(query), 12)), Y=np.ones((len(query), 3)))
    def fake_fit(x_input, y_input, fit_rows, forecast_rows):
        assert label_end[fit_rows].max() < query_date[forecast_rows].min()
        return np.ones((int(forecast_rows.sum()), 3))
    _, fake_tr, _, fake_trace = safe_background(fake, fit_function=fake_fit)
    assert fake_trace[0]['status'].startswith('skipped')
    assert {record['prediction_year'] for record in fake_trace if record['status'] == 'fit_preyear_only'} == {2023, 2024, 2025}
    assert (fake['date'][fake_tr] >= np.datetime64('2023-01-01')).all()
    total = np.array([1.123456789e-4, 2.345678901e-4])
    p = mean_risks_np(total, np.array([[.1, .2, .7], [.3, .3, .4]]))
    assert np.array_equal(p[:, 2], total / 22)
    # Zero-range queries have no observed angle: preserve neutrality and draw no donor.
    zero_library = {key: value.copy() for key, value in library.items()}
    origin = int(origins[0])
    zero_library['eta'][origin] = 0
    zero_library['xi'][origin] = 0
    zero_library['zeta'][origin] = 0
    zero_library['zero_range'][origin] = True
    zero_library['valid'][origin] = False
    zero_d = dict(t=origins[:1], date=dates[origins[:1]], shape=np.zeros((1, 7), dtype=np.float32))
    zero_shape, zero_audit, zero_details = conditional_placebo(zero_d, zero_library)
    np.testing.assert_array_equal(zero_shape, zero_d['shape'])
    assert zero_details['donor_origin'][0] == -1 and np.isnat(zero_details['donor_date'][0])
    assert zero_audit['zero_range_queries_kept_neutral'] == 1
    # Population shares must remain w0 under any common positive amplitude multiplier.
    reference = torch.tensor([[.05, .15, .8]], dtype=torch.float64)
    distorted = torch.tensor([[.1, .3, .6]], dtype=torch.float64)
    unit = torch.ones(1, dtype=torch.float64)
    for amplitude in (.5, 2., 7.):
        logits = torch.log(reference).detach().clone().requires_grad_(True)
        shares = torch.softmax(logits, dim=1)
        synthetic_y = amplitude * reference
        proper = objective_rows(shares, unit, synthetic_y, unit, 'allocation').mean()
        gradient = torch.autograd.grad(proper, logits)[0]
        assert torch.allclose(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)
        alt = objective_rows(distorted, unit, synthetic_y, unit, 'allocation').mean()
        assert float((alt - proper).detach()) > 0
    # Fixed forecast total=1, true total=2: optimizing only near horizons favors stealing.
    synthetic_y = 2 * reference
    short_difference = float((objective_rows(distorted, unit, synthetic_y, unit, 'short_qlike')
                              - objective_rows(reference, unit, synthetic_y, unit, 'short_qlike')).mean())
    allocation_difference = float((objective_rows(distorted, unit, synthetic_y, unit, 'allocation')
                                   - objective_rows(reference, unit, synthetic_y, unit, 'allocation')).mean())
    far_truth = synthetic_y[:, 2] / 17
    far_before = unit * reference[:, 2] / 17
    far_after = unit * distorted[:, 2] / 17
    far_difference = float((B15.loss_rows(far_after, far_truth) - B15.loss_rows(far_before, far_truth)).mean())
    assert short_difference < 0 and allocation_difference > 0 and far_difference > 0
    fake_budget = known_past_budget(dict(x=np.column_stack([np.zeros(3), np.zeros(3), np.log([.01, .02, .03])])))
    np.testing.assert_allclose(fake_budget, 22 * np.array([.01, .02, .03]), rtol=1e-12)
    return dict(market_data_opened=False, past_only_donors=True,
                donor_prefix_invariance=True, geometric_legality=True,
                neutral_raw_units=True, zero_head_nests_background=True,
                neutral_wick_nests_with_nonzero_head=True,
                skipped_early_preyear_oof_checked=True,
                equilibrium_detailed_balance=True,
                equal_parameter_counts=capacities[0], double_budget_exact=True,
                zero_range_queries_stay_neutral=True, known_past_budget_only=True,
                allocation_optimum_independent_of_common_amplitude=True,
                short_qlike_can_steal_far_budget=True,
                amplitude_counterexample=dict(short_qlike_difference=short_difference,
                                              allocation_difference=allocation_difference,
                                              far_bucket_qlike_difference=far_difference))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instrument', choices=list(SOURCES), default='KRBN')
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--objective', choices=['allocation', 'short_qlike'], default='allocation')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--check-only', action='store_true', help='Run synthetic checks only; do not read market data.')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.check_only:
        print(json.dumps(synthetic_checks(), indent=2), flush=True)
        return
    if args.epochs < 0 or not args.seeds:
        raise ValueError('epochs must be nonnegative; at least one seed is required')
    tag = args.tag or f'wick_counterfactual_round15_{args.instrument}_{args.objective}_development'
    if not re.fullmatch(r'wick_counterfactual_round15_[A-Za-z0-9_-]+', tag):
        raise ValueError('Use a new tag starting wick_counterfactual_round15_; old result tags are protected.')
    paths = dict(manifest=HERE / f'{tag}_manifest.json',
                 results=HERE / f'{tag}_results.json',
                 predictions=HERE / f'{tag}_validation_predictions.npz')
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite an existing result; choose another --tag: {path}')
    started = time.time()
    source = SOURCES[args.instrument]
    source_paths = dict(wick_counterfactual_script=Path(__file__).resolve(),
                        bridge_budget_script=Path(B15.__file__).resolve(),
                        range_clock_script=Path(B15.R.__file__).resolve(), csv=source)
    hashes = {name: dict(path=str(path), sha256=sha256(path)) for name, path in source_paths.items()}
    original_source = B15.R.SOURCE
    with selected_source(args.instrument):
        D, audit = B15.prepare()
    assert B15.R.SOURCE == original_source
    assert audit['feature_names'][2] == 'logrange22'
    library = donor_library(source)
    np.testing.assert_array_equal(library['date'][D['t']], D['date'])
    background, tr, va, trace = safe_background(D)
    placebo_shape, placebo_audit, placebo_detail = conditional_placebo(D, library)
    total = background[va].sum(axis=1, dtype=np.float64)
    w0 = background[va] / total[:, None]
    targets = np.column_stack([D['Y'][va, 0], D['Y'][va, :2].sum(axis=1) / 5,
                               D['Y'][va].sum(axis=1) / 22])
    pred = dict(level_allocator=mean_risks_np(total, w0),
                uniform=np.repeat((total / 22)[:, None], 3, axis=1))
    weights_out = dict(level_allocator=w0,
                       uniform=np.broadcast_to(np.array([1., 4., 17.]) / 22, w0.shape).copy())
    np.testing.assert_array_equal(va, np.flatnonzero(D['val']))
    full_x = np.column_stack([D['x'], D['shape']])
    full_bucket = B15.fit_bucket_model(full_x, D['Y'], D['train'], D['val'])
    full_shares = full_bucket / full_bucket.sum(axis=1, keepdims=True)
    pred['HGB_full_allocator'] = mean_risks_np(total, full_shares)
    weights_out['HGB_full_allocator'] = full_shares
    config = dict(config=vars(args), tag=tag,
        status='DEVELOPMENT ONLY; 2025 validation selects epochs; no 2026 targets or forecasts',
        target='future mean Parkinson intraday-range variance, not returns RV or true IV',
        horizons=[1, 5, 22], bucket_lengths=[1, 4, 17], audit=audit, hashes=hashes,
        background_trace=trace, head_training_origins=len(tr), validation_origins=len(va),
        strong_baseline=dict(name='HGB_full_allocator', training_origins=int(D['train'].sum()),
                             inputs='same full context plus actual seven geometry features',
                             note='uses all pre2025 mature training rows, potentially more than OOF residual heads; same float64 T'),
        head_training_years=sorted(np.unique(D['date'][tr].astype('datetime64[Y]').astype(int) + 1970).tolist()),
        objective=args.objective,
        objective_formula=('-sum_b(Y_b/c_t)*log(w_b)' if args.objective == 'allocation'
                           else 'mean h1/h5 QLIKE; old amplitude-sensitive sensitivity objective'),
        allocation_normalizer='c_t=22*exp(raw D.x[:,2]); known past22 mean Parkinson budget; never future realized total',
        estimand='allocation optimum is conditional bucket-mean ratio; finite-capacity fitting still weights states by future total/c_t',
        selection='2025 same-objective loss difference versus fixed w0; epoch zero included; no test selection',
        loss='QLIKE log(v)+y/v is a performance report; paired differences, never percent changes in possibly negative scores',
        modes=MODES, shape_features=SHAPE_NAMES, body_columns=list(BODY_COLUMNS), wick_columns=list(WICK_COLUMNS),
        equilibrium_control='softmax(log(w0)+[0,2*delta1,2*(delta1+delta2)]), stationary law of the same tilted generator',
        shared_tilt='all heads use 1.25*tanh(actual-neutral) and 0.001*mean(delta^2) training regularization',
        information_note='background already contains body and range; this is conditional incremental wick geometry',
        neutral='preserve eta, eta^2, abs(eta); set xi, eta*xi, xi^2, abs(xi) to raw zero using train-only normalization',
        placebo=placebo_audit, seed_aggregation='arithmetic mean allocation shares; identical float64 total budget',
        precision='float32 neural fitting; checked shares promoted and normalized in float64; exact shared h22=total/22',
        checks=synthetic_checks(), outputs={name: str(path) for name, path in paths.items()}, fits={}, scores={})
    paths['manifest'].write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    for mode in MODES:
        shape = placebo_shape if mode == 'conditional_wick_placebo' else D['shape']
        shares_by_seed, fits = [], []
        for seed in args.seeds:
            _, shares, fit = fit_head(D, background, tr, va, shape, mode, seed, args.epochs,
                                      objective=args.objective)
            shares_by_seed.append(shares)
            fits.append(fit)
            print(args.instrument, mode, seed, 'best_epoch', fit['best_epoch'],
                  'val_delta', round(fit['validation_objective_loss_difference'], 7), flush=True)
        shares = np.mean(shares_by_seed, axis=0, dtype=np.float64)
        shares /= shares.sum(axis=1, keepdims=True)
        if all(fit['best_epoch'] == 0 for fit in fits):
            shares = w0.copy()
        pred[mode] = mean_risks_np(total, shares)
        weights_out[mode] = shares
        config['fits'][mode] = fits
    capacity = {fit['parameters'] for fits in config['fits'].values() for fit in fits}
    assert len(capacity) == 1
    for name, p in pred.items():
        assert p.dtype == np.float64 and np.isfinite(p).all() and (p > 0).all()
        np.testing.assert_array_equal(p[:, 2], total / 22)
    past_budget_va = known_past_budget(D)[va]
    bucket_lengths = np.array([1., 4., 17.])
    bucket_truth = D['Y'][va] / bucket_lengths[None, :]
    bucket_predictions = {name: total[:, None] * shares / bucket_lengths[None, :]
                          for name, shares in weights_out.items()}
    allocation_losses = {name: allocation_rows_np(shares, D['Y'][va], past_budget_va)
                         for name, shares in weights_out.items()}
    bucket_losses = {name: np.log(value) + bucket_truth / value
                     for name, value in bucket_predictions.items()}
    config['allocation_scores'] = {}
    config['bucket_mean_scores'] = {}
    config['three_bucket_mean_QLIKE_difference'] = {}
    for name in pred:
        config['allocation_scores'][name] = dict(
            n=len(va), mean_loss=float(allocation_losses[name].mean()),
            loss_difference_vs_level=float((allocation_losses[name] - allocation_losses['level_allocator']).mean()))
        config['three_bucket_mean_QLIKE_difference'][name] = float(
            (bucket_losses[name] - bucket_losses['level_allocator']).mean())
    for j, bucket in enumerate(['step1', 'steps2_5', 'steps6_22']):
        config['bucket_mean_scores'][bucket] = {}
        for name, value in bucket_predictions.items():
            config['bucket_mean_scores'][bucket][name] = dict(
                n=len(va), bucket_length=int(bucket_lengths[j]),
                loss_difference_vs_level=float((bucket_losses[name][:, j] - bucket_losses['level_allocator'][:, j]).mean()),
                MSE=float(np.mean((value[:, j] - bucket_truth[:, j]) ** 2)))
    for j, horizon in enumerate([1, 5, 22]):
        base_loss = np.log(pred['level_allocator'][:, j]) + targets[:, j] / pred['level_allocator'][:, j]
        config['scores'][str(horizon)] = {}
        for name, p in pred.items():
            loss = np.log(p[:, j]) + targets[:, j] / p[:, j]
            config['scores'][str(horizon)][name] = dict(
                n=len(va), loss_difference_vs_level=float((loss - base_loss).mean()),
                MSE=float(np.mean((p[:, j] - targets[:, j]) ** 2)))
        print(args.instrument, 'h', horizon,
              {name: round(score['loss_difference_vs_level'], 7)
               for name, score in config['scores'][str(horizon)].items()}, flush=True)
    # Detect an accidental source change during the run; never edit those sources.
    for name, path in source_paths.items():
        assert sha256(path) == hashes[name]['sha256'], f'Source changed during run: {name}'
    config['elapsed_seconds'] = time.time() - started
    config['h22_exact_shared_budget'] = True
    paths['results'].write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    arrays = dict(date=D['date'][va], origin=D['t'][va], label_end=D['end'][va],
                  horizons=np.array([1, 5, 22]), target=targets,
                  total_budget=total, baseline_weights=w0, known_past_budget=past_budget_va,
                  bucket_target_sums=D['Y'][va], bucket_target_means=bucket_truth,
                  actual_shape=D['shape'][va], placebo_shape=placebo_shape[va])
    arrays.update(pred)
    arrays.update({f'{name}_weights': shares for name, shares in weights_out.items()})
    arrays.update({f'{name}_bucket_means': value for name, value in bucket_predictions.items()})
    arrays.update({f'{name}_allocation_loss': value for name, value in allocation_losses.items()})
    arrays.update({f'{name}_bucket_mean_QLIKE': value for name, value in bucket_losses.items()})
    arrays.update({f'placebo_{name}': value[va] for name, value in placebo_detail.items()})
    np.savez_compressed(paths['predictions'], **arrays)
    print('DEVELOPMENT FINISHED; RESERVED PERIOD UNUSED', round(config['elapsed_seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
