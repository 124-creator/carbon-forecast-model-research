"""Round 16: Brownian-shape measurement precision for Parkinson-proxy forecasts.

Independent development-only experiment. Remove 2026 before observations,
features, or labels. --check-only uses synthetic arrays and never opens a CSV.
All modes use the same bounded AR1 form/initialization/training procedure, with
parameters fitted separately per mode; dynamics are not numerically tied.
"""
from __future__ import annotations
import os
for _thread_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_thread_var] = '1'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
from torch import nn
import range_clock_round14 as R14

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / 'brownian_reference_round16' / 'brownian_shape_models.json'
SOURCES = {
    'KRBN': HERE / 'data_acquisition_round13' / 'KRBN_ETF_USD_daily_completed_20200731_20261020.csv',
    'KCCA': HERE / 'data_acquisition_round13' / 'KCCA_ETF_USD_daily_completed_20211005_20261020.csv',
}
HORIZONS = (1, 5, 22)
MODES = ('raw_constant', 'bias_constant', 'fixed_precision', 'neural_precision', 'neural_constant_precision')
NEURAL_MODES = ('neural_precision', 'neural_constant_precision')
EPS = 1e-12
CUT_DATE = np.datetime64('2025-01-01')
RESERVED = np.datetime64('2026-01-01')
DTYPE = torch.float64


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_cells(a, z, model):
    edges_a, edges_z = np.asarray(model['edges_a']), np.asarray(model['edges_z'])
    na, nz = len(edges_a) - 1, len(edges_z) - 1
    ia = np.clip(np.searchsorted(edges_a, np.clip(a, 0, 1), side='right') - 1, 0, na - 1)
    iz = np.clip(np.searchsorted(edges_z, np.clip(z, 0, 1), side='right') - 1, 0, nz - 1)
    return ia * nz + iz


def observation_arrays(d):
    """All rows are already pre2026. Missing angles are neutral and flagged."""
    assert d.date.is_monotonic_increasing and d.date.is_unique
    assert d.date.max() < pd.Timestamp(RESERVED)
    px = d[['open', 'high', 'low', 'close']].to_numpy(float)
    assert np.isfinite(px).all() and (px > 0).all()
    assert (d.high >= d[['open', 'close']].max(axis=1)).all()
    assert (d.low <= d[['open', 'close']].min(axis=1)).all()
    r = np.log(d.close.to_numpy(float) / d.open.to_numpy(float))
    u = np.log(d.high.to_numpy(float) / d.open.to_numpy(float))
    low = np.log(d.low.to_numpy(float) / d.open.to_numpy(float))
    q = u - low
    observed = q > EPS
    eta = np.divide(r, q, out=np.zeros(len(d)), where=observed)
    xi = np.divide(u + low - r, q, out=np.zeros(len(d)), where=observed)
    assert (np.abs(eta) + np.abs(xi) <= 1 + 1e-6).all()
    a = np.abs(eta)
    z = np.divide(np.abs(xi), 1 - a, out=np.zeros(len(d)), where=observed & ((1 - a) > EPS))
    assert (z >= 0).all() and (z <= 1 + 1e-6).all()
    z = np.clip(z, 0, 1)
    proxy = q ** 2 / (4 * np.log(2))
    log_proxy = np.zeros(len(d), dtype=float)
    log_proxy[observed] = np.log(proxy[observed])
    volume = d.volume.to_numpy(float)
    assert np.isfinite(volume).all() and (volume >= 0).all()
    with np.errstate(divide='ignore', invalid='ignore'):
        volume_ratio = np.log(volume / pd.Series(volume).rolling(22).mean().to_numpy())
    even = np.column_stack([a, z, volume_ratio])
    return dict(proxy=proxy, log_proxy=log_proxy, observed=observed, zero_range=~observed,
                eta=eta, xi=xi, even_raw=even)


def prepare(instrument, reference_grid):
    source = SOURCES[instrument]
    original = R14.SOURCE
    try:
        R14.SOURCE = source
        old, audit = R14.build()
    finally:
        R14.SOURCE = original
    raw = pd.read_csv(source, parse_dates=['date'])
    d = raw.loc[raw.date < pd.Timestamp(RESERVED)].copy().reset_index(drop=True)
    obs = observation_arrays(d)
    n = len(d)
    cut = int(np.searchsorted(d.date.to_numpy(), CUT_DATE))
    assert cut == old['cut'] and 100 < cut < n
    model = json.loads(REFERENCE.read_text(encoding='utf-8'))[f'grid{reference_grid}']
    cells = reference_cells(obs['even_raw'][:, 0], obs['even_raw'][:, 1], model)
    cell_b = np.asarray(model['b_by_cell'], dtype=float)[cells]
    cell_R = np.asarray(model['R_by_cell'], dtype=float)[cells]
    assert np.isfinite(cell_b).all() and np.isfinite(cell_R).all() and (cell_R > 0).all()
    ids = np.flatnonzero(old['h'] == 22)
    origin = old['t'][ids]
    labels = np.column_stack([np.array([obs['proxy'][t + 1:t + h + 1].mean() for t in origin]) for h in HORIZONS])
    np.testing.assert_allclose(labels[:, 2], old['y'][ids], rtol=2e-6, atol=1e-12)
    tr = np.flatnonzero(origin + 22 < cut)
    va = np.flatnonzero(origin >= cut)
    assert len(tr) > 100 and len(va) > 100
    assert len(tr) + len(va) == len(origin)
    assert (origin[tr] + 22 < cut).all() and (origin[va] + 22 < n).all()
    assert (d.date.to_numpy()[origin[va]] >= CUT_DATE).all()
    assert (d.date.to_numpy()[origin[va] + 22] < RESERVED).all()
    # Fixed normalization and initial level use pre2025 observed data only.
    pre = (np.arange(n) < cut) & obs['observed']
    means, stds = [], []
    for j in range(3):
        mask = pre & np.isfinite(obs['even_raw'][:, j])
        assert mask.any()
        means.append(float(obs['even_raw'][mask, j].mean()))
        stds.append(float(obs['even_raw'][mask, j].std()) + 1e-5)
    means, stds = np.array(means), np.array(stds)
    even = (obs['even_raw'] - means) / stds
    missing = ~np.isfinite(even)
    even[missing] = 0  # training-mean imputation after train-only normalization
    initial_mu = float((obs['log_proxy'][pre] - model['global_b']).mean())
    x_baseline = np.column_stack([old['x'][ids], even[origin]])
    assert np.isfinite(x_baseline).all()
    D = dict(n=n, cut=cut, date=d.date.to_numpy(), origin=origin, y=labels,
             tr=tr, va=va, log_proxy=obs['log_proxy'], observed=obs['observed'],
             even=even, cell_b=cell_b, cell_R=cell_R,
             global_b=float(model['global_b']), global_R=float(model['global_R_ddof1']),
             constant_R=float(model['shape_bias_constant_R_train_residual_mean_square']),
             initial_mu=initial_mu, ewma=old['ewma'][ids].astype(float),
             baseline_x=x_baseline, cell=cells, zero_range=obs['zero_range'])
    assert D['global_R'] > 0 and D['constant_R'] > 0
    audit.update(common_h22_origins=True, target='future mean Parkinson intraday-range proxy; not true QV or return RV',
                 train_common_origins=len(tr), validation_common_origins=len(va),
                 last_training_origin=str(pd.Timestamp(d.date.iloc[origin[tr].max()]).date()),
                 last_training_label=str(pd.Timestamp(d.date.iloc[(origin[tr] + 22).max()]).date()),
                 first_validation_origin=str(pd.Timestamp(d.date.iloc[origin[va].min()]).date()),
                 last_validation_label=str(pd.Timestamp(d.date.iloc[(origin[va] + 22).max()]).date()),
                 initial_mu=initial_mu, initialization_observations=int(pre.sum()),
                 neural_features=['abs_eta', 'abs_xi/(1-abs_eta)', 'log(volume/past22_mean_volume)'],
                 neural_mean=means.tolist(), neural_std=stds.tolist(),
                 normalization='pre2025 nonzero-range observations only; missing feature becomes normalized zero',
                 imputed_feature_entries=int(missing.sum()), zero_range_observations=int((~obs['observed']).sum()),
                 reference_grid=reference_grid, reference_label=model['label'],
                 global_b=D['global_b'], global_R=D['global_R'],
                 shape_bias_constant_R=D['constant_R'],
                 calibration='fixed synthetic table; no real-data fitting of b_cell or R_cell')
    return D, audit


class PrecisionFilter(nn.Module):
    def __init__(self, mode, initial_mu, seed=11):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        torch.manual_seed(seed)
        self.mode = mode
        self.register_buffer('initial_mu', torch.tensor(float(initial_mu), dtype=DTYPE))
        self.raw_mu = nn.Parameter(torch.zeros((), dtype=DTYPE))
        self.raw_phi = nn.Parameter(torch.tensor(np.log((.95 / .995) / (1 - .95 / .995)), dtype=DTYPE))
        self.raw_Q = nn.Parameter(torch.zeros((), dtype=DTYPE))
        self.raw_s_R = nn.Parameter(torch.zeros((), dtype=DTYPE))
        if mode in NEURAL_MODES:
            self.precision_net = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 1)).to(dtype=DTYPE)
            nn.init.zeros_(self.precision_net[-1].weight)
            nn.init.zeros_(self.precision_net[-1].bias)

    def parameters_bounded(self):
        mu = self.initial_mu + 3 * torch.tanh(self.raw_mu)
        phi = .995 * torch.sigmoid(self.raw_phi)
        Q = torch.exp(np.log(.02) + 2 * torch.tanh(self.raw_Q))
        s_R = 2 * torch.tanh(self.raw_s_R)
        return mu, phi, Q, s_R

    def initial_state(self, theta=None):
        mu, phi, Q, _ = self.parameters_bounded() if theta is None else theta
        return mu, Q / (1 - phi.square())

    def noise_variance(self, features, cell_R, global_R, constant_R, theta=None):
        theta = self.parameters_bounded() if theta is None else theta
        if self.mode == 'raw_constant':
            base = torch.full_like(cell_R, float(global_R))
        elif self.mode in ('bias_constant', 'neural_constant_precision'):
            base = torch.full_like(cell_R, float(constant_R))
        else:
            base = cell_R
        tilt = theta[3]
        if self.mode in NEURAL_MODES:
            tilt = tilt + 1.5 * torch.tanh(self.precision_net(features).squeeze(-1))
        return base * torch.exp(tilt)

    def step(self, state, x, observed, noise, theta=None):
        mu, phi, Q, _ = self.parameters_bounded() if theta is None else theta
        m, P = state
        prior_m = mu + phi * (m - mu)
        prior_P = phi.square() * P + Q
        if not observed:
            return (prior_m, prior_P), torch.zeros_like(prior_P)
        K = prior_P / (prior_P + noise)
        return (prior_m + K * (x - prior_m), (1 - K) * prior_P), K

    def future_daily_means(self, m, P, theta=None):
        mu, phi, Q, _ = self.parameters_bounded() if theta is None else theta
        k = torch.arange(1, 23, dtype=DTYPE, device=m.device)
        persistence = phi.pow(k)
        square_persistence = phi.pow(2 * k)
        future_m = mu + persistence * (m[..., None] - mu)
        future_P = square_persistence * P[..., None] + Q * (1 - square_persistence) / (1 - phi.square())
        return torch.exp(future_m + .5 * future_P)

    def forecast(self, m, P, theta=None):
        daily = self.future_daily_means(m, P, theta)
        sums = daily.cumsum(dim=-1)
        return torch.stack([sums[..., h - 1] / h for h in HORIZONS], dim=-1)


def tensors(D):
    return {name: torch.tensor(D[name], dtype=DTYPE) for name in ('log_proxy', 'even', 'cell_b', 'cell_R', 'y', 'ewma')}


def observation_values(D, T, mode):
    return T['log_proxy'] - (D['global_b'] if mode == 'raw_constant' else T['cell_b'])


def qlike(v, y, known_scale=None):
    loss = torch.log(v.clamp_min(EPS)) + y / v.clamp_min(EPS)
    if known_scale is not None:
        loss = loss - torch.log(known_scale.clamp_min(EPS))[:, None]
    return loss


def filter_prefix(net, D, T, stop):
    """Fixed parameters, chronological updates, no parameter update of any kind."""
    theta = net.parameters_bounded()
    noise = net.noise_variance(T['even'][:stop], T['cell_R'][:stop], D['global_R'], D['constant_R'], theta)
    x = observation_values(D, T, net.mode)
    state = net.initial_state(theta)
    means, variances, gains = [], [], []
    for t in range(stop):
        state, gain = net.step(state, x[t], bool(D['observed'][t]), noise[t], theta)
        means.append(state[0]); variances.append(state[1]); gains.append(gain)
    return torch.stack(means), torch.stack(variances), torch.stack(gains), noise


def evaluate(net, D, T, which):
    net.eval()
    query = D['origin'][which]
    with torch.no_grad():
        m, P, K, noise = filter_prefix(net, D, T, int(query.max()) + 1)
        pred = net.forecast(m[query], P[query])
        score = float(qlike(pred, T['y'][which], T['ewma'][which]).mean())
    assert torch.isfinite(pred).all() and torch.all(pred > 0)
    return pred.numpy(), score, dict(m=m[query].numpy(), P=P[query].numpy(),
                                     K=K[query].numpy(), R=noise[query].numpy())


def train_one(D, mode, seed, epochs, chunk=128):
    T = tensors(D)
    net = PrecisionFilter(mode, D['initial_mu'], seed)
    opt = torch.optim.Adam(net.parameters(), lr=.002)
    _, best, _ = evaluate(net, D, T, D['va'])
    state_best = copy.deepcopy(net.state_dict())
    best_epoch, bad = 0, 0
    history = [dict(epoch=0, validation_QLIKE_known_scale=best, training_QLIKE_known_scale=None)]
    lookup = np.full(D['n'], -1, dtype=int)
    lookup[D['origin'][D['tr']]] = D['tr']
    assert (D['origin'][D['tr']] + 22 < D['cut']).all()
    for epoch in range(1, epochs + 1):
        net.train()
        state = net.initial_state()
        losses, counts = [], []
        # Never filter a validation observation inside the training routine.
        for lo in range(0, D['cut'], chunk):
            hi = min(lo + chunk, D['cut'])
            theta = net.parameters_bounded()
            noise = net.noise_variance(T['even'][lo:hi], T['cell_R'][lo:hi], D['global_R'], D['constant_R'], theta)
            x = observation_values(D, T, mode)
            selected_m, selected_P, selected_ids = [], [], []
            for t in range(lo, hi):
                state, _ = net.step(state, x[t], bool(D['observed'][t]), noise[t - lo], theta)
                if lookup[t] >= 0:
                    selected_m.append(state[0]); selected_P.append(state[1]); selected_ids.append(lookup[t])
            if selected_ids:
                ids = np.asarray(selected_ids)
                assert (D['origin'][ids] < D['cut']).all()
                assert (D['origin'][ids] + 22 < D['cut']).all()
                pred = net.forecast(torch.stack(selected_m), torch.stack(selected_P), theta)
                loss = qlike(pred, T['y'][ids], T['ewma'][ids]).mean()
                assert torch.isfinite(loss)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 2)
                opt.step()
                losses.append(float(loss.detach())); counts.append(len(ids))
            state = tuple(value.detach() for value in state)
        _, score, _ = evaluate(net, D, T, D['va'])
        history.append(dict(epoch=epoch, validation_QLIKE_known_scale=score,
                            training_QLIKE_known_scale=float(np.average(losses, weights=counts))))
        if score < best - 1e-5:
            best, best_epoch, bad = score, epoch, 0
            state_best = copy.deepcopy(net.state_dict())
        else:
            bad += 1
        if bad >= 15 and epoch >= 20:
            break
    net.load_state_dict(state_best)
    pred, score, aux = evaluate(net, D, T, D['va'])
    with torch.no_grad():
        mu, phi, Q, s_R = [float(value) for value in net.parameters_bounded()]
    return pred, aux, dict(seed=seed, mode=mode, best_epoch=best_epoch,
        epochs_run=len(history) - 1, best_validation_QLIKE_known_scale=best,
        returned_validation_QLIKE_known_scale=score, parameters=sum(p.numel() for p in net.parameters()),
        dynamics=dict(mu=mu, phi=phi, Q=Q, s_R=s_R), history=history,
        training_observation_stop_exclusive=int(D['cut']),
        max_training_target_end=int((D['origin'][D['tr']] + 22).max()),
        validation='from first observation with frozen parameters, no_grad; current observed prices update state only')


def direct_baselines(D):
    tr, va = D['tr'], D['va']
    X = D['baseline_x']
    means, stds = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-5
    design = np.column_stack([np.ones(len(X)), (X - means) / stds])
    har = np.empty((len(va), 3)); hgb = np.empty_like(har)
    for j in range(3):
        log_target = np.log(np.maximum(D['y'][tr, j], EPS))
        coef = np.linalg.lstsq(design[tr], log_target, rcond=None)[0]
        smear = float(np.exp(log_target - design[tr] @ coef).mean())
        har[:, j] = np.exp(design[va] @ coef) * smear
        scale = float(D['y'][tr, j].mean())
        assert scale > 0
        model = HistGradientBoostingRegressor(loss='poisson', max_iter=150, max_leaf_nodes=7,
            learning_rate=.04, l2_regularization=1, min_samples_leaf=20, early_stopping=False, random_state=11)
        model.fit(X[tr], D['y'][tr, j] / scale)
        hgb[:, j] = model.predict(X[va]) * scale
    return dict(EWMA22=np.repeat(D['ewma'][va, None], 3, axis=1), HAR_geometry=har, HGB_geometry=hgb)


def synthetic_checks():
    """No file reads; all calibration and observations below are synthetic."""
    schema = dict(edges_a=[0, .2, .4, .6, .8, 1], edges_z=[0, .2, .4, .6, .8, 1])
    np.testing.assert_array_equal(reference_cells(np.array([0, 1, .4]), np.array([0, 1, .8]), schema), [0, 24, 14])
    n = 48
    rng = np.random.default_rng(81)
    D = dict(n=n, initial_mu=-8., log_proxy=-8 + rng.normal(0, .3, n),
             even=rng.normal(size=(n, 3)), cell_b=np.linspace(-.5, .2, n),
             cell_R=np.linspace(.2, .35, n), observed=np.ones(n, dtype=bool),
             global_b=-.23, global_R=.35, constant_R=.27,
             y=np.full((3, 3), .0005), ewma=np.full(3, .0005))
    D['observed'][12] = False
    T = tensors(D)
    for mode in MODES:
        net = PrecisionFilter(mode, -8.)
        with torch.no_grad():
            m, P, K, noise = filter_prefix(net, D, T, n)
            daily = net.future_daily_means(m, P)
            pred = net.forecast(m, P)
            assert torch.isfinite(pred).all() and torch.all(pred > 0) and torch.all(P > 0)
            assert torch.all(noise > 0) and torch.all((K >= 0) & (K < 1))
            assert torch.all(torch.diff(daily.cumsum(dim=1), dim=1) > 0)
            assert K[12] == 0
            theta = net.parameters_bounded()
            mu, phi, Q, _ = theta
            expected_m = mu + phi * (m[11] - mu)
            expected_P = phi.square() * P[11] + Q
            torch.testing.assert_close(m[12], expected_m, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(P[12], expected_P, rtol=1e-12, atol=1e-12)
            one = net.future_daily_means(torch.tensor(-8., dtype=DTYPE), torch.tensor(.4, dtype=DTYPE))[0]
            first_m = mu + phi * (-8 - mu)
            first_P = phi.square() * .4 + Q
            torch.testing.assert_close(one, torch.exp(first_m + .5 * first_P), rtol=1e-12, atol=1e-12)
            assert float(one) > float(torch.exp(first_m))
        changed = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in D.items()}
        changed['log_proxy'][25:] += 50
        changed['even'][25:] *= -30
        changed['cell_b'][25:] -= 20
        with torch.no_grad():
            cm, cP, _, _ = filter_prefix(net, changed, tensors(changed), n)
        torch.testing.assert_close(m[:25], cm[:25], rtol=0, atol=0)
        torch.testing.assert_close(P[:25], cP[:25], rtol=0, atol=0)
    fixed = PrecisionFilter('fixed_precision', -8.)
    neural = PrecisionFilter('neural_precision', -8.)
    constant = PrecisionFilter('bias_constant', -8.)
    neural_constant = PrecisionFilter('neural_constant_precision', -8.)
    with torch.no_grad():
        for left, right in ((fixed, neural), (constant, neural_constant)):
            lm, lP, _, lR = filter_prefix(left, D, T, n)
            rm, rP, _, rR = filter_prefix(right, D, T, n)
            torch.testing.assert_close(lR, rR, rtol=0, atol=0)
            torch.testing.assert_close(lm, rm, rtol=0, atol=0)
            torch.testing.assert_close(lP, rP, rtol=0, atol=0)
    # Backpropagate a synthetic sequence through covariance, gain, and lognormal forecasts.
    T['log_proxy'].requires_grad_(True)
    m, P, _, _ = filter_prefix(neural, D, T, 18)
    synthetic_loss = qlike(neural.forecast(m[-3:], P[-3:]), torch.full((3, 3), .0005, dtype=DTYPE)).mean()
    synthetic_loss.backward()
    assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in neural.parameters())
    assert torch.equal(T['log_proxy'].grad[18:], torch.zeros(n - 18, dtype=DTYPE))
    tiny = dict(n=80, cut=50, initial_mu=-8., log_proxy=-8 + rng.normal(0, .3, 80),
                even=rng.normal(size=(80, 3)), cell_b=np.linspace(-.5, .2, 80),
                cell_R=np.linspace(.2, .35, 80), observed=np.ones(80, dtype=bool),
                global_b=-.23, global_R=.35, constant_R=.27,
                origin=np.array([5, 10, 15, 20, 25, 55]), tr=np.arange(5), va=np.array([5]),
                y=np.full((6, 3), .0005), ewma=np.full(6, .0005))
    tiny['observed'][12] = False
    tiny_pred, _, tiny_fit = train_one(tiny, 'neural_precision', 11, epochs=1, chunk=16)
    assert tiny_pred.shape == (1, 3) and np.isfinite(tiny_pred).all()
    assert tiny_fit['training_observation_stop_exclusive'] == 50
    assert tiny_fit['max_training_target_end'] == 47
    assert tiny_fit['epochs_run'] == 1
    # Geometry is reflection-even, and a genuinely zero range never becomes log(EPS) data.
    frame = pd.DataFrame(dict(date=pd.date_range('2024-01-01', periods=24), open=np.ones(24),
        close=np.ones(24), high=np.full(24, 1.02), low=np.full(24, .98), volume=np.full(24, 100.)))
    frame.loc[10, ['high', 'low']] = 1.
    obs = observation_arrays(frame)
    assert not obs['observed'][10] and obs['log_proxy'][10] == 0 and obs['zero_range'][10]
    assert obs['eta'][10] == 0 and obs['xi'][10] == 0
    a, xi = .3, -.2
    z = abs(xi) / (1 - a)
    assert z == abs(-xi) / (1 - abs(-a))
    return dict(market_data_opened=False, calibration_file_opened=False,
        reference_cell_mapping=True, positive_variance_and_state_covariance=True,
        future_mutation_prefix_invariant=True, zero_range_is_skipped_not_logEPS=True,
        lognormal_mean_includes_half_variance=True, zero_neural_nests_fixed_precision=True,
        zero_neural_nests_shape_bias_constant=True, differentiable_filter_gradients_finite=True,
        zero_training_gradient_to_future_observations=True, synthetic_tbptt_and_maturity_checked=True,
        reflected_geometry_is_even=True, positive_shared_daily_curve=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instrument', choices=list(SOURCES), default='KRBN')
    parser.add_argument('--reference-grid', type=int, choices=[512, 2048], default=512)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--chunk', type=int, default=128)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.check_only:
        print(json.dumps(synthetic_checks(), indent=2), flush=True)
        return
    if args.epochs < 0 or args.chunk < 1 or not args.seeds:
        raise ValueError('Invalid epochs/chunk/seeds')
    tag = args.tag or f'bridge_precision_round16_{args.instrument}_grid{args.reference_grid}_development'
    if not re.fullmatch(r'bridge_precision_round16_[A-Za-z0-9_-]+', tag):
        raise ValueError('Use a new bridge_precision_round16_ tag')
    outputs = dict(manifest=HERE / f'{tag}_manifest.json', results=HERE / f'{tag}_results.json',
                   predictions=HERE / f'{tag}_validation_predictions.npz')
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f'Existing output is protected; select a new --tag: {path}')
    sources = dict(script=Path(__file__).resolve(), range_clock_script=Path(R14.__file__).resolve(),
                   brownian_calibration=REFERENCE, csv=SOURCES[args.instrument])
    hashes = {name: dict(path=str(path), sha256=sha256(path)) for name, path in sources.items()}
    started = time.time()
    D, audit = prepare(args.instrument, args.reference_grid)
    config = dict(config=vars(args), status='DEVELOPMENT ONLY; 2025 validation selection; 2026 unused',
        audit=audit, hashes=hashes, modes=list(MODES), horizons=list(HORIZONS), checks=synthetic_checks(),
        dynamics='same bounded AR1 form, boundaries, initial values, optimizer and training schedule; separately fitted parameters per mode',
        bounds=dict(mu='pre2025 initial_mu + 3*tanh(raw)', phi='.995*sigmoid(raw)',
                    Q='0.02*exp(2*tanh(raw))', s_R='2*tanh(raw)', neural_log_R_tilt='1.5*tanh(g_even)'),
        initialization='common pre2025 mean(logP-global_b), phi=.95, Q=.02, s_R=0, stationary P; neural output head=0',
        modes_definition=dict(raw_constant='global b and global R', bias_constant='cell b and synthetic shape-corrected residual-best constant R',
            fixed_precision='cell b and cell R', neural_precision='cell b; cell R*exp(s_R+1.5*tanh(g_even))',
            neural_constant_precision='cell b; residual-best constant R*exp(s_R+1.5*tanh(g_even))'),
        train='chronological TBPTT; all observations strictly before 2025; detach state at chunk boundaries; supervised origins require t+22<cut',
        selection='epoch0 included; mean same-direction QLIKE across h1/5/22 common 2025 origins; known EWMA22 log constant subtracted',
        validation='fresh chronological filtering from first observation, no_grad, fixed parameters; validation prices update state only',
        forecasting='propagate AR1 m/P for k1..22; exp(m_k+P_k/2); all h average the same positive daily curve',
        baselines='EWMA22, fixed-spec Poisson HGB and log-linear HAR extension; common mature origins and same geometry inputs',
        information='raw Parkinson proxy only; real quadratic variation is not identified; calibration is a diffusion reference, not verified true measurement noise',
        seed_note='non-neural modes fit only the first seed; later seed slots explicitly reuse that deterministic fit and are not independent repeats; both neural modes fit every seed; seeds never constitute independent datasets',
        outputs={key: str(value) for key, value in outputs.items()}, fits={}, scores={})
    outputs['manifest'].write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    pred = direct_baselines(D)
    arrays = dict(date=D['date'][D['origin'][D['va']]], origin=D['origin'][D['va']],
        label_end=D['date'][D['origin'][D['va']] + 22], target=D['y'][D['va']], horizons=np.array(HORIZONS),
        observed_at_origin=D['observed'][D['origin'][D['va']]], reference_cell=D['cell'][D['origin'][D['va']]])
    for mode in MODES:
        seed_predictions, fits = [], []
        deterministic_cache = None
        for seed in args.seeds:
            if mode not in NEURAL_MODES and deterministic_cache is not None:
                cached_value, cached_aux, cached_fit, fitted_seed = deterministic_cache
                value = cached_value.copy()
                aux = {name: state.copy() for name, state in cached_aux.items()}
                fit = copy.deepcopy(cached_fit)
                fit.update(seed=seed, fit_executed=False,
                           deterministic_reuse_from_seed=fitted_seed,
                           effective_fit_seed=fitted_seed, independent_repeat=False)
            else:
                value, aux, fit = train_one(D, mode, seed, args.epochs, args.chunk)
                fit.update(fit_executed=True, deterministic_reuse_from_seed=None,
                           effective_fit_seed=seed, independent_repeat=False)
                if mode not in NEURAL_MODES:
                    deterministic_cache = (value.copy(),
                        {name: state.copy() for name, state in aux.items()}, copy.deepcopy(fit), seed)
            seed_predictions.append(value); fits.append(fit)
            pred[f'{mode}_seed{seed}'] = value
            arrays[f'{mode}_seed{seed}'] = value
            for name, state in aux.items():
                arrays[f'{mode}_seed{seed}_{name}'] = state
            print(args.instrument, mode, seed, 'epoch', fit['best_epoch'],
                  'validation', round(fit['best_validation_QLIKE_known_scale'], 7), flush=True)
        pred[mode] = (seed_predictions[0].copy() if mode not in NEURAL_MODES
                      else np.mean(seed_predictions, axis=0, dtype=np.float64))
        config['fits'][mode] = fits
    y = D['y'][D['va']]
    for j, h in enumerate(HORIZONS):
        baseline = np.log(pred['EWMA22'][:, j]) + y[:, j] / pred['EWMA22'][:, j]
        config['scores'][str(h)] = {}
        for name, value in pred.items():
            assert value.shape == y.shape and np.isfinite(value).all() and (value > 0).all()
            score = np.log(value[:, j]) + y[:, j] / value[:, j]
            config['scores'][str(h)][name] = dict(n=len(y), mean_QLIKE=float(score.mean()),
                loss_difference_vs_EWMA22=float((score - baseline).mean()), MSE=float(np.mean((value[:, j] - y[:, j]) ** 2)))
    arrays.update(pred)
    for name, path in sources.items():
        assert sha256(path) == hashes[name]['sha256'], f'Source changed during run: {name}'
    config['elapsed_seconds'] = time.time() - started
    outputs['results'].write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(outputs['predictions'], **arrays)
    print('DEVELOPMENT FINISHED; 2026 UNUSED', round(config['elapsed_seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
