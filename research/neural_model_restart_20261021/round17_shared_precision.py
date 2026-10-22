"""Round 17: prespecified two-asset, centered geometry corrections.

Code-only delivery: --check-only uses synthetic arrays and opens no data files.
Real execution uses the frozen R16 preparation and simulation-train proxy frame.
2026 is removed before all market observations, features, and targets.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
import json
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import bridge_precision_proxy_round16 as PROXY

B = PROXY.B
HERE = Path(__file__).resolve().parent
ASSETS = ('KRBN', 'KCCA')
MODES = ('fixed', 'local_precision', 'shared_precision',
         'shared_bounded_linear_precision', 'shared_observation_mean')
DTYPE = torch.float64
LAMBDA = .05
LR = .01


def prepare_pair():
    reference = json.loads(PROXY.REFERENCE_RESULTS.read_text(encoding='utf-8'))
    c = float(reference['resolution_sensitivity']['mean_P_by_train_resolution']['512'])
    datasets, audits, raw_features = {}, {}, {}
    for asset in ASSETS:
        original, audit = B.prepare(asset, 512)
        D = PROXY.proxy_frame(original, c)
        # Reverse only fixed pre2025 normalizers; geometry gains no new inputs.
        mean = np.asarray(audit['neural_mean'], dtype=float)
        sd = np.asarray(audit['neural_std'], dtype=float)
        recovered = D['even'] * sd + mean
        assert (recovered[:, :2] >= -1e-10).all() and (recovered[:, :2] <= 1 + 1e-8).all()
        # Read the SAME volume column to retain its actual missingness, not the
        # prior per-asset mean imputation. Filter before the rolling calculation.
        raw = pd.read_csv(B.SOURCES[asset], parse_dates=['date'])
        d = raw.loc[raw.date < pd.Timestamp(B.RESERVED)].copy().reset_index(drop=True)
        np.testing.assert_array_equal(d.date.to_numpy(), D['date'])
        with np.errstate(divide='ignore', invalid='ignore'):
            volume_ratio = np.log(d.volume.to_numpy(float) /
                d.volume.rolling(22).mean().to_numpy(float))
        F = recovered.copy()
        F[:, :2] = np.clip(F[:, :2], 0, 1)
        F[D['zero_range'], :2] = 0
        F[:, 2] = volume_ratio
        stats_mask = (np.arange(D['n']) < D['cut']) & D['observed'] & np.isfinite(F[:, 2])
        assert stats_mask.any()
        audit.update(proxy_frame='log(c512*sigma^2)', simulation_train_mean_P=c,
                     initial_mu=D['initial_mu'], global_b=D['global_b'],
                     reference_bank='pre2025 observed positive range, row>=21 only')
        datasets[asset], audits[asset], raw_features[asset] = D, audit, F
    normalizer = install_features(datasets, raw_features)
    for asset, D in datasets.items():
        bank = D['bank_index']
        audits[asset].update(
            bank_count=len(bank), bank_first=str(pd.Timestamp(D['date'][bank[0]]).date()),
            bank_last=str(pd.Timestamp(D['date'][bank[-1]]).date()),
            bank_rows=bank.tolist(),
            volume_missing_imputed=int((~np.isfinite(raw_features[asset][:, 2])).sum()),
            geometry_mapping='2*a-1,2*z-1, no fitted geometry location/scale',
            model_features='same a,z,volume logratio as round16; no new feature or target')
    return datasets, dict(assets=audits, proxy_c512=c, normalization=normalizer)


def install_features(datasets, raw_features):
    """Equal-asset mixture variance includes the between-asset mean component."""
    moments = {}
    for asset in ASSETS:
        D, F = datasets[asset], raw_features[asset]
        mask = (np.arange(D['n']) < D['cut']) & D['observed'] & np.isfinite(F[:, 2])
        assert mask.any()
        values = F[mask, 2]
        moments[asset] = dict(n=len(values), mean=float(values.mean()), variance=float(values.var()))
    pooled_mean = sum(moments[a]['mean'] for a in ASSETS) / len(ASSETS)
    pooled_var = sum(moments[a]['variance'] + (moments[a]['mean'] - pooled_mean) ** 2
                     for a in ASSETS) / len(ASSETS)
    pooled_sd = max(float(np.sqrt(pooled_var)), 1e-5)
    for asset in ASSETS:
        D, F = datasets[asset], raw_features[asset]
        normalized = F.copy()
        normalized[:, :2] = 2 * F[:, :2] - 1
        volume = np.where(np.isfinite(F[:, 2]), F[:, 2], pooled_mean)
        normalized[:, 2] = (volume - pooled_mean) / pooled_sd
        assert np.isfinite(normalized).all()
        D['F17'] = normalized
        D['bank_index'] = np.flatnonzero((np.arange(D['n']) < D['cut']) &
                                          D['observed'] & (np.arange(D['n']) >= 21))
        assert len(D['bank_index']) > 0
        assert (D['bank_index'] < D['cut']).all()
        if 'date' in D:
            assert (D['date'][D['bank_index']] < B.CUT_DATE).all()
        if 'baseline_x' in D:
            assert D['baseline_x'].shape[1] == 15
            D['baseline_x'] = np.column_stack([D['baseline_x'][:, :12], normalized[D['origin']]])
    return dict(asset_weight=.5, asset_moments=moments, pooled_mean=pooled_mean,
                pooled_variance=pooled_var, pooled_std=pooled_sd,
                rule='equal-asset mixture moments, pre2025 observed rows only; missing volume becomes training pooled mean',
                geometry='fixed 2*a-1 and 2*z-1', missing_volume_normalized_value=0.)


def small_net():
    net = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 1)).to(dtype=DTYPE)
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class JointPrecision(nn.Module):
    def __init__(self, datasets, mode, seed):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.filters = nn.ModuleDict({asset: B.PrecisionFilter('fixed_precision', datasets[asset]['initial_mu'], seed)
                                      for asset in ASSETS})
        for asset in ASSETS:
            D = datasets[asset]
            self.register_buffer(f'bank_{asset}', torch.tensor(D['F17'][D['bank_index']].copy(), dtype=DTYPE))
        torch.manual_seed(seed)
        if mode == 'local_precision':
            self.local = nn.ModuleDict({asset: small_net() for asset in ASSETS})
        elif mode in ('shared_precision', 'shared_observation_mean'):
            self.shared = small_net()
        elif mode == 'shared_bounded_linear_precision':
            self.shared = nn.Linear(3, 1, bias=False).to(dtype=DTYPE)
            nn.init.zeros_(self.shared.weight)

    def raw_tilt(self, asset, features):
        if self.mode == 'fixed':
            return features.new_zeros(len(features))
        net = self.local[asset] if self.mode == 'local_precision' else self.shared
        return .5 * torch.tanh(net(features).squeeze(-1))

    def reference_delta(self, asset):
        raw = self.raw_tilt(asset, getattr(self, f'bank_{asset}'))
        return raw - raw.mean()

    def delta(self, asset, features):
        # Recompute this graph for EVERY chunk, including the reference mean.
        bank_raw = self.raw_tilt(asset, getattr(self, f'bank_{asset}'))
        return self.raw_tilt(asset, features) - bank_raw.mean()

    def penalty(self):
        return LAMBDA * sum(self.reference_delta(asset).square().mean() for asset in ASSETS) / len(ASSETS)

    def inputs(self, asset, T, lo, hi, theta):
        delta = self.delta(asset, T['F17'][lo:hi])
        x = T['log_proxy'][lo:hi] - T['cell_b'][lo:hi]
        if self.mode == 'shared_observation_mean':
            x = x + delta
            noise = T['cell_R'][lo:hi] * torch.exp(theta[3])
        else:
            noise = T['cell_R'][lo:hi] * torch.exp(theta[3] + delta)
        return x, noise, delta


def make_tensors(datasets):
    out = {}
    for asset in ASSETS:
        out[asset] = B.tensors(datasets[asset])
        out[asset]['F17'] = torch.tensor(datasets[asset]['F17'], dtype=DTYPE)
    return out


def filter_asset(model, asset, D, T, stop):
    process = model.filters[asset]
    theta = process.parameters_bounded()
    x, noise, delta = model.inputs(asset, T, 0, stop, theta)
    state = process.initial_state(theta)
    m, P, K = [], [], []
    for t in range(stop):
        state, gain = process.step(state, x[t], bool(D['observed'][t]), noise[t], theta)
        m.append(state[0]); P.append(state[1]); K.append(gain)
    return torch.stack(m), torch.stack(P), torch.stack(K), noise, delta


def evaluate_joint(model, datasets, T):
    model.eval()
    predictions, auxiliary, scores = {}, {}, {}
    with torch.no_grad():
        for asset in ASSETS:
            D = datasets[asset]
            query = D['origin'][D['va']]
            m, P, K, noise, delta = filter_asset(model, asset, D, T[asset], int(query.max()) + 1)
            pred = model.filters[asset].forecast(m[query], P[query])
            assert torch.isfinite(pred).all() and torch.all(pred > 0)
            scores[asset] = float(B.qlike(pred, T[asset]['y'][D['va']], T[asset]['ewma'][D['va']]).mean())
            predictions[asset] = pred.numpy()
            auxiliary[asset] = dict(m=m[query].numpy(), P=P[query].numpy(), K=K[query].numpy(),
                                     R=noise[query].numpy(), delta=delta[query].numpy())
    return predictions, sum(scores.values()) / len(ASSETS), scores, auxiliary


def train_epoch(model, datasets, T, optimizer, chunk):
    """Fixed parameters until one optimizer step after both assets and penalty."""
    model.train()
    optimizer.zero_grad()
    asset_loss, weights, terminals = {}, {}, {}
    for asset in ASSETS:
        D, tensor, process = datasets[asset], T[asset], model.filters[asset]
        lookup = np.full(D['n'], -1, dtype=int)
        lookup[D['origin'][D['tr']]] = D['tr']
        ntrain = len(D['tr'])
        assert ntrain > 0 and (D['origin'][D['tr']] + 22 < D['cut']).all()
        state = None
        running, weight_sum = 0., 0.
        for lo in range(0, D['cut'], chunk):
            hi = min(lo + chunk, D['cut'])
            # Fresh theta and fresh centering graph prevent backward graph reuse.
            theta = process.parameters_bounded()
            if state is None:
                state = process.initial_state(theta)
            x, noise, _ = model.inputs(asset, tensor, lo, hi, theta)
            selected_m, selected_P, selected = [], [], []
            for t in range(lo, hi):
                state, _ = process.step(state, x[t - lo], bool(D['observed'][t]), noise[t - lo], theta)
                if lookup[t] >= 0:
                    selected_m.append(state[0]); selected_P.append(state[1]); selected.append(lookup[t])
            if selected:
                ids = np.asarray(selected)
                assert (D['origin'][ids] + 22 < D['cut']).all()
                pred = process.forecast(torch.stack(selected_m), torch.stack(selected_P), theta)
                loss = B.qlike(pred, tensor['y'][ids], tensor['ewma'][ids]).mean()
                weight = .5 * len(ids) / ntrain
                assert torch.isfinite(loss)
                (weight * loss).backward()
                running += len(ids) / ntrain * float(loss.detach())
                weight_sum += weight
            state = tuple(value.detach() for value in state)
        assert abs(weight_sum - .5) < 1e-12
        asset_loss[asset], weights[asset] = running, weight_sum
        terminals[asset] = [float(value) for value in state]
    # This is a fresh graph too. No data gradient or reference graph is cached.
    penalty = model.penalty()
    assert torch.isfinite(penalty)
    if penalty.requires_grad:
        penalty.backward()
    gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 2))
    assert np.isfinite(gradient_norm)
    optimizer.step()
    return dict(training_asset_QLIKE=asset_loss, training_joint_QLIKE=sum(asset_loss.values()) / 2,
                penalty=float(penalty.detach()), total_training_objective=sum(asset_loss.values()) / 2 + float(penalty.detach()),
                asset_gradient_weights=weights, pre_step_terminal_states=terminals,
                gradient_norm_before_clip=gradient_norm, optimizer_steps=1)


def distribution(values):
    values = np.asarray(values, dtype=float)
    assert len(values) and np.isfinite(values).all()
    return dict(n=len(values), mean=float(values.mean()), std=float(values.std()),
                min=float(values.min()), max=float(values.max()),
                q05=float(np.quantile(values, .05)), median=float(np.median(values)), q95=float(np.quantile(values, .95)))


def summarize_model(model, datasets, T):
    result = {}
    with torch.no_grad():
        for asset in ASSETS:
            D = datasets[asset]
            train_delta = model.reference_delta(asset).numpy()
            val_index = np.flatnonzero((np.arange(D['n']) >= D['cut']) & D['observed'])
            val_delta = model.delta(asset, T[asset]['F17'][val_index]).numpy()
            mu, phi, Q, s_R = [float(x) for x in model.filters[asset].parameters_bounded()]
            result[asset] = dict(dynamics=dict(mu=mu, phi=phi, Q=Q, s_R=s_R),
                delta_train_reference=distribution(train_delta), delta_validation_observed=distribution(val_delta),
                validation_distribution_only='never used for centering or normalization',
                zero_range_observations=int((~D['observed']).sum()))
    return result


def train_one(datasets, mode, seed, epochs, chunk=128):
    T = make_tensors(datasets)
    model = JointPrecision(datasets, mode, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    _, best, per_asset, _ = evaluate_joint(model, datasets, T)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch, bad = 0, 0
    history = [dict(epoch=0, validation_joint_QLIKE=best, validation_asset_QLIKE=per_asset)]
    for epoch in range(1, epochs + 1):
        training = train_epoch(model, datasets, T, optimizer, chunk)
        _, score, scores, _ = evaluate_joint(model, datasets, T)
        history.append(dict(epoch=epoch, validation_joint_QLIKE=score, validation_asset_QLIKE=scores, **training))
        if score < best - 1e-5:
            best, best_epoch, bad = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if bad >= 15 and epoch >= 20:
            break
    model.load_state_dict(best_state)
    pred, score, per_asset, aux = evaluate_joint(model, datasets, T)
    return pred, aux, dict(seed=seed, mode=mode, best_joint_epoch=best_epoch, epochs_run=len(history) - 1,
        best_joint_validation_QLIKE=best, returned_joint_validation_QLIKE=score,
        selected_epoch_asset_scores=per_asset, parameters=sum(p.numel() for p in model.parameters()),
        parameters_by_asset={a: sum(p.numel() for p in model.filters[a].parameters()) for a in ASSETS},
        state_and_delta=summarize_model(model, datasets, T), history=history,
        selection='single joint epoch for both assets, including local_precision; never per-asset epoch selection')


def synthetic_pair():
    rng = np.random.default_rng(1701)
    datasets, raw_features = {}, {}
    for j, asset in enumerate(ASSETS):
        n, cut = 94 + j * 10, 60 + j * 8
        origins = np.array([5, 11, 17, 23, 29, 35] + ([41] if j else []) + [cut + 2])
        tr = np.flatnonzero(origins + 22 < cut)
        va = np.flatnonzero(origins >= cut)
        dates = np.empty(n, dtype='datetime64[D]')
        dates[:cut] = np.datetime64('2024-01-01') + np.arange(cut)
        dates[cut:] = np.datetime64('2025-01-01') + np.arange(n - cut)
        observed = np.ones(n, dtype=bool); observed[12] = False
        D = dict(n=n, cut=cut, date=dates, initial_mu=-8. + j * .1,
            log_proxy=-8 + rng.normal(0, .3, n), cell_b=np.linspace(-.5, .2, n),
            cell_R=np.linspace(.2, .35, n), observed=observed,
            global_b=-.23, global_R=.35, constant_R=.27,
            even=np.zeros((n, 3)), origin=origins, tr=tr, va=va,
            y=np.full((len(origins), 3), .0005), ewma=np.full(len(origins), .0005))
        raw = np.column_stack([rng.uniform(.05, .9, n), rng.uniform(.05, .9, n), rng.normal(j, 1, n)])
        raw[:21, 2] = np.nan
        datasets[asset], raw_features[asset] = D, raw
    install_features(datasets, raw_features)
    return datasets, raw_features


def synthetic_checks():
    datasets, raw = synthetic_pair()
    T = make_tensors(datasets)
    # Both calibration and all market inputs remain synthetic in this function.
    fixed = JointPrecision(datasets, 'fixed', 11)
    with torch.no_grad():
        fixed_pred, _, _, _ = evaluate_joint(fixed, datasets, T)
        for mode in MODES:
            model = JointPrecision(datasets, mode, 11)
            values, _, _, _ = evaluate_joint(model, datasets, T)
            for asset in ASSETS:
                np.testing.assert_array_equal(values[asset], fixed_pred[asset])
                assert torch.equal(model.reference_delta(asset), torch.zeros_like(model.reference_delta(asset)))
    # A learned shared linear tilt centers on TRAIN banks, not the query batch.
    model = JointPrecision(datasets, 'shared_bounded_linear_precision', 11)
    with torch.no_grad():
        model.shared.weight.fill_(.2)
        for asset in ASSETS:
            delta = model.reference_delta(asset)
            assert abs(float(delta.mean())) < 1e-12 and float(delta.abs().max()) <= 1
            shifted_query = getattr(model, f'bank_{asset}') + 5
            assert abs(float(model.delta(asset, shifted_query).mean())) > 1e-5
        expected = .05 * sum(model.reference_delta(a).square().mean() for a in ASSETS) / 2
        torch.testing.assert_close(model.penalty(), expected, rtol=0, atol=0)
    # Mutate validation covariates only: neither normalizers nor banks may change.
    changed_D = {a: copy.deepcopy(datasets[a]) for a in ASSETS}
    changed_raw = {a: raw[a].copy() for a in ASSETS}
    for asset in ASSETS:
        changed_raw[asset][datasets[asset]['cut']:, :] += 50
    norm_changed = install_features(changed_D, changed_raw)
    original_D = {a: copy.deepcopy(datasets[a]) for a in ASSETS}
    norm_original = install_features(original_D, raw)
    assert norm_changed == norm_original
    for asset in ASSETS:
        np.testing.assert_array_equal(changed_D[asset]['F17'][:datasets[asset]['cut']], datasets[asset]['F17'][:datasets[asset]['cut']])
        np.testing.assert_array_equal(datasets[asset]['F17'][:21, 2], np.zeros(21))
    # Future observation mutations preserve earlier filtered states with fixed banks.
    for asset in ASSETS:
        with torch.no_grad():
            before = filter_asset(model, asset, datasets[asset], T[asset], 48)
        changed_t = {k: v.clone() for k, v in T[asset].items()}
        changed_t['log_proxy'][40:] += 30
        changed_t['F17'][40:] += 3
        with torch.no_grad():
            after = filter_asset(model, asset, datasets[asset], changed_t, 48)
        for left, right in zip(before[:2], after[:2]):
            torch.testing.assert_close(left[:40], right[:40], rtol=0, atol=0)
    # One step ONLY per epoch; parameter-constant forward states equal fresh filtering.
    class CountingAdam(torch.optim.Adam):
        calls = 0
        def step(self, *args, **kwargs):
            self.calls += 1
            return super().step(*args, **kwargs)
    for mode in MODES:
        model = JointPrecision(datasets, mode, 11)
        initial_model = copy.deepcopy(model)
        expected_states = {}
        with torch.no_grad():
            for asset in ASSETS:
                m, P, _, _, _ = filter_asset(initial_model, asset, datasets[asset], T[asset], datasets[asset]['cut'])
                expected_states[asset] = np.array([float(m[-1]), float(P[-1])])
        optimizer = CountingAdam(model.parameters(), lr=.01)
        info = train_epoch(model, datasets, T, optimizer, chunk=16)
        assert optimizer.calls == 1
        for asset in ASSETS:
            np.testing.assert_allclose(info['pre_step_terminal_states'][asset], expected_states[asset], rtol=1e-12, atol=1e-12)
            assert abs(info['asset_gradient_weights'][asset] - .5) < 1e-12
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.filters[asset].parameters())
        pred, joint, scores, aux = evaluate_joint(model, datasets, T)
        assert abs(joint - sum(scores.values()) / 2) < 1e-12
        assert all(np.isfinite(value).all() and (value > 0).all() for value in pred.values())
        if mode == 'shared_observation_mean':
            for asset in ASSETS:
                with torch.no_grad():
                    theta = model.filters[asset].parameters_bounded()
                    x, noise, delta = model.inputs(asset, T[asset], 0, 30, theta)
                    torch.testing.assert_close(noise, T[asset]['cell_R'][:30] * torch.exp(theta[3]))
                    torch.testing.assert_close(x, T[asset]['log_proxy'][:30] - T[asset]['cell_b'][:30] + delta)
    return dict(market_files_opened=False, reference_files_opened=False,
        all_initial_models_exactly_nest_fixed=True, training_bank_centering_zero_mean=True,
        no_validation_centering=True, normalizer_ignores_validation_mutations=True,
        first21_missing_volume_is_pooled_training_mean=True, future_mutation_causality=True,
        multi_backward_graphs_are_fresh=True, equal_asset_gradient_weight=True,
        one_optimizer_step_per_epoch=True, no_chunk_parameter_drift=True,
        mean_control_modifies_x_not_R=True, positive_predictions=True)


def metrics(pred, D):
    y = D['y'][D['va']]
    ewma = np.repeat(D['ewma'][D['va'], None], 3, axis=1)
    baseline = np.log(ewma) + y / ewma
    result = {}
    for name, value in pred.items():
        assert value.shape == y.shape and np.isfinite(value).all() and (value > 0).all()
        loss = np.log(value) + y / value
        result[name] = {str(h): dict(n=len(y), mean_QLIKE=float(loss[:, j].mean()),
            loss_difference_vs_EWMA22=float((loss[:, j] - baseline[:, j]).mean()),
            MSE=float(((value[:, j] - y[:, j]) ** 2).mean())) for j, h in enumerate(B.HORIZONS)}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--chunk', type=int, default=128)
    ap.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    ap.add_argument('--tag', default='round17_shared_precision_development')
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.check_only:
        print(json.dumps(synthetic_checks(), indent=2), flush=True)
        return
    assert 0 <= args.epochs <= 100 and args.chunk > 0 and args.seeds
    assert re.fullmatch(r'round17_shared_precision_[A-Za-z0-9_-]+', args.tag)
    paths = dict(manifest=HERE / f'{args.tag}_manifest.json', results=HERE / f'{args.tag}_results.json',
                 predictions=HERE / f'{args.tag}_validation_predictions.npz')
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f'Existing output protected; choose a new --tag: {path}')
    sources = dict(runner=Path(__file__).resolve(), filter=Path(B.__file__).resolve(),
        proxy_frame=Path(PROXY.__file__).resolve(), range_clock=Path(B.R14.__file__).resolve(),
        reference_models=B.REFERENCE, reference_results=PROXY.REFERENCE_RESULTS,
        **{f'csv_{asset}': B.SOURCES[asset] for asset in ASSETS})
    hashes = {name: dict(path=str(path), sha256=B.sha256(path)) for name, path in sources.items()}
    started = time.time()
    datasets, audit = prepare_pair()
    meta = dict(config=vars(args), status='DEVELOPMENT ONLY; no 2026 targets or forecasts',
        assets=list(ASSETS), modes=list(MODES), audit=audit, hashes=hashes, checks=synthetic_checks(),
        objective='equal-asset mean of three-horizon QLIKE + .05 equal-asset mean centered-reference-delta squared',
        centering='0.5*tanh(g(F))-mean(pre2025 asset bank 0.5*tanh(g(F_ref))); never validation-mean centering',
        scale_note='centering fixes average LOG correction, not arithmetic mean R; centered delta absolute bound is 1',
        information='unchanged a,z,volume; fixed geometry mapping, equal-asset pre2025 volume moments',
        training='sequential TBPTT; parameters unchanged throughout epoch; fresh theta and reference graph every chunk; .5*nchunk/Ntrain_asset weighting; penalty once; clip2; Adam .01 step once',
        selection='single equal-asset/h joint validation epoch including 0, also for local; patience15 min20 max100',
        process='asset-specific mu/phi/Q/s_R, same bounded AR1 form as frozen R16; same positive daily curve and exp(m+P/2)',
        seed_note='fixed fit reused deterministically after first seed; other four modes fit all seeds; seed slots are not independent datasets',
        mean_control='centered delta added to CURRENT x only; R=R_cell*exp(s_R)',
        outputs={k: str(v) for k, v in paths.items()}, fits={}, scores={}, joint_scores={})
    paths['manifest'].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    pred = {asset: B.direct_baselines(datasets[asset]) for asset in ASSETS}
    arrays = {}
    for asset, D in datasets.items():
        query = D['origin'][D['va']]
        bank = D['bank_index']
        arrays.update({f'{asset}_date': D['date'][query], f'{asset}_origin': query,
            f'{asset}_label_end': D['date'][query + 22], f'{asset}_target': D['y'][D['va']],
            f'{asset}_F17': D['F17'][query], f'{asset}_bank_origin': bank,
            f'{asset}_bank_date': D['date'][bank], f'{asset}_bank_F17': D['F17'][bank]})
    arrays['horizons'] = np.array(B.HORIZONS)
    for mode in MODES:
        values = {a: [] for a in ASSETS}; fits = []; cache = None
        for seed in args.seeds:
            if mode == 'fixed' and cache is not None:
                value, aux, info = copy.deepcopy(cache)
                info.update(seed=seed, fit_executed=False, deterministic_reuse_from_seed=args.seeds[0])
            else:
                value, aux, info = train_one(datasets, mode, seed, args.epochs, args.chunk)
                info.update(fit_executed=True, deterministic_reuse_from_seed=None)
                if mode == 'fixed':
                    cache = copy.deepcopy((value, aux, info))
            fits.append(info)
            for asset in ASSETS:
                values[asset].append(value[asset])
                pred[asset][f'{mode}_seed{seed}'] = value[asset]
                for key, state in aux[asset].items():
                    arrays[f'{asset}_{mode}_seed{seed}_{key}'] = state
            print(mode, seed, 'joint_epoch', info['best_joint_epoch'], 'joint_validation',
                  round(info['best_joint_validation_QLIKE'], 7), flush=True)
        for asset in ASSETS:
            pred[asset][mode] = values[asset][0].copy() if mode == 'fixed' else np.mean(values[asset], axis=0)
        meta['fits'][mode] = fits
    for asset in ASSETS:
        meta['scores'][asset] = metrics(pred[asset], datasets[asset])
        arrays.update({f'{asset}_{name}': value for name, value in pred[asset].items()})
    common_names = set(pred[ASSETS[0]]) & set(pred[ASSETS[1]])
    for name in sorted(common_names):
        perasset = []
        for asset in ASSETS:
            D = datasets[asset]
            value = pred[asset][name]
            y, scale = D['y'][D['va']], D['ewma'][D['va']]
            perasset.append(float((np.log(value) + y / value - np.log(scale)[:, None]).mean()))
        meta['joint_scores'][name] = dict(mean_known_scale_QLIKE=sum(perasset) / 2,
                                           asset_known_scale_QLIKE=dict(zip(ASSETS, perasset)))
    for name, path in sources.items():
        assert B.sha256(path) == hashes[name]['sha256'], f'Source changed during run: {name}'
    meta['elapsed_seconds'] = time.time() - started
    paths['results'].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(paths['predictions'], **arrays)
    print('JOINT DEVELOPMENT FINISHED; 2026 UNUSED', round(meta['elapsed_seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
