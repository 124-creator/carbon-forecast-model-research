"""Round18: observation fields over independently selected, frozen R16 moments.

Only this new source is changed. --check-only uses pure synthetic fixtures;
real execution must first match the original R16 baseline arrays. No 2026
market targets are constructed. Every result is a reused-2025 development
diagnostic, not causal identification of measurement noise.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import re
import time
import numpy as np
import torch
from torch import nn
import round17_shared_precision as S17
import kalman_prefix_utils as KP

B = S17.B
HERE = Path(__file__).resolve().parent
ASSETS = S17.ASSETS
MODES = ('frozen', 'shared_mean', 'shared_precision', 'shared_both', 'shared_bounded_linear_both')
DTYPE = torch.float64
LAMBDA, LR = .05, .01
PREFIX_SHA256 = '7d18735778154be2912eab1ebdb56881c2c27ca6fe39b4f4704c99b65fec7b36'
STAGE16 = {asset: dict(
    results=HERE / f'bridge_precision_proxy_round16_{asset}_grid512_development_results.json',
    predictions=HERE / f'bridge_precision_proxy_round16_{asset}_grid512_development_validation_predictions.npz')
    for asset in ASSETS}


def load_stage16_parameters():
    """Executed only in the real runner, never by synthetic checks."""
    parameters, provenance = {}, {}
    for asset in ASSETS:
        path = STAGE16[asset]['results']
        meta = json.loads(path.read_text(encoding='utf-8'))
        assert meta['config']['instrument'] == asset and meta['config']['reference_grid'] == 512
        fit = meta['fits']['fixed_precision'][0]
        bounded = {key: float(fit['dynamics'][key]) for key in ('mu', 'phi', 'Q', 's_R')}
        assert all(np.isfinite(list(bounded.values())))
        assert 0 < bounded['phi'] < 1 and bounded['Q'] > 0
        # This prevents silently reusing predictions after any upstream change.
        source = dict(runner=Path(S17.PROXY.__file__).resolve(), filter=Path(B.__file__).resolve(),
            range_clock=Path(B.R14.__file__).resolve(), reference_models=B.REFERENCE,
            reference_results=S17.PROXY.REFERENCE_RESULTS, csv=B.SOURCES[asset])
        for key, current in source.items():
            assert B.sha256(current) == meta['hashes'][key]['sha256'], f'{asset} R16 source mismatch: {key}'
        parameters[asset] = bounded
        provenance[asset] = dict(parameters=bounded, source_results=str(path),
            source_predictions=str(STAGE16[asset]['predictions']), selected_seed=fit['seed'],
            selected_stage16_epoch=fit['best_epoch'],
            reference_c512=meta['audit']['reference_train_mean_proxy'],
            selection_note='fixed_precision first seed, final bounded parameters selected on 2025 in stage16; not refitted here')
    return parameters, provenance


class FrozenProcess(nn.Module):
    def __init__(self, bounded):
        super().__init__()
        for name in ('mu', 'phi', 'Q', 's_R'):
            value = float(bounded[name])
            assert np.isfinite(value)
            self.register_buffer(name, torch.tensor(value, dtype=DTYPE))
        assert 0 < float(self.phi) < 1 and float(self.Q) > 0

    def parameters_bounded(self):
        return self.mu, self.phi, self.Q, self.s_R

    def initial_state(self, theta=None):
        mu, phi, Q, _ = self.parameters_bounded() if theta is None else theta
        return mu, Q / (1 - phi.square())

    def future_daily_means(self, m, P, theta=None):
        return B.PrecisionFilter.future_daily_means(self, m, P, theta)

    def forecast(self, m, P, theta=None):
        return B.PrecisionFilter.forecast(self, m, P, theta)


def make_head(outputs, linear):
    if linear:
        net = nn.Linear(3, outputs, bias=False).to(dtype=DTYPE)
        nn.init.zeros_(net.weight)
    else:
        net = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, outputs)).to(dtype=DTYPE)
        nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
    return net


class ObservationFields(nn.Module):
    def __init__(self, datasets, parameters, mode, seed):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.processes = nn.ModuleDict({asset: FrozenProcess(parameters[asset]) for asset in ASSETS})
        for asset in ASSETS:
            D = datasets[asset]
            self.register_buffer(f'bank_{asset}', torch.tensor(D['F17'][D['bank_index']].copy(), dtype=DTYPE))
        torch.manual_seed(seed)
        if mode != 'frozen':
            self.head = make_head(2 if mode in ('shared_both', 'shared_bounded_linear_both') else 1,
                                  mode == 'shared_bounded_linear_both')
        assert not any(True for _ in self.processes.parameters())

    def raw_fields(self, features):
        if self.mode == 'frozen':
            return features.new_zeros((len(features), 2))
        raw = .5 * torch.tanh(self.head(features))
        zero = torch.zeros_like(raw[:, :1])
        if self.mode == 'shared_mean':
            return torch.cat([raw, zero], dim=1)
        if self.mode == 'shared_precision':
            return torch.cat([zero, raw], dim=1)
        return raw

    def reference_delta(self, asset):
        raw = self.raw_fields(getattr(self, f'bank_{asset}'))
        return raw - raw.mean(dim=0)

    def delta(self, asset, features):
        reference = self.raw_fields(getattr(self, f'bank_{asset}'))
        return self.raw_fields(features) - reference.mean(dim=0)

    def penalty(self):
        # Sum channels, not mean over channels: each active channel gets lambda.
        return LAMBDA * sum(self.reference_delta(a).square().mean(dim=0).sum() for a in ASSETS) / 2

    def inputs(self, asset, T, lo, hi):
        delta = self.delta(asset, T['F17'][lo:hi])
        x = T['log_proxy'][lo:hi] - T['cell_b'][lo:hi] + delta[:, 0]
        noise = T['cell_R'][lo:hi] * torch.exp(self.processes[asset].s_R + delta[:, 1])
        return x, noise, delta


def filter_asset(model, asset, D, T, stop, engine='prefix'):
    process = model.processes[asset]
    mu, phi, Q, _ = process.parameters_bounded()
    x, noise, delta = model.inputs(asset, T, 0, stop)
    m0, P0 = process.initial_state()
    fn = KP.kalman_prefix if engine == 'prefix' else KP.frozen_serial
    m, P, K = fn(x, noise, D['observed'][:stop], phi, Q, mu, m0, P0, validate=False)
    return m, P, K, noise, delta


def evaluate(model, datasets, T, engine='prefix'):
    model.eval()
    pred, scores, aux = {}, {}, {}
    with torch.no_grad():
        for asset in ASSETS:
            D = datasets[asset]; query = D['origin'][D['va']]
            m, P, K, R, delta = filter_asset(model, asset, D, T[asset], int(query.max()) + 1, engine)
            value = model.processes[asset].forecast(m[query], P[query])
            assert torch.isfinite(value).all() and torch.all(value > 0)
            pred[asset] = value.numpy()
            scores[asset] = float(B.qlike(value, T[asset]['y'][D['va']], T[asset]['ewma'][D['va']]).mean())
            aux[asset] = dict(m=m[query].numpy(), P=P[query].numpy(), K=K[query].numpy(),
                              R=R[query].numpy(), delta_mean=delta[query, 0].numpy(), delta_precision=delta[query, 1].numpy())
    return pred, sum(scores.values()) / 2, scores, aux


def assert_baseline_arrays(D, saved, actual):
    query = D['origin'][D['va']]
    dates = saved['date']; label_end = saved['label_end']
    # Check dates BEFORE accessing saved target or prediction arrays.
    assert (dates < B.RESERVED).all() and (label_end < B.RESERVED).all()
    np.testing.assert_array_equal(dates, D['date'][query])
    np.testing.assert_array_equal(label_end, D['date'][query + 22])
    np.testing.assert_array_equal(saved['origin'], query)
    np.testing.assert_array_equal(saved['horizons'], np.array(B.HORIZONS))
    np.testing.assert_allclose(saved['target'], D['y'][D['va']], rtol=1e-11, atol=1e-14)
    expected = saved['fixed_precision']
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-14)
    diff = np.abs(actual - expected)
    return dict(n=len(query), max_absolute_error=float(diff.max()),
                max_relative_error=float((diff / np.maximum(np.abs(expected), np.finfo(float).tiny)).max()),
                tolerance='rtol=1e-11,atol=1e-14', exact_dates_origin_horizons=True, targets_same_source=True)


def verify_real_baseline(datasets, parameters):
    model = ObservationFields(datasets, parameters, 'frozen', 11)
    pred, joint, per_asset, _ = evaluate(model, datasets, S17.make_tensors(datasets))
    matching = {}
    for asset in ASSETS:
        with np.load(STAGE16[asset]['predictions'], allow_pickle=False) as saved:
            matching[asset] = assert_baseline_arrays(datasets[asset], saved, pred[asset])
    return pred, dict(assets=matching, joint_known_scale_QLIKE=joint,
                     asset_known_scale_QLIKE=per_asset, status='zero-field matches frozen stage16 before any neural training')


def train_epoch(model, datasets, T, optimizer, chunk):
    model.train(); optimizer.zero_grad()
    losses, weights, terminals = {}, {}, {}
    for asset in ASSETS:
        D, tensor, process = datasets[asset], T[asset], model.processes[asset]
        lookup = np.full(D['n'], -1, dtype=int); lookup[D['origin'][D['tr']]] = D['tr']
        ntrain = len(D['tr']); assert ntrain > 0
        state = process.initial_state(); total_loss = 0.; weight_sum = 0.
        for lo in range(0, D['cut'], chunk):
            hi = min(lo + chunk, D['cut'])
            theta = process.parameters_bounded()
            mu, phi, Q, _ = theta
            # Fresh neural/centering graph every chunk. Frozen theta is buffers.
            x, noise, _ = model.inputs(asset, tensor, lo, hi)
            m, P, _ = KP.kalman_prefix(x, noise, D['observed'][lo:hi], phi, Q, mu, *state, validate=False)
            position = np.flatnonzero(lookup[lo:hi] >= 0)
            if len(position):
                ids = lookup[lo:hi][position]
                assert (D['origin'][ids] + 22 < D['cut']).all()
                value = process.forecast(m[position], P[position], theta)
                loss = B.qlike(value, tensor['y'][ids], tensor['ewma'][ids]).mean()
                weight = .5 * len(ids) / ntrain
                assert torch.isfinite(loss)
                (weight * loss).backward()
                total_loss += len(ids) / ntrain * float(loss.detach()); weight_sum += weight
            state = (m[-1].detach(), P[-1].detach())
        assert abs(weight_sum - .5) < 1e-12
        losses[asset], weights[asset] = total_loss, weight_sum
        terminals[asset] = [float(x) for x in state]
    penalty = model.penalty()
    assert torch.isfinite(penalty)
    penalty.backward()  # fresh graph, once after both assets
    norm = float(nn.utils.clip_grad_norm_(model.parameters(), 2))
    assert np.isfinite(norm)
    optimizer.step()
    return dict(train_asset_QLIKE=losses, train_joint_QLIKE=sum(losses.values()) / 2,
                penalty=float(penalty.detach()), gradient_weights=weights,
                pre_step_terminal_states=terminals, gradient_norm_before_clip=norm, optimizer_steps=1)


def diagnostics(model, datasets, T):
    out = {}
    with torch.no_grad():
        for asset in ASSETS:
            D = datasets[asset]
            train = model.reference_delta(asset).numpy()
            val_ids = np.flatnonzero((np.arange(D['n']) >= D['cut']) & D['observed'])
            val = model.delta(asset, T[asset]['F17'][val_ids]).numpy()
            out[asset] = dict(
                frozen_parameters={key: float(getattr(model.processes[asset], key)) for key in ('mu', 'phi', 'Q', 's_R')},
                delta_train={name: S17.distribution(train[:, j]) for j, name in enumerate(('mean', 'precision'))},
                delta_validation={name: S17.distribution(val[:, j]) for j, name in enumerate(('mean', 'precision'))},
                validation_note='diagnostic only; validation means never used for centering')
    return out


def train_one(datasets, parameters, mode, seed, epochs, chunk=128):
    T = S17.make_tensors(datasets)
    model = ObservationFields(datasets, parameters, mode, seed)
    pred, best, scores, aux = evaluate(model, datasets, T)
    history = [dict(epoch=0, validation_joint_QLIKE=best, validation_asset_QLIKE=scores)]
    best_epoch, bad = 0, 0
    before = {name: value.clone() for name, value in model.processes.state_dict().items()}
    if mode != 'frozen':
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_state = copy.deepcopy(model.state_dict())
        for epoch in range(1, epochs + 1):
            training = train_epoch(model, datasets, T, optimizer, chunk)
            _, score, scores, _ = evaluate(model, datasets, T)
            history.append(dict(epoch=epoch, validation_joint_QLIKE=score, validation_asset_QLIKE=scores, **training))
            if score < best - 1e-5:
                best, best_epoch, bad = score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
            if bad >= 20 and epoch >= 30:
                break
        model.load_state_dict(best_state)
        pred, returned, scores, aux = evaluate(model, datasets, T)
    else:
        returned = best
    for name, value in model.processes.state_dict().items():
        assert torch.equal(value, before[name]), f'Frozen dynamic buffer changed: {name}'
    return pred, aux, dict(mode=mode, seed=seed, best_joint_epoch=best_epoch, epochs_run=len(history) - 1,
        best_joint_validation_QLIKE=best, returned_joint_validation_QLIKE=returned,
        selected_epoch_asset_scores=scores, head_parameters=sum(p.numel() for p in model.parameters()),
        trainable_dynamic_parameters=0, observations_only=True, state_and_delta=diagnostics(model, datasets, T), history=history,
        selection='one joint epoch across both assets including epoch0; stage16 checkpoint was itself chosen on reused2025 validation')


def _synthetic_parameters():
    return dict(KRBN=dict(mu=-8.05, phi=.95, Q=.02, s_R=.1),
                KCCA=dict(mu=-7.9, phi=.88, Q=.035, s_R=-.15))


def synthetic_checks():
    datasets, _ = S17.synthetic_pair()
    parameters = _synthetic_parameters(); T = S17.make_tensors(datasets)
    baseline_model = ObservationFields(datasets, parameters, 'frozen', 11)
    baseline, _, _, _ = evaluate(baseline_model, datasets, T)
    assert sum(p.numel() for p in baseline_model.parameters()) == 0
    assert float(baseline_model.processes['KRBN'].phi) != float(baseline_model.processes['KCCA'].phi)
    with torch.no_grad():
        for mode in MODES:
            model = ObservationFields(datasets, parameters, mode, 11)
            values, _, _, _ = evaluate(model, datasets, T)
            scalar, _, _, _ = evaluate(model, datasets, T, engine='scalar')
            for asset in ASSETS:
                np.testing.assert_array_equal(values[asset], baseline[asset])
                np.testing.assert_allclose(values[asset], scalar[asset], rtol=1e-11, atol=1e-14)
    # Pure dictionaries exercise the baseline identity gate without opening NPZs.
    for asset in ASSETS:
        D = datasets[asset]; q = D['origin'][D['va']]
        saved = dict(date=D['date'][q], label_end=D['date'][q + 22], origin=q,
                     horizons=np.array(B.HORIZONS), target=D['y'][D['va']], fixed_precision=baseline[asset])
        assert_baseline_arrays(D, saved, baseline[asset])
        bad = dict(saved); bad['fixed_precision'] = baseline[asset] * 1.001
        try:
            assert_baseline_arrays(D, bad, baseline[asset])
        except AssertionError:
            pass
        else:
            raise AssertionError('Baseline identity gate failed to reject a changed baseline')
    # Nonzero two-channel field: prefix state and HEAD gradients match scalar.
    left = ObservationFields(datasets, parameters, 'shared_both', 11)
    with torch.no_grad():
        left.head[-1].weight.fill_(.08); left.head[-1].bias.copy_(torch.tensor([.02, -.03], dtype=DTYPE))
    right = copy.deepcopy(left)
    def gradient(model, engine):
        model.zero_grad(); objective = model.penalty()
        for asset in ASSETS:
            D = datasets[asset]; q = D['origin'][D['tr']]
            m, P, _, _, _ = filter_asset(model, asset, D, T[asset], D['cut'], engine)
            value = model.processes[asset].forecast(m[q], P[q])
            objective = objective + .5 * B.qlike(value, T[asset]['y'][D['tr']], T[asset]['ewma'][D['tr']]).mean()
        objective.backward()
        return [p.grad.clone() for p in model.parameters()]
    for a, b in zip(gradient(left, 'prefix'), gradient(right, 'scalar')):
        torch.testing.assert_close(a, b, rtol=2e-8, atol=2e-10)
    class CountingAdam(torch.optim.Adam):
        calls = 0
        def step(self, *args, **kwargs):
            self.calls += 1
            return super().step(*args, **kwargs)
    for mode in MODES[1:]:
        model = ObservationFields(datasets, parameters, mode, 11)
        frozen_before = {name: x.clone() for name, x in model.processes.state_dict().items()}
        expected_states = {}
        with torch.no_grad():
            for asset in ASSETS:
                m, P, _, _, _ = filter_asset(model, asset, datasets[asset], T[asset], datasets[asset]['cut'])
                expected_states[asset] = [float(m[-1]), float(P[-1])]
        optimizer = CountingAdam(model.parameters(), lr=.01)
        info = train_epoch(model, datasets, T, optimizer, 16)
        assert optimizer.calls == 1
        for name, x in model.processes.state_dict().items():
            assert torch.equal(x, frozen_before[name])
        with torch.no_grad():
            for asset in ASSETS:
                np.testing.assert_allclose(info['pre_step_terminal_states'][asset], expected_states[asset], rtol=1e-12, atol=1e-12)
                assert abs(info['gradient_weights'][asset] - .5) < 1e-12
                delta_ref = model.reference_delta(asset)
                assert torch.all(delta_ref.mean(dim=0).abs() < 1e-12)
                assert float(delta_ref.abs().max()) <= 1
                x, R, delta = model.inputs(asset, T[asset], 0, 30)
                base_x = T[asset]['log_proxy'][:30] - T[asset]['cell_b'][:30]
                base_R = T[asset]['cell_R'][:30] * torch.exp(model.processes[asset].s_R)
                if mode == 'shared_mean':
                    torch.testing.assert_close(R, base_R, rtol=0, atol=0)
                if mode == 'shared_precision':
                    torch.testing.assert_close(x, base_x, rtol=0, atol=0)
            expected_penalty = .05 * sum(model.reference_delta(a).square().mean(dim=0).sum() for a in ASSETS) / 2
            torch.testing.assert_close(model.penalty(), expected_penalty, rtol=0, atol=0)
        output, joint, scores, _ = evaluate(model, datasets, T)
        assert abs(joint - sum(scores.values()) / 2) < 1e-12
        assert all(np.isfinite(v).all() and (v > 0).all() for v in output.values())
    # Missing rows and future mutations follow the already verified prefix engine.
    for asset in ASSETS:
        D = datasets[asset]
        with torch.no_grad():
            before = filter_asset(left, asset, D, T[asset], 48)
            changed = {k: v.clone() for k, v in T[asset].items()}
            changed['log_proxy'][40:] += 30; changed['F17'][40:] += 3
            after = filter_asset(left, asset, D, changed, 48)
            for a, b in zip(before[:2], after[:2]):
                torch.testing.assert_close(a[:40], b[:40], rtol=0, atol=0)
            assert before[2][12] == 0
    return dict(market_files_opened=False, stage16_results_opened=False,
        frozen_dynamics_are_buffers_only=True, asset_checkpoint_differences_preserved=True,
        all_zero_heads_exactly_nest_frozen=True, baseline_identity_gate_rejects_change=True,
        prefix_scalar_predictions_and_head_gradients_match=True, fixed_buffers_unchanged_after_training=True,
        fresh_chunk_centering_graphs=True, one_step_per_epoch=True, equal_asset_gradients=True,
        penalty_sums_channels=True, mean_and_precision_only_controls_valid=True,
        zero_range_skipped=True, future_mutation_prefix_exact=True, positive_shared_curve=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--chunk', type=int, default=128)
    ap.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    ap.add_argument('--tag', default='round18_frozen_moments_development')
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    assert B.sha256(Path(KP.__file__).resolve()) == PREFIX_SHA256, 'Frozen prefix utility changed'
    if args.check_only:
        print(json.dumps(synthetic_checks(), indent=2), flush=True)
        return
    assert 0 <= args.epochs <= 200 and args.chunk > 0 and args.seeds
    assert re.fullmatch(r'round18_frozen_moments_[A-Za-z0-9_-]+', args.tag)
    paths = {key: HERE / f'{args.tag}{suffix}' for key, suffix in
             dict(manifest='_manifest.json', results='_results.json', predictions='_validation_predictions.npz').items()}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f'Protected existing output; use new --tag: {path}')
    sources = dict(runner=Path(__file__).resolve(), shared_prepare=Path(S17.__file__).resolve(),
        frozen_filter=Path(B.__file__).resolve(), proxy_frame=Path(S17.PROXY.__file__).resolve(),
        range_clock=Path(B.R14.__file__).resolve(), prefix=Path(KP.__file__).resolve(),
        reference_models=B.REFERENCE, reference_results=S17.PROXY.REFERENCE_RESULTS,
        **{f'csv_{a}': B.SOURCES[a] for a in ASSETS},
        **{f'stage16_{a}_{k}': p for a in ASSETS for k, p in STAGE16[a].items()})
    hashes = {name: dict(path=str(path), sha256=B.sha256(path)) for name, path in sources.items()}
    started = time.time()
    parameters, provenance = load_stage16_parameters()
    datasets, audit = S17.prepare_pair()
    for a in ASSETS:
        assert abs(provenance[a]['reference_c512'] - audit['proxy_c512']) < 1e-14
    baseline, matching = verify_real_baseline(datasets, parameters)
    meta = dict(config=vars(args), status='REUSED 2025 DEVELOPMENT DIAGNOSTIC; no 2026 targets or causal noise identification',
        assets=list(ASSETS), modes=list(MODES), audit=audit, hashes=hashes, stage16_checkpoints=provenance,
        baseline_matching=matching, frozen_parameters=parameters, checks=synthetic_checks(),
        model='only shared observation fields train; per-asset R16 mu/phi/Q/s_R buffers never change',
        selection='unique joint equal-asset/h2025 epoch incl0; max200 patience20 min30; stage16 parameters were already2025-selected',
        fields='two canonical channels [mean,logprecision]; .5*tanh(g)-asset pre2025 bank mean',
        regularization='lambda=.05*equal-asset SUM_channel mean(delta_ref^2), not averaged over channels',
        training='frozen epoch parameters; chunk128 detach; fresh centering graph; equal-asset chunk gradients; separate penalty backward; clip2; Adam .01 one step',
        filtering='frozen validated Mobius/affine prefix; future mean exp(m_k+P_k/2); all h same positive curve',
        interpretation='2025-selected checkpoints and repeated validation; development prediction calibration only, not true QV or causal measurement noise',
        external_benchmarks='raw16 and MEM/HAR/HGB merged externally; not refitted or chosen here',
        outputs={k: str(v) for k, v in paths.items()}, fits={}, scores={}, joint_scores={})
    paths['manifest'].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    pred = {a: {} for a in ASSETS}; arrays = {'horizons': np.array(B.HORIZONS)}
    for asset, D in datasets.items():
        q = D['origin'][D['va']]; bank = D['bank_index']
        arrays.update({f'{asset}_date': D['date'][q], f'{asset}_origin': q,
            f'{asset}_label_end': D['date'][q + 22], f'{asset}_target': D['y'][D['va']],
            f'{asset}_bank_date': D['date'][bank], f'{asset}_bank_F17': D['F17'][bank]})
    for mode in MODES:
        values = {a: [] for a in ASSETS}; fits = []; cache = None
        for seed in args.seeds:
            if mode == 'frozen' and cache is not None:
                value, aux, info = copy.deepcopy(cache)
                info.update(seed=seed, fit_executed=False, deterministic_reuse_from_seed=args.seeds[0])
            else:
                value, aux, info = train_one(datasets, parameters, mode, seed, args.epochs, args.chunk)
                info.update(fit_executed=mode != 'frozen', deterministic_reuse_from_seed=None)
                if mode == 'frozen':
                    cache = copy.deepcopy((value, aux, info))
            fits.append(info)
            for asset in ASSETS:
                values[asset].append(value[asset]); pred[asset][f'{mode}_seed{seed}'] = value[asset]
                for key, state in aux[asset].items():
                    arrays[f'{asset}_{mode}_seed{seed}_{key}'] = state
            print(mode, seed, 'joint_epoch', info['best_joint_epoch'], 'joint_score',
                  round(info['best_joint_validation_QLIKE'], 7), flush=True)
        for asset in ASSETS:
            pred[asset][mode] = values[asset][0].copy() if mode == 'frozen' else np.mean(values[asset], axis=0)
        meta['fits'][mode] = fits
    for asset in ASSETS:
        meta['scores'][asset] = S17.metrics(pred[asset], datasets[asset])
        # Add a checkpoint-specific comparison without replacing external EWMA scores.
        D = datasets[asset]; y = D['y'][D['va']]
        base_loss = np.log(baseline[asset]) + y / baseline[asset]
        for name, value in pred[asset].items():
            loss = np.log(value) + y / value
            for j, h in enumerate(B.HORIZONS):
                meta['scores'][asset][name][str(h)]['loss_difference_vs_frozen16'] = float((loss[:, j] - base_loss[:, j]).mean())
        arrays.update({f'{asset}_{name}': v for name, v in pred[asset].items()})
    for name in sorted(set(pred[ASSETS[0]]) & set(pred[ASSETS[1]])):
        scores = {}
        for asset in ASSETS:
            D = datasets[asset]; v = pred[asset][name]; y = D['y'][D['va']]; scale = D['ewma'][D['va']]
            scores[asset] = float((np.log(v) + y / v - np.log(scale)[:, None]).mean())
        joint_mse = sum(float(np.mean((pred[a][name] - datasets[a]['y'][datasets[a]['va']]) ** 2)) for a in ASSETS) / 2
        joint_qlike = sum(scores.values()) / 2
        meta['joint_scores'][name] = dict(mean_known_scale_QLIKE=joint_qlike, asset_scores=scores,
            loss_difference_vs_frozen16=joint_qlike - matching['joint_known_scale_QLIKE'], MSE=joint_mse)
    for name, path in sources.items():
        assert B.sha256(path) == hashes[name]['sha256'], f'Frozen input changed: {name}'
    meta['elapsed_seconds'] = time.time() - started
    paths['results'].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(paths['predictions'], **arrays)
    print('FROZEN-MOMENT DEVELOPMENT FINISHED; 2026 UNUSED', round(meta['elapsed_seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
