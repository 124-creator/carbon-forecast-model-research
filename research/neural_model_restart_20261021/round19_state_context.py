"""R19: matched F5 state-context-conditioned predictive corrections.

Code delivery defaults to pure synthetic checks; --run-development is opt-in.
Observer's affine residual is algebraically the frozen mean-field KF, not a new
KF algorithm. No claim of identifying true measurement bias. No new observations.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import time
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import round18_direct_controls as C18

S18, S17, B = C18.S18, C18.S17, C18.B
KP = S18.KP
ASSETS, DTYPE = C18.ASSETS, C18.DTYPE
HERE = Path(__file__).resolve().parent
MODES = ('observer', 'current_only', 'direct_readout')
SEEDS = C18.SEEDS
LAMBDA, LR = C18.LAMBDA, C18.LR
MAX_EPOCHS, PATIENCE, MIN_EPOCHS = C18.MAX_EPOCHS, C18.PATIENCE, C18.MIN_EPOCHS
precompute_baseline = C18.precompute_baseline
train_epoch = C18.train_epoch
evaluate = C18.evaluate


def observer_residual(phi, K, delta):
    """e[-1]=0; e[t]=phi*(1-K[t])*e[t-1]+K[t]*delta[t].

    Full prefix, no detach/TBPTT. Delta depends only on frozen F5 context.
    Missing observations have K=0, hence e propagates by phi without update.
    """
    assert K.shape == delta.shape and K.ndim == 1
    return KP.affine_prefix(phi * (1 - K), K * delta)[1]


class StateContext(C18.DirectControls):
    """Three modes differ only in how the SAME F5 MLP delta corrects m0."""
    def __init__(self, datasets, parameters, baseline, mode, seed):
        if mode not in MODES:
            raise ValueError(mode)
        # Reuse exact frozen processes, baseline copies and bank validation.
        super().__init__(datasets, parameters, baseline, 'current_only', seed)
        self.mode = mode
        torch.manual_seed(seed)
        self.head = nn.Sequential(nn.Linear(5, 8), nn.Tanh(), nn.Linear(8, 1)).to(dtype=DTYPE)
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        assert sum(p.numel() for p in self.parameters()) == 57

    def context(self, a, features, stop):
        # The final two columns are buffers, never updated by gradients.
        return torch.cat([features[:stop], getattr(self, f'K0_{a}')[:stop, None],
            (getattr(self, f'm0_{a}')[:stop] - self.processes[a].mu)[:, None]], dim=1)

    def raw_sequence(self, asset, features, stop, detach_every=None):
        return .5 * torch.tanh(self.head(self.context(asset, features, stop)).squeeze(-1))

    def fields(self, a, D, T, stop, detach_every=None):
        assert D['cut'] <= stop <= D['n']
        # One complete prefix graph; bank location/membership is reused from C18.
        raw = self.raw_sequence(a, T['F17'], stop)
        bank = getattr(self, f'bank_{a}')
        f5 = self.context(a, T['F17'], D['cut'])
        bank_raw = .5 * torch.tanh(self.head(f5[bank]).squeeze(-1))
        # Preserve S18/C18 mean-field bank reduction convention.
        center = torch.stack([bank_raw, torch.zeros_like(bank_raw)], 1).mean(dim=0)[0]
        return raw - center, bank_raw - center, center

    def corrected_state(self, a, delta):
        if self.mode != 'observer':
            return super().corrected_state(a, delta)
        n = len(delta); K = getattr(self, f'K0_{a}')[:n]
        e = observer_residual(self.processes[a].phi, K, delta)
        return getattr(self, f'm0_{a}')[:n] + e, getattr(self, f'P0_{a}')[:n]


def objective(model, datasets, T):
    # C18 objective calls fields on t<cut only, then one equal-asset/horizon mean.
    return C18.objective(model, datasets, T, detach_every=0)


def field_diagnostics(model, datasets, T):
    out = {}
    with torch.no_grad():
        for a in ASSETS:
            D = datasets[a]
            delta, ref, center = model.fields(a, D, T[a], D['n'])
            m, _ = model.corrected_state(a, delta)
            val = np.arange(D['n']) >= D['cut']
            out[a] = dict(bankcenter=float(center), bank_delta=S17.distribution(ref.numpy()),
                validation_delta=S17.distribution(delta[val].numpy()),
                train_state_correction=S17.distribution((m - getattr(model, f'm0_{a}'))[:D['cut']].numpy()),
                validation_state_correction=S17.distribution((m - getattr(model, f'm0_{a}'))[val].numpy()),
                validation_statistics='diagnostics only, never used for centering')
    return out


def train_one(datasets, parameters, baseline, mode, seed):
    T = S17.make_tensors(datasets)
    model = StateContext(datasets, parameters, baseline, mode, seed)
    immutable = {k: v.clone() for k, v in model.named_buffers()}
    _, best, scores, _, centers = evaluate(model, datasets, T)
    history = [dict(epoch=0, validation_joint_QLIKE=best, validation_asset_QLIKE=scores, bankcenter=centers)]
    best_state = copy.deepcopy(model.state_dict()); best_epoch = bad = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, MAX_EPOCHS + 1):
        info = train_epoch(model, datasets, T, optimizer)
        _, score, scores, _, centers = evaluate(model, datasets, T)
        history.append(dict(epoch=epoch, validation_joint_QLIKE=score,
                            validation_asset_QLIKE=scores, bankcenter=centers, **info))
        if score < best - 1e-5:
            best, best_epoch, bad = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if bad >= PATIENCE and epoch >= MIN_EPOCHS:
            break
    model.load_state_dict(best_state)
    pred, returned, scores, aux, centers = evaluate(model, datasets, T)
    for k, v in model.named_buffers():
        assert torch.equal(v, immutable[k]), f'Frozen buffer changed: {k}'
    return pred, aux, dict(mode=mode, seed=seed, best_joint_epoch=best_epoch,
        epochs_run=len(history) - 1, best_joint_validation_QLIKE=best,
        returned_joint_validation_QLIKE=returned, selected_epoch_asset_scores=scores,
        head_parameters=57, trainable_dynamic_parameters=0, bankcenter=centers,
        fields=field_diagnostics(model, datasets, T), history=history,
        selection='C18 rule: one joint equal-asset/horizon 2025 epoch including0',
        seed_interpretation='same reused dataset, repeated initialization; not independent datasets')


def _nonzero(model):
    return C18._nonzero(model)


def _synthetic_checks():
    datasets, raw = S17.synthetic_pair()
    parameters = S18._synthetic_parameters(); T = S17.make_tensors(datasets)
    baseline = precompute_baseline(datasets, parameters, T)
    frozen = S18.ObservationFields(datasets, parameters, 'frozen', 11)
    expected, _, _, _ = S18.evaluate(frozen, datasets, T)
    checks = {}
    # Arbitrary nonzero delta, before centering: compare to both original KF engines.
    generator = torch.Generator().manual_seed(1901)
    max_error, max_grad_error = 0., 0.
    for a in ASSETS:
        D = datasets[a]; process = frozen.processes[a]
        mu, phi, Q, _ = process.parameters_bounded()
        x, R, _ = frozen.inputs(a, T[a], 0, D['n'])
        delta = (.3 * torch.randn(D['n'], generator=generator, dtype=DTYPE)).requires_grad_()
        e = observer_residual(phi, baseline[a]['K0'], delta)
        m = baseline[a]['m0'] + e
        weights = torch.linspace(.3, 1.7, D['n'], dtype=DTYPE)
        loss = (weights * (m + 7.8).square()).mean()
        g = torch.autograd.grad(loss, delta, retain_graph=True)[0]
        for engine in (KP.kalman_prefix, KP.frozen_serial):
            mm, pp, kk = engine(x + delta, R, D['observed'], phi, Q, mu, *process.initial_state(), validate=False)
            torch.testing.assert_close(m, mm, rtol=5e-12, atol=5e-12)
            torch.testing.assert_close(pp, baseline[a]['P0'], rtol=5e-12, atol=5e-12)
            torch.testing.assert_close(kk, baseline[a]['K0'], rtol=5e-12, atol=5e-12)
            gg = torch.autograd.grad((weights * (mm + 7.8).square()).mean(), delta)[0]
            torch.testing.assert_close(g, gg, rtol=2e-8, atol=2e-10)
            max_error = max(max_error, float((m - mm).detach().abs().max()))
            max_grad_error = max(max_grad_error, float((g - gg).abs().max()))
        assert g[12] == 0
        K = baseline[a]['K0']; A = phi * (1 - K)
        assert (A >= 0).all() and (A <= phi).all() and (A + K <= 1 + 1e-15).all()
        # Includes constant signs and general varying delta; zero initial residual.
        for d in (torch.full_like(K, .37), torch.full_like(K, -.37), delta.detach()):
            ee = observer_residual(phi, K, d)
            assert ee.abs().max() <= d.abs().max() + 1e-14
        with torch.no_grad():
            changed = delta.detach().clone(); changed[40:] += 5
            torch.testing.assert_close(observer_residual(phi, K, changed)[:40], e[:40], rtol=0, atol=0)
    checks.update(affine_observer_matches_original_KF_means_gradients=True, P_and_K_unchanged=True,
        missing_observation_delta_gradient_zero=True, stable_A_between_zero_and_phi=True,
        constant_and_arbitrary_delta_residual_bounded=True, affine_future_mutation_exact=True,
        max_KF_state_error=max_error, max_KF_delta_gradient_error=max_grad_error)
    for mode in MODES:
        model = StateContext(datasets, parameters, baseline, mode, 11)
        values, _, _, _, center = evaluate(model, datasets, T)
        for a in ASSETS:
            np.testing.assert_array_equal(values[a], expected[a]); assert center[a] == 0
        assert sum(p.numel() for p in model.parameters()) == 57
    checks['zero_head_exact_frozen_nesting_all_modes'] = True
    models = [_nonzero(StateContext(datasets, parameters, baseline, mode, 22)) for mode in MODES]
    for a in ASSETS:
        D = datasets[a]
        outputs = [m.fields(a, D, T[a], D['n']) for m in models]
        for out in outputs[1:]:
            for left, right in zip(outputs[0], out):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
        f5 = models[0].context(a, T[a]['F17'], D['n'])
        torch.testing.assert_close(f5[:, :3], T[a]['F17'], rtol=0, atol=0)
        torch.testing.assert_close(f5[:, 3], baseline[a]['K0'], rtol=0, atol=0)
        torch.testing.assert_close(f5[:, 4], baseline[a]['m0'] - models[0].processes[a].mu, rtol=0, atol=0)
    checks['identical_F5_head_weights_delta_and_centers_across_modes'] = True
    for model in models:
        immutable = {k: v.clone() for k, v in model.named_buffers()}
        changed_T = {a: {k: v.clone() for k, v in T[a].items()} for a in ASSETS}
        for a in ASSETS:
            boundary = datasets[a]['cut'] + 4
            changed_T[a]['F17'][boundary:] += 9
            changed_T[a]['log_proxy'][boundary:] += 4
        changed_base = precompute_baseline(datasets, parameters, changed_T)
        other = StateContext(datasets, parameters, changed_base, model.mode, 22)
        other.head.load_state_dict(model.head.state_dict())
        with torch.no_grad():
            for a in ASSETS:
                D = datasets[a]; boundary = D['cut'] + 4
                d, ref, c = model.fields(a, D, T[a], D['n'])
                dd, _, cc = other.fields(a, D, changed_T[a], D['n'])
                torch.testing.assert_close(c, cc, rtol=0, atol=0)
                torch.testing.assert_close(d[:boundary], dd[:boundary], rtol=0, atol=0)
                m, P = model.corrected_state(a, d); mm, _ = other.corrected_state(a, dd)
                torch.testing.assert_close(m[:boundary], mm[:boundary], rtol=0, atol=0)
                assert abs(float(ref.mean())) < 1e-14
                daily = model.processes[a].future_daily_means(m, P)
                v = model.processes[a].forecast(m, P)
                torch.testing.assert_close(v, torch.stack([daily[:, :h].mean(1) for h in B.HORIZONS], 1))
                assert (daily > 0).all()
        grad_T = {a: {**T[a], 'F17': T[a]['F17'].clone().requires_grad_(),
                        'y': T[a]['y'].clone().requires_grad_()} for a in ASSETS}
        loss, info = objective(model, datasets, grad_T)
        grads = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True)
        explicit = 0
        for a in ASSETS:
            _, ref, _ = model.fields(a, datasets[a], grad_T[a], datasets[a]['cut'])
            explicit = explicit + .5 * (info['losses'][a] + LAMBDA * ref.square().mean())
        reference_grads = torch.autograd.grad(explicit, tuple(model.parameters()), retain_graph=True)
        for left, right in zip(grads, reference_grads):
            torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-12)
        # Full head gradient also matches observation-shift KF in observer mode.
        if model.mode == 'observer':
            reference = 0
            for a in ASSETS:
                D = datasets[a]; ids = D['tr']; q = D['origin'][ids]; process = model.processes[a]
                d, ref, _ = model.fields(a, D, grad_T[a], D['cut'])
                x, R, _ = frozen.inputs(a, grad_T[a], 0, D['cut'])
                mu, phi, Q, _ = process.parameters_bounded()
                m, P, _ = KP.frozen_serial(x + d, R, D['observed'][:D['cut']], phi, Q, mu,
                                           *process.initial_state(), validate=False)
                value = process.forecast(m[q], P[q])
                reference = reference + .5 * (B.qlike(value, grad_T[a]['y'][ids], grad_T[a]['ewma'][ids]).mean() + LAMBDA * ref.square().mean())
            rg = torch.autograd.grad(reference, tuple(model.parameters()))
            for left, right in zip(grads, rg):
                torch.testing.assert_close(left, right, rtol=2e-8, atol=2e-10)
        loss.backward()
        for a in ASSETS:
            D = datasets[a]; g = grad_T[a]['F17'].grad
            assert g[:D['cut']].abs().sum() > 0 and (g[D['cut']:] == 0).all()
            assert (grad_T[a]['y'].grad[D['va']] == 0).all()
            changed_T[a]['F17'][D['cut']:] += 99; changed_T[a]['y'][D['va']] *= 1e6
        after, _ = objective(model, datasets, changed_T)
        torch.testing.assert_close(loss.detach(), after.detach(), rtol=0, atol=0)
        class CountingAdam(torch.optim.Adam):
            calls = 0
            def step(self, *args, **kwargs):
                self.calls += 1
                return super().step(*args, **kwargs)
        optimizer = CountingAdam(model.parameters(), lr=LR)
        with patch('torch.autograd.backward', wraps=torch.autograd.backward) as backward:
            train_epoch(model, datasets, T, optimizer)
        assert backward.call_count == optimizer.calls == 1
        for k, v in model.named_buffers():
            assert torch.equal(v, immutable[k])
    # Confirm exact reused normalizer ignores all postcut raw feature mutations.
    dd, rr = copy.deepcopy(datasets), copy.deepcopy(raw)
    before = S17.install_features(copy.deepcopy(datasets), raw)
    for a in ASSETS:
        rr[a][datasets[a]['cut']:] = 1e6
    assert S17.install_features(dd, rr) == before
    for a in ASSETS:
        np.testing.assert_array_equal(dd[a]['F17'][:datasets[a]['cut']], datasets[a]['F17'][:datasets[a]['cut']])
    checks.update(observer_full_head_gradient_matches_original_KF=True,
        future_feature_observation_mutation_preserves_past_and_center=True,
        train_has_no_val_feature_target_gradient_or_val_center=True,
        equal_asset_horizon_and_centered_penalty_differentiable=True,
        actual_one_backward_and_optimizer_step=True, frozen_buffers_unchanged=True,
        normalizer_trainonly_reused=True, shared_positive_dailycurve_1_5_22=True,
        head_parameters={mode: 57 for mode in MODES}, real_training_executed=False)
    return checks


def synthetic_checks():
    # Fail loudly if any purecheck ever acquires a dependency on real inputs.
    readers = ('pandas.read_csv', 'numpy.load', 'pathlib.Path.read_text',
        'round17_shared_precision.prepare_pair', 'round18_frozen_moments.load_stage16_parameters',
        'round18_frozen_moments.verify_real_baseline')
    with ExitStack() as stack:
        for target in readers:
            stack.enter_context(patch(target, side_effect=AssertionError('real input forbidden in purechecks')))
        checks = _synthetic_checks()
    checks.update(real_input_readers_guarded=True, market_files_opened=False, stage16_results_opened=False)
    return checks


def source_hashes():
    out = C18.source_hashes()
    out['direct_controls'] = out.pop('runner')
    p = Path(__file__).resolve()
    out['runner'] = dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    return out


def run_development(tag):
    """Future review-gated real development entry; no 2026 targets constructed."""
    assert re.fullmatch(r'round19_state_context_[A-Za-z0-9_-]+', tag)
    paths = {k: HERE / f'{tag}{s}' for k, s in dict(results='_results.json', predictions='_validation_predictions.npz').items()}
    for p in paths.values():
        if p.exists():
            raise FileExistsError(f'Protected existing output: {p}')
    started = time.time(); hashes = source_hashes()
    assert hashes['prefix']['sha256'] == S18.PREFIX_SHA256
    parameters, provenance = S18.load_stage16_parameters(); datasets, audit = S17.prepare_pair()
    for a in ASSETS:
        assert abs(provenance[a]['reference_c512'] - audit['proxy_c512']) < 1e-14
        assert (datasets[a]['date'] < B.RESERVED).all()
    frozen_pred, matching = S18.verify_real_baseline(datasets, parameters)
    T = S17.make_tensors(datasets); baseline = precompute_baseline(datasets, parameters, T)
    arrays = dict(horizons=np.array(B.HORIZONS)); predictions = {a: dict(frozen=frozen_pred[a]) for a in ASSETS}
    meta = dict(config=dict(modes=MODES, seeds=SEEDS, lr=LR, regularization=LAMBDA, head_parameters=57,
        max_epochs=MAX_EPOCHS, patience=PATIENCE, min_epochs=MIN_EPOCHS), hashes=hashes,
        status='REUSED 2025 DEVELOPMENT; seeds are not independent datasets; no holdout targets',
        stage16_checkpoints=provenance, audit=audit, baseline_matching=matching, frozen_parameters=parameters,
        fits={}, scores={}, joint_scores={}, checks=synthetic_checks(),
        interpretation='state-context-conditioned predictive correction; true measurement bias not identified',
        features='all modes identical F5=[F17,K0,m0-mu] from frozen baseline; no new observations',
        observer='exact frozen mean-field KF residual algebra via KP.affine_prefix, not a new KF algorithm',
        training='full training prefix, no TBPTT; delta independent of corrected state; one data+penalty backward/Adam step',
        comparison='observer full-prefix gradients differ from prior S18/GRU detach128; identical information, no new observations',
        center='reused asset pre2025 bank only, differentiable; no validation means',
        forecast='S18.FrozenProcess.forecast(corrected_m,P0); same positive exp(m+P/2) dailycurve',
        selection='C18 unique joint epoch including0; stage16 previously2025-selected; no GRU retraining')
    for a in ASSETS:
        D = datasets[a]; q = D['origin'][D['va']]
        arrays.update({f'{a}_date': D['date'][q], f'{a}_origin': q, f'{a}_label_end': D['date'][q + 22],
            f'{a}_target': D['y'][D['va']], f'{a}_bank_date': D['date'][D['bank_index']],
            f'{a}_bank_F17': D['F17'][D['bank_index']]})
        for k, v in baseline[a].items():
            arrays[f'{a}_{k}'] = v.numpy()
        example = StateContext(datasets, parameters, baseline, 'observer', 11)
        arrays[f'{a}_bank_F5'] = example.context(a, T[a]['F17'], D['cut'])[D['bank_index']].numpy()
    for mode in MODES:
        fits, values = [], {a: [] for a in ASSETS}
        for seed in SEEDS:
            pred, aux, info = train_one(datasets, parameters, baseline, mode, seed); fits.append(info)
            for a in ASSETS:
                name = f'{mode}_seed{seed}'; predictions[a][name] = pred[a]; values[a].append(pred[a])
                for k, v in aux[a].items():
                    arrays[f'{a}_{name}_{k}'] = v
            print(mode, seed, 'joint_epoch', info['best_joint_epoch'], flush=True)
        meta['fits'][mode] = fits
        for a in ASSETS:
            predictions[a][f'{mode}_ensemble'] = np.mean(values[a], axis=0)
    for a in ASSETS:
        meta['scores'][a] = S17.metrics(predictions[a], datasets[a])
        arrays.update({f'{a}_{k}': v for k, v in predictions[a].items()})
    for name in predictions[ASSETS[0]]:
        perasset, perh = {}, {}
        for a in ASSETS:
            D = datasets[a]; v = predictions[a][name]; y = D['y'][D['va']]; scale = D['ewma'][D['va']]
            perasset[a] = dict(known_scale_QLIKE=float((np.log(v) + y / v - np.log(scale)[:, None]).mean()),
                              MSE=float(((v - y) ** 2).mean()))
        for h in B.HORIZONS:
            perh[str(h)] = {metric: sum(meta['scores'][a][name][str(h)][metric] for a in ASSETS) / 2
                           for metric in ('mean_QLIKE', 'loss_difference_vs_EWMA22', 'MSE')}
        meta['joint_scores'][name] = dict(asset_scores=perasset, per_horizon=perh,
            mean_known_scale_QLIKE=sum(x['known_scale_QLIKE'] for x in perasset.values()) / 2,
            MSE=sum(x['MSE'] for x in perasset.values()) / 2)
    assert source_hashes() == hashes
    meta['elapsed_seconds'] = time.time() - started
    paths['results'].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(paths['predictions'], **arrays)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    gate = ap.add_mutually_exclusive_group()
    gate.add_argument('--check-only', action='store_true'); gate.add_argument('--run-development', action='store_true')
    ap.add_argument('--tag', default='round19_state_context_development')
    args = ap.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.run_development:
        run_development(args.tag)
    else:
        print(json.dumps(dict(checks=synthetic_checks(), source_hashes=source_hashes()), indent=2), flush=True)


if __name__ == '__main__':
    main()
