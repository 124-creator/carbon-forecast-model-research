"""R18 direct controls; default CLI is pure synthetic, real runner is opt-in.
Seeds repeat initialization on one reused-2025 dataset, not independent datasets.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
import torch
from torch import nn
import round18_frozen_moments as S18

S17, B = S18.S17, S18.B
ASSETS, DTYPE = S18.ASSETS, S18.DTYPE
HERE = Path(__file__).resolve().parent
MODES = ('current_only', 'direct_readout', 'shared_GRU')
SEEDS = (11, 22, 33)
LAMBDA, LR, CHUNK = .05, .01, 128
MAX_EPOCHS, PATIENCE, MIN_EPOCHS = 200, 20, 30


def precompute_baseline(datasets, parameters, T):
    """All frozen IPO-to-end states; no trainable control head enters this path."""
    frozen = S18.ObservationFields(datasets, parameters, 'frozen', 11)
    baseline = {}
    with torch.no_grad():
        for a in ASSETS:
            D = datasets[a]
            m, P, K, _, delta = S18.filter_asset(frozen, a, D, T[a], D['n'], 'prefix')
            assert not delta.any() and not any(p.requires_grad for p in frozen.parameters())
            baseline[a] = dict(m0=m.detach().clone(), P0=P.detach().clone(), K0=K.detach().clone())
            assert (K[~D['observed']] == 0).all()
    return baseline


class DirectControls(nn.Module):
    def __init__(self, datasets, parameters, baseline, mode, seed):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.processes = nn.ModuleDict({a: S18.FrozenProcess(parameters[a]) for a in ASSETS})
        for a in ASSETS:
            D = datasets[a]; bank = np.asarray(D['bank_index'])
            assert len(bank) and (bank < D['cut']).all() and D['observed'][bank].all()
            if 'date' in D:
                assert (D['date'][bank] < B.CUT_DATE).all()
            self.register_buffer(f'bank_{a}', torch.tensor(bank, dtype=torch.long))
            for k in ('m0', 'P0', 'K0'):
                self.register_buffer(f'{k}_{a}', baseline[a][k].detach().clone())
        torch.manual_seed(seed)
        if mode == 'shared_GRU':
            self.gru = nn.GRU(5, 8, num_layers=1, batch_first=True).to(dtype=DTYPE)
            self.head = nn.Linear(8, 1).to(dtype=DTYPE)
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        else:
            self.head = S18.make_head(1, False)
        assert not any(True for _ in self.processes.parameters())

    def raw_sequence(self, asset, features, stop, detach_every=CHUNK):
        """Slice before graph construction; hidden0 at IPO; detach preserves values."""
        f = features[:stop]
        assert len(f) == stop and stop > 0
        if self.mode != 'shared_GRU':
            return .5 * torch.tanh(self.head(f).squeeze(-1))
        K = getattr(self, f'K0_{asset}')[:stop]
        dev = getattr(self, f'm0_{asset}')[:stop] - self.processes[asset].mu
        x = torch.cat([f, K[:, None], dev[:, None]], dim=1)
        hidden = x.new_zeros((1, 1, 8)); outputs = []
        width = detach_every if detach_every else stop
        assert width > 0
        for lo in range(0, stop, width):
            out, hidden = self.gru(x[None, lo:lo + width], hidden)
            outputs.append(.5 * torch.tanh(self.head(out[0]).squeeze(-1)))
            if detach_every:
                hidden = hidden.detach()
        return torch.cat(outputs)

    def fields(self, a, D, T, stop, detach_every=CHUNK):
        assert D['cut'] <= stop <= D['n']
        raw = self.raw_sequence(a, T['F17'], stop, detach_every)
        bank = getattr(self, f'bank_{a}')
        # Match S18's separate bank MLP evaluation, including floating-point order.
        # GRU bank values must come from the causal IPO rollout, never isolated rows.
        bank_raw = (raw[bank] if self.mode == 'shared_GRU' else
                    .5 * torch.tanh(self.head(T['F17'][bank]).squeeze(-1)))
        if self.mode == 'shared_GRU':
            center = bank_raw.mean()
        else:
            center = torch.stack([bank_raw, torch.zeros_like(bank_raw)], dim=1).mean(dim=0)[0]
        delta = raw - center
        return delta, bank_raw - center, center

    def corrected_state(self, a, delta):
        n = len(delta); base = getattr(self, f'm0_{a}')[:n]
        correction = getattr(self, f'K0_{a}')[:n] * delta if self.mode == 'current_only' else delta
        return base + correction, getattr(self, f'P0_{a}')[:n]


def objective(model, datasets, T, detach_every=CHUNK):
    """Equal asset/horizon common-h22 mature origins and bank penalty in ONE graph."""
    losses, penalties, centers = {}, {}, {}
    for a in ASSETS:
        D = datasets[a]; ids = D['tr']; q = D['origin'][ids]
        assert len(ids) and (q + 22 < D['cut']).all()
        delta, reference, center = model.fields(a, D, T[a], D['cut'], detach_every)
        m, P = model.corrected_state(a, delta)
        v = model.processes[a].forecast(m[q], P[q])
        losses[a] = B.qlike(v, T[a]['y'][ids], T[a]['ewma'][ids]).mean()
        penalties[a] = reference.square().mean(); centers[a] = center
    data = sum(losses.values()) / len(ASSETS)
    penalty = LAMBDA * sum(penalties.values()) / len(ASSETS)
    return data + penalty, dict(data=data, penalty=penalty, losses=losses, centers=centers)


def train_epoch(model, datasets, T, optimizer):
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss, info = objective(model, datasets, T)
    assert torch.isfinite(loss)
    loss.backward()
    norm = float(nn.utils.clip_grad_norm_(model.parameters(), 2))
    assert np.isfinite(norm)
    optimizer.step()
    return dict(train_joint_QLIKE=float(info['data'].detach()),
        train_asset_QLIKE={a: float(x.detach()) for a, x in info['losses'].items()},
        penalty=float(info['penalty'].detach()),
        pre_step_bankcenter={a: float(x.detach()) for a, x in info['centers'].items()},
        gradient_norm_before_clip=norm, optimizer_steps=1, backward_calls=1,
        gradient_asset_weights={a: 1 / len(ASSETS) for a in ASSETS})


def evaluate(model, datasets, T):
    model.eval(); predictions, scores, aux, centers = {}, {}, {}, {}
    with torch.no_grad():
        for a in ASSETS:
            D = datasets[a]; q = D['origin'][D['va']]
            # Fresh IPO hidden0, full no_grad sequence; center uses training bank.
            delta, reference, center = model.fields(a, D, T[a], D['n'], detach_every=0)
            m, P = model.corrected_state(a, delta)
            v = model.processes[a].forecast(m[q], P[q])
            assert torch.isfinite(v).all() and (v > 0).all()
            predictions[a] = v.numpy()
            scores[a] = float(B.qlike(v, T[a]['y'][D['va']], T[a]['ewma'][D['va']]).mean())
            aux[a] = dict(m=m[q].numpy(), P=P[q].numpy(), K0=getattr(model, f'K0_{a}')[q].numpy(),
                          delta=delta[q].numpy(), bank_delta=reference.numpy())
            centers[a] = float(center)
    return predictions, sum(scores.values()) / len(ASSETS), scores, aux, centers


def train_one(datasets, parameters, baseline, mode, seed):
    T = S17.make_tensors(datasets)
    model = DirectControls(datasets, parameters, baseline, mode, seed)
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
        head_parameters=sum(p.numel() for p in model.parameters()), trainable_dynamic_parameters=0,
        bankcenter=centers, history=history, selection='one joint equal-asset/horizon 2025 epoch including0',
        seed_interpretation='same dataset, different initialization; not independent datasets')


def _nonzero(model):
    with torch.no_grad():
        head = model.head if model.mode == 'shared_GRU' else model.head[-1]
        head.weight.copy_(torch.linspace(-.12, .15, 8, dtype=DTYPE)[None]); head.bias.fill_(.03)
    return model


def synthetic_checks():
    """In-memory only; never calls prepare_pair/load_stage16_parameters."""
    datasets, raw = S17.synthetic_pair()
    parameters = S18._synthetic_parameters(); T = S17.make_tensors(datasets)
    baseline = precompute_baseline(datasets, parameters, T)
    frozen = S18.ObservationFields(datasets, parameters, 'frozen', 11)
    expected, _, _, _ = S18.evaluate(frozen, datasets, T)
    checks = {}
    for mode in MODES:
        model = DirectControls(datasets, parameters, baseline, mode, 11)
        pred, _, _, _, centers = evaluate(model, datasets, T)
        for a in ASSETS:
            np.testing.assert_array_equal(pred[a], expected[a]); assert centers[a] == 0
        assert sum(p.numel() for p in model.parameters()) == (369 if mode == 'shared_GRU' else 41)
    checks['zero_head_exact_frozen_nesting'] = True
    model = _nonzero(DirectControls(datasets, parameters, baseline, 'current_only', 11))
    shared = S18.ObservationFields(datasets, parameters, 'shared_mean', 11)
    shared.head.load_state_dict(model.head.state_dict())
    for a in ASSETS:
        D = datasets[a]; delta, reference, _ = model.fields(a, D, T[a], D['n'])
        torch.testing.assert_close(delta, shared.delta(a, T[a]['F17'])[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(reference, shared.reference_delta(a)[:, 0], rtol=0, atol=0)
        m, P = model.corrected_state(a, delta)
        torch.testing.assert_close(m, baseline[a]['m0'] + baseline[a]['K0'] * delta, rtol=0, atol=0)
        assert m[12] == baseline[a]['m0'][12] and baseline[a]['K0'][12] == 0
        torch.testing.assert_close(P, baseline[a]['P0'], rtol=0, atol=0)
        theta = frozen.processes[a].parameters_bounded(); mu, phi, Q, _ = theta
        x, R, _ = frozen.inputs(a, T[a], 0, D['n'])
        perturbed = x.clone(); perturbed[30] += .17
        mm, pp, _ = S18.KP.kalman_prefix(perturbed, R, D['observed'], phi, Q, mu,
                                       *frozen.processes[a].initial_state(), validate=False)
        torch.testing.assert_close(mm[30] - baseline[a]['m0'][30], baseline[a]['K0'][30] * .17)
        torch.testing.assert_close(pp, baseline[a]['P0'], rtol=0, atol=0)
        changed = delta.clone(); changed[29] += .7
        cm, _ = model.corrected_state(a, changed)
        torch.testing.assert_close(cm[30:], m[30:], rtol=0, atol=0)
    checks['current_only_same_S18_MLP_center_and_K_no_carry'] = True
    mutated_data, mutated_raw = copy.deepcopy(datasets), copy.deepcopy(raw)
    original_stats = S17.install_features(copy.deepcopy(datasets), raw)
    for a in ASSETS:
        mutated_raw[a][datasets[a]['cut']:, :] = 1e6
    changed_stats = S17.install_features(mutated_data, mutated_raw)
    assert original_stats == changed_stats
    for a in ASSETS:
        np.testing.assert_array_equal(mutated_data[a]['F17'][:datasets[a]['cut']], T[a]['F17'][:datasets[a]['cut']].numpy())
    checks['normalizer_trainonly_exact_reuse'] = True
    for mode in MODES:
        model = _nonzero(DirectControls(datasets, parameters, baseline, mode, 22))
        buffers = {k: v.clone() for k, v in model.named_buffers()}
        changed_T = {a: {k: v.clone() for k, v in T[a].items()} for a in ASSETS}
        for a in ASSETS:
            boundary = datasets[a]['cut'] + 4
            changed_T[a]['F17'][boundary:] += 9; changed_T[a]['log_proxy'][boundary:] += 4
        changed_baseline = precompute_baseline(datasets, parameters, changed_T)
        changed_model = DirectControls(datasets, parameters, changed_baseline, mode, 22)
        for name, p in changed_model.named_parameters():
            with torch.no_grad():
                p.copy_(dict(model.named_parameters())[name])
        with torch.no_grad():
            for a in ASSETS:
                D = datasets[a]; boundary = D['cut'] + 4
                left, ref, center = model.fields(a, D, T[a], D['n'], 0)
                right, _, newcenter = changed_model.fields(a, D, changed_T[a], D['n'], 0)
                torch.testing.assert_close(center, newcenter, rtol=0, atol=0)
                torch.testing.assert_close(left[:boundary], right[:boundary], rtol=0, atol=0)
                for k in ('m0', 'P0', 'K0'):
                    torch.testing.assert_close(baseline[a][k][:boundary], changed_baseline[a][k][:boundary], rtol=0, atol=0)
                assert abs(float(ref.mean())) < 1e-14
                delta, _, _ = model.fields(a, D, T[a], D['n'])
                m, P = model.corrected_state(a, delta)
                daily = model.processes[a].future_daily_means(m, P)
                forecast = model.processes[a].forecast(m, P)
                torch.testing.assert_close(forecast, torch.stack([daily[:, :h].mean(1) for h in B.HORIZONS], 1))
                assert (daily > 0).all()
                if mode != 'current_only':
                    torch.testing.assert_close(m, baseline[a]['m0'] + delta, rtol=0, atol=0)
        grad_T = {a: {**T[a], 'F17': T[a]['F17'].clone().requires_grad_()} for a in ASSETS}
        loss, info = objective(model, datasets, grad_T, detach_every=16)
        torch.testing.assert_close(loss, (info['losses'][ASSETS[0]] + info['losses'][ASSETS[1]]) / 2 + info['penalty'])
        grads = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True)
        reference_loss = 0
        for a in ASSETS:
            D = datasets[a]; _, ref, _ = model.fields(a, D, grad_T[a], D['cut'], 16)
            reference_loss = reference_loss + .5 * (info['losses'][a] + LAMBDA * ref.square().mean())
        expected_grads = torch.autograd.grad(reference_loss, tuple(model.parameters()), retain_graph=True)
        for x, y in zip(grads, expected_grads):
            torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-12)
        loss.backward()
        for a in ASSETS:
            gradient = grad_T[a]['F17'].grad
            assert gradient[:datasets[a]['cut']].abs().sum() > 0
            assert (gradient[datasets[a]['cut']:] == 0).all()
        for a in ASSETS:
            changed_T[a]['F17'][datasets[a]['cut']:] += 99
            changed_T[a]['y'][datasets[a]['va']] *= 1e6
        newloss, _ = objective(model, datasets, changed_T, 16)
        torch.testing.assert_close(loss.detach(), newloss.detach(), rtol=0, atol=0)
        class CountingAdam(torch.optim.Adam):
            calls = 0
            def step(self, *args, **kwargs):
                self.calls += 1
                return super().step(*args, **kwargs)
        optimizer = CountingAdam(model.parameters(), lr=LR)
        from unittest.mock import patch
        with patch('torch.autograd.backward', wraps=torch.autograd.backward) as backward:
            train_info = train_epoch(model, datasets, T, optimizer)
        assert backward.call_count == 1
        assert optimizer.calls == train_info['optimizer_steps'] == 1
        assert train_info['backward_calls'] == 1
        for k, v in model.named_buffers():
            assert torch.equal(v, buffers[k])
        assert all(torch.equal(baseline[a][k], getattr(model, f'{k}_{a}')) for a in ASSETS for k in baseline[a])
    checks.update(future_mutation_prefix_and_bankcenter_exact=True, frozen_baseline_and_dynamics_unchanged=True,
        direct_readout_no_K=True, train_has_no_validation_feature_or_target_gradient=True,
        differentiable_equal_asset_horizon_data_and_bank_penalty=True, one_combined_backward_and_one_Adam_step=True,
        all_horizons_same_positive_dailycurve=True, zero_range_baseline_skipped=True)
    gru = _nonzero(DirectControls(datasets, parameters, baseline, 'shared_GRU', 33))
    for a in ASSETS:
        feature = T[a]['F17'].clone().requires_grad_(); n = datasets[a]['cut']
        full = gru.raw_sequence(a, feature, n, 0); chunked = gru.raw_sequence(a, feature, n, 16)
        torch.testing.assert_close(chunked, full, rtol=1e-12, atol=1e-14)
        changed = feature.detach().clone(); changed[24:] += 8
        torch.testing.assert_close(gru.raw_sequence(a, changed, n, 0)[:24], full[:24], rtol=0, atol=0)
        g = torch.autograd.grad(chunked[20], feature, retain_graph=True)[0]
        assert (g[:16] == 0).all() and g[16:21].abs().sum() > 0 and (g[21:] == 0).all()
        gfull = torch.autograd.grad(full[20], feature)[0]
        assert gfull[:16].abs().sum() > 0
        K = getattr(gru, f'K0_{a}')[16:n]; dev = getattr(gru, f'm0_{a}')[16:n] - gru.processes[a].mu
        x = torch.cat([feature.detach()[16:n], K[:, None], dev[:, None]], 1)
        out, _ = gru.gru(x[None], torch.zeros(1, 1, 8, dtype=DTYPE))
        reset = .5 * torch.tanh(gru.head(out[0]).squeeze(-1))
        assert not torch.allclose(reset[:3], chunked[16:19], rtol=1e-8, atol=1e-10)
    checks.update(GRU_causal_raw_outputs=True, GRU_boundary_hidden_forward_continuous=True,
        GRU_boundary_detaches_gradients_only=True, market_files_opened=False, stage16_results_opened=False,
        real_training_executed=False, parameter_counts=dict(current_only=41, direct_readout=41, shared_GRU=369))
    return checks


def source_hashes():
    sources = dict(runner=Path(__file__), frozen_moments=Path(S18.__file__), shared_prepare=Path(S17.__file__),
        frozen_filter=Path(B.__file__), proxy_frame=Path(S17.PROXY.__file__), range_clock=Path(B.R14.__file__),
        prefix=Path(S18.KP.__file__))
    return {name: dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for name, p in sources.items()}


def run_development(tag):
    """Explicit later review gate; default/check-only execution never enters."""
    assert re.fullmatch(r'round18_direct_controls_[A-Za-z0-9_-]+', tag)
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
    meta = dict(config=dict(modes=MODES, seeds=SEEDS, lr=LR, regularization=LAMBDA, hidden=8, chunk=CHUNK,
        max_epochs=MAX_EPOCHS, patience=PATIENCE, min_epochs=MIN_EPOCHS),
        status='REUSED 2025 DEVELOPMENT; no holdout targets; seeds are not independent datasets', hashes=hashes,
        stage16_checkpoints=provenance, audit=audit, baseline_matching=matching, frozen_parameters=parameters,
        fits={}, scores={}, joint_scores={}, checks=synthetic_checks(),
        training='train-prefix outputs+bank centers; one data+penalty backward; Adam .01 clip2; GRU detach128 only',
        features='MLP exactly F17; GRU F17 plus frozen K0 and m0-mu; no new observations',
        center='per-asset pre2025 bank; differentiable train center, never val mean',
        forecast='S18.FrozenProcess.forecast(corrected_m,P0), same positive exp(m+P/2) dailycurve',
        selection='unique joint epoch including0; stage16 also selected on reused2025; no causal identification')
    for a in ASSETS:
        D = datasets[a]; q = D['origin'][D['va']]
        arrays.update({f'{a}_date': D['date'][q], f'{a}_origin': q, f'{a}_label_end': D['date'][q + 22],
            f'{a}_target': D['y'][D['va']], f'{a}_bank_date': D['date'][D['bank_index']], f'{a}_bank_F17': D['F17'][D['bank_index']]})
        arrays.update({f'{a}_{k}': v.numpy() for k, v in baseline[a].items()})
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
            perasset[a] = dict(known_scale_QLIKE=float((np.log(v) + y / v - np.log(scale)[:, None]).mean()), MSE=float(((v - y) ** 2).mean()))
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
    ap.add_argument('--tag', default='round18_direct_controls_development')
    args = ap.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.run_development:
        run_development(args.tag)
    else:
        print(json.dumps(dict(checks=synthetic_checks(), source_hashes=source_hashes()), indent=2), flush=True)


if __name__ == '__main__':
    main()
