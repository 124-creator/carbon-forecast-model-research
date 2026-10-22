# -*- coding: utf-8 -*-
"""Five-seed, matched-capacity mark-tilt ablation; frozen parent is read-only.

Only empirical_tilt_round3_followup* is written. Each seed is selected once on
purged inner validation and refitted once on all precut labels. Seed mixtures
average full probabilities on a shared bank, never seed scores or quantiles.
"""
from __future__ import annotations
import os
import sys
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
sys.dont_write_bytecode = True
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import empirical_tilt_round3 as T

P = Path(__file__).resolve().parent
PREFIX = 'empirical_tilt_round3_followup'
SEEDS = [11, 22, 33, 44, 55]
VARIANTS = ['full', 'state_only', 'horizon_only']
EPOCHS = 64


class MaskedTilt(T.TiltNet):
    def __init__(self, nf, variant):
        super().__init__(nf, 'tilt_rule')
        self.variant = variant

    def weights(self, x, local, t, bank, causal=False):
        if self.variant == 'horizon_only':
            mask = torch.zeros_like(x)
            mask[:, 26] = x[:, 26]
            x = mask
        return super().weights(x, local, t, bank, causal)


def distribution(D, X, banks, weights):
    te = D['te']
    Q = np.empty((len(te), len(T.TT)))
    C = np.empty(len(te))
    probability = np.empty(len(te))
    for hi, bank in enumerate(banks):
        rows = np.flatnonzero(D['hi'][te] == hi)
        idx = torch.tensor(te[rows])
        w = weights[hi]
        assert torch.isfinite(w).all() and torch.all(w >= 0)
        assert torch.allclose(w.sum(1), torch.ones(len(w)), atol=2e-6, rtol=0)
        scale = D['scale'][te[rows]]
        Q[rows] = T.quantiles(w, bank['support']).numpy()*scale[:, None]
        C[rows] = T.crps(X['z'][idx], w, bank['support']).numpy()*scale
        probability[rows] = 1-w[:, bank['zero_pos']].numpy()
    assert np.isfinite(Q).all() and np.isfinite(C).all() and C.min() > -1e-6
    assert np.all(np.diff(Q, axis=1) >= -1e-8)
    return Q, C, probability


def one_fit(D, variant, seed, Xi, bi, Xf, bf):
    torch.manual_seed(seed)
    net = MaskedTilt(Xi['x'].shape[1], variant)
    info = T.optimize(net, Xi, bi, D['tr'], EPOCHS, seed, torch.tensor(D['va']))
    torch.manual_seed(seed)
    final = MaskedTilt(Xf['x'].shape[1], variant)
    ep = info['best_epoch']
    if ep:
        refit = T.optimize(final, Xf, bf, D['rf'], ep, seed)
    else:
        refit = {'eligible_training_queries': None}
    weights, betas, probabilities = {}, {}, {}
    with torch.no_grad():
        for hi, (ii, w, prob, beta) in T.forward(final, Xf, torch.tensor(D['te']), bf).items():
            weights[hi] = w.detach().clone()
            betas[hi] = beta.detach().numpy()
            probabilities[hi] = prob.detach().numpy()
    info.update(seed=seed, variant=variant, refit_epochs=ep,
                refit_eligible_queries=refit['eligible_training_queries'],
                allocated_parameters=sum(p.numel() for p in final.parameters()))
    return distribution(D, Xf, bf, weights), weights, betas, info


def add_rows(rows, market, D, name, seed, prediction, kind):
    q, c, prob = prediction
    y = D['y'][D['te']]
    hi = D['hi'][D['te']]
    for hidx, h in enumerate(T.HOR):
        keep = hi == hidx
        rows.append(dict(market=market, horizon=int(h), model=name, seed=str(seed), kind=kind,
                         n=int(keep.sum()), MQS=float(T.B.score(y[keep], q[keep]).mean()),
                         CRPS=float(np.mean(c[keep])) if c is not None else None,
                         Brier=float(np.mean((prob[keep]-(np.abs(y[keep]) > 1e-12))**2)) if prob is not None else None))


def store_prediction(out, key, prediction):
    q, c, p = prediction
    out[key+'_Q'] = q
    if c is not None:
        out[key+'_CRPS'] = c
    if p is not None:
        out[key+'_p'] = p


def summarize(rows):
    frame = pd.DataFrame(rows)
    cells = {(r['market'], r['horizon'], r['model'], r['seed']): r for r in rows}
    summaries = []
    refs = [('uniform_rule', 'fixed'), ('uniform_static', 'fixed'),
            ('legacy_empirical', 'fixed'), ('gaussian', 'fixed'),
            ('hgb_augmented', 'fixed'), ('hgb_state_only', 'fixed')]
    for variant in VARIANTS:
        for seed in [*map(str, SEEDS), 'ensemble']:
            selected = [r for r in rows if r['model'] == variant and r['seed'] == seed]
            comparisons = refs + [(v, seed) for v in VARIANTS if v != variant]
            for base, base_seed in comparisons:
                for metric in ['MQS', 'CRPS']:
                    paired = [(r, cells.get((r['market'], r['horizon'], base, base_seed))) for r in selected]
                    paired = [(r, b) for r, b in paired if b and r[metric] is not None and b[metric] is not None]
                    if not paired:
                        continue
                    ratios = np.array([r[metric]/b[metric] for r, b in paired])
                    summaries.append(dict(model=variant, seed=seed, baseline=base, baseline_seed=base_seed,
                        metric=metric, cells=len(ratios), wins=int((ratios < 1).sum()),
                        median_ratio=float(np.median(ratios)),
                        geometric_gain_pct=float(100*(1-np.exp(np.log(ratios).mean())))))
    return summaries


def write_tables(report, rows, arrays):
    report['cells'] = rows
    report['summary'] = summarize(rows)
    (P/f'{PREFIX}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(P/f'{PREFIX}_cells.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(report['summary']).to_csv(P/f'{PREFIX}_summary.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(P/f'{PREFIX}_predictions.npz', **arrays)


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    panel = pd.read_csv(T.B.PANEL, parse_dates=['date'])
    frozen_paths = [P/'empirical_tilt_round3.py', P/'empirical_tilt_round3.json',
                    P/'empirical_tilt_round3_predictions.npz']
    frozen_hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in frozen_paths}
    report = dict(status='exploratory component check on reused historical tests; no novelty claim',
        seeds=SEEDS, variants=VARIANTS, max_epochs=EPOCHS, threads=2,
        fitting='each seed inner selection once and selected-epoch full-precut refit once; epoch0 eligible',
        model='identical 36-input TiltNet 24/16 hidden, 7-output head; only 5 bounded mark coefficients active',
        masks={'full': 'all 36 normalized columns', 'state_only': 'T.prepare aug=False, last8 zero',
               'horizon_only': 'only normalized x index26; all other columns zero'},
        p='same external querylocal rule 0.8*local+0.2*bank.p in all three versions',
        mark='5 coefficients beta=.35*tanh(head[:5]); zero atom explicit',
        selection='T.optimize inner equal-horizon standardized pinball; exact CRPS training with causal maturity mask',
        ensemble='average five complete weights on identical precut support, then recompute Q and exact weighted ECDF CRPS',
        weights_layout='one NPZ per market; shared standardized support_hH and test_positions_hH; each seed/ensemble weights_hH',
        baseline_policy='canonical origin/h/targets fixed; tilt lookback128 versus HGB250; information and input differences are not architecture effects',
        empirical_policy='legacy MQS interpolates sample quantiles; its exact CRPS uses complete ECDF; uniform_static uses coherent inverse ECDF',
        hgb_policy='MQS only; stored float32 targets must equal canonical target cast to same dtype exactly; CRPS unavailable from quantile grid',
        frozen_sha256_before=frozen_hashes, sanity=T.check_crps(), markets={})
    arrays = {'taus': T.B.TAUS}
    rows = []
    hgb_path = P/'context_probe_round3_predictions.npz'
    hgb = np.load(hgb_path) if hgb_path.exists() else None
    previous = np.load(P/'empirical_tilt_round3_predictions.npz')
    with threadpool_limits(limits=2):
        for market in T.B.CUT:
            d = panel[panel.market == market].sort_values('date').reset_index(drop=True)
            D = T.add_context(d, T.E.build(d))
            assert D['x_state'].shape[1] == 28 and D['x_aug'].shape[1] == 36
            assert np.array_equal(D['x_state'][:, 26], np.log(T.HOR[D['hi']])/np.log(20), equal_nan=False) or np.allclose(D['x_state'][:, 26], np.log(T.HOR[D['hi']])/np.log(20), atol=1e-7)
            te = D['te']
            key = market+'_'
            assert np.array_equal(D['t'][te], previous[key+'origin'])
            assert np.array_equal(T.HOR[D['hi'][te]], previous[key+'horizon'])
            assert np.array_equal(D['y'][te], previous[key+'target'])
            arrays[key+'origin'] = D['t'][te]
            arrays[key+'origin_date'] = d.date.to_numpy(dtype='datetime64[D]')[D['t'][te]]
            arrays[key+'horizon'] = T.HOR[D['hi'][te]]
            arrays[key+'target'] = D['y'][te]
            arrays[key+'scale'] = D['scale'][te]
            Xi, bi = T.prepare(D, D['tr'], True)
            Xf, bf = T.prepare(D, D['rf'], True)
            Xis, bis = T.prepare(D, D['tr'], False)
            Xfs, bfs = T.prepare(D, D['rf'], False)
            assert torch.equal(Xi['x'][:, :28], Xis['x'][:, :28])
            assert torch.count_nonzero(Xis['x'][:, 28:]) == 0
            for hi in range(4):
                assert torch.equal(bf[hi]['support'], bfs[hi]['support'])
            weights_out = {}
            for hi, h in enumerate(T.HOR):
                weights_out[f'support_h{h}'] = bf[hi]['support'].numpy()
                weights_out[f'zero_position_h{h}'] = np.array(bf[hi]['zero_pos'])
                weights_out[f'nonzero_mature_h{h}'] = bf[hi]['mature'].numpy()
                weights_out[f'test_positions_h{h}'] = np.flatnonzero(D['hi'][te] == hi)
            fits, diagnostics = {}, {}
            control_predictions = {}
            for mode in ['uniform_static', 'uniform_rule']:
                torch.manual_seed(11)
                control = T.TiltNet(36, mode)
                with torch.no_grad():
                    ww = {hi: w for hi, (ii, w, p, b) in T.forward(control, Xf, torch.tensor(te), bf).items()}
                    prediction = distribution(D, Xf, bf, ww)
                control_predictions[mode] = prediction
                store_prediction(arrays, key+mode, prediction)
                add_rows(rows, market, D, mode, 'fixed', prediction, 'statistical_control')
                for hi, h in enumerate(T.HOR):
                    weights_out[f'{mode}_weights_h{h}'] = ww[hi].numpy()
            q0, c0, p0 = T.E.baseline_scores(D)
            for name, old_name in [('legacy_empirical', 'empirical_constant'), ('gaussian', 'gaussian_constant')]:
                prediction = (q0[old_name], c0[old_name], p0[old_name])
                store_prediction(arrays, key+name, prediction)
                add_rows(rows, market, D, name, 'fixed', prediction, 'legacy_statistical_baseline')
            max_error = float(np.max(np.abs(control_predictions['uniform_static'][1]-c0['empirical_constant'])))
            assert np.allclose(control_predictions['uniform_static'][1], c0['empirical_constant'], rtol=1e-4, atol=1e-6)
            if hgb is not None:
                canonical_keys = list(zip(D['t'][te].tolist(), T.HOR[D['hi'][te]].tolist()))
                hgb_keys = list(zip(hgb[key+'origin'].tolist(), hgb[key+'horizon'].tolist()))
                assert len(set(canonical_keys)) == len(canonical_keys)
                assert len(set(hgb_keys)) == len(hgb_keys) and set(canonical_keys) == set(hgb_keys)
                lookup = {pair: j for j, pair in enumerate(hgb_keys)}
                order = np.array([lookup[pair] for pair in canonical_keys])
                assert np.array_equal(D['t'][te], hgb[key+'origin'][order])
                assert np.array_equal(T.HOR[D['hi'][te]], hgb[key+'horizon'][order])
                stored_target = hgb[key+'target'][order]
                assert np.array_equal(D['y'][te].astype(stored_target.dtype), stored_target)
                hgb_align = dict(origins_exact=True, horizons_exact=True, target_same_dtype_exact=True,
                    reordered_by_unique_origin_horizon=True,
                    target_storage_dtype=str(stored_target.dtype),
                    max_target_storage_error=float(np.max(np.abs(D['y'][te]-stored_target))),
                    compared_on='same canonical float64 target; raw-return MQS, algorithms have different lookback/scale')
                for group in ['augmented', 'state_only']:
                    prediction = (hgb[key+'hgb_'+group][order], None, None)
                    store_prediction(arrays, key+'hgb_'+group, prediction)
                    add_rows(rows, market, D, 'hgb_'+group, 'fixed', prediction, 'existing_HGB_no_retraining')
            else:
                hgb_align = {'available': False}
            for variant in VARIANTS:
                inner, inner_bank, final, final_bank = (Xis, bis, Xfs, bfs) if variant == 'state_only' else (Xi, bi, Xf, bf)
                sums = {hi: torch.zeros_like(torch.tensor(weights_out[f'uniform_rule_weights_h{h}'])) for hi, h in enumerate(T.HOR)}
                fits[variant] = []
                seed_crps = []
                max_rule_p_error = 0.
                for seed in SEEDS:
                    prediction, ww, beta, info = one_fit(D, variant, seed, inner, inner_bank, final, final_bank)
                    fits[variant].append(info)
                    modelkey = key+f'{variant}_seed{seed}'
                    store_prediction(arrays, modelkey, prediction)
                    add_rows(rows, market, D, variant, seed, prediction, 'single_seed')
                    seed_crps.append(prediction[1])
                    max_rule_p_error = max(max_rule_p_error, float(np.max(np.abs(prediction[2]-control_predictions['uniform_rule'][2]))))
                    for hi, h in enumerate(T.HOR):
                        weights_out[f'{variant}_seed{seed}_weights_h{h}'] = ww[hi].numpy()
                        weights_out[f'{variant}_seed{seed}_beta_h{h}'] = beta[hi]
                        sums[hi] += ww[hi]/len(SEEDS)
                    print(market, variant, seed, 'epoch', info['best_epoch'],
                          'elapsed_s', round(time.perf_counter()-started, 1), flush=True)
                with torch.no_grad():
                    mixture = distribution(D, final, final_bank, sums)
                store_prediction(arrays, key+variant+'_ensemble', mixture)
                add_rows(rows, market, D, variant, 'ensemble', mixture, 'probability_mixture')
                for hi, h in enumerate(T.HOR):
                    weights_out[f'{variant}_ensemble_weights_h{h}'] = sums[hi].numpy()
                    means = np.mean([weights_out[f'{variant}_seed{s}_weights_h{h}'] for s in SEEDS], 0)
                    assert np.allclose(means, sums[hi].numpy(), atol=3e-8, rtol=1e-6)
                assert max_rule_p_error < 5e-7
                diagnostics[variant] = dict(max_rule_probability_error=max_rule_p_error,
                    mixture_CRPS_minus_mean_seed_CRPS=float(np.mean(mixture[1]-np.mean(seed_crps, 0))),
                    ensemble_computed_from_weights=True)
            np.savez_compressed(P/f'{PREFIX}_{market}_weights.npz', **weights_out)
            report['markets'][market] = dict(audit=D['audit'], fits=fits,
                uniform_static_exact_CRPS_max_error=max_error, hgb_alignment=hgb_align,
                mixture_diagnostics=diagnostics,
                supports={str(h): len(bf[hi]['support']) for hi, h in enumerate(T.HOR)})
            report['elapsed_seconds'] = time.perf_counter()-started
            write_tables(report, rows, arrays)
            print('saved', market, 'elapsed_s', round(time.perf_counter()-started, 1), flush=True)
    report['frozen_sha256_after'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in frozen_paths}
    assert report['frozen_sha256_after'] == frozen_hashes
    report['elapsed_seconds'] = time.perf_counter()-started
    write_tables(report, rows, arrays)
    selected = [r for r in report['summary'] if r['seed'] == 'ensemble' and r['baseline'] in ['uniform_rule', 'legacy_empirical', 'horizon_only', 'state_only']]
    print(json.dumps(selected, ensure_ascii=False, indent=2), flush=True)
    print('elapsed_seconds', report['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
