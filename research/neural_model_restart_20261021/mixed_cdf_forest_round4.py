# -*- coding: utf-8 -*-
"""Four genuinely fitted mixed-CDF forests; all earlier artifacts are read-only.

Finite CDF-grid supervised coordinates learn partitions only. Prediction uses
conditional leaf probabilities of the original scalar standardized responses.
Five seed probabilities, not quantiles or scores, form every forest ensemble.
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
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
import qrf_round4 as Q
import empirical_tilt_round3 as T

P = Path(__file__).resolve().parent
TAG = 'mixed_cdf_forest_round4'
SEEDS = [11, 22, 33, 44, 55]
VARIANTS = ['per_h_CDF', 'pooled_MSE', 'pooled_CDF', 'pooled_zero_CDF']
PARAMS = dict(n_estimators=80, min_samples_leaf=25, criterion='squared_error',
              max_features=1., bootstrap=True, n_jobs=2)


def cdf_grid(z):
    knots = np.unique(np.quantile(np.asarray(z, np.float64), np.linspace(0., 1., 33)))
    assert len(knots) > 1
    width = np.diff(knots)
    midpoint = (knots[1:]+knots[:-1])/2
    assert np.all(width > 0)
    return dict(knots=knots.tolist(), midpoints=midpoint.tolist(), widths=width.tolist(),
                n_coordinates=len(width), max_width_share=float(width.max()/width.sum()))


def response(z, zero, grid, extra_zero):
    midpoint = np.array(grid['midpoints'], np.float64)
    width = np.array(grid['widths'], np.float64)
    y = ((z[:, None] <= midpoint[None, :])*np.sqrt(width)[None, :]).astype(np.float32)
    if extra_zero:
        y = np.column_stack([y, np.sqrt(.5)*np.asarray(zero, np.float32)])
    assert np.isfinite(y).all()
    return np.ascontiguousarray(y)


def summary(rows):
    lookup = {(r['market'], r['horizon'], r['model'], r['seed']): r for r in rows}
    references = ['qrf_original', 'qrf_rule_atom', 'tilt_full_5seed', 'lstm_qr', 'uniform_rule', 'legacy_empirical']
    pairs = [(v, str(s), b, 'fixed') for v in VARIANTS for s in [*SEEDS, 'ensemble'] for b in references]
    pairs += [(a, 'ensemble', b, 'ensemble') for a, b in [
        ('pooled_CDF', 'pooled_MSE'), ('pooled_zero_CDF', 'pooled_CDF'),
        ('pooled_zero_CDF', 'pooled_MSE'), ('pooled_CDF', 'per_h_CDF')]]
    result = []
    for model, seed, base, base_seed in pairs:
        selected = [r for r in rows if r['model'] == model and r['seed'] == seed]
        for metric in ['MQS', 'CRPS', 'WIS', 'MAE']:
            pairs2 = [(r, lookup.get((r['market'], r['horizon'], base, base_seed))) for r in selected]
            pairs2 = [(r, b) for r, b in pairs2 if b and r[metric] is not None and b[metric] is not None]
            if not pairs2:
                continue
            ratios = np.array([r[metric]/b[metric] for r, b in pairs2])
            result.append(dict(model=model, seed=seed, baseline=base, baseline_seed=base_seed,
                metric=metric, cells=len(ratios), wins=int((ratios < 1).sum()), median_ratio=float(np.median(ratios)),
                geometric_gain_pct=float(100*(1-np.exp(np.log(ratios).mean())))))
    return result


def save(report, rows, arrays):
    report['cells'], report['summary'] = rows, summary(rows)
    (P/f'{TAG}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(P/f'{TAG}_cells.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(report['summary']).to_csv(P/f'{TAG}_summary.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(P/f'{TAG}_predictions.npz', **arrays)


def store(out, key, prediction):
    q, c, p = prediction
    out[key+'_Q'] = q
    if c is not None:
        out[key+'_CRPS'] = c
    if p is not None:
        out[key+'_p'] = p


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    baseline_files = {'qrf': 'qrf_round4_predictions.npz',
        'rule': 'qrf_round4_probability_control_predictions.npz',
        'tilt': 'empirical_tilt_round3_followup_predictions.npz', 'deep': 'deep_seq_round4_predictions.npz'}
    frozen_names = [*baseline_files.values(), 'qrf_round4.py', 'qrf_round4.json',
        'qrf_round4_probability_control.py', 'qrf_round4_probability_control.json',
        'empirical_tilt_round3.py', 'empirical_tilt_round3_followup.py',
        'empirical_tilt_round3_followup.json', 'deep_seq_round4.py', 'deep_seq_round4.json',
        'marked_event_round2.py', 'marked_event_round1.py', 'neural_restart.py']
    frozen_names += [f'qrf_round4_{m}_weights.npz' for m in T.B.CUT]
    frozen = {name: hashlib.sha256((P/name).read_bytes()).hexdigest() for name in frozen_names}
    cached = {k: np.load(P/v) for k, v in baseline_files.items()}
    panel = pd.read_csv(T.B.PANEL, parse_dates=['date'])
    manifest = dict(status='all settings frozen before any fit; model innovation experiment',
        variants=VARIANTS, seeds=SEEDS, forest_params=PARAMS,
        max_depth={'per_h_CDF': 6, 'pooled_MSE': 8, 'pooled_CDF': 8, 'pooled_zero_CDF': 8},
        fitting='fixed settings; all valid precut rows; no hyperparameter search or optional inner diagnostic fits',
        inner_diagnostics=False, expected_final_forests=175,
        context='same T.add_context(E.build(d)) 36 columns; final T.prepare uses rf only, h column26 retained',
        CDF_response='fit-only 33 quantiles at0..1, deduplicate; psi_j=sqrt(delta_j)*I(z<=midpoint_j)',
        zero_response='pooled_zero_CDF appends sqrt(.5)*I(raw_y==0)',
        split_interpretation='squared-error multioutput CDF variance approximates integrated CRPS impurity on finite fit grid; extra zero coordinate adds .5 Brier impurity',
        splitting_precision='sklearn averages output impurities by common output count; that common factor does not change split ordering',
        final_scoring='exact complete scalar weighted-ECDF CRPS, not finite-grid CRPS approximation',
        leaf_probability='original scalar z labels, zero rows retained, uniform original rows within each terminal leaf; bootstrap partitions only',
        pooling='all h rows pooled with their own t_i+h_i<fit boundary; source-h standardized z scaled by query own sigma_t*sqrt(h)',
        comparison='matched pooled_MSE vs pooled_CDF vs pooled_zero_CDF; pooling changes borrowing information and is not solely a zero-module effect',
        budget='n_jobs2/threadpoolctl2, fixed80 trees, no test selection or test-memory appending',
        claim_scope='carbon zero-aware proper-score splitting and multi-horizon borrowing experiment; no new forest family or first CDF/DRF claim',
        frozen_sha256_before=frozen, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        panel_sha256=hashlib.sha256(T.B.PANEL.read_bytes()).hexdigest(), markets={})
    datasets = {}
    for market in T.B.CUT:
        d = panel[panel.market == market].sort_values('date').reset_index(drop=True)
        D = T.add_context(d, T.E.build(d))
        te, key = D['te'], market+'_'
        target_end = D['t']+T.HOR[D['hi']]
        vs, cut = D['audit']['validation_start_index'], D['audit']['cut_index']
        assert np.all(target_end[D['tr']] < vs)
        assert np.all(D['t'][D['va']] >= vs) and np.all(target_end[D['va']] < cut)
        assert np.all(target_end[D['rf']] < cut) and np.all(D['t'][te] >= cut)
        assert np.array_equal(D['rf'], np.flatnonzero(target_end < cut))
        for source in cached.values():
            for suffix, value in [('origin', D['t'][te]), ('horizon', T.HOR[D['hi'][te]]),
                                  ('target', D['y'][te]), ('scale', D['scale'][te])]:
                assert np.array_equal(value, source[key+suffix])
        xf, _ = T.prepare(D, D['rf'], True)
        x, z = np.ascontiguousarray(xf['x'].numpy()), xf['z'].numpy()
        grids = {'pooled': cdf_grid(z[D['rf']])}
        for hi, h in enumerate(T.HOR):
            grids[str(h)] = cdf_grid(z[D['rf'][D['hi'][D['rf']] == hi]])
        raw = D['x_aug']
        manifest['markets'][market] = dict(audit=D['audit'], strict_alignment=True, CDF_grids=grids,
            scaler_mean=raw[D['rf']].mean(0).tolist(), scaler_std=(raw[D['rf']].std(0)+1e-5).tolist(),
            inner_train_last_target=int(target_end[D['tr']].max()),
            refit_last_target=int(target_end[D['rf']].max()),
            h_column=26, pooled_rows=len(D['rf']))
        datasets[market] = (D, x, z)
    (P/f'{TAG}_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('MANIFEST_SAVED_BEFORE_FIRST_FIT', flush=True)
    report = dict(manifest=manifest, markets={}, sanity=dict(max_leaf_bruteforce_error=0.,
        max_independent_CDF_CRPS_error=0., max_probability_mass_error=0., max_CRPS_convexity_error=0.))
    rows, arrays = [], {'taus': T.B.TAUS}
    fit_count = 0
    with threadpool_limits(limits=2):
        for market, (D, x, z) in datasets.items():
            te, rf, key = D['te'], D['rf'], market+'_'
            for suffix, value in [('origin', D['t'][te]), ('horizon', T.HOR[D['hi'][te]]),
                                  ('target', D['y'][te]), ('scale', D['scale'][te])]:
                arrays[key+suffix] = value
            arrays[key+'origin_date'] = cached['qrf'][key+'origin_date']
            mf = {}
            for variant in VARIANTS:
                pooled = variant != 'per_h_CDF'
                groups = [(None, rf, np.arange(len(te)))] if pooled else [
                    (hi, rf[D['hi'][rf] == hi], np.flatnonzero(D['hi'][te] == hi)) for hi in range(4)]
                ensembles, weight_output, details = {}, {}, {}
                single = {str(s): [np.empty((len(te), 15)), np.empty(len(te)), np.empty(len(te))] for s in SEEDS}
                for hi, fitids, positions in groups:
                    targets = D['t'][fitids]+T.HOR[D['hi'][fitids]]
                    assert np.all(targets < D['audit']['cut_index'])
                    queries = te[positions]
                    order = np.argsort(z[fitids], kind='stable')
                    support, sortedids = z[fitids][order], fitids[order]
                    groupkey = 'pooled' if pooled else f'h{T.HOR[hi]}'
                    grid = manifest['markets'][market]['CDF_grids']['pooled' if pooled else str(T.HOR[hi])]
                    target = z[fitids] if variant == 'pooled_MSE' else response(z[fitids],
                        np.abs(D['y'][fitids]) <= 1e-12, grid, variant == 'pooled_zero_CDF')
                    average = np.zeros((len(positions), len(fitids)), dtype=np.float32)
                    seed_crps, seed_hashes, seed_times = [], {}, {}
                    for seed in SEEDS:
                        tick = time.perf_counter()
                        forest = RandomForestRegressor(max_depth=8 if pooled else 6,
                            random_state=seed, **PARAMS)
                        forest.fit(x[fitids], target)
                        fit_count += 1
                        w, error = Q.leaf_weights(forest, x[fitids], x[queries], sanity=True)
                        report['sanity']['max_leaf_bruteforce_error'] = max(report['sanity']['max_leaf_bruteforce_error'], error)
                        w = w[:, order]
                        prediction, error = Q.ecdf(w, support, z[queries], D['scale'][queries])
                        report['sanity']['max_independent_CDF_CRPS_error'] = max(report['sanity']['max_independent_CDF_CRPS_error'], error)
                        mass_error = float(np.max(np.abs(w.sum(1, dtype=np.float64)-1.)))
                        assert mass_error < 2e-6
                        report['sanity']['max_probability_mass_error'] = max(report['sanity']['max_probability_mass_error'], mass_error)
                        for j in range(3):
                            single[str(seed)][j][positions] = prediction[j]
                        seed_crps.append(prediction[1])
                        seed_hashes[str(seed)] = hashlib.sha256(w.tobytes(order='C')).hexdigest()
                        average += w/len(SEEDS)
                        seed_times[str(seed)] = time.perf_counter()-tick
                        print(market, variant, groupkey, seed, 'elapsed_s', round(time.perf_counter()-started, 1), flush=True)
                        del w, forest
                    prediction, error = Q.ecdf(average, support, z[queries], D['scale'][queries])
                    report['sanity']['max_independent_CDF_CRPS_error'] = max(report['sanity']['max_independent_CDF_CRPS_error'], error)
                    convexity = float(np.max(prediction[1]-np.mean(seed_crps, 0)))
                    assert convexity < 3e-7
                    report['sanity']['max_CRPS_convexity_error'] = max(report['sanity']['max_CRPS_convexity_error'], convexity)
                    ensembles[groupkey] = (positions, prediction)
                    weight_output[groupkey+'_support'] = support
                    weight_output[groupkey+'_fit_D_indices'] = sortedids
                    weight_output[groupkey+'_fit_origin'] = D['t'][sortedids]
                    weight_output[groupkey+'_fit_horizon'] = T.HOR[D['hi'][sortedids]]
                    weight_output[groupkey+'_fit_mature'] = D['t'][sortedids]+T.HOR[D['hi'][sortedids]]
                    weight_output[groupkey+'_test_positions'] = positions
                    weight_output[groupkey+'_ensemble_weights'] = average
                    borrowed = {}
                    if pooled:
                        for qhi, qh in enumerate(T.HOR):
                            take = D['hi'][queries] == qhi
                            borrowed[str(qh)] = {str(sh): float(average[take][:, D['hi'][sortedids] == shi].sum(1, dtype=np.float64).mean())
                                                 for shi, sh in enumerate(T.HOR)}
                    details[groupkey] = dict(n_fit=len(fitids), n_query=len(queries),
                        target_dimension=1 if target.ndim == 1 else target.shape[1],
                        source_horizon_probability_by_query_h=borrowed, seed_weight_sha256=seed_hashes,
                        fit_and_predict_seconds=seed_times, last_mature_target=int(targets.max()))
                ensemble = [np.empty((len(te), 15)), np.empty(len(te)), np.empty(len(te))]
                for positions, prediction in ensembles.values():
                    for j in range(3):
                        ensemble[j][positions] = prediction[j]
                for seed, prediction in single.items():
                    Q.append_scores(rows, market, D, variant, seed, prediction)
                    store(arrays, key+variant+'_seed'+seed, prediction)
                Q.append_scores(rows, market, D, variant, 'ensemble', ensemble)
                store(arrays, key+variant+'_ensemble', ensemble)
                np.savez_compressed(P/f'{TAG}_{market}_{variant}_weights.npz', **weight_output)
                mf[variant] = details
                report['markets'][market] = dict(audit=D['audit'], fits=mf)
                report['forest_fits_completed'] = fit_count
                report['elapsed_seconds'] = time.perf_counter()-started
                save(report, rows, arrays)
                del weight_output, ensembles, single
            for name, source, oldkey in [('qrf_original', cached['qrf'], 'qrf_ensemble'),
                ('qrf_rule_atom', cached['rule'], 'qrf_rule_atom'), ('tilt_full_5seed', cached['tilt'], 'full_ensemble'),
                ('lstm_qr', cached['deep'], 'lstm_qr'), ('uniform_rule', cached['tilt'], 'uniform_rule'),
                ('legacy_empirical', cached['tilt'], 'legacy_empirical')]:
                prediction = (source[key+oldkey+'_Q'], source[key+oldkey+'_CRPS'] if key+oldkey+'_CRPS' in source else None,
                    source[key+oldkey+'_p'] if key+oldkey+'_p' in source else None)
                Q.append_scores(rows, market, D, name, 'fixed', prediction)
                store(arrays, key+name, prediction)
            report['elapsed_seconds'] = time.perf_counter()-started
            save(report, rows, arrays)
            print('MARKET_SAVED', market, 'elapsed_s', round(time.perf_counter()-started, 1), flush=True)
    after = {name: hashlib.sha256((P/name).read_bytes()).hexdigest() for name in frozen_names}
    assert after == frozen
    assert fit_count == 175
    report.update(frozen_sha256_after=after, elapsed_seconds=time.perf_counter()-started)
    save(report, rows, arrays)
    print(json.dumps([r for r in report['summary'] if r['seed'] == 'ensemble' and r['baseline'] in ['qrf_original', 'qrf_rule_atom', 'lstm_qr', 'pooled_MSE', 'pooled_CDF', 'per_h_CDF']], ensure_ascii=False, indent=2), flush=True)
    print('elapsed_seconds', report['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
