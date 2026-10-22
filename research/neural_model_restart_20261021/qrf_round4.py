# -*- coding: utf-8 -*-
"""Standard QRF baseline: mean-RF splits, conditional terminal-leaf ECDF.

Only qrf_round4* outputs. All fitting targets are before the frozen outer cut.
Each tree assigns equal mass to every original fitting row in the query leaf;
bootstrap sampling learns tree partitions, without duplicating response mass.
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
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
import empirical_tilt_round3 as T

P = Path(__file__).resolve().parent
TAG = 'qrf_round4'
SEEDS = [11, 22, 33, 44, 55]
LEAVES = [10, 25, 50]
RF_PARAMS = dict(n_estimators=80, max_depth=6, criterion='squared_error',
                 max_features=1., bootstrap=True, n_jobs=2)
BATCH = 128


def leaf_weights(forest, fit_x, query_x, sanity=False):
    """W = mean_tree 1(query_leaf==fit_leaf)/n_original_fit_rows_in_leaf."""
    fit_leaf = forest.apply(fit_x)
    query_leaf = forest.apply(query_x)
    n, trees = fit_leaf.shape
    nodes = [tree.tree_.node_count for tree in forest.estimators_]
    offsets = np.r_[0, np.cumsum(nodes[:-1])]
    total = int(sum(nodes))
    fit_nodes = (fit_leaf+offsets[None, :]).ravel()
    counts = np.bincount(fit_nodes, minlength=total)
    assert np.all(counts[fit_nodes] > 0)
    membership = sparse.csr_matrix((1./counts[fit_nodes],
        (fit_nodes, np.repeat(np.arange(n), trees))), shape=(total, n))
    out = np.empty((len(query_x), n), dtype=np.float32)
    for start in range(0, len(query_x), BATCH):
        end = min(start+BATCH, len(query_x))
        qnodes = (query_leaf[start:end]+offsets[None, :]).ravel()
        assert np.all(counts[qnodes] > 0)
        query = sparse.csr_matrix((np.full(len(qnodes), 1./trees),
            (np.repeat(np.arange(end-start), trees), qnodes)), shape=(end-start, total))
        out[start:end] = (query@membership).toarray().astype(np.float32)
    assert np.isfinite(out).all() and np.all(out >= 0)
    assert np.allclose(out.sum(1), 1., atol=2e-6, rtol=0)
    error = 0.
    if sanity:
        for i in range(min(4, len(query_x))):
            same = fit_leaf == query_leaf[i:i+1]
            brute = (same/same.sum(0, keepdims=True)).mean(1)
            error = max(error, float(np.max(np.abs(brute-out[i]))))
        assert error < 1e-7
    return out, error


def cdf_crps(y, w, support):
    """Independent exact squared-CDF integral for numerical cross-check."""
    w, support = w.astype(np.float64), support.astype(np.float64)
    cw = w.cumsum(1)
    dz = np.diff(support)
    left = np.clip(np.minimum(support[None, 1:], y[:, None])-support[None, :-1], 0., dz)
    value = (cw[:, :-1]**2*left+(cw[:, :-1]-1)**2*(dz[None, :]-left)).sum(1)
    return value+np.maximum(support[0]-y, 0.)+np.maximum(y-support[-1], 0.)


def ecdf(w, support, z, scale):
    q = np.empty((len(w), len(T.TT)))
    c = np.empty(len(w))
    error = 0.
    for start in range(0, len(w), BATCH):
        end = min(start+BATCH, len(w))
        wt = torch.tensor(w[start:end])
        st = torch.tensor(support)
        yy = torch.tensor(np.asarray(z[start:end], np.float32))
        q[start:end] = T.quantiles(wt, st).numpy()*scale[start:end, None]
        cr = T.crps(yy, wt, st).numpy()*scale[start:end]
        c[start:end] = cr
        independent = cdf_crps(yy.numpy().astype(np.float64), w[start:end], support)*scale[start:end]
        error = max(error, float(np.max(np.abs(independent-cr))))
        assert np.allclose(independent, cr, atol=3e-7, rtol=3e-5)
    zero = np.abs(support) <= 1e-12
    p = 1-w[:, zero].sum(1)
    assert np.all(np.diff(q, axis=1) >= -1e-8) and np.isfinite(q).all()
    assert np.isfinite(c).all() and c.min() > -1e-7
    assert p.min() >= -2e-6 and p.max() <= 1+2e-6
    return (q, c, np.clip(p, 0., 1.)), error


def append_scores(rows, market, D, name, seed, prediction):
    q, c, p = prediction
    y = D['y'][D['te']]
    for hi, h in enumerate(T.HOR):
        take = D['hi'][D['te']] == hi
        yh, qh = y[take], q[take]
        rows.append(dict(market=market, horizon=int(h), model=name, seed=str(seed), n=int(take.sum()),
            MQS=float(T.B.score(yh, qh).mean()), CRPS=float(c[take].mean()) if c is not None else None,
            WIS=float(T.B.wis(yh, qh).mean()), MAE=float(np.abs(yh-qh[:, 7]).mean()),
            coverage80=float(np.mean((yh >= qh[:, 2])&(yh <= qh[:, 12]))),
            coverage95=float(np.mean((yh >= qh[:, 0])&(yh <= qh[:, 14]))),
            Brier=float(np.mean((p[take]-(np.abs(yh) > 1e-12))**2)) if p is not None else None))


def summarize(rows):
    result = []
    lookup = {(r['market'], r['horizon'], r['model'], r['seed']): r for r in rows}
    comparisons = [('qrf', str(s), b, 'fixed') for s in [*SEEDS, 'ensemble']
                   for b in ['tilt_full_5seed', 'uniform_rule', 'uniform_static', 'legacy_empirical', 'gaussian', 'hgb_augmented', 'hgb_state_only']]
    comparisons += [('tilt_full_5seed', 'fixed', 'qrf', 'ensemble')]
    for model, seed, base, base_seed in comparisons:
        selected = [r for r in rows if r['model'] == model and r['seed'] == seed]
        for metric in ['MQS', 'CRPS', 'WIS', 'MAE']:
            pairs = [(r, lookup.get((r['market'], r['horizon'], base, base_seed))) for r in selected]
            pairs = [(r, b) for r, b in pairs if b and r[metric] is not None and b[metric] is not None]
            if not pairs:
                continue
            ratios = np.array([r[metric]/b[metric] for r, b in pairs])
            result.append(dict(model=model, seed=seed, baseline=base, baseline_seed=base_seed, metric=metric,
                cells=len(ratios), wins=int((ratios < 1).sum()), median_ratio=float(np.median(ratios)),
                geometric_gain_pct=float(100*(1-np.exp(np.log(ratios).mean())))))
    return result


def save(report, rows, arrays):
    report['cells'] = rows
    report['summary'] = summarize(rows)
    (P/f'{TAG}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(P/f'{TAG}_cells.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(report['summary']).to_csv(P/f'{TAG}_summary.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(P/f'{TAG}_predictions.npz', **arrays)


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    frozen_files = [P/name for name in ['empirical_tilt_round3.py', 'empirical_tilt_round3.json',
        'empirical_tilt_round3_predictions.npz', 'empirical_tilt_round3_followup.py',
        'empirical_tilt_round3_followup.json', 'empirical_tilt_round3_followup_predictions.npz',
        'marked_event_round2.py', 'marked_event_round1.py', 'neural_restart.py']]
    frozen = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in frozen_files}
    old = np.load(P/'empirical_tilt_round3_followup_predictions.npz')
    panel = pd.read_csv(T.B.PANEL, parse_dates=['date'])
    report = dict(status='standard QRF strong baseline on frozen historical canonical targets',
        algorithm='RandomForest squared-error mean splitting; terminal-leaf conditional empirical distribution; not DRF/MMD splitting',
        formula='w_i(x)=mean_seed mean_tree I(leaf(x)=leaf(x_i))/n_original_fit_rows_in_leaf',
        bootstrap='tree partitions use bootstrap; all original fit rows each contribute once to leaf distribution',
        seeds=SEEDS, forest_params=RF_PARAMS, leaves=LEAVES, selection_seed=11,
        selection='per market/h inner normalized MQS selects minleaf from 10/25/50; test never selects setting or seed',
        input='same T.add_context(d,E.build(d)) 36-column normalized/clipped context; T.prepare inner tr, refit rf',
        target='same standardized z float32; raw scores use exactly same canonical float64 y as frozen 5seed tilt',
        support='all precut fitting z sorted stably including every zero return; repeated support points retained',
        zero_mass='sum weights of all zero labels; no smooth prior replaces empirical zero mass',
        ensemble='five seed conditional probability matrices averaged, then inverse ECDF and exact weighted ECDF CRPS',
        no_test_memory_updates=True, threads=2, weight_batch_queries=BATCH,
        frozen_sha256_before=frozen, panel_sha256=hashlib.sha256(T.B.PANEL.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), markets={})
    rows, arrays = [], {'taus': T.B.TAUS}
    sanity = {'max_leaf_bruteforce_error': 0., 'max_CDF_CRPS_error': 0., 'max_ensemble_weight_error': 0.}
    # Independent uniform one-leaf case: all sample weights must be equal.
    toy_x = np.arange(12, dtype=np.float32)[:, None]
    toy_z = np.array([-2., -1., 0., 0., 0., .5, .5, 1., 1., 2., 3., 4.], np.float32)
    toy = RandomForestRegressor(n_estimators=3, min_samples_leaf=12, bootstrap=False, n_jobs=2, random_state=11).fit(toy_x, toy_z)
    toy_weights, err = leaf_weights(toy, toy_x, np.array([[1.], [7.]], np.float32), True)
    assert np.allclose(toy_weights, 1/12, atol=1e-7)
    sanity['uniform_single_leaf_check'] = True
    sanity['zero_labels_preserved_in_uniform_check'] = bool(np.allclose(toy_weights[:, toy_z == 0].sum(1), .25))
    with threadpool_limits(limits=2):
        for market in T.B.CUT:
            d = panel[panel.market == market].sort_values('date').reset_index(drop=True)
            D = T.add_context(d, T.E.build(d))
            te, key = D['te'], market+'_'
            assert np.array_equal(D['t'][te], old[key+'origin'])
            assert np.array_equal(T.HOR[D['hi'][te]], old[key+'horizon'])
            assert np.array_equal(D['y'][te], old[key+'target'])
            assert np.array_equal(D['scale'][te], old[key+'scale'])
            Xi, _ = T.prepare(D, D['tr'], True)
            Xf, _ = T.prepare(D, D['rf'], True)
            inner, final = Xi['x'].numpy(), Xf['x'].numpy()
            z = Xi['z'].numpy()
            assert inner.shape[1] == 36
            arrays[key+'origin'], arrays[key+'horizon'] = D['t'][te], T.HOR[D['hi'][te]]
            arrays[key+'target'], arrays[key+'scale'] = D['y'][te], D['scale'][te]
            arrays[key+'origin_date'] = old[key+'origin_date']
            predicted = {str(s): [np.empty((len(te), 15)), np.empty(len(te)), np.empty(len(te))] for s in SEEDS}
            predicted['ensemble'] = [np.empty((len(te), 15)), np.empty(len(te)), np.empty(len(te))]
            weight_arrays, selections = {}, {}
            for hi, h in enumerate(T.HOR):
                train = D['tr'][D['hi'][D['tr']] == hi]
                valid = D['va'][D['hi'][D['va']] == hi]
                refit = D['rf'][D['hi'][D['rf']] == hi]
                test_pos = np.flatnonzero(D['hi'][te] == hi)
                test = te[test_pos]
                scores = []
                order_inner = np.argsort(z[train], kind='stable')
                inner_support = z[train][order_inner]
                for leaf in LEAVES:
                    forest = RandomForestRegressor(min_samples_leaf=leaf, random_state=11, **RF_PARAMS)
                    forest.fit(inner[train], z[train])
                    w, err = leaf_weights(forest, inner[train], inner[valid], sanity=(leaf == LEAVES[0]))
                    sanity['max_leaf_bruteforce_error'] = max(sanity['max_leaf_bruteforce_error'], err)
                    prediction, err = ecdf(w[:, order_inner], inner_support, z[valid], np.ones(len(valid)))
                    sanity['max_CDF_CRPS_error'] = max(sanity['max_CDF_CRPS_error'], err)
                    scores.append(dict(leaf=leaf, MQS=float(T.B.score(z[valid], prediction[0]).mean()),
                                       CRPS=float(prediction[1].mean())))
                chosen = min(scores, key=lambda s: s['MQS'])['leaf']
                selections[str(h)] = dict(chosen_minleaf=chosen, inner_scores=scores,
                    n_train=len(train), n_validation=len(valid), n_refit=len(refit), n_test=len(test),
                    final_zero_labels=int(np.sum(np.abs(D['y'][refit]) <= 1e-12)))
                order = np.argsort(z[refit], kind='stable')
                support = z[refit][order]
                weight_arrays[f'support_h{h}'] = support
                weight_arrays[f'fit_D_row_indices_h{h}'] = refit[order]
                weight_arrays[f'fit_origin_h{h}'] = D['t'][refit[order]]
                weight_arrays[f'test_positions_h{h}'] = test_pos
                total = np.zeros((len(test), len(refit)), dtype=np.float32)
                for seed in SEEDS:
                    forest = RandomForestRegressor(min_samples_leaf=chosen, random_state=seed, **RF_PARAMS)
                    forest.fit(final[refit], z[refit])
                    w, err = leaf_weights(forest, final[refit], final[test], sanity=True)
                    sanity['max_leaf_bruteforce_error'] = max(sanity['max_leaf_bruteforce_error'], err)
                    w = w[:, order]
                    prediction, err = ecdf(w, support, z[test], D['scale'][test])
                    sanity['max_CDF_CRPS_error'] = max(sanity['max_CDF_CRPS_error'], err)
                    for j in range(3):
                        predicted[str(seed)][j][test_pos] = prediction[j]
                    weight_arrays[f'seed{seed}_weights_h{h}'] = w
                    total += w/len(SEEDS)
                prediction, err = ecdf(total, support, z[test], D['scale'][test])
                sanity['max_CDF_CRPS_error'] = max(sanity['max_CDF_CRPS_error'], err)
                for j in range(3):
                    predicted['ensemble'][j][test_pos] = prediction[j]
                weight_arrays[f'ensemble_weights_h{h}'] = total
                independent_mean = np.mean([weight_arrays[f'seed{s}_weights_h{h}'] for s in SEEDS], 0)
                error = float(np.max(np.abs(independent_mean-total)))
                sanity['max_ensemble_weight_error'] = max(sanity['max_ensemble_weight_error'], error)
                assert np.allclose(independent_mean, total, atol=3e-8, rtol=1e-6)
                print(market, 'h', h, 'selected_leaf', chosen, 'elapsed_s', round(time.perf_counter()-started, 1), flush=True)
            for seed, prediction in predicted.items():
                append_scores(rows, market, D, 'qrf', seed, prediction)
                arrays[key+f'qrf_{seed}_Q'], arrays[key+f'qrf_{seed}_CRPS'], arrays[key+f'qrf_{seed}_p'] = prediction
            for name, oldkey in [('tilt_full_5seed', 'full_ensemble'), ('uniform_rule', 'uniform_rule'),
                                 ('uniform_static', 'uniform_static'), ('legacy_empirical', 'legacy_empirical'),
                                 ('gaussian', 'gaussian'), ('hgb_augmented', 'hgb_augmented'), ('hgb_state_only', 'hgb_state_only')]:
                q = old[key+oldkey+'_Q']
                c = old[key+oldkey+'_CRPS'] if key+oldkey+'_CRPS' in old else None
                p = old[key+oldkey+'_p'] if key+oldkey+'_p' in old else None
                append_scores(rows, market, D, name, 'fixed', (q, c, p))
                arrays[key+name+'_Q'] = q
                if c is not None:
                    arrays[key+name+'_CRPS'] = c
                if p is not None:
                    arrays[key+name+'_p'] = p
            np.savez_compressed(P/f'{TAG}_{market}_weights.npz', **weight_arrays)
            report['markets'][market] = dict(audit=D['audit'], selection=selections,
                alignment=dict(origin_exact=True, horizon_exact=True, target_exact=True, scale_exact=True,
                               tilt_reference='frozen full_ensemble from five seeds 11/22/33/44/55'))
            report['sanity'] = sanity
            report['elapsed_seconds'] = time.perf_counter()-started
            save(report, rows, arrays)
    after = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in frozen_files}
    assert after == frozen
    report['frozen_sha256_after'] = after
    report['elapsed_seconds'] = time.perf_counter()-started
    save(report, rows, arrays)
    print(json.dumps([s for s in report['summary'] if (s['model'] == 'qrf' and s['seed'] == 'ensemble') or s['model'] == 'tilt_full_5seed'], ensure_ascii=False, indent=2), flush=True)
    print('elapsed_seconds', report['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
