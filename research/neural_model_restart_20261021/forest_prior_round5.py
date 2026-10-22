# -*- coding: utf-8 -*-
"""Build inner/final forest priors for the frozen sequence-transport row order.

Rebuild frozen QRF trees only to obtain new training-query predictions. All old
artifacts remain read-only. Each cache bank uses T.make_bank's stable NZ order.
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
TAG = 'forest_prior_round5'
SEEDS = [11, 22, 33, 44, 55]
GAMMA = .05


def nonzero_ids(D, fit, hi):
    ids = fit[D['hi'][fit] == hi]
    ids = ids[np.abs(D['y'][ids]) > 1e-12]
    return ids[np.argsort(D['z'][ids], kind='stable')]


def make_reference(raw, allowed, valid_sid):
    count = allowed.sum(1)
    uniform = allowed.astype(np.float64)/np.maximum(count, 1)[:, None]
    masked = raw.astype(np.float64)*allowed
    mass = masked.sum(1)
    fallback = (mass <= 0)|(np.logical_not(valid_sid))
    forest = np.divide(masked, mass[:, None], out=np.zeros_like(masked), where=mass[:, None] > 0)
    forest[fallback] = uniform[fallback]
    forest[count == 0] = 0.
    u = ((1-GAMMA)*forest+GAMMA*uniform).astype(np.float32)
    forest = forest.astype(np.float32)
    assert np.isfinite(u).all() and np.all(u >= 0) and np.all(u[~allowed] == 0)
    assert np.allclose(u[count > 0].sum(1, dtype=np.float64), 1., atol=2e-6, rtol=0)
    assert np.all(u[count == 0] == 0)
    return u, forest, uniform.astype(np.float32), count, fallback


def gamma0_check(D, hi, row_sid, bank, nzids, forest, raw, oldweights, rule):
    h = int(T.HOR[hi])
    positions = np.flatnonzero(D['hi'][D['te']] == hi)
    rows = D['te'][positions]
    sids = row_sid[rows]
    oldids = oldweights[f'fit_D_row_indices_h{h}']
    lookup = {int(i): j for j, i in enumerate(oldids)}
    cols = np.array([lookup[int(i)] for i in nzids])
    frozen_nz = oldweights[f'ensemble_weights_h{h}'][:, cols]
    raw_error = float(np.max(np.abs(raw[sids]-frozen_nz)))
    assert np.allclose(raw[sids], frozen_nz, atol=3e-8, rtol=1e-6)
    conditioned = frozen_nz.astype(np.float64)
    conditioned /= conditioned.sum(1, keepdims=True)
    conditional_error = float(np.max(np.abs(forest[sids]-conditioned)))
    assert conditional_error < 2e-7
    z, mark = bank['z'], torch.tensor(forest[sids])
    local = torch.tensor(D['local'][rows])
    prob = .8*local+.2*torch.full_like(local, bank['p'])
    zero_pos = int(bank['zero_pos'])
    weight = torch.cat([prob[:, None]*mark[:, :zero_pos], (1-prob)[:, None],
                        prob[:, None]*mark[:, zero_pos:]], 1)
    q = T.quantiles(weight, bank['support']).numpy()*D['scale'][rows, None]
    c = T.crps(torch.tensor(D['z'][rows], dtype=torch.float32), weight, bank['support']).numpy()*D['scale'][rows]
    frozen_q = rule[f"{D['market']}_qrf_rule_atom_Q"][positions]
    frozen_c = rule[f"{D['market']}_qrf_rule_atom_CRPS"][positions]
    frozen_p = rule[f"{D['market']}_qrf_rule_atom_p"][positions]
    crps_error = float(np.max(np.abs(c-frozen_c)))
    assert np.allclose(c, frozen_c, atol=3e-7, rtol=3e-5)
    assert np.allclose(prob.numpy(), frozen_p, atol=2e-7, rtol=0)
    return dict(horizon=h, n=len(rows), original_weight_NZ_Dindex_reorder_max_error=raw_error,
        gamma0_conditional_mark_max_error=conditional_error, gamma0_CRPS_max_error=crps_error,
        gamma0_quantile_max_error=float(np.max(np.abs(q-frozen_q))),
        gamma0_quantile_different_values=int(np.sum(q != frozen_q)),
        gamma0_MQS_mean=float(T.B.score(D['y'][rows], q).mean()),
        frozen_rule_MQS_mean=float(T.B.score(D['y'][rows], frozen_q).mean()),
        gamma0_p_rule_max_error=float(np.max(np.abs(prob.numpy()-frozen_p))))


def load_cache(market, phase, D=None):
    cache = torch.load(P/f'{TAG}_{market}.pt', map_location='cpu', weights_only=False)
    if D is not None:
        for key in ['t', 'hi', 'y', 'scale', 'tr', 'va', 'rf', 'te']:
            assert np.array_equal(cache['rows'][key].numpy(), D[key])
        times, _, inverse = np.unique(D['t'], return_index=True, return_inverse=True)
        assert np.array_equal(cache['times'].numpy(), times)
        assert np.array_equal(cache['row_sid'].numpy(), inverse)
    return cache['phases'][phase]


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source_files = ['qrf_round4.py', 'qrf_round4.json', 'qrf_round4_predictions.npz',
        'qrf_round4_probability_control_predictions.npz', 'empirical_tilt_round3.py',
        'seq_transport_round4.py', 'deep_seq_round4.py', 'marked_event_round2.py', 'marked_event_round1.py', 'neural_restart.py']
    source_files += [f'qrf_round4_{m}_weights.npz' for m in T.B.CUT]
    frozen = {name: hashlib.sha256((P/name).read_bytes()).hexdigest() for name in source_files}
    baseline = json.loads((P/'qrf_round4.json').read_text(encoding='utf-8'))
    rules = np.load(P/'qrf_round4_probability_control_predictions.npz')
    panel = pd.read_csv(T.B.PANEL, parse_dates=['date'])
    report = dict(status='inner/final conditional forest reference caches; no new test tuning',
        seeds=SEEDS, forest_params=Q.RF_PARAMS, gamma=GAMMA, phases={'inner': 'tr', 'final': 'rf'},
        sid='np.unique(D.t,return_inverse=True), identical to seq_transport.prepare; shared across phases/h',
        order='bank_D_indices exactly T.make_bank raw-y nonzero selection then stable argsort float64 D.z; never map duplicates by z value',
        vintage_boundary='forest/scaler/phi use complete allowed fit partition, maturity mask restricts response retrieval only; not all-parameter query-vintage',
        uniform_fallback='only currently matured NZ scenarios; count0 gives all0 and is excluded by count>=32',
        invalid_sid_h='no D row for that h: reference=matured uniform, valid_sid=False, never used for fitting/evaluation',
        frozen_sha256_before=frozen, markets={})
    (P/f'{TAG}_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    fits = 0
    checks = []
    with threadpool_limits(limits=2):
        for market in T.B.CUT:
            d = panel[panel.market == market].sort_values('date').reset_index(drop=True)
            D = T.add_context(d, T.E.build(d));D['market'] = market
            for suffix, value in [('origin', D['t'][D['te']]), ('horizon', T.HOR[D['hi'][D['te']]]),
                                  ('target', D['y'][D['te']]), ('scale', D['scale'][D['te']])]:
                assert np.array_equal(value, rules[market+'_'+suffix])
            times, first, inverse = np.unique(D['t'], return_index=True, return_inverse=True)
            valid = np.zeros((len(times), 4), dtype=bool)
            valid[inverse, D['hi']] = True
            common_rows = {k: torch.tensor(D[k]) for k in ['t', 'hi', 'y', 'scale', 'z', 'local', 'x_state', 'x_aug', 'tr', 'va', 'rf', 'te']}
            cache = dict(schema_version=1, market=market, gamma=GAMMA, times=torch.tensor(times),
                unique_times=torch.tensor(times), origin_times=torch.tensor(times), first=torch.tensor(first), row_sid=torch.tensor(inverse),
                sid=torch.tensor(inverse), valid_sid_h=torch.tensor(valid), rows=common_rows,
                phases={}, frozen_sha256=frozen, parameters=Q.RF_PARAMS, seeds=SEEDS)
            market_info = dict(phases={}, final_test_checks=[])
            for phase, part in [('inner', 'tr'), ('final', 'rf')]:
                fit = D[part]
                boundary = D['audit']['validation_start_index' if part == 'tr' else 'cut_index']
                maturity = D['t'][fit]+T.HOR[D['hi'][fit]]
                assert np.all(maturity < boundary)
                X, banks = T.prepare(D, fit, True)
                xx, z = X['x'].numpy(), X['z'].numpy()
                mean = D['x_aug'][fit].mean(0);std = D['x_aug'][fit].std(0)+1e-5
                phase_cache = dict(phase=phase, part=part, boundary=int(boundary), fit_D_indices=torch.tensor(fit),
                    fit_indices=torch.tensor(fit), scaler={'mean': torch.tensor(mean), 'std': torch.tensor(std)},
                    x=X['x'], times=cache['times'], unique_times=cache['unique_times'], origin_times=cache['times'], sid=cache['row_sid'],
                    row_sid=cache['row_sid'], valid_sid_h=cache['valid_sid_h'], banks=banks)
                phase_info = {}
                for hi, h in enumerate(T.HOR):
                    h = int(h)
                    fit_h = fit[D['hi'][fit] == hi]
                    nzids = nonzero_ids(D, fit, hi)
                    b = banks[hi]
                    assert np.array_equal(b['z'].numpy(), D['z'][nzids].astype(np.float32))
                    assert np.array_equal(b['mature'].numpy(), D['t'][nzids]+h)
                    lookup = {int(i): j for j, i in enumerate(fit_h)}
                    cols = np.array([lookup[int(i)] for i in nzids])
                    query_rows = np.flatnonzero(D['hi'] == hi)
                    qsids = inverse[query_rows]
                    assert len(np.unique(qsids)) == len(qsids)
                    raw = np.zeros((len(times), len(nzids)), dtype=np.float32)
                    leaf = int(baseline['markets'][market]['selection'][str(h)]['chosen_minleaf'])
                    for seed in SEEDS:
                        forest = RandomForestRegressor(min_samples_leaf=leaf, random_state=seed, **Q.RF_PARAMS)
                        forest.fit(xx[fit_h], z[fit_h]);fits += 1
                        ww, error = Q.leaf_weights(forest, xx[fit_h], xx[query_rows], sanity=(seed == 11))
                        raw[qsids] += ww[:, cols]/len(SEEDS)
                    allowed = b['mature'].numpy()[None, :] <= times[:, None]
                    u, forest_only, uniform, count, fallback = make_reference(raw, allowed, valid[:, hi])
                    m0 = (u.astype(np.float64)*b['z'].numpy()[None, :]).sum(1)
                    v0 = (u.astype(np.float64)*(b['z'].numpy()[None, :]-m0[:, None])**2).sum(1)
                    ut, ft, at, ct = torch.tensor(u), torch.tensor(forest_only), torch.tensor(allowed), torch.tensor(count)
                    b.update(u=ut, forest=ft, u_forest=ft,
                        uniform=torch.tensor(uniform), allowed=at, count=ct,
                        mature_mask=at, mature_count=ct,
                        m0=torch.tensor(m0, dtype=torch.float32), v0=torch.tensor(v0, dtype=torch.float32),
                        phi2=b['phi'][:, :2], bank_D_indices=torch.tensor(nzids), D_indices=torch.tensor(nzids),
                        bank_t=torch.tensor(D['t'][nzids]), bank_horizon=h, p_static=float(b['p']),
                        valid_sid=torch.tensor(valid[:, hi]), fallback=torch.tensor(fallback),
                        eligible_sid=torch.tensor(valid[:, hi] & (count >= 32)))
                    phase_info[str(h)] = dict(leaf=leaf, n_fit=len(fit_h), n_bank=len(nzids), n_sid=len(times),
                        n_valid_sid=int(valid[:, hi].sum()), missing_sid_h=int((~valid[:, hi]).sum()),
                        empty_mature_sid=int((count == 0).sum()),
                        valid_zero_leaf_mature_mass_fallback=int((fallback & valid[:, hi] & (count > 0)).sum()),
                        min_eligible_count=int(count[valid[:, hi] & (count >= 32)].min()),
                        mass_max_error=float(np.max(np.abs(u[count > 0].sum(1, dtype=np.float64)-1.))),
                        future_mass_max=float(np.max(np.where(allowed, 0., u))),
                        first_fit_origin=int(D['t'][fit_h].min()), last_fit_target=int((D['t'][fit_h]+h).max()))
                    if phase == 'final':
                        oldw = np.load(P/f'qrf_round4_{market}_weights.npz')
                        check = gamma0_check(D, hi, inverse, b, nzids, forest_only, raw, oldw, rules)
                        check['market'] = market
                        market_info['final_test_checks'].append(check);checks.append(check)
                    print(market, phase, 'h', h, 'ready', 'elapsed_s', round(time.perf_counter()-started, 1), flush=True)
                    del raw, ww, forest
                cache['phases'][phase] = phase_cache
                market_info['phases'][phase] = phase_info
            expected = f'{TAG}_{market}.pt'
            target = (P/expected).resolve();temporary = (P/(expected+'.partial')).resolve()
            assert target.parent == P.resolve() and target.name == expected
            assert temporary.parent == P.resolve() and temporary.name == expected+'.partial'
            torch.save(cache, temporary)
            os.replace(temporary, target)
            market_info.update(cache_file=target.name, bytes=target.stat().st_size,
                               elapsed_seconds=time.perf_counter()-started)
            report['markets'][market] = market_info
            report['forest_fits_completed'] = fits
            (P/f'{TAG}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            pd.DataFrame(checks).to_csv(P/f'{TAG}_baseline_checks.csv', index=False, encoding='utf-8-sig')
            print('CACHE_READY', market, str(target), flush=True)
            del cache, banks, phase_cache
    after = {name: hashlib.sha256((P/name).read_bytes()).hexdigest() for name in source_files}
    assert after == frozen and fits == 200
    report.update(frozen_sha256_after=after, elapsed_seconds=time.perf_counter()-started)
    (P/f'{TAG}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('ALL_CACHES_READY', 'elapsed_seconds', report['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
