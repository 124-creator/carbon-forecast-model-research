# -*- coding: utf-8 -*-
"""Prefix-forest training-query priors; preserve full-phase forecast references.

Only forest_prior_honest_round5* is written. Keeps old phase NN features, scaler,
phi, banks and all post-boundary references unchanged. Four fixed time blocks;
no per-query full-parameter vintage or invalidity claim about previous models.
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
import torch
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
import forest_prior_round5 as F

P = Path(__file__).resolve().parent
TAG = 'forest_prior_honest_round5'
FRACTIONS = [0., .25, .5, .75, 1.]
SEEDS = F.SEEDS
GAMMA = F.GAMMA


def tensor_digest(value):
    if torch.is_tensor(value):
        return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return hashlib.sha256(repr(value).encode('utf-8')).hexdigest()


def immutable_phase_digest(phase):
    out = {'x': tensor_digest(phase['x']),
           'scaler_mean': tensor_digest(phase['scaler']['mean']),
           'scaler_std': tensor_digest(phase['scaler']['std']),
           'fit_indices': tensor_digest(phase['fit_D_indices'])}
    keys = ['z', 'phi', 'mature', 'support', 'zero_pos', 'p', 'n', 'uniform',
            'allowed', 'count', 'bank_D_indices', 'bank_t', 'bank_horizon',
            'p_static', 'valid_sid', 'eligible_sid', 'phi2']
    for hi, b in enumerate(phase['banks']):
        for key in keys:
            out[f'{hi}_{key}'] = tensor_digest(b[key])
    return out


def load_cache(market, phase, D=None):
    cache = torch.load(P/f'{TAG}_{market}.pt', map_location='cpu', weights_only=False)
    if D is not None:
        for key in ['t', 'hi', 'y', 'scale', 'tr', 'va', 'rf', 'te']:
            assert np.array_equal(cache['rows'][key].numpy(), D[key])
        times, inverse = np.unique(D['t'], return_inverse=True)
        assert np.array_equal(cache['times'].numpy(), times)
        assert np.array_equal(cache['row_sid'].numpy(), inverse)
    return cache['phases'][phase]


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    old_names = ['forest_prior_round5.py', 'forest_prior_round5.json', 'qrf_round4.py',
                 'qrf_round4.json', 'empirical_tilt_round3.py', 'seq_transport_round4.py']
    old_names += [f'forest_prior_round5_{market}.pt' for market in F.T.B.CUT]
    frozen = {name: hashlib.sha256((P/name).read_bytes()).hexdigest() for name in old_names}
    old_report = json.loads((P/'forest_prior_round5.json').read_text(encoding='utf-8'))
    qrf = json.loads((P/'qrf_round4.json').read_text(encoding='utf-8'))
    manifest = dict(status='model training-reference alignment experiment; not a claim previous evaluation invalid',
        fractions=FRACTIONS, seeds=SEEDS, gamma=GAMMA, forest_params=F.Q.RF_PARAMS,
        blocks='start+floor(f*(phase_boundary-start)); queries in [b_j,b_j+1)',
        prefix='phase fitids with each row t_i+h_i<block_start; scalar normalized z; old selected leaf',
        prefix_scaler='mean/std+1e-5 on all-h prefixfit raw x_aug, clipped [-8,8]',
        preserved='all phase NN scaler/x/phi/support/fitindices/mature/allowed/count; all query t>=phase boundary reference values',
        uniform='currently matured full-phase NZ support, may include labels matured after block_start but no later than query origin',
        coldstart='no prefix or <=1 row per h -> matured NZ uniform; no legal NN training query dropped',
        boundary_scope='only forest training-query references prefix-fitted; NN phase parameters/scaler/phi remain ordinary full-fit',
        frozen_sha256_before=frozen, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), markets={})
    for market in F.T.B.CUT:
        manifest['markets'][market] = {}
        for phase in ['inner', 'final']:
            start = min(v['first_fit_origin'] for v in old_report['markets'][market]['phases'][phase].values())
            boundary = qrf['markets'][market]['audit']['validation_start_index' if phase == 'inner' else 'cut_index']
            bounds = (start+np.floor(np.array(FRACTIONS)*(boundary-start)).astype(int)).tolist()
            manifest['markets'][market][phase] = dict(start=start, boundary=boundary, boundaries=bounds)
    (P/f'{TAG}_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    report = dict(manifest=manifest, markets={}, forest_fits_completed=0)
    fits = 0
    with threadpool_limits(limits=2):
        for market in F.T.B.CUT:
            source_name = f'forest_prior_round5_{market}.pt'
            cache = torch.load(P/source_name, map_location='cpu', weights_only=False)
            D = {key: value.numpy() for key, value in cache['rows'].items()}
            times = cache['times'].numpy();inverse = cache['row_sid'].numpy()
            rawx = np.ascontiguousarray(D['x_aug']);z = D['z'].astype(np.float32)
            ends = D['t']+F.T.HOR[D['hi']]
            cache['source_cache_file'] = source_name
            cache['source_cache_sha256'] = frozen[source_name]
            cache['honest_reference_policy'] = manifest
            cache['honest_blocks'] = {}
            info = dict(phases={}, source_cache_sha256=frozen[source_name])
            for phase, part in [('inner', 'tr'), ('final', 'rf')]:
                ph = cache['phases'][phase]
                fit = ph['fit_D_indices'].numpy();boundary = int(ph['boundary'])
                start = int(D['t'][D[part]].min())
                bounds = start+np.floor(np.array(FRACTIONS)*(boundary-start)).astype(int)
                assert bounds.tolist() == manifest['markets'][market][phase]['boundaries']
                assert np.all(ends[fit] < boundary)
                digest_before = immutable_phase_digest(ph)
                keep = torch.tensor(np.flatnonzero(times >= boundary))
                protected_keys = ['u', 'forest', 'u_forest', 'm0', 'v0', 'fallback']
                protected = [{key: b[key][keep].clone() for key in protected_keys} for b in ph['banks']]
                original_u_hash = [tensor_digest(b['u']) for b in ph['banks']]
                for b in ph['banks']:
                    b['old_fullphase_fallback'] = b['fallback'].clone()
                    b['honest_query_block_sid'] = torch.full((len(times),), -1, dtype=torch.int64)
                phase_blocks, block_reports = [], []
                query_block_sid = torch.full((len(times),), -1, dtype=torch.int64)
                for j, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
                    left, right = int(left), int(right)
                    block_sid = np.flatnonzero((times >= left)&(times < right))
                    query_block_sid[torch.tensor(block_sid)] = j
                    prefix = fit[ends[fit] < left]
                    assert np.all(ends[prefix] < left)
                    if len(prefix):
                        mean, std = rawx[prefix].mean(0), rawx[prefix].std(0)+1e-5
                        xx = np.clip((rawx-mean)/std, -8, 8)
                        scaler = {'mean': torch.tensor(mean), 'std': torch.tensor(std)}
                    else:
                        xx, scaler = None, None
                    meta = dict(block=j, start=left, stop=right, prefix_fit_D_indices=torch.tensor(prefix),
                        prefix_target_indices=torch.tensor(ends[prefix]), scaler=scaler,
                        query_sid_indices=torch.tensor(block_sid), per_h=[])
                    tiny = dict(block=j, start=left, stop=right, prefix_n=len(prefix), query_n=len(block_sid),
                        prefix_last_target=int(ends[prefix].max()) if len(prefix) else None, per_h=[])
                    local_sid = {int(sid): k for k, sid in enumerate(block_sid)}
                    for hi, h in enumerate(F.T.HOR):
                        h = int(h);b = ph['banks'][hi]
                        bh = prefix[D['hi'][prefix] == hi]
                        query_rows = np.flatnonzero((D['hi'] == hi)&(D['t'] >= left)&(D['t'] < right))
                        qsids = inverse[query_rows]
                        qlocal = np.array([local_sid[int(sid)] for sid in qsids], dtype=int)
                        bank_ids = b['bank_D_indices'].numpy()
                        bank_lookup = {int(i): k for k, i in enumerate(bank_ids)}
                        nz_pos = np.flatnonzero(np.abs(D['y'][bh]) > 1e-12)
                        nz_ids = bh[nz_pos]
                        columns = np.array([bank_lookup[int(i)] for i in nz_ids], dtype=int)
                        assert np.intersect1d(bh, query_rows).size == 0
                        assert np.all(ends[bh] < left)
                        raw = np.zeros((len(block_sid), len(bank_ids)), dtype=np.float32)
                        leaf = int(qrf['markets'][market]['selection'][str(h)]['chosen_minleaf'])
                        trained = len(bh) > 1 and len(query_rows) > 0
                        if trained:
                            for seed in SEEDS:
                                forest = RandomForestRegressor(min_samples_leaf=leaf, random_state=seed, **F.Q.RF_PARAMS)
                                forest.fit(xx[bh], z[bh]);fits += 1
                                weight, error = F.Q.leaf_weights(forest, xx[bh], xx[query_rows], sanity=(seed == 11))
                                if len(columns):
                                    raw[np.ix_(qlocal, columns)] += weight[:, nz_pos]/len(SEEDS)
                            del forest, weight
                        allowed = b['allowed'][torch.tensor(block_sid)].numpy()
                        valid = b['valid_sid'][torch.tensor(block_sid)].numpy()
                        u, forest_only, uniform, count, fallback = F.make_reference(raw, allowed, valid)
                        assert np.array_equal(uniform, b['uniform'][torch.tensor(block_sid)].numpy())
                        bt = torch.tensor(block_sid)
                        b['u'][bt] = torch.tensor(u)
                        b['forest'][bt] = torch.tensor(forest_only)
                        b['u_forest'] = b['forest']
                        m0 = (u.astype(np.float64)*b['z'].numpy()[None, :]).sum(1)
                        v0 = (u.astype(np.float64)*(b['z'].numpy()[None, :]-m0[:, None])**2).sum(1)
                        b['m0'][bt] = torch.tensor(m0, dtype=torch.float32)
                        b['v0'][bt] = torch.tensor(v0, dtype=torch.float32)
                        b['fallback'][bt] = torch.tensor(fallback)
                        b['honest_query_block_sid'][bt] = j
                        assert np.all(u[~allowed] == 0)
                        mass_error = float(np.max(np.abs(u[count > 0].sum(1, dtype=np.float64)-1.))) if np.any(count > 0) else 0.
                        recent = allowed.copy()
                        recent &= b['mature'].numpy()[None, :] >= left
                        recent_uniform_mass = uniform[recent].sum(dtype=np.float64)
                        per_h = dict(horizon=h, leaf=leaf, trained_five_forests=bool(trained),
                            prefix_h_D_indices=torch.tensor(bh), prefix_h_target_indices=torch.tensor(ends[bh]),
                            prefix_NZ_D_indices=torch.tensor(nz_ids), full_bank_columns=torch.tensor(columns),
                            query_D_indices=torch.tensor(query_rows), query_sid_indices=torch.tensor(qsids))
                        meta['per_h'].append(per_h)
                        small = dict(horizon=h, leaf=leaf, prefix_h_n=len(bh), prefix_NZ_n=len(nz_ids),
                            valid_query_n=len(query_rows), n_bank=len(bank_ids), trained_five_forests=bool(trained),
                            last_forest_target=int(ends[bh].max()) if len(bh) else None,
                            no_own_query_or_future_partition_labels=True,
                            uniform_fallback_valid_queries=int((fallback & valid & (count > 0)).sum()),
                            mass_max_error=mass_error, future_mass_exact_zero=True,
                            recent_mature_uniform_total_mass=float(recent_uniform_mass))
                        tiny['per_h'].append(small)
                    phase_blocks.append(meta);block_reports.append(tiny)
                    print(market, phase, 'block', j, 'ready', 'elapsed_s', round(time.perf_counter()-started, 1), flush=True)
                ph['honest_blocks'] = phase_blocks
                ph['query_block_sid'] = query_block_sid
                cache['honest_blocks'][phase] = phase_blocks
                assert torch.equal(query_block_sid[times >= boundary], torch.full_like(query_block_sid[times >= boundary], -1))
                assert torch.all(query_block_sid[times < boundary] >= 0)
                assert digest_before == immutable_phase_digest(ph)
                for hi, b in enumerate(ph['banks']):
                    for key in protected_keys:
                        assert torch.equal(b[key][keep], protected[hi][key]), (market, phase, hi, key)
                    evaluation_rows = D['va' if phase == 'inner' else 'te']
                    assert np.all(D['t'][evaluation_rows] >= boundary)
                    sid_eval = inverse[evaluation_rows[D['hi'][evaluation_rows] == hi]]
                    lookup_keep = {int(s): j for j, s in enumerate(keep.tolist())}
                    ix = torch.tensor([lookup_keep[int(s)] for s in sid_eval])
                    assert torch.equal(b['u'][torch.tensor(sid_eval)], protected[hi]['u'][ix])
                info['phases'][phase] = dict(boundaries=bounds.tolist(), blocks=block_reports,
                    post_boundary_all_reference_fields_bitwise_equal=True,
                    validation_or_test_u_bitwise_equal=True, immutable_NN_and_bank_fields_equal=True,
                    source_fullphase_u_sha256=original_u_hash)
            expected = f'{TAG}_{market}.pt'
            target, temp = (P/expected).resolve(), (P/(expected+'.partial')).resolve()
            assert target.parent == P.resolve() and target.name == expected
            assert temp.parent == P.resolve() and temp.name == expected+'.partial'
            torch.save(cache, temp);os.replace(temp, target)
            info.update(cache_file=target.name, bytes=target.stat().st_size, elapsed_seconds=time.perf_counter()-started)
            report['markets'][market] = info
            report['forest_fits_completed'] = fits
            (P/f'{TAG}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print('CACHE_READY', market, str(target), flush=True)
            del cache, ph, protected, b
    after = {name: hashlib.sha256((P/name).read_bytes()).hexdigest() for name in old_names}
    assert after == frozen
    report.update(frozen_sha256_after=after, elapsed_seconds=time.perf_counter()-started)
    (P/f'{TAG}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('ALL_HONEST_CACHES_READY', report['elapsed_seconds'], 'forest_fits', fits, flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
