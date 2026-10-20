"""Frozen, finite experiment runner; private per-origin outputs stay separate."""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from data import H, HORIZONS, SPECS, Series, load_series, make_case

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
NEURAL = ['local_profile', 'uniform_profile', 'aux_one', 'aux_five', 'profile',
          'transport', 'transport_uniform', 'covariance_profile', 'diagonal_profile',
          'local_gru', 'pooled_gru']
MODELS = NEURAL + ['ridge', 'rw']
SEEDS = [17, 29, 43]
DEFAULT = dict(max_steps=600, learning_rate=.01, meta_learning_rate=.1,
               ridge_lambda=.1, validation_interval=10, patience=10, min_delta=1e-6,
               episode_seed=17018, feature_dim=8, hidden=32, dtype='float64',
               seed=17, gru_learning_rate=.003, gru_weight_decay=1e-4,
               gru_hidden=16, source_support=64, source_query=32)


def baseline_config(config):
    return {'lr': config['gru_learning_rate'], 'weight_decay': config['gru_weight_decay'],
            'max_steps': config['max_steps'], 'batch_size': 64,
            'validation_every': config['validation_interval'], 'patience': config['patience'],
            'min_delta': config['min_delta'], 'sampler_seed': config['episode_seed']}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(x):
    if isinstance(x, dict):
        return {str(k): serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serial(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def add_schedule(case, max_steps):
    rng = np.random.default_rng(17018)
    schedule, hasher = [], hashlib.sha256()
    for step in range(max_steps):
        item = {}
        for name, batch in case['sources'].items():
            q0 = int(rng.integers(64+H, len(batch['origin'])-32+1))
            s, q = np.arange(q0-H-64, q0-H), np.arange(q0, q0+32)
            if batch['origin'][s[-1]]+H >= batch['origin'][q[0]]:
                raise AssertionError('Source episode purge failed')
            item[name] = {'support': s, 'query': q}
            hasher.update(name.encode()); hasher.update(s.astype('<i8').tobytes()); hasher.update(q.astype('<i8').tobytes())
        schedule.append(item)
    case['source_episodes'] = schedule
    case['receipt']['episode_schedule_sha256'] = hasher.hexdigest()
    case['receipt']['episode_seed'] = 17018
    return case


def scores(prediction, batch):
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != batch['delta'].shape or not np.isfinite(prediction).all():
        raise ValueError('Invalid scaled incremental forecast')
    residual = prediction * batch['sigma'] - batch['delta']
    mse, rw = np.mean(residual**2, axis=0), np.mean(batch['delta']**2, axis=0)
    if np.any(rw <= 0):
        raise ValueError('Degenerate RW denominator; no metric floors permitted')
    ratio = mse/rw
    return {'score': float(np.mean(ratio[HORIZONS])),
            'horizons': [{'h': int(h+1), 'mse': float(mse[h]), 'rw_mse': float(rw[h]),
                          'rmse': float(np.sqrt(mse[h])), 'relative_mse': float(ratio[h]),
                          'mae': float(np.mean(np.abs(residual[:,h])))} for h in HORIZONS],
            'n_origins': len(prediction), 'all20_relative_mse': ratio.tolist()}


def fit_ridge(case):
    # A strong conventional baseline can use every target supervised training
    # window, including the20/80 meta windows counted in the declared budget.
    sx = np.concatenate([case['target_support']['x'], case['target_meta']['x']]).reshape(-1,60)
    sy = np.concatenate([case['target_support']['y'], case['target_meta']['y']])
    vx = case['target_val']['x'].reshape(-1,60)
    tx = case['target_test']['x'].reshape(-1,60)
    # Add and regularize an intercept; inputs/scalers are causal and identical
    # to neural methods. It is not a newly engineered feature baseline.
    sx = np.column_stack((sx, np.ones(len(sx))))
    vx = np.column_stack((vx, np.ones(len(vx))))
    tx = np.column_stack((tx, np.ones(len(tx))))
    c, b = sx.T@sx/len(sx), sx.T@sy/len(sx)
    candidates = []
    best = None
    for a in (.01, .1, 1., 10., 100.):
        w = np.linalg.solve(c + a*np.eye(sx.shape[1]), b)
        vp = vx@w
        score = scores(vp, case['target_val'])['score']
        candidates.append({'alpha': a, 'validation_score': score})
        if best is None or score < best[0]:
            best = (score, a, w.copy(), vp.copy())
    return {'predictions': tx@best[2], 'val_predictions': best[3], 'state': {'w': best[2]},
            'traces': candidates,
            'receipt': {'selected_alpha': best[1], 'ridge_fits': len(candidates), 'support_plus_meta_n': len(sx),
                        'validation_score': best[0], 'test_forward_calls': 1}}


def worker(case, model, config, public_dir, private_dir):
    import torch
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float64)
    job = f"{case['target']}_b{case['budget']}_{model}_s{config['seed']}"
    started, t0 = utc(), time.perf_counter()
    try:
        if model == 'rw':
            result = {'predictions': np.zeros_like(case['target_test']['y']),
                      'val_predictions': np.zeros_like(case['target_val']['y']),
                      'state': {}, 'traces': [], 'receipt': {'fits': 0, 'test_forward_calls': 0}}
        elif model == 'ridge':
            result = fit_ridge(case)
        elif model.endswith('_gru'):
            from baselines import fit_baseline
            result = fit_baseline(case, model, config['seed'], baseline_config(config))
        else:
            from learning import fit
            result = fit(case, model, config['seed'], config)
        if result.get('receipt', {}).get('eligible') is False:
            raise RuntimeError('Learner declared ineligible: '+str(result['receipt'].get('failure') or result['receipt'].get('status')))
        pred = np.asarray(result['predictions'], dtype=np.float64)
        evaluation = scores(pred, case['target_test'])
        valpred = np.asarray(result['val_predictions'], dtype=np.float64)
        validation = scores(valpred, case['target_val'])
        private = Path(private_dir)
        private.mkdir(parents=True, exist_ok=True)
        prediction_path = private/f'{job}_PREDICTIONS.npz'
        np.savez_compressed(prediction_path, prediction=pred, delta=case['target_test']['delta'],
            sigma=case['target_test']['sigma'], origin=case['target_test']['origin'], date=case['target_test']['date'],
            last_label_date=case['target_test']['last_label_date'],
            val_prediction=valpred, val_delta=case['target_val']['delta'], val_sigma=case['target_val']['sigma'],
            val_origin=case['target_val']['origin'], val_date=case['target_val']['date'],
            val_last_label_date=case['target_val']['last_label_date'])
        state_path = private/f'{job}_STATE.pt'
        torch.save(result['state'], state_path)
        full_trace_path = private/f'{job}_TRACE.json'
        write_json(full_trace_path, result['traces'])
        # Preserve full traces privately, and publish all validation checkpoints
        # plus any model-independent scalar summaries of a step.
        traces = [r for r in result['traces'] if 'step' not in r or int(r['step']) % 10 == 0]
        summary = {'job': job, 'status': 'COMPLETE', 'model': model, 'target': case['target'],
            'budget': case['budget'], 'seed': config['seed'], 'started_utc': started,
            'completed_utc': utc(), 'wall_seconds': time.perf_counter()-t0,
            'data_receipt': case['receipt'], 'evaluation': evaluation, 'validation': validation,
            'fit_receipt': result['receipt'], 'validation_traces': traces,
            'private_prediction_sha256': sha(prediction_path), 'private_state_sha256': sha(state_path),
            'private_full_trace_sha256': sha(full_trace_path)}
        write_json(Path(public_dir)/f'{job}_SUMMARY.json', summary)
        return {'job': job, 'status': 'COMPLETE', 'wall_seconds': summary['wall_seconds']}
    except Exception as exc:
        import traceback
        failure = {'job': job, 'status': 'FAILED', 'started_utc': started, 'completed_utc': utc(),
                   'wall_seconds': time.perf_counter()-t0, 'exception': repr(exc), 'traceback': traceback.format_exc()}
        if 'result' in locals() and isinstance(result, dict):
            failure['fit_receipt'] = result.get('receipt')
            if result.get('traces') is not None:
                trace_path = Path(private_dir)/f'{job}_FAILED_TRACE.json'
                write_json(trace_path, result['traces'])
                failure['private_failure_trace_sha256'] = sha(trace_path)
        write_json(Path(public_dir)/f'{job}_FAILED.json', failure)
        return failure


def summarize(public_dir):
    rows = [json.loads(p.read_text()) for p in sorted(Path(public_dir).glob('*_SUMMARY.json'))]
    failures = [json.loads(p.read_text()) for p in sorted(Path(public_dir).glob('*_FAILED.json'))]
    grouped = []
    for model in MODELS:
        selected = [r for r in rows if r['model'] == model]
        if not selected:
            continue
        grouped.append({'model': model, 'score': float(np.mean([r['evaluation']['score'] for r in selected])),
                        'budget100': float(np.mean([r['evaluation']['score'] for r in selected if r['budget']==100])),
                        'budget400': float(np.mean([r['evaluation']['score'] for r in selected if r['budget']==400])),
                        'jobs': len(selected)})
    lookup = {r['model']: r['score'] for r in grouped}
    expected_jobs = 10*(len(NEURAL)*len(SEEDS)+2)
    gate = {'all_jobs_completed': not failures and len(rows)==expected_jobs}
    if all(k in lookup for k in ('transport','profile','covariance_profile','rw')):
        gate.update({'beats_profile_by_half_percent': lookup['transport'] < .995*lookup['profile'],
                     'beats_covariance_by_half_percent': lookup['transport'] < .995*lookup['covariance_profile'],
                     'does_not_lose_to_rw': lookup['transport'] <= 1.})
    gate['advance'] = len(gate)==4 and all(gate.values())
    result = {'completed_utc': utc(), 'completed_jobs': len(rows), 'failed_jobs': len(failures),
              'ranking': sorted(grouped, key=lambda x:x['score']), 'resource_gate': gate,
              'meaning': 'exposed2022 development; three neural seeds, static forecasts executed once percase; nested supervision budget; no independence or publishability claim'}
    write_json(Path(public_dir)/'AGGREGATE.json', result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('prepare','run','summarize'))
    p.add_argument('--eex-dev', type=Path)
    p.add_argument('--protocol', type=Path, default=STUDY/'provenance/PROTOCOL.json')
    p.add_argument('--public-dir', type=Path, default=STUDY/'results/D1')
    p.add_argument('--private-dir', type=Path, required=True)
    p.add_argument('--year', type=int, default=2022)
    args = p.parse_args()
    if args.mode == 'summarize':
        print(json.dumps(summarize(args.public_dir), ensure_ascii=False, indent=2)); return
    if args.eex_dev is None:
        p.error('--eex-dev must point to the exact approved development CSV')
    config = DEFAULT.copy()
    if args.mode == 'run':
        protocol = json.loads(args.protocol.read_text())
        for path, expected in protocol['source_sha256'].items():
            if sha(STUDY/path) != expected:
                raise ValueError('Frozen source changed: '+path)
        config.update(protocol['config'])
        if protocol['year'] != args.year or protocol['models'] != MODELS or protocol['seeds'] != SEEDS:
            raise ValueError('Unfrozen execution task')
    series = load_series(REPO, args.eex_dev)
    cases = [add_schedule(make_case(series,name,budget,args.year),config['max_steps'])
             for name,_,_ in SPECS for budget in (100,400)]
    for name,_,_ in SPECS:
        pair=[c for c in cases if c['target']==name]
        assert pair[0]['receipt']['sources']==pair[1]['receipt']['sources']
        assert pair[0]['receipt']['episode_schedule_sha256']==pair[1]['receipt']['episode_schedule_sha256']
    if args.mode == 'prepare':
        out={'created_utc':utc(),'cases':[c['receipt'] for c in cases], 'new_shenzhen_read':False,
             'rows':{k:len(v.price) for k,v in series.items()},'price_diagnostics_or_model_fits':False}
        write_json(STUDY/'provenance/DATA_PREPARATION.json',out)
        print(json.dumps({'rows':out['rows'],'cases':len(cases),'status':'prepared causal features; no fits'},ensure_ascii=False));return
    if args.public_dir.exists() and any(args.public_dir.iterdir()):
        raise ValueError('Refuse an existing run directory; no silent resume/retry')
    args.public_dir.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for case in cases:
        for model in MODELS:
            for seed in (SEEDS if model in NEURAL else [17]):
                fit_config=dict(config,seed=seed)
                jobs.append((case,model,fit_config))
    write_json(args.public_dir/'STARTED.json',{'utc':utc(),'protocol_sha256':sha(args.protocol),
        'jobs':len(jobs),'config':config,'seeds':SEEDS,'source_sha256':protocol['source_sha256']})
    t0=time.perf_counter(); completed=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(worker,c,m,cfg,str(args.public_dir),str(args.private_dir)) for c,m,cfg in jobs]
        for f in concurrent.futures.as_completed(futures):
            r=f.result();completed.append(r)
            print(json.dumps({'completed':len(completed),'total':len(futures),'job':r['job'],'status':r['status'],
                              'wall_seconds':round(time.perf_counter()-t0,2)}),flush=True)
    summary=summarize(args.public_dir)
    write_json(args.public_dir/'COMPLETE.json',{'utc':utc(),'wall_seconds':time.perf_counter()-t0,
        'completed':len(completed),'failed':summary['failed_jobs'],'market_retries':0,
        'protocol_sha256':sha(args.protocol),'runtime_source_unchanged':all(sha(STUDY/k)==v for k,v in protocol['source_sha256'].items())})
    print(json.dumps({'status':'COMPLETE','jobs':len(completed),'failures':summary['failed_jobs']},ensure_ascii=False))


if __name__ == '__main__':
    main()
