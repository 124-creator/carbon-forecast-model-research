"""Frozen, checkpointed CPU pilot; refuses overwrites and incomplete rankings."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from rum_core import MODES
from rum_fit import fit, predict
from fair_data import load_prefix

ROOT = Path(__file__).resolve().parent
CUTS = (459, 619, 779, 899)
SEEDS = (17, 29)


def planned_tasks():
    return [(m, s, t) for t in CUTS for s in SEEDS for m in MODES]


def stem(task):
    family, seed, cut = task
    return f'{family}_s{seed}_t{cut}'


def saved_stems(root):
    root = Path(root)
    logs = {p.stem for p in root.glob('*.json')}
    arrays = {p.stem for p in root.glob('*.npz')}
    if logs != arrays or not logs.issubset({stem(t) for t in planned_tasks()}):
        raise ValueError('Unpaired or unexpected fit; do not overwrite/repair silently')
    return logs


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def check_sources():
    pre = json.loads((ROOT / 'PRE_RUN.json').read_text(encoding='utf-8'))
    for name, digest in pre['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen source changed: ' + name)
    return pre


def initialize():
    if (ROOT / 'local_only').exists() or (ROOT / 'PRE_RUN.json').exists():
        raise FileExistsError('Fresh pilot only; existing run cannot be overwritten')
    names = ['rum_core.py', 'rum_fit.py', 'rum_run.py', 'rum_verify.py', 'test_rum.py', 'PROTOCOL.md',
             '../fair_recurrent_tuning_20261016/fair_core.py', '../fair_recurrent_tuning_20261016/fair_data.py']
    price, dates, data_receipt = load_prefix()
    (ROOT / 'local_only' / 'fits').mkdir(parents=True)
    with (ROOT / 'local_only' / 'INPUTS.npz').open('xb') as stream:
        np.savez(stream, price=price, dates=dates)
    started = time.monotonic()
    pre = dict(started_at_utc=datetime.now(timezone.utc).isoformat(), started_monotonic=started,
               deadline_monotonic=started + 1200, budget_seconds=1200, workers=3, batch_limit=6,
               tasks=[list(t) for t in planned_tasks()], formal_expected=32, data=data_receipt,
               source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in names},
               environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                                cuda=torch.cuda.is_available(), dtype='float64', math_threads=1),
               unexposed_test=False, novelty_established=False)
    write_json(ROOT / 'PRE_RUN.json', pre)
    print('FROZEN', len(pre['tasks']), 'fits; absolute budget', pre['budget_seconds'], flush=True)


def run_one(task, deadline):
    family, seed, cut = task
    root = ROOT / 'local_only'
    target = root / 'fits' / stem(task)
    if target.with_suffix('.json').exists() or target.with_suffix('.npz').exists():
        raise FileExistsError('Saved fit must not be repeated')
    with np.load(root / 'INPUTS.npz', allow_pickle=False) as z:
        price = np.array(z['price'], copy=True)
    model, scale, receipt = fit(price, cut, family, seed, deadline=deadline)
    stop = CUTS[CUTS.index(cut) + 1] if cut != CUTS[-1] else 930
    origins = np.arange(cut, stop)
    forecasts, diagnostics = predict(model, scale, price, origins)
    boundary, _ = predict(model, scale, price, np.array([cut - 1]))
    if not np.isfinite(forecasts).all() or time.monotonic() > deadline:
        raise TimeoutError('Nonfinite or over-deadline output cannot complete a fit')
    receipt.update(origin_first=int(origins[0]), origin_last=int(origins[-1]), diagnostics=diagnostics,
                   completed_monotonic=time.monotonic())
    with target.with_suffix('.npz').open('xb') as stream:
        np.savez(stream, origins=origins, predictions=forecasts, boundary_prediction=boundary)
    write_json(target.with_suffix('.json'), receipt)
    return dict(task=stem(task), epochs=receipt['epochs'], selected_epoch=receipt['selected_epoch'],
                seconds=receipt['seconds'], memory=diagnostics)


def batch():
    pre = check_sources()
    if (ROOT / 'RUN_COMPLETE.json').exists():
        raise FileExistsError('Pilot complete; no repeated fitting')
    if time.monotonic() >= pre['deadline_monotonic']:
        raise TimeoutError('Original pilot deadline passed; no implicit extension')
    saved = saved_stems(ROOT / 'local_only' / 'fits')
    pending = [t for t in planned_tasks() if stem(t) not in saved]
    if not pending:
        print('ALL_FITS_SAVED; aggregate without refitting', flush=True)
        return
    chosen = pending[:6]
    batch_id = len(list(ROOT.glob('BATCH_*_START.json')))
    write_json(ROOT / f'BATCH_{batch_id:02d}_START.json', dict(tasks=[stem(t) for t in chosen],
               started_monotonic=time.monotonic(), original_deadline=pre['deadline_monotonic']))
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_one, t, pre['deadline_monotonic']) for t in chosen]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    saved = saved_stems(ROOT / 'local_only' / 'fits')
    write_json(ROOT / f'BATCH_{batch_id:02d}_DONE.json', dict(saved=len(saved), missing=32 - len(saved),
               finished_monotonic=time.monotonic()))
    print('INVENTORY', len(saved), '/32', flush=True)


def producer_scores(price, dates, origins, forecasts):
    cells, annual, seed_year = [], {}, []
    for year in ('2020', '2021'):
        mask = (np.char.startswith(dates[origins], year) & np.char.startswith(dates[origins + 20], year))
        t = origins[mask]
        values = []
        for h in (1, 5, 20):
            change = price[t + h] - price[t]
            ratios = np.mean((forecasts[:, mask, h - 1] - change) ** 2, axis=1) / np.mean(change ** 2)
            values.append(ratios)
            cells.append(dict(year=year, horizon=h, count=len(t), ratio=float(ratios.mean())))
        values = np.mean(values, axis=0)
        seed_year.append(values)
        annual[year] = float(values.mean())
    seeds = np.mean(seed_year, axis=0)
    return dict(score=float(seeds.mean()), seeds=seeds.tolist(), annual=annual, cells=cells)


def aggregate():
    pre = check_sources()
    root = ROOT / 'local_only'
    if len(saved_stems(root / 'fits')) != 32:
        raise ValueError('All 32 formal fits are required before ranking')
    if (ROOT / 'RUN_COMPLETE.json').exists() or (ROOT / 'RESULTS.json').exists():
        raise FileExistsError('Aggregate output already exists')
    with np.load(root / 'INPUTS.npz', allow_pickle=False) as z:
        price, dates = np.array(z['price']), np.array(z['dates'])
    models, records = {}, []
    for family in MODES:
        forecasts = []
        for seed in SEEDS:
            blocks = []
            for cut in CUTS:
                name = stem((family, seed, cut))
                record = json.loads((root / 'fits' / (name + '.json')).read_text(encoding='utf-8'))
                if record['completed_monotonic'] > pre['deadline_monotonic']:
                    raise ValueError('Saved fit exceeded original deadline')
                records.append(dict(artifact=name, receipt=record))
                with np.load(root / 'fits' / (name + '.npz'), allow_pickle=False) as z:
                    blocks.append(np.array(z['predictions']))
            forecasts.append(np.concatenate(blocks))
        models[family] = producer_scores(price, dates, np.arange(459, 930), np.stack(forecasts))
    advance = models['utility']['score'] < models['learned']['score'] and models['utility']['score'] <= .995 * models['gru']['score']
    write_json(ROOT / 'RESULTS.json', dict(formal_fits=32, models=models, advance_candidate=advance,
               unexposed_test=False, novelty_established=False, candidate_qualified=False))
    write_json(ROOT / 'TRAINING_LOGS.json', dict(formal_fits=32, records=records))
    elapsed = max(r['receipt']['completed_monotonic'] for r in records) - pre['started_monotonic']
    write_json(ROOT / 'RUN_COMPLETE.json', dict(formal_fits=32, fits_per_family=8,
               elapsed_seconds=elapsed, budget_seconds=1200, completed_within_deadline=True))
    print(json.dumps(dict(scores={k: v['score'] for k, v in models.items()}, advance_candidate=advance,
                          elapsed_seconds=elapsed)), flush=True)


def boundary():
    check_sources()
    if not (ROOT / 'RUN_COMPLETE.json').exists() or (ROOT / 'BOUNDARY.json').exists():
        raise ValueError('Need complete pilot and no existing boundary result')
    path = ROOT / 'BOUNDARY_PRE_RUN.json'
    if not path.exists():
        write_json(path, dict(deadline_monotonic=time.monotonic() + 180, budget_seconds=180,
                             families=['gru', 'utility'], seed=17, cut=459, fresh_fits=2))
    before = json.loads(path.read_text(encoding='utf-8'))
    with np.load(ROOT / 'local_only' / 'INPUTS.npz', allow_pickle=False) as z:
        changed = np.array(z['price'], copy=True)
    changed[459:] += 10000 + 5 * np.arange(len(changed) - 459)
    checks = []
    for family in ('gru', 'utility'):
        receipt_path = ROOT / f'BOUNDARY_{family}.json'
        if receipt_path.exists():
            checks.append(json.loads(receipt_path.read_text(encoding='utf-8')))
            continue
        original = json.loads((ROOT / 'local_only' / 'fits' / (stem((family, 17, 459)) + '.json')).read_text(encoding='utf-8'))
        model, scale, record = fit(changed, 459, family, 17, deadline=before['deadline_monotonic'])
        forecast, _ = predict(model, scale, changed, np.array([458]))
        with np.load(ROOT / 'local_only' / 'fits' / (stem((family, 17, 459)) + '.npz'), allow_pickle=False) as z:
            difference = float(np.max(np.abs(forecast - z['boundary_prediction'])))
        passed = record['state_sha256'] == original['state_sha256'] and difference <= 1e-11
        result = dict(family=family, passed=passed, parameter_digest_equal=record['state_sha256'] == original['state_sha256'],
                      prediction_max_difference=difference, fresh_epochs=record['epochs'], seconds=record['seconds'])
        write_json(receipt_path, result)
        checks.append(result)
        if not passed:
            raise ValueError('Real suffix refit changed earlier parameters or forecast')
    write_json(ROOT / 'BOUNDARY.json', dict(passed=all(r['passed'] for r in checks), fresh_fits=2, checks=checks,
               scope='Representative fresh same-implementation suffix refits, not full rerun or third-party replication'))
    print('BOUNDARY_PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('init', 'batch', 'aggregate', 'boundary'))
    args = parser.parse_args()
    {'init': initialize, 'batch': batch, 'aggregate': aggregate, 'boundary': boundary}[args.action]()
