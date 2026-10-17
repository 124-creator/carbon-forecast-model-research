"""Bounded annual hindsight grid; all fits fresh and no partial champion."""
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
from hd_core import configuration_grid
from hd_fit import fit, predict
from fair_core import reference
from fair_data import load_prefix

ROOT = Path(__file__).resolve().parent
CUTS = (459, 703)


def planned_tasks():
    return [(c['id'], cut) for c in configuration_grid() for cut in CUTS]


def stem(task):
    return f'{task[0]}_t{task[1]}'


def inventory(path):
    path = Path(path)
    logs, arrays = {p.stem for p in path.glob('*.json')}, {p.stem for p in path.glob('*.npz')}
    if logs != arrays or not logs.issubset({stem(t) for t in planned_tasks()}):
        raise ValueError('Unpaired/unexpected fit, not a valid completed case')
    return logs


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def check_sources():
    before = json.loads((ROOT / 'PRE_RUN.json').read_text(encoding='utf-8'))
    for name, digest in before['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen source changed: ' + name)
    return before


def initialize():
    if (ROOT / 'PRE_RUN.json').exists() or (ROOT / 'local_only').exists():
        raise FileExistsError('Do not overwrite/restart an existing run')
    price, dates, data = load_prefix()
    if tuple(int((dates < y).sum()) for y in ('2020-01-01', '2021-01-01')) != CUTS:
        raise ValueError('Annual boundaries differ')
    timing = json.loads((ROOT / 'TIMING.json').read_text(encoding='utf-8'))
    if timing['timing_fits'] != 5 or timing['formal_fits_started']:
        raise ValueError('Missing truthful preflight')
    names = ['hd_core.py', 'hd_fit.py', 'hd_run.py', 'hd_analysis.py', 'hd_verify.py', 'test_hd.py', 'PROTOCOL.md', 'TIMING.json',
             '../fair_recurrent_tuning_20261016/fair_core.py', '../fair_recurrent_tuning_20261016/fair_data.py']
    (ROOT / 'local_only' / 'fits').mkdir(parents=True)
    with (ROOT / 'local_only' / 'INPUTS.npz').open('xb') as stream:
        np.savez(stream, price=price, dates=dates, reference=reference(price)[0])
    started = time.monotonic()
    write_json(ROOT / 'PRE_RUN.json', dict(started_at_utc=datetime.now(timezone.utc).isoformat(), started_monotonic=started,
               deadline_monotonic=started + 2700, budget_seconds=2700, workers=2, batch_limit=8,
               formal_expected=len(planned_tasks()), configs=configuration_grid(), tasks=planned_tasks(), cuts=CUTS,
               seed=17, max_epochs=60, data=data, source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in names},
               environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, cuda=torch.cuda.is_available(), dtype='float64', math_threads=1),
               hindsight_selection_authorized=True, unexposed_test=False, novelty_established=False))
    print('FROZEN', len(planned_tasks()), 'fits; 2700 seconds; 2 CPU workers', flush=True)


def run_one(task, deadline):
    cid, cut = task
    config = next(c for c in configuration_grid() if c['id'] == cid)
    target = ROOT / 'local_only' / 'fits' / stem(task)
    if target.with_suffix('.json').exists() or target.with_suffix('.npz').exists():
        raise FileExistsError('Do not overwrite or silently repeat a fit')
    with np.load(ROOT / 'local_only' / 'INPUTS.npz', allow_pickle=False) as z:
        price, dates = np.array(z['price']), np.array(z['dates'])
    model, receipt = fit(price, dates, cut, config, seed=17, deadline=deadline)
    origins = np.arange(cut, 703 if cut == 459 else 930)
    train_origins = np.arange(receipt['pool_start'], receipt['pool_start'] + receipt['train_n'])
    validation_origins = np.arange(receipt['validation_first_origin'], cut - 20)
    predictions = predict(model, price, dates, origins)
    training = predict(model, price, dates, train_origins)
    validation = predict(model, price, dates, validation_origins)
    boundary = predict(model, price, dates, np.array([cut - 1]))
    if not all(np.isfinite(x).all() for x in (predictions, training, validation, boundary)) or time.monotonic() > deadline:
        raise TimeoutError('Nonfinite or over-deadline fit cannot be completed')
    receipt.update(completed_monotonic=time.monotonic(), first_origin=int(origins[0]), last_origin=int(origins[-1]))
    with target.with_suffix('.npz').open('xb') as stream:
        np.savez(stream, origins=origins, predictions=predictions, training_origins=train_origins, training_predictions=training,
                 validation_origins=validation_origins, validation_predictions=validation, boundary_prediction=boundary)
    write_json(target.with_suffix('.json'), receipt)
    return dict(case=stem(task), family=config['family'], epochs=receipt['epochs'], seconds=receipt['seconds'])


def batch():
    before = check_sources()
    if (ROOT / 'RUN_COMPLETE.json').exists():
        raise FileExistsError('Formal run already complete')
    if time.monotonic() >= before['deadline_monotonic']:
        raise TimeoutError('Original deadline passed; no implicit continuation')
    saved = inventory(ROOT / 'local_only' / 'fits')
    pending = [t for t in planned_tasks() if stem(t) not in saved]
    if not pending:
        print('ALL_FITS_SAVED', flush=True)
        return
    chosen = pending[:8]
    index = len(list(ROOT.glob('BATCH_*_START.json')))
    write_json(ROOT / f'BATCH_{index:02d}_START.json', dict(tasks=[stem(t) for t in chosen], started_monotonic=time.monotonic(), original_deadline=before['deadline_monotonic']))
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_one, t, before['deadline_monotonic']) for t in chosen]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    saved = inventory(ROOT / 'local_only' / 'fits')
    write_json(ROOT / f'BATCH_{index:02d}_DONE.json', dict(completed=len(saved), missing=len(planned_tasks()) - len(saved), finished_monotonic=time.monotonic()))
    print('INVENTORY', len(saved), '/', len(planned_tasks()), flush=True)


def aggregate():
    before = check_sources()
    if len(inventory(ROOT / 'local_only' / 'fits')) != len(planned_tasks()):
        raise ValueError('No champion from an incomplete grid')
    if (ROOT / 'RUN_COMPLETE.json').exists() or (ROOT / 'RESULTS.json').exists():
        raise FileExistsError('Already aggregated; do not overwrite')
    from hd_analysis import analyze
    result, effects, records, csv_text = analyze(ROOT)
    if any(r['completed_monotonic'] > before['deadline_monotonic'] for r in records):
        raise ValueError('Formal run exceeded original deadline')
    write_json(ROOT / 'RESULTS.json', result)
    write_json(ROOT / 'FACTOR_EFFECTS.json', effects)
    write_json(ROOT / 'TRAINING_SUMMARY.json', dict(formal_fits=len(records), records=records, full_traces_local_only=True))
    with (ROOT / 'ALL_CONFIGS.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        stream.write(csv_text)
    elapsed = max(r['completed_monotonic'] for r in records) - before['started_monotonic']
    write_json(ROOT / 'RUN_COMPLETE.json', dict(formal_fits=len(records), configs=len(before['configs']), elapsed_seconds=elapsed,
               budget_seconds=2700, completed_within_deadline=True, hindsight_only=True))
    print(json.dumps(dict(fits=len(records), elapsed_seconds=elapsed, selections=result['selections'])), flush=True)


def boundary():
    check_sources()
    if not (ROOT / 'RUN_COMPLETE.json').exists() or (ROOT / 'BOUNDARY.json').exists():
        raise ValueError('Need complete formal run and no previous final boundary')
    path = ROOT / 'BOUNDARY_PRE_RUN.json'
    cases = ['c022', 'c046', 'c093']
    if not path.exists():
        write_json(path, dict(cases=cases, seed=17, cut=459, budget_seconds=180, deadline_monotonic=time.monotonic() + 180))
    before = json.loads(path.read_text(encoding='utf-8'))
    with np.load(ROOT / 'local_only' / 'INPUTS.npz', allow_pickle=False) as z:
        price, dates = np.array(z['price']), np.array(z['dates'])
    changed = price.copy()
    changed[459:] += 10000 + np.arange(len(price) - 459) * 10
    checks = []
    for cid in cases:
        target = ROOT / f'BOUNDARY_{cid}.json'
        if target.exists():
            result = json.loads(target.read_text(encoding='utf-8'))
        else:
            config = next(c for c in configuration_grid() if c['id'] == cid)
            original = json.loads((ROOT / 'local_only' / 'fits' / (stem((cid, 459)) + '.json')).read_text(encoding='utf-8'))
            model, record = fit(changed, dates, 459, config, deadline=before['deadline_monotonic'])
            forecast = predict(model, changed, dates, np.array([458]))
            with np.load(ROOT / 'local_only' / 'fits' / (stem((cid, 459)) + '.npz'), allow_pickle=False) as z:
                diff = float(np.max(np.abs(forecast - z['boundary_prediction'])))
            result = dict(config_id=cid, family=config['family'], state_digest_equal=record['state_sha256'] == original['state_sha256'],
                          prediction_max_difference=diff, epochs=record['epochs'], seconds=record['seconds'],
                          passed=record['state_sha256'] == original['state_sha256'] and diff <= 1e-10)
            write_json(target, result)
        if not result['passed']:
            raise ValueError('Representative suffix refit failed')
        checks.append(result)
    write_json(ROOT / 'BOUNDARY.json', dict(passed=True, fresh_fits=len(checks), checks=checks,
               scope='Three representative fresh same-implementation fits, not all 192 repeated'))
    print('BOUNDARY_PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('init', 'batch', 'aggregate', 'boundary'))
    args = parser.parse_args()
    {'init': initialize, 'batch': batch, 'aggregate': aggregate, 'boundary': boundary}[args.action]()
