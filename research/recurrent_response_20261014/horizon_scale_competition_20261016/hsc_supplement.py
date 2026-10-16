"""Separate, fixed-task completion; never rewrites the interrupted-run verdict."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import datetime
import hashlib
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from hsc_run import HERE, MODES, CUTS, SEEDS, source_hashes, dump, save_fit, market_metrics
from hsc_fit import fit, predict

SUP = HERE / 'supplement_20261017'
NEW = HERE / 'local_only' / 'supplement_20261017' / 'market_fits'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def merged_paths(original, supplemental):
    result = {}
    for root in (Path(original), Path(supplemental)):
        logs = {p.stem for p in root.glob('*.json')}
        arrays = {p.stem for p in root.glob('*.npz')}
        if logs != arrays or set(result).intersection(logs):
            raise ValueError('Duplicate or unpaired market fit')
        result.update({stem: root / stem for stem in logs})
    return result


def file_hashes(root, paths):
    root = Path(root)
    return {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def check_originals(root, hashes):
    for name, digest in hashes.items():
        if hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Original artifact changed: ' + name)


def sources():
    paths = [HERE / 'hsc_supplement.py', HERE / 'hsc_verify.py', SUP / 'PROTOCOL.md']
    return file_hashes(HERE, paths)


def initialize():
    if (SUP / 'PRE_RUN.json').exists() or NEW.exists():
        raise FileExistsError('Supplement already initialized')
    before, old = read(HERE / 'PRE_RUN.json'), read(HERE / 'INCOMPLETE.json')
    if source_hashes() != before['source_sha256']:
        raise ValueError('Frozen algorithm or protocol differs')
    selection = read(HERE / 'SELECTION.json')
    paths = merged_paths(HERE / 'local_only' / 'market_fits', NEW)
    expected = {f'{m}_{selection[m]["config_id"]}_s{s}_t{t}': (m, s, t)
                for m in MODES for s in SEEDS for t in CUTS}
    absent = sorted(set(expected) - set(paths))
    if len(paths) != 85 or absent != old['missing_fits'] or len(absent) != 59:
        raise ValueError('Original saved/missing grid differs')
    tasks = [dict(stem=stem, mode=expected[stem][0], seed=expected[stem][1], cut=expected[stem][2],
                  config=next(c for c in before['configurations'] if c['id'] == selection[expected[stem][0]]['config_id']))
             for stem in absent]
    tasks.sort(key=lambda r: (r['cut'], r['seed'], r['mode']))
    originals = list((HERE / 'local_only' / 'search_fits').glob('*'))
    originals += list((HERE / 'local_only' / 'market_fits').glob('*'))
    originals += [p for p in HERE.glob('*.json')]
    originals += [HERE / 'local_only' / 'INPUTS.npz', HERE / 'local_only' / 'TUNING_INPUT.npz']
    manifest = file_hashes(HERE, [p for p in originals if p.is_file()])
    dump(SUP / 'PRE_RUN.json', dict(initialized_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         original_formal_saved=133, expected_supplemental=59, tasks=tasks, wall_limit=3600, batch_max=6,
         source_sha256=before['source_sha256'], supplement_source_sha256=sources(),
         original_artifact_sha256=manifest, original_run_remains_incomplete=True,
         scope='Post-exposure missing-fit completion only; no rescue tuning or new data'))
    NEW.mkdir(parents=True)
    print('INITIALIZED59_MISSING_FITS', flush=True)


def check(pre):
    if source_hashes() != pre['source_sha256'] or sources() != pre['supplement_source_sha256']:
        raise ValueError('Frozen source changed')
    check_originals(HERE, pre['original_artifact_sha256'])


def one(task):
    item, deadline = task
    with np.load(HERE / 'local_only' / 'INPUTS.npz', allow_pickle=False) as z:
        price = z['price']
    model, scale, log = fit(price, item['cut'], item['mode'], item['config'], item['seed'], deadline=deadline)
    ids = np.arange(item['cut'], min(item['cut'] + 40, 930))
    forecast, weights = predict(model, scale, price, ids)
    log.update(phase='market', config_id=item['config']['id'], cut=item['cut'],
               origin_first=int(ids[0]), origin_last=int(ids[-1]), supplemental_execution=True)
    save_fit(NEW, item['stem'], model, scale, log, forecast, weights, ids)
    print('SUPPLEMENT ' + item['stem'] + f' epochs={log["epochs"]} selected={log["selected_epoch"]}', flush=True)
    return item['stem']


def batch():
    pre = read(SUP / 'PRE_RUN.json')
    check(pre)
    if (SUP / 'RUN_COMPLETE.json').exists() or (SUP / 'FAILURE.json').exists():
        raise ValueError('Supplement already closed')
    started = SUP / 'STARTED.json'
    if not started.exists():
        dump(started, dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    elapsed = (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(read(started)['started_at'])).total_seconds()
    remaining = pre['wall_limit'] - elapsed
    if remaining <= 0:
        dump(SUP / 'FAILURE.json', dict(reason='Supplementary3600-second budget expired', completed=len(list(NEW.glob('*.json')))))
        raise TimeoutError('Supplement expired')
    pairs = merged_paths(HERE / 'local_only' / 'market_fits', NEW)
    pending = [t for t in pre['tasks'] if t['stem'] not in pairs]
    taken = pending[:pre['batch_max']]
    number = len(list(SUP.glob('BATCH_*_START.json')))
    for old_start in SUP.glob('BATCH_*_START.json'):
        if not old_start.with_name(old_start.name.replace('_START', '_DONE')).exists():
            raise ValueError('Prior batch did not finish; preserve failure, do not silently retry')
    dump(SUP / f'BATCH_{number:02d}_START.json', dict(stems=[t['stem'] for t in taken], remaining_wall_seconds=remaining))
    deadline = time.monotonic() + remaining
    try:
        with ProcessPoolExecutor(max_workers=3) as pool:
            for f in as_completed([pool.submit(one, (task, deadline)) for task in taken]):
                f.result()
        check(pre)
        dump(SUP / f'BATCH_{number:02d}_DONE.json', dict(completed=len(taken), stems=[t['stem'] for t in taken]))
    except Exception as error:
        dump(SUP / 'FAILURE.json', dict(error=type(error).__name__, reason=str(error), traceback=traceback.format_exc()))
        raise
    print(json.dumps(dict(supplemental_saved=len(list(NEW.glob('*.json'))), remaining=len(pending)-len(taken))), flush=True)


def aggregate():
    pre = read(SUP / 'PRE_RUN.json')
    check(pre)
    pairs = merged_paths(HERE / 'local_only' / 'market_fits', NEW)
    selection = read(HERE / 'SELECTION.json')
    expected = {f'{m}_{selection[m]["config_id"]}_s{s}_t{t}' for m in MODES for s in SEEDS for t in CUTS}
    if set(pairs) != expected or len(list(NEW.glob('*.json'))) != 59:
        raise ValueError('Combined market grid is incomplete')
    with np.load(HERE / 'local_only' / 'INPUTS.npz', allow_pickle=False) as z:
        price, dates = z['price'], z['dates']
    results, diagnostics = {}, {}
    for mode in MODES:
        curves, weights = [], []
        for seed in SEEDS:
            blocks, gates = [], []
            for cut in CUTS:
                path = pairs[f'{mode}_{selection[mode]["config_id"]}_s{seed}_t{cut}']
                with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
                    blocks.append(z['prediction'])
                    gates.append(z['weights'])
            curves.append(np.concatenate(blocks))
            weights.append(np.concatenate(gates))
        results[mode] = market_metrics(price, dates, np.arange(459, 930), np.stack(curves))
        mean = np.mean(weights, axis=(0, 1))
        diagnostics[mode] = dict(mean_weights_by_scale_horizon=mean.tolist(), horizon_weight_range=float(np.max(np.ptp(mean, axis=1))))
    dump(SUP / 'RESULTS.json', dict(complete=True, models=results, routing=diagnostics, status='SUPPLEMENTAL_DEVELOPMENT_ONLY',
         original_run_complete=False, unseen_test=False, novelty_established=False))
    dump(SUP / 'RUN_COMPLETE.json', dict(complete=True, formal_fits=192, fits=dict(search=48, market=144),
         reused_formal_fits=133, supplementary_fits=59, original_run_complete=False,
         seconds_since_supplement_start=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(read(SUP/'STARTED.json')['started_at'])).total_seconds(),
         source_sha256=pre['source_sha256']))
    print(json.dumps({m:r['score'] for m,r in results.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('init', 'batch', 'aggregate'))
    args = parser.parse_args()
    {'init': initialize, 'batch': batch, 'aggregate': aggregate}[args.action]()
