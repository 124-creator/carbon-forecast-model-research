"""Independent original-price labels and arithmetic; no trainer/scorer imports."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np

def independent_scores(price, dates, origins, prediction):
    origins = np.asarray(origins, dtype=np.int64)
    if prediction.ndim != 3 or prediction.shape[1:] != (len(origins), 20) or not np.isfinite(prediction).all():
        raise ValueError('Invalid or incomplete numeric forecasts')
    if len(origins) == 0 or origins.min() < 0 or origins.max() + 20 >= len(price):
        raise ValueError('Label support exceeds approved prices')
    eligible = np.array([dates[t][:4] == dates[t + 20][:4] for t in origins])
    years = np.array([dates[t][:4] for t in origins])
    seed_rows, ensemble_rows, cells, annual, counts = [], [], [], {}, {}
    for year in ('2020', '2021'):
        selected = np.flatnonzero(eligible & (years == year))
        if not len(selected):
            raise ValueError('Missing complete same-year support')
        partial = independent_span(price, origins[selected], prediction[:, selected])
        counts[year] = len(selected)
        seed_rows.append(partial['seeds'])
        ensemble_rows.append(partial['ensemble'])
        annual[year] = partial['score']
        cells.extend(dict(cell, year=year) for cell in partial['cells'])
    seeds = np.mean(seed_rows, axis=0)
    return dict(score=float(seeds.mean()), ensemble=float(np.mean(ensemble_rows)),
                seeds=seeds.tolist(), cells=cells, annual=annual, counts=counts)


def independent_span(price, origins, prediction):
    if prediction.ndim != 3 or prediction.shape[1:] != (len(origins), 20) or not np.isfinite(prediction).all():
        raise ValueError('Invalid numeric support')
    seed_cells, ensemble_cells, cells = [], [], []
    current = price[origins]
    for horizon in (1, 5, 20):
        actual_price = price[origins + horizon]
        walk_error = current - actual_price
        denominator = float(np.dot(walk_error, walk_error) / len(origins))
        if denominator <= 0:
            raise ValueError('Undefined RW denominator')
        errors = prediction[:, :, horizon - 1] + current - actual_price
        ratios = np.array([np.dot(row, row) / len(origins) / denominator for row in errors])
        average_error = prediction[:, :, horizon - 1].mean(axis=0) + current - actual_price
        ensemble_cells.append(float(np.dot(average_error, average_error) / len(origins) / denominator))
        seed_cells.append(ratios)
        cells.append(dict(horizon=horizon, count=len(origins), ratio=float(ratios.mean())))
    seeds = np.array(seed_cells).mean(axis=0)
    return dict(score=float(seeds.mean()), ensemble=float(np.mean(ensemble_cells)), seeds=seeds.tolist(), cells=cells)

def require_complete(directory):
    directory = Path(directory)
    target = directory / 'RUN_COMPLETE.json'
    if (directory / 'INCOMPLETE.json').exists() or not target.exists():
        raise ValueError('Incomplete diagnostics cannot pass replay')
    receipt = json.loads(target.read_text(encoding='utf-8'))
    if receipt.get('complete') is not True:
        raise ValueError('Run is not complete')
    return receipt


def replay(directory):
    directory = Path(directory)
    completion = require_complete(directory)
    root = directory / 'local_only'
    original = directory.parents[2] / 'data/parsed/figshare_28682348/guangzhou_reported.csv'
    with original.open('rb', buffering=0) as stream:
        raw = stream.read(29204)
    prefix_hash = hashlib.sha256(raw).hexdigest()
    if len(raw) != 29204 or prefix_hash != '1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070':
        raise ValueError('Original prefix differs')
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    price = np.array([float(row['author_reported_price']) for row in rows])
    dates = np.array([row['date'] for row in rows])
    if len(price) != 950 or dates[-1] != '2021-12-31' or dates[458] != '2019-12-31':
        raise ValueError('Approved chronological support differs')
    with np.load(root / 'INPUTS.npz', allow_pickle=False) as data:
        if not np.array_equal(price, data['price']) or not np.array_equal(dates, data['dates']):
            raise ValueError('Runner input arrays differ from original CSV')
    with np.load(root / 'TUNING_INPUT.npz', allow_pickle=False) as data:
        if not np.array_equal(price[:459], data['price']):
            raise ValueError('Tuning includes different or post-2019 prices')
    before = json.loads((directory / 'PRE_RUN.json').read_text(encoding='utf-8'))
    for name, digest in before['source_sha256'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen execution source differs: ' + name)
    families = ('gru_dense', 'gru_profile', 'gru_rank6', 'lstm_dense')
    if before['plan']['families'] != list(families) or before['plan']['search_cuts'] != [359, 419]:
        raise ValueError('Frozen model schedule differs')
    grid = {config['id']: config for config in before['configurations']}
    if len(grid) != 8:
        raise ValueError('Frozen grid size differs')
    selection = json.loads((directory / 'SELECTION.json').read_text(encoding='utf-8'))
    market = json.loads((directory / 'MARKET_PLAN.json').read_text(encoding='utf-8'))
    results = json.loads((directory / 'RESULTS.json').read_text(encoding='utf-8'))
    differences, search_scores, search_keys = [], {}, set()
    traces, epoch_limit, selected_initial = 0, 0, 0

    def inspect_log(log, end):
        nonlocal traces, epoch_limit, selected_initial
        ids = np.arange(max(60, end - 500), end - 20)
        ntrain = len(ids) - 80
        if log['end'] != end or log['train_n'] != ntrain or log['validation_n'] != 60:
            raise ValueError('Logged fit support differs')
        if log['train_max_target'] != int(ids[ntrain - 1] + 20) or log['validation_first_origin'] != int(ids[-60]):
            raise ValueError('Train/validation purge boundary differs')
        if log['validation_max_target'] != end - 1:
            raise ValueError('Validation targets cross fit cutoff')
        inputs = np.concatenate([np.diff(price[t - 60:t + 1]) for t in ids[:ntrain]])
        if abs(max(float(inputs.std()), 1e-6) - log['scale']) > 1e-12:
            raise ValueError('Scale is not trained from permitted input windows')
        trace = np.asarray(log['trace'], dtype=float)
        if trace.shape != (log['epochs'], 4) or not np.isfinite(trace).all():
            raise ValueError('Malformed training trace')
        chosen = log['selected_epoch']
        if chosen < 0 or chosen > log['epochs'] or log['epochs'] > 60:
            raise ValueError('Invalid checkpoint or epoch budget')
        best = log['initial_validation'] if chosen == 0 else trace[chosen - 1, 2]
        if best != log['best_validation']:
            raise ValueError('Checkpoint validation receipt differs')
        traces += len(trace)
        epoch_limit += int(log['epochs'] == 60)
        selected_initial += int(chosen == 0)

    for family in families:
        for cid in grid:
            per_fold = []
            for cut in (359, 419):
                stem = f'{family}_{cid}_t{cut}'
                search_keys.add(stem)
                log = json.loads((root / 'search_fits' / (stem + '.json')).read_text(encoding='utf-8'))
                inspect_log(log, cut)
                if log['family'] != family or log['config'] != grid[cid] or log['seed'] != 17:
                    raise ValueError('Tuning identity differs')
                expected_origins = np.arange(cut, cut + 20)
                with np.load(root / 'search_fits' / (stem + '.npz'), allow_pickle=False) as saved:
                    if not np.array_equal(saved['origins'], expected_origins):
                        raise ValueError('Tuning issuance support differs')
                    value = independent_span(price, expected_origins, saved['prediction'][None])['score']
                if log['outer_max_target'] != cut + 39 or log['outer_max_target'] >= 459:
                    raise ValueError('Tuning sees post-2019 labels')
                differences.append(abs(value - log['selection_score']))
                per_fold.append(value)
            search_scores[(family, cid)] = sum(per_fold) / len(per_fold)
            differences.append(abs(search_scores[(family, cid)] - selection[family]['all_scores'][cid]))
        best_score = min(search_scores[(family, cid)] for cid in grid)
        chosen = selection[family]['config_id']
        if search_scores[(family, chosen)] > best_score + 1e-12:
            raise ValueError('Chosen configuration is not an independent minimum')
    actual_search = {path.stem for path in (root / 'search_fits').glob('*.json')}
    if actual_search != search_keys or completion['search_fits'] != len(search_keys):
        raise ValueError('Search not complete or contains extra fits')
    expected_jobs = {}
    aliases = {}
    for family in families:
        aliases[family] = dict(default='c07', selected=selection[family]['config_id'])
        for cid in dict.fromkeys(aliases[family].values()):
            for seed in (17, 29):
                expected_jobs[f'{family}_{cid}_s{seed}'] = (family, cid, seed)
    if market['aliases'] != aliases or results['aliases'] != aliases or {job['key'] for job in market['jobs']} != set(expected_jobs):
        raise ValueError('Selected/default aliases or job sets differ')
    origins = np.arange(459, 930)
    curves, market_keys = {}, set()
    for key, (family, cid, seed) in expected_jobs.items():
        with np.load(root / 'curves' / (key + '.npz'), allow_pickle=False) as data:
            curve = data['prediction'].copy()
        if curve.shape != (471, 20) or not np.isfinite(curve).all():
            raise ValueError('Incomplete market curve')
        for block in range(12):
            cut = 459 + 40 * block
            subset = (origins - 459) // 40 == block
            stem = f'{key}_b{block:02d}'
            market_keys.add(stem)
            log = json.loads((root / 'market_fits' / (stem + '.json')).read_text(encoding='utf-8'))
            inspect_log(log, cut)
            if log['family'] != family or log['config'] != grid[cid] or log['seed'] != seed or log['block'] != block:
                raise ValueError('Market model identity differs')
            with np.load(root / 'market_fits' / (stem + '.npz'), allow_pickle=False) as saved:
                if not np.array_equal(saved['origins'], origins[subset]) or not np.array_equal(saved['prediction'], curve[subset]):
                    raise ValueError('Saved fit predictions differ from complete trajectory')
        curves[key] = curve
    actual_market = {path.stem for path in (root / 'market_fits').glob('*.json')}
    actual_curves = {path.stem for path in (root / 'curves').glob('*.npz')}
    if actual_market != market_keys or actual_curves != set(expected_jobs) or completion['market_fits'] != len(market_keys):
        raise ValueError('Market fit grid is not complete')
    checked_main = 0
    replayed = {}
    for family in families:
        for variant, cid in aliases[family].items():
            name = family + '__' + variant
            prediction = np.stack([curves[f'{family}_{cid}_s{seed}'] for seed in (17, 29)])
            metrics = independent_scores(price, dates, origins, prediction)
            if metrics['counts'] != {'2020': 224, '2021': 227}:
                raise ValueError('Primary origin counts differ')
            reported = results['models'][name]
            for field in ('score', 'ensemble'):
                differences.append(abs(metrics[field] - reported[field]))
                checked_main += 1
            for a, b in zip(metrics['seeds'], reported['seeds']):
                differences.append(abs(a - b))
            for year in ('2020', '2021'):
                differences.append(abs(metrics['annual'][year] - reported['annual'][year]))
            for a, b in zip(metrics['cells'], reported['cells']):
                if any(a[field] != b[field] for field in ('horizon', 'count', 'year')):
                    raise ValueError('Metric cell identity differs')
                differences.append(abs(a['ratio'] - b['ratio']))
            replayed[name] = metrics
    maximum = max(differences)
    if maximum > 1e-11:
        raise ValueError('Independent metric replay differs')
    return dict(status='INDEPENDENT_REPLAY_PASSED', models=replayed, search_trials=len(search_keys),
                market_fits=len(market_keys), unique_trajectories=len(curves),
                primary_and_ensemble_compared=checked_main, metric_scalars_compared=len(differences),
                maximum_absolute_difference=maximum, inspected_epochs=traces,
                fits_at_epoch_limit=epoch_limit, fits_selecting_initialization=selected_initial,
                source_hashes_verified=len(before['source_sha256']), prefix_sha256=prefix_hash,
                scope='Independent original-price labels, saved forecast arithmetic, selection and chronological receipts; not independent retraining or novelty validation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.directory)
    if args.output.exists():
        raise FileExistsError('Preserve the first replay output')
    with args.output.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'models'}, indent=2))
