"""Independent bounded original-price scoring; no model/producer imports."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np


def independent_scores(price, dates, origins, predictions):
    ids = np.asarray(origins, dtype=np.int64)
    if not len(ids) or ids.min() < 0 or ids.max() + 20 >= len(price):
        raise ValueError('Unsupported labels')
    if predictions.ndim != 3 or predictions.shape[1:] != (len(ids), 20) or not np.isfinite(predictions).all():
        raise ValueError('Incomplete/nonfinite forecasts')
    counts, annual, cells, seeds, ensemble = {}, {}, [], [], []
    for year in ('2020', '2021'):
        selected = np.array([i for i, t in enumerate(ids) if dates[t][:4] == year and dates[t + 20][:4] == year])
        if not len(selected):
            raise ValueError('Missing year')
        where = ids[selected]
        counts[year] = len(where)
        year_seed, year_ensemble = [], []
        for h in (1, 5, 20):
            actual = price[where + h]
            walk = price[where]
            denominator = float(np.dot(actual - walk, actual - walk) / len(where))
            if denominator <= 0:
                raise ValueError('Undefined RW risk')
            errors = predictions[:, selected, h - 1] + walk - actual
            values = np.array([float(np.dot(e, e) / len(where) / denominator) for e in errors])
            mean_error = predictions[:, selected, h - 1].mean(0) + walk - actual
            year_seed.append(values)
            year_ensemble.append(float(np.dot(mean_error, mean_error) / len(where) / denominator))
            cells.append(dict(year=year, horizon=h, count=len(where), ratio=float(values.mean()),
                              rmse=float(np.sqrt(np.mean(errors ** 2))), mae=float(np.mean(np.abs(errors))),
                              mape_percent=float(100 * np.mean(np.abs(errors) / actual))))
        year_values = np.mean(year_seed, axis=0)
        annual[year] = float(year_values.mean())
        seeds.append(year_values)
        ensemble.append(float(np.mean(year_ensemble)))
    all_seeds = np.mean(seeds, axis=0)
    return dict(score=float(all_seeds.mean()), seeds=all_seeds.tolist(), ensemble=float(np.mean(ensemble)),
                annual=annual, counts=counts, cells=cells)


def original_prefix(directory):
    original = directory.parents[2] / 'data/parsed/figshare_28682348/guangzhou_reported.csv'
    with original.open('rb', buffering=0) as stream:
        raw = stream.read(29204)
    if len(raw) != 29204 or hashlib.sha256(raw).hexdigest() != '1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070':
        raise ValueError('Original prefix identity differs')
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    price = np.array([float(r['author_reported_price']) for r in rows])
    dates = np.array([r['date'] for r in rows])
    if len(rows) != 950 or dates[-1] != '2021-12-31' or dates[458] != '2019-12-31':
        raise ValueError('Original date boundary differs')
    return price, dates


def saved_market_inventory(root, selected, supplemental=None):
    root = Path(root)
    expected = {f'{m}_{v["config_id"]}_s{s}_t{t}'
                for m, v in selected.items() for s in (17, 29) for t in range(459, 900, 40)}
    logs = {p.stem for p in root.glob('*.json')}
    arrays = {p.stem for p in root.glob('*.npz')}
    if supplemental is not None:
        new_logs = {p.stem for p in Path(supplemental).glob('*.json')}
        new_arrays = {p.stem for p in Path(supplemental).glob('*.npz')}
        if new_logs != new_arrays or logs.intersection(new_logs):
            raise ValueError('Duplicate or unpaired supplemental fit')
        logs |= new_logs
        arrays |= new_arrays
    if logs != arrays or not logs.issubset(expected):
        raise ValueError('Unpaired or unexpected saved fits')
    missing = sorted(expected - logs)
    complete = [m for m in selected if not any(s.startswith(m + '_') for s in missing)]
    return logs, missing, complete


def replay(directory, allow_partial=False, supplement=False):
    directory = Path(directory)
    result_directory = directory / 'supplement_20261017' if supplement else directory
    new_root = directory / 'local_only' / 'supplement_20261017' / 'market_fits' if supplement else None
    partial = not (result_directory / 'RUN_COMPLETE.json').exists() or (not supplement and (directory / 'INCOMPLETE.json').exists())
    if partial and not allow_partial:
        raise ValueError('Incomplete run cannot pass replay')
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    before = read(directory / 'PRE_RUN.json')
    complete = None if partial else read(result_directory / 'RUN_COMPLETE.json')
    results = None if partial else read(result_directory / 'RESULTS.json')
    selected = read(directory / 'SELECTION.json')
    if supplement:
        supplemental_pre = read(result_directory / 'PRE_RUN.json')
        if not (directory / 'INCOMPLETE.json').exists() or supplemental_pre['expected_supplemental'] != 59:
            raise ValueError('Original interrupted-run evidence missing')
        for name, digest in supplemental_pre['original_artifact_sha256'].items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Original artifact changed: ' + name)
        for name, digest in supplemental_pre['supplement_source_sha256'].items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Supplement source changed: ' + name)
    if complete is not None and (complete['formal_fits'] != 192 or complete['fits'] != dict(search=48, market=144)):
        raise ValueError('Incorrect fit totals')
    for name, digest in before['source_sha256'].items():
        if hashlib.sha256((directory.parent / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen producer source differs: ' + name)
    price, dates = original_prefix(directory)
    root = directory / 'local_only'
    with np.load(root / 'INPUTS.npz', allow_pickle=False) as z:
        if not np.array_equal(z['price'], price) or not np.array_equal(z['dates'], dates):
            raise ValueError('Runner inputs differ from original')
    with np.load(root / 'TUNING_INPUT.npz', allow_pickle=False) as z:
        if not np.array_equal(z['price'], price[:459]):
            raise ValueError('Search inputs cross 2019 boundary')
    modes = ('raw_gru', 'raw_lstm', 'scale_mean', 'gate_only', 'shared_competition', 'horizon_competition')
    grid = {c['id']: c for c in before['configurations']}
    if set(grid) != {'c00', 'c01', 'c02', 'c03'} or before['modes'] != list(modes):
        raise ValueError('Frozen model/grid mismatch')
    if set(selected) != set(modes):
        raise ValueError('Selected model set differs')
    saved_market, missing, complete_models = saved_market_inventory(root / 'market_fits', selected, supplemental=new_root)
    differences, search_keys, market_keys, traces = [], set(), set(), []
    model_metrics = {}

    def inspect(log, family, cid, seed, cut):
        if log['family'] != family or log['config'] != grid[cid] or log['seed'] != seed or log['end'] != cut:
            raise ValueError('Fit identity mismatch')
        origins = np.arange(max(60, cut - 500), cut - 20)
        train = origins[:-80]
        if log['train_n'] != len(train) or log['validation_n'] != 60:
            raise ValueError('Wrong support size')
        if log['train_max_target'] != train[-1] + 20 or log['validation_first_origin'] != origins[-60] or log['validation_max_target'] != cut - 1:
            raise ValueError('Maturity/purge mismatch')
        scale = max(float(np.concatenate([np.diff(price[t - 60:t + 1]) for t in train]).std()), 1e-6)
        if abs(scale - log['scale']) > 1e-12:
            raise ValueError('Scale differs from training windows')
        trace = np.array(log['trace'])
        if trace.shape != (log['epochs'], 7) or not np.isfinite(trace).all() or not 1 <= len(trace) <= 60:
            raise ValueError('Bad epoch trace')
        chosen = log['selected_epoch']
        value = log['initial_validation'] if chosen == 0 else trace[chosen - 1, 2]
        if abs(value - log['best_validation']) > 1e-12 or value > min(log['initial_validation'], trace[:, 2].min()) + 1.1e-6:
            raise ValueError('Checkpoint is not validation selected')
        traces.append(dict(family=family, epochs=len(trace), initialization=chosen == 0, params=log['parameters']))

    for mode in modes:
        search_scores = {}
        for cid in grid:
            fold_scores = []
            for cut in (359, 419):
                stem = f'{mode}_{cid}_s17_t{cut}'
                search_keys.add(stem)
                log = read(root / 'search_fits' / (stem + '.json'))
                inspect(log, mode, cid, 17, cut)
                ids = np.arange(cut, cut + 20)
                with np.load(root / 'search_fits' / (stem + '.npz'), allow_pickle=False) as z:
                    if not np.array_equal(z['origins'], ids):
                        raise ValueError('Wrong search origins')
                    pred = z['prediction']
                ratios = []
                for h in (1, 5, 20):
                    y = price[ids + h] - price[ids]
                    ratios.append(float(np.mean((pred[:, h - 1] - y) ** 2) / np.mean(y ** 2)))
                val = float(np.mean(ratios))
                differences.append(abs(val - log['score']))
                fold_scores.append(val)
            search_scores[cid] = float(np.mean(fold_scores))
            differences.append(abs(search_scores[cid] - selected[mode]['all_scores'][cid]))
        cid = selected[mode]['config_id']
        if search_scores[cid] > min(search_scores.values()) + 1e-12:
            raise ValueError('Wrong selected configuration')
        curves = []
        for seed in (17, 29):
            parts = []
            for cut in range(459, 900, 40):
                stem = f'{mode}_{cid}_s{seed}_t{cut}'
                if stem not in saved_market:
                    continue
                market_keys.add(stem)
                block_root = root / 'market_fits'
                if new_root is not None and not (block_root / (stem + '.json')).exists():
                    block_root = new_root
                log = read(block_root / (stem + '.json'))
                inspect(log, mode, cid, seed, cut)
                with np.load(block_root / (stem + '.npz'), allow_pickle=False) as z:
                    if not np.array_equal(z['origins'], np.arange(cut, min(cut + 40, 930))):
                        raise ValueError('Wrong market origins')
                    if not np.isfinite(z['weights']).all() or np.max(np.abs(z['weights'].sum(1) - 1)) > 1e-12 or np.min(z['weights']) < 0:
                        raise ValueError('Invalid convex weights')
                    parts.append(z['prediction'])
            if mode in complete_models:
                curves.append(np.concatenate(parts))
        if mode not in complete_models:
            continue
        metric = independent_scores(price, dates, np.arange(459, 930), np.stack(curves))
        if metric['counts'] != {'2020': 224, '2021': 227}:
            raise ValueError('Primary counts differ')
        model_metrics[mode] = metric
        if results is None:
            continue
        original = results['models'][mode]
        for f in ('score', 'ensemble'):
            differences.append(abs(metric[f] - original[f]))
        differences.extend(abs(np.array(metric['seeds']) - original['seeds']).tolist())
        for year in ('2020', '2021'):
            differences.append(abs(metric['annual'][year] - original['annual'][year]))
        for a, b in zip(metric['cells'], original['cells']):
            if (a['year'], a['horizon'], a['count']) != (b['year'], b['horizon'], b['count']):
                raise ValueError('Cell identity differs')
            differences.extend(abs(a[f] - b[f]) for f in ('ratio', 'rmse', 'mae', 'mape_percent'))
    if {p.stem for p in (root / 'search_fits').glob('*.json')} != search_keys or saved_market != market_keys:
        raise ValueError('Actual saved fit grid differs')
    if max(differences) > 1e-11:
        raise ValueError('Independent numeric scores differ')
    return dict(passed=True, search_fits=len(search_keys), market_fits=len(market_keys),
                full_run_complete=not partial, candidate_qualified=False, missing_fits=missing,
                supplemental_completion=supplement, original_run_complete=False if supplement else not partial,
                complete_models=complete_models, model_metrics=model_metrics,
                independent_numeric_comparisons=len(differences), max_absolute_difference=max(differences),
                epochs=sum(t['epochs'] for t in traces), initialization_selected=sum(t['initialization'] for t in traces),
                scope='Independent saved-prediction replay, not independent implementation/refitting or novelty qualification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--supplement', action='store_true')
    args = parser.parse_args()
    result = replay(Path(__file__).resolve().parent, allow_partial=args.allow_partial, supplement=args.supplement)
    with Path(args.output).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))
