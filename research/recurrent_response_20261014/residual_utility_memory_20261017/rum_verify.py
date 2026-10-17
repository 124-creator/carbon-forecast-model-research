"""Independent raw-price score replay; imports neither model nor trainer."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np

MODES = ('gru', 'fixed', 'learned', 'utility')
CUTS = (459, 619, 779, 899)
SEEDS = (17, 29)


def score_arrays(price, dates, origins, predictions):
    ids = np.asarray(origins, dtype=np.int64)
    if (ids.ndim != 1 or not len(ids) or ids.min() < 0 or np.any(np.diff(ids) <= 0) or
            ids.max() + 20 >= len(price) or predictions.shape != (2, len(ids), 20) or
            not np.isfinite(predictions).all()):
        raise ValueError('Invalid or incomplete score support')
    totals, annual, counts, cells = [], {}, {}, []
    for year in ('2020', '2021'):
        positions = [i for i, t in enumerate(ids) if dates[t][:4] == year and dates[t + 20][:4] == year]
        if not positions:
            raise ValueError('Missing same-year labels')
        t = ids[positions]
        counts[year] = len(t)
        year_risks = []
        for h in (1, 5, 20):
            actual_change = price[t + h] - price[t]
            rw = float(np.dot(actual_change, actual_change) / len(t))
            if not np.isfinite(rw) or rw <= 0:
                raise ValueError('Invalid random-walk reference')
            risks = []
            for seed in range(2):
                error = predictions[seed, positions, h - 1] - actual_change
                risks.append(float(np.dot(error, error) / len(t) / rw))
            year_risks.append(risks)
            cells.append(dict(year=year, horizon=h, count=len(t), ratio=float(np.mean(risks))))
        year_seed = np.asarray(year_risks).mean(axis=0)
        totals.append(year_seed)
        annual[year] = float(year_seed.mean())
    values = np.asarray(totals).mean(axis=0)
    return dict(score=float(values.mean()), seeds=values.tolist(), annual=annual, counts=counts, cells=cells)


def replay(directory):
    directory = Path(directory)
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    complete = read(directory / 'RUN_COMPLETE.json')
    if complete['formal_fits'] != 32:
        raise ValueError('Incomplete pilot')
    before = read(directory / 'PRE_RUN.json')
    for name, digest in before['source_sha256'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Executed source changed: ' + name)
    original = directory.parents[2] / 'data/parsed/figshare_28682348/guangzhou_reported.csv'
    with original.open('rb', buffering=0) as stream:
        raw = stream.read(29204)
    if len(raw) != 29204 or hashlib.sha256(raw).hexdigest() != '1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070':
        raise ValueError('Approved original prefix differs')
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    price = np.array([float(r['author_reported_price']) for r in rows])
    dates = np.array([r['date'] for r in rows])
    if len(price) != 950 or dates[-1] != '2021-12-31':
        raise ValueError('Approved date boundary differs')
    root = directory / 'local_only'
    with np.load(root / 'INPUTS.npz', allow_pickle=False) as z:
        if not np.array_equal(z['price'], price) or not np.array_equal(z['dates'], dates):
            raise ValueError('Input buffer differs from original')
    expected = {f'{m}_s{s}_t{t}' for m in MODES for s in SEEDS for t in CUTS}
    logs = {p.stem for p in (root / 'fits').glob('*.json')}
    arrays = {p.stem for p in (root / 'fits').glob('*.npz')}
    if logs != expected or arrays != expected:
        raise ValueError('Missing, duplicate or unexpected fit')
    models, differences, traces = {}, [], []
    results = read(directory / 'RESULTS.json')
    for family in MODES:
        seed_predictions = []
        for seed in SEEDS:
            blocks = []
            for i, cut in enumerate(CUTS):
                name = f'{family}_s{seed}_t{cut}'
                receipt = read(root / 'fits' / (name + '.json'))
                ids = np.arange(max(60, cut - 500), cut - 20)
                split = len(ids) - 60
                train, validation = np.arange(split - 20), np.arange(split, len(ids))
                scale = max(float(np.stack([np.diff(price[t - 60:t + 1]) for t in ids[train]]).std()), 1e-6)
                if (receipt['family'], receipt['seed'], receipt['end'], receipt['train_n'], receipt['validation_n']) != (family, seed, cut, len(train), 60):
                    raise ValueError('Fit identity/support differs')
                if receipt['train_max_target'] != int(ids[train[-1]] + 20) or receipt['validation_first_origin'] != int(ids[validation[0]]) or receipt['validation_max_target'] >= cut:
                    raise ValueError('Maturity/purge differs')
                if abs(scale - receipt['scale']) > 1e-12:
                    raise ValueError('Train scale differs')
                changes = price[ids[train, None] + np.arange(1, 21)] - price[ids[train], None]
                variance = np.maximum(np.var(changes / scale, axis=0), 1e-6)
                if not np.allclose(variance, receipt['horizon_variance'], rtol=0, atol=1e-11):
                    raise ValueError('Teacher scale uses different support')
                best, chosen = receipt['initial_validation'], 0
                trace = np.asarray(receipt['trace'])
                if trace.shape != (receipt['epochs'], 5) or not np.isfinite(trace).all():
                    raise ValueError('Invalid training trace')
                for epoch, _, _, value, _ in trace:
                    if value < best - 1e-6:
                        best, chosen = float(value), int(epoch)
                if chosen != receipt['selected_epoch'] or abs(best - receipt['best_validation']) > 1e-12:
                    raise ValueError('Checkpoint choice differs')
                traces.append(dict(family=family, selected_epoch=chosen, epochs=receipt['epochs']))
                stop = CUTS[i + 1] if i + 1 < len(CUTS) else 930
                with np.load(root / 'fits' / (name + '.npz'), allow_pickle=False) as z:
                    if not np.array_equal(z['origins'], np.arange(cut, stop)) or z['predictions'].shape != (stop - cut, 20):
                        raise ValueError('Prediction block differs')
                    blocks.append(np.array(z['predictions'], copy=True))
            seed_predictions.append(np.concatenate(blocks))
        metric = score_arrays(price, dates, np.arange(459, 930), np.stack(seed_predictions))
        if metric['counts'] != {'2020': 224, '2021': 227}:
            raise ValueError('Primary same-year origin counts differ')
        other = results['models'][family]
        differences.append(abs(metric['score'] - other['score']))
        differences.extend(abs(np.asarray(metric['seeds']) - other['seeds']).tolist())
        differences.extend(abs(metric['annual'][y] - other['annual'][y]) for y in ('2020', '2021'))
        differences.extend(abs(a['ratio'] - b['ratio']) for a, b in zip(metric['cells'], other['cells']))
        models[family] = metric
    if max(differences) > 1e-11:
        raise ValueError('Independent scores differ')
    advance = models['utility']['score'] < models['learned']['score'] and models['utility']['score'] <= .995 * models['gru']['score']
    if bool(results['advance_candidate']) != advance:
        raise ValueError('Frozen continuation rule differs')
    return dict(passed=True, fits=32, models=models, numeric_comparisons=len(differences),
                max_absolute_difference=max(differences), advance_candidate=advance,
                initialization_selected=sum(t['selected_epoch'] == 0 for t in traces),
                scope='Independent saved-prediction scoring, not third-party refitting or novelty certification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = replay(Path(__file__).resolve().parent)
    with Path(args.output).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'models'}))
