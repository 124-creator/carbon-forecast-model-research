"""Independent scores: imports no trainer, model, input builder, runner or IO helper."""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import numpy as np


def read_original_rows(carbon_file, prefix_n):
    if prefix_n <= 0:
        raise ValueError('Empty original-price prefix')
    with Path(carbon_file).open('rb', buffering=0) as stream:
        if prefix_n == 950:
            raw = stream.read(29204)
            if len(raw) != 29204 or hashlib.sha256(raw).hexdigest() != '1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070':
                raise ValueError('Approved original-price bytes differ')
        else:
            raw = stream.readline() + b''.join(stream.readline() for _ in range(prefix_n))
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    if len(rows) != prefix_n:
        raise ValueError('Original-price prefix is incomplete')
    return rows


def replay(folder, carbon_file, prefix_n=950):
    folder = Path(folder)
    rows = read_original_rows(carbon_file, prefix_n)
    dates = np.array([row['date'] for row in rows], dtype='datetime64[D]').astype('int64')
    calendar = dates.astype('datetime64[D]').astype(str)
    price = np.array([float(row['author_reported_price']) for row in rows])
    if not np.isfinite(price).all() or not np.all(np.diff(dates) > 0) or calendar[-1] > '2021-12-31':
        raise ValueError('Invalid permitted original-price prefix')
    years = np.array([s[:4] for s in calendar])
    origins = np.array([t for t in range(60, len(rows) - 20)
                        if years[t] in ('2020', '2021') and years[t] == years[t + 20]], dtype=np.int64)
    truth = np.array([[price[t + h] - price[t] for h in range(1, 21)] for t in origins])
    names = ('price', 'clock', 'raw_hold', 'raw_event', 'hold', 'event')
    runs = folder / 'local_only/runs'
    frozen_file = folder / 'PRE_RUN.json'
    frozen = json.loads(frozen_file.read_text(encoding='utf-8')) if frozen_file.exists() else {}
    frozen_hashes = frozen.get('source_sha256', {})
    for name, expected in frozen_hashes.items():
        if Path(name).name != name or hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen executed source changed: ' + name)
    input_identity = False
    if prefix_n == 950:
        audit = json.loads((folder / 'DATA_AUDIT.json').read_text(encoding='utf-8'))
        local = folder / 'local_only'
        if hashlib.sha256((local / 'INPUTS.npz').read_bytes()).hexdigest() != audit['local_bundle_sha256']:
            raise ValueError('Frozen input bundle changed')
        event_bytes = (local / 'source_events.json').read_bytes()
        if hashlib.sha256(event_bytes).hexdigest() != audit['source_event_sha256']:
            raise ValueError('Prepared source event list changed')
        events = json.loads(event_bytes.decode('utf-8'))
        for event in events:
            if dt.date.fromisoformat(event['available_date']) != dt.date.fromisoformat(event['published']) + dt.timedelta(days=1):
                raise ValueError('Source availability is not literal publication plus one day')
        sd = np.array([event['available_date'] for event in events], dtype='datetime64[D]').astype('int64')
        sv = np.array([event['values'] for event in events])
        with np.load(local / 'INPUTS.npz', allow_pickle=False) as inputs:
            if not all(np.array_equal(inputs[key], value) for key, value in
                       (('price', price), ('dates', dates), ('source_dates', sd), ('source_values', sv))):
                raise ValueError('Raw-price/source-event to training-bundle link mismatch')
        input_identity = True
    logs = []
    for path in sorted(runs.glob('*_b*.json')):
        log = json.loads(path.read_text(encoding='utf-8'))
        if not log['train_max_target'] < log['validation_first_origin'] or not log['validation_max_target'] < log['end']:
            raise ValueError('Training/validation label maturity violation')
        if log['source_fit_date'] > dates[log['train_max_target'] - 20]:
            raise ValueError('Source scaler exceeds training-input support')
        if log['end'] != frozen['blocks'][log['block']]['end']:
            raise ValueError('Refresh clock changed')
        logs.append(log)
    metrics, missing = {}, []
    for mode in names:
        predictions = []
        for seed in (17, 29):
            target = runs / f'{mode}_s{seed}.npz'
            if not target.exists():
                missing.append(dict(mode=mode, seed=seed))
                continue
            with np.load(target, allow_pickle=False) as saved:
                if not np.array_equal(saved['origins'], origins):
                    raise ValueError('Trajectory origin support changed')
                prediction = saved['prediction'].copy()
            pieces = []
            for block, item in enumerate(frozen['blocks']):
                expected_origins = origins[(origins >= item['end']) & (origins < item['end'] + 40)]
                with np.load(runs / f'{mode}_s{seed}_b{block:02d}.npz', allow_pickle=False) as saved:
                    if not np.array_equal(saved['origins'], expected_origins):
                        raise ValueError('Block origin support changed')
                    pieces.append(saved['prediction'].copy())
            if not np.array_equal(np.concatenate(pieces), prediction):
                raise ValueError('Completed trajectory differs from saved fitted blocks')
            predictions.append(prediction)
        if len(predictions) == 2:
            metrics[mode] = score_predictions(np.stack(predictions), truth, years[origins])
    errors = []
    reported_file = folder / 'ALL_MODEL_METRICS.json'
    if reported_file.exists():
        reported = json.loads(reported_file.read_text(encoding='utf-8'))
        if set(reported) != set(metrics):
            raise ValueError('Declared complete model count disagrees with collected support')
        for mode, score in metrics.items():
            other = reported[mode]
            errors.extend(abs(score[k] - other[k]) for k in ('score', 'ensemble'))
            errors.extend(abs(a - b) for a, b in zip(score['seeds'], other['seeds']))
            errors.extend(abs(score['annual'][year] - other['annual'][year]) for year in ('2020', '2021'))
            for a, b in zip(score['cells'], other['cells']):
                if (a['year'], a['horizon'], a['count']) != (b['year'], b['horizon'], b['count']):
                    raise ValueError('Reported cell metadata changed')
                errors.append(abs(a['ratio'] - b['ratio']))
        if max(errors, default=0) > 1e-12:
            raise ValueError('Independent scores disagree with producer')
    complete_file = folder / 'COMPLETE.json'
    causal_file = folder / 'CAUSAL.json'
    marker = json.loads(complete_file.read_text(encoding='utf-8')) if complete_file.exists() else {}
    causal = json.loads(causal_file.read_text(encoding='utf-8')) if causal_file.exists() else []
    causal_ids = {(record['mode'], record['kind']) for record in causal}
    expected_ids = {(mode, kind) for mode in ('hold', 'event') for kind in ('price', 'source')}
    complete = (input_identity and len(metrics) == 6 and len(logs) == frozen.get('expected_market_fits')
                and marker.get('complete', False) and marker.get('market_fits') == len(logs)
                and marker.get('boundary_refits') == 4 and len(causal) == 4 and causal_ids == expected_ids
                and all(record['exact_trace_state_scale_prediction_match'] for record in causal)
                and not (folder / 'INCOMPLETE.json').exists())
    return dict(metrics=metrics, verdict=verdict(metrics, complete), origins=len(origins),
                year_counts={year: int(np.sum(years[origins] == year)) for year in ('2020', '2021')},
                missing_seed_trajectories=missing, saved_completed_market_fits=len(logs),
                original_price_and_source_bundle_link_verified=input_identity,
                frozen_source_hashes_verified=len(frozen_hashes), producer_scalar_comparisons=len(errors),
                producer_max_absolute_difference=max(errors) if errors else None,
                qualification_scope='Independent original-price scoring, not independent market retraining or historical vintage certification')


def verdict(metrics, complete):
    controls = ('price', 'clock', 'raw_hold', 'raw_event', 'hold')
    comparisons = {}
    if 'event' in metrics:
        event = metrics['event']
        for mode in controls:
            if mode not in metrics:
                continue
            other = metrics[mode]
            mean = event['score'] <= .99 * other['score']
            seeds = bool(np.all(np.array(event['seeds']) < np.array(other['seeds'])))
            annual = all(event['annual'][y] <= other['annual'][y] for y in ('2020', '2021'))
            cells = all(a['ratio'] <= 1.05 * b['ratio'] for a, b in zip(event['cells'], other['cells']))
            comparisons[mode] = dict(mean_one_percent=mean, both_seeds=seeds,
                                     both_years_nonworse=annual, cell_guard=cells,
                                     passed=bool(mean and seeds and annual and cells))
    source_gate = None
    witnesses = []
    source_modes = ('raw_hold', 'raw_event', 'hold', 'event')
    if all(mode in metrics for mode in ('price', 'clock')):
        for mode in source_modes:
            if mode in metrics and all(metrics[mode]['score'] <= .99 * metrics[control]['score']
                   and np.all(np.array(metrics[mode]['seeds']) < np.array(metrics[control]['seeds']))
                   for control in ('price', 'clock')):
                witnesses.append(mode)
        if witnesses or all(mode in metrics for mode in source_modes):
            source_gate = bool(witnesses)
    failure = any(not c['passed'] for c in comparisons.values())
    numerical = bool(len(comparisons) == len(controls) and not failure)
    return dict(complete=bool(complete), paper_acceptance_passed=False,
                source_value_gate_passed=source_gate, source_value_witnesses=witnesses,
                event_numerical_gate_passed=numerical,
                decisive_necessary_gate_failure=failure,
                development_gate_passed=bool(complete and source_gate and numerical),
                comparisons=comparisons, independent_novelty_passed=False, unseen_validation=False)


def score_predictions(prediction, truth, years):
    if prediction.ndim != 3 or prediction.shape[1:] != truth.shape or prediction.shape[0] != 2:
        raise ValueError('Exactly two full seed predictions required')
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError('Nonfinite evaluation')
    per_cell, cells, ensemble_cells, annual = [], [], [], {}
    for year in ('2020', '2021'):
        ids = np.flatnonzero(years == year)
        if len(ids) == 0:
            raise ValueError('Missing required calendar year')
        annual_cells = []
        for horizon in (1, 5, 20):
            y = truth[ids, horizon - 1]
            baseline = float(np.dot(y, y) / len(y))
            if baseline <= 0:
                raise ValueError('Undefined random-walk ratio')
            error = prediction[:, ids, horizon - 1] - y
            ratios = np.sum(error * error, axis=1) / (len(ids) * baseline)
            per_cell.append(ratios)
            annual_cells.append(float(np.mean(ratios)))
            average_error = prediction[:, ids, horizon - 1].mean(0) - y
            ensemble_cells.append(float(np.dot(average_error, average_error) / len(ids) / baseline))
            cells.append(dict(year=year, horizon=horizon, count=len(ids), ratio=float(np.mean(ratios))))
        annual[year] = float(np.mean(annual_cells))
    seeds = np.array(per_cell).mean(0)
    return dict(score=float(seeds.mean()), seeds=seeds.tolist(), ensemble=float(np.mean(ensemble_cells)),
                cells=cells, annual=annual)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    result = replay(here, here.parents[2] / 'data/parsed/figshare_28682348/guangzhou_reported.csv')
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        Path(args.output).write_text(text, encoding='utf-8')
    print(text)
