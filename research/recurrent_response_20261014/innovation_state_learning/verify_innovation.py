"""Independent original-price score replay. Imports no model, trainer, IO helper or runner."""
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODES = ('innovation_shift', 'raw_shift', 'innovation_diagonal', 'constant_shift',
         'gru_profile', 'gru_rank_window', 'lstm_profile')


def independent_metrics(prediction, prices, dates, origins):
    if prediction.ndim != 3 or prediction.shape[1:] != (len(origins), 20) or not np.isfinite(prediction).all():
        raise ValueError('Invalid prediction shape/value')
    if not np.all(dates[origins].astype('U4') == dates[origins + 20].astype('U4')):
        raise ValueError('Cross-year targets not eligible')
    cell_seed_losses, cell_ensemble_losses, cells = [], [], []
    annual = {}
    for year in ('2020', '2021'):
        members = np.flatnonzero(dates[origins].astype('U4') == year)
        if len(members) == 0:
            raise ValueError('Required year absent')
        annual_cells = []
        for horizon in (1, 5, 20):
            truth = prices[origins[members] + horizon] - prices[origins[members]]
            rw_loss = float(np.dot(truth, truth) / len(truth))
            if rw_loss <= 0:
                raise ValueError('Undefined reference loss')
            values = []
            for seed in range(len(prediction)):
                difference = prediction[seed, members, horizon - 1] - truth
                values.append(float(np.dot(difference, difference) / len(truth) / rw_loss))
            ensemble_error = np.mean(prediction[:, members, horizon - 1], axis=0) - truth
            cell_ensemble_losses.append(float(np.dot(ensemble_error, ensemble_error) / len(truth) / rw_loss))
            cell_seed_losses.append(values)
            mean_value = float(np.mean(values))
            annual_cells.append(mean_value)
            cells.append(dict(year=year, horizon=horizon, count=len(members), ratio=mean_value))
        annual[year] = float(np.mean(annual_cells))
    seed_scores = np.mean(np.array(cell_seed_losses), axis=0)
    return dict(score=float(seed_scores.mean()), seeds=seed_scores.tolist(), ensemble=float(np.mean(cell_ensemble_losses)),
                cells=cells, annual=annual)


def partial_verdict(scores):
    ordinary = [name for name in ('constant_shift', 'gru_profile', 'gru_rank_window', 'lstm_profile') if name in scores]
    verdict: dict[str, object] = dict(complete=False, acceptance_passed=False, decisive_necessary_gate_failure=False)
    if 'innovation_shift' in scores and ordinary:
        best = min(ordinary, key=lambda name: scores[name]['score'])
        improvement = 1 - scores['innovation_shift']['score'] / scores[best]['score']
        verdict.update(best_completed_ordinary=best, improvement_over_completed_ordinary=improvement,
                       decisive_necessary_gate_failure=bool(improvement < .01))
    return verdict


def replay_incomplete():
    """Never accepts partial results; may falsify a necessary frozen condition."""
    incomplete = json.loads((HERE / 'INCOMPLETE.json').read_text(encoding='utf-8'))
    if incomplete['complete']:
        raise ValueError('Expected an explicitly incomplete run')
    with (ROOT / 'data/parsed/figshare_28682348/guangzhou_reported.csv').open('rb', buffering=0) as source:
        raw = source.read(29204)
    if len(raw) != 29204 or hashlib.sha256(raw).hexdigest() != '1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070':
        raise ValueError('Approved raw prefix mismatch')
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    if len(rows) != 950 or rows[-1]['date'] != '2021-12-31':
        raise ValueError('Approved boundary mismatch')
    price = np.array([float(row['author_reported_price']) for row in rows])
    dates = np.array([row['date'] for row in rows])
    all_origins = np.arange(459, 930)
    eligible = np.array([dates[t][:4] == dates[t + 20][:4] for t in all_origins])
    origins = all_origins[eligible]
    if len(origins) != 451:
        raise ValueError('Independent support mismatch')
    before = json.loads((HERE / 'PRE_RUN.json').read_text(encoding='utf-8'))
    for name, expected in before['source_sha256'].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen source changed: ' + name)
    scores, excluded = {}, []
    for mode in MODES:
        seeds = (17,) if mode == 'constant_shift' else (17, 29)
        paths = [HERE / 'local_only' / f'{mode}_s{seed}.npz' for seed in seeds]
        if not all(path.exists() for path in paths):
            excluded.append(mode)
            continue
        arrays = []
        for seed, path in zip(seeds, paths):
            with np.load(path, allow_pickle=False) as saved:
                array = saved['prediction'].copy()
            if array.shape != (471, 20) or not np.isfinite(array).all():
                raise ValueError('Incomplete trajectory array')
            for block in range(12):
                selection = np.flatnonzero((all_origins - 459) // 40 == block)
                with np.load(HERE / 'local_only' / f'{mode}_s{seed}_b{block:02d}.npz', allow_pickle=False) as saved:
                    if not np.array_equal(saved['origins'], all_origins[selection]) or not np.array_equal(saved['prediction'], array[selection]):
                        raise ValueError('Block/trajectory mismatch')
            arrays.append(array[eligible])
        scores[mode] = independent_metrics(np.stack(arrays), price, dates, origins)
    logs = [json.loads(path.read_text(encoding='utf-8')) for path in (HERE / 'local_only').glob('*_b*.json')]
    keys = {(row['mode'], row['seed'], row['block']) for row in logs}
    expected_keys = {(mode, seed, block) for mode, seed in before['jobs'] for block in range(12)}
    if len(keys) != len(logs) or not keys.issubset(expected_keys):
        raise ValueError('Invalid completed-fit register')
    for row in logs:
        if not row['train_max_target'] < row['validation_first_origin'] or not row['validation_max_target'] < row['end']:
            raise ValueError('Stored maturity/purge violation')
        if row['epochs'] > 60:
            raise ValueError('Epoch budget violation')
    return dict(complete=False, replay_verified=True, acceptance_passed=False,
                expected_market_fits=len(expected_keys), saved_completed_block_fits=len(logs),
                fits_in_complete_trajectories=incomplete['completed_trajectory_fits'],
                incomplete_or_excluded_modes=excluded, scored_complete_modes=len(scores), origins=len(origins),
                source_freeze_unchanged=True, boundary_refits_executed=0, independent_retraining=False,
                scores=scores, verdict=partial_verdict(scores))


def main():
    with (ROOT / 'data/parsed/figshare_28682348/guangzhou_reported.csv').open('rb', buffering=0) as source:
        raw = source.read(29204)
    if len(raw) != 29204 or hashlib.sha256(raw).hexdigest() != '1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070':
        raise ValueError('Approved prefix mismatch')
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    if len(rows) != 950 or rows[-1]['date'] != '2021-12-31':
        raise ValueError('Raw prefix boundary mismatch')
    prices = np.array([float(row['author_reported_price']) for row in rows])
    dates = np.array([row['date'] for row in rows])
    all_origins = np.arange(459, 930)
    origins = np.array([t for t in all_origins if dates[t][:4] == dates[t + 20][:4]])
    counts = {year: sum(dates[t][:4] == year for t in origins) for year in ('2020', '2021')}
    if len(origins) != 451 or counts != {'2020': 224, '2021': 227}:
        raise ValueError('Independent common support mismatch')
    result = json.loads((HERE / 'RESULTS.json').read_text(encoding='utf-8'))
    if not result['complete']:
        raise ValueError('Cannot certify an incomplete run')
    scores, errors = {}, []
    for mode in MODES:
        with np.load(HERE / 'local_only' / (mode + '.npz'), allow_pickle=False) as saved:
            pr = saved['prediction'].copy()
        expected_seeds = 1 if mode == 'constant_shift' else 2
        if pr.shape != (expected_seeds, 451, 20):
            raise ValueError('Saved seed/origin count mismatch')
        value = independent_metrics(pr, prices, dates, origins)
        scores[mode] = value
        claimed = result['scores'][mode]
        for name in ('score', 'ensemble'):
            errors.append(abs(value[name] - claimed[name]))
        errors.extend(abs(a - b) for a, b in zip(value['seeds'], claimed['seeds']))
        errors.extend(abs(a['ratio'] - b['ratio']) for a, b in zip(value['cells'], claimed['cells']))
        errors.extend(abs(value['annual'][year] - claimed['annual'][year]) for year in ('2020', '2021'))
    best = min(('constant_shift', 'gru_profile', 'gru_rank_window', 'lstm_profile'), key=lambda key: scores[key]['score'])
    candidate = scores['innovation_shift']
    improvement = 1 - candidate['score'] / scores[best]['score']
    seed_wins = int(np.sum(np.asarray(candidate['seeds']) < np.asarray(scores[best]['seeds'])))
    cell_wins = sum(a['ratio'] < b['ratio'] for a, b in zip(candidate['cells'], scores[best]['cells']))
    gate = improvement >= .01 and seed_wins == 2 and cell_wins >= 4
    gate = gate and all(candidate['score'] < scores[mode]['score'] for mode in ('raw_shift', 'innovation_diagonal', 'constant_shift'))
    if gate != result['candidate_gate'] or best != result['best_ordinary']:
        raise ValueError('Independent candidate-gate mismatch')
    logs = json.loads((HERE / 'FIT_LOGS.json').read_text(encoding='utf-8'))
    keys = {(r['mode'], r['seed'], r['block']) for r in logs}
    if len(keys) != 156 or len(logs) != 156:
        raise ValueError('Complete unique fit total mismatch')
    for row in logs:
        if not row['train_max_target'] < row['validation_first_origin'] or not row['validation_max_target'] < row['end']:
            raise ValueError('Stored maturity/purge metadata failed')
        if row['epochs'] > 60 or row['selected_epoch'] > row['epochs']:
            raise ValueError('Frozen epoch budget failed')
    before = json.loads((HERE / 'PRE_RUN.json').read_text(encoding='utf-8'))
    for name, expected in before['source_sha256'].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Pre-run source changed: ' + name)
    largest = max(errors)
    if largest > 1e-12:
        raise ValueError('Independent scalar replay failed')
    report = dict(verified=True, origins=len(origins), origin_counts=counts, modes=len(scores), scalar_comparisons=len(errors),
                  max_abs_score_error=largest, unique_market_fits=len(keys), source_freeze_unchanged=True,
                  independently_replayed_gate=bool(gate), independent_retraining=False, scores=scores)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
