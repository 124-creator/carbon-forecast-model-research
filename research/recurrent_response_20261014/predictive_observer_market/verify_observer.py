"""Independent raw-price score replay; not independent retraining."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

R = Path(__file__).resolve().parent
KINDS = ['GRUpatch', 'RKNglobal', 'RKNconditional']


def score_raw_prices(prediction, actual, issued, years):
    cells = []
    for year in ['2020', '2021']:
        mask = years == year
        for h in [1, 5, 20]:
            rw_mse = np.mean((actual[mask, h-1] - issued[mask]) ** 2)
            estimates = prediction[:, mask, h-1] + issued[mask][None]
            mse = np.mean(np.mean((estimates - actual[mask, h-1][None]) ** 2, axis=1))
            cells.append(dict(year=year, h=h, ratio=float(mse / rw_mse)))
    return float(np.mean([c['ratio'] for c in cells])), cells


if __name__ == '__main__':
    sys.path.insert(0, str(R.parents[1] / 'torch_segmented_backbone_20261014'))
    from prefix_loader import read_prefix
    rows, audit = read_prefix(R.parents[2] / 'data/parsed/figshare_28682348/guangzhou_reported.csv', 950, '2021-12-31')
    p = np.array([float(row['author_reported_price']) for row in rows])
    dates = [row['date'] for row in rows]
    origins = np.array([i for i in range(459, len(p)-20) if dates[i][:4] == dates[i+20][:4]])
    years = np.array([dates[i][:4] for i in origins])
    actual = np.stack([p[origins + h] for h in range(1, 21)], axis=1)
    result = json.loads((R / 'RESULTS.json').read_text())
    logs = json.loads((R / 'FIT_LOGS.json').read_text())
    causal = json.loads((R / 'CAUSAL.json').read_text())
    if len(origins) != 451 or not result['complete'] or result['fits'] != 72 or len(logs) != 72:
        raise ValueError('Incomplete experiment')
    expected = {(kind, seed, block) for kind in KINDS for seed in [17, 29] for block in range(12)}
    if {(row['kind'], row['seed'], row['block']) for row in logs} != expected:
        raise ValueError('Fit index mismatch')
    for row in logs:
        end = 459 + 40 * row['block']
        mature = list(range(max(60, end-500), end-20))
        if row['fit_end'] != end or row['n'] != len(mature):
            raise ValueError('Mature support mismatch')
        if row['max_training_target'] != mature[-81] + 20 or row['validation_first'] != mature[-60] or row['max_validation_target'] != end-1:
            raise ValueError('Purging index mismatch')
        if not row['max_training_target'] < row['validation_first'] or not row['max_validation_target'] < end:
            raise ValueError('Immature training or validation target')
        if len(row['history']) != row['epochs'] or not 0 <= row['selected'] <= row['epochs'] <= 200:
            raise ValueError('Invalid epoch selection')
        if not np.isfinite(row['history']).all():
            raise ValueError('Invalid training history')
        if row['parameters'] != (1444 if row['kind'] == 'GRUpatch' else 1311):
            raise ValueError('Parameter count mismatch')
    if len(causal) != 3 or {row['kind'] for row in causal} != set(KINDS) or not all(row['passed'] for row in causal):
        raise ValueError('Representative future refit receipts missing')
    pre = json.loads((R / 'PRE_RUN.json').read_text())
    for name, expected_hash in pre['source_sha256'].items():
        if hashlib.sha256((R.parents[2] / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Executed-source snapshot changed: ' + name)
    scores, year_scores = {}, {}
    with np.load(R / 'local_only/PREDICTIONS.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['origins'], origins) or not np.array_equal(saved['years'], years):
            raise ValueError('Evaluation origins mismatch')
        if not np.array_equal(saved['truth'], actual - p[origins, None]):
            raise ValueError('Saved targets disagree with permitted original price prefix')
        for kind in KINDS:
            prediction = np.stack([saved[kind + '_' + str(seed)] for seed in [17, 29]])
            if prediction.shape != (2, 451, 20) or not np.isfinite(prediction).all():
                raise ValueError('Missing predictions')
            score, cells = score_raw_prices(prediction, actual, p[origins], years)
            if not np.isclose(score, result['scores'][kind]['score'], atol=1e-12, rtol=0):
                raise ValueError('Score mismatch')
            for a, b in zip(cells, result['scores'][kind]['cells']):
                if a['year'] != b['year'] or a['h'] != b['h'] or not np.isclose(a['ratio'], b['ratio'], atol=1e-12, rtol=0):
                    raise ValueError('Cell mismatch')
            scores[kind] = score
            year_scores[kind] = {year: float(np.mean([c['ratio'] for c in cells if c['year'] == year])) for year in ['2020', '2021']}
    gate = all(scores['RKNconditional'] < scores[k] and all(year_scores['RKNconditional'][y] <= year_scores[k][y] for y in ['2020', '2021']) for k in ['GRUpatch', 'RKNglobal'])
    if result['development_gate'] != gate:
        raise ValueError('Gate mismatch')
    output = dict(passed=True, optimized=not __debug__, fits=72, origins=len(origins), scores=scores,
                  development_gate=gate, original_prefix=audit,
                  scope='original-prefix raw-price replay and receipt audit; not independent retraining')
    print(json.dumps(output, ensure_ascii=False))
