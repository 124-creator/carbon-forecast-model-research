"""Original-price score replay; not an independent training implementation."""
import numpy as np
from pathlib import Path
import hashlib
import json
import sys


def score_raw_prices(prediction, actual, issued, years):
    years = np.asarray(years)
    cells = []
    for year in ['2020', '2021']:
        mask = years == year
        for h in [1, 5, 20]:
            denominator = float(np.mean((actual[mask, h-1]-issued[mask])**2))
            if not np.any(mask) or denominator <= 0:
                raise ValueError('Missing year or invalid RW denominator')
            levels = prediction[:, mask, h-1]+issued[mask][None]
            mse = float(np.mean(np.mean((levels-actual[mask, h-1][None])**2, axis=1)))
            cells.append(dict(year=year, h=h, ratio=mse/denominator))
    return float(np.mean([cell['ratio'] for cell in cells])), cells


if __name__ == '__main__':
    directory = Path(__file__).resolve().parent
    sys.path.insert(0, str(directory.parents[1]/'torch_segmented_backbone_20261014'))
    from prefix_loader import read_prefix
    rows, audit = read_prefix(directory.parents[2]/'data/parsed/figshare_28682348/guangzhou_reported.csv', 950, '2021-12-31')
    prices = np.array([float(row['author_reported_price']) for row in rows])
    dates = [row['date'] for row in rows]
    origins = np.array([i for i in range(459, len(prices)-20) if dates[i][:4] == dates[i+20][:4]])
    years = np.array([dates[i][:4] for i in origins])
    actual = np.stack([prices[origins+h] for h in range(1, 21)], axis=1)
    kinds = ['plain', 'fixed', 'global', 'coupled']
    results = json.loads((directory/'RESULTS.json').read_text())
    logs = json.loads((directory/'FIT_LOGS.json').read_text())
    causal = json.loads((directory/'CAUSAL.json').read_text())
    if len(origins) != 451 or not results['complete'] or results['fits'] != 96 or len(logs) != 96:
        raise ValueError('Experiment is not complete')
    expected = {(kind, seed, block) for kind in kinds for seed in [17, 29] for block in range(12)}
    if {(row['kind'], row['seed'], row['block']) for row in logs} != expected:
        raise ValueError('Fit grid mismatch')
    for row in logs:
        end = 459+40*row['block']
        mature = list(range(max(60, end-500), end-20))
        if row['fit_end'] != end or row['n'] != len(mature):
            raise ValueError('Incorrect mature support')
        if row['max_training_target'] != mature[-81]+20 or row['validation_first'] != mature[-60] or row['max_validation_target'] != end-1:
            raise ValueError('Incorrect purge or target indexing')
        if not row['max_training_target'] < row['validation_first'] or not row['max_validation_target'] < end:
            raise ValueError('Target leakage')
        if len(row['history']) != row['epochs'] or not 0 <= row['selected'] <= row['epochs'] <= 200 or not np.isfinite(row['history']).all():
            raise ValueError('Invalid optimization receipt')
        in_features = 1 if row['kind'] == 'plain' else 4
        parameters = 3*16*(in_features+16+2)+(16+1)*20+{'plain': 0, 'fixed': 0, 'global': 3, 'coupled': 51}[row['kind']]
        if row['parameters'] != parameters or row['active_parameters'] != parameters:
            raise ValueError('Incorrect parameter count')
        if row['kind'] != 'plain' and row['diagnostics']['reconstruction_max_error'] > 1e-4:
            raise ValueError('Numerical component conservation failed')
    if len(causal) != 4 or {row['kind'] for row in causal} != set(kinds) or not all(row['passed'] and row['seed'] == 17 and row['fit_end'] == 459 for row in causal):
        raise ValueError('Representative actual future-refit receipts missing')
    pre = json.loads((directory/'PRE_RUN.json').read_text())
    for name, expected_hash in pre['source_sha256'].items():
        if hashlib.sha256((directory.parents[2]/name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Executed source changed: '+name)
    scores, per_year = {}, {}
    with np.load(directory/'local_only/PREDICTIONS.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['origins'], origins) or not np.array_equal(saved['years'], years):
            raise ValueError('Issuance support mismatch')
        if not np.array_equal(saved['truth'], actual-prices[origins, None]):
            raise ValueError('Original target mismatch')
        for kind in kinds:
            prediction = np.stack([saved[kind+'_'+str(seed)] for seed in [17, 29]])
            if prediction.shape != (2, 451, 20) or not np.isfinite(prediction).all():
                raise ValueError('Predictions incomplete')
            value, cells = score_raw_prices(prediction, actual, prices[origins], years)
            if not np.isclose(value, results['scores'][kind]['score'], atol=1e-12, rtol=0):
                raise ValueError('Score mismatch')
            for first, second in zip(cells, results['scores'][kind]['cells']):
                if first['year'] != second['year'] or first['h'] != second['h'] or not np.isclose(first['ratio'], second['ratio'], atol=1e-12, rtol=0):
                    raise ValueError('Cell mismatch')
            scores[kind] = value
            per_year[kind] = {year: float(np.mean([cell['ratio'] for cell in cells if cell['year'] == year])) for year in ['2020', '2021']}
    mechanism = all(scores['coupled'] < scores[kind] for kind in ['plain', 'fixed', 'global']) and all(per_year['coupled'][year] <= per_year['global'][year] for year in ['2020', '2021'])
    escalation = mechanism and scores['coupled'] < .9497932301207621
    if results['mechanism_gate'] != mechanism or results['direction_escalation_gate'] != escalation:
        raise ValueError('Decision gate mismatch')
    print(json.dumps(dict(passed=True, optimized=not __debug__, fits=96, origins=len(origins), scores=scores,
        mechanism_gate=mechanism, direction_escalation_gate=escalation, input_audit=audit,
        scope='Independent original-price score replay and receipt audit, not independent retraining'), ensure_ascii=False))
