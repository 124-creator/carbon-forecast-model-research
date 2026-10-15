"""Independent raw-prefix scoring. Does not import the model, trainer or runner."""
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODES = ['window_joint', 'full_joint', 'window_profile', 'full_profile',
         'full_prequential_complete', 'full_prequential_mature']
SEEDS = [17, 29]


def check_fit_log(row):
    if not row['train_max_target'] < row['validation_first_origin']:
        raise ValueError('Training targets overlap validation origins')
    if not row['validation_max_target'] < row['end'] or row['parameters'] != 1232:
        raise ValueError('Fit boundary or parameter count mismatch')


def verify():
    with (ROOT / 'data/parsed/figshare_28682348/guangzhou_reported.csv').open('rb', buffering=0) as handle:
        prefix = handle.read(29204)
    if hashlib.sha256(prefix).hexdigest() != '1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070':
        raise ValueError('Wrong price prefix')
    rows = list(csv.DictReader(io.StringIO(prefix.decode('utf-8-sig'))))
    if len(rows) != 950 or rows[-1]['date'] != '2021-12-31':
        raise ValueError('Wrong prefix boundary')
    price = np.array([float(row['author_reported_price']) for row in rows])
    dates = [row['date'] for row in rows]
    every_origin = np.arange(459, 930)
    keep = np.array([dates[t][:4] == dates[t + 20][:4] for t in every_origin])
    origins = every_origin[keep]
    years = np.array([dates[t][:4] for t in origins])
    if len(origins) != 451 or int(sum(years == '2020')) != 224 or int(sum(years == '2021')) != 227:
        raise ValueError('Origin population mismatch')
    actual = np.array([[price[t + h] - price[t] for h in range(1, 21)] for t in origins])
    result = json.loads((HERE / 'RESULTS.json').read_text(encoding='utf-8'))
    log = json.loads((HERE / 'FIT_LOGS.json').read_text(encoding='utf-8'))
    grid = {(mode, seed, block) for mode in MODES for seed in SEEDS for block in range(12)}
    if len(log) != 144 or {(r['mode'], r['seed'], r['block']) for r in log} != grid:
        raise ValueError('Incomplete or duplicated 144-fit grid')
    for row in log:
        check_fit_log(row)
        if row['end'] != 459 + row['block'] * 40 or row['epochs'] != len(row['history']) or row['epochs'] > 80:
            raise ValueError('Unexpected budget or refresh')
    pre = json.loads((HERE / 'PRE_RUN.json').read_text(encoding='utf-8'))
    for name, expected in pre['source_sha256'].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Executed-source snapshot changed: ' + name)
    comparisons = []
    replays = {}
    for mode in MODES:
        with np.load(HERE / 'local_only' / (mode + '.npz'), allow_pickle=False) as saved:
            prediction = saved['prediction']
        if prediction.shape != (2, 451, 20) or not np.isfinite(prediction).all():
            raise ValueError('Invalid numeric prediction array')
        for si, seed in enumerate(SEEDS):
            pieces = []
            for block in range(12):
                selected = every_origin[(every_origin - 459) // 40 == block]
                path = HERE / 'local_only' / f'{mode}_s{seed}_b{block:02d}.npz'
                with np.load(path, allow_pickle=False) as batch:
                    if not np.array_equal(batch['origins'], selected):
                        raise ValueError('Prediction origin order mismatch')
                    count = sum(batch[name].size for name in batch.files if name.startswith('state__'))
                    if count != 1232:
                        raise ValueError('Saved parameter count mismatch')
                    pieces.append(batch['prediction'])
            if not np.array_equal(np.concatenate(pieces)[keep], prediction[si]):
                raise ValueError('Per-refresh and final predictions differ')
        per_seed = []
        ensemble = []
        year_scores = {}
        cell_scores = []
        for year in ['2020', '2021']:
            mask = years == year
            subtotal = []
            for h in [1, 5, 20]:
                denominator = (actual[mask, h - 1] ** 2).sum()
                ratios = ((prediction[:, mask, h - 1] - actual[None, mask, h - 1]) ** 2).sum(axis=1) / denominator
                per_seed.append(ratios)
                subtotal.append(float(ratios.mean()))
                cell_scores.append(float(ratios.mean()))
                ensemble.append(float(((prediction[:, mask, h - 1].mean(0) - actual[mask, h - 1]) ** 2).sum() / denominator))
            year_scores[year] = float(np.mean(subtotal))
        seed_scores = np.array(per_seed).mean(axis=0)
        score = float(seed_scores.mean())
        stored = result['scores'][mode]
        discrepancies = [abs(score - stored['score']), abs(np.mean(ensemble) - stored['ensemble'])]
        discrepancies.extend(abs(seed_scores - stored['seeds']))
        discrepancies.extend(abs(year_scores[y] - stored['annual'][y]) for y in year_scores)
        discrepancies.extend(abs(a - b['ratio']) for a, b in zip(cell_scores, stored['cells']))
        if max(discrepancies) > 1e-12:
            raise ValueError('Independent scoring mismatch: ' + mode)
        comparisons.extend(discrepancies)
        replays[mode] = dict(score=score, annual=year_scores, seeds=seed_scores.tolist(), cells=cell_scores)
    checks = json.loads((HERE / 'CAUSAL.json').read_text(encoding='utf-8'))
    if len(checks) != 4 or sum(c['actual_fits'] for c in checks) != 8:
        raise ValueError('Representative full-refit checks incomplete')
    if not all(c['parameters_equal'] and c['first_fit_equal'] and c['predictions_equal'] for c in checks):
        raise ValueError('Representative refit check failed')
    return dict(verified=True, formal_fits=144, additional_full_refits=8, origins=451,
                maximum_score_difference=float(max(comparisons)), source_snapshot_matches=True,
                independent_label_reconstruction=True, scores=replays,
                scope='Independent scoring, numeric snapshot and fit-log checks, not independent training implementation or novelty validation')


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2, allow_nan=False))
