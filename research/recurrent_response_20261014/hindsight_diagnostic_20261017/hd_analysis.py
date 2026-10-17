"""Exposed-development oracle bounds versus previously observable selection."""
import numpy as np
import json
from itertools import combinations
from pathlib import Path
import csv
import io


def summarize(risks, validation, configs):
    risks, validation = np.asarray(risks), np.asarray(validation)
    if risks.shape != (len(configs), 2, 3) or validation.shape != (len(configs), 2) or not np.isfinite(risks).all() or not np.isfinite(validation).all():
        raise ValueError('Invalid complete diagnostic risk table')
    ids = [c['id'] for c in configs]
    annual, scores = risks.mean(2), risks.mean((1, 2))
    fixed = int(scores.argmin())
    yearly, prior = annual.argmin(0), validation.argmin(0)
    cells = risks.argmin(0)
    return dict(hindsight_static=dict(id=ids[fixed], score=float(scores[fixed])),
                hindsight_year_ids=[ids[int(i)] for i in yearly],
                hindsight_year_score=float(np.mean([annual[yearly[y], y] for y in range(2)])),
                hindsight_cell_ids=[[ids[int(i)] for i in row] for row in cells],
                hindsight_cell_score=float(risks.min(0).mean()),
                prior_validation_ids=[ids[int(i)] for i in prior],
                prior_validation_score=float(np.mean([annual[prior[y], y] for y in range(2)])),
                transfer_2020_winner_to_2021=dict(id=ids[int(yearly[0])], score=float(annual[yearly[0], 1])),
                reverse_2021_winner_to_2020=dict(id=ids[int(yearly[1])], score=float(annual[yearly[1], 0])),
                unexposed_test=False, global_optimum_established=False,
                selection_scope='Limited frozen candidate set; hindsight bounds are not deployable forecast rules')


def paired_effects(configs, risks):
    result = {}
    subset = [(i, c) for i, c in enumerate(configs) if c.get('slice') == 'factorial']
    for factor in ('family', 'group', 'length', 'anchor', 'shrink'):
        values = sorted({c[factor] for _, c in subset})
        result[factor] = {}
        for left, right in combinations(values, 2):
            def pool(value):
                return {json.dumps({k: v for k, v in c.items() if k not in ('id', 'slice', factor)}, sort_keys=True): i
                        for i, c in subset if c[factor] == value}
            a, b = pool(left), pool(right)
            keys = sorted(set(a) & set(b))
            difference = np.stack([risks[b[k]] - risks[a[k]] for k in keys])
            means = difference.mean((1, 2))
            result[factor][f'{left}->{right}'] = dict(pairs=len(keys), mean_right_minus_left=float(means.mean()),
                        median_right_minus_left=float(np.median(means)), fraction_right_better=float(np.mean(means < 0)),
                        annual_right_minus_left=difference.mean((0, 2)).tolist())
    return result


def analyze(directory):
    directory = Path(directory)
    if not (directory / 'PRE_RUN.json').exists():
        raise ValueError('Missing frozen run; no retrospective ranking')
    before = json.loads((directory / 'PRE_RUN.json').read_text(encoding='utf-8'))
    configs = before['configs']
    root = directory / 'local_only'
    expected = {f'{cid}_t{cut}' for cid, cut in before['tasks']}
    if ({p.stem for p in (root / 'fits').glob('*.json')} != expected or
            {p.stem for p in (root / 'fits').glob('*.npz')} != expected):
        raise ValueError('Incomplete or unexpected grid; no partial winner')
    with np.load(root / 'INPUTS.npz', allow_pickle=False) as z:
        price, dates, reference = np.array(z['price']), np.array(z['dates']), np.array(z['reference'])
    ids = np.arange(459, 930)
    predictions, validations, records = [], [], []
    for config in configs:
        blocks, valid = [], []
        for cut in (459, 703):
            name = f"{config['id']}_t{cut}"
            record = json.loads((root / 'fits' / (name + '.json')).read_text(encoding='utf-8'))
            if record['config'] != config or record['end'] != cut:
                raise ValueError('Saved configuration identity differs')
            with np.load(root / 'fits' / (name + '.npz'), allow_pickle=False) as z:
                expected_ids = np.arange(cut, 703 if cut == 459 else 930)
                if not np.array_equal(z['origins'], expected_ids) or z['predictions'].shape != (len(expected_ids), 20):
                    raise ValueError('Saved forecast support differs')
                blocks.append(np.array(z['predictions']))
            valid.append(record['validation_primary'])
            records.append({k: v for k, v in record.items() if k != 'trace'})
        predictions.append(np.concatenate(blocks))
        validations.append(valid)
    configs = configs + [dict(id='rw', family='reference', group='reference', anchor='rw'),
                         dict(id='stat', family='reference', group='reference', anchor='stat')]
    predictions.extend([np.zeros((len(ids), 20)), reference[ids]])
    for control in ('rw', 'stat'):
        values = []
        for cut in (459, 703):
            t = np.arange(cut - 80, cut - 20)
            target = price[t[:, None] + np.array([1, 5, 20])] - price[t, None]
            prediction = np.zeros_like(target) if control == 'rw' else reference[t][:, [0, 4, 19]]
            values.append(float(np.mean(np.mean((prediction - target) ** 2, axis=0) / np.mean(target ** 2, axis=0))))
        validations.append(values)
    predictions, validations = np.stack(predictions), np.array(validations)
    risks = np.zeros((len(configs), 2, 3))
    pointwise = np.zeros((2, 3))
    counts = []
    for yi, year in enumerate(('2020', '2021')):
        mask = np.char.startswith(dates[ids], year) & np.char.startswith(dates[ids + 20], year)
        t = ids[mask]
        counts.append(len(t))
        for hi, h in enumerate((1, 5, 20)):
            target = price[t + h] - price[t]
            squared = (predictions[:, mask, h - 1] - target) ** 2
            denominator = np.mean(target ** 2)
            risks[:, yi, hi] = squared.mean(1) / denominator
            pointwise[yi, hi] = squared.min(0).mean() / denominator
    if counts != [224, 227] or not np.isfinite(risks).all():
        raise ValueError('Primary score support differs')
    rows = []
    for i, config in enumerate(configs):
        rows.append(dict(id=config['id'], config=config, score=float(risks[i].mean()),
                         annual=dict(zip(('2020', '2021'), risks[i].mean(1).tolist())),
                         cells=risks[i].tolist(), prior_validation=validations[i].tolist()))
    selections = summarize(risks, validations, configs)
    selections.update(pointwise_oracle_score=float(pointwise.mean()), pointwise_oracle_cells=pointwise.tolist(),
                      pointwise_oracle_is_deployable=False)
    best_by_family = {family: min((r for r in rows if r['config']['family'] == family), key=lambda r: r['score'])
                      for family in ('ridge', 'gru', 'lstm', 'reference')}
    result = dict(formal_fits=len(records), trained_configs=len(before['configs']), controls=2, score_origins=sum(counts),
                  year_origin_counts=counts, seed=17, selections=selections, rows=rows, best_by_family=best_by_family,
                  unexposed_test=False, novelty_established=False, diagnostic_only=True)
    effects = dict(sign='Negative right-minus-left favors the right setting; these are exposed finite-grid sensitivity results, not causal market effects',
                   factorial=paired_effects(configs, risks))
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['id', 'family', 'feature_group', 'window', 'anchor', 'shrink', 'hidden', 'lr', 'alpha', 'slice', 'score',
                     '2020', '2021', 'prior_validation_2020', 'prior_validation_2021', '2020_h1', '2020_h5', '2020_h20', '2021_h1', '2021_h5', '2021_h20'])
    for row in sorted(rows, key=lambda r: r['score']):
        c = row['config']
        writer.writerow([row['id'], c['family'], c['group'], c.get('length'), c['anchor'], c.get('shrink'), c.get('hidden'),
                         c.get('lr'), c.get('alpha'), c.get('slice', 'reference'), row['score'], row['annual']['2020'], row['annual']['2021'],
                         *row['prior_validation'], *np.ravel(row['cells']).tolist()])
    return result, effects, records, output.getvalue()
