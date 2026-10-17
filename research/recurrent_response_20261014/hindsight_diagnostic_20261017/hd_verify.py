"""Independent hindsight risk replay: no producer/trainer/feature imports."""
import numpy as np
import argparse
import csv
from datetime import date
import hashlib
import io
import json
from pathlib import Path


def risk_table(price, dates, ids, forecasts):
    ids = np.asarray(ids, dtype=np.int64)
    if (ids.ndim != 1 or not len(ids) or ids.min() < 0 or np.any(np.diff(ids) <= 0) or
            ids.max() + 20 >= len(price) or forecasts.ndim != 3 or
            forecasts.shape[1:] != (len(ids), 20) or not np.isfinite(forecasts).all()):
        raise ValueError('Invalid independent scoring support')
    table, counts = np.zeros((len(forecasts), 2, 3)), []
    for yi, year in enumerate(('2020', '2021')):
        positions = [i for i, t in enumerate(ids) if str(dates[t])[:4] == year and str(dates[t + 20])[:4] == year]
        if not positions:
            raise ValueError('Missing complete same-year score support')
        counts.append(len(positions))
        origin = ids[positions]
        for hi, h in enumerate((1, 5, 20)):
            target = price[origin + h] - price[origin]
            denominator = float(np.dot(target, target))
            if denominator <= 0 or not np.isfinite(denominator):
                raise ValueError('Invalid RW denominator')
            for ci, prediction in enumerate(forecasts):
                error = prediction[positions, h - 1] - target
                table[ci, yi, hi] = np.dot(error, error) / denominator
    return table, counts


def replay(directory):
    directory = Path(directory)
    if not (directory / 'RUN_COMPLETE.json').exists() or not (directory / 'PRE_RUN.json').exists():
        raise ValueError('No complete frozen run to verify')
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    before, complete = read(directory / 'PRE_RUN.json'), read(directory / 'RUN_COMPLETE.json')
    if len(before['configs']) != 96 or complete['formal_fits'] != 192:
        raise ValueError('Formal search count differs')
    for name, digest in before['source_sha256'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen source changed: ' + name)
    original = directory.parents[2] / 'data/parsed/figshare_28682348/guangzhou_reported.csv'
    with original.open('rb', buffering=0) as stream:
        raw = stream.read(29204)
    if len(raw) != 29204 or hashlib.sha256(raw).hexdigest() != '1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070':
        raise ValueError('Original approved prefix differs')
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    price = np.array([float(r['author_reported_price']) for r in rows])
    dates = np.array([r['date'] for r in rows])
    if len(price) != 950 or dates[-1] != '2021-12-31':
        raise ValueError('Approved original row/date boundary differs')
    root = directory / 'local_only'
    with np.load(root / 'INPUTS.npz', allow_pickle=False) as z:
        if not np.array_equal(z['price'], price) or not np.array_equal(z['dates'], dates):
            raise ValueError('Numeric input buffer differs')
        reference = np.array(z['reference'])
    expected = {f"{c['id']}_t{t}" for c in before['configs'] for t in (459, 703)}
    if {p.stem for p in (root / 'fits').glob('*.json')} != expected or {p.stem for p in (root / 'fits').glob('*.npz')} != expected:
        raise ValueError('Missing, duplicate or unexpected fit')
    delta = np.r_[0., np.diff(price)]
    raw_columns = [delta, np.log(price)]
    for length in (5, 20):
        raw_columns.append(np.array([0. if t == 0 else np.mean(np.diff(price[max(0, t - length):t + 1])) for t in range(len(price))]))
    raw_columns.append(np.array([0. if t == 0 else np.std(np.diff(price[max(0, t - 20):t + 1])) for t in range(len(price))]))
    day = [date.fromisoformat(str(x)) for x in dates]
    gap = np.array([0] + [(day[i] - day[i - 1]).days for i in range(1, len(day))])
    weekday = np.array([d.weekday() for d in day])
    banks = {'price': delta[:, None], 'dynamics': np.stack(raw_columns, axis=1),
             'calendar': np.stack(raw_columns + [np.log1p(gap), np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7)], axis=1)}
    differences, records, predictions, validation_values = [], [], [], []
    def compare(a, b):
        if np.shape(a) != np.shape(b):
            raise ValueError('Numeric shape differs')
        values = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).ravel()
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite independent comparison')
        differences.extend(values.tolist())
    for c in before['configs']:
        blocks, v = [], []
        for cut in (459, 703):
            name = f"{c['id']}_t{cut}"
            record = read(root / 'fits' / (name + '.json'))
            ids = np.arange(max(60, cut - 500), cut - 20)
            train, validation = ids[:-80], ids[-60:]
            if record['config'] != c or record['seed'] != 17 or record['end'] != cut or record['train_n'] != len(train) or record['validation_n'] != 60:
                raise ValueError('Training identity/support differs')
            if record['train_max_target'] != int(train[-1] + 20) or record['validation_first_origin'] != int(validation[0]) or record['validation_max_target'] >= cut:
                raise ValueError('Label maturity/purge differs')
            if record['completed_monotonic'] > before['deadline_monotonic']:
                raise ValueError('Fit exceeded original deadline')
            scale = max(np.stack([np.diff(price[t - 60:t + 1]) for t in train]).std(), 1e-6)
            compare(scale, record['target_scale'])
            bank = banks[c['group']]
            window = bank[train[:, None] - np.arange(c['length'] - 1, -1, -1)]
            compare(window.mean((0, 1)), record['feature_mean'])
            compare(np.maximum(window.std((0, 1)), 1e-6), record['feature_sd'])
            count = (c['length'] * bank.shape[1] + 1) * 20 if c['family'] == 'ridge' else (3 if c['family'] == 'gru' else 4) * c['hidden'] * (bank.shape[1] + c['hidden'] + 2) + (c['hidden'] + 1) * 20
            if count != record['parameter_count']:
                raise ValueError('Model parameter count differs')
            with np.load(root / 'fits' / (name + '.npz'), allow_pickle=False) as z:
                origin = np.arange(cut, 703 if cut == 459 else 930)
                if not np.array_equal(z['origins'], origin) or not np.array_equal(z['training_origins'], train) or not np.array_equal(z['validation_origins'], validation):
                    raise ValueError('Saved prediction support differs')
                if z['predictions'].shape != (len(origin), 20) or z['training_predictions'].shape != (len(train), 20) or z['validation_predictions'].shape != (60, 20):
                    raise ValueError('Saved prediction dimensions differ')
                blocks.append(np.array(z['predictions']))
                train_target = price[train[:, None] + np.arange(1, 21)] - price[train, None]
                valid_target = price[validation[:, None] + np.arange(1, 21)] - price[validation, None]
                compare(np.mean(((z['training_predictions'] - train_target) / scale) ** 2), record['train_mse'])
                compare(np.mean(((z['validation_predictions'] - valid_target) / scale) ** 2), record['best_validation'])
                h = np.array([0, 4, 19])
                primary = float(np.mean(np.mean((z['validation_predictions'][:, h] - valid_target[:, h]) ** 2, axis=0) / np.mean(valid_target[:, h] ** 2, axis=0)))
                compare(primary, record['validation_primary'])
                v.append(primary)
            if c['family'] != 'ridge':
                best, selected, stale = record['initial_validation'], 0, 0
                if not 1 <= record['epochs'] <= 60 or len(record['trace']) != record['epochs']:
                    raise ValueError('Invalid epoch count')
                for i, row in enumerate(record['trace'], start=1):
                    if row[0] != i or not np.isfinite(row).all() or stale >= 12:
                        raise ValueError('Invalid training/checkpoint trajectory')
                    if row[3] < best - 1e-6:
                        best, selected, stale = row[3], i, 0
                    else:
                        stale += 1
                if record['epochs'] < 60 and stale != 12:
                    raise ValueError('Unexpected early termination')
                if selected != record['selected_epoch']:
                    raise ValueError('Selected epoch differs')
                compare(best, record['best_validation'])
            records.append({k: value for k, value in record.items() if k != 'trace'})
        predictions.append(np.concatenate(blocks)); validation_values.append(v)
    ids = np.arange(459, 930)
    predictions.extend([np.zeros((len(ids), 20)), reference[ids]])
    for control in ('rw', 'stat'):
        values = []
        for cut in (459, 703):
            t = np.arange(cut - 80, cut - 20)
            losses = []
            for h in (1, 5, 20):
                target = price[t + h] - price[t]
                prediction = np.zeros(len(t)) if control == 'rw' else reference[t, h - 1]
                losses.append(np.dot(prediction - target, prediction - target) / np.dot(target, target))
            values.append(float(np.mean(losses)))
        validation_values.append(values)
    predictions, validation_values = np.stack(predictions), np.array(validation_values)
    table, counts = risk_table(price, dates, ids, predictions)
    if counts != [224, 227]:
        raise ValueError('Year origin counts differ')
    results = read(directory / 'RESULTS.json')
    ids_config = [c['id'] for c in before['configs']] + ['rw', 'stat']
    if [r['id'] for r in results['rows']] != ids_config or len(results['rows']) != 98:
        raise ValueError('Published model set/order differs')
    for i, row in enumerate(results['rows']):
        compare(table[i], row['cells']); compare(table[i].mean(), row['score'])
        compare(table[i].mean(1), [row['annual']['2020'], row['annual']['2021']])
        compare(validation_values[i], row['prior_validation'])
    annual, scores = table.mean(2), table.mean((1, 2))
    best, yearly, prior = int(scores.argmin()), annual.argmin(0), validation_values.argmin(0)
    selected = results['selections']
    if selected['hindsight_static']['id'] != ids_config[best] or selected['hindsight_year_ids'] != [ids_config[x] for x in yearly] or selected['prior_validation_ids'] != [ids_config[x] for x in prior]:
        raise ValueError('Hindsight/prior selector identity differs')
    if selected['hindsight_cell_ids'] != [[ids_config[x] for x in row] for row in table.argmin(0)]:
        raise ValueError('Cell oracle differs')
    compare(scores[best], selected['hindsight_static']['score'])
    compare(np.mean([annual[yearly[y], y] for y in range(2)]), selected['hindsight_year_score'])
    compare(table.min(0).mean(), selected['hindsight_cell_score'])
    compare(np.mean([annual[prior[y], y] for y in range(2)]), selected['prior_validation_score'])
    compare(annual[yearly[0], 1], selected['transfer_2020_winner_to_2021']['score'])
    compare(annual[yearly[1], 0], selected['reverse_2021_winner_to_2020']['score'])
    pointwise = []
    for year in ('2020', '2021'):
        pos = [i for i, t in enumerate(ids) if dates[t][:4] == year and dates[t + 20][:4] == year]
        t = ids[pos]
        values = []
        for h in (1, 5, 20):
            target = price[t + h] - price[t]
            error = predictions[:, pos, h - 1] - target
            values.append(float(np.min(error ** 2, axis=0).sum() / np.dot(target, target)))
        pointwise.append(values)
    compare(pointwise, selected['pointwise_oracle_cells']); compare(np.mean(pointwise), selected['pointwise_oracle_score'])
    if results['unexposed_test'] or not results['diagnostic_only'] or results['novelty_established'] or selected['pointwise_oracle_is_deployable']:
        raise ValueError('Diagnostic scope overstated')
    if read(directory / 'TRAINING_SUMMARY.json')['records'] != records:
        raise ValueError('Published training summary differs')
    with (directory / 'ALL_CONFIGS.csv').open(encoding='utf-8-sig', newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != 98 or {r['id'] for r in csv_rows} != set(ids_config):
        raise ValueError('Published CSV count/scope differs')
    for row in csv_rows:
        compare(float(row['score']), scores[ids_config.index(row['id'])])
    if max(differences) > 1e-8:
        raise ValueError('Independent numeric replay differs')
    return dict(passed=True, formal_fits=192, trained_configs=96, controls=2, score_origins=sum(counts),
                numeric_comparisons=len(differences), max_absolute_difference=max(differences),
                selected_config=ids_config[best], selected_score=float(scores[best]), unexposed_test=False,
                scope='Independent approved-original label/scaler/score/selection replay, not model retraining, causal diagnosis or novelty certification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = replay(Path(__file__).resolve().parent)
    with Path(args.output).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))
