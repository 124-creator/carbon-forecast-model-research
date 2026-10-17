"""Independent Stage A verification. No trainer imports and no market fitting.

Approved CSV -> independent labels/support/scalers -> saved NPZ/JSON comparison.
This verifier does not certify novelty, unexposed testing, or third-party retraining.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import platform
from typing import Any

import numpy as np

APPROVED_BYTES = 29204
APPROVED_SHA256 = '1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070'
HORIZONS = (1, 5, 20)
YEARS = ('2020', '2021')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_approved(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    # Deliberately do not ask the stream for the next byte or read a later suffix.
    with path.open('rb', buffering=0) as stream:
        content = stream.read(APPROVED_BYTES)
    require(len(content) == APPROVED_BYTES, 'Approved byte count differs')
    digest = hashlib.sha256(content).hexdigest()
    require(digest == APPROVED_SHA256, 'Approved prefix SHA256 differs')
    reader = csv.reader(io.StringIO(content.decode('utf-8-sig')))
    header = next(reader)
    require(header == ['date', 'author_reported_price', 'source_excel_row', 'source_date_kind'], 'Approved header differs')
    rows = list(reader)
    require(len(rows) == 950 and all(len(row) == 4 for row in rows), 'Approved record shape differs')
    dates = np.asarray([row[0] for row in rows])
    price = np.asarray([float(row[1]) for row in rows], dtype=np.float64)
    require(dates[0] == '2018-01-03' and dates[-1] == '2021-12-31', 'Approved date limits differ')
    require(bool(np.all(dates[:-1] < dates[1:])), 'Original dates must be strictly increasing')
    require(bool(np.isfinite(price).all() and (price > 0).all()), 'Original prices invalid')
    return price, dates, {
        'bytes_read': len(content), 'records': len(rows), 'sha256': digest,
        'first_date': str(dates[0]), 'last_date': str(dates[-1]),
        'requested_next_byte': False, 'metadata_used_as_features': False,
    }


def chronological_support(dates: np.ndarray, cut: int) -> dict[str, np.ndarray]:
    require(cut in (459, 703), 'Unexpected annual cut')
    pool = np.asarray(list(range(max(60, cut - 500), cut - 20)), dtype=np.int64)
    training, validation = pool[:-80], pool[-60:]
    purge = pool[-80:-60]
    require(len(validation) == 60 and len(purge) == 20, 'Purge/validation count differs')
    require(int(training[-1]) + 20 < int(validation[0]), 'Training label overlaps validation origin')
    require(int(validation[-1]) + 20 < cut, 'Unmature validation label at fit time')
    expected_year = '2020' if cut == 459 else '2021'
    require(str(dates[cut])[:4] == expected_year and str(dates[cut - 1])[:4] != expected_year, 'Cut is not first observed point of the year')
    prediction = np.asarray(list(range(cut, 703 if cut == 459 else 930)), dtype=np.int64)
    return {'pool': pool, 'training': training, 'purge': purge, 'validation': validation, 'prediction': prediction}


def price_change_labels(price: np.ndarray, origins: np.ndarray) -> np.ndarray:
    require(bool(np.all(origins >= 0) and np.all(origins + 20 < len(price))), 'Label indices out of approved support')
    return np.asarray([[price[int(t) + h] - price[int(t)] for h in range(1, 21)] for t in origins], dtype=np.float64)


def target_scale_from_training(price: np.ndarray, training: np.ndarray) -> float:
    # 60 observed increments ending at each training origin; no validation support.
    windows = np.asarray([[price[int(t) - 59 + j] - price[int(t) - 60 + j] for j in range(60)] for t in training])
    return max(float(np.std(windows, ddof=0)), 1e-6)


def six_cell_scores(price: np.ndarray, dates: np.ndarray, origins: np.ndarray, delta: np.ndarray) -> dict[str, Any]:
    require(np.array_equal(origins, np.arange(459, 930)), 'Prediction origins must exactly cover 459..929')
    require(delta.shape == (471, 20) and bool(np.isfinite(delta).all()), 'Invalid saved forecast shape/value')
    cells, counts = [], []
    for year in YEARS:
        positions = [i for i, origin in enumerate(origins) if str(dates[origin])[:4] == year and str(dates[origin + 20])[:4] == year]
        counts.append(len(positions))
        for h in HORIZONS:
            actual = [float(price[int(origins[i]) + h] - price[int(origins[i])]) for i in positions]
            errors = [float(delta[i, h - 1]) - target for i, target in zip(positions, actual)]
            sse = math.fsum(error * error for error in errors)
            rw_sse = math.fsum(target * target for target in actual)
            require(rw_sse > 0 and math.isfinite(rw_sse), 'Nonpositive/nonfinite RW risk')
            cells.append({'year': year, 'horizon': h, 'origins': len(positions), 'mse': sse / len(positions), 'rw_mse': rw_sse / len(positions), 'ratio': sse / rw_sse})
    require(counts == [224, 227], 'Main complete-H20 year counts differ')
    return {'cells': cells, 'score': math.fsum(c['ratio'] for c in cells) / 6, 'main_origins': sum(counts), 'year_counts': dict(zip(YEARS, counts))}


def support_receipt(price: np.ndarray, dates: np.ndarray) -> dict[str, Any]:
    out = {}
    for cut in (459, 703):
        s = chronological_support(dates, cut)
        out[str(cut)] = {
            'train_count': len(s['training']), 'train_first_origin': int(s['training'][0]),
            'train_last_origin': int(s['training'][-1]), 'train_latest_label': int(s['training'][-1] + 20),
            'purged_origins': len(s['purge']), 'validation_count': len(s['validation']),
            'validation_first_origin': int(s['validation'][0]), 'validation_last_origin': int(s['validation'][-1]),
            'validation_latest_label': int(s['validation'][-1] + 20),
            'first_prediction_origin': int(s['prediction'][0]), 'last_prediction_origin': int(s['prediction'][-1]),
            'target_scale': target_scale_from_training(price, s['training']),
        }
    return out


def independent_reference(price: np.ndarray) -> np.ndarray:
    """Scalar candidate-by-candidate implementation of the documented reference.

    No import of fair_core; each state at t uses only price[0:t+1], and a
    candidate choice at t uses prediction errors strictly before t.
    """
    candidates = [(0.0, 1.0)] + [(r, q) for r in (0.0, .25, .5, .75, .9, .97) for q in (.02, .1, .25, .5, .8)]
    states = np.zeros((len(candidates), len(price)))
    errors = np.zeros_like(states)
    variances = np.ones_like(states)
    for k, (rho, permanent) in enumerate(candidates):
        state, covariance = 0.0, (1.0 - permanent) / (1.0 - rho * rho)
        for t in range(1, len(price)):
            error = float(price[t] - price[t - 1]) - (rho - 1.0) * state
            innovation = (rho - 1.0) ** 2 * covariance + 1.0
            cross = rho * (rho - 1.0) * covariance + 1.0 - permanent
            state = rho * state + cross * error / innovation
            covariance = max(rho * rho * covariance + 1.0 - permanent - cross * cross / innovation, 0.0)
            states[k, t], errors[k, t], variances[k, t] = state, error, innovation
    prediction = np.zeros((len(price), 20))
    selected = 0
    for t in range(len(price)):
        if t >= 60 and (t - 60) % 20 == 0:
            start = max(1, t - 240)
            criteria = []
            for k in range(len(candidates)):
                scale = max(float(np.mean(errors[k, start:t] ** 2 / variances[k, start:t])), 1e-12)
                criteria.append(float(np.log(variances[k, start:t]).sum()) + (t - start) * math.log(scale))
            selected = int(np.argmin(criteria))
        rho = candidates[selected][0]
        for h in range(1, 21):
            prediction[t, h - 1] = (rho ** h - 1.0) * states[selected, t]
    return prediction


def input_scalers(price: np.ndarray, training: np.ndarray, linear: bool) -> tuple[float, float]:
    if linear:
        windows = np.asarray([[price[int(t) - 19 + j] for j in range(20)] for t in training])
    else:
        windows = np.asarray([[price[int(t) - 19 + j] - price[int(t) - 20 + j] for j in range(20)] for t in training])
    return float(windows.mean()), max(float(windows.std()), 1e-6)


def numpy_forward(name: str, state: dict[str, np.ndarray], price: np.ndarray,
                  origins: np.ndarray, mean: float, sd: float, scale: float,
                  reference: np.ndarray) -> np.ndarray:
    """Inference from saved arrays using independent NumPy equations, no training."""
    linear = name in ('NLinear', 'DLinear')
    if linear:
        raw = np.asarray([[price[int(t) - 19 + j] for j in range(20)] for t in origins])
        x = (raw - mean) / sd
        if name == 'NLinear':
            last = x[:, -1:]
            level = (x - last) @ state['Linear.weight'].T + state['Linear.bias'] + last
        else:
            extended = np.pad(x, ((0, 0), (12, 12)), mode='edge')
            trend = np.asarray([extended[:, j:j + 25].mean(axis=1) for j in range(20)]).T
            seasonal = x - trend
            level = (seasonal @ state['Linear_Seasonal.weight'].T + state['Linear_Seasonal.bias']
                     + trend @ state['Linear_Trend.weight'].T + state['Linear_Trend.bias'])
        return level * sd + mean - price[origins, None]
    raw = np.asarray([[price[int(t) - 19 + j] - price[int(t) - 20 + j] for j in range(20)] for t in origins])
    x = (raw - mean) / sd
    hidden = np.zeros((len(origins), 16))
    cell = np.zeros_like(hidden)
    weight_ih, weight_hh = state['encoder.weight_ih_l0'], state['encoder.weight_hh_l0']
    bias_ih, bias_hh = state['encoder.bias_ih_l0'], state['encoder.bias_hh_l0']
    sigmoid = lambda v: 1.0 / (1.0 + np.exp(-v))
    for j in range(20):
        incoming = x[:, j:j + 1] @ weight_ih.T + bias_ih
        recurrent = hidden @ weight_hh.T + bias_hh
        if name == 'c003_GRU':
            ir, iz, inn = np.split(incoming, 3, axis=1)
            hr, hz, hn = np.split(recurrent, 3, axis=1)
            reset, update = sigmoid(ir + hr), sigmoid(iz + hz)
            proposed = np.tanh(inn + reset * hn)
            hidden = (1.0 - update) * proposed + update * hidden
        else:
            incoming_gate, forget_gate, proposed, output_gate = np.split(incoming + recurrent, 4, axis=1)
            cell = sigmoid(forget_gate) * cell + sigmoid(incoming_gate) * np.tanh(proposed)
            hidden = sigmoid(output_gate) * np.tanh(cell)
    correction = np.column_stack([hidden, np.ones(len(origins))]) @ state['readout']
    base = reference[origins] if name == 'c003_GRU' else 0.0
    return base + scale * correction


def verify(stage: Path) -> dict[str, Any]:
    workspace = stage.parents[1]
    read_json = lambda path: json.loads(path.read_text(encoding='utf-8'))
    pre = read_json(stage / 'STAGE_A_PRE_RUN.json')
    done = read_json(stage / 'STAGE_A_RUN_COMPLETE.json')
    aggregate = read_json(stage / 'STAGE_A_RESULTS.json')
    require(not (stage / 'STAGE_A_FAILED.json').exists(), 'Failed/incomplete run cannot pass')
    models = ('NLinear', 'DLinear', 'c003_GRU', 'c025_LSTM')
    seeds = (17, 29, 43)
    tasks = [{'model': name, 'seed': seed, 'cut': cut} for seed in seeds for name in models for cut in (459, 703)]
    stems = ['{model}_seed{seed}_cut{cut}'.format(**task) for task in tasks]
    require(pre['tasks'] == tasks and pre['formal_expected'] == 24 and done['formal_fits'] == 24, 'Frozen task set differs')
    require(done['completed_cases'] == stems and done['completed_within_deadline'], 'Incomplete or unordered formal tasks')
    require(pre['cuts'] == [459, 703] and pre['seeds'] == list(seeds) and pre['models'] == list(models), 'Frozen identities differ')
    require(pre['workers'] == 1 and pre['math_threads'] == 1 and pre['dtype'] == 'float64' and pre['device'] == 'cpu', 'Unexpected execution resources')
    require(pre['budget_seconds'] == 600 and pre['deadline_monotonic'] - pre['started_monotonic'] == 600, 'Deadline changed')
    require(done['elapsed_seconds'] < 600, 'Run exceeded fixed deadline')
    for relative, expected_sha in pre['source_sha256'].items():
        require(hashlib.sha256((workspace / relative).read_bytes()).hexdigest() == expected_sha, 'Frozen source changed: ' + relative)
    source_manifest = read_json(workspace / 'provenance/SOURCE_READ_MANIFEST.json')
    for record in source_manifest:
        if record['runtime_path'] in pre['source_sha256']:
            require(pre['source_sha256'][record['runtime_path']] == record['sha256'], 'Approved historical infrastructure modified: ' + record['runtime_path'])
    identity = read_json(stage / 'audit/AUTHOR_CORE_IDENTITY.json')
    require(identity['author_core_identity_passed'], 'Author-core independent fetch identity not verified')
    for row in identity['files']:
        require(hashlib.sha256((stage / 'vendor/ltsf_linear' / row['file']).read_bytes()).hexdigest() == row['local_sha256'], 'Author core changed after independent retrieval')

    numeric_count, max_difference, max_label = 0, 0.0, ''
    def compare(observed: Any, expected: Any, label: str, atol: float = 1e-8) -> None:
        nonlocal numeric_count, max_difference, max_label
        a, b = np.asarray(observed, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        require(a.shape == b.shape, 'Shape differs at ' + label)
        require(bool(np.isfinite(a).all() and np.isfinite(b).all()), 'Nonfinite values at ' + label)
        count = a.size
        difference = float(np.max(np.abs(a - b))) if count else 0.0
        numeric_count += count
        if difference > max_difference:
            max_difference, max_label = difference, label
        require(difference <= atol, f'Numeric mismatch at {label}: {difference} > {atol}')

    price, dates, data_receipt = read_approved(workspace / 'data/parsed/figshare_28682348/guangzhou_reported.csv')
    require(pre['data']['prefix_sha256'] == APPROVED_SHA256 and pre['data']['records'] == 950, 'Frozen data identity differs')
    reference = independent_reference(price)
    local = stage / 'local_only/stage_a'
    with np.load(local / 'INPUTS.npz', allow_pickle=False) as saved:
        require(np.array_equal(saved['dates'], dates), 'Saved original dates differ')
        compare(saved['price'], price, 'original numeric input', 0.0)
        compare(saved['reference'], reference, 'independent causal statistical reference')
    require({p.stem for p in (local / 'fits').glob('*.npz')} == set(stems), 'Unexpected/missing prediction NPZ')
    require({p.stem for p in (stage / 'stage_a_receipts').glob('*.json')} == set(stems), 'Unexpected/missing fit receipt')
    attempts = [json.loads(row) for row in (stage / 'STAGE_A_ATTEMPTS.jsonl').read_text().splitlines()]
    require(len(attempts) == 24, 'Unexpected market attempts or retries')
    fit_audit, predictions, validations, summaries = [], {}, {}, []
    npz_hashes = {}
    parameter_counts = {'NLinear': 420, 'DLinear': 840, 'c003_GRU': 1252, 'c025_LSTM': 1556}
    for ordinal, (task, stem) in enumerate(zip(tasks, stems), start=1):
        name, seed, cut = task['model'], task['seed'], task['cut']
        record = read_json(stage / 'stage_a_receipts' / (stem + '.json'))
        require(record['task'] == task and record['seed'] == seed and record['end'] == cut and record['ordinal'] == ordinal, 'Fit identity differs: ' + stem)
        require(attempts[ordinal - 1]['task'] == task and attempts[ordinal - 1]['ordinal'] == ordinal, 'Attempt identity differs: ' + stem)
        require(pre['started_monotonic'] <= attempts[ordinal - 1]['monotonic'] <= record['completed_monotonic'] < pre['deadline_monotonic'], 'Fit deadline/order violation: ' + stem)
        support = chronological_support(dates, cut)
        train, val, pred_ids = support['training'], support['validation'], support['prediction']
        require(record['train_n'] == len(train) and record['validation_n'] == len(val), 'Training/validation count differs: ' + stem)
        expected_fields = {'pool_start': int(train[0]), 'train_max_origin': int(train[-1]), 'train_max_target': int(train[-1] + 20), 'validation_first_origin': int(val[0]), 'validation_max_target': int(val[-1] + 20), 'first_origin': int(pred_ids[0]), 'last_origin': int(pred_ids[-1]), 'boundary_origin': cut - 1}
        for key, value in expected_fields.items():
            require(record[key] == value, key + ' differs: ' + stem)
        linear = name in ('NLinear', 'DLinear')
        mean, sd = input_scalers(price, train, linear)
        scale = target_scale_from_training(price, train)
        if linear:
            compare([record['input_global_mean'], record['input_global_sd'], record['fair_support_increment_scale']], [mean, sd, scale], stem + ' train-only linear normalization')
            require(record['author_commit'] == '0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6', 'Author commit differs')
            require(record['config'] == {'seq_len': 20, 'pred_len': 20, 'enc_in': 1, 'individual': False}, 'Linear core settings changed')
        else:
            compare(record['feature_mean'], [mean], stem + ' train-only increment mean')
            compare(record['feature_sd'], [sd], stem + ' train-only increment sd')
            compare(record['target_scale'], scale, stem + ' target scale')
            require(record['config'] == pre['configs'][name], 'Historical strong-control config changed')

        path = local / 'fits' / (stem + '.npz')
        npz_hashes[stem] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as saved:
            for field, ids in [('origins', pred_ids), ('training_origins', train), ('validation_origins', val)]:
                require(np.array_equal(saved[field], ids), 'Saved support differs: ' + stem + '/' + field)
            state = {key[len('state_'):]: np.array(saved[key]) for key in saved.files if key.startswith('state_')}
            require(all(value.dtype == np.float64 and np.isfinite(value).all() for value in state.values()), 'State dtype/value differs')
            require(sum(value.size for value in state.values()) == parameter_counts[name] == record['parameter_count'], 'Parameter count differs')
            state_hash = hashlib.sha256(b''.join(key.encode() + value.tobytes() for key, value in sorted(state.items()))).hexdigest()
            require(state_hash == record['state_sha256'], 'Saved state hash differs')
            for field, ids in [('predictions', pred_ids), ('training_predictions', train), ('validation_predictions', val), ('boundary_prediction', np.array([cut - 1]))]:
                calculated = numpy_forward(name, state, price, ids, mean, sd, scale, reference)
                compare(saved[field], calculated, stem + ' independent NumPy inference ' + field)
            require(hashlib.sha256(saved['boundary_prediction'].tobytes()).hexdigest() == record['boundary_forecast_sha256'], 'Boundary prediction hash differs')
            require(record['frozen_fit_suffix_prediction_difference'] == 0.0, 'Original frozen boundary check failed')
            # Independently rerun frozen inference with all later observations replaced.
            altered = price.copy()
            altered[cut:] = 100000.0 + 7.0 * np.arange(len(price) - cut)
            boundary = numpy_forward(name, state, altered, np.array([cut - 1]), mean, sd, scale, reference)
            compare(saved['boundary_prediction'], boundary, stem + ' independent future-suffix frozen inference')
            train_target, val_target = price_change_labels(price, train), price_change_labels(price, val)
            divisor = sd if linear else scale
            compare(float(np.mean(((saved['training_predictions'] - train_target) / divisor) ** 2)), record['train_mse'], stem + ' training loss')
            compare(float(np.mean(((saved['validation_predictions'] - val_target) / divisor) ** 2)), record['best_validation'], stem + ' selected validation loss')
            h = [0, 4, 19]
            validation_primary = float(np.mean(np.mean((saved['validation_predictions'][:, h] - val_target[:, h]) ** 2, axis=0) / np.mean(val_target[:, h] ** 2, axis=0)))
            compare(validation_primary, record['validation_primary'], stem + ' validation primary score')
            predictions[(name, seed, cut)] = np.array(saved['predictions'])
            validations[(name, seed, cut)] = validation_primary
        best, selected, stale = record['initial_validation'], 0, 0
        require(1 <= record['epochs'] <= 60 and len(record['trace']) == record['epochs'], 'Epoch count differs')
        for epoch, row in enumerate(record['trace'], start=1):
            require(row[0] == epoch and np.isfinite(row).all() and stale < 12, 'Trace/early-stop sequence differs')
            value = row[2] if linear else row[3]
            if value < best - 1e-6:
                best, selected, stale = value, epoch, 0
            else:
                stale += 1
        require(record['epochs'] == 60 or stale == 12, 'Unexpected early termination')
        require(record['selected_epoch'] == selected, 'Best checkpoint choice differs')
        compare(best, record['best_validation'], stem + ' trace-selected loss')
        summaries.append({key: value for key, value in record.items() if key != 'trace'})
        fit_audit.append({'case': stem, 'train_n': len(train), 'validation_n': len(val), 'epochs': record['epochs'], 'selected_epoch': selected, 'saved_state_matches_predictions': True, 'state_sha256': state_hash})

    require(read_json(stage / 'STAGE_A_TRAINING_SUMMARY.json')['records'] == summaries, 'Public training summary differs from complete receipts')
    controls = ('RW', 'stat')
    independent_scores, per_cell = [], []
    for seed in seeds:
        for name in models + controls:
            delta = (np.vstack([predictions[(name, seed, cut)] for cut in (459, 703)]) if name in models
                     else np.zeros((471, 20)) if name == 'RW' else reference[459:930])
            scored = six_cell_scores(price, dates, np.arange(459, 930), delta)
            prior = [validations[(name, seed, cut)] for cut in (459, 703)] if name in models else []
            independent_scores.append({'model': name, 'seed': seed, 'score': scored['score'], 'cells': [cell['ratio'] for cell in scored['cells']], 'prior_validation': prior})
            for cell in scored['cells']:
                per_cell.append({'model': name, 'seed': seed, **cell})
    require(len(aggregate['per_seed']) == 18 and len(aggregate['per_cell']) == 108 and len(aggregate['rows']) == 6, 'Published row counts differ')
    for observed, expected in zip(aggregate['per_seed'], independent_scores):
        require((observed['model'], observed['seed']) == (expected['model'], expected['seed']), 'Per-seed identity/order differs')
        for key in ['score', 'cells', 'prior_validation']:
            compare(observed[key], expected[key], f"per-seed {expected['model']}/{expected['seed']} {key}")
        require(observed['diagnostic_only'] and not observed['unexposed_test'], 'Per-seed scope overstated')
    for observed, expected in zip(aggregate['per_cell'], per_cell):
        require((observed['model'], observed['seed'], observed['year'], observed['horizon'], observed['origins']) == (expected['model'], expected['seed'], expected['year'], expected['horizon'], expected['origins']), 'Per-cell identity differs')
        compare([observed['mse'], observed['rw_mse'], observed['relative_mse']], [expected['mse'], expected['rw_mse'], expected['ratio']], 'per-cell MSE/RW score')
    independent_models = []
    for name in models + controls:
        rows = [row for row in independent_scores if row['model'] == name]
        scores = np.asarray([row['score'] for row in rows])
        cell_array = np.asarray([row['cells'] for row in rows])
        independent_models.append({'model': name, 'seeds': list(seeds), 'mean_score': float(scores.mean()), 'sd_score': float(scores.std(ddof=1)), 'min_score': float(scores.min()), 'max_score': float(scores.max()), 'cells_mean': cell_array.mean(axis=0).tolist(), 'cells_sd': cell_array.std(axis=0, ddof=1).tolist()})
    for observed, expected in zip(aggregate['rows'], independent_models):
        require(observed['model'] == expected['model'] and observed['seeds'] == expected['seeds'], 'Per-model identity differs')
        for key in ['mean_score', 'sd_score', 'min_score', 'max_score', 'cells_mean', 'cells_sd']:
            compare(observed[key], expected[key], 'per-model ' + expected['model'] + ' ' + key)
    with (stage / 'STAGE_A_PER_SEED.csv').open(newline='', encoding='utf-8') as stream:
        csv_seed = list(csv.DictReader(stream))
    require(len(csv_seed) == 18, 'Per-seed CSV row count differs')
    for observed, expected in zip(csv_seed, independent_scores):
        require(observed['model'] == expected['model'] and int(observed['seed']) == expected['seed'], 'Per-seed CSV identity differs')
        compare([float(observed['score'])] + [float(observed[key]) for key in aggregate['cell_order']], [expected['score']] + expected['cells'], 'per-seed CSV values')
    with (stage / 'STAGE_A_PER_CELL.csv').open(newline='', encoding='utf-8') as stream:
        csv_cells = list(csv.DictReader(stream))
    require(len(csv_cells) == 108, 'Per-cell CSV row count differs')
    for observed, expected in zip(csv_cells, aggregate['per_cell']):
        require(observed['model'] == expected['model'] and int(observed['seed']) == expected['seed'] and observed['year'] == expected['year'] and int(observed['horizon']) == expected['horizon'], 'Per-cell CSV identity differs')
        for key in ['mse', 'rw_mse', 'relative_mse', 'mae', 'rmse']:
            compare(float(observed[key]), expected[key], 'per-cell CSV ' + key)
    require(aggregate['formal_fits'] == 24 and aggregate['score_origins'] == 451 and aggregate['annual_origin_counts'] == [224, 227] and aggregate['horizons'] == [1, 5, 20], 'Aggregate protocol metadata differs')
    require(aggregate['diagnostic_only'] and not aggregate['unexposed_test'] and not aggregate['novelty_established'], 'Aggregate scope overstated')
    timing = read_json(stage / 'STAGE_A_TIMING.json')
    require(timing['timing_fits'] == 3 and not timing['score_2020_2021_computed'] and timing['source_sha256'] == pre['source_sha256'], 'Timing/freeze evidence differs')
    require(len(list((stage / 'timing_receipts').glob('*.json'))) == 3, 'Unexpected timing fit count')
    return {
        'passed': True, 'scope': 'Independent approved-CSV label/support/normalization/reference reconstruction, NumPy saved-state inference, checkpoint/score/CSV replay; zero market fits',
        'third_party_training_reproduction': False, 'unexposed_test': False, 'novelty_certified': False,
        'data': data_receipt, 'support': support_receipt(price, dates),
        'formal_fits_checked': 24, 'timing_fits_checked_from_receipts': 3,
        'market_fits_performed_by_auditor': 0, 'main_score_origins': 451, 'annual_counts': [224, 227],
        'per_seed_rows_checked': 18, 'six_cell_rows_checked': 108,
        'independent_numeric_comparisons': int(numeric_count), 'maximum_absolute_difference': max_difference,
        'maximum_difference_location': max_label, 'tolerance': 1e-8,
        'numpy_only_inference_models': list(models), 'fit_checks': fit_audit,
        'independent_per_model': independent_models, 'independent_per_seed': independent_scores,
        'source_sha256': pre['source_sha256'], 'prediction_npz_sha256': npz_hashes,
        'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__},
        'limitations': ['Fresh-fit suffix perturbation was not rerun by the auditor; inspected prefix slicing and independently checked frozen inference.', 'Saved state and trace consistency verify submitted artifacts, not the full optimizer trajectory by retraining.', 'Protocol adaptation preserves author model core but does not reproduce the original paper benchmark or original full training recipe.', 'RW/stat repeated across seed rows are deterministic controls, not six additional independently trained models.'],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--support-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    stage = Path(__file__).resolve().parents[1]
    workspace = stage.parents[1]
    price, dates, source = read_approved(workspace / 'data/parsed/figshare_28682348/guangzhou_reported.csv')
    if args.support_only:
        rw = six_cell_scores(price, dates, np.arange(459, 930), np.zeros((471, 20)))
        print(json.dumps({'source': source, 'support': support_receipt(price, dates), 'rw': rw}, indent=2))
        return
    result = verify(stage)
    output = args.output or stage / 'audit/AUDIT_STAGE_A.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ['passed', 'formal_fits_checked', 'market_fits_performed_by_auditor', 'independent_numeric_comparisons', 'maximum_absolute_difference', 'maximum_difference_location', 'independent_per_model']}, indent=2))


if __name__ == '__main__':
    main()
