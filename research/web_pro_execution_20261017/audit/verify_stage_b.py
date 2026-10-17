"""Independent Stage B artifact audit: NumPy inference, no trainer/torch imports.

Reads only the approved 950-row prefix. Does not fit, take gradients, update a
market model, or reproduce a paper benchmark. Shared data/support/score helpers
come from this auditor's separate Stage A verifier, not either trainer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any

import numpy as np

from verify_stage_a import (
    APPROVED_SHA256, chronological_support, independent_reference, input_scalers,
    price_change_labels, read_approved, require, six_cell_scores, support_receipt,
    target_scale_from_training,
)

ARMS = ('NLinear_OGD', 'PROCEED', 'PROCEED_no_drift')
SEEDS = (17, 29, 43)
CUTS = (459, 703)
AUTHOR_COMMIT = '0a8646199f6dcbf5346f933508b33c5e35b5708b'
G = 'generator.bottlenecks.Linear_60.'
CONFIG = {'seq_len': 20, 'pred_len': 20, 'enc_in': 1, 'individual': False,
          'freeze': False, 'concept_dim': 16, 'bottleneck_dim': 4, 'act': 'sigmoid',
          'individual_generator': False, 'ema': 0.0, 'tune_mode': 'down_up',
          'merge_weights': True, 'do_predict': True, 'wo_clip': False}
ONLINE_COLUMNS = ['issue_origin', 'mature_origin', 'mature_input_min', 'mature_input_max',
                  'mature_target_min', 'mature_target_max', 'forecast_input_min',
                  'forecast_input_max', 'previous_recent_label_max', 'new_recent_label_max',
                  'normalized_mature_MSE', 'unclipped_gradient_norm']


def parameter_shapes(arm: str) -> dict[str, tuple[int, ...]]:
    if arm == 'NLinear_OGD':
        return {'Linear.weight': (20, 20), 'Linear.bias': (20,)}
    return {'backbone.Linear.weight': (20, 20), 'backbone.Linear.bias': (20,),
            G + 'biases.0': (1, 1, 4), G + 'biases.1': (1, 1, 60),
            G + 'weights.1': (1, 1, 60, 4), G + 'weights.0.weight': (4, 16),
            'mlp1.1.weight': (16, 20), 'mlp1.1.bias': (16,),
            'mlp1.3.weight': (16, 16), 'mlp1.3.bias': (16,),
            'mlp2.1.weight': (16, 40), 'mlp2.1.bias': (16,),
            'mlp2.3.weight': (16, 16), 'mlp2.3.bias': (16,)}


def state_hash(state: dict[str, np.ndarray], names: list[str] | None = None) -> str:
    value = hashlib.sha256()
    for name in sorted(state if names is None else names):
        array = state[name]
        value.update(name.encode())
        value.update(str(tuple(array.shape)).encode())
        value.update(array.tobytes())
    return value.hexdigest()


def groups(arm: str) -> dict[str, list[str]]:
    prefix = '' if arm == 'NLinear_OGD' else 'backbone.'
    fixed = [] if arm == 'NLinear_OGD' else [G + 'biases.1']
    back_weight, back_bias = [prefix + 'Linear.weight'], [prefix + 'Linear.bias']
    dynamic = [name for name in parameter_shapes(arm)
               if name not in back_weight + back_bias + fixed]
    return {'backbone_weights': back_weight, 'backbone_bias': back_bias,
            'dynamic_adapter': dynamic, 'fixed_adapter_bias': fixed}


def normalized_window(observed: np.ndarray, origin: int, mean: float, sd: float) -> np.ndarray:
    require(origin >= 19 and origin < len(observed), 'Input exceeds available observations')
    return ((observed[origin - 19:origin + 1] - mean) / sd)[None, :]


def mature_context(observed: np.ndarray, origin: int, mean: float, sd: float) -> np.ndarray:
    require(origin >= 19 and origin + 20 < len(observed), 'Context label has not matured')
    return ((observed[origin - 19:origin + 21] - mean) / sd)[None, :, None]


def gelu(x: np.ndarray) -> np.ndarray:
    # torch.nn.GELU defaults to the exact Gaussian CDF, not its tanh approximation.
    values = np.fromiter((math.erf(float(v) / math.sqrt(2.0)) for v in x.flat),
                         dtype=np.float64, count=x.size).reshape(x.shape)
    return .5 * x * (1.0 + values)


def mlp(x: np.ndarray, state: dict[str, np.ndarray], name: str) -> np.ndarray:
    hidden = gelu(x @ state[name + '.1.weight'].T + state[name + '.1.bias'])
    return hidden @ state[name + '.3.weight'].T + state[name + '.3.bias']


def normalized_forward(arm: str, state: dict[str, np.ndarray], x: np.ndarray,
                       recent: np.ndarray) -> np.ndarray:
    """Author's one-channel NLinear/down-up/generator expressed independently.

    In particular, the original Linear.bias is NOT multiplied by output scale.
    The input to the concept encoder is the globally standardized price window;
    centering at the last value happens inside the NLinear backbone afterwards.
    """
    require(x.ndim == 2 and x.shape[1] == 20, 'Wrong input shape')
    last = x[:, -1:]
    if arm == 'NLinear_OGD':
        return (x - last) @ state['Linear.weight'].T + state['Linear.bias'] + last
    require(recent.ndim == 3 and recent.shape[1:] == (40, 1), 'Wrong recent context shape')
    current = mlp(x, state, 'mlp1')
    previous = mlp(recent[:, :, 0], state, 'mlp2').mean(axis=0)
    drift = current - previous
    if arm == 'PROCEED_no_drift':
        drift = np.zeros_like(drift)
    # Equivalent to author clip at max_norm=1, including its forward at zero.
    drift = drift / np.maximum(np.linalg.norm(drift, axis=-1, keepdims=True), 1.0)
    hidden = drift @ state[G + 'weights.0.weight'].T + state[G + 'biases.0'][0, 0]
    hidden = 1.0 / (1.0 + np.exp(-hidden))
    coefficients = hidden @ state[G + 'weights.1'][0, 0].T + state[G + 'biases.1'][0, 0]
    out_scale = coefficients[:, :20] + 1.0
    in_scale = coefficients[:, 20:40] + 1.0
    shift = coefficients[:, 40:60]
    value = ((x - last) * in_scale) @ state['backbone.Linear.weight'].T
    return value * out_scale + state['backbone.Linear.bias'] + shift + last


def forecast(arm: str, state: dict[str, np.ndarray], observed: np.ndarray,
             origin: int, mean: float, sd: float, recent: np.ndarray) -> np.ndarray:
    require(len(observed) == origin + 1, 'Inference requires exact observed prefix')
    output = normalized_forward(arm, state, normalized_window(observed, origin, mean, sd), recent)
    return output[0] * sd + mean - observed[origin]


def validation_forecasts(arm: str, state: dict[str, np.ndarray], price: np.ndarray,
                         origins: np.ndarray, mean: float, sd: float) -> np.ndarray:
    rows = []
    for t in origins:
        observed = price[:int(t) + 1]
        context = mature_context(observed, int(t) - 20, mean, sd)
        rows.append(forecast(arm, state, observed, int(t), mean, sd, context))
    return np.asarray(rows)


def check_epoch_selection(record: dict[str, Any], compare: Any, stem: str) -> int:
    best, selected, stale = record['initial_validation'], 0, 0
    require(1 <= record['epochs'] <= 60 and len(record['trace']) == record['epochs'],
            'Epoch count differs: ' + stem)
    for epoch, row in enumerate(record['trace'], start=1):
        require(len(row) == 4 and row[0] == epoch and np.isfinite(row).all()
                and min(row[1:]) >= 0 and stale < 12, 'Invalid offline trace: ' + stem)
        if row[2] < best - 1e-6:
            best, selected, stale = row[2], epoch, 0
        else:
            stale += 1
    require(record['epochs'] == 60 or stale == 12, 'Unexplained termination: ' + stem)
    require(record['selected_epoch'] == selected, 'Selected epoch differs: ' + stem)
    compare(record['best_validation'], best, stem + ' trace-selected best validation')
    return selected


def verify(stage: Path) -> dict[str, Any]:
    workspace = stage.parents[1]
    read_json = lambda path: json.loads(path.read_text(encoding='utf-8'))
    pre = read_json(stage / 'STAGE_B_PRE_RUN.json')
    complete = read_json(stage / 'STAGE_B_RUN_COMPLETE.json')
    aggregate = read_json(stage / 'STAGE_B_RESULTS.json')
    require(not (stage / 'STAGE_B_FAILED.json').exists(), 'Failed/incomplete run cannot pass')
    tasks = [{'arm': arm, 'seed': seed, 'cut': cut} for seed in SEEDS for arm in ARMS for cut in CUTS]
    stems = ['{arm}_seed{seed}_cut{cut}'.format(**row) for row in tasks]
    require(pre['tasks'] == tasks and pre['formal_expected'] == complete['formal_fits'] == 18,
            'Frozen formal task count/identities differ')
    require(complete['completed_cases'] == stems and complete['completed_within_deadline'],
            'Formal runs incomplete or reordered')
    require(pre['online_expected'] == complete['online_updates'] == 4239, 'Online count differs')
    require(pre['cuts'] == list(CUTS) and pre['seeds'] == list(SEEDS) and pre['arms'] == list(ARMS),
            'Frozen arms/cuts/seeds differ')
    require(pre['stream_ranges_inclusive'] == [[459, 702], [703, 929]]
            and pre['stream_counts_per_cut'] == [244, 227], 'Online stream support differs')
    require(pre['workers'] == pre['math_threads'] == 1 and pre['device'] == 'cpu'
            and pre['environment']['dtype'] == 'float64', 'Execution resources differ')
    require(pre['budget_seconds'] == 900 and pre['deadline_monotonic'] - pre['started_monotonic'] == 900
            and complete['elapsed_seconds'] < 900, 'Fixed absolute budget exceeded/changed')
    require(pre['config'] == CONFIG and pre['pretraining_learning_rate'] == .003
            and pre['online_learning_rate'] == .0003 and pre['online_minibatch'] == pre['steps_per_origin'] == 1,
            'Frozen model/learning configuration differs')
    require(pre['no_test_per_epoch'] and pre['protocol_adaptation']
            and not any(pre[key] for key in ('historical_predictions_or_weights_used',
                       'original_benchmark_reproduction', 'unexposed_test', 'novelty_established')),
            'Frozen scientific scope overstated')
    for relative, expected_hash in pre['source_sha256'].items():
        require(hashlib.sha256((workspace / relative).read_bytes()).hexdigest() == expected_hash,
                'Frozen source changed: ' + relative)
    for row in read_json(workspace / 'provenance/SOURCE_READ_MANIFEST.json'):
        if row['runtime_path'] in pre['source_sha256']:
            require(pre['source_sha256'][row['runtime_path']] == row['sha256'],
                    'Historical infrastructure changed: ' + row['runtime_path'])
    identity = read_json(stage / 'audit/PROCEED_AUTHOR_CORE_IDENTITY.json')
    require(identity['independent_core_identity_passed'] and identity['commit'] == AUTHOR_COMMIT,
            'Independent author identity unavailable')
    for row in identity['files']:
        raw = (stage / 'vendor/proceed' / row['path']).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == row['local_sha256'], 'Core changed after independent fetch')
        require(hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
                == row['github_blob_sha'], 'Core Git blob identity differs')
    preflight_path = stage / 'audit/AUDIT_STAGE_B_PREFLIGHT.json'
    preflight = read_json(preflight_path) if preflight_path.exists() else None
    if preflight is not None:
        require(preflight['status'] == 'PASS_STATIC_PREFLIGHT', 'Static preflight did not pass')
        for relative, value in preflight['source_sha256'].items():
            require(hashlib.sha256((stage / relative).read_bytes()).hexdigest() == value,
                    'Implementation differs from static preflight: ' + relative)
    # A new isolated rerun can retain the immutable author identity without
    # presenting the original run's historical preflight timestamps as its own.

    count, max_difference, max_label = 0, 0.0, ''
    def compare(observed: Any, expected: Any, label: str, atol: float = 1e-8) -> None:
        nonlocal count, max_difference, max_label
        a, b = np.asarray(observed, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        require(a.shape == b.shape, 'Shape differs: ' + label)
        require(bool(np.isfinite(a).all() and np.isfinite(b).all()), 'Nonfinite numeric values: ' + label)
        error = float(np.max(np.abs(a - b))) if a.size else 0.0
        count += a.size
        if error > max_difference:
            max_difference, max_label = error, label
        require(error <= atol, f'Numeric mismatch {label}: {error} > {atol}')

    price, dates, data_receipt = read_approved(workspace / 'data/parsed/figshare_28682348/guangzhou_reported.csv')
    require(pre['data']['prefix_sha256'] == APPROVED_SHA256 and pre['data']['records'] == 950
            and pre['data']['bytes'] == 29204 and not pre['data']['next_byte_requested'], 'Data freeze differs')
    reference = independent_reference(price)
    local = stage / 'local_only/stage_b'
    with np.load(local / 'INPUTS.npz', allow_pickle=False) as saved:
        require(np.array_equal(saved['dates'], dates), 'Saved original dates differ')
        compare(saved['price'], price, 'Approved original numeric inputs', 0.0)
        compare(saved['reference'], reference, 'Independent causal statistical reference')
    require({p.stem for p in (local / 'fits').glob('*.npz')} == set(stems), 'Unexpected/missing fit NPZ')
    require({p.stem for p in (stage / 'stage_b_receipts').glob('*.json')} == set(stems), 'Unexpected/missing receipt')
    attempts = [json.loads(line) for line in (stage / 'STAGE_B_ATTEMPTS.jsonl').read_text().splitlines() if line]
    require(len(attempts) == 18, 'Unexpected number of formal attempts')
    predictions, validations, records = {}, {}, {}
    npz_hashes, fit_checks, summary_rows = {}, [], []
    total_updates, total_frozen_scalar_endpoint_checks, total_reported_step_parameter_checks = 0, 0, 0
    previous_completion = pre['started_monotonic']
    for ordinal, task in enumerate(tasks, start=1):
        arm, seed, cut = task['arm'], task['seed'], task['cut']
        stem = '{arm}_seed{seed}_cut{cut}'.format(**task)
        record = read_json(stage / 'stage_b_receipts' / (stem + '.json'))
        records[(arm, seed, cut)] = record
        require(record['task'] == task and record['ordinal'] == ordinal and record['formal_stage'] == 'B',
                'Receipt task identity differs: ' + stem)
        attempt = attempts[ordinal - 1]
        require(attempt['task'] == task and attempt['ordinal'] == ordinal and attempt['status'] == 'started'
                and attempt['original_deadline_monotonic'] == pre['deadline_monotonic'], 'Attempt identity/deadline differs')
        require(previous_completion <= attempt['monotonic'] <= record['completed_monotonic'] < pre['deadline_monotonic'],
                'Nonserial/out-of-budget completion: ' + stem)
        previous_completion = record['completed_monotonic']
        require(record['arm'] == arm and record['seed'] == seed and record['cut'] == cut
                and record['author_commit'] == AUTHOR_COMMIT and record['config'] == CONFIG
                and not record['historical_predictions_or_weights_used']
                and record['no_drift'] == (arm == 'PROCEED_no_drift'), 'Fit identity/configuration differs')
        s = chronological_support(dates, cut)
        train, validation, ids = s['training'], s['validation'], s['prediction']
        expected_support = {'train_n': len(train), 'validation_n': len(validation),
                            'pool_start': int(train[0]), 'train_first_origin': int(train[0]),
                            'train_max_origin': int(train[-1]), 'train_max_target': int(train[-1] + 20),
                            'validation_first_origin': int(validation[0]),
                            'validation_max_target': int(validation[-1] + 20), 'boundary_origin': cut - 1}
        for key, value in expected_support.items():
            require(record[key] == value, 'Support field differs: ' + stem + '/' + key)
        mean, sd = input_scalers(price, train, True)
        scale = target_scale_from_training(price, train)
        compare([record['input_global_mean'], record['input_global_sd'], record['fair_support_increment_scale']],
                [mean, sd, scale], stem + ' independent train-only scalers')
        require(record['optimizer'] == {'name': 'Adam', 'lr': .003, 'batch_size': 32,
                                       'max_epochs': 60, 'patience': 12,
                                       'minimum_validation_improvement': 1e-6,
                                       'weight_decay': 0.0, 'gradient_clipping': False}, 'Offline optimizer differs')
        selected = check_epoch_selection(record, compare, stem)
        online = record['online']
        for key, value in {'updates': len(ids), 'first_origin': int(ids[0]), 'last_origin': int(ids[-1]),
                           'first_mature_origin': int(ids[0] - 20), 'last_mature_origin': int(ids[-1] - 20)}.items():
            require(online[key] == value, 'Online support differs: ' + stem + '/' + key)
        require(online['trace_columns'] == ONLINE_COLUMNS, 'Online trace columns differ')
        require(all(online[key] for key in ('all_label_maxima_equal_issue_origin',
                    'all_input_maxima_equal_issue_origin',
                    'all_prior_context_label_maxima_equal_issue_origin_minus_one',
                    'frozen_parameters_bitwise_unchanged')), 'Online causal/freeze summary failed')
        require(online['optimizer']['name'] == 'Adam' and online['optimizer']['lr'] == .0003
                and online['optimizer']['batch_size'] == online['optimizer']['steps_per_origin'] == 1
                and online['optimizer']['fresh_online_optimizer'], 'Online optimizer differs')
        total_updates += len(ids)
        path = local / 'fits' / (stem + '.npz')
        npz_hashes[stem] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as saved:
            for field, origins in [('origins', ids), ('training_origins', train), ('validation_origins', validation)]:
                require(np.array_equal(saved[field], origins), 'Saved support differs: ' + stem + '/' + field)
            compare([saved['input_global_mean'], saved['input_global_sd']], [mean, sd], stem + ' NPZ scalers')
            expected_shapes = parameter_shapes(arm)
            states = {}
            for prefix in ('pretrained_', 'final_'):
                state = {name[len(prefix):]: np.asarray(saved[name]) for name in saved.files
                         if name.startswith(prefix) and name != prefix + 'recent_batch'}
                require(set(state) == set(expected_shapes), 'Saved parameter identity differs: ' + stem)
                for name, shape in expected_shapes.items():
                    require(state[name].shape == shape and state[name].dtype == np.float64
                            and np.isfinite(state[name]).all(), 'Saved parameter shape/dtype/value differs: ' + name)
                require(sum(value.size for value in state.values()) == record['parameter_count']
                        == (420 if arm == 'NLinear_OGD' else 2324), 'Parameter count differs')
                digest = record['state_sha256'] if prefix == 'pretrained_' else record['final_state_sha256']
                require(state_hash(state) == digest, 'Complete saved state hash differs: ' + stem)
                states[prefix] = state
            before, after = states['pretrained_'], states['final_']
            online_trainable = 0
            frozen_tensor_count = 0
            for group, names in groups(arm).items():
                offline_record = record['selected_pretraining_parameter_audit'][group]
                start_record, end_record = online['initial_parameter_audit'][group], online['final_parameter_audit'][group]
                n = sum(before[name].size for name in names)
                trainable = n if group in ('backbone_weights', 'fixed_adapter_bias') or arm == 'NLinear_OGD' else 0
                offline_trainable = n if group != 'backbone_bias' or arm == 'NLinear_OGD' else 0
                for row, required_trainable in [(offline_record, offline_trainable),
                                                 (start_record, trainable), (end_record, trainable)]:
                    require(set(row['names']) == set(names) and row['parameter_count'] == n
                            and row['trainable_count'] == required_trainable, 'Parameter group identity/count differs')
                require(offline_record['sha256'] == start_record['sha256'] == end_record['initial_sha256']
                        == state_hash(before, names), 'Initial/selected parameter group hash differs')
                require(offline_record['initial_sha256'] == record['initial_parameter_audit'][group]['sha256'],
                        'Offline initial group receipts disagree')
                require(end_record['sha256'] == state_hash(after, names), 'Final parameter group hash differs')
                change = max([float(np.max(np.abs(after[name] - before[name]))) for name in names] or [0.0])
                unchanged = all(np.array_equal(after[name], before[name]) for name in names)
                compare(end_record['maximum_absolute_change'], change, stem + ' parameter group change ' + group, 0.0)
                require(end_record['bitwise_unchanged'] == unchanged, 'Parameter group unchanged flag differs')
                online_trainable += trainable
                if n and not trainable:
                    require(unchanged, 'Frozen parameter changed at stream endpoints: ' + stem + '/' + group)
                    total_frozen_scalar_endpoint_checks += n
                    frozen_tensor_count += len(names)
            require(online_trainable == (420 if arm == 'NLinear_OGD' else 460), 'Author online trainable count differs')
            require(online['frozen_parameter_checks'] == len(ids) * frozen_tensor_count,
                    'Reported per-step frozen tensor-check count differs')
            total_reported_step_parameter_checks += online['frozen_parameter_checks']
            gradient_names = record['parameters_with_nonzero_gradient_in_any_attempted_pretraining_batch']
            require(set(gradient_names) <= set(expected_shapes), 'Unknown pretraining gradient parameter')
            if arm != 'NLinear_OGD':
                require('backbone.Linear.bias' not in gradient_names, 'Frozen original bias received gradients')
            if arm == 'PROCEED_no_drift':
                inactive = [name for name in expected_shapes if name.startswith('mlp') or name == G + 'weights.0.weight']
                require(not set(inactive).intersection(gradient_names), 'No-drift encoder/input generator unexpectedly had gradients')

            prediction = np.asarray(saved['predictions'])
            require(prediction.shape == (len(ids), 20) and np.isfinite(prediction).all(), 'Invalid issued predictions')
            trace = np.asarray(saved['online_trace'])
            require(trace.shape == (len(ids), 12) and np.isfinite(trace).all()
                    and bool((trace[:, 10:] >= 0).all()), 'Invalid online trace')
            columns = np.column_stack([ids, ids - 20, ids - 39, ids - 20, ids - 19,
                                       ids, ids - 19, ids, ids - 1, ids])
            compare(trace[:, :10], columns, stem + ' all online maturity/context indices', 0.0)
            require(hashlib.sha256(trace.tobytes()).hexdigest() == online['trace_sha256'], 'Online trace hash differs')
            compare([trace[:, 10].mean(), trace[:, 10].max(), trace[:, 11].mean(), trace[:, 11].max()],
                    [online['loss_mean'], online['loss_max'], online['gradient_norm_mean'], online['gradient_norm_max']],
                    stem + ' online trace summaries')
            first_context = mature_context(price[:cut], cut - 21, mean, sd)
            final_context = mature_context(price[:int(ids[-1]) + 1], int(ids[-1]) - 20, mean, sd)
            if arm == 'NLinear_OGD':
                require(all(saved[key].shape == (0, 40, 1) for key in
                            ('pretrained_recent_batch', 'online_initial_recent_batch', 'final_recent_batch')),
                        'Plain NLinear should not have a recent context buffer')
            else:
                compare(saved['online_initial_recent_batch'], first_context, stem + ' initial mature context')
                compare(saved['final_recent_batch'], final_context, stem + ' final mature context')
                trained_context = np.asarray(saved['pretrained_recent_batch'])
                if selected == 0:
                    compare(trained_context, np.zeros((1, 40, 1)), stem + ' epoch-zero saved context', 0.0)
                else:
                    require(trained_context.shape == (len(train) % 32 or 32, 40, 1), 'Best context minibatch count differs')
                    candidates = np.concatenate([mature_context(price[:cut], int(t), mean, sd) for t in train], axis=0)
                    for row in trained_context:
                        require(float(np.min(np.max(np.abs(candidates - row), axis=(1, 2)))) < 1e-12,
                                'Saved best context includes a nontraining sample')
                if arm == 'PROCEED_no_drift':
                    x = normalized_window(price[:cut], cut - 1, mean, sd)
                    no_context = normalized_forward(arm, before, x, np.zeros((1, 40, 1)))
                    extreme_context = normalized_forward(arm, before, x, np.full((1, 40, 1), 123.0))
                    compare(no_context, extreme_context, stem + ' independent no-drift context invariance', 0.0)

            boundary = forecast(arm, before, price[:cut], cut - 1, mean, sd, first_context)
            compare(saved['boundary_prediction'], boundary, stem + ' independent boundary inference')
            require(hashlib.sha256(saved['boundary_prediction'].tobytes()).hexdigest()
                    == record['boundary_forecast_sha256'], 'Boundary prediction hash differs')
            require(record['frozen_boundary_suffix_difference'] == 0.0, 'Reported frozen suffix check failed')
            altered = price.copy()
            altered[cut:] = 100000.0 + 7.0 * np.arange(len(price) - cut)
            altered_context = mature_context(altered[:cut], cut - 21, mean, sd)
            altered_prediction = forecast(arm, before, altered[:cut], cut - 1, mean, sd, altered_context)
            compare(boundary, altered_prediction, stem + ' independent frozen suffix invariance', 0.0)
            final_prediction = forecast(arm, after, price[:int(ids[-1]) + 1], int(ids[-1]), mean, sd, final_context)
            compare(prediction[-1], final_prediction, stem + ' independent last online inference')
            # Read-only first mature loss. No gradient or optimizer replay is used.
            mature_origin = cut - 20
            first_x = normalized_window(price[:cut + 1], mature_origin, mean, sd)
            first_level = normalized_forward(arm, before, first_x, first_context)[0]
            first_y = (price[mature_origin + 1:cut + 1] - mean) / sd
            compare(trace[0, 10], float(np.mean((first_level - first_y) ** 2)), stem + ' first mature preupdate loss')
            valid = validation_forecasts(arm, before, price[:cut], validation, mean, sd)
            compare(saved['validation_predictions'], valid, stem + ' independent causal validation inference')
            target = price_change_labels(price, validation)
            compare(record['best_validation'], float(np.mean(((valid - target) / sd) ** 2)),
                    stem + ' independent selected validation loss')
            hidx = [0, 4, 19]
            primary = float(np.mean(np.mean((valid[:, hidx] - target[:, hidx]) ** 2, axis=0)
                                    / np.mean(target[:, hidx] ** 2, axis=0)))
            compare(record['validation_primary'], primary, stem + ' independent prior validation score')
            predictions[(arm, seed, cut)] = prediction.copy()
            validations[(arm, seed, cut)] = primary
        summary_rows.append({key: value for key, value in record.items() if key != 'trace'})
        fit_checks.append({'case': stem, 'train_n': len(train), 'validation_n': len(validation),
                           'epochs': record['epochs'], 'selected_epoch': selected,
                           'online_updates': len(ids), 'online_trainable_parameters': online_trainable,
                           'all_maturity_indices_verified': True,
                           'pretrained_and_final_state_hashes_verified': True,
                           'independent_forward_checks': ['60 causal validation origins', 'pre-online boundary',
                                                          'last online prediction', 'first mature preupdate loss'],
                           'frozen_endpoint_arrays_bitwise_equal': True})

    require(total_updates == 4239, 'Total independently counted updates differs')
    training_summary = read_json(stage / 'STAGE_B_TRAINING_SUMMARY.json')
    require(training_summary['records'] == summary_rows and training_summary['formal_fits'] == 18
            and training_summary['online_updates'] == 4239, 'Public training summary differs from full receipts')
    for seed in SEEDS:
        for cut in CUTS:
            full = records[('PROCEED', seed, cut)]['initial_parameter_audit']
            control = records[('PROCEED_no_drift', seed, cut)]['initial_parameter_audit']
            require(full == control, 'Full/no-drift initial parameter groups are not paired')

    independent_seeds, per_cell = [], []
    for seed in SEEDS:
        for arm in ARMS + ('RW', 'stat'):
            delta = (np.vstack([predictions[(arm, seed, cut)] for cut in CUTS]) if arm in ARMS
                     else np.zeros((471, 20)) if arm == 'RW' else reference[459:930])
            score = six_cell_scores(price, dates, np.arange(459, 930), delta)
            prior = [validations[(arm, seed, cut)] for cut in CUTS] if arm in ARMS else []
            independent_seeds.append({'arm': arm, 'seed': seed, 'score': score['score'],
                                      'cells': [cell['ratio'] for cell in score['cells']], 'prior_validation': prior})
            for cell in score['cells']:
                scored = [t for t in range(459, 930) if dates[t][:4] == cell['year']
                          and dates[t + 20][:4] == cell['year']]
                h = cell['horizon']
                errors = [float(delta[t - 459, h - 1]) - float(price[t + h] - price[t]) for t in scored]
                per_cell.append({'arm': arm, 'seed': seed, 'year': cell['year'], 'horizon': h,
                                 'origins': len(scored), 'mse': cell['mse'], 'rw_mse': cell['rw_mse'],
                                 'relative_mse': cell['ratio'], 'mae': math.fsum(abs(e) for e in errors) / len(scored),
                                 'rmse': math.sqrt(cell['mse'])})
    require(len(aggregate['per_seed']) == 15 and len(aggregate['per_cell']) == 90
            and len(aggregate['rows']) == 5, 'Published aggregate row counts differ')
    for observed, expected in zip(aggregate['per_seed'], independent_seeds):
        require((observed['arm'], observed['seed']) == (expected['arm'], expected['seed']), 'Per-seed identity/order differs')
        for key in ('score', 'cells', 'prior_validation'):
            compare(observed[key], expected[key], f"per-seed {expected['arm']}/{expected['seed']} {key}")
    for observed, expected in zip(aggregate['per_cell'], per_cell):
        for key in ('arm', 'seed', 'year', 'horizon', 'origins'):
            require(observed[key] == expected[key], 'Per-cell identity/order differs')
        for key in ('mse', 'rw_mse', 'relative_mse', 'mae', 'rmse'):
            compare(observed[key], expected[key], 'independent per-cell ' + key)
    independent_arms = []
    for arm in ARMS + ('RW', 'stat'):
        rows = [row for row in independent_seeds if row['arm'] == arm]
        values = np.asarray([row['score'] for row in rows])
        cells = np.asarray([row['cells'] for row in rows])
        independent_arms.append({'arm': arm, 'seeds': list(SEEDS), 'mean_score': float(values.mean()),
                                 'sd_score': float(values.std(ddof=1)), 'min_score': float(values.min()),
                                 'max_score': float(values.max()), 'cells_mean': cells.mean(axis=0).tolist(),
                                 'cells_sd': cells.std(axis=0, ddof=1).tolist()})
    for observed, expected in zip(aggregate['rows'], independent_arms):
        require(observed['arm'] == expected['arm'] and observed['seeds'] == expected['seeds'], 'Per-arm identity differs')
        for key in ('mean_score', 'sd_score', 'min_score', 'max_score', 'cells_mean', 'cells_sd'):
            compare(observed[key], expected[key], 'independent per-arm ' + expected['arm'] + '/' + key)
    with (stage / 'STAGE_B_PER_SEED.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 15, 'Per-seed CSV row count differs')
    for row, expected in zip(rows, independent_seeds):
        require(row['arm'] == expected['arm'] and int(row['seed']) == expected['seed'], 'Per-seed CSV identity differs')
        compare([float(row['score'])] + [float(row[key]) for key in aggregate['cell_order']],
                [expected['score']] + expected['cells'], 'independent per-seed CSV')
    with (stage / 'STAGE_B_PER_CELL.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 90, 'Per-cell CSV row count differs')
    for row, expected in zip(rows, per_cell):
        require(row['arm'] == expected['arm'] and int(row['seed']) == expected['seed']
                and row['year'] == expected['year'] and int(row['horizon']) == expected['horizon']
                and int(row['origins']) == expected['origins'], 'Per-cell CSV identity differs')
        for key in ('mse', 'rw_mse', 'relative_mse', 'mae', 'rmse'):
            compare(float(row[key]), expected[key], 'independent per-cell CSV ' + key)
    mean_by_arm = {row['arm']: row['mean_score'] for row in independent_arms}
    seed_scores = {(row['arm'], row['seed']): row['score'] for row in independent_seeds}
    differences = [seed_scores[('PROCEED', seed)] - seed_scores[('PROCEED_no_drift', seed)] for seed in SEEDS]
    mechanism = {'full_minus_no_drift_by_seed': differences,
                 'full_beats_no_drift_mean': mean_by_arm['PROCEED'] < mean_by_arm['PROCEED_no_drift'],
                 'full_beats_plain_mean': mean_by_arm['PROCEED'] < mean_by_arm['NLinear_OGD'],
                 'full_beats_RW_mean': mean_by_arm['PROCEED'] < 1.0,
                 'full_beats_stat_mean': mean_by_arm['PROCEED'] < mean_by_arm['stat']}
    for key, value in mechanism.items():
        if key == 'full_minus_no_drift_by_seed':
            compare(aggregate['mechanism_diagnostic'][key], value, 'Paired no-drift score differences')
        else:
            require(aggregate['mechanism_diagnostic'][key] == value, 'Mechanism diagnostic sign differs')
    require(aggregate['formal_fits'] == 18 and aggregate['online_updates'] == 4239
            and aggregate['score_origins'] == 451 and aggregate['annual_origin_counts'] == [224, 227]
            and aggregate['horizons'] == [1, 5, 20], 'Aggregate protocol metadata differs')
    require(aggregate['protocol_adaptation'] and aggregate['no_selection_of_new_hyperparameters']
            and not aggregate['unexposed_test'] and not aggregate['novelty_established']
            and not aggregate['original_benchmark_reproduction'], 'Aggregate scope overstated')
    timing = read_json(stage / 'STAGE_B_TIMING.json')
    timing_plan = read_json(stage / 'STAGE_B_TIMING_PLAN.json')
    require(timing['timing_fits'] == timing_plan['timing_fits'] == 3
            and timing['market_online_updates'] == timing_plan['market_online_updates'] == 0
            and not timing['development_year_scores_computed']
            and not timing_plan['development_year_scores_computed']
            and timing['source_sha256'] == timing_plan['source_sha256'] == pre['source_sha256'], 'Timing/freeze evidence differs')
    require(timing['environment'] == pre['environment'], 'Environment changed between timing and formal run')
    require({p.stem for p in (stage / 'stage_b_timing_receipts').glob('*.json')} == set(ARMS), 'Unexpected timing fit count')
    for arm in ARMS:
        row = read_json(stage / 'stage_b_timing_receipts' / (arm + '.json'))
        require(row['arm'] == arm and row['cut'] == 459 and row['seed'] == 113 and row['epochs'] == 3
                and row['optimizer']['max_epochs'] == 3, 'Timing scope differs')
    return {'passed': True,
            'scope': 'Independent approved-prefix reconstruction, author-source identity, mature-label and saved-state/forecast audit; no training or gradients',
            'market_fits_performed_by_auditor': 0, 'market_online_updates_performed_by_auditor': 0,
            'market_backward_calls_performed_by_auditor': 0, 'third_party_training_reproduction': False,
            'unexposed_test': False, 'novelty_certified': False, 'data': data_receipt,
            'support': support_receipt(price, dates), 'formal_fits_checked': 18,
            'timing_fits_checked_from_receipts': 3, 'online_updates_checked': total_updates,
            'main_score_origins': 451, 'annual_counts': [224, 227], 'online_origins_per_seed_per_arm': 471,
            'cross_year_unscored_origins_per_seed_per_arm': 20,
            'per_seed_rows_checked': 15, 'six_cell_rows_checked': 90,
            'independent_numeric_comparisons': int(count), 'maximum_absolute_difference': max_difference,
            'maximum_difference_location': max_label, 'tolerance': 1e-8,
            'independent_frozen_scalar_endpoint_checks': total_frozen_scalar_endpoint_checks,
            'reported_per_step_frozen_tensor_checks_reconciled': total_reported_step_parameter_checks,
            'independent_validation_prediction_vectors': 18 * 60,
            'independent_boundary_prediction_vectors': 18, 'independent_last_online_prediction_vectors': 18,
            'independent_first_mature_loss_checks': 18, 'fit_checks': fit_checks,
            'independent_per_arm': independent_arms, 'independent_per_seed': independent_seeds,
            'independent_per_cell': per_cell, 'independent_mechanism_diagnostic': mechanism,
            'source_sha256': pre['source_sha256'], 'prediction_npz_sha256': npz_hashes,
            'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'shared_auditor_helper_sha256': hashlib.sha256((stage / 'audit/verify_stage_a.py').read_bytes()).hexdigest(),
            'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch_imported': False},
            'limitations': [
                'Intermediate online parameter states were not saved. All issued forecasts are rescored, but only the final online forecast of each stream is independently reproduced from online endpoint weights.',
                'The first mature preupdate loss, all validation predictions, and pre-online boundary predictions are reconstructed; no optimizer or backward pass is replayed.',
                'Stepwise parameter immutability was enforced by the reviewed runner; the auditor independently compares saved start/end arrays and reconciles the reported count, not unsaved intermediate arrays.',
                'Identical per-arm sampler seeds and order generation are statically verified; individual offline minibatch permutations were not logged.',
                'Full/no-drift nominal modules and initialization match; removing drift also eliminates effective encoder/input-generator gradients, so this is not a matched-effective-capacity experiment.',
                'Plain NLinear has no generator. Its online optimizer metadata generic generator_frozen/fixed_adapter_bias_trainable flags are inapplicable; actual parameter groups and counts are audited.',
                'RW/stat repeated seed rows are deterministic controls, not additional independent models or markets.',
                'Author core is preserved under explicit local training/validation/data/update adaptations; original benchmark replication, independent test performance, and method novelty are not certified.',
            ]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    stage = Path(__file__).resolve().parents[1]
    if args.support_only:
        price, dates, source = read_approved(stage.parents[1] / 'data/parsed/figshare_28682348/guangzhou_reported.csv')
        print(json.dumps({'source': source, 'support': support_receipt(price, dates),
                          'rw': six_cell_scores(price, dates, np.arange(459, 930), np.zeros((471, 20))),
                          'expected_fits': 18, 'expected_online_updates': 4239}, indent=2))
        return
    result = verify(stage)
    output = args.output or stage / 'audit/AUDIT_STAGE_B.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('passed', 'formal_fits_checked', 'online_updates_checked',
                    'market_fits_performed_by_auditor', 'independent_numeric_comparisons',
                    'maximum_absolute_difference', 'maximum_difference_location',
                    'independent_per_arm', 'independent_mechanism_diagnostic')}, indent=2))


if __name__ == '__main__':
    main()
