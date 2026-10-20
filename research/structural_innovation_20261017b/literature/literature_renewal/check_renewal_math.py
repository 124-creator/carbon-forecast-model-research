"""Pure synthetic / source-semantics checks. No market files or model fitting.

This is an independent mathematical audit, not execution of the MXNet author
network and not a trained forecasting model. Only the adjacent source manifest
and extracted author code are read.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def branches(age: int, sign: int, symmetric: bool = False):
    q = 0.15 + 0.20 * age + 0.04 * (sign + 1)
    plus = 0.5 if symmetric else 0.25 + 0.15 * age + 0.10 * sign
    magnitude = 1 if symmetric else 1 + int(age > 0)
    return [
        ((min(age + 1, 2), sign), 0, 1.0 - q),
        ((0, 1), magnitude, q * plus),
        ((0, -1), -1, q * (1.0 - plus)),
    ]


STATES = [(a, s) for a in range(3) for s in (-1, 1)]
INDEX = {z: i for i, z in enumerate(STATES)}


def kernel(frequency: float, symmetric: bool = False):
    k = np.zeros((len(STATES), len(STATES)), dtype=np.complex128)
    for i, (age, sign) in enumerate(STATES):
        for z, reward, probability in branches(age, sign, symmetric):
            k[i, INDEX[z]] += probability * np.exp(1j * frequency * reward)
    return k


def enumerate_law(horizon: int, symmetric: bool = False):
    # State and accumulated reward are retained separately; no Fourier recursion.
    law = {((0, 1), 0): 1.0}
    for _ in range(horizon):
        new = defaultdict(float)
        for ((age, sign), total), probability in law.items():
            for z, reward, transition in branches(age, sign, symmetric):
                new[(z, total + reward)] += probability * transition
        law = dict(new)
    return law


def main():
    manifest = json.loads((ROOT / 'AUTHOR_SOURCE_MANIFEST.json').read_text())
    source_identities = []
    for entry in manifest['files']:
        body = (ROOT / entry['path']).read_bytes()
        computed = hashlib.sha256(body).hexdigest()
        assert computed == entry['sha256']
        source_identities.append({'path': entry['path'], 'sha256': computed})
    net = (ROOT / 'author_gluonts/src/gluonts/mx/model/renewal/_network.py').read_text()
    feature = (ROOT / 'author_gluonts/src/gluonts/transform/feature.py').read_text()
    source_conditions = {
        'interval_training_target_Q_minus_1': 'dist_interval.log_prob(data_interval - 1)' in net,
        'strict_first_interval_acceptance': 'interval_sample > repeated_time_remaining' in net,
        'fallback_if_less_or_equal': 'interval_sample <= repeated_time_remaining' in net,
        'fifty_rejection_attempts': 'for j in range(50):' in net,
        'subtract_elapsed_after_sampling': 'interval_samples[0] = interval_samples[0] - repeated_time_remaining' in net,
        'elapsed_is_number_of_trailing_zero_observations': 'len(target) - len(np.trim_zeros(target, "b"))' in feature,
    }
    assert all(source_conditions.values())

    start = np.zeros(len(STATES))
    start[INDEX[(0, 1)]] = 1.0
    one = np.ones(len(STATES))
    p = kernel(0.0).real
    reward_mean = np.array([
        sum(value * prob for _, value, prob in branches(*z)) for z in STATES
    ])
    max_cf_error = 0.0
    max_mean_error = 0.0
    all_means = []
    for h in range(1, 7):
        law = enumerate_law(h)
        expected = sum(total * prob for (_, total), prob in law.items())
        propagated = sum(float(start @ np.linalg.matrix_power(p, j) @ reward_mean) for j in range(h))
        max_mean_error = max(max_mean_error, abs(expected - propagated))
        all_means.append({'h': h, 'enumerated_mean': expected, 'matrix_mean': propagated})
        for u in (0.0, 0.2, 1.2, 4.1):
            direct = sum(prob * np.exp(1j * u * total) for (_, total), prob in law.items())
            matrix = start @ np.linalg.matrix_power(kernel(u), h) @ one
            max_cf_error = max(max_cf_error, float(abs(direct - matrix)))
    assert max_cf_error < 1e-12 and max_mean_error < 1e-12
    assert np.max(np.abs(p.sum(axis=1) - 1.0)) < 1e-14

    symmetric_means = []
    symmetric_variances = []
    for h in (1, 5, 20):
        law = enumerate_law(h, symmetric=True)
        mean = sum(total * prob for (_, total), prob in law.items())
        variance = sum(total * total * prob for (_, total), prob in law.items()) - mean * mean
        symmetric_means.append(mean)
        symmetric_variances.append(variance)
    assert max(abs(x) for x in symmetric_means) < 1e-12
    assert min(symmetric_variances) > 0.0

    # Finite equally spaced CF frequencies do not identify arbitrary laws.
    spacing = 0.5
    u = spacing * np.arange(-10, 11)
    alias_shift = 2.0 * math.pi / spacing
    cf_grid_error = float(np.max(np.abs(np.ones_like(u) - np.exp(1j * u * alias_shift))))
    assert cf_grid_error < 1e-12
    finite_grid_counterexample = {
        'spacing': spacing,
        'delta_0_and_delta_shift_have_same_sampled_CF': True,
        'shift': alias_shift,
        'maximum_sampled_CF_difference': cf_grid_error,
        'exact_CRPS_at_y0_for_delta_0': 0.0,
        'exact_CRPS_at_y0_for_delta_shift': alias_shift,
    }

    # X=Q-1 ~ geometric(p) on {0,1,...}. Q>a is equivalent to X>=a.
    # The extracted branch accepts X>a and otherwise forces W=1 after 50 tries.
    waiting = []
    geometric_p = 0.5
    for age in (0, 1, 5, 20):
        cdf_at_age = -math.expm1((age + 1) * math.log1p(-geometric_p))
        fallback = cdf_at_age ** 50
        waiting.append({
            'elapsed_zero_observations': age,
            'correct_P_remaining_wait_1': geometric_p,
            'source_branch_P_remaining_wait_1': fallback,
            'correct_mean_remaining_wait': 1.0 / geometric_p,
            'source_branch_mean_remaining_wait': 1.0 + (1.0 - fallback) / geometric_p,
        })
    assert waiting[0]['source_branch_P_remaining_wait_1'] < 1e-12
    assert waiting[-1]['source_branch_P_remaining_wait_1'] > 0.99

    closure = {
        'future_path_A': [1, 1],
        'future_path_B': [-1, 1],
        'same_fixed_origin_context': True,
        'same_end_age_and_last_sign': [0, 1],
        'true_next_mark_rule': 'equals penultimate future event sign',
        'true_next_mark_mean_A': 1,
        'true_next_mark_mean_B': -1,
        'conclusion': 'Fixed-context age/last-sign closure cannot represent both conditional continuations; this is not a proof that no chosen finite set of marginals can coincide accidentally.',
    }

    out = {
        'status': 'PASS',
        'scope': 'Independent pure synthetic enumeration and source-semantics algebra; no MXNet author class execution.',
        'market_files_read': 0,
        'confirmation_files_read': 0,
        'market_fits': 0,
        'synthetic_model_fits': 0,
        'author_source_identities': source_identities,
        'source_conditions': source_conditions,
        'markov_additive_equivalence': {'max_cf_error': max_cf_error, 'max_mean_error': max_mean_error, 'means': all_means},
        'hazard_predictability_without_signed_mean': {'horizons': [1, 5, 20], 'means': symmetric_means, 'variances': symmetric_variances},
        'finite_CF_grid_not_strictly_proper_in_general': finite_grid_counterexample,
        'first_wait_source_semantics_counterexample': waiting,
        'finite_state_closure_counterexample': closure,
        'net_zero_does_not_identify_no_latent_events': {'latent_paths': [[], [1, -1]], 'same_observed_net_change': 0},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    dest = ROOT / 'RENEWAL_SYNTHETIC.json'
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': out['status'], 'receipt': str(dest), 'cf_error': max_cf_error, 'mean_error': max_mean_error, 'first_wait': waiting}, ensure_ascii=False))


if __name__ == '__main__':
    main()
