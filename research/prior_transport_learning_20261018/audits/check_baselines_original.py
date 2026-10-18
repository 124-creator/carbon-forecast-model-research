"""Synthetic-only baseline smoke. Does not import data.py or open market files."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'carbon_workspace/research/prior_transport_learning_20261018/source/baselines.py'
spec = importlib.util.spec_from_file_location('reviewed_baselines', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestFeaturesOnly(dict):
    def __init__(self, x):
        super().__init__(x=x)
        self.accesses = 0

    def __getitem__(self, key):
        if key != 'x':
            raise AssertionError('Test-label or other test-field access: ' + str(key))
        self.accesses += 1
        return super().__getitem__(key)


def main():
    rng = np.random.default_rng(17019)
    checks = []
    def check(name, condition, **evidence):
        assert bool(condition), (name, evidence)
        checks.append({'name': name, 'passed': True, **evidence})

    # Explicit constructor dtype must remove dependence on global default dtype.
    original_dtype = module.torch.get_default_dtype()
    initial_states = []
    try:
        for dtype in (module.torch.float32, module.torch.float64):
            module.torch.set_default_dtype(dtype)
            with module.torch.random.fork_rng(devices=[]):
                module.torch.manual_seed(17)
                m = module.SupervisedGRU(('TARGET',))
            initial_states.append(module.state_sha256(m.state_dict()))
    finally:
        module.torch.set_default_dtype(original_dtype)
    check('explicit_float64_initialization_independent_of_global_default', initial_states[0] == initial_states[1])

    def batch(n):
        x = rng.normal(size=(n, 20, 3))
        signal = 0.6 * x[:, -1, 0] + 0.3 * x[:, :, 1].mean(1)
        y = signal[:, None] * np.linspace(1, 0.4, 20)[None, :]
        sigma = np.exp(rng.normal(0, .2, (n, 1))) * np.sqrt(np.arange(1, 21))[None, :]
        return {'x': x, 'y': y, 'sigma': sigma, 'delta': y * sigma}

    support, meta, validation = batch(240), batch(80), batch(80)
    sources = {'SOURCE_A': batch(137), 'SOURCE_B': batch(151)}
    test_x = rng.normal(size=(19, 20, 3))

    def subset(source, n):
        ids = np.rint(np.linspace(0, len(source['x']) - 1, n)).astype(int)
        return {k: v[ids].copy() for k, v in source.items()}

    def case(budget):
        return {'target': 'TARGET', 'budget': budget, 'year': 'SYNTHETIC',
                'target_support': subset(support, int(.6 * budget)),
                'target_meta': subset(meta, int(.2 * budget)),
                'target_val': subset(validation, int(.2 * budget)),
                'target_test': TestFeaturesOnly(test_x.copy()),
                'sources': sources, 'source_cutoff': 'SYNTHETIC_NO_DATES'}

    config = {'max_steps': 4, 'validation_every': 1, 'patience': 10}
    runs = []
    for name, budget in [('local_gru', 100), ('local_gru', 100), ('pooled_gru', 100), ('pooled_gru', 400)]:
        c = case(budget)
        out = module.fit_baseline(c, name, seed=17, config=config)
        r, tr = out['receipt'], out['traces']
        prefix = f'{len(runs)}_{name}_{budget}'
        check(prefix + '_prediction_shape', out['predictions'].shape == (19, 20))
        check(prefix + '_finite_predictions', np.isfinite(out['predictions']).all())
        check(prefix + '_test_features_once', c['target_test'].accesses == 1 and r['test_forward_calls'] == 1)
        check(prefix + '_test_labels_not_accessed', r['test_labels_accessed'] is False)
        check(prefix + '_initial_rw', r['initial_predictions_equal_rw'] and tr[0]['validation']['score'] == 1.0)
        check(prefix + '_target_support_plus_meta', r['target_training_count'] == int(.8 * budget))
        check(prefix + '_optimizer_actually_updated', r['steps_completed'] == 4 and r['last_state_sha256'] != r['initial_state_sha256'])
        check(prefix + '_selected_state_identity', r['selected_state_sha256'] == module.state_sha256(out['state']))
        check(prefix + '_selected_by_validation', r['best_validation_score'] == tr[r['best_step']]['validation']['score'])
        check(prefix + '_double_cpu', all(t.dtype == module.torch.float64 and t.device.type == 'cpu' for t in out['state'].values()))
        selected_scores = [row['validation']['score'] for row in tr if row['selected']]
        check(prefix + '_min_delta_selection', all(a - b > 1e-6 for a, b in zip(selected_scores, selected_scores[1:])))
        for row in tr[1:]:
            weighted = sum(r['market_loss_weights'][m] * v for m, v in row['per_market_training_mse'].items())
            check(prefix + '_weighted_loss_' + str(row['step']), abs(weighted - row['training_loss']) < 1e-12)
        if name == 'local_gru':
            check(prefix + '_local_only', r['source_counts'] == {} and r['market_loss_weights'] == {'TARGET': 1.0})
        else:
            check(prefix + '_pooled_target_half', r['market_loss_weights']['TARGET'] == .5)
            check(prefix + '_pooled_sources_equal_half', r['market_loss_weights']['SOURCE_A'] == .25 and r['market_loss_weights']['SOURCE_B'] == .25)
        runs.append(out)

    check('repeat_same_seed_exact_predictions', np.array_equal(runs[0]['predictions'], runs[1]['predictions']))
    check('repeat_same_seed_exact_state', runs[0]['receipt']['selected_state_sha256'] == runs[1]['receipt']['selected_state_sha256'])
    for market in ('SOURCE_A', 'SOURCE_B'):
        check('fixed_source_schedule_across_budgets_' + market,
              runs[2]['receipt']['sampled_indices_sha256'][market] == runs[3]['receipt']['sampled_indices_sha256'][market])
    check('architecture_parameter_counts', runs[0]['receipt']['parameter_count'] == 1348 and runs[2]['receipt']['parameter_count'] == 2028)

    receipt = {'status': 'PASS', 'created_utc': datetime.now(timezone.utc).isoformat(),
               'source': str(SOURCE), 'baseline_source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'checks': len(checks), 'synthetic_training_calls': 4,
               'synthetic_optimizer_updates': 16, 'synthetic_test_forward_calls': 4,
               'market_files_read': 0, 'confirmation_files_read': 0, 'market_fits': 0,
               'case_source': 'NumPy-generated tensors, no data.py import, no price series',
               'training_override': config,
               'formal_defaults_not_executed': module.DEFAULT_CONFIG,
               'details': checks,
               'run_receipts': [r['receipt'] for r in runs]}
    output = Path(__file__).with_name('BASELINES_SYNTHETIC_RECEIPT.json')
    if output.exists():
        raise FileExistsError('Preserve prior synthetic receipt')
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('details', 'run_receipts')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
