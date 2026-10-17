"""New hindsight diagnostics, never old forecast/checkpoint fixtures."""
import unittest
import numpy as np


class DiagnosticTests(unittest.TestCase):
    def test_observed_feature_banks_ignore_future_prices(self):
        from hd_core import feature_bank
        price = 40 + np.exp(np.arange(280) / 300)
        dates = np.datetime_as_string(np.datetime64('2018-01-01') + np.arange(280), unit='D')
        other = price.copy()
        other[201:] += np.arange(79) * 100
        for group, width in [('price', 1), ('dynamics', 5), ('calendar', 8)]:
            actual = feature_bank(price, dates, group)
            changed = feature_bank(other, dates, group)
            self.assertEqual(actual.shape, (280, width))
            np.testing.assert_array_equal(actual[:201], changed[:201])
            self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_array_equal(feature_bank(price, dates, 'price')[1:, 0], np.diff(price))

    def test_configuration_space_has_exact_balanced_factorial_and_extensions(self):
        import json
        from collections import Counter
        from hd_core import configuration_grid
        configs = configuration_grid()
        self.assertEqual(len(configs), 96)
        self.assertEqual(Counter(c['family'] for c in configs), {'gru': 30, 'lstm': 30, 'ridge': 36})
        self.assertEqual(len({c['id'] for c in configs}), 96)
        self.assertEqual(len({json.dumps({k: v for k, v in c.items() if k not in ('id', 'slice')}, sort_keys=True) for c in configs}), 96)
        self.assertEqual(sum(c['slice'] == 'factorial' for c in configs), 48)

    def test_actual_fresh_fits_ignore_unavailable_suffix(self):
        from hd_core import configuration_grid
        from hd_fit import fit, predict
        price = 70 + np.cumsum(np.random.default_rng(7).normal(size=310))
        dates = np.datetime_as_string(np.datetime64('2018-01-01') + np.arange(310), unit='D')
        changed = price.copy()
        changed[281:] += np.arange(29) * 500
        for family in ('ridge', 'gru', 'lstm'):
            config = next(c for c in configuration_grid() if c['family'] == family and c['group'] == 'calendar' and c['length'] == 60 and c['anchor'] == 'rw')
            a, ra = fit(price, dates, 281, config, max_epochs=2)
            b, rb = fit(changed, dates, 281, config, max_epochs=2)
            self.assertEqual(ra['state_sha256'], rb['state_sha256'])
            self.assertLess(ra['train_max_target'], ra['validation_first_origin'])
            x = predict(a, price, dates, np.arange(277, 281))
            y = predict(b, changed, dates, np.arange(277, 281))
            np.testing.assert_array_equal(x, y)
            self.assertTrue(np.isfinite(x).all())
            self.assertEqual(x.shape, (4, 20))

    def test_hindsight_selection_is_not_confused_with_prior_selection(self):
        from hd_analysis import summarize
        risks = np.array([[[.5] * 3, [1.5] * 3], [[1.] * 3, [.6] * 3]])
        validation = np.array([[.2, .2], [.8, .8]])
        result = summarize(risks, validation, [{'id': 'a'}, {'id': 'b'}])
        self.assertEqual(result['hindsight_static']['id'], 'b')
        self.assertAlmostEqual(result['hindsight_static']['score'], .8)
        self.assertEqual(result['hindsight_year_ids'], ['a', 'b'])
        self.assertAlmostEqual(result['hindsight_year_score'], .55)
        self.assertEqual(result['prior_validation_ids'], ['a', 'a'])
        self.assertAlmostEqual(result['prior_validation_score'], 1.)
        self.assertEqual(result['transfer_2020_winner_to_2021']['id'], 'a')
        self.assertAlmostEqual(result['transfer_2020_winner_to_2021']['score'], 1.5)
        self.assertFalse(result['unexposed_test'])

    def test_independent_risk_table_identifies_rw_and_perfect_forecasts(self):
        from hd_verify import risk_table
        price = np.arange(200, dtype=float) + 50
        dates = np.array(['2020-01-01'] * 100 + ['2021-01-01'] * 100)
        ids = np.r_[np.arange(20, 80), np.arange(120, 180)]
        forecasts = np.zeros((2, len(ids), 20))
        forecasts[1] = np.arange(1, 21)
        table, counts = risk_table(price, dates, ids, forecasts)
        np.testing.assert_array_equal(table[0], np.ones((2, 3)))
        np.testing.assert_array_equal(table[1], np.zeros((2, 3)))
        self.assertEqual(counts, [60, 60])
        forecasts[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            risk_table(price, dates, ids, forecasts)

    def test_formal_inventory_refuses_unpaired_cases(self):
        import os
        import tempfile
        from pathlib import Path
        from hd_run import planned_tasks, inventory
        self.assertEqual(len(planned_tasks()), 192)
        self.assertEqual(len(set(planned_tasks())), 192)
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            root = Path(tmp)
            self.assertEqual(inventory(root), set())
            (root / 'c000_t459.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                inventory(root)
            (root / 'c000_t459.npz').write_bytes(b'inventory only')
            self.assertEqual(inventory(root), {'c000_t459'})

    def test_factor_effects_match_only_otherwise_identical_configs(self):
        from hd_analysis import paired_effects
        configs = [dict(id=f'x{i}', family='gru', group='price', length=length, anchor=anchor,
                        shrink=0., hidden=16, lr=.003, weight_decay=.001, slice='factorial')
                   for i, (length, anchor) in enumerate([(20, 'rw'), (20, 'stat'), (60, 'rw'), (60, 'stat')])]
        risks = np.array([np.full((2, 3), x) for x in (1., .9, 1.2, 1.)])
        result = paired_effects(configs, risks)
        effect = result['anchor']['rw->stat']
        self.assertEqual(effect['pairs'], 2)
        self.assertAlmostEqual(effect['mean_right_minus_left'], -.15)

    def test_analysis_refuses_to_rank_incomplete_grid(self):
        import os
        import tempfile
        from pathlib import Path
        from hd_analysis import analyze
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                analyze(root)
            self.assertFalse((root / 'RESULTS.json').exists())

    def test_independent_replay_rejects_an_absent_run(self):
        import os
        import tempfile
        from pathlib import Path
        from hd_verify import replay
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            with self.assertRaises(ValueError):
                replay(Path(tmp))


if __name__ == '__main__':
    unittest.main()
