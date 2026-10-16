"""Behavior tests for the scale/horizon candidate; no market data."""
import unittest
import numpy as np
import torch


class ScaleCoreTests(unittest.TestCase):
    def test_forward_uses_only_available_prefix_and_valid_convex_weights(self):
        from hsc_core import ScaleNet, configure
        configure()
        torch.manual_seed(17)
        price = np.cumsum(np.random.default_rng(2).normal(size=95)) + 30
        context = np.zeros((95, 3))
        model = ScaleNet('horizon_competition', 8)
        origins = np.array([25, 40, 70])
        result = model(price, context, origins, 1.)
        changed = price.copy()
        changed[71:] += 1000
        replay = model(changed, context, origins, 1.)
        self.assertEqual(result['prediction'].shape, (3, 20))
        self.assertEqual(result['experts'].shape, (3, 3, 20))
        torch.testing.assert_close(result['weights'].sum(1), torch.ones(3, 20))
        self.assertTrue(bool((result['weights'] >= 0).all()))
        torch.testing.assert_close(result['prediction'], replay['prediction'], rtol=0, atol=0)
        torch.testing.assert_close(result['prediction'], (result['experts'] * result['weights']).sum(1))

    def test_competition_supervises_router_without_target_gradient(self):
        from hsc_core import ScaleNet, configure, training_loss
        configure()
        torch.manual_seed(5)
        model = ScaleNet('horizon_competition', 8)
        g = torch.randn(7, 3, 9)
        target = torch.randn(7, 20, requires_grad=True)
        with torch.no_grad():
            model.initialize(g, target.detach())
        output = model.combine(g)
        loss, info = training_loss(model, output, target.detach())
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertGreater(float(model.route_interaction.grad.abs().sum()), 0.)
        self.assertGreater(info['competition'], 0.)
        shared = ScaleNet('shared_competition', 8).combine(g)
        torch.testing.assert_close(shared['weights'][:, :, :1].expand_as(shared['weights']), shared['weights'])

    def test_raw_control_matches_existing_fair_features_and_initial_head(self):
        from hsc_core import ScaleNet, configure
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'fair_recurrent_tuning_20261016'))
        from fair_core import RecurrentNet
        configure()
        x = np.cumsum(np.random.default_rng(6).normal(size=90)) + 30
        q = np.zeros((90, 3))
        ids = np.arange(20, 70)
        torch.manual_seed(17)
        old = RecurrentNet('gru_dense', 8)
        torch.manual_seed(17)
        new = ScaleNet('raw_gru', 8)
        a = old.features(x, q, ids, 2.)
        b = new.features(x, q, ids, 2.)
        torch.testing.assert_close(a, b[:, 0], rtol=0, atol=0)
        target = torch.randn(len(ids), 20)
        old.initialize(a, target)
        new.initialize(b, target)
        torch.testing.assert_close(old.correction(a), new.combine(b)['prediction'])

    def test_fresh_fit_is_unchanged_by_unavailable_suffix(self):
        from hsc_fit import fit, predict
        price = np.cumsum(np.random.default_rng(8).normal(size=280)) + 80
        altered = price.copy()
        altered[240:] += 500
        cfg = dict(id='tiny', hidden=4, lr=.003, weight_decay=.001)
        a, scale_a, log_a = fit(price, 240, 'horizon_competition', cfg, 17, max_epochs=2)
        b, scale_b, log_b = fit(altered, 240, 'horizon_competition', cfg, 17, max_epochs=2)
        self.assertEqual(scale_a, scale_b)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
        self.assertLess(log_a['train_max_target'], log_a['validation_first_origin'])
        self.assertEqual(log_b['validation_max_target'], 239)
        prediction, weights = predict(a, scale_a, price, np.arange(240, 245))
        self.assertEqual(prediction.shape, (5, 20))
        self.assertEqual(weights.shape, (5, 3, 20))
        self.assertTrue(np.isfinite(prediction).all())

    def test_selection_requires_complete_search_and_score_uses_seed_losses(self):
        from hsc_run import configurations, select, span_metrics
        from hsc_core import MODES
        records = [dict(family=m, config_id=c['id'], cut=t, score=float(i + 1))
                   for m in MODES for i, c in enumerate(configurations()) for t in (359, 419)]
        self.assertEqual(set(select(records)), set(MODES))
        self.assertTrue(all(v['config_id'] == 'c00' for v in select(records).values()))
        with self.assertRaises(ValueError):
            select(records[:-1])
        actual = np.ones((3, 20))
        prediction = np.stack([np.zeros_like(actual), 2 * actual])
        result = span_metrics(prediction, actual)
        self.assertEqual(result['score'], 1.)
        self.assertEqual(result['ensemble'], 0.)

    def test_independent_original_price_scores_and_invalid_support(self):
        from hsc_verify import independent_scores
        price = np.linspace(10., 20., 70)
        dates = np.array(['2020-01-01'] * 35 + ['2021-01-01'] * 35)
        ids = np.array([5, 6, 40, 41])
        prediction = np.zeros((2, 4, 20))
        result = independent_scores(price, dates, ids, prediction)
        self.assertAlmostEqual(result['score'], 1.)
        self.assertEqual(result['counts'], {'2020': 2, '2021': 2})
        from hsc_run import market_metrics
        produced = market_metrics(price, dates, ids, prediction)
        self.assertAlmostEqual(result['score'], produced['score'])
        for a, b in zip(result['cells'], produced['cells']):
            for key in ('ratio', 'rmse', 'mae', 'mape_percent'):
                self.assertAlmostEqual(a[key], b[key])
        prediction[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            independent_scores(price, dates, ids, prediction)

    def test_real_boundary_comparison_rejects_a_changed_parameter(self):
        from hsc_boundary import compare_parameters
        a = {'x': np.array([1., 2.])}
        compare_parameters(a, {'x': np.array([1., 2.])})
        with self.assertRaises(ValueError):
            compare_parameters(a, {'x': np.array([1., 3.])})

    def test_missing_task_planner_never_repeats_a_saved_pair(self):
        from hsc_resume import missing_tasks
        import tempfile
        import os
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as temp:
            root = Path(temp)
            config = dict(id='c02', hidden=16, lr=.001, weight_decay=.001)
            selection = {'raw_gru': {'config_id': 'c02'}}
            tasks = missing_tasks(root, selection, {'c02': config}, modes=('raw_gru',), cuts=(459,), seeds=(17,))
            self.assertEqual(len(tasks), 1)
            stem = root / 'raw_gru_c02_s17_t459'
            stem.with_suffix('.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                missing_tasks(root, selection, {'c02': config}, modes=('raw_gru',), cuts=(459,), seeds=(17,))
            np.savez(stem.with_suffix('.npz'), prediction=np.zeros((1, 20)))
            self.assertEqual(missing_tasks(root, selection, {'c02': config}, modes=('raw_gru',), cuts=(459,), seeds=(17,)), [])

    def test_partial_inventory_never_qualifies_an_incomplete_model(self):
        from hsc_verify import saved_market_inventory
        import os
        import tempfile
        from pathlib import Path
        selection = {'raw_gru': {'config_id': 'c02'}}
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as temp:
            root = Path(temp)
            for seed in (17, 29):
                for cut in range(459, 900, 40):
                    stem = root / f'raw_gru_c02_s{seed}_t{cut}'
                    stem.with_suffix('.json').write_text('{}', encoding='utf-8')
                    np.savez(stem.with_suffix('.npz'), prediction=np.zeros((1, 20)))
            saved, missing, complete = saved_market_inventory(root, selection)
            self.assertEqual(len(saved), 24)
            self.assertEqual(missing, [])
            self.assertEqual(complete, ['raw_gru'])
            (root / 'raw_gru_c02_s29_t899.json').unlink()
            with self.assertRaises(ValueError):
                saved_market_inventory(root, selection)
            (root / 'raw_gru_c02_s29_t899.npz').unlink()
            saved, missing, complete = saved_market_inventory(root, selection)
            self.assertEqual(len(saved), 23)
            self.assertEqual(missing, ['raw_gru_c02_s29_t899'])
            self.assertEqual(complete, [])

    def test_first_refresh_audit_accepts_partial_run_only_with_all_six_pairs(self):
        from hsc_boundary import first_refresh_paths
        from hsc_core import MODES
        import os
        import tempfile
        from pathlib import Path
        import json
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as temp:
            root = Path(temp)
            (root / 'local_only' / 'market_fits').mkdir(parents=True)
            (root / 'INCOMPLETE.json').write_text(json.dumps({'complete': False}), encoding='utf-8')
            selection = {m: {'config_id': 'c02'} for m in MODES}
            with self.assertRaises(ValueError):
                first_refresh_paths(root, selection)
            for mode in MODES:
                path = root / 'local_only' / 'market_fits' / f'{mode}_c02_s17_t459.npz'
                np.savez(path, x=np.array([1.]))
                path.with_suffix('.json').write_text('{}', encoding='utf-8')
            self.assertEqual(len(first_refresh_paths(root, selection)), 6)


if __name__ == '__main__':
    unittest.main()
