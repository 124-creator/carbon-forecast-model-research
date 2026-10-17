"""Tests for bounded residual-utility memory; no historical checkpoints."""
import unittest
import numpy as np


class MemoryTests(unittest.TestCase):
    def test_memory_candidates_are_mature_and_ignore_unavailable_suffix(self):
        from rum_core import candidate_cache
        price = np.cumsum(np.random.default_rng(17).normal(size=300)) + 50
        ids = np.arange(60, 220)
        cache = candidate_cache(price, ids, lookback=60, horizon=20, k=8)
        keys = ids[cache['index']]
        self.assertTrue(np.all((keys + 20 <= ids[:, None] - 60)[cache['valid']]))
        changed = price.copy()
        changed[220:] += np.arange(80) * 500
        other = candidate_cache(changed, ids, lookback=60, horizon=20, k=8)
        for key in ('index', 'valid', 'similarity', 'age'):
            np.testing.assert_array_equal(cache[key], other[key])
        self.assertFalse(cache['valid'][:80].any())
        self.assertTrue(cache['valid'][80:].any())

    def test_utility_teacher_rewards_helpful_keys_and_has_detached_null(self):
        import torch
        from rum_core import utility_teacher
        error = torch.tensor([[1., -1.]], requires_grad=True)
        memory = torch.tensor([[[4., -4.], [-4., 4.]]], requires_grad=True)
        valid = torch.tensor([[True, True]])
        teacher = utility_teacher(error, memory, valid, torch.ones(2))
        self.assertFalse(teacher.requires_grad)
        self.assertTrue(torch.all(teacher[:, 1] > teacher[:, 0]))
        self.assertTrue(torch.all(teacher[:, 0] > teacher[:, 2]))
        torch.testing.assert_close(teacher.sum(1), torch.ones_like(error))
        empty = utility_teacher(error, memory, ~valid, torch.ones(2))
        torch.testing.assert_close(empty[:, 0], torch.ones_like(error))
        self.assertEqual(int(torch.count_nonzero(empty[:, 1:])), 0)

    def test_fresh_utility_fit_is_unchanged_by_unavailable_suffix(self):
        import torch
        from rum_fit import fit, predict
        price = np.cumsum(np.random.default_rng(29).normal(size=310)) + 60
        other = price.copy()
        other[281:] += np.arange(29) * 80
        first, scale, receipt = fit(price, 281, 'utility', 17, max_epochs=2)
        second, other_scale, other_receipt = fit(other, 281, 'utility', 17, max_epochs=2)
        self.assertEqual(scale, other_scale)
        self.assertEqual(receipt['state_sha256'], other_receipt['state_sha256'])
        self.assertEqual(receipt['epochs'], 2)
        self.assertLess(receipt['train_max_target'], receipt['validation_first_origin'])
        a, diagnostics = predict(first, scale, price, np.arange(277, 281))
        b, _ = predict(second, other_scale, other, np.arange(277, 281))
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isfinite(a).all())
        self.assertLessEqual(diagnostics['maximum_memory_weight'], 1.)
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)

    def test_gru_control_matches_actual_fresh_common_baseline(self):
        from rum_fit import fit, predict
        from fit_recurrent import fit as common_fit, predict as common_predict
        price = np.cumsum(np.random.default_rng(31).normal(size=310)) + 60
        model, scale, _ = fit(price, 281, 'gru', 17, max_epochs=2)
        common, common_scale, _ = common_fit(price, 281, 'gru_dense',
                                            dict(hidden=16, lr=.003, weight_decay=.001), 17, max_epochs=2)
        self.assertEqual(scale, common_scale)
        actual, _ = predict(model, scale, price, np.arange(281, 286))
        expected = common_predict(common, common_scale, price, np.arange(281, 286))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)

    def test_independent_score_reproduces_rw_and_rejects_nonfinite(self):
        from rum_verify import score_arrays
        price = np.arange(200, dtype=float) + 50
        dates = np.array(['2020-01-01'] * 100 + ['2021-01-01'] * 100)
        origins = np.r_[np.arange(20, 80), np.arange(120, 180)]
        forecasts = np.zeros((2, len(origins), 20))
        result = score_arrays(price, dates, origins, forecasts)
        self.assertEqual(result['score'], 1.)
        self.assertEqual(result['counts'], {'2020': 60, '2021': 60})
        forecasts[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            score_arrays(price, dates, origins, forecasts)

    def test_frozen_plan_and_saved_inventory_reject_partial_pairs(self):
        import os
        import tempfile
        from pathlib import Path
        from rum_run import planned_tasks, saved_stems
        tasks = planned_tasks()
        self.assertEqual(len(tasks), 32)
        self.assertEqual(len(set(tasks)), 32)
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            root = Path(tmp)
            self.assertEqual(saved_stems(root), set())
            (root / 'gru_s17_t459.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                saved_stems(root)
            (root / 'gru_s17_t459.npz').write_bytes(b'inventory only')
            self.assertEqual(saved_stems(root), {'gru_s17_t459'})


if __name__ == '__main__':
    unittest.main()
