"""End-to-end chronological-fit contract, including all ordinary controls."""
import importlib
import importlib.util
import unittest
import numpy as np
import torch


class TrainingTests(unittest.TestCase):
    def test_fit_ignores_prices_after_exclusive_boundary(self):
        self.assertIsNotNone(importlib.util.find_spec('train_innovation'), 'new trainer missing')
        module = importlib.import_module('train_innovation')
        rng = np.random.default_rng(31)
        prices = 30 + np.cumsum(rng.normal(0, .2, 310))
        changed = prices.copy()
        changed[260:] += np.linspace(100, 200, len(prices) - 260)
        for mode in module.MODES:
            with self.subTest(mode=mode):
                first, scale, log = module.fit(prices, 260, mode, 17, max_epochs=2)
                second, scale2, other = module.fit(changed, 260, mode, 17, max_epochs=2)
                self.assertEqual(scale, scale2)
                self.assertEqual(log, other)
                self.assertLess(log['train_max_target'], log['validation_first_origin'])
                self.assertLess(log['validation_max_target'], 260)
                for key, value in first.state_dict().items():
                    torch.testing.assert_close(value, second.state_dict()[key], atol=0, rtol=0)
                pred = module.predict(first, scale, prices[:280], np.array([260, 279]), mode)
                self.assertEqual(pred.shape, (2, 20))
                self.assertTrue(np.isfinite(pred).all())


if __name__ == '__main__':
    unittest.main()
