import importlib
import unittest
import numpy as np
import torch

try:
    trainer = importlib.import_module('train_state')
except ModuleNotFoundError:
    trainer = None


class FitBoundaryTests(unittest.TestCase):
    def test_future_prices_do_not_change_scale_or_recurrent_fit(self):
        self.assertIsNotNone(trainer, 'Missing training implementation')
        p = 100 + np.cumsum(np.random.default_rng(77).normal(0, .1, 950))
        future = p.copy()
        future[459:] += 1000
        for mode in ['full_profile', 'full_prequential_mature']:
            first, scale_a, log_a = trainer.fit(p, 459, mode, 17, max_epochs=2)
            altered, scale_b, log_b = trainer.fit(future, 459, mode, 17, max_epochs=2)
            self.assertEqual(scale_a, scale_b)
            self.assertEqual(log_a['history'], log_b['history'])
            for key, value in first.state_dict().items():
                self.assertTrue(torch.equal(value, altered.state_dict()[key]), key)
            self.assertLess(log_a['train_max_target'], log_a['validation_first_origin'])
            self.assertLess(log_a['validation_max_target'], 459)


if __name__ == '__main__':
    unittest.main(verbosity=2)
