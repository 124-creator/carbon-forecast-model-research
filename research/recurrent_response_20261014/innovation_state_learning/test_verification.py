"""Independent evaluation must derive its own raw-price targets."""
import importlib
import importlib.util
import unittest
import numpy as np


class VerificationTests(unittest.TestCase):
    def test_independent_metrics_reconstruct_observation_step_targets(self):
        self.assertIsNotNone(importlib.util.find_spec('verify_innovation'), 'independent scorer missing')
        module = importlib.import_module('verify_innovation')
        prices = 30 + np.cumsum(np.linspace(.1, .5, 80))
        dates = np.concatenate((np.arange(np.datetime64('2020-01-01'), np.datetime64('2020-02-10')),
                                np.arange(np.datetime64('2021-01-01'), np.datetime64('2021-02-10')))).astype('U10')
        origins = np.array([0, 1, 2, 3, 40, 41, 42, 43])
        exact = prices[origins[:, None] + np.arange(1, 21)] - prices[origins, None]
        metrics = module.independent_metrics(np.stack((np.zeros_like(exact), exact)), prices, dates, origins)
        self.assertAlmostEqual(metrics['score'], float(np.mean([1., 0.])))
        np.testing.assert_allclose(metrics['seeds'], [1., 0.])
        self.assertEqual(len(metrics['cells']), 6)


    def test_incomplete_run_can_never_be_accepted(self):
        module = importlib.import_module('verify_innovation')
        self.assertTrue(hasattr(module, 'partial_verdict'), 'incomplete-run verdict missing')
        winning = {'innovation_shift': {'score': .7}, 'gru_profile': {'score': 1.}}
        losing = {'innovation_shift': {'score': 1.}, 'gru_profile': {'score': .9}}
        for scores in (winning, losing):
            self.assertFalse(module.partial_verdict(scores)['acceptance_passed'])
        self.assertFalse(module.partial_verdict(winning)['decisive_necessary_gate_failure'])
        self.assertTrue(module.partial_verdict(losing)['decisive_necessary_gate_failure'])


if __name__ == '__main__':
    unittest.main()
