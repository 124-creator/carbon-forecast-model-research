"""Golden scoring distinguishes mean seed losses from forecast ensembles."""
import importlib
import importlib.util
import unittest
import numpy as np


class ScoringTests(unittest.TestCase):
    def test_scores_average_seed_losses_not_predictions(self):
        self.assertIsNotNone(importlib.util.find_spec('experiment_io'), 'scoring interface missing')
        module = importlib.import_module('experiment_io')
        truth = np.arange(1, 161, dtype=float).reshape(8, 20) / 10
        prediction = np.stack((np.zeros_like(truth), truth))
        score = module.summarize(prediction, truth, np.array(['2020'] * 4 + ['2021'] * 4))
        self.assertAlmostEqual(score['score'], float(np.mean([1., 0.])))
        expected_ensemble = np.mean((truth / 2) ** 2) / np.mean(truth ** 2)
        self.assertAlmostEqual(score['ensemble'], float(expected_ensemble))
        self.assertEqual(len(score['cells']), 6)
        np.testing.assert_allclose(score['seeds'], [1, 0])


if __name__ == '__main__':
    unittest.main()
