import importlib
import unittest
import numpy as np

try:
    runner = importlib.import_module('run_state')
except ModuleNotFoundError:
    runner = None


class ScoreTests(unittest.TestCase):
    def test_zero_change_forecast_has_exact_rw_score(self):
        self.assertIsNotNone(runner, 'Missing experiment runner')
        truth = np.random.default_rng(11).normal(size=(80, 20))
        years = np.array(['2020'] * 40 + ['2021'] * 40)
        actual = runner.summarize(np.zeros((2, 80, 20)), truth, years)
        self.assertEqual(actual['score'], 1.)
        self.assertEqual(actual['ensemble'], 1.)
        self.assertEqual(actual['seeds'], [1., 1.])


if __name__ == '__main__':
    unittest.main(verbosity=2)
