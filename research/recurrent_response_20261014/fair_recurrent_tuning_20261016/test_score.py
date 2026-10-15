import unittest
import numpy as np
from scoring import score

class ScoreTest(unittest.TestCase):
    def test_seed_loss_is_not_prediction_ensemble(self):
        truth = np.ones((80, 20))
        pred = np.stack((truth + 1, truth - 1))
        years = np.array(['2020'] * 40 + ['2021'] * 40)
        result = score(pred, truth, years)
        self.assertEqual(result['score'], 1.)
        self.assertEqual(result['ensemble'], 0.)
        self.assertEqual(result['seeds'], [1., 1.])
        with self.assertRaises(ValueError):
            score(pred[:, :40], truth[:40], years[:40])

if __name__ == '__main__':
    unittest.main()
