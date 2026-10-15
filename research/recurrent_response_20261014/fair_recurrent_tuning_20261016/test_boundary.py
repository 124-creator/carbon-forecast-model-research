import unittest
import numpy as np
from boundary_audit import audit_pair

class BoundaryTest(unittest.TestCase):
    def test_pair_repeats_training_not_just_forecasts(self):
        rng = np.random.default_rng(113)
        synthetic = 20 + np.cumsum(rng.normal(.01, .15, 280))
        config = dict(id='unit_fixture', hidden=4, lr=.003, weight_decay=.001)
        receipt = audit_pair(synthetic, 240, 'gru_dense', config, 17, 2)
        self.assertEqual(receipt['actual_fits'], 2)
        self.assertEqual(receipt['original_epochs'], 2)
        self.assertEqual(receipt['altered_epochs'], 2)
        self.assertEqual(receipt['parameter_max_difference'], 0.)
        self.assertEqual(receipt['unchanged_prefix_prediction_max_difference'], 0.)
        self.assertFalse(receipt['independent_implementation'])

if __name__ == '__main__':
    unittest.main()
