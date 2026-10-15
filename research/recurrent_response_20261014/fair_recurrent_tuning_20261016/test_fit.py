import unittest
import numpy as np
from fit_recurrent import fit, predict

class FitTest(unittest.TestCase):
    def test_each_complete_fit_ignores_future_price(self):
        rng = np.random.default_rng(14)
        price = 20 + np.cumsum(rng.normal(0, .2, 280))
        altered = price.copy()
        altered[240:] += 500
        config = dict(id='synthetic_test_only', hidden=4, lr=.003, weight_decay=.001)
        for family in ('gru_dense', 'gru_profile', 'gru_rank6', 'lstm_dense'):
            first, scale, receipt = fit(price, 240, family, config, 17, max_epochs=2)
            second, other_scale, other_receipt = fit(altered, 240, family, config, 17, max_epochs=2)
            self.assertEqual(scale, other_scale)
            self.assertLess(receipt['train_max_target'], receipt['validation_first_origin'])
            self.assertLess(receipt['validation_max_target'], 240)
            self.assertEqual(receipt['epochs'], 2)
            for key, value in first.state_dict().items():
                np.testing.assert_array_equal(value.numpy(), second.state_dict()[key].numpy())
            origins = np.array([200, 220, 239])
            np.testing.assert_array_equal(predict(first, scale, price, origins), predict(second, scale, altered, origins))

if __name__ == '__main__':
    unittest.main()
