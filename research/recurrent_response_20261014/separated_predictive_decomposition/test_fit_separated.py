import unittest
import numpy as np
import torch


class SeparatedFitTests(unittest.TestCase):
    def test_fit_and_scaler_ignore_prices_after_the_maturity_cutoff(self):
        from run_separated import fit
        prices = np.cumsum(np.random.default_rng(3).normal(size=600))
        model, scale, receipt = fit(prices, 459, 'coupled', 17, max_epochs=1)
        changed = prices.copy()
        changed[459:] += 10000
        other, other_scale, _ = fit(changed, 459, 'coupled', 17, max_epochs=1)
        self.assertEqual(scale, other_scale)
        self.assertLess(receipt['max_training_target'], receipt['validation_first'])
        self.assertLess(receipt['max_validation_target'], 459)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, other.state_dict()[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
