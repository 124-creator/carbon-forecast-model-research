import unittest
import numpy as np
import torch
from train_events import fit, predict


class TrainingTests(unittest.TestCase):
    def test_full_fit_ignores_both_future_price_and_future_released_vectors(self):
        dates = np.arange(280) + 17000
        sd = dates[8::15]
        sv = 450 + np.arange(len(sd))[:, None] * np.array([[2., 3., 4.]])
        data = dict(price=30 + np.sin(np.arange(280) / 6), dates=dates, source_dates=sd, source_values=sv)
        model, stats, log = fit(data, 240, 'event', 17, max_epochs=2)
        changed = {k: v.copy() for k, v in data.items()}
        changed['price'][240:] += 900
        changed['source_values'][sd > dates[239]] += 10000
        other, other_stats, other_log = fit(changed, 240, 'event', 17, max_epochs=2)
        self.assertEqual(stats, other_stats)
        self.assertEqual(log['history'], other_log['history'])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, other.state_dict()[name], rtol=0, atol=0)
        np.testing.assert_array_equal(predict(model, stats, data, [239]), predict(other, other_stats, changed, [239]))
        self.assertLess(log['train_max_target'], log['validation_first_origin'])


if __name__ == '__main__':
    unittest.main()
