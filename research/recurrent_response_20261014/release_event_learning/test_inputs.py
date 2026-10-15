import unittest
import numpy as np
from event_inputs import make_arrays


class InputTests(unittest.TestCase):
    def test_future_source_values_do_not_change_prefix_or_training_scaler(self):
        dates = np.arange(260) + 17000
        source_dates = dates[8::15]
        values = 450 + np.arange(len(source_dates))[:, None] * np.array([[2., 3., 4.]])
        price = 30 + np.sin(np.arange(260) / 6)
        a, stats = make_arrays(price, dates, source_dates, values, 220, 130, .2)
        changed = values.copy()
        changed[source_dates > dates[219]] += 10000
        b, other = make_arrays(price, dates, source_dates, changed, 220, 130, .2)
        self.assertEqual(stats, other)
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
        self.assertEqual(len(a['delta']), 219)
        self.assertTrue(np.all(a['source_index'] >= 0))
        self.assertTrue(np.any(a['clock'][:, 0] == 0))


if __name__ == '__main__':
    unittest.main()
