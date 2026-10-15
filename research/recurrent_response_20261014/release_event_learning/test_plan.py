import unittest
import numpy as np
from run_events import plan


class PlanTests(unittest.TestCase):
    def test_frozen_plan_uses_both_years_and_never_targets_the_next_year(self):
        dates = np.arange(np.datetime64('2018-01-01'), np.datetime64('2022-01-01'), dtype='datetime64[D]').astype('int64')
        origins, blocks, jobs = plan(dates)
        calendar = dates.astype('datetime64[D]').astype(str)
        years = np.array([s[:4] for s in calendar])
        self.assertTrue(np.all(years[origins] == years[origins + 20]))
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len({(m, s) for m, s in jobs}), 12)
        np.testing.assert_array_equal(np.concatenate([b['origins'] for b in blocks]), origins)
        self.assertTrue(all(b['end'] <= b['origins'][0] for b in blocks))


if __name__ == '__main__':
    unittest.main()
