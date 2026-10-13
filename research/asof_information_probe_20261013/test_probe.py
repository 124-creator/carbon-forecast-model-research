import unittest
import numpy as np
try:
    from probe import delay_filter
except ImportError:
    delay_filter=None
class Tests(unittest.TestCase):
    def test_future_releases_cannot_change_current_state(self):
        self.assertIsNotNone(delay_filter,'delayed-observation filter not implemented')
        events=[(2,0,3.,.1),(8,4,100.,.1)]
        a=delay_filter(np.zeros(10),events,.9,.2,0.,1.)
        b=delay_filter(np.zeros(10),[(2,0,3.,.1),(8,4,-100.,.1)],.9,.2,0.,1.)
        np.testing.assert_array_equal(a[:8],b[:8])
        self.assertAlmostEqual(a[3],.9*a[2],places=12)
        self.assertGreater(abs(a[2]),0.)
    def test_hold_age_has_observable_age_and_variance(self):
        from probe import features
        x=features(np.zeros(10),[(2,0,3.,.1)],0.,'hold_age',.9)
        self.assertEqual(x.shape,(10,8))
        np.testing.assert_array_equal(x[3,-3:], [3.,3.,.1])
if __name__=='__main__':unittest.main()
