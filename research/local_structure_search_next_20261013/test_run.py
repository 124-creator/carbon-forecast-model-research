import unittest
import numpy as np
try:
    from run import mature_fit,load
except ImportError:
    mature_fit=load=None
class RunTest(unittest.TestCase):
    def test_future_invariance(self):
        self.assertIsNotNone(mature_fit,'causal trainer missing')
        rng=np.random.default_rng(4);p=np.cumsum(rng.normal(size=200));X=rng.normal(size=(200,6))
        a,_=mature_fit(X,p,100,80,'corrected');p[101:]+=10000
        b,_=mature_fit(X,p,100,80,'corrected');np.testing.assert_array_equal(a,b)
if __name__=='__main__': unittest.main()
