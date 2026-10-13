import unittest
import numpy as np
try:
    from prototype import fit
except ImportError:
    fit=None

class CoreTest(unittest.TestCase):
    def test_noiseless_rank_one(self):
        self.assertIsNotNone(fit,'candidate implementation missing')
        rng=np.random.default_rng(3)
        X=rng.normal(size=(100,5)); B=np.outer(rng.normal(size=5),[1,2,3]);Y=X@B
        b,info=fit(X,Y,'corrected',0)
        np.testing.assert_allclose(X@b,Y,atol=1e-9)
        self.assertLess(info['objective'],1e-16)

if __name__=='__main__':unittest.main()
