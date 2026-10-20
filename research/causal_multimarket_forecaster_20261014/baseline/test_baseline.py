import unittest
import numpy as np
from pathlib import Path

class Contracts(unittest.TestCase):
    def test_maturity_and_future_invariance(self):
        import benchmark as b
        p,_=b.generate(17,True)
        splits=b.origins(len(p))
        self.assertLess(splits['train'][-1]+20,splits['val'][0])
        self.assertLess(splits['val'][-1]+20,splits['test'][0])
        mu,sd=b.scaler(p)
        t=int(splits['test'][0]); q=p.copy(); q[t+1:]+=10000
        np.testing.assert_array_equal(b.features(p,[t],mu,sd),b.features(q,[t],mu,sd))
        np.testing.assert_array_equal(b.scaler(p)[0],b.scaler(q)[0])
    def test_analytic_gradient(self):
        import benchmark as b
        self.assertLess(b.gradient_check(),1e-5)
    def test_pcgrad_conflict(self):
        import benchmark as b
        g=np.array([[1.,0.],[-1.,1.]])
        out=b.project(g,np.random.default_rng(1))
        self.assertGreaterEqual(float(out[0]@g[1]),-1e-12)

if __name__=='__main__': unittest.main()
