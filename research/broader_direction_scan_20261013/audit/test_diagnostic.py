import unittest
import numpy as np
from diagnostic import support, ridge_predict

class TestDiagnostic(unittest.TestCase):
    def test_maturity_and_linear_solution(self):
        self.assertEqual(support(100,1,40,'per_h').tolist(),list(range(60,99)))
        self.assertEqual(support(100,20,40,'common').tolist(),list(range(60,80)))
        p=np.arange(180,dtype=float)
        x=np.zeros((180,3))
        for mode in ['common','per_h']:
            pred,trace=ridge_predict(p,x,100,np.array([100,101]),mode,40,1.)
            np.testing.assert_allclose(pred,np.tile([1.,5.,20.],(2,1)),atol=1e-10)
            for row in trace:
                self.assertLess(row['max_label_endpoint'],100)
                self.assertLess(row['normal_equation_relative_residual'],1e-8)
        q=p.copy();q[102:]=1e9
        np.testing.assert_array_equal(ridge_predict(p,x,100,np.array([100,101]),'per_h',40,1.)[0],ridge_predict(q,x,100,np.array([100,101]),'per_h',40,1.)[0])

if __name__=='__main__':unittest.main()
