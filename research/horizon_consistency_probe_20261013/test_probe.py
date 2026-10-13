import unittest
import numpy as np
try:
 from probe import solve_head
except ImportError:
 solve_head=None
class TestProbe(unittest.TestCase):
 def test_consistency_is_augmented_fixed_kernel(self):
  self.assertIsNotNone(solve_head,'缺少真实训练实现')
  r=np.random.default_rng(2); A=r.normal(size=(12,9)); C=r.normal(size=(7,9)); y=r.normal(size=12); d=r.normal(size=7)
  w,dual=solve_head(A,y,C,d,.1,.3,True)
  self.assertLess(float(np.max(abs(w-dual))),1e-10)
if __name__=='__main__':unittest.main()
