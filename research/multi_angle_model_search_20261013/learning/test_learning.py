import unittest,importlib.util
from pathlib import Path
import numpy as np
class Tests(unittest.TestCase):
 def test_df_cap_and_ridge_equivalence(self):
  path=Path(__file__).with_name('run.py')
  self.assertTrue(path.exists(),'Missing executable diagnostic')
  spec=importlib.util.spec_from_file_location('learning',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  rng=np.random.default_rng(19);A=rng.normal(size=(120,44));A-=A.mean(0);Y=rng.normal(size=(120,3))
  B,lam,df=m.ridge(A,Y,3.)
  self.assertLessEqual(df,3.00000001)
  ref=np.linalg.solve(A.T@A+lam*np.eye(44),A.T@Y)
  self.assertLess(np.max(np.abs(B-ref)),1e-10)
  self.assertLessEqual(m.ess(np.cumsum(rng.normal(size=120))),120)
if __name__=='__main__':unittest.main()
