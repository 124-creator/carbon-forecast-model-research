import unittest, importlib.util
from pathlib import Path
import numpy as np
class Tests(unittest.TestCase):
 def test_causal_prediction_and_maturity(self):
  path=Path(__file__).with_name('run.py')
  self.assertTrue(path.exists(),'state implementation missing')
  spec=importlib.util.spec_from_file_location('state_run',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  p=30+np.cumsum(np.random.default_rng(5).normal(size=600))
  bank,features=m.history(p,300)
  q=p.copy();q[351:]+=1000
  b2,f2=m.history(q,300)
  np.testing.assert_allclose(bank[:351],b2[:351]);np.testing.assert_allclose(features[:351],f2[:351])
  c=dict(family='candidate',alpha=1.,seed=None)
  a,logs=m.correct(p,bank,features,340,np.array([340,350]),c)
  b,_=m.correct(q,b2,f2,340,np.array([340,350]),c)
  np.testing.assert_allclose(a,b)
  for log in logs:self.assertLess(log['max_target'],340)
  short,fs=m.history(p[:351],300)
  np.testing.assert_allclose(bank[:351],short);np.testing.assert_allclose(features[:351],fs)
if __name__=='__main__':unittest.main()
