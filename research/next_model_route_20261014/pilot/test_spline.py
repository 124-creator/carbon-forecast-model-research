import unittest, importlib.util
from pathlib import Path
import numpy as np
class Learnability(unittest.TestCase):
 def test_convex_spline_recovers_nonlinearity(self):
  path=Path(__file__).with_name('spline.py')
  self.assertTrue(path.exists(),'Spline estimator not implemented')
  spec=importlib.util.spec_from_file_location('spline',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  r=np.random.default_rng(91);x=r.normal(size=(1800,4));y=(x[:,0]**2)[:,None]+r.normal(size=(1800,1))*.05
  fit=m.fit(x[:480],y[:480],'additive',.01)
  pred=m.predict(fit,x[480:]);self.assertLess(np.mean((pred-y[480:])**2),.15)
if __name__=='__main__':unittest.main()
