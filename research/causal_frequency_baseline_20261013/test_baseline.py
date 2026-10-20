import unittest, importlib.util
from pathlib import Path
import numpy as np
class TestSSA(unittest.TestCase):
 def test_reconstruction(self):
  path=Path(__file__).with_name('run.py')
  self.assertTrue(path.exists(),'SSA implementation missing')
  spec=importlib.util.spec_from_file_location('newrun',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  p=np.random.default_rng(7).normal(size=100).cumsum()
  raw,ssa=m.feature_at(p,80)
  np.testing.assert_allclose(ssa.reshape(3,59).sum(0),raw,atol=1e-10)
if __name__=='__main__':unittest.main()
