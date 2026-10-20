import unittest, importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
class JointTest(unittest.TestCase):
 def test_private_adapted_meta_gradient(self):
  self.assertTrue((ROOT/'run.py').exists(),'joint adaptation mechanism is missing')
  spec=importlib.util.spec_from_file_location('joint',ROOT/'run.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
  x,y=m.pilot.data(17); x=x[:,:12]; y=y[:,:12]; p=m.pilot.init(17)
  q,g=m.adapt_and_gradient(p,x,y); self.assertTrue(np.array_equal(q[m.mask],p[m.mask]))
  for j in np.flatnonzero(m.mask)[::13]:
   a=p.copy(); b=p.copy(); a[j]+=1e-5; b[j]-=1e-5
   def f(z):
    z=m.private_adapt(z,x,y); return m.pilot.objective(z,x,y,'shared_private')[0]
   self.assertAlmostEqual((f(a)-f(b))/2e-5,g[j],places=6)
if __name__=='__main__': unittest.main()
