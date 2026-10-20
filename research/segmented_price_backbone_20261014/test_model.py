import unittest
import numpy as np
class Test(unittest.TestCase):
 def test_price_only(self):
  try: from model import initialize,predict
  except ImportError: self.fail('price-only backbone missing')
  x=np.random.RandomState(2).normal(size=(4,60)); p=initialize(7)
  for mode in ['segmented','unsegmented']:
   a=predict(p,x,mode); self.assertEqual(a.shape,(4,3))
   np.testing.assert_allclose(a,predict(p,x+10,mode),atol=1e-12)
if __name__=='__main__': unittest.main()
