import unittest
from pathlib import Path
class Delivery(unittest.TestCase):
 def test_fixed_synthetic_delivery(self):
  p=Path(__file__).with_name('run.py')
  self.assertTrue(p.exists(),'selective baseline runner missing')
  import importlib.util
  s=importlib.util.spec_from_file_location('selective',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
  prices,_=m.b.generate(17,True)
  result,receipt=m.fit(prices)
  self.assertEqual(len(result),5)
  self.assertTrue(all(v.shape==(340,3,3) for v in result.values()))
  changed=prices.copy(); changed[1441:]+=10000
  result2,receipt2=m.fit(changed)
  self.assertEqual(receipt,receipt2)
  for k in result: self.assertTrue(m.np.array_equal(result[k][0],result2[k][0]))
if __name__=='__main__': unittest.main()
