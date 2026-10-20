import unittest, importlib.util
from pathlib import Path
class Contract(unittest.TestCase):
 def test_complete_labels_and_purged_validation(self):
  spec=importlib.util.find_spec('run')
  self.assertIsNotNone(spec,'training implementation missing')
  import run
  import numpy as np
  d=np.array(['2019-01-01']*459); p=np.arange(459,dtype=float)
  pack=run.make_pack(p)
  self.assertEqual(pack['complete'],379)
  self.assertLess(int(pack['train_origins'][-1])+20,int(pack['val_origins'][0]))
  self.assertEqual(pack['x'].shape[1],60)
if __name__=='__main__': unittest.main()
