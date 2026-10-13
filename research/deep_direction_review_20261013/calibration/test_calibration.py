import unittest
import importlib.util
from pathlib import Path
import numpy as np

class CalibrationTests(unittest.TestCase):
    def test_end_to_end_causal_contract(self):
        path=Path(__file__).with_name('calibrate.py')
        self.assertTrue(path.exists(), 'Missing calibration implementation')
        spec=importlib.util.spec_from_file_location('calibrate',path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        p=100+np.cumsum(np.random.default_rng(71).normal(size=300))
        origins=np.arange(240,260)
        for config in m.configurations():
            a,_=m.predict(p,240,origins,config)
            changed=p.copy();changed[260:]+=1000
            b,_=m.predict(changed,240,origins,config)
            np.testing.assert_allclose(a,b,rtol=0,atol=0)
            self.assertEqual(a.shape,(20,3))
        truth=np.ones((4,3)); pred=np.zeros_like(truth)
        score,rows=m.score(pred,truth,np.array(['2020','2020','2021','2021']))
        self.assertEqual(score,1.)
        self.assertEqual(len(rows),2)

if __name__=='__main__':unittest.main()
