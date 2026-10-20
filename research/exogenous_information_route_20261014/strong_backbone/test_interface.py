import importlib.util
from pathlib import Path
import unittest
import numpy as np

class InterfaceTest(unittest.TestCase):
    def test_absent_extras_preserve_price(self):
        path=Path(__file__).with_name('backbone.py')
        self.assertTrue(path.exists(), 'strong backbone interface missing')
        spec=importlib.util.spec_from_file_location('adapter',path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        dates,p=m.read_prefix(); initial=int(np.sum(dates<'2020-01-01'))
        origins=np.arange(initial,initial+40)
        expected,_=m.base.predict(p,initial,origins,{'family':'MLE','window':480})
        got,receipt=m.forecast(p,initial,origins)
        np.testing.assert_array_equal(got,expected)
        missing={'coal_price':np.full(len(origins),np.nan),'available_at':np.full(len(origins),np.nan)}
        fallback,r=m.forecast(p,initial,origins,missing)
        np.testing.assert_array_equal(fallback,expected)
        self.assertEqual(r['route'],'price_only_fallback')

if __name__=='__main__':unittest.main()
