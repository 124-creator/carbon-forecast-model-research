import unittest, importlib.util
from pathlib import Path
import numpy as np

class TestCausal(unittest.TestCase):
    def test_vintage_and_maturity(self):
        path=Path(__file__).with_name('run.py')
        self.assertTrue(path.exists(), 'causal implementation missing')
        spec=importlib.util.spec_from_file_location('pilot',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        p=np.arange(150,dtype=float)+np.sin(np.arange(150))
        X,rev=m.features(p,5)
        for t in [40,70,100]:
            q=p.copy();q[t+1:]+=10000
            Y,_=m.features(q,5)
            np.testing.assert_allclose(X[:t+1],Y[:t+1],atol=1e-10)
            Z,_=m.features(p[:t+1],5)
            np.testing.assert_allclose(X[t],Z[t],atol=1e-10)
            s=t-6;expected=rev[s-19:s+1]
            np.testing.assert_allclose(X[t,-2:],[expected.mean(),np.sqrt(np.mean(expected**2))])
        np.testing.assert_allclose(X[70,:8]+X[70,8:16],np.diff(p[62:71])[::-1],atol=1e-10)

if __name__=='__main__':unittest.main()
