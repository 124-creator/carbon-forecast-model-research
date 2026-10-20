import unittest
import importlib.util
from pathlib import Path
class TestCore(unittest.TestCase):
    def test_design_kernel_and_reconstruction(self):
        p=Path(__file__).with_name('core.py')
        self.assertTrue(p.exists(), 'design core not implemented')
        spec=importlib.util.spec_from_file_location('core',p); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        import torch
        model=m.Route('full'); d=torch.randn(3,59,dtype=torch.float64)
        k=model.kernel();self.assertEqual(float(k[0]),1);self.assertEqual(float(k[12]),0)
        self.assertTrue(bool((k[1:]<=k[:-1]).all()))
        f=m.features(d,4);pred,e,T,q=model(d,f,True)
        self.assertLess(float((T@e.unsqueeze(-1)-d.unsqueeze(-1)).abs().max()),1e-10)
        self.assertEqual(sum(x.numel() for x in model.parameters()),30)
if __name__=='__main__': unittest.main()
