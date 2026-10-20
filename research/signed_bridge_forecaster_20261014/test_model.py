import unittest
import torch
import model
class Tests(unittest.TestCase):
 def test_exact_bridge_inverse(self):
  self.assertTrue(hasattr(model,'bridge_matrix'),'bridge transform not implemented')
  for n in (4,8):
   e,d=model.bridge_matrix(n)
   x=torch.arange(n,dtype=torch.float64).reshape(1,n)-2
   self.assertTrue(torch.allclose((x@e.T)@d.T,x,atol=1e-10))
 def test_route_and_gradients(self):
  self.assertTrue(hasattr(model,'Forecaster'),'forecast mechanism missing')
  for kind in ['bridge','no_bridge','no_route','dst','haar','rawAvg','rawMax']:
   torch.manual_seed(17); m=model.Forecaster(kind).double()
   x=torch.randn(3,64,dtype=torch.float64,requires_grad=True)
   y=m(x); self.assertEqual(tuple(y.shape),(3,20)); y.square().mean().backward()
   self.assertTrue(torch.isfinite(x.grad).all())
   self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
  m=model.Forecaster('bridge').double()
  c=torch.randn(3,3,8,dtype=torch.float64); c[:,:,0]=0
  out=m.decode(c).cumsum(-1)
  self.assertTrue(torch.allclose(out[:,[7,15,19]],torch.zeros(3,3,dtype=torch.float64),atol=1e-10))
  z=torch.randn(3,64,dtype=torch.float64)
  self.assertTrue(torch.allclose(m.encode(z)@model.bridge_matrix(8)[1].T,z.reshape(3,8,8),atol=1e-10))
 def test_partial_block_and_endpoint_perturbation(self):
  m=model.Forecaster('bridge').double(); c=torch.randn(4,3,8,dtype=torch.float64)
  baseline=m.decode(c); altered=c.clone(); altered[:,2,4:]+=10000
  self.assertTrue(torch.equal(baseline,m.decode(altered)))
  detail=c.clone(); detail[:,:,0]=0
  self.assertTrue(torch.allclose(m.decode(detail).cumsum(-1)[:,[7,15,19]],torch.zeros(4,3,dtype=torch.float64),atol=1e-10))
  for b,n in enumerate([8,8,4]):
   segment=baseline[:,sum([8,8,4][:b]):sum([8,8,4][:b+1])]
   self.assertTrue(torch.allclose(segment.sum(-1),n*c[:,b,0],atol=1e-10))
  import itertools
  x=torch.tensor(list(itertools.product([-2.,0.,2.],repeat=4)),dtype=torch.float64)
  e,d=model.bridge_matrix(4)
  self.assertTrue(torch.allclose((x@e.T)@d.T,x,atol=1e-12))
if __name__=='__main__': unittest.main()
