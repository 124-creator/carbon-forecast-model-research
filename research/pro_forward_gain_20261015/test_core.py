"""Own algebra and causality fixtures; not forecasting-performance claims."""
import unittest
import numpy as np
import torch
import core

class Tests(unittest.TestCase):
 def setUp(self):
  core.configure();torch.manual_seed(7);self.rng=np.random.default_rng(7)
 def fixture(self):
  return torch.randn(120,9),torch.randn(120,20)
 def test_ridge_stationarity(self):
  z,r=self.fixture();d=core.ridge(z,r).detach().requires_grad_()
  loss=((r-z@d).square().sum()+(z@d).square().sum())/len(z)+core.RIDGE*d.square().sum()
  self.assertLess(float(torch.autograd.grad(loss,d)[0].abs().max()),1e-12)
 def test_ridge_unique(self):
  z,r=self.fixture();matrix=2*z.T@z/len(z)+core.RIDGE*torch.eye(9)
  self.assertGreaterEqual(float(torch.linalg.eigvalsh(matrix)[0]),core.RIDGE-1e-12)
 def test_ridge_derivative(self):
  z,r=self.fixture();t=torch.tensor(.8,requires_grad=True)
  fn=lambda v:core.ridge(z*v,r).square().sum()
  g=float(torch.autograd.grad(fn(t),t)[0]);e=1e-6
  fd=float((fn(t.detach()+e)-fn(t.detach()-e))/(2*e));self.assertAlmostEqual(g,fd,places=8)
 def test_trace_identity(self):
  z,r=self.fixture();a=np.arange(60);b=np.arange(80,120);s,d=core.gain_matrix(z,r,a,b);p=core.spectral(s)[0];u=z[b]@d
  direct=((r[b]-u@p).square().sum()+(u@p).square().sum())/len(b)+core.RIDGE*(d@p).square().sum()
  closed=r[b].square().sum()/len(b)-torch.trace(p@s)
  self.assertAlmostEqual(float(direct),float(closed),places=12)
 def test_projector(self):
  a=torch.randn(20,20);p,vals,take=core.spectral((a+a.T)/2)
  torch.testing.assert_close(p@p,p,rtol=0,atol=1e-13);torch.testing.assert_close(p.T,p,rtol=0,atol=1e-14)
  self.assertLessEqual(int(take.sum()),6)
 def test_optimal_rank(self):
  a=torch.randn(20,20);s=(a+a.T)/2;p,_,_=core.spectral(s);best=float(torch.trace(p@s))
  for rank in range(1,7):
   for _ in range(10):
    q=torch.linalg.qr(torch.randn(20,rank)).Q
    self.assertLessEqual(float(torch.trace(q.T@s@q)),best+1e-12)
 def test_negative_gain(self):
  p,v,t=core.spectral(-torch.eye(20));self.assertEqual(float(p.abs().max()),0.);self.assertEqual(int(t.sum()),0)
 def test_zero_gain(self):
  p,v,t=core.spectral(torch.zeros(20,20));self.assertEqual(float(p.abs().max()),0.)
 def test_stable_reversal(self):
  r=torch.eye(20);u=r*.5;u[0]*=-1
  s=r.T@u+u.T@r-2*u.T@u;p,_,_=core.spectral(s)
  self.assertEqual(float(p[0,0]),0.)
  self.assertGreater(float(torch.trace(p)),0.)
 def test_eigen_objective_derivative(self):
  a=torch.randn(20,20);s=(a+a.T)/2;t=torch.tensor(.3,requires_grad=True);b=torch.diag(torch.arange(20)/20)
  fn=lambda v:torch.linalg.eigvalsh(s+v*b)[-6:].clamp(min=0).sum()
  g=float(torch.autograd.grad(fn(t),t)[0]);e=1e-6;fd=float((fn(t.detach()+e)-fn(t.detach()-e))/(2*e))
  self.assertAlmostEqual(g,fd,places=7)
 def test_fixed_subspace(self):
  p=core.fixed_projector();self.assertAlmostEqual(float(torch.trace(p)),6.,places=12)
  torch.testing.assert_close(p@p,p,rtol=0,atol=1e-14)
 def test_fixed_transform_equivalence(self):
  z,r=self.fixture();q=torch.linalg.qr(core.initial_basis(6)).Q
  torch.testing.assert_close(core.ridge(z,r)@q,core.ridge(z,r@q),rtol=1e-12,atol=1e-12)
 def test_rotation_invariance(self):
  a=torch.randn(20,20);s=(a+a.T)/2;o=torch.linalg.qr(torch.randn(20,20)).Q
  p,_,_=core.spectral(s);po,_,_=core.spectral(o.T@s@o)
  torch.testing.assert_close(po,o.T@p@o,rtol=1e-11,atol=1e-11)
 def test_in_sample_is_psd(self):
  z,r=self.fixture();ids=np.arange(len(z));s,d=core.gain_matrix(z,r,ids,ids)
  self.assertGreater(float(torch.linalg.eigvalsh(s)[0]),-1e-12)
 def test_population_penalty_shrink(self):
  target=torch.tensor(4.);c=torch.tensor(2.,requires_grad=True);l=(target-c)**2+c*c
  self.assertEqual(float(torch.autograd.grad(l,c)[0]),0.)
 def test_common_encoder_init(self):
  models=[core.make_model(k,17) for k in core.PROFILE_MODES]
  for k,v in models[0].named_parameters():
   for m in models[1:]:torch.testing.assert_close(v,dict(m.named_parameters())[k],rtol=0,atol=0)
 def test_maturity(self):
  p=self.rng.normal(size=950).cumsum()
  for end in (459,699,899):
   ids,x,y,tr,va,sd=core.support(p,end)
   self.assertLess(ids[tr[-1]]+20,ids[va[0]]);self.assertLess(ids[-1]+20,end)
   for a,b in core.folds(len(tr)):self.assertLess(ids[tr[a[-1]]]+20,ids[tr[b[0]]])
 def test_scale_only_earliest_support(self):
  p=self.rng.normal(size=950).cumsum();ids,x,y,tr,va,sd=core.support(p,459)
  a,b=core.folds(len(tr))[0];q=p.copy();q[ids[tr[b[0]]]:]+=1000
  self.assertEqual(sd,core.support(q,459)[-1])
 def test_reference_causality(self):
  p=self.rng.normal(size=250).cumsum();q=p.copy();q[180:]+=1000
  a=core.reference(p);b=core.reference(q)
  for k in a:np.testing.assert_array_equal(a[k][:180],b[k][:180])
 def test_price_error(self):
  p=torch.randn(4,1);d=torch.randn(4,20);y=torch.randn(4,20)
  torch.testing.assert_close((d-y).square(),((p+d)-(p+y)).square(),rtol=1e-12,atol=1e-12)

if __name__=='__main__':unittest.main(verbosity=2)
