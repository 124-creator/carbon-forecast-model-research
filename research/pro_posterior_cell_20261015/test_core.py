"""Algebraic/implementation fixtures, not synthetic carbon performance scores."""
import unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
import core

class Checks(unittest.TestCase):
 def setUp(self):
  core.configure();torch.manual_seed(8);self.rng=np.random.default_rng(8)
 def test_integral(self):
  nodes,weights=hermgauss(80)
  for m,v,a,b in [(.3,.2,1.1,.7),(-.5,2.,-.8,.9),(0.,0.,2.,1.),(1.,.8,2.2,-.2)]:
   numeric=np.sum(weights*np.sin(b+a*(m+np.sqrt(2*v)*nodes)))/np.sqrt(np.pi)
   exact=np.exp(-a*a*v/2)*np.sin(b+a*m)
   self.assertAlmostEqual(numeric,exact,places=13)
 def test_gradient(self):
  a=torch.tensor(.7,requires_grad=True);m=.2;v=.4;b=.8
  f=lambda aa:torch.exp(-aa*aa*v/2)*torch.sin(b+aa*m)
  g=torch.autograd.grad(f(a),a)[0].item();eps=1e-6
  fd=(f(a.detach()+eps)-f(a.detach()-eps)).item()/(2*eps)
  self.assertAlmostEqual(g,fd,places=9)
 def args(self,n=3,l=12):
  return torch.randn(n,l),torch.randn(n,l),torch.rand(n,l),torch.randn(n,3),torch.randn(n,20)
 def test_point_zero_var(self):
  x,m,v,q,b=self.args(); v.zero_()
  a=core.make_model('INTEGRATED',17);p=core.make_model('POINT',17)
  torch.testing.assert_close(a.encode(x,m,v),p.encode(x,m,v),rtol=0,atol=0)
 def test_tied_free_equivalence(self):
  x,m,v,_,_=self.args();a=core.make_model('INTEGRATED',17);f=core.make_model('FREE_DAMP',17)
  torch.testing.assert_close(a.encode(x,m,v),f.encode(x,m,v),rtol=0,atol=0)
 def test_variance_changes_mean(self):
  x,m,v,_,_=self.args();a=core.make_model('INTEGRATED',17)
  self.assertGreater(float((a.encode(x,m,v)-a.encode(x,m,v*0)).abs().max()),1e-5)
 def test_bounded_hidden(self):
  x,m,v,_,_=self.args(5,120);a=core.make_model('INTEGRATED',17)
  self.assertLessEqual(float(a.encode(x,m,v).abs().max()),1.)
 def test_zero_head_reference(self):
  args=self.args();a=core.make_model('INTEGRATED',17);p,c=a(*args)
  torch.testing.assert_close(p,args[-1],rtol=0,atol=0);self.assertEqual(float(c.abs().max()),0.)
 def test_constant_case(self):
  x,m,v,_,_=self.args();v.fill_(.4)
  a=core.make_model('INTEGRATED',17);b=core.make_model('CONSTANT',17,.4)
  torch.testing.assert_close(a.encode(x,m,v),b.encode(x,m,v),rtol=0,atol=0)
 def test_common_initialization(self):
  models=[core.make_model(k,17) for k in core.MODES[:4]]
  for name,val in models[0].named_parameters():
   for mod in models[1:]:torch.testing.assert_close(val,dict(mod.named_parameters())[name],rtol=0,atol=0)
 def test_counts(self):
  counts=[sum(v.numel() for v in core.make_model(k,17).parameters()) for k in core.MODES]
  self.assertEqual(counts,[1268,1268,1284,1268,1232,1252])
 def test_full_projection(self):
  a=core.make_model('FREE_DAMP',17)
  with torch.no_grad():a.a.fill_(8);a.damp.fill_(-8)
  a.project();self.assertEqual(float(a.a.max()),3.);self.assertEqual(float(a.damp.min()),-3.)
 def test_posterior_prefix(self):
  p=np.cumsum(self.rng.normal(size=180));q=p.copy();q[130:]+=10000
  a=core.reference(p);b=core.reference(q)
  for k in a:np.testing.assert_array_equal(a[k][:130],b[k][:130])
 def test_scale_translation(self):
  p=np.cumsum(self.rng.normal(size=200));a=core.reference(p);b=core.reference(p*3+20)
  np.testing.assert_array_equal(a['selected'],b['selected'])
  np.testing.assert_allclose(3*a['mu'],b['mu'],atol=1e-12)
  np.testing.assert_allclose(9*a['var'],b['var'],atol=1e-12)
 def test_variance_finite(self):
  p=np.cumsum(self.rng.normal(size=950));r=core.reference(p)
  self.assertTrue(np.isfinite(r['var']).all());self.assertTrue(np.all(r['var']>=0))
 def test_flat_reference(self):
  r=core.reference(np.ones(200)*15)
  np.testing.assert_array_equal(r['b'],np.zeros((200,20)))
 def test_support_maturity(self):
  p=np.cumsum(self.rng.normal(size=950))
  for end in [459,499,899]:
   ids,x,y,tr,va,sd=core.support(p,end)
   self.assertLess(ids[tr[-1]]+20,ids[va[0]]);self.assertLess(ids[-1]+20,end)
 def test_training_only_scale(self):
  p=np.cumsum(self.rng.normal(size=950));ids,x,y,tr,va,sd=core.support(p,459)
  q=p.copy();q[ids[va[0]]:]+=5000
  self.assertEqual(sd,core.support(q,459)[-1])
 def test_mean_error_price_error(self):
  p=self.rng.normal(size=10);d=self.rng.normal(size=(10,20));y=self.rng.normal(size=(10,20))
  np.testing.assert_allclose((d-y)**2,((d+p[:,None])-(y+p[:,None]))**2,atol=1e-14)
 def test_not_exact_full_marginalization(self):
  # The variance propagation of a complete random trajectory is not represented.
  nodes,w=hermgauss(80);ez=np.sum(w*np.sin(.7+np.sqrt(.8)*nodes))/np.sqrt(np.pi)
  nested=np.sum(w*np.sin(np.sin(.7+np.sqrt(.8)*nodes)))/np.sqrt(np.pi)
  self.assertGreater(abs(np.sin(ez)-nested),1e-4)
 def test_counterexample_mean_not_unique(self):
  self.assertNotEqual(np.exp(-.1/2)*np.sin(.6),np.exp(-2/2)*np.sin(.6))

if __name__=='__main__':unittest.main(verbosity=2)
