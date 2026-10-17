"""Pre-fit algebra/causality checks, not market forecast performance tests."""
import unittest, numpy as np, torch
import core

class Tests(unittest.TestCase):
 def setUp(self):core.configure();self.rng=np.random.default_rng(701)
 def test_overlap_psd(self):
  a=self.rng.integers(1,180,12)
  for h in [1,5,20]:
   c=core.overlap_matrix(a,h);self.assertGreater(np.linalg.eigvalsh(c).min(),-1e-12)
   incidence=np.array([[float(j<u<=j+h) for u in range(1,220)] for j in a])/np.sqrt(h)
   np.testing.assert_allclose(c,incidence@incidence.T,atol=1e-14)
 def test_h1_identity(self):np.testing.assert_array_equal(core.overlap_matrix(np.array([1,2,5,8]),1),np.eye(4))
 def test_nonoverlap(self):np.testing.assert_array_equal(core.overlap_matrix(np.array([1,21,60]),20),np.eye(3))
 def test_solve_stationarity(self):
  pi=torch.softmax(torch.tensor([[.2,1.,-.5]]),-1);c=torch.tensor(core.overlap_matrix(np.array([[1,3,50]]),20))[:,None]
  mask=torch.ones_like(pi,dtype=torch.bool);w=core.gls_weights(pi,c,mask)
  self.assertAlmostEqual(w.sum().item(),1.,places=13)
  m=c+torch.diag_embed(1/pi)[:,None];g=(m@w[...,None])[0,0,:,0]
  self.assertLess((g-g[0]).abs().max().item(),1e-12)
 def test_solve_gradient(self):
  x=torch.tensor([[.4,-.2,.8]],requires_grad=True);c=torch.tensor(core.overlap_matrix(np.array([[0,1,50]]),20))[:,None]
  f=lambda z:(core.gls_weights(torch.softmax(z,-1),c,torch.ones_like(z,dtype=torch.bool))*torch.tensor([1.,2.,-1.])).sum()
  g=torch.autograd.grad(f(x),x)[0];eps=1e-6
  for i in range(3):
   a=x.detach().clone();b=a.clone();a[0,i]+=eps;b[0,i]-=eps
   self.assertAlmostEqual(g[0,i].item(),((f(a)-f(b))/(2*eps)).item(),places=8)
 def test_uniform(self):
  pi=torch.ones(2,4)/4;c=torch.eye(4)[None,None].expand(2,20,-1,-1)
  np.testing.assert_allclose(core.gls_weights(pi,c,torch.ones_like(pi,dtype=torch.bool)).numpy(),.25,atol=1e-14)
 def test_masks(self):
  pi=torch.tensor([[.4,.6,0.],[0.,0.,0.]]);mask=pi>0;c=torch.ones(2,20,3,3)
  w=core.gls_weights(pi,c,mask);self.assertTrue(torch.isfinite(w).all());self.assertEqual(float(w[1].abs().max()),0.);self.assertEqual(float(w[0,:,2].abs().max()),0.)
 def data(self,n=240):
  p=np.cumsum(self.rng.normal(size=n));r=core.reference(p);return p,r,core.MemoryData(p,r)
 def test_reference_prefix(self):
  p,r,m=self.data();p[160:]+=1e4;q=core.reference(p)
  for k in r:np.testing.assert_array_equal(r[k][:160],q[k][:160])
 def test_memory_prefix(self):
  p,r,m=self.data();p[160:]+=1e4;z=core.MemoryData(p,core.reference(p))
  for k in m.candidates:
   np.testing.assert_array_equal(m.candidates[k][:160],z.candidates[k][:160]);np.testing.assert_array_equal(m.values[k][:160],z.values[k][:160])
 def test_memory_maturity(self):
  _,_,m=self.data();ids=np.arange(240)[:,None]
  for k in m.candidates:self.assertTrue(np.all((m.candidates[k]+20<ids)[m.masks[k]]))
 def test_thin(self):
  _,_,m=self.data()
  for a,mask in zip(m.candidates['thin'],m.masks['thin']):
   x=a[mask];dist=abs(x[:,None]-x[None,:]);self.assertTrue(np.all(dist[~np.eye(len(x),dtype=bool)]>=20))
 def test_common_initialization(self):
  a=core.make_model('OVERLAP',17).state_dict()
  for mode in ['DIAGONAL','SOFTMAX','SHUFFLED','THIN','FULL']:
   b=core.make_model(mode,17).state_dict()
   for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
 def test_initial_reference(self):
  p,r,m=self.data();b=m.tensors(1.,[160,170,180],'OVERLAP');m0=core.make_model('OVERLAP',17)
  torch.testing.assert_close(m0(b),b['base'],rtol=0,atol=0)
 def test_zero_memory_is_full(self):
  _,_,m=self.data();b=m.tensors(1.,[160,170,180],'OVERLAP');a=core.make_model('OVERLAP',17);f=core.make_model('FULL',17)
  with torch.no_grad():a.O.weight.fill_(.1);f.O.weight.fill_(.1)
  torch.testing.assert_close(a(b),f(b),rtol=0,atol=0)
 def test_model_future_invariance(self):
  p,r,m=self.data();q=p.copy();q[180:]+=123.;n=core.MemoryData(q,core.reference(q));a=core.make_model('OVERLAP',17)
  with torch.no_grad():a.O.weight.fill_(.1);a.beta.fill_(.3)
  torch.testing.assert_close(a(m.tensors(1.,[150,160,179],'OVERLAP')),a(n.tensors(1.,[150,160,179],'OVERLAP')),rtol=0,atol=0)
 def test_support(self):
  p=np.cumsum(self.rng.normal(size=950))
  for e in [459,499,899]:
   ids,x,y,tr,va,s=core.support(p,e);self.assertLess(ids[tr[-1]]+20,ids[va[0]]);self.assertLess(ids[-1]+20,e)
 def test_training_scale(self):
  p=np.cumsum(self.rng.normal(size=950));ids,x,y,tr,va,s=core.support(p,459);p[ids[va[0]]:]+=9999
  self.assertEqual(s,core.support(p,459)[-1])
 def test_translation_scale(self):
  p,r,m=self.data();q=3*p+100;n=core.MemoryData(q,core.reference(q))
  np.testing.assert_array_equal(m.candidates['regular'],n.candidates['regular'])
  np.testing.assert_allclose(m.values['regular'],n.values['regular'],atol=1e-10)
 def test_no_label_in_candidate_choice(self):
  p,r,m=self.data();r2={k:v.copy() for k,v in r.items()};r2['b']+=12345;n=core.MemoryData(p,r2)
  for k in m.candidates:np.testing.assert_array_equal(m.candidates[k],n.candidates[k])
 def test_working_risk_not_guarantee(self):
  c=core.overlap_matrix(np.array([0,1,40]),20);pi=torch.ones(1,3)/3
  w=core.gls_weights(pi,torch.tensor(c)[None,None],torch.ones_like(pi,dtype=torch.bool))[0,0].numpy();u=np.ones(3)/3
  self.assertLess(w@c@w,u@c@u)
  bias=np.array([0.,0.,10.]);self.assertGreater((w@bias)**2+w@c@w,(u@bias)**2+u@c@u)
if __name__=='__main__':unittest.main(verbosity=2)
