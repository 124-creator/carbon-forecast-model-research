"""Finite algebra and implementation fixtures; not carbon performance evidence."""
import unittest
import numpy as np
import torch
import core

class Checks(unittest.TestCase):
 def setUp(self):core.configure();self.rng=np.random.default_rng(51);torch.manual_seed(51)
 def test_covariance_gram(self):
  n=31
  for h in [1,5,20]:
   a=np.zeros((n,n+h))
   for i in range(n):a[i,i+1:i+h+1]=1/np.sqrt(h)
   c=.5*np.eye(n)+.5*a@a.T;v=np.linalg.inv(c);v*=n/np.trace(v)
   np.testing.assert_allclose(v,core.precision(n,h),atol=1e-12)
 def test_positive_symmetric_part(self):
  for mode in ['CAUSAL','SHUFFLED']:
   u=core.operator_array(33,mode)
   self.assertGreater(np.linalg.eigvalsh((u+u.transpose(0,2,1))/2).min(),0)
   self.assertEqual(float(abs(np.tril(u,-1)).max()),0.)
 def test_symmetric_equals_working_precision(self):
  u=core.operator_array(31,'CAUSAL')
  for h in range(1,21):np.testing.assert_allclose((u[h-1]+u[h-1].T)/2,core.precision(31,h),atol=1e-14)
 def test_h1_identity(self):
  for mode in ['CAUSAL','SYMMETRIC','SHUFFLED']:np.testing.assert_array_equal(core.operator_array(31,mode)[0],np.eye(31))
 def test_head_equation(self):
  g=torch.randn(31,8);y=torch.randn(31,20);u=torch.tensor(core.operator_array(31,'CAUSAL'))
  b=core.solve_head(g,y,u);score=core.score_direction(g@b.T,y,u)
  residual=g.T@score/31+core.ALPHA*b.T
  self.assertLess(float(residual.abs().max()),1e-12)
 def test_head_identity_ridge(self):
  g=torch.randn(31,8);y=torch.randn(31,20);u=torch.tensor(core.operator_array(31,'IDENTITY'))
  a=core.solve_head(g,y,u).T;b=torch.linalg.solve(2*g.T@g/31+.01*torch.eye(8),g.T@y/31)
  torch.testing.assert_close(a,b,rtol=0,atol=1e-12)
 def test_no_covariance_scalar_loss_equivalence(self):
  g=torch.randn(31,3,requires_grad=True);b=torch.randn(20,3);y=torch.randn(31,20);u=torch.tensor(core.operator_array(31,'CAUSAL'))
  pred=g@b.T;score=core.score_direction(pred,y,u)
  correct=torch.autograd.grad(2*(pred*score.detach()).mean(),g,retain_graph=True)[0]
  # Differentiating the quadratic instead symmetrizes U: it is NOT this update.
  wrong=torch.autograd.grad(((pred-y)*((u@(pred-y).T.unsqueeze(-1)).squeeze(-1).T)).mean()+core.GAMMA*(pred**2).mean(),g)[0]
  self.assertGreater(float((correct-wrong).abs().max()),1e-5)
 def test_direction_finite_difference_fixed_score(self):
  g=torch.randn(31,3,requires_grad=True);b=torch.randn(20,3);y=torch.randn(31,20);u=torch.tensor(core.operator_array(31,'CAUSAL'))
  score=core.score_direction(g@b.T,y,u).detach();fn=lambda x:2*(x@b.T*score).mean()
  grad=torch.autograd.grad(fn(g),g)[0][0,0].item();d=torch.zeros_like(g);d[0,0]=1e-6
  fd=((fn(g.detach()+d)-fn(g.detach()-d))/(2e-6)).item();self.assertAlmostEqual(grad,fd,places=8)
 def test_null_population_moment(self):
  n=31;i=np.arange(n)[:,None];j=np.arange(n)[None,:]
  for h in [1,5,20]:
   cov=((i>j)&(i<=j+h)).astype(float)
   for mode in ['CAUSAL','IDENTITY','SHUFFLED']:self.assertEqual(float(np.sum(core.operator_array(n,mode)[h-1]*cov)),0.)
   if h>1:self.assertLess(float(np.sum(core.precision(n,h)*cov)),0.)
 def test_full_prefix_future_input(self):
  p=np.cumsum(self.rng.normal(size=180));q=p.copy();q[131:]+=10000
  m=core.make_model('CAUSAL',17)
  with torch.no_grad():m.B.fill_(.2)
  np.testing.assert_array_equal(core.predict(m,p,core.reference(p),[90,130],1),core.predict(m,q,core.reference(q),[90,130],1))
 def test_full_prefix_not_window_reset(self):
  p=np.cumsum(self.rng.normal(size=180));r=core.reference(p);bb=core.batch(p,r,[130],1,'CAUSAL')
  self.assertEqual(tuple(bb['x'].shape),(1,130,1));self.assertEqual(int(bb['indices'][0]),129)
 def test_reference_prefix(self):
  p=np.cumsum(self.rng.normal(size=180));q=p.copy();q[131:]+=1000
  a=core.reference(p);b=core.reference(q)
  for k in a:np.testing.assert_array_equal(a[k][:131],b[k][:131])
 def test_maturity(self):
  p=np.cumsum(self.rng.normal(size=950))
  for end in [459,499,899]:
   ids,x,y,tr,va,sd=core.support(p,end)
   self.assertLess(ids[tr[-1]]+20,ids[va[0]]);self.assertLess(ids[-1]+20,end)
 def test_scale_training_only(self):
  p=np.cumsum(self.rng.normal(size=950));ids,x,y,tr,va,s=core.support(p,459);q=p.copy();q[ids[va[0]]:]+=1000
  self.assertEqual(s,core.support(q,459)[-1])
 def test_parameter_count(self):
  for mode in core.MODES:
   count=sum(p.numel() for p in core.make_model(mode,17).parameters());self.assertEqual(count,1252 if mode=='GRU_WINDOW' else 1232)
 def test_common_initialization(self):
  a=core.make_model('CAUSAL',17)
  for mode in ['IDENTITY','SYMMETRIC','SHUFFLED','FROZEN']:
   b=core.make_model(mode,17)
   for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
 def test_zero_features_finite_head(self):
  b=core.solve_head(torch.zeros(31,8),torch.randn(31,20),torch.tensor(core.operator_array(31,'CAUSAL')))
  self.assertEqual(float(b.abs().max()),0.)
 def test_regularization_not_zero_bias_claim(self):
  g=torch.ones(31,1);y=torch.ones(31,20);b=core.solve_head(g,y,torch.tensor(core.operator_array(31,'IDENTITY')))
  self.assertAlmostEqual(float(b[0,0]),1/2.01)

if __name__=='__main__':unittest.main(verbosity=2)
