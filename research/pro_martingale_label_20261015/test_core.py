"""Finite algebra and causal implementation checks, not predictive performance."""
import unittest
import numpy as np
import torch
from scipy.linalg import expm
import core

class Checks(unittest.TestCase):
 def setUp(self):
  core.configure();self.rng=np.random.default_rng(211);self.c=np.array([-4.,-2.,-.5,0.,.5,2.,4.])
 def test_generator_mass(self):np.testing.assert_allclose(core.generator(self.c).sum(1),0,atol=1e-14)
 def test_generator_moment(self):np.testing.assert_allclose(core.generator(self.c)@self.c,self.c*0,atol=1e-14)
 def test_offdiagonal(self):
  q=core.generator(self.c);q-=np.diag(q.diagonal());self.assertTrue((q>=0).all())
 def test_absorbing_boundaries(self):np.testing.assert_array_equal(core.generator(self.c)[[0,-1]],np.zeros((2,len(self.c))))
 def test_transition_mass(self):np.testing.assert_allclose(expm(core.generator(self.c)*1.2).sum(1),1,atol=1e-13)
 def test_transition_nonnegative(self):self.assertGreaterEqual(float(expm(core.generator(self.c)*1.2).min()),-1e-14)
 def test_transition_moment(self):np.testing.assert_allclose(expm(core.generator(self.c)*1.2)@self.c,self.c,atol=1e-13)
 def test_semigroup(self):
  q=core.generator(self.c);np.testing.assert_allclose(expm(q*.3)@expm(q*.6),expm(q*.9),atol=1e-13)
 def test_twohot(self):
  c=np.tile(self.c,(20,1));y=self.rng.uniform(-4,4,(12,20));q=core.two_hot(y,c)
  np.testing.assert_allclose(q.sum(-1),1,atol=1e-14);np.testing.assert_allclose((q*c).sum(-1),y,atol=1e-14)
 def test_no_silent_clipping(self):
  with self.assertRaises(ValueError):core.two_hot(np.ones((2,20))*5,np.tile(self.c,(20,1)))
 def test_fisher_mean_fixture(self):
  c,p,labels,T,_=core.label_system(self.rng.normal(size=(25,20)))
  np.testing.assert_allclose((labels['MP_DIFF'].mean(0)*c).sum(-1),(labels['TWOHOT'].mean(0)*c).sum(-1),atol=1e-12)
 def test_scalar_risk_identity(self):
  r=self.rng.normal(size=100);a=self.rng.normal(size=100)
  np.testing.assert_allclose((r-a)**2+a*a,2*(a-r/2)**2+r*r/2,atol=1e-13)
 def test_target_is_not_original_law(self):
  _,_,d,_,_=core.label_system(np.ones((10,20)))
  self.assertGreater(float(np.max(np.abs(d['MP_DIFF']-d['TWOHOT']))),.01)
 def test_prior_symmetric_mean(self):
  c,p,_,_,_=core.label_system(self.rng.normal(size=(100,20)))
  np.testing.assert_allclose((np.exp(p)*c).sum(-1),0,atol=1e-13)
 def test_zero_head(self):
  c,p,*_=core.label_system(self.rng.normal(size=(20,20)));m=core.make_model('MP_DIFF',17,c,p)
  x=torch.randn(4,60);q=torch.randn(4,3);b=torch.randn(4,20)
  torch.testing.assert_close(m(x,q,b),b,atol=1e-12,rtol=0)
 def test_shared_initialization(self):
  a=core.make_model('MP_DIFF',17);b=core.make_model('TWOHOT',17)
  for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
 def test_finite_gradient(self):
  c,p,d,*_=core.label_system(self.rng.normal(size=(20,20)));m=core.make_model('MP_DIFF',17,c,p)
  x=torch.randn(20,60);q=torch.randn(20,3);b=torch.zeros(20,20)
  _,info=m(x,q,b,True);loss=-(torch.as_tensor(d['MP_DIFF'])*info['logprob']).sum(-1).mean();loss.backward()
  self.assertTrue(all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None))
 def test_reference_causal(self):
  p=self.rng.normal(size=200).cumsum();q=p.copy();q[140:]+=100
  a=core.reference(p);b=core.reference(q)
  for k in a:np.testing.assert_array_equal(a[k][:140],b[k][:140])
 def test_support(self):
  p=self.rng.normal(size=950).cumsum()
  for end in (459,499,899):
   ids,_,_,tr,va,_=core.support(p,end);self.assertLess(ids[tr[-1]]+20,ids[va[0]]);self.assertLess(ids[-1]+20,end)
 def test_scalar_scale_source(self):
  p=self.rng.normal(size=950).cumsum();ids,x,y,tr,va,s=core.support(p,459);q=p.copy();q[ids[va[0]]:]+=4000
  self.assertEqual(s,core.support(q,459)[-1])
 def test_parameter_counts(self):
  counts={k:sum(p.numel() for p in core.make_model(k,17).parameters()) for k in core.MODES}
  self.assertEqual(counts['MP_DIFF'],counts['TWOHOT']);self.assertEqual(counts['MP_DIFF'],counts['CAT_MSE'])
  self.assertGreater(counts['IND_DIFF'],counts['MP_DIFF']);self.assertEqual(counts['GRU'],1252)
 def test_gauss_not_mean_preserving(self):
  c=np.tile(self.c,(20,1));y=np.ones((10,20))*3.7
  g=core.gaussian_probs(y,c,np.ones(20))
  self.assertGreater(float(np.max(np.abs((g*c).sum(-1)-y))),.01)

if __name__=='__main__':unittest.main(verbosity=2)
