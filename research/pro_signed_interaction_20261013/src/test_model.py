import unittest
from pathlib import Path
import sys,numpy as np,torch
from scipy.linalg import solve
from model import *
from experiment import audit,load_market,generate

class Tests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(123);self.p=100+np.cumsum(self.rng.normal(size=400))
    def test_01_block_orthogonal(self):np.testing.assert_allclose(Q@Q.T,np.eye(8),atol=1e-15)
    def test_02_prefix(self):
        a=prepare(self.p);pp=self.p.copy();pp[250:]+=999
        b=prepare(pp)
        for v,w in zip(a,b):np.testing.assert_array_equal(v[:250],w[:250])
    def test_03_scale_and_shift(self):
        x,s,_=prepare(self.p);xx,ss,_=prepare(3*self.p+600)
        np.testing.assert_allclose(x,xx,atol=1e-12);np.testing.assert_allclose(ss,3*s,atol=1e-12)
    def test_04_reflection(self):
        x,s,_=prepare(self.p);xx,ss,_=prepare(-self.p+600)
        np.testing.assert_allclose(x,-xx,atol=1e-12);np.testing.assert_allclose(ss,s,atol=1e-12)
    def test_05_basis_odd(self):
        e=Encoder('INTERACTION',7);x=torch.as_tensor(self.rng.normal(size=(35,8)))
        np.testing.assert_allclose(e(-x).detach(),-e(x).detach(),atol=1e-14)
    def test_06_flat_rw(self):
        x,s,_=prepare(np.ones(100));e=Encoder('INTERACTION',7);B=torch.ones((12,3));pr=ProfilePredictor(e,B,np.ones(3),{}).predict(x,s)
        np.testing.assert_array_equal(pr,np.zeros((100,3)))
    def test_07_linear_submodel(self):
        e=Encoder('INTERACTION',7);x=self.rng.normal(size=(30,8));B=torch.zeros((12,3));B[:8]=torch.as_tensor(self.rng.normal(size=(8,3)));pr=ProfilePredictor(e,B,np.ones(3),{}).predict(x,np.ones(30))
        np.testing.assert_allclose(pr,x@B[:8].numpy(),atol=1e-14)
    def test_08_nonlinearity(self):
        e=Encoder('INTERACTION',7);a=torch.as_tensor(self.rng.normal(size=(40,8)));b=torch.as_tensor(self.rng.normal(size=(40,8)))
        self.assertGreater(float((e(a+b)-e(a)-e(b)).detach().abs().max()),.01)
    def test_09_head_normal_equations(self):
        e=Encoder('INTERACTION',7);x=torch.as_tensor(self.rng.normal(size=(40,8)));s=torch.ones(40);y=torch.as_tensor(self.rng.normal(size=(40,3)));hs=torch.ones(3)
        loss,B=profiled(e,x,s,y,hs,.1);F=e(x).detach().numpy()
        expected=np.linalg.solve(F.T@F/40+.1*np.eye(12),F.T@y.numpy()/40)
        np.testing.assert_allclose(B.detach().numpy(),expected,atol=1e-12)
    def test_10_frozen_kernel_identity(self):
        e=Encoder('INTERACTION',7);x=torch.as_tensor(self.rng.normal(size=(40,8)));s=torch.ones(40);y=torch.as_tensor(self.rng.normal(size=(40,3)));hs=torch.ones(3)
        _,B=profiled(e,x,s,y,hs,.1);F=e(x).detach().numpy();C=np.linalg.solve(F@F.T+4*np.eye(40),y.numpy())
        np.testing.assert_allclose(F@B.detach().numpy(),F@F.T@C,atol=2e-12)
    def test_11_gradient(self):
        e=Encoder('INTERACTION',7);x=torch.as_tensor(self.rng.normal(size=(30,8)));s=torch.as_tensor(np.exp(self.rng.normal(size=30)));y=torch.as_tensor(self.rng.normal(size=(30,3)));hs=torch.as_tensor([1.,2.,3.])
        loss,_=profiled(e,x,s,y,hs,.1);loss.backward();ana=float(e.theta.grad[0,0,0]);old=float(e.theta[0,0,0].detach());eps=1e-5
        with torch.no_grad():e.theta[0,0,0]=old+eps
        a=float(profiled(e,x,s,y,hs,.1)[0].detach())
        with torch.no_grad():e.theta[0,0,0]=old-eps
        b=float(profiled(e,x,s,y,hs,.1)[0].detach())
        self.assertAlmostEqual(ana,(a-b)/(2*eps),places=7)
    def test_12_maturity(self):
        ii=support(300,240);self.assertEqual(ii.max()+20,299);self.assertGreaterEqual(ii.min(),60)
    def test_13_training_scale_future(self):
        a=training_scales(self.p,200);p=self.p.copy();p[200:]*=100
        np.testing.assert_array_equal(a,training_scales(p,200))
    def test_14_parameter_counts(self):
        self.assertEqual(Encoder('INTERACTION',0).count(),132)
        self.assertEqual(Encoder('MLP',0).count(),135)
        self.assertEqual(Encoder('ODD_MLP',0).count(),134)
    def test_15_market_cutoff_metadata(self):
        ds,p=load_market();self.assertLessEqual(max(ds.tolist()),'2021-12-31');self.assertEqual(p.ndim,1)
        self.assertEqual(set(audit()[0]['model_columns']),{'date','author_reported_price'})
    def test_16_odd_kernel(self):
        x=self.rng.normal(size=(30,8));k=kernel(x,x,2,True)
        np.testing.assert_allclose(k,k.T,atol=1e-14);self.assertGreater(np.linalg.eigvalsh(k).min(),-1e-12)
        np.testing.assert_allclose(kernel(-x,x,2,True),-k,atol=1e-14)
    def test_17_horizon_targets(self):
        p=np.arange(200,dtype=float);np.testing.assert_array_equal(targets(p,[61,62]),np.tile(H,(2,1)))
    def test_18_price_only_generator(self):
        p,mu=generate('interaction',300);self.assertEqual(p.shape,(1800,));self.assertEqual(mu.shape,(1800,));self.assertTrue(np.isfinite(p).all())
    def test_19_basis_bound(self):
        x,s,_=prepare(self.p);self.assertLessEqual(np.max(np.linalg.norm(x,axis=1)),np.sqrt(60)+1e-12)
    def test_20_fit_is_deterministic(self):
        x,s,_=prepare(self.p);idx=support(200);y=targets(self.p,idx);hs=training_scales(self.p,200)
        a=fit_profile('INTERACTION',x[idx],s[idx],y,hs,.1,7,steps=3)
        b=fit_profile('INTERACTION',x[idx],s[idx],y,hs,.1,7,steps=3)
        np.testing.assert_array_equal(a.predict(x[200:205],s[200:205]),b.predict(x[200:205],s[200:205]))

if __name__=='__main__':unittest.main(verbosity=2)
