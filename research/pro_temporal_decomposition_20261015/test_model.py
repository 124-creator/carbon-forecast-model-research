"""Implementation tests only. Synthetic fixtures are not performance claims."""
import tempfile,unittest,copy,io,hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import model

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(109)
        self.x=torch.randn(4,12,dtype=torch.float64)
        self.m=model.make_model('D_TEMP',17,torch.float64)
        self.p=np.cumsum(np.random.default_rng(601).normal(size=620))+100
    def test_01_parameter_counts(self):
        for mode,n in [('S_ERM',1202),('D_TEMP',1208),('GRU',1252)]:
            self.assertEqual(sum(p.numel() for p in model.make_model(mode,17).parameters()),n)
    def test_02_reconstruct(self):
        with torch.no_grad():self.m.router.phi.uniform_(-2,2)
        a,b,w=self.m.router(self.x)
        torch.testing.assert_close(a+b,self.x,atol=2e-15,rtol=0)
        torch.testing.assert_close(w.sum(-1),torch.ones_like(a),atol=2e-15,rtol=0)
    def test_03_backward_mean(self):
        for k in [1,5,20]:
            true=torch.stack([self.x[:,max(0,t-k+1):t+1].mean(1) for t in range(12)],1)
            torch.testing.assert_close(model.backward_mean(self.x,k),true,atol=1e-15,rtol=1e-14)
    def test_04_context_strict_past(self):
        _,q=model.statistics(self.x); xx=self.x.clone();xx[:,6:]+=50
        _,qq=model.statistics(xx)
        torch.testing.assert_close(q[:,:7],qq[:,:7],atol=0,rtol=0)
    def test_05_decomposition_prefix(self):
        a,b,w=self.m.router(self.x);aa,bb,ww=self.m.router(self.x[:,:7])
        torch.testing.assert_close(a[:,:7],aa,atol=1e-15,rtol=1e-14)
        torch.testing.assert_close(w[:,:7],ww,atol=1e-15,rtol=1e-14)
    def test_06_static_dynamic_boundary(self):
        s=model.make_model('S_TEMP',17,torch.float64)
        torch.testing.assert_close(s(self.x),self.m(self.x),atol=0,rtol=0)
    def test_07_shared_theta_initialization(self):
        models=[model.make_model(mode,29) for mode in model.MODES[:4]]
        for m in models[1:]:
            for (n,p),(nn,q) in zip(model.theta_items(models[0]),model.theta_items(m)):
                self.assertEqual(n,nn);torch.testing.assert_close(p,q,atol=0,rtol=0)
    def test_08_mature_support(self):
        for end in [459,619,899]:
            p=np.pad(self.p,(0,max(0,end-len(self.p))),mode='edge')
            ids,x,y,F,C,A,B,pool,sd,r=model.training_arrays(p,end)
            self.assertLess(ids[-1]+20,end)
            self.assertLess(ids[A[-1]]+20,ids[B[0]])
            self.assertLess(ids[F[-1]]+20,ids[C[0]])
            self.assertEqual(len(C),60)
    def test_09_scale_A_only(self):
        args=model.training_arrays(self.p,459);ids=args[0];A=args[5]
        q=self.p.copy();q[ids[A[-1]]+1:]+=999
        new=model.training_arrays(q,459)
        self.assertEqual(args[-2],new[-2])
    def test_10_training_prefix(self):
        a=model.training_arrays(self.p,459);q=self.p.copy();q[459:]+=1000
        b=model.training_arrays(q,459)
        for i in range(9):np.testing.assert_array_equal(a[i],b[i])
    def test_11_functional_update(self):
        y=torch.randn(4,20,dtype=torch.float64)
        J,upd,*_=model.temporal_objective(self.m,self.x,y,self.x,y)
        other=copy.deepcopy(self.m)
        with torch.no_grad():
            for n,p in model.theta_items(other):p.copy_(upd[n])
        val=(other(self.x)-y).square().mean()+.001*other.router.phi.square().sum()
        torch.testing.assert_close(J,val,atol=1e-14,rtol=1e-14)
    def test_12_hypergradient_finite_difference(self):
        y=torch.randn(4,20,dtype=torch.float64);xb=self.x.flip(0)*.7;yb=y*.8
        with torch.no_grad():self.m.router.phi.uniform_(-.2,.2)
        J,*_=model.temporal_objective(self.m,self.x,y,xb,yb)
        g=torch.autograd.grad(J,self.m.router.phi)[0].detach().numpy();fd=[]
        phi=self.m.router.phi.detach().clone();eps=1e-5
        for k in range(8):
            vv=[]
            for sign in [-1,1]:
                with torch.no_grad():self.m.router.phi.copy_(phi);self.m.router.phi[k]+=sign*eps
                vv.append(float(model.temporal_objective(self.m,self.x,y,xb,yb)[0].detach()))
            fd.append((vv[1]-vv[0])/(2*eps))
        np.testing.assert_allclose(g,fd,atol=1e-8,rtol=1e-5)
    def test_13_mixed_derivative_nonzero(self):
        y=torch.randn(4,20,dtype=torch.float64);xb=self.x*1.3;yb=y*.5
        J,upd,*_=model.temporal_objective(self.m,self.x,y,xb,yb,eta=.1)
        exact=torch.autograd.grad(J,self.m.router.phi)[0]
        params={n:v.detach() for n,v in upd.items()};params['router.phi']=self.m.router.phi
        first=(model.functional_call(self.m,params,(xb,),strict=True)-yb).square().mean()+.001*self.m.router.phi.square().sum()
        g=torch.autograd.grad(first,self.m.router.phi)[0]
        self.assertGreater(float((exact-g).norm()),1e-9)
    def test_14_clip_bound_and_zero(self):
        gs=[torch.ones(5),torch.ones(4)];new,n=model.differentiable_clip(gs)
        self.assertLessEqual(float(torch.sqrt(sum(a.square().sum() for a in new))),1.000001)
        z,_=model.differentiable_clip([torch.zeros(3)])
        self.assertTrue(torch.isfinite(z[0]).all())
    def test_15_linear_reparameterization(self):
        # Fixed static decomposition is a linear transformation; a linear reader can absorb it.
        s=model.Decomposition(False).double();L=12
        cI,dI,_=s(torch.eye(L,dtype=torch.float64))
        wc=torch.randn(L,dtype=torch.float64);wd=torch.randn(L,dtype=torch.float64)
        c,d,_=s(self.x);a=c@wc+d@wd;b=self.x@(cI@wc+dI@wd)
        torch.testing.assert_close(a,b,atol=1e-14,rtol=1e-14)
    def test_16_dynamic_non_linearity(self):
        with torch.no_grad():self.m.router.phi.copy_(torch.tensor([.8,0,0,0,0,.3,0,0],dtype=torch.float64))
        x=self.x[:2];y=self.x[2:]
        cxy=self.m.router(x+y)[0];cx=self.m.router(x)[0];cy=self.m.router(y)[0]
        self.assertGreater(float((cxy-cx-cy).detach().norm()),.001)
    def test_17_no_tail_or_metadata_features(self):
        path=Path(__file__).parent/'data/guangzhou_prefix950.csv'
        p,d,r=model.read_prefix(path)
        raw=path.read_bytes();rows=raw.decode().splitlines()
        changed=[rows[0]]+[(','.join(line.split(',')[:2])+',999999,irrelevant_format') for line in rows[1:]]
        text=('\n'.join(changed)+'\n').encode()
        class Guard(io.BytesIO):
            def read(self,n=-1):
                if self.tell()>=len(text):raise AssertionError('requested a tail byte')
                return super().read(n)
        with patch.object(Path,'open',lambda *a,**kw:Guard(text+b'FORBIDDEN_TAIL')):
            pp,dd,rr=model.read_prefix('unused',False)
        np.testing.assert_array_equal(p,pp);np.testing.assert_array_equal(d,dd)
    def test_18_full_fit_future_and_replay_fixture(self):
        lim=dict(warmup=1,outer=2,erm=3,final=3,joint=3,patience=3)
        a,s,log=model.fit_model(self.p,459,'D_TEMP',17,lim)
        q=self.p.copy();q[459:]+=10000
        b,ss,ll=model.fit_model(q,459,'D_TEMP',17,lim)
        self.assertEqual(s,ss)
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
        pred,_=model.forecast(a,s,self.p,[458]);predb,_=model.forecast(b,ss,q,[458]);np.testing.assert_array_equal(pred,predb)
    def test_19_frozen_phi_final(self):
        lim=dict(warmup=1,outer=2,erm=3,final=2,joint=2,patience=2)
        a,s,log=model.fit_model(self.p,459,'D_ERM',17,lim)
        self.assertFalse(a.router.phi.requires_grad)
        self.assertGreater(float(a.router.phi.norm()),0.)
    def test_20_flat_series(self):
        p=np.ones(620)*10
        args=model.training_arrays(p,459)
        self.assertEqual(args[-2],1e-6)
        self.assertTrue(np.isfinite(args[1]).all())

if __name__=='__main__':unittest.main(verbosity=2)
