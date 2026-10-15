"""Algebra and causality fixtures; these are not forecasting performance claims."""
import unittest,copy,tempfile
from pathlib import Path
import numpy as np
import torch
from model import *
from experiment import fit_one,matured_mix

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.p=100+np.cumsum(np.random.default_rng(500).normal(size=950))
        cls.ref=reference(cls.p)
    def test_01_capacity(self):
        self.assertEqual(sum(x.numel() for x in InteractionGRU('MIXED').parameters()),1232)
        self.assertEqual(sum(x.numel() for x in PlainGRU().parameters()),1252)
    def test_02_initialization(self):
        ss=[]
        for mode in MODES:
            torch.manual_seed(17);ss.append(InteractionGRU(mode).state_dict())
        for other in ss[1:]:
            for k,v in ss[0].items():self.assertTrue(torch.equal(v,other[k]))
    def test_03_zero_output(self):
        m=InteractionGRU('MIXED');ids=np.arange(459,470)
        pred=predict(m,1.3,self.p,self.ref,ids)
        np.testing.assert_array_equal(pred,self.ref['b'][ids])
    def test_04_factorization(self):
        torch.manual_seed(6);m=InteractionGRU('FULL').double();torch.nn.init.normal_(m.O.weight)
        h=torch.randn(5,16,dtype=torch.float64);q=torch.randn(5,3,dtype=torch.float64)
        full=m.correction_from_hidden(h,q);m.mode='ADDITIVE';add=m.correction_from_hidden(h,q)
        m.mode='MIXED';mix=m.correction_from_hidden(h,q)
        torch.testing.assert_close(full,add+mix,atol=1e-12,rtol=1e-12)
    def test_05_zero_reference(self):
        m=InteractionGRU('MIXED').double();torch.nn.init.normal_(m.O.weight)
        r=m.correction_from_hidden(torch.randn(4,16,dtype=torch.float64),torch.zeros(4,3,dtype=torch.float64))
        torch.testing.assert_close(r,torch.zeros_like(r),atol=1e-12,rtol=0)
    def test_06_zero_history(self):
        m=InteractionGRU('MIXED').double();torch.nn.init.normal_(m.O.weight)
        r=m.correction_from_hidden(torch.zeros(4,16,dtype=torch.float64),torch.randn(4,3,dtype=torch.float64))
        torch.testing.assert_close(r,torch.zeros_like(r),atol=1e-12,rtol=0)
    def test_07_affine_mixed_zero(self):
        rng=np.random.default_rng(1);a,c,b=rng.normal(size=(3,6));w=rng.normal(size=(6,4))
        np.testing.assert_allclose((a+c+b)@w-(a+b)@w-(c+b)@w+b@w,0,atol=1e-14)
    def test_08_gradient(self):
        torch.manual_seed(11);m=InteractionGRU('MIXED').double();torch.nn.init.normal_(m.O.weight)
        h=torch.randn(2,16,dtype=torch.float64);q=torch.randn(2,3,dtype=torch.float64)
        val=m.correction_from_hidden(h,q).square().mean();g=torch.autograd.grad(val,m.V.weight)[0][0,0].item()
        with torch.no_grad():
            old=m.V.weight[0,0].item();eps=1e-6
            m.V.weight[0,0]=old+eps;a=m.correction_from_hidden(h,q).square().mean().item()
            m.V.weight[0,0]=old-eps;b=m.correction_from_hidden(h,q).square().mean().item()
            m.V.weight[0,0]=old
        self.assertAlmostEqual(g,(a-b)/(2*eps),places=7)
    def test_09_causal_reference(self):
        q=self.p.copy();q[500:]+=500
        for w in (120,240,480):
            a,b=reference(self.p,w),reference(q,w)
            for key in a:np.testing.assert_array_equal(a[key][:500],b[key][:500])
    def test_10_reference_2d(self):
        states,_,_=filter_bank(self.p[:100])
        for j,(rho,q) in enumerate(BANK):
            r=1-q;v=r/(1-rho*rho);mean=np.array([self.p[0],0.]);cov=np.array([[v,-v],[-v,v]])
            F=np.diag([1.,rho]);Q=np.diag([q,r]);H=np.ones(2)
            for t in range(1,100):
                mu=F@mean;pp=F@cov@F.T+Q;gain=pp@H/(H@pp@H)
                mean=mu+gain*(self.p[t]-H@mu);cov=pp-np.outer(gain,H@pp)
                self.assertAlmostEqual(mean[1],states[t,j],places=9)
    def test_11_rw_state(self):
        states,e,s=filter_bank(self.p)
        np.testing.assert_array_equal(states[:,0],0)
    def test_12_maturity(self):
        for end in range(459,900,40):
            ids,ts,tr,va,sd,r=tensor_arrays(self.p,self.ref,end)
            self.assertLess(ids[-1]+20,end)
            self.assertLess(ids[tr[-1]]+20,ids[va[0]])
    def test_13_scaler_support(self):
        ids,ts,tr,va,sd,r=tensor_arrays(self.p,self.ref,459)
        manual=np.array([np.diff(self.p[t-60:t+1]) for t in ids[tr]])
        self.assertEqual(sd,float(manual.std()))
    def test_14_future_arrays(self):
        q=self.p.copy();q[459:]*=10
        a=tensor_arrays(self.p,self.ref,459);b=tensor_arrays(q,reference(q),459)
        self.assertEqual(a[-2],b[-2])
        for u,v in zip(a[1],b[1]):torch.testing.assert_close(u,v,rtol=0,atol=0)
    def test_15_zero_context_rw(self):
        rf=dict(z=np.zeros(3),q=np.ones(3),rho=np.zeros(3))
        np.testing.assert_array_equal(state_features(rf,np.arange(3),1),np.zeros((3,3)))
    def test_16_nonlinearity(self):
        m=InteractionGRU('MIXED').double()
        with torch.no_grad():
            m.U.weight.zero_();m.V.weight.zero_();m.O.weight.zero_();m.bias.fill_(.7)
            m.U.weight[0,0]=1;m.V.weight[0,0]=1;m.O.weight[0,0]=1
        h=torch.zeros(1,16,dtype=torch.float64);h[0,0]=.7;q=torch.zeros(1,3,dtype=torch.float64);q[0,0]=.8
        self.assertGreater(abs(float(m.correction_from_hidden(h,q)[0,0].detach())),.01)
    def test_17_mixture_causality(self):
        a=np.ones((950,20))*.1;b=np.ones((950,20))*(-.1)
        p=self.p.copy();q=p.copy();q[600:]+=100
        x,w=matured_mix(p,a,b);y,v=matured_mix(q,a,b)
        np.testing.assert_array_equal(x[:600],y[:600]);np.testing.assert_array_equal(w[:600],v[:600])
    def test_18_numeric_parameters(self):
        m=InteractionGRU('MIXED')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'w.npz';np.savez(path,**{k:v.detach().numpy() for k,v in m.state_dict().items()})
            with np.load(path,allow_pickle=False) as z:
                self.assertTrue(all(z[k].dtype.kind in 'fiu' for k in z.files))
    def test_19_prefix_extra_sentinel(self):
        path=Path(__file__).parents[2]/'data/parsed/figshare_28682348/guangzhou_reported.csv'
        if not path.exists():self.skipTest('Prefix supplied to runner separately')
        # Read exactly approved bytes; no full-file read even in the test.
        with path.open('rb',buffering=0) as f:raw=f.read(29204)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.csv';p.write_bytes(raw+b'NEVER_PARSE_THIS_SENTINEL\n')
            _,_,receipt=load_prefix(p);self.assertFalse(receipt['next_byte_requested'])
    def test_20_prefix_reject_modified(self):
        path=Path(__file__).parents[2]/'data/parsed/figshare_28682348/guangzhou_reported.csv'
        if not path.exists():self.skipTest('Prefix supplied to runner separately')
        with path.open('rb',buffering=0) as f:raw=f.read(29204)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.csv';p.write_bytes(raw.replace(b'12.93',b'12.94',1))
            with self.assertRaises(ValueError):load_prefix(p)

if __name__=='__main__':unittest.main(verbosity=2)
