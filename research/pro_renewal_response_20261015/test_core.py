"""Finite algebra/causality fixtures; not market-performance evidence."""
import unittest, tempfile, hashlib
from pathlib import Path
import numpy as np
import torch
import core

class Checks(unittest.TestCase):
    def setUp(self):
        core.configure();torch.manual_seed(8);self.rng=np.random.default_rng(8)
    def args(self,n=4):
        return torch.randn(n,60),torch.randn(n,3),torch.randn(n,20)
    def components(self):
        P=core.stochastic_matrix(torch.randn(3,4,4))
        pi=torch.softmax(torch.randn(3,4),-1)
        v=torch.randn(3,4);d=torch.randn(3);b=torch.randn(3,4)
        return pi,P,v,d,b
    def test_zero_duration_markov(self):
        pi,P,v,d,b=self.components();b.zero_()
        actual,_=core.renewal_curve(pi,P,v,d,b);expected,_=core.markov_curve(pi,P,v,d)
        torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
    def test_zero_duration_clock(self):
        pi,P,v,d,b=self.components();b.zero_()
        a,_=core.renewal_curve(pi,P,v,d,b,True);e,_=core.markov_curve(pi,P,v,d)
        torch.testing.assert_close(a,e,rtol=1e-13,atol=1e-13)
    def test_mass_nonnegative(self):
        pi,P,v,d,b=self.components();_,info=core.renewal_curve(pi,P,v,d,b)
        self.assertGreaterEqual(float(info['final_age_mass'].min()),0)
        torch.testing.assert_close(info['occupancy'].sum(-1),torch.ones(3,20),rtol=1e-13,atol=1e-13)
    def test_one_step_same(self):
        pi,P,v,d,b=self.components();a,_=core.renewal_curve(pi,P,v,d,b);e,_=core.markov_curve(pi,P,v,d)
        torch.testing.assert_close(a[:,0],e[:,0],rtol=1e-13,atol=1e-13)
    def test_age_not_clock(self):
        pi,P,v,d,b=self.components();a,_=core.renewal_curve(pi,P,v,d,b);e,_=core.renewal_curve(pi,P,v,d,b,True)
        self.assertGreater(float((a-e).abs().max()),1e-5)
    def test_lifted_markov_equivalence(self):
        pi,P,v,d,b=self.components();T=7;a,inf=core.renewal_curve(pi[:1],P[:1],v[:1],d[:1],b[:1],steps=T)
        P=P[0].numpy();pi=pi[0].numpy();beta=b[0].numpy();v=v[0].numpy();d=float(d[0])
        e=1-np.diag(P);Q=P*(1-np.eye(4))/e[:,None];n=4*(T+1);L=np.zeros((n,n))
        for s in range(4):
            for age in range(T+1):
                hazard=1/(1+np.exp(-(np.log(e[s]/(1-e[s]))+beta[s]*np.log1p(age)/np.log(20))))
                i=s*(T+1)+age;L[i,s*(T+1)+min(age+1,T)]+=1-hazard
                for r in range(4):L[i,r*(T+1)]+=hazard*Q[s,r]
        mass=np.zeros(n);mass[np.arange(4)*(T+1)]=pi;ys=[]
        for h in range(1,T+1):
            mass=mass@L;ys.append((mass.reshape(4,T+1).sum(1)-pi)@v+h/20*d)
        np.testing.assert_allclose(ys,a[0].numpy(),atol=1e-13,rtol=1e-13)
    def test_gradient_finite_difference(self):
        pi,P,v,d,b=self.components();b.requires_grad_(True)
        score=lambda x:core.renewal_curve(pi,P,v,d,x)[0].square().mean()
        analytic=torch.autograd.grad(score(b),b)[0]
        eps=1e-6;differences=[]
        for s in range(4):
            x=b.detach().clone();x[0,s]+=eps;y=b.detach().clone();y[0,s]-=eps
            fd=(score(x)-score(y))/(2*eps);differences.append(abs(float(fd-analytic[0,s])))
        self.assertLess(max(differences),2e-8)
    def test_state_permutation(self):
        pi,P,v,d,b=self.components();idx=torch.tensor([2,0,3,1])
        a,_=core.renewal_curve(pi,P,v,d,b)
        c,_=core.renewal_curve(pi[:,idx],P[:,idx][:,:,idx],v[:,idx],d,b[:,idx])
        torch.testing.assert_close(a,c,rtol=1e-13,atol=1e-13)
    def test_value_shift(self):
        pi,P,v,d,b=self.components();a,_=core.renewal_curve(pi,P,v,d,b);c,_=core.renewal_curve(pi,P,v+5,d,b)
        torch.testing.assert_close(a,c,rtol=1e-12,atol=1e-12)
    def test_neutral_initialization(self):
        args=self.args()
        for kind in core.MODES[:-1]:
            m=core.make_model(kind,17);out=m(*args)
            torch.testing.assert_close(out,args[-1],rtol=0,atol=0)
    def test_common_parameters(self):
        models=[core.make_model(m,17) for m in ('RENEWAL','CLOCK','MARKOV')]
        for key,p in models[-1].state_dict().items():
            for m in models[:2]:torch.testing.assert_close(p,m.state_dict()[key],rtol=0,atol=0)
    def test_basis_fixed_learned_initial(self):
        a=core.make_model('RANK6',17);b=core.make_model('FIXED6',17)
        for key,p in a.state_dict().items():torch.testing.assert_close(p,b.state_dict()[key],rtol=0,atol=0)
    def test_rank_bound(self):
        rng=self.rng
        for r in (4,6):
            B=core.initial_basis(r).numpy();coeff=rng.normal(size=(100,r));d=rng.normal(size=(100,1))
            F=coeff@B.T+d*np.arange(1,21)/20
            self.assertLessEqual(np.linalg.matrix_rank(F),r+1)
    def test_basis_partition(self):
        for r in (4,6):
            B=core.initial_basis(r).numpy();self.assertTrue((B>=0).all())
            np.testing.assert_allclose(B.sum(1),1,rtol=0,atol=1e-14)
    def test_counts(self):
        counts=[sum(p.numel() for p in core.make_model(m,17).parameters()) for m in core.MODES]
        self.assertEqual(counts,[1264,1264,1228,1197,1255,1135,1232,1252])
    def test_prefix_reference(self):
        p=np.cumsum(self.rng.normal(size=210));q=p.copy();q[130:]+=1000
        a=core.reference(p);b=core.reference(q)
        for key in a:np.testing.assert_array_equal(a[key][:130],b[key][:130])
    def test_scale_translation_reference(self):
        p=np.cumsum(self.rng.normal(size=210));a=core.reference(p);b=core.reference(3*p+10)
        np.testing.assert_array_equal(a['selected'],b['selected']);np.testing.assert_allclose(3*a['b'],b['b'],atol=1e-12)
    def test_maturity(self):
        p=np.cumsum(self.rng.normal(size=950))
        for end in (459,499,899):
            ids,x,y,tr,va,sd=core.support(p,end)
            self.assertLess(ids[tr[-1]]+20,ids[va[0]]);self.assertLess(ids[-1]+20,end)
    def test_scale_train_only(self):
        p=np.cumsum(self.rng.normal(size=950));ids,x,y,tr,va,sd=core.support(p,459)
        q=p.copy();q[ids[va[0]]:]+=1000
        self.assertEqual(sd,core.support(q,459)[-1])
    def test_price_target_identity(self):
        a=self.rng.normal(size=(10,20));b=self.rng.normal(size=(10,20));p=self.rng.normal(size=(10,1))
        np.testing.assert_allclose((a-b)**2,((a+p)-(b+p))**2,atol=1e-13)
    def test_deterministic_same_seed(self):
        args=self.args();a=core.make_model('RENEWAL',17);b=core.make_model('RENEWAL',17)
        with torch.no_grad():a.payoff.weight.fill_(.1);b.payoff.weight.fill_(.1)
        torch.testing.assert_close(a(*args),b(*args),rtol=0,atol=0)
    def test_flat_reference(self):
        a=core.reference(np.ones(950)*13);np.testing.assert_array_equal(a['b'],np.zeros((950,20)))

if __name__=='__main__':unittest.main(verbosity=2)
