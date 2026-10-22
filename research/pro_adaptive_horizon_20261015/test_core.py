"""Pre-fit algebra and data-boundary fixtures; not forecasting-performance evidence."""
import unittest
import numpy as np
import torch
import core

class Checks(unittest.TestCase):
    def setUp(self):
        core.configure();torch.manual_seed(11);self.rng=np.random.default_rng(11)
    def test_base_nodes(self):
        k=core.knots_from_logits(torch.zeros(4,5))
        torch.testing.assert_close(k,torch.tensor([1.,2.,3.,6.,11.,20.])[None].expand(4,-1),atol=5e-15,rtol=0)
    def test_order_and_endpoints(self):
        k=core.knots_from_logits(100*torch.randn(200,5))
        self.assertTrue(torch.all(torch.diff(k,dim=-1)>=.25-1e-12))
        torch.testing.assert_close(k[:,0],torch.ones(200),atol=0,rtol=0)
        torch.testing.assert_close(k[:,-1],20*torch.ones(200),atol=1e-12,rtol=0)
    def test_fixed_interpolation(self):
        k=core.knots_from_logits(torch.zeros(7,5));a=torch.randn(7,6)
        torch.testing.assert_close(core.interpolate(k,a),a@core.initial_basis(6).T,atol=3e-14,rtol=0)
    def test_exact_knot_values(self):
        k=core.knots_from_logits(torch.randn(1,5));a=torch.randn(1,6)
        torch.testing.assert_close(core.interpolate(k,a,k[0]),a,atol=2e-15,rtol=0)
    def test_origin(self):
        k=core.knots_from_logits(torch.randn(5,5));a=torch.randn(5,6)
        torch.testing.assert_close(core.interpolate(k,a,[0.]),torch.zeros(5,1),atol=0,rtol=0)
    def test_affine_reproduction(self):
        k=core.knots_from_logits(torch.randn(5,5));a=.7*k-2
        query=torch.linspace(1,20,93)
        torch.testing.assert_close(core.interpolate(k,a,query),(.7*query-2)[None].expand(5,-1),atol=4e-14,rtol=0)
    def test_convex_range(self):
        k=core.knots_from_logits(torch.randn(5,5));a=torch.randn(5,6);p=core.interpolate(k,a)
        self.assertTrue(torch.all(p<=a.max(-1).values[:,None]+1e-12))
        self.assertTrue(torch.all(p>=a.min(-1).values[:,None]-1e-12))
    def test_constant_values(self):
        k=core.knots_from_logits(torch.randn(5,5));a=torch.ones(5,6)*2
        torch.testing.assert_close(core.interpolate(k,a),torch.ones(5,20)*2,atol=2e-15,rtol=0)
    def test_partition_unity(self):
        k=core.knots_from_logits(torch.randn(1,5));basis=core.interpolate(k.expand(6,-1),torch.eye(6)).T
        torch.testing.assert_close(basis.sum(-1),torch.ones(20),atol=2e-15,rtol=0)
        self.assertTrue(torch.all(basis>=-1e-15))
    def test_geometry_derivative(self):
        z=torch.tensor([[.11,-.2,.23,.31,-.15]],requires_grad=True);a=torch.tensor([[1.,-.3,.7,.2,-.2,.6]])
        def f(zz):return core.interpolate(core.knots_from_logits(zz),a,[1.31,4.91,8.37,17.21]).square().sum()
        g=torch.autograd.grad(f(z),z)[0].detach().numpy();eps=1e-6;num=np.zeros((1,5))
        for j in range(5):
            dz=torch.zeros_like(z);dz[0,j]=eps
            num[0,j]=(f(z.detach()+dz).item()-f(z.detach()-dz).item())/(2*eps)
        np.testing.assert_allclose(g,num,rtol=1e-6,atol=1e-8)
    def test_common_initialization(self):
        mods=[core.make_model(m,17) for m in ['FIXED6','GLOBAL','TILT','MESH']]
        for key,p in mods[0].named_parameters():
            for m in mods[1:]:torch.testing.assert_close(p,dict(m.named_parameters())[key],atol=0,rtol=0)
    def test_reference_initial_prediction(self):
        x=torch.randn(4,60);q=torch.randn(4,3);base=torch.randn(4,20)
        for mode in core.MODES[:-1]:
            pred=core.make_model(mode,17)(x,q,base)
            torch.testing.assert_close(pred,base,rtol=0,atol=0)
    def test_static_limit(self):
        m=core.make_model('MESH',17);g=core.make_model('GLOBAL',17);f=torch.randn(9,8)
        with torch.no_grad():m.geometry.bias.copy_(torch.tensor([.1,-.2,.3,-.4]));g.geometry.copy_(m.geometry.bias)
        torch.testing.assert_close(m.knot_locations(f),g.knot_locations(f),rtol=0,atol=0)
    def test_dynamic_geometry(self):
        m=core.make_model('MESH',17)
        with torch.no_grad():m.geometry.weight.fill_(.2)
        k=m.knot_locations(torch.stack([torch.zeros(8),torch.ones(8)]))
        self.assertGreater(float((k[0]-k[1]).abs().max()),.1)
    def test_counts(self):
        self.assertEqual([sum(p.numel() for p in core.make_model(m,17).parameters()) for m in core.MODES],[1171,1139,1144,1135,1197,1255,1232,1252])
    def test_shifted_kink_not_in_fixed_span(self):
        # Algebraic counterexample only: it is not a market target or a performance test.
        k=torch.tensor([[1.,2.,3.,8.,11.,20.]])
        y=core.interpolate(k,torch.relu(k-8.))[0].numpy()
        B=core.initial_basis(6).numpy();a=np.linalg.lstsq(B,y,rcond=None)[0]
        self.assertGreater(float(np.mean((B@a-y)**2)),.01)
        np.testing.assert_allclose(y,np.maximum(np.arange(1,21)-8.,0),atol=1e-14)
    def test_continuity(self):
        k=core.knots_from_logits(torch.randn(1,5));a=torch.randn(1,6);eps=1e-8
        p=core.interpolate(k,a,k[0,1:-1]+eps);q=core.interpolate(k,a,k[0,1:-1]-eps)
        self.assertLess(float((p-q).abs().max()),1e-6)
    def test_reference_causality(self):
        p=np.cumsum(self.rng.normal(size=950));q=p.copy();q[500:]+=10000
        a=core.reference(p);b=core.reference(q)
        for key in a:np.testing.assert_array_equal(a[key][:500],b[key][:500])
    def test_mature_support(self):
        p=np.cumsum(self.rng.normal(size=950))
        for end in [459,499,899]:
            ids,x,y,tr,va,sd=core.support(p,end)
            self.assertLess(ids[tr[-1]]+20,ids[va[0]])
            self.assertLess(ids[-1]+20,end)
    def test_training_scale(self):
        p=np.cumsum(self.rng.normal(size=950));ids,x,y,tr,va,s=core.support(p,459)
        q=p.copy();q[ids[va[0]]:]+=1000
        self.assertEqual(s,core.support(q,459)[-1])
    def test_price_error_identity(self):
        y=self.rng.normal(size=(4,20));d=self.rng.normal(size=(4,20));p=self.rng.normal(size=(4,1))
        np.testing.assert_allclose((y-d)**2,((p+y)-(p+d))**2,atol=2e-14)
    def test_flat_reference(self):
        r=core.reference(np.ones(950)*10)
        np.testing.assert_array_equal(r['b'],np.zeros((950,20)))

if __name__=='__main__':unittest.main(verbosity=2)
