"""Deterministic science/implementation checks; no market outcomes used."""
import unittest, numpy as np
from scipy.optimize import minimize
from lowdim import H,filter_bank,bank_parameters,profile_nll,generate,features,targets,ridge_predict
from experiment import project_hull,COUNTERS,load_market_prices

class Controls(unittest.TestCase):
    def setUp(self):self.p=generate(180,44,.65,.15,.85)
    def test_null_exact(self):
        f,_,_=filter_bank(self.p,np.array([[0.,1.]]));np.testing.assert_array_equal(f,0)
    def test_prefix_invariance(self):
        f,l,_=filter_bank(self.p);short=filter_bank(self.p[:70])
        np.testing.assert_array_equal(f[:70],short[0]);np.testing.assert_array_equal(l[:70],short[1])
    def test_future_perturbation(self):
        p=self.p.copy();p[100:]+=np.arange(80)*7
        np.testing.assert_array_equal(filter_bank(p)[0][:100],filter_bank(self.p)[0][:100])
    def test_independent_two_state_kalman(self):
        for rho,q in [(0.,.1),(.5,.25),(.97,.8)]:
            r=1-q;A=np.diag([1.,rho]);Q=np.diag([q,r]);h=np.ones(2)
            x=np.array([self.p[0],0.]);V=r/(1-rho*rho);P=V*np.array([[1.,-1.],[-1.,1.]])
            expected=[]
            for y in self.p[1:]:
                xp=A@x;PP=A@P@A.T+Q;S=h@PP@h;K=PP@h/S
                x=xp+K*(y-h@xp);P=PP-np.outer(K,h@PP)
                expected.append([x[0]+rho**int(k)*x[1]-y for k in H])
            actual=filter_bank(self.p,np.array([[rho,q]]))[0][1:,0]
            np.testing.assert_allclose(actual,expected,atol=1e-11)
    def test_affine_price_equivariance(self):
        f,_,_=filter_bank(self.p);f2,_,_=filter_bank(3*self.p+9)
        np.testing.assert_allclose(f2,3*f,atol=1e-11)
    def test_profile_rank_scale_invariance(self):
        l=filter_bank(self.p)[1];l2=filter_bank(3*self.p+9)[1]
        a=profile_nll(l,1,150);b=profile_nll(l2,1,150)
        np.testing.assert_allclose(a-a.min(),b-b.min(),atol=1e-9)
    def test_ridge_maturity_and_future_invariance(self):
        p=self.p.copy();p[121:]+=900
        a=ridge_predict(self.p,100,np.array([100,110,120]),5,.1)
        b=ridge_predict(p,100,np.array([100,110,120]),5,.1)
        np.testing.assert_array_equal(a,b)
        ii=np.arange(5,100-20);self.assertLessEqual(int((ii[:,None]+H).max()),99)
    def test_feature_lags(self):
        X=features(self.p,5)
        np.testing.assert_array_equal(X[20],np.diff(self.p)[15:20][::-1])
    def test_hull_singleton(self):
        m=np.array([[1.,2.,3.]])
        np.testing.assert_array_equal(project_hull(m),m[0])
    def test_hull_contains_zero(self):
        m=np.array([[1.,2.,3.],[-1.,-2.,-3.]])
        np.testing.assert_allclose(project_hull(m),0,atol=1e-7)
    def test_non_radial_equivalence_counterexample(self):
        m=np.array([[1.,0.,0.],[0.,2.,0.]])
        np.testing.assert_allclose(project_hull(m),[.8,.4,0],atol=1e-7)
    def test_minimax_equals_minimum_norm_hull(self):
        rng=np.random.default_rng(901)
        for _ in range(20):
            M=rng.normal(size=(5,3))+[1,1,1]
            proj=project_hull(M)
            def objective(x):return x[3]
            def con(x):return x[3]-x[:3]@x[:3]+2*M@x[:3]
            sol=minimize(objective,np.zeros(4),method='SLSQP',constraints={'type':'ineq','fun':con},options={'ftol':1e-11,'maxiter':400})
            self.assertTrue(sol.success)
            np.testing.assert_allclose(sol.x[:3],proj,atol=3e-5)
    def test_projection_kkt(self):
        rng=np.random.default_rng(12)
        for _ in range(20):
            M=rng.normal(size=(8,3))+1;v=project_hull(M)
            self.assertGreaterEqual(float(np.min(M@v-v@v)),-1e-6)
    def test_arma111_spectrum_equivalence(self):
        # (1-rho B) Dp = (1-rho B) eps + (1-B) eta has MA(1) covariance.
        for rho,q in bank_parameters():
            r=1-q;a=q*(1+rho*rho)+2*r;b=-rho*q-r
            theta=0. if abs(b)<1e-14 else (a-np.sqrt(max(0.,a*a-4*b*b)))/(2*b)
            v=a/(1+theta*theta)
            for w in [.01,.4,1.2,2.5]:
                z=np.exp(-1j*w)
                uc=q+r*abs(1-z)**2/abs(1-rho*z)**2
                ar=v*abs(1+theta*z)**2/abs(1-rho*z)**2
                self.assertAlmostEqual(uc,ar,places=10)
    def test_variance_fraction_boundary_identification(self):
        # With r=0, changing rho cannot change any observable forecast.
        a=filter_bank(self.p,np.array([[.1,1.],[.9,1.]]))[0]
        np.testing.assert_array_equal(a[:,0],a[:,1])
    def test_market_cutoff_and_metadata_exclusion(self):
        import tempfile, pathlib, pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);p=root/'data/parsed/figshare_28682348/guangzhou_reported.csv';p.parent.mkdir(parents=True)
            frame=pd.DataFrame({'date':['2021-12-30','2021-12-31','2022-01-03','2023-01-03'],'author_reported_price':[3.,4.,5.,6.],'source_excel_row':[1,2,3,4],'source_date_kind':['a']*4})
            frame.to_csv(p,index=False);first=load_market_prices(root)
            frame.loc[2:,'author_reported_price']=99999;frame['source_excel_row']=[900,800,700,600];frame['source_date_kind']='corrupted-metadata'
            frame.to_csv(p,index=False);second=load_market_prices(root)
            pd.testing.assert_frame_equal(first,second)
            self.assertEqual(list(first.columns),['date','author_reported_price']);self.assertEqual(len(first),2)
    def test_invalid_input(self):
        with self.assertRaises(ValueError):filter_bank([1,np.nan,2])
        with self.assertRaises(ValueError):filter_bank([1,2],np.array([[1.,.2]]))
        with self.assertRaises(ValueError):project_hull(np.empty((0,3)))

if __name__=='__main__':unittest.main(verbosity=2)
