import unittest
import numpy as np

class Contract(unittest.TestCase):
    def test_mature_l60_twenty_outputs(self):
        from runner import training_data
        p=np.arange(200,dtype=float)**2
        x,y,s=training_data(p,100)
        if s.tolist()!=list(range(60,80)): raise RuntimeError('strict maturity')
        np.testing.assert_array_equal(x[0],np.diff(p[:61]))
        np.testing.assert_array_equal(y[0],p[61:81]-p[60])
        q=p.copy();q[100:]+=999
        a,b,c=training_data(q,100)
        np.testing.assert_array_equal(x,a);np.testing.assert_array_equal(y,b)

    def test_fixed_author_fit_future_invariance(self):
        from runner import fit_model, model_signature
        rng=np.random.default_rng(7);p=np.cumsum(rng.normal(size=140))
        x,y,s=training_data_local(p,100)
        m=fit_model('MBTR',x,y,2)
        q=p.copy();q[100:]+=10000
        a,b,c=training_data_local(q,100);n=fit_model('MBTR',a,b,2)
        if model_signature(m)!=model_signature(n):raise RuntimeError('future changed fitted parameters')
        np.testing.assert_array_equal(m.predict(x[-3:]),n.predict(x[-3:]))

def training_data_local(p,t):
    from runner import training_data
    return training_data(p,t)

if __name__=='__main__': unittest.main()
