import unittest
import numpy as np
import run

class Core(unittest.TestCase):
    def test_features_causal_and_observation_steps(self):
        self.assertTrue(hasattr(run,'features'), 'Missing causal diagnostic features')
        p=np.array([1.,1.,1.,2.,2.,3.])
        d=np.array(['2020-01-01','2020-01-02','2020-01-06','2020-01-07','2020-01-08','2020-01-09'])
        x=run.features(p,d,np.ones((6,3)), 'quote')
        self.assertAlmostEqual(x[2,0],np.log1p(2))
        self.assertAlmostEqual(x[2,1],np.log1p(4))
        q=p.copy();q[4:]+=99
        np.testing.assert_array_equal(x[:4],run.features(q,d,np.ones((6,3)),'quote')[:4])

    def test_full_run_and_future_invariance(self):
        self.assertTrue(hasattr(run,'probe'), 'Missing mature residual diagnostic')
        p=np.cumsum(np.sin(np.arange(140)))
        d=np.arange(np.datetime64('2020-01-01'),np.datetime64('2020-01-01')+140).astype(str)
        a,log=run.probe(p,d,100,np.array([100,103]),'all',1.)
        q=p.copy();q[104:]+=777
        b,_=run.probe(q,d,100,np.array([100,103]),'all',1.)
        np.testing.assert_array_equal(a,b)
        self.assertLess(log['max_training_target'],100)
        self.assertEqual(a.shape,(2,3))

if __name__=='__main__':unittest.main()
