import unittest
import numpy as np
import prototype as p

class Contract(unittest.TestCase):
    def test_asof_excludes_unreleased_and_immature(self):
        release=np.array([0,4,2,5])
        self.assertEqual(p.mature_indices(release,3,240).tolist(),[0,2])
    def test_joint_fit_separates_sensor_noise(self):
        rng=np.random.default_rng(99); z=rng.normal(size=2000)
        x=np.column_stack([z+rng.normal(size=2000)*.2,z+rng.normal(size=2000)*2])
        y=1.2*z+rng.normal(size=2000)*.4
        m=p.joint_fit(x,y,30)
        self.assertGreater(m['r'][1],m['r'][0]*5)
        self.assertGreater(m['beta'],.5)
    def test_null_gate_is_zero_for_harmful_predictions(self):
        self.assertEqual(p.gate(np.ones(80),np.zeros(80)),0.)
    def test_future_and_unreleased_values_cannot_change_fitted_prediction(self):
        (x,release,y),_=p.generate('reliability_shift',3)
        t=720
        model,_=p.fit_models(x,release,y,t,None)
        x2=x.copy(); x2[release>t]=1e9
        y2=y.copy(); y2[t:]=1e9
        changed,_=p.fit_models(x2,release,y2,t,None)
        np.testing.assert_array_equal(p.deploy_predict(x,release,t,model),
                                      p.deploy_predict(x2,release,t,changed))
    def test_deployment_forbidden_extra_inputs(self):
        with self.assertRaises(TypeError):
            p.joint_fit(np.ones((10,2)),np.ones(10),30,latent=np.ones(10))

if __name__=='__main__': unittest.main()
