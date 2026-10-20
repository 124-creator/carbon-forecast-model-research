import unittest
import numpy as np

class CoreTest(unittest.TestCase):
    def test_segrnn_shape_translation(self):
        try:
            from model import initialize, predict
        except ImportError:
            self.fail('missing self-owned SegRNN implementation')
        p=initialize(7)
        x=np.arange(120.).reshape(2,60)/100
        g=np.ones_like(x)
        a,_,_=predict(p,x,g,'original')
        b,_,_=predict(p,x+9,g,'original')
        self.assertEqual(a.shape,(2,3))
        np.testing.assert_allclose(b,a,atol=1e-12)

    def test_gradient_each_parameter_group(self):
        from model import initialize, loss
        from autograd import grad
        from autograd.misc.flatten import flatten
        p=initialize(7)
        x=np.random.RandomState(8).normal(size=(4,60))
        g=np.ones_like(x); y=np.ones((4,3)); k=np.tile([0,1,2],(4,1))
        reps=np.tile([.25,.5,1.],(3,1))
        f,unflat=flatten(p)
        fun=lambda v:loss(unflat(v),x,g,y,k,'marginal',reps)
        derivative=unflat(grad(fun)(f)); rng=np.random.RandomState(9)
        for name,value in p.items():
            direction=rng.normal(size=value.shape); direction/=np.linalg.norm(direction)
            plus={a:b.copy() for a,b in p.items()}; minus={a:b.copy() for a,b in p.items()}
            plus[name]+=1e-5*direction; minus[name]-=1e-5*direction
            fd=(loss(plus,x,g,y,k,'marginal',reps)-loss(minus,x,g,y,k,'marginal',reps))/2e-5
            analytical=np.sum(derivative[name]*direction)
            self.assertLess(abs(fd-analytical),1e-5,name)

if __name__=='__main__': unittest.main()
