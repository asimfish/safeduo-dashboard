import unittest
import numpy as np
import torch
from predictors import Predictor, nonnegative_fit

class Tests(unittest.TestCase):
    def test_inertia_can_close_with_opening_target(self):
        q=torch.zeros(2,26); v=torch.full_like(q,-1); target=torch.full_like(q,.1)
        p=Predictor('empirical', np.tile([.8,.01],(26,1)))
        out=p.predict(q,v,target,.02)
        torch.testing.assert_close(out,torch.full_like(q,-.015))
        self.assertTrue((Predictor('target_snap').predict(q,v,target,.02)>0).all())

    def test_observable_baselines(self):
        q=torch.arange(26,dtype=torch.float64)[None];v=torch.full_like(q,2);t=q+.5
        torch.testing.assert_close(Predictor('velocity').predict(q,v,t,.1),torch.full_like(q,.2))
        torch.testing.assert_close(Predictor('static').predict(q,v,t,.1),torch.zeros_like(q))
        torch.testing.assert_close(Predictor('target_snap').predict(q,v,t,.1),torch.full_like(q,.5))

    def test_fit_known_system_and_boundary(self):
        x=np.array([[1,0],[0,1],[1,1],[-1,2]],float)
        y=x@np.array([.8,.03]);c=nonnegative_fit(x.T@x,x.T@y,y@y)
        np.testing.assert_allclose(c,[.8,.03],atol=1e-12)
        x=np.eye(2);y=np.array([-1.,2.]);np.testing.assert_equal(nonnegative_fit(x.T@x,x.T@y,y@y),[0.,2.])

    def test_invalid_and_unidentifiable(self):
        with self.assertRaises(ValueError):nonnegative_fit(np.ones((2,2)),np.ones(2),1.)
        q=torch.zeros(1,26)
        with self.assertRaises(ValueError):Predictor('velocity').predict(q,q,q,0.)
        q[0,0]=float('nan')
        with self.assertRaises(ValueError):Predictor('velocity').predict(q,q,q,.02)

if __name__=='__main__':unittest.main()
