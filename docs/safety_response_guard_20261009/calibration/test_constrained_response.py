import unittest
import numpy as np
from constrained_response import advance_constrained

class NativeStopResponseTests(unittest.TestCase):
    def test_stop_reaction_changes_other_joint_velocity(self):
        # Analytic constrained momentum: v0=0, M10*v0+M11*v1=-1 => v1=-.5.
        q=np.array([[0.,1.]]);v=np.array([[-1.,0.]])
        nq,nv,info=advance_constrained(q,v,q,np.array([[[2.,1.],[1.,2.]]]),np.zeros_like(q),np.zeros_like(q),np.ones_like(q),np.full_like(q,10),np.zeros_like(q),np.array([[[0.,10.],[-10.,10.]]]),.1)
        np.testing.assert_allclose(nv,[[0.,-.5]],atol=1e-9);np.testing.assert_allclose(nq,[[0.,.95]],atol=1e-9)
        self.assertTrue(info['converged'].all())
        # Old independent velocity clipping cannot satisfy this coupled stop reaction.
        self.assertGreater(np.linalg.norm(np.clip(v,[[0,-10]],[[10,10]])-nv),.49)
    def test_saturated_drive_and_upper_stop(self):
        q=np.zeros((1,1));one=np.ones((1,1))
        nq,nv,info=advance_constrained(q,q,one,np.ones((1,1,1)),100*one,q,2*one,10*one,q,np.array([[[0,.01]]]),.1)
        np.testing.assert_allclose(nq,[[.01]],atol=1e-10);np.testing.assert_allclose(nv,[[.1]],atol=1e-10)
        np.testing.assert_allclose(info['torque'],[[2]],atol=1e-10);self.assertTrue(info['converged'].all())
    def test_free_implicit_drive_analytic_solution(self):
        q=np.zeros((1,1));one=np.ones((1,1))
        nq,nv,info=advance_constrained(q,q,one,np.ones((1,1,1)),10*one,2*one,100*one,100*one,q,np.array([[[-10,10]]]),.1)
        np.testing.assert_allclose(nv,[[1/1.3]],atol=1e-9);self.assertTrue(info['converged'].all())
    def test_infeasible_stop_velocity_box_refused(self):
        q=np.array([[-1.]]);zero=np.zeros_like(q);one=np.ones_like(q)
        with self.assertRaisesRegex(ValueError,'infeasible'):
            advance_constrained(q,zero,q,np.ones((1,1,1)),one,one,one,one,zero,np.array([[[0,10]]]),.1)

if __name__=='__main__':unittest.main()
