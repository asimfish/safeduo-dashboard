"""Behavioral algebra tests, not native or physical safety qualification."""
import unittest
import numpy as np
from contact_disturbance_motor_effort_v1 import observed_disturbance,disturbance_backward_euler_torque
from fourhand_native_probe_v55.coupled_implicit_motor_effort_v1 import coupled_backward_euler_torque


class DisturbanceTests(unittest.TestCase):
    def test_zero_disturbance_preserves_free_controller_exactly(self):
        rng=np.random.default_rng(2026101156)
        for n in (1,6,13):
            B=rng.normal(size=(n,n));M=B.T@B+np.eye(n)*.002
            args=(M,np.eye(n),rng.normal(size=n),rng.normal(size=n),rng.normal(size=n),rng.uniform(1,100,n),rng.uniform(.1,15,n),.001041625)
            old=coupled_backward_euler_torque(*args)
            new=disturbance_backward_euler_torque(*args,np.zeros(n))
            np.testing.assert_array_equal(new.torque,old.torque)
            np.testing.assert_array_equal(new.predicted_velocity,old.predicted_free_velocity)

    def test_blocked_static_load_has_nominal_spring_equilibrium(self):
        # Ideal fixture oracle: zero acceleration, external reaction = -K*error.
        M=np.array([[2e-5,1e-6],[1e-6,3e-5]]);K=np.array([20.,30.]);D=np.array([2.,3.]);error=np.array([.02,.01]);h=.001041625
        result=disturbance_backward_euler_torque(M,np.eye(2),error,np.zeros(2),np.zeros(2),K,D,h,-K*error)
        np.testing.assert_allclose(result.torque,K*error,atol=2e-15,rtol=0)
        np.testing.assert_allclose(result.predicted_velocity,0.,atol=2e-15,rtol=0)
        np.testing.assert_allclose(M@result.predicted_velocity/h,result.torque-K*error,atol=2e-15,rtol=0)

    def test_observer_uses_applied_clipped_torque_and_masks_arm(self):
        M=np.array([[.3,.02],[.02,.001]]);v0=np.array([.01,.02]);v1=np.array([.04,0.]);h=.001
        applied=np.array([.5,.03]);g=np.array([.2,.001]);c=np.array([.01,-.002]);mask=np.array([False,True])
        full=M@((v1-v0)/h)-applied+g+c
        result=observed_disturbance(M,v0,v1,applied,g,c,h,mask)
        self.assertEqual(result[0],0.);self.assertEqual(result[1],full[1])
        different=observed_disturbance(M,v0,v1,np.array([.5,3.]),g,c,h,mask)
        self.assertAlmostEqual(different[1]-result[1],-2.97)

    def test_blocked_scalar_feedback_converges_without_force_floor(self):
        M=2e-5;K=20.;D=2.;h=.001041625;e=.02;reaction=0.;last_error=None
        for _ in range(1000):
            r=disturbance_backward_euler_torque(np.array([[M]]),np.eye(1),np.array([e]),np.zeros(1),np.zeros(1),np.array([K]),np.array([D]),h,np.array([reaction]))
            reaction=-r.torque[0] # Ideal rigid stop, actual velocity remains zero.
            error=abs(K*e+reaction)
            if last_error is not None:self.assertLessEqual(error,last_error+1e-15)
            last_error=error
        self.assertLess(last_error,1e-4)

    def test_rejects_nonfinite_or_wrong_mask_inputs(self):
        with self.assertRaises(ValueError):observed_disturbance(np.eye(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),.001,np.ones(1))
        with self.assertRaises(ValueError):disturbance_backward_euler_torque(np.eye(1),np.eye(1),np.zeros(1),np.zeros(1),np.zeros(1),np.ones(1),np.ones(1),.001,np.array([np.nan]))


if __name__=='__main__':unittest.main()
