"""Free-coordinate energy and boundary oracles; not native safety coverage."""
import unittest
import numpy as np
from stable_hand_motor_effort_v2 import stable_motor_torque


class StableMotorEffortTests(unittest.TestCase):
    def test_rest_is_exact_equilibrium(self):
        np.testing.assert_array_equal(stable_motor_torque([.4,.1],[.4,.1],[0,0],[1e-4,2e-4],5,.5,.00208325),[0,0])

    def test_free_implicit_step_energy_dissipates(self):
        rng = np.random.default_rng(3201)
        mass = np.exp(rng.uniform(-14,1,2000))
        stiffness = np.exp(rng.uniform(-5,7,2000))
        damping = np.exp(rng.uniform(-7,4,2000))
        q,v,target = rng.normal(size=(3,2000))
        for dt in [1e-5, .00208325, .1]:
            torque = stable_motor_torque(target,q,v,mass,stiffness,damping,dt)
            vn = v+dt*torque/mass
            qn = q+dt*vn
            before = .5*mass*v*v+.5*stiffness*(target-q)**2
            after = .5*mass*vn*vn+.5*stiffness*(target-qn)**2
            dissipation = dt*damping*vn*vn+.5*mass*(vn-v)**2+.5*stiffness*(qn-q)**2
            np.testing.assert_allclose(after-before,-dissipation,rtol=2e-9,atol=2e-9)
            self.assertTrue(np.all(after<=before+2e-9))
    def test_damping_only_opposes_velocity(self):
        velocity = np.linspace(-10,10,101)
        torque = stable_motor_torque(0,0,velocity,1e-4,0,.5,.002)
        self.assertTrue(np.all(torque*velocity<=0))

    def test_zero_gains_yield_zero_torque(self):
        np.testing.assert_array_equal(stable_motor_torque([1,2],[-1,-2],[5,-5],.001,0,0,.002),[0,0])

    def test_invalid_physical_input_rejected(self):
        for inertia in [0,-1,np.nan,np.inf]:
            with self.assertRaises(ValueError):stable_motor_torque(1,0,0,inertia,5,.5,.002)
        for dt in [0,-1,np.nan,np.inf,True]:
            with self.assertRaises(ValueError):stable_motor_torque(1,0,0,.001,5,.5,dt)
        for position in [np.nan,np.inf]:
            with self.assertRaises(ValueError):stable_motor_torque(1,position,0,.001,5,.5,.002)

    def test_vector_values_match_separate_motor_energy_oracle(self):
        q=np.array([.1,.2,.3]);target=np.array([.2,.1,.6]);v=np.array([-.1,.4,.3]);m=np.array([1e-4,2e-4,3e-4])
        dt=.00208325;k=5;d=.5
        tau=stable_motor_torque(target,q,v,m,k,d,dt)
        vn=v+dt*tau/m
        np.testing.assert_allclose(m*(vn-v)/dt,k*(target-q-dt*vn)-d*vn,rtol=1e-12,atol=1e-12)


if __name__=='__main__':
    unittest.main()
