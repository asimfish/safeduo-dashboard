import unittest
import numpy as np
from actual_state_arm_guard_v2 import ActualStateArmGuard,ARMS
from test_actual_state_guard_v1 import setup


def fixture():
    old,q,v,a,p,d=setup()
    guard=ActualStateArmGuard(old.gap_function,old.bounds,old.speed_caps,old.ids,old.retreat)
    previous={a:q[a][guard.ids[a]].astype(np.float32) for a in ARMS}
    velocity={a:np.zeros_like(previous[a]) for a in ARMS}
    return guard,q,v,a,p,d,previous,velocity


class SlewContracts(unittest.TestCase):
    def test_large_raw_jump_does_not_jump_actuator_target(self):
        g,q,v,a,p,d,previous,velocity=fixture();a['F_R'][:]=1.5
        r=g.decide(q,v,a,p,d,previous,velocity)
        delta=r.target['F_R']-previous['F_R']
        self.assertTrue((abs(delta)<=.5*(1/60.)**2+1e-8).all())

    def test_direction_reversal_has_bounded_command_acceleration(self):
        g,q,v,a,p,d,previous,velocity=fixture()
        velocity['F_R'][:]=.1;a['F_R'][:]=-1.
        r=g.decide(q,v,a,p,d,previous,velocity)
        nextv=(r.target['F_R']-previous['F_R'])*60.
        np.testing.assert_allclose(nextv,np.full(7,.1-.5/60.),atol=1e-7)

    def test_all_modes_preserve_slew_over_sequence(self):
        g,q,v,a,p,d,previous,velocity=fixture();rng=np.random.default_rng(12901)
        modes=set()
        for _ in range(150):
            a={arm:rng.uniform(-1.8,1.8,len(previous[arm])) for arm in ARMS}
            r=g.decide(q,v,a,p,d,previous,velocity);modes.add(r.mode)
            for arm in ARMS:
                nextv=(r.target[arm].astype(float)-previous[arm].astype(float))*60.
                self.assertTrue((abs(nextv)<=.1+1e-6).all())
                self.assertTrue((abs(nextv-velocity[arm])<=.5/60.+1e-6).all())
                velocity[arm]=nextv;previous[arm]=r.target[arm].copy()
        self.assertGreaterEqual(len(modes),2)

    def test_previous_actuator_state_is_required(self):
        g,q,v,a,p,d,previous,velocity=fixture();previous['F_R']=np.zeros(6)
        with self.assertRaises(ValueError):g.decide(q,v,a,p,d,previous,velocity)


if __name__=='__main__':unittest.main(verbosity=2)
