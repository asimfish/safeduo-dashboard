"""Storage isolation contracts; not an Isaac integration test."""
from types import SimpleNamespace
import unittest
import numpy as np
from native_state_capture import capture_native,ARM_KEYS


class NativeCopy(unittest.TestCase):
    def setUp(self):
        self.q=np.ones((2,9),dtype=np.float32)
        self.root=np.zeros((2,7),dtype=np.float32)
        view=SimpleNamespace(get_dof_positions=lambda:self.q,get_dof_velocities=lambda:self.q,
                             get_root_transforms=lambda:self.root,get_root_velocities=lambda:self.root[:,:6])
        targets={a:np.zeros((2,6),dtype=np.float32) for a in ARM_KEYS}
        self.env=SimpleNamespace(_arms={a:SimpleNamespace(root_physx_view=view) for a in ARM_KEYS},
            _joint_idx={a:np.arange(6) for a in ARM_KEYS},_targets=targets,
            _evaluation_actuator_delay=SimpleNamespace(queue=SimpleNamespace(pending=[targets for _ in range(6)])))

    def test_full_getter_state_and_fifo_copy(self):
        out=capture_native(self.env,7,'pre_physics')
        self.q[:]=9;self.root[:]=3;self.env._targets['F_L'][:]=8
        np.testing.assert_array_equal(out['F_L_native_q'],np.ones((2,9)))
        np.testing.assert_array_equal(out['F_L_native_root_xyzw'],np.zeros((2,7)))
        np.testing.assert_array_equal(out['F_L_pending_controlled_targets'],np.zeros((6,2,6)))

    def test_capture_does_not_mutate_input(self):
        capture_native(self.env,7,'post_physics_pre_render')
        np.testing.assert_array_equal(self.q,np.ones((2,9)))

    def test_bad_boundary_or_incomplete_queue_rejected(self):
        with self.assertRaises(ValueError):capture_native(self.env,0,'during_render')
        self.env._evaluation_actuator_delay.queue.pending.pop()
        with self.assertRaises(ValueError):capture_native(self.env,0,'pre_physics')


if __name__=='__main__':unittest.main(verbosity=2)
