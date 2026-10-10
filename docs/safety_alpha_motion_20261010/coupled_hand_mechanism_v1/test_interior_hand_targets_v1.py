import unittest
import numpy as np
from interior_hand_targets_v1 import filtered_hand_targets


class InteriorTargetTests(unittest.TestCase):
    def setUp(self):
        self.names=['right_thumb_1_joint','right_thumb_2_joint','right_thumb_3_joint','right_thumb_4_joint','right_index_1_joint']
        self.limits=np.array([[0,1],[0,1],[0,.567],[0,.396],[0,1.]])
        self.previous=np.full(5,.03);self.dt=.016666

    def test_backoff_cannot_lower_loaded_distal_target_to_stop(self):
        target=self.previous.copy()
        for _ in range(12):target=filtered_hand_targets(self.names,np.full(5,.03),target,self.limits,self.dt,True)
        self.assertAlmostEqual(target[2],.1);self.assertAlmostEqual(target[3],.0792)
        self.assertAlmostEqual(target[0],.03);self.assertAlmostEqual(target[4],.03)

    def test_release_can_reach_real_open_pose_without_target_jump(self):
        target=np.full(5,.2)
        for _ in range(30):
            before=target.copy();target=filtered_hand_targets(self.names,self.previous,target,self.limits,self.dt,False)
            self.assertLessEqual(abs(target-before).max(),.6*self.dt+1e-12)
        np.testing.assert_allclose(target,self.previous)

    def test_finger_targets_do_not_receive_distal_thumb_floor(self):
        target=filtered_hand_targets(self.names,self.previous,self.previous,self.limits,self.dt,True)
        np.testing.assert_allclose(target[[0,1,4]],self.previous[[0,1,4]])

    def test_invalid_previous_target_is_rejected(self):
        previous=self.previous.copy();previous[2]=-.001
        with self.assertRaises(ValueError):filtered_hand_targets(self.names,self.previous,previous,self.limits,self.dt,True)


if __name__=='__main__':unittest.main()
