import unittest
import numpy as np
from operation_metrics import safe_operation,pairwise_joint_visits

class Metrics(unittest.TestCase):
    def test_unsafe_jump_is_excluded_but_safe_duration_is_reported(self):
        q0=np.zeros((2,26));q=np.array([np.full((2,26),x) for x in [.1,.2,.9]])
        margins=np.ones((3,2,4));margins[2,0,3]=-.001
        limits=np.broadcast_to([0.,1.],(2,26,2)).copy()
        r=safe_operation(q0,q,margins,limits,.1)
        self.assertEqual(r['safe_transitions_before_first_violation'],[2,3])
        self.assertAlmostEqual(r['mean_safe_duration_s'],.25)
        self.assertAlmostEqual(r['mean_safe_prefix_joint_range'],.55)
        self.assertAlmostEqual(r['mean_safe_prefix_joint_path_rad'],14.3)

    def test_pairwise_grid_is_marginal_and_outside_samples_do_not_add_cells(self):
        q0=np.zeros((1,26));q=np.array([np.full((1,26),x) for x in [.15,1.,2.]])
        limits=np.broadcast_to([0.,1.],(1,26,2)).copy()
        r=pairwise_joint_visits(q0,q,limits)
        self.assertEqual(r['joint_pairs'],325);self.assertEqual(len(r['groups']),10)
        self.assertTrue(all(x['cells']==3 for x in r['rows']))
        self.assertEqual(r['outside_soft_joint_sample_fraction'],.25)

if __name__=='__main__':unittest.main()
