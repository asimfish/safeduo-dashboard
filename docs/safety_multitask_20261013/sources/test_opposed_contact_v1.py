"""Observable contact opposition, including the actual long-beam regression."""
import os
import unittest
import numpy as np


def legacy_opposed(bodies, normals, points, forces):
    for i in range(len(bodies)):
        for j in range(len(bodies)):
            if bodies[i] != bodies[j] and abs(forces[i]) > .1 and abs(forces[j]) > .1:
                if normals[i] @ normals[j] <= -.5 and points[i] @ points[j] < 0:
                    return True
    return False


if os.environ.get('SAFEDUO_CONTACT_REGRESSION_OLD') == '1':
    opposed = legacy_opposed
else:
    from opposed_contact_v1 import opposed_contact as opposed


class ContactTests(unittest.TestCase):
    def test_long_beam_opposite_faces_at_same_end(self):
        # Rounded actual native105 contacts: both at y<0, opposite X faces.
        self.assertTrue(opposed(['index', 'thumb'],
            np.array([[-.977, .019, .210], [1., -.010, -.009]]),
            np.array([[-.0221, -.2386, .0402], [.0173, -.2764, .0154]]),
            np.array([-1.8489, -2.988])))

    def test_translation_does_not_change_local_opposition(self):
        n=np.array([[-1.,0,0],[1.,0,0]])
        p=np.array([[-.02,-.25,0],[.02,-.25,0]])
        for translation in [np.zeros(3),np.array([100.,30.,-50.])]:
            self.assertTrue(opposed(['index','thumb'],n,p+translation,np.ones(2)))

    def test_parallel_normals_not_opposed(self):
        self.assertFalse(opposed(['index','thumb'],np.array([[1.,0,0],[1.,0,0]]),
                                np.array([[-.02,0,0],[.02,0,0]]),np.ones(2)))

    def test_same_body_or_weak_contact_not_grasp(self):
        n=np.array([[-1.,0,0],[1.,0,0]]);p=np.array([[-.02,0,0],[.02,0,0]])
        self.assertFalse(opposed(['thumb','thumb'],n,p,np.ones(2)))
        self.assertFalse(opposed(['index','thumb'],n,p,np.array([.01,1.])))

    def test_coincident_or_reversed_geometry_not_grasp(self):
        n=np.array([[-1.,0,0],[1.,0,0]])
        self.assertFalse(opposed(['index','thumb'],n,np.zeros((2,3)),np.ones(2)))
        self.assertFalse(opposed(['index','thumb'],n,np.array([[.02,0,0],[-.02,0,0]]),np.ones(2)))


if __name__ == '__main__': unittest.main()
