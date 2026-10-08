import unittest
import numpy as np
from scene_geometry import corners,pose_matrix,predict,NativeSceneGuard

class GeometryContracts(unittest.TestCase):
    def scene(self):
        names=['finger_joint'];meta={a:dict(joint_names=names,hand_ids=[0],body_names=['wrist_3_link','tip']) for a in ['U_L','U_R']}
        js=lambda:dict(name=names[0],parent='wrist_3_link',child='tip',axis='Z',local0=[0,0,0,0,0,0,1],local1=[0,0,0,0,0,0,1])
        cs=[]
        for a in meta:cs.append(dict(path=a+'/tip',arm=a,owner=a+'/tip',body='tip',hand=True,corners=corners([.995,-.005,-.005],[1.005,.005,.005]).tolist(),contact_offset_m=0))
        s=dict(joints={a:[js()] for a in meta},colliders=cs)
        links={'U_L':np.array([[0,0,1,0,0,0,1],[1,0,1,0,0,0,1]],float),'U_R':np.array([[5,0,1,0,0,0,1],[6,0,1,0,0,0,1]],float)}
        return s,links,meta,{'U_L':[0.],'U_R':[0.]}

    def test_midpath_collision_cannot_be_certified_by_endpoints(self):
        s,l,m,op=self.scene();s['colliders'].append(dict(path='fixture',owner=None,arm=None,hand=False,corners=corners([-.01,.99,.99],[.01,1.01,1.01]).tolist(),contact_offset_m=0))
        r=predict(s,'U_L',l,m,op,{'U_L':[np.pi],'U_R':[0.]},{},reserve=0)
        self.assertFalse(r['admitted']);self.assertEqual(r['witness']['obstacle'],'fixture');self.assertAlmostEqual(r['witness']['alpha'],.5,places=2)

    def test_clear_path_admits(self):
        s,l,m,op=self.scene();self.assertTrue(predict(s,'U_L',l,m,op,{'U_L':[.1],'U_R':[0.]},{},reserve=0)['admitted'])

    def test_other_hand_future_sweep_is_obstacle(self):
        s,l,m,op=self.scene();l['U_R'][0,:3]=[1,-1,1]
        self.assertFalse(predict(s,'U_L',l,m,op,{'U_L':[0.],'U_R':[np.pi]},{},reserve=0)['admitted'])

    def test_empty_or_nonfinite_geometry_rejected(self):
        for lo,hi in [([1,1,1],[-1,-1,-1]),([np.nan,0,0],[1,1,1]),([-3e38]*3,[3e38]*3)]:
            with self.assertRaises(ValueError):corners(lo,hi)

    def test_nonunit_pose_rejected(self):
        with self.assertRaises(ValueError):pose_matrix([0,0,0,0,0,0,0])

    def test_missing_obstacle_state_fails_closed(self):
        g=object.__new__(NativeSceneGuard)
        def missing(*args):raise KeyError('fixture state')
        g._check=missing
        self.assertEqual(g.check(0,'U_L',np.zeros(12),0)['reason'],'missing_or_invalid_scene_geometry')

if __name__=='__main__':unittest.main()
