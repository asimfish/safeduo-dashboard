"""Synthetic state/math tests only. No images generated or visually inspected."""
import copy
import unittest
import numpy as np

from astra_tracking_camera_core import ARMS, DOFS, frustum_planes, native_state_binding, select_groups


class FrustumTests(unittest.TestCase):
    K=np.array([[1.,0,1],[0,1.,1],[0,0,1]])

    def test_six_normalized_plane_units_and_radius(self):
        result=frustum_planes([[0,0,-2]],[.25],np.eye(4),self.K,[.5,10],2,2)
        expected=np.array([[np.sqrt(2)-.25]*4+[1.25,7.75]])
        np.testing.assert_allclose(result,expected,rtol=0,atol=1e-15)

    def test_every_cropped_direction_detected(self):
        points=[[-3,0,-2],[3,0,-2],[0,3,-2],[0,-3,-2],[0,0,-.1],[0,0,-11]]
        result=frustum_planes(points,np.zeros(6),np.eye(4),self.K,[.5,10],2,2)
        self.assertTrue((np.diag(result)<0).all())

    def test_world_row_transform_and_rotation_invariance(self):
        W=np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[4,5,6,1]],float)
        camera=np.array([[.25,.5,-2],[-.5,-.25,-4]],float)
        world=(np.c_[camera,np.ones(2)]@W)[:,:3]
        a=frustum_planes(world,[.1,.2],W,self.K,[.5,10],2,2)
        b=frustum_planes(camera,[.1,.2],np.eye(4),self.K,[.5,10],2,2)
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-15)

    def test_touching_plane_is_not_strictly_contained(self):
        margin=frustum_planes([[0,0,-1.5]],[1.],np.eye(4),self.K,[.5,10],2,2)
        self.assertEqual(margin[0,4],0.)
        self.assertFalse((margin>0).all())

    def test_nonfinite_or_invalid_optics_rejected(self):
        for radius,clip in [([-1],[.5,10]),([np.nan],[.5,10]),([.1],[10,.5])]:
            with self.assertRaises(ValueError):frustum_planes([[0,0,-2]],radius,np.eye(4),self.K,clip,2,2)


def native_fixture(t=3,e=1):
    steps,envs=8,2
    q=(np.arange(steps*envs*26,dtype=np.float32).reshape(steps,envs,26)/1024)
    qd=q/np.float32(2)
    initial=np.full((envs,26),.5,np.float32)
    controller=np.stack([initial+np.float32((k+1)/64) for k in range(steps)])
    stream=np.concatenate((np.repeat(initial[None],6,axis=0),controller))
    pending=np.stack([stream[k:k+6] for k in range(steps)])
    dense=dict(q=q,pre_qd_compact=qd,controller_target=controller,actuator_target=stream[:steps],
               pre_pending_actuator_targets=pending,q_initial=np.full((envs,26),-1,np.float32))
    before={};s=dict(env_id=e,step=t,fresh_native_controlled_joint_indices={},fresh_native_all_joint_names={},
        fresh_native_selected={},arms={},controller_target={},actuator_target={},pending_actuator_targets=[])
    velocity=qd[t+1] if t+1<steps else np.full((envs,26),.3125,np.float32)
    start=0
    for arm,n in zip(ARMS,DOFS):
        aq=np.zeros((envs,n+2),np.float32);av=aq.copy()
        aq[:,:n]=q[t,:,start:start+n];av[:,:n]=velocity[:,start:start+n]
        before[arm+'_q']=aq;before[arm+'_qd']=av
        before[arm+'_root']=np.zeros((envs,7),np.float32)
        before[arm+'_root_vel']=np.zeros((envs,6),np.float32)
        s['fresh_native_controlled_joint_indices'][arm]=list(range(n))
        s['fresh_native_all_joint_names'][arm]=[f'joint{i}' for i in range(n+2)]
        s['fresh_native_selected'][arm]={k:before[arm+'_'+k][e].tolist() for k in ('q','qd','root','root_vel')}
        s['arms'][arm]=dict(q=aq[e,:n].tolist(),qd=av[e,:n].tolist())
        s['controller_target'][arm]=controller[t,e,start:start+n].tolist()
        s['actuator_target'][arm]=stream[t,e,start:start+n].tolist()
        start+=n
    post=np.concatenate((pending[t,1:,e],controller[t,e][None]))
    for target in post:
        start=0;entry={}
        for arm,n in zip(ARMS,DOFS):entry[arm]=target[start:start+n].tolist();start+=n
        s['pending_actuator_targets'].append(entry)
    return s,before,{k:v.copy() for k,v in before.items()},dense


class NativeBindingTests(unittest.TestCase):
    def test_exact_all_native_and_own_state_bindings(self):
        result=native_state_binding(*native_fixture())
        self.assertEqual(result['native_all_env_fields_bound'],16)
        self.assertEqual(result['native_all_env_count'],2)
        self.assertTrue(result['post_qd_next_pre_available'])

    def test_initial_target_is_not_assumed_q_initial(self):
        result=native_state_binding(*native_fixture(t=0))
        self.assertEqual(result['post_pending'][0],[.5]*26)

    def test_final_post_velocity_only_native_no_missing_dense_endpoint(self):
        result=native_state_binding(*native_fixture(t=7))
        self.assertFalse(result['post_qd_next_pre_available'])
        self.assertTrue(result['final_post_qd_only_native_when_no_next_pre'])
        self.assertEqual(result['qd'],[.3125]*26)

    def test_change_in_uncaptured_environment_detected(self):
        s,b,a,d=native_fixture();a['F_L_root_vel'][0,0]=np.float32(1)
        with self.assertRaisesRegex(ValueError,'render changed native'):native_state_binding(s,b,a,d)

    def test_wrong_current_applied_and_queue_phase_detected(self):
        for kind in ('issued','applied','post_queue'):
            s,b,a,d=native_fixture()
            if kind=='issued':s['controller_target']=copy.deepcopy(s['actuator_target'])
            elif kind=='applied':s['actuator_target']=copy.deepcopy(s['controller_target'])
            else:s['pending_actuator_targets'][-1]=copy.deepcopy(s['pending_actuator_targets'][0])
            with self.assertRaises(ValueError):native_state_binding(s,b,a,d)

    def test_next_pre_velocity_mismatch_detected_without_tolerance(self):
        s,b,a,d=native_fixture();d['pre_qd_compact'][4,1,0]=np.nextafter(d['pre_qd_compact'][4,1,0],np.float32(np.inf))
        with self.assertRaisesRegex(ValueError,'next pre velocity'):native_state_binding(s,b,a,d)


class SelectionTests(unittest.TestCase):
    def fixture(self):
        selection=dict(modes=['joint_reference','delay_reserve'],scheduled_groups=[dict(env=0,step=75),dict(env=56,step=480),dict(env=40,step=959)])
        receipts=[dict(capture_kind='scheduled',env_id=p['env'],step=p['step']) for p in selection['scheduled_groups']]
        return selection,receipts

    def test_three_scheduled_plus_earliest_failure_tie_lowest_env(self):
        s,r=self.fixture()
        r.extend([dict(capture_kind='first_failure',env_id=e,step=t) for e,t in [(0,60),(7,20),(3,20)]])
        chosen=select_groups(s,r,'joint_reference')
        self.assertEqual(len(chosen),4)
        self.assertEqual((chosen[-1]['step'],chosen[-1]['env_id']),(20,3))

    def test_no_failure_is_not_replaced_by_other_scheduled_group(self):
        s,r=self.fixture();self.assertEqual(len(select_groups(s,r,'delay_reserve')),3)

    def test_missing_or_duplicate_preselected_group_rejected(self):
        s,r=self.fixture()
        for bad in (r[:-1],r+[r[0]]):
            with self.assertRaisesRegex(ValueError,'missing/duplicated'):select_groups(s,bad,'joint_reference')


if __name__=='__main__':unittest.main(verbosity=2)
