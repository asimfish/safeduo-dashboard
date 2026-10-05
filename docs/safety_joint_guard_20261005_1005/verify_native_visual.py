"""Independent offline equality, optics, frustum and own-state binding for two replays."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
RAW=Path('/mnt/nas/data/lyf/double_hand')/HERE.name
ARMS=['F_L','F_R','U_L','U_R']
SLICES=[slice(0,7),slice(7,14),slice(14,20),slice(20,26)]


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def one(mode):
    root=RAW/'visual'/mode
    protocol=json.loads((root/'visual_protocol.json').read_text())
    reg=json.loads((HERE/'visual_registration_v5.json').read_text())
    assert protocol['status']=='complete' and protocol['completed_windows']==64 and protocol['steps']==960
    assert protocol['observer_sha256']==reg['visual_runner_sha256']==sha(HERE/'visual_runner_v5.py')
    assert protocol['actor_sha256']==json.loads((HERE/'holdout_plan_v2.json').read_text())['checkpoint_sha256']
    d=load(root/'cell_001.npz');assert d['q'].shape==(960,64,26)
    for v in d.values():
        if v.dtype.kind=='f':assert np.isfinite(v).all()
    assert np.array_equal(d['actuator_target'],np.concatenate([np.repeat(d['q_initial'][None],6,0),d['controller_target'][:-6]]))
    numerical=load(RAW/'holdout'/f'{mode}_{reg["command_seed"]}/cell_001.npz')
    assert np.array_equal(d['q_initial'],numerical['q_initial']) and np.array_equal(d['cmd'],numerical['cmd'])
    identity=json.loads((root/'full_row_identity.json').read_text())
    physical=np.array(identity['class_id']);classes=physical.copy()
    sphere_arm=np.array(identity['sphere_arm_id']);pair=np.array(identity['pair_sphere_idx']);radii=np.array(identity['sphere_radii_m'])
    assert identity['rows']==9021 and physical.shape==(9021,) and pair.shape==(9021,2)
    assert len(sphere_arm)==len(radii)==len(identity['sphere_names']) and (radii>0).all()
    classes[(physical==1)&(sphere_arm[pair[:,0]]>=2)]=2;classes[physical==2]=3
    receipts=json.loads((root/'camera_receipts.json').read_text())
    assert receipts['groups']==21 and receipts['PNG']==189 and len(receipts['receipts'])==21
    expected_groups={(e,t) for e in reg['slots'] for t in reg['capture_steps']}
    actual_groups=[(x['env_id'],x['step']) for x in receipts['receipts']]
    assert len(set(actual_groups))==21 and set(actual_groups)==expected_groups
    expected_views={'overview','front','reverse','f_pair','f_opposite_low','f_opposite_high','u_pair','u_opposite_low','u_opposite_high'}
    assert len(reg['slots'])==7 and len(reg['capture_steps'])==3
    images=0;views=0;stale=0;details=[];all_image_paths=set();minimum_center_depth=np.inf;maximum_sphere_depth=-np.inf
    raw_chunks={}
    for item in receipts['receipts']:
        state_file=root/item['state'];assert sha(state_file)==item['sha256']
        s=json.loads(state_file.read_text());e,t=s['env_id'],s['step']
        assert s['visual_mode']==mode and s['environment_count']==64 and s['visibility_only']
        assert (e,t)==(item['env_id'],item['step'])
        before_file=root/s['fresh_native_snapshot'];after_file=root/s['fresh_native_after_snapshot']
        assert sha(before_file)==s['fresh_native_before_sha256'] and sha(after_file)==s['fresh_native_after_sha256']
        before=load(before_file);after=load(after_file);assert set(before)==set(after)
        for key in before:
            assert before[key].shape[0]==64 and np.isfinite(before[key]).all()
            assert np.array_equal(before[key],after[key]),(mode,t,e,key)
        for arm,sl in zip(ARMS,SLICES):
            idx=s['fresh_native_controlled_joint_indices'][arm];assert len(idx)==sl.stop-sl.start
            assert all(isinstance(i,int) and not isinstance(i,bool) for i in idx) and len(set(idx))==len(idx)
            assert min(idx)>=0 and max(idx)<before[arm+'_q'].shape[1]
            names=s['fresh_native_all_joint_names'][arm];assert len(names)==before[arm+'_q'].shape[1]
            q=before[arm+'_q'][:,idx];qd=before[arm+'_qd'][:,idx]
            assert np.array_equal(q,d['q'][t,:,sl])
            assert np.array_equal(q[e],np.array(s['arms'][arm]['q'],q.dtype))
            assert np.array_equal(qd[e],np.array(s['arms'][arm]['qd'],qd.dtype))
            if t<959:assert np.array_equal(qd,d['pre_qd_compact'][t+1,:,sl])
            for key,value in s['fresh_native_selected'][arm].items():
                assert np.array_equal(np.array(value,before[arm+'_'+key].dtype),before[arm+'_'+key][e])
            for key in ['controller_target','actuator_target']:
                assert np.array_equal(np.array(s[key][arm],np.float32),d[key][t,e,sl])
        centers=np.array(s['sphere_centers_world_m'],np.float32)
        assert np.array_equal(centers,d['sphere_centers'][t,e])
        raw=np.array(s['full_distances_m'],np.float32);exempt=np.array(s['full_exempt'],bool)
        assert raw.shape==(9021,) and np.isfinite(raw).all()
        measured=np.array([raw[(classes==c)&~exempt].min() for c in range(4)])
        assert np.array_equal(measured,d['official_margins'][t,e])
        assert np.array_equal(np.array(s['sphere_radii_m']),radii)
        assert s['full_row_identity_file']=='full_row_identity.json'
        for key in ['full_braking_dmin_m','full_closing_m_s']:
            value=np.array(s[key],np.float32);assert value.shape==(9021,) and np.isfinite(value).all()
        queue=np.stack([
            np.concatenate([np.array(v[a],np.float32) for a in ARMS]) for v in s['pending_actuator_targets']])
        assert queue.shape==(6,26)
        if t<959:
            assert np.array_equal(queue,d['pre_pending_actuator_targets'][t+1,:,e])
            start=((t+1)//32)*32
            if start not in raw_chunks:raw_chunks[start]=load(root/'full_forecast'/f'steps_{start:04d}_{start+32:04d}.npz')
            chunk=raw_chunks[start];pos=t+1-start
            assert np.array_equal(raw,chunk['measured_d'][pos,e])
            assert np.array_equal(np.array(s['full_braking_dmin_m'],np.float32),chunk['dmin'][pos,e])
            assert np.array_equal(exempt,np.unpackbits(chunk['exempt'][pos,e])[:9021].astype(bool))
        else:
            assert np.array_equal(queue,d['controller_target'][954:960,e])
        assert set(s['views'])==expected_views
        image_names=[Path(x['path']).stem.split('_',2)[-1] for x in item['images']]
        assert len(image_names)==9 and set(image_names)==expected_views
        for name,v in s['views'].items():
            matrix=np.array(v['actual_camera_to_world_row_matrix']);pos=np.array(v['actual_position_world_m'])
            assert np.isfinite(matrix).all() and np.allclose(pos,np.array(v['eye'])+s['origin_world_m'],atol=1e-5,rtol=0)
            assert np.allclose(matrix[3,:3],pos,atol=1e-8,rtol=0)
            assert np.allclose(matrix[:3,:3]@matrix[:3,:3].T,np.eye(3),atol=1e-5)
            K=np.array(v['actual_intrinsic_matrix']);opt=v['actual_usd_optics']
            fx=1280*opt['focal_length']/opt['horizontal_aperture'];fy=720*opt['focal_length']/opt['vertical_aperture']
            expected=np.array([[fx,0,640+fx*opt['horizontal_aperture_offset']],
                               [0,fy,360+fy*opt['vertical_aperture_offset']],[0,0,1]])
            assert np.allclose(K,expected,atol=2e-4,rtol=0)
            assert opt['horizontal_aperture_offset']==0 and opt['vertical_aperture_offset']==0
            mask=sphere_arm<2 if name.startswith('f_') else sphere_arm>=2 if name.startswith('u_') else np.ones(len(sphere_arm),bool)
            pts=np.c_[centers[mask],np.ones(mask.sum())]@np.linalg.inv(matrix)
            x,y=pts[:,0],pts[:,1];depth=-pts[:,2];r=radii[mask]
            fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
            margins=np.stack([(fx*x+cx*depth)/np.hypot(fx,cx)-r,((1280-cx)*depth-fx*x)/np.hypot(fx,1280-cx)-r,
                (cy*depth-fy*y)/np.hypot(fy,cy)-r,((720-cy)*depth+fy*y)/np.hypot(fy,720-cy)-r,depth-r],-1)
            assert (margins>0).all() and v['observed_sphere_frustum']['all_contained']
            assert abs(float(margins.min())-v['observed_sphere_frustum']['minimum_plane_margin_m'])<1e-6
            assert margins.min()>=.050-2e-6
            minimum_center_depth=min(minimum_center_depth,float((depth-r).min()))
            maximum_sphere_depth=max(maximum_sphere_depth,float((depth+r).max()))
            pixels=np.c_[fx*x/depth+cx,cy-fy*y/depth]
            assert np.allclose(pixels.min(0),v['observed_center_projection']['pixel_min'],atol=1e-5,rtol=0)
            assert np.allclose(pixels.max(0),v['observed_center_projection']['pixel_max'],atol=1e-5,rtol=0)
            stale+=not v['sdk_position_matches_actual'];views+=1
        for image in item['images']:
            assert image['path'] not in all_image_paths;all_image_paths.add(image['path'])
            file=root/image['path'];assert sha(file)==image['sha256']
            with Image.open(file) as im:assert im.size==(1280,720) and np.array(im).std()>5
            images+=1
        details.append(dict(mode=mode,env=e,step=t,all64_native_before_after_exact=True,nine_actual_calibrated_views=True))
    keys=['q','ee','cmd','exec','alpha','official_margins','controller_target','actuator_target']
    equality={key:bool(np.array_equal(d[key],numerical[key])) for key in keys}
    differences={}
    for key in keys:
        bad=(d[key]!=numerical[key]).reshape(960,-1).any(-1)
        differences[key]=dict(first_different_step=int(np.flatnonzero(bad)[0]) if bad.any() else None,
            maximum_absolute_difference=float(np.abs(d[key].astype(np.float64)-numerical[key]).max()))
    return dict(mode=mode,full_steps=960,replay_windows=64,groups=21,images=images,actual_calibrated_views=views,
        native_before_files=3,native_after_files=21,recorded_stale_sdk_views=stale,
        identity_sha256=sha(root/'full_row_identity.json'),exact_registered_group_and_view_coverage=True,
        side_and_positive_depth_planes='5planes verified; actual clipping_range not captured',
        observed_sphere_depth_min_m=minimum_center_depth,observed_sphere_depth_max_m=maximum_sphere_depth,
        dmin_and_exempt_binding='71/75 to next full pre chunk; terminal producer values finite only',
        full_closing_scope='producer values finite and source bound, no full offline velocity oracle',
        all64_native_q_bound_to_dense_trace=True,all64_qd_bound_to_next_pre_steps=[71,75],
        final_qd_binding='full native before/after exact; selected slots linked to capture state; no next dense frame',
        numerical_binding_status='EXACT' if all(equality.values()) else 'FAIL_EXACT_REPLAY',trajectory_exact_to_numerical=equality,
        trajectory_differences=differences,render_noninterference_scope='instantaneous full native before/after only; crossrun forward trajectories compared separately',
        violations=int((d['official_margins']<0).any((0,2)).sum()),
        numerical_violations=int((numerical['official_margins']<0).any((0,2)).sum()),details=details)


if __name__=='__main__':
    runs=[one(mode) for mode in ['baseline_guard','joint_guard']]
    result=dict(status='PASS_BOUNDED_NATIVE_CAMERA_EVIDENCE',runs=runs,groups=42,images=378,actual_calibrated_views=378,
        render_invariance='independent offline equality of all64 q/qd/root/root_vel before AND after each9-view group',
        unique_new_initials_added=0,unique_new_tapes_added=0,production_promoted=False,hardware_approved=False,
        limitation='represented target spheres inside actual side planes with positive depth; actual near/far planes not captured; no full-link silhouette, pixel occlusion, continuous contact or object-carry safety claim',
        verification_source_sha256=sha(__file__))
    (HERE/'native_visual_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','groups','images','actual_calibrated_views']}))
