"""Verify actual six-plane camera math, native snapshots and own visual state."""
from pathlib import Path
import json,hashlib,numpy as np
HERE=Path(__file__).resolve().parent;RAW=Path('/mnt/nas/data/lyf/double_hand')/HERE.name
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
    runs=[];galleries=[];images=0
    d=json.loads((HERE/'RANDOM_EXPERIMENT_DESIGN.json').read_text());seed=d['rows'][0]['command_seed']
    for mode in ('joint_reference','motion_admission'):
        root=Path(json.loads((HERE/'VISUAL_EFFECTIVE_PLAN.json').read_text())['actual_visual_roots'][mode]);p=json.loads((root/'visual_protocol.json').read_text());assert p['status']=='complete' and p['steps']==960 and p['completed_windows']==64
        visual=load(root/'cell_001.npz');num=load(RAW/'holdout_0'/f'{mode}_{seed}'/'cell_001.npz');identity=json.loads((root/'full_row_identity.json').read_text())
        rec=json.loads((root/'camera_receipts.json').read_text());assert rec['scheduled_groups']==21 and 0<=rec['first_failure_groups']<=7
        assert rec['groups']==21+rec['first_failure_groups'] and rec['PNG']==9*rec['groups']
        assignments=load(root/'bank_assignment.npz')['risk_pair_index'];bad_frames=(visual['official_margins']<0).any(-1)
        scheduled=set();causal={}
        for cap in rec['receipts']:
            state_path=root/cap['state'];assert sha(state_path)==cap['sha256'];s=json.loads(state_path.read_text());e,t=s['env_id'],s['step']
            assert s['visual_mode']==mode and s['environment_count']==64 and len(s['pending_actuator_targets'])==6
            assert s['capture_kind']==cap['capture_kind'] and s['risk_stratum']==int(assignments[e])
            if s['capture_kind']=='scheduled':scheduled.add((e,t));assert not s['first_negative_capture']
            else:
                label=s['risk_stratum'];assert label not in causal and s['first_negative_capture']
                hit=bad_frames[:,assignments==label].any(-1);assert hit.any() and int(hit.argmax())==t
                assert e==int(np.flatnonzero((assignments==label)&bad_frames[t])[0]);causal[label]=(e,t)
            bf=root/s['fresh_native_snapshot'];af=root/s['fresh_native_after_snapshot'];assert sha(bf)==s['fresh_native_before_sha256'] and sha(af)==s['fresh_native_after_sha256']
            before,after=load(bf),load(af);assert set(before)==set(after) and len(before)==16
            for key in before:assert np.isfinite(before[key]).all() and np.array_equal(before[key],after[key]),key
            compact=[];qd=[]
            for a in ('F_L','F_R','U_L','U_R'):
                idx=s['fresh_native_controlled_joint_indices'][a]
                assert np.array_equal(before[a+'_q'][e,idx],np.array(s['arms'][a]['q'],dtype=before[a+'_q'].dtype))
                compact.extend(s['arms'][a]['q']);qd.extend(s['arms'][a]['qd'])
            assert np.array_equal(np.array(compact,dtype=visual['q'].dtype),visual['q'][t,e])
            centers=np.array(s['sphere_centers_world_m']);radii=np.array(s['sphere_radii_m']);arms=np.array(identity['sphere_arm_id'])
            assert centers.shape==(142,3) and len(radii)==142
            cls=np.array(identity['class_id']);pairs=np.array(identity['pair_sphere_idx']);evalcls=cls.copy();evalcls[(cls==1)&(arms[pairs[:,0]]>=2)]=2;evalcls[cls==2]=3
            distances=np.array(s['full_distances_m'],np.float32);exempt=np.array(s['full_exempt'],bool)
            margin=np.array([np.where(exempt|(evalcls!=k),np.inf,distances).min() for k in range(4)],np.float32)
            assert np.array_equal(margin,visual['official_margins'][t,e])
            for name,v in s['views'].items():
                W=np.array(v['actual_camera_to_world_row_matrix']);K=np.array(v['actual_intrinsic_matrix']);clip=np.array(v['actual_clipping_range_m'])
                assert np.isfinite(W).all() and np.isfinite(K).all() and 0<=clip[0]<clip[1]
                assert np.allclose(W[3,:3],v['actual_position_world_m'],rtol=0,atol=1e-8)
                assert np.allclose(W[:3,:3]@W[:3,:3].T,np.eye(3),rtol=0,atol=1e-6)
                o=v['actual_usd_optics'];assert o['horizontal_aperture_offset']==0 and o['vertical_aperture_offset']==0
                expected=np.array([[1280*o['focal_length']/o['horizontal_aperture'],0,640],[0,720*o['focal_length']/o['vertical_aperture'],360],[0,0,1.]])
                assert np.allclose(K,expected,rtol=1e-5,atol=2e-4),(name,K,expected)
                mask=arms<2 if name.startswith('f_') else arms>=2 if name.startswith('u_') else np.ones(142,bool)
                pts=centers[mask];rad=radii[mask];pc=np.c_[pts,np.ones(len(pts))]@np.linalg.inv(W);x,y,z=pc[:,0],pc[:,1],-pc[:,2]
                fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
                planes=np.stack([(fx*x+cx*z)/np.hypot(fx,cx)-rad,((1280-cx)*z-fx*x)/np.hypot(fx,1280-cx)-rad,
                    (cy*z-fy*y)/np.hypot(fy,cy)-rad,((720-cy)*z+fy*y)/np.hypot(fy,720-cy)-rad,z-clip[0]-rad,clip[1]-z-rad],-1)
                assert (planes>0).all() and np.allclose(planes.min(0),v['observed_sphere_frustum']['minimum_by_plane_m'],rtol=0,atol=1e-7)
                assert v['observed_sphere_frustum']['plane_names']==['left','right','top','bottom','near','far']
            for item in cap['images']:
                file=root/item['path'];assert sha(file)==item['sha256']
                from PIL import Image
                with Image.open(file) as im:assert im.size==(1280,720)
                images+=1
            galleries.append(dict(mode=mode,env=e,step=t,kind=s['capture_kind'],state=cap['state'],state_sha256=cap['sha256'],before=s['fresh_native_snapshot'],after=s['fresh_native_after_snapshot'],
                before_sha256=s['fresh_native_before_sha256'],after_sha256=s['fresh_native_after_sha256'],
                images=[{**item,'view':Path(item['path']).stem.split('_',2)[-1]} for item in cap['images']]))
        assert scheduled=={(e,t) for e in [0,8,16,24,32,40,56] for t in [75,480,959]}
        assert set(causal)=={int(label) for label in [-1,0,1,2,3,4,5] if bad_frames[:,assignments==label].any()}
        diff=(visual['q']!=num['q']).any((1,2));exact=not diff.any()
        runs.append(dict(mode=mode,visual_violations=int((visual['official_margins']<0).any((0,2)).sum()),numerical_violations=int((num['official_margins']<0).any((0,2)).sum()),
            q_first_difference=int(np.flatnonzero(diff)[0]) if diff.any() else None,q_max_difference=float(np.abs(visual['q'].astype(np.float64)-num['q']).max()),
            first_failure_groups=len(causal),scheduled_groups=len(scheduled),binding_status='PASS_EXACT_REPLAY' if exact and all(np.array_equal(visual[k],num[k]) for k in ('q','ee','cmd','exec','official_margins','controller_target','actuator_target')) else 'FAIL_EXACT_REPLAY'))
    receipt=dict(status='PASS_BOUNDED_ACTUAL_SIX_PLANE_CAMERA',images=images,groups=len(galleries),scheduled_groups=42,first_failure_groups=len(galleries)-42,runs=runs,galleries=galleries,
        scope='actual target-group sphere frustum, instantaneous all64 native equality and own visual-state binding; occlusion, mesh silhouette and forward noninterference not certified')
    (HERE/'visual_verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['status'],images,[r['binding_status'] for r in runs])
if __name__=='__main__':main()
