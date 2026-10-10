"""Source-hull frustum coverage with native camera/body/object readback custody."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(version):
    P=H/f'fourhand_native_probe_v{version}';reg=json.loads((P/'REGISTRATION_V1.json').read_text());audit=json.loads((P/'INDEPENDENT_CAPTURE_AUDIT_V1.json').read_text());assert audit['record_integrity_verified']
    raw=Path(json.loads((P/'RESOURCE_EXECUTION_V1.json').read_text())['out']);frames=json.loads((raw/'camera_frames.json').read_text());params=np.load(raw/'native_parameters.npz')
    ext=Path(reg['complete_hand_extraction']);meta=json.loads(ext.read_text());vp=ext.parent/'ALL_HAND_COMPONENT_HULLS_V1.npz';vertices=np.load(vp);identity=json.loads((raw/'native_task_identity.json').read_text());origin=np.array(identity['scene_origin'])[0];spec=reg['camera_framing_gate'];records=[];sources={}
    pieces=[]
    for chunk in json.loads((raw/'physics_chunks.json').read_text()):
        p=raw/chunk['path'];assert sha(p)==chunk['sha256'];sources[str(p)]=sha(p)
        with np.load(p) as z:pieces.append({k:z[k].copy() for k in z.files if k=='object_state' or any(x in k for x in ['_body_position_local','_body_quaternion_wxyz'])})
    d={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    for f in frames:
        image=raw/f['path'];assert sha(image)==f['sha256'];sources[str(image)]=sha(image)
        assert f['camera_pose_metadata_source']=='PublicCameraResetPoseBuffer_and_independent_USD_world_transform'
        M=np.array(f['camera_world_xform_row_major']);pos=np.array(f['camera_pos_world_m']);quat=np.array(f['camera_quaternion_opengl_wxyz']);K=np.array(f['intrinsic_matrix']);R=M[:3,:3].T
        np.testing.assert_allclose(pos,reg['camera_views'][f['view']]['eye'],atol=1e-5,rtol=0);np.testing.assert_allclose(pos,M[3,:3],atol=1e-5,rtol=0);np.testing.assert_allclose(R,Rotation.from_quat(quat[[1,2,3,0]]).as_matrix(),atol=1e-5,rtol=0)
        clouds={a:[] for a in ['F_L','F_R','U_L','U_R']};event=f['control']*2-1
        for a,poses in f['native_body_link_transforms_xyzw'].items():
            poses=np.array(poses);names=params[a+'_body_names'].tolist();assert len(poses)==len(names)
            if event>=0:
                np.testing.assert_allclose(poses[:,:3]-origin,d[a+'_body_position_local'][event,0],atol=1e-5,rtol=0)
                actual_R=Rotation.from_quat(poses[:,3:]).as_matrix();saved_q=d[a+'_body_quaternion_wxyz'][event,0]
                np.testing.assert_allclose(actual_R,Rotation.from_quat(saved_q[:,[1,2,3,0]]).as_matrix(),atol=1e-5,rtol=0)
            for c in meta['records']:
                if c['arm']==a:
                    t=poses[names.index(c['body'])];clouds[a].append(vertices[c['array_key']]@Rotation.from_quat(t[3:]).as_matrix().T+t[:3])
            clouds[a]=np.concatenate(clouds[a])
        t=np.array(f['native_object_transform_xyzw']);corners=np.array([[x,y,z] for x in [-.02,.02] for y in [-.35,.35] for z in [-.04,.04]]);clouds['beam']=corners@Rotation.from_quat(t[3:]).as_matrix().T+t[:3]
        if event>=0:
            np.testing.assert_allclose(t[:3]-origin,d['object_state'][event,:3],atol=1e-5,rtol=0)
            np.testing.assert_allclose(Rotation.from_quat(t[3:]).as_matrix(),Rotation.from_quat(d['object_state'][event,[4,5,6,3]]).as_matrix(),atol=1e-5,rtol=0)
        boxes={};all_visible=True
        for name,C in clouds.items():
            local=(C-pos)@R;depth=-local[:,2];front=bool((depth>0).all());uv=(local[:,:2]*np.array([1,-1]))/depth[:,None];uv=uv@K[:2,:2].T+K[:2,2];low=uv.min(0);high=uv.max(0)
            margin=float(min(*low,f['image_width']-high[0],f['image_height']-high[1]));readable=True if name!='beam' else float((high-low).min())>=spec['minimum_beam_bbox_short_edge_px']
            visible=front and margin>=spec['minimum_margin_px'] and readable;all_visible=all_visible and visible;boxes[name]=dict(bbox_min_px=low.tolist(),bbox_max_px=high.tolist(),minimum_frame_margin_px=margin,all_source_vertices_in_front=front,passes_declared_frustum_and_readability=visible)
        critical=f['phase_after_control'] in spec['critical_phases'];records.append(dict(view=f['view'],control=f['control'],phase=f['phase_after_control'],critical_phase=critical,all4_hands_and_beam_source_frustum_pass=all_visible,boxes=boxes))
    required=[r for r in records if r['critical_phase'] and r['view'] in ['top','front','side']];assert required,'No manipulation-phase framing evidence'
    passed=all(r['all4_hands_and_beam_source_frustum_pass'] for r in required)
    for p in [Path(__file__),P/'REGISTRATION_V1.json',P/'INDEPENDENT_CAPTURE_AUDIT_V1.json',raw/'camera_frames.json',raw/'native_parameters.npz',raw/'native_task_identity.json',ext,vp]:sources[str(p)]=sha(p)
    report=dict(status='CLOSED_INDEPENDENT_NATIVE_CAMERA_POSE_AND_SOURCE_FRUSTUM_AUDIT',camera_pose_metadata_integrity_pass=True,manipulation_three_view_source_frustum_pass=passed,critical_view_frames=len(required),all_frame_records=records,source_sha256=sources,visual_occlusion_or_cooked_geometry_certified=False,full_four_arm_visual_mesh_visibility_certified=False,all_task_phases_observed=audit['four_hand_task_development_pass'],fullSystem0_accepted=False)
    p=P/'INDEPENDENT_CAMERA_FRAMING_V1.json';assert not p.exists();p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(report['status'],'three_view_source_frustum',passed,'frames',len(required),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('version',type=int);main(p.parse_args().version)
