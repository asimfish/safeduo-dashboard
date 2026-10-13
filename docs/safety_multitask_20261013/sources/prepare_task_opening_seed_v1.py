"""Counterfactual task startup: open source roots before initialization only."""
from pathlib import Path
import hashlib,json,sys
import numpy as np

K=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
sys.path[:0]=['/mnt/nas/data/lyf/double_hand/safeduo_full50_geometry_20261013',str(H/'fourhand_native_probe_v118'),str(H/'mesh_blind_pair_v1')]
from full74_geometry_v2 import Full74Geometry,ARMS
from hand_kinematic_replay_v1 import quaternion_matrix
from convex_distance_v1 import convex_distance
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    reg=json.loads((H/'fourhand_native_probe_v118/REGISTRATION_V1.json').read_text());raw=Path(json.loads((H/'fourhand_native_probe_v118/RESOURCE_EXECUTION_V1.json').read_text())['out'])
    p=dict(np.load(raw/'native_parameters.npz'));seed=dict(np.load(K/'PAIR_TASK_NATIVE_SEED_V1.npz'));g=Full74Geometry(p,json.loads(Path(reg['coupled_assets']).read_text())['assets'])
    refs=dict(np.load(reg['coupled_reference_endpoints']));roots=dict(np.load(reg['motion_test']['command_bank']));changes=[]
    for a in ['F_L','F_R']:
        ids=roots[a+'_full50_root_indices'];names=p[a+'_joint_names'].tolist();delta=np.zeros(len(ids),np.float32)
        hi=refs[a+'_hand_indices'];openq=dict(zip(hi,refs[a+'_hand_open_q']))
        for j,i in enumerate(ids):
            if i in openq and 'thumb_1' not in names[i]:
                distance=float(seed[a+'_native_q_seed'][0,i]-openq[i]);delta[j]=-np.sign(distance)*min(.08,abs(distance))
                changes.append({'arm':a,'root':names[i],'initial_only_delta_rad':float(delta[j])})
        full_delta=delta@p[a+'_full_motor_velocity_mapping'].T
        seed[a+'_native_q_seed']+=full_delta[None]
        seed[a+'_actual_full_position_target_seed']+=full_delta[None]
        seed[a+'_actual_full_position_target_tape']+=full_delta[None,None]
        seed[a+'_hand_nominal_position_target_tape']+=full_delta[hi][None]
    q={a:seed[a+'_native_q_seed'] for a in ARMS};poses=g.transforms(q);gap=float(g.gaps(q).min())
    extraction=json.loads((H/'all_hand_object_mesh_v3/ALL_HAND_EXTRACTION_V1.json').read_text());verts=dict(np.load(H/'all_hand_object_mesh_v3/ALL_HAND_COMPONENT_HULLS_V1.npz'))
    s=seed['object_seed'];R=quaternion_matrix(s[3:7][None])[0];center=s[:3]-p['scene_origin'][0]
    corners=np.array([[x,y,z] for x in [-.02,.02] for y in [-.35,.35] for z in [-.04,.04]])@R.T+center
    rows=[]
    for c in extraction['records']:
        a=c['arm'];body=c['body'];T=poses[a][body][0]
        cloud=verts[c['array_key']]@T[:3,:3].T+T[:3,3]
        lower=float(np.linalg.norm(np.maximum(0,np.maximum(cloud.min(0)-corners.max(0),corners.min(0)-cloud.max(0)))))
        if lower<.002:
            result=convex_distance(cloud,corners,tolerance=1e-8,max_iterations=96)
            lower=float(result['lower_bound_m'])
            status=result['status']
        else:status='MODEL_AABB_SEPARATING_DISTANCE_BOUND'
        rows.append({'arm':a,'component':c['array_key'],'body':body,'beam_enclosing_box_lower_gap_m':lower,'status':status})
    frows=[v for v in rows if v['arm'].startswith('F')]
    hard=all(np.all((q[a]>=p[a+'_hard_limits'][...,0])&(q[a]<=p[a+'_hard_limits'][...,1])) for a in ARMS)
    admitted=gap>=.0005 and hard and min(r['beam_enclosing_box_lower_gap_m'] for r in frows)>=.002
    result={'status':'PREINITIALIZATION_ONLY_OPENING_ADMISSION','closed_native_reference':118,'new_candidate':119,
        'initial_only_motor_changes':changes,'unmodified_arm_and_upper_hand_states':True,'unchanged_velocities':True,
        'all74_initial_hard_limits_pass':hard,'all9021_source_initial_gap_m':gap,'hand_components_screened':len(rows),
        'minimum_F_hand_to_beam_enclosing_box_gap_m':min(r['beam_enclosing_box_lower_gap_m'] for r in frows),
        'admitted':admitted,'hand_beam_source_rows':rows,'cooked_contact_offsets_not_certified':True,
        'runtime_state_writes':0,'native_episodes':0,'fullSystem0_accepted':False,
        'source_sha256':{str(K/'PAIR_TASK_NATIVE_SEED_V1.npz'):sha(K/'PAIR_TASK_NATIVE_SEED_V1.npz'),str(Path(__file__)):sha(__file__)}}
    for name in ['PAIR_TASK_OPENING_ADMISSION_V1.json','PAIR_TASK_OPENING_NATIVE_SEED_V1.npz']:assert not (K/name).exists()
    np.savez_compressed(K/'PAIR_TASK_OPENING_NATIVE_SEED_V1.npz',**seed)
    (K/'PAIR_TASK_OPENING_ADMISSION_V1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['hand_beam_source_rows','source_sha256']}),flush=True)


if __name__=='__main__':main()
