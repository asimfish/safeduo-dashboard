"""Calibrate opening direction per hand from original source collider hulls."""
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
    roots=dict(np.load(reg['motion_test']['command_bank']));ex=json.loads((H/'all_hand_object_mesh_v3/ALL_HAND_EXTRACTION_V1.json').read_text());vertices=dict(np.load(H/'all_hand_object_mesh_v3/ALL_HAND_COMPONENT_HULLS_V1.npz'))
    s=seed['object_seed'];R=quaternion_matrix(s[3:7][None])[0];center=s[:3]-p['scene_origin'][0]
    beam=np.array([[x,y,z] for x in [-.02,.02] for y in [-.35,.35] for z in [-.04,.04]])@R.T+center
    rows=[];selected={};deltas={};closure={}
    for a in ['F_L','F_R']:
        ids=roots[a+'_full50_root_indices'];names=p[a+'_joint_names'].tolist();components=[c for c in ex['records'] if c['arm']==a]
        for fi,th in [(-1,-1),(1,-1),(-1,1),(1,1)]:
            root_delta=np.zeros(len(ids),np.float32)
            for j,i in enumerate(ids):
                if any(w in names[i] for w in ['index','middle','ring','pinky']):root_delta[j]=fi*.08
                elif 'thumb_2' in names[i]:root_delta[j]=th*.08
            full_delta=root_delta@p[a+'_full_motor_velocity_mapping'].T;q={arm:seed[arm+'_native_q_seed'].copy() for arm in ARMS};q[a]+=full_delta[None]
            hard=bool(np.all((q[a]>=p[a+'_hard_limits'][...,0])&(q[a]<=p[a+'_hard_limits'][...,1])))
            poses=g.transforms(q)[a];minimum=float('inf');component_rows=[]
            for c in components:
                T=poses[c['body']][0];cloud=vertices[c['array_key']]@T[:3,:3].T+T[:3,3]
                lower=float(np.linalg.norm(np.maximum(0,np.maximum(cloud.min(0)-beam.max(0),beam.min(0)-cloud.max(0)))))
                if lower<.002:
                    d=convex_distance(cloud,beam,tolerance=1e-8,max_iterations=96);lower=float(d['lower_bound_m'])
                minimum=min(minimum,lower);component_rows.append({'component':c['array_key'],'body':c['body'],'lower_gap_m':lower})
            row={'arm':a,'finger_sign':fi,'thumb2_sign':th,'root_delta_rad':root_delta.tolist(),'all_initial_hard_pass':hard,
                'minimum_source_beam_enclosing_box_gap_m':minimum,'admitted':bool(hard and minimum>=.002),'components':component_rows}
            rows.append(row)
            if row['admitted'] and a not in selected:
                selected[a]={'finger_sign':fi,'thumb2_sign':th,'minimum_gap_m':minimum};deltas[a]=full_delta
                closure[a]={names[i]:float(-np.sign(root_delta[j])) for j,i in enumerate(ids) if root_delta[j]!=0}
    admitted=set(selected)=={'F_L','F_R'}
    if admitted:
        for a,delta in deltas.items():
            seed[a+'_native_q_seed']+=delta[None];seed[a+'_actual_full_position_target_seed']+=delta[None]
            seed[a+'_actual_full_position_target_tape']+=delta[None,None]
            handids=np.array([i for i,n in enumerate(p[a+'_joint_names'].tolist()) if any(v in n for v in ['thumb','index','middle','ring','pinky'])])
            seed[a+'_hand_nominal_position_target_tape']+=delta[handids][None]
        minimum_geometry=float(g.gaps({a:seed[a+'_native_q_seed'] for a in ARMS}).min());admitted=minimum_geometry>=.0005
        np.savez_compressed(K/'PAIR_TASK_OPENING_NATIVE_SEED_V2.npz',**seed)
    else:minimum_geometry=None
    result={'status':'SOURCE_CALIBRATED_TASK_OPENING_SEARCH','selected':selected,'closure_direction_by_motor':closure,
        'admitted':admitted,'initial_all9021_gap_m':minimum_geometry,'candidate_hand_postures':rows,
        'selection_rule':'First declared finger/thumb sign pair with all original hard limits and every source hand component at least2mm from the enclosing beam box',
        'source_geometry_is_not_cooked_contact_offset_certificate':True,'new_native_episodes':0,'fullSystem0_accepted':False,
        'source_sha256':{str(K/'PAIR_TASK_NATIVE_SEED_V1.npz'):sha(K/'PAIR_TASK_NATIVE_SEED_V1.npz'),str(Path(__file__)):sha(__file__)}}
    assert not (K/'PAIR_TASK_OPENING_ADMISSION_V2.json').exists();(K/'PAIR_TASK_OPENING_ADMISSION_V2.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['candidate_hand_postures','source_sha256']}),flush=True)


if __name__=='__main__':main()
