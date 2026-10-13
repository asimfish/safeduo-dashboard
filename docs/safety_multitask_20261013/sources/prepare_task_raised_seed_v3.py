"""Source-admitted pregrasp arm height; every finger and velocity stays original."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch

K=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
sys.path[:0]=['/mnt/nas/data/lyf/double_hand/safeduo_full50_geometry_20261013',str(H/'fourhand_native_probe_v118'),str(H/'mesh_blind_pair_v1')]
from full74_geometry_v2 import Full74Geometry,ARMS
from hand_kinematic_replay_v1 import quaternion_matrix
from convex_distance_v1 import convex_distance
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    reg=json.loads((H/'fourhand_native_probe_v118/REGISTRATION_V1.json').read_text());raw=Path(json.loads((H/'fourhand_native_probe_v118/RESOURCE_EXECUTION_V1.json').read_text())['out'])
    p=dict(np.load(raw/'native_parameters.npz'));seed=dict(np.load(K/'PAIR_TASK_NATIVE_SEED_V1.npz'));g=Full74Geometry(p,json.loads(Path(reg['coupled_assets']).read_text())['assets'])
    height=.06;IK=[]
    for a in ['F_L','F_R']:
        ids=g.armids[a];kin=g.provider.kin[a[0]];base,yaw=g.provider.layout.base_pose(a)
        qi=torch.tensor(seed[a+'_native_q_seed'][:,ids],dtype=torch.float32);f=kin.fk(qi,base,yaw);target=f['t_flange']+torch.tensor([0,0,height]);R=f['R_flange'];bounds=torch.tensor(p[a+'_hard_limits'][0,ids],dtype=torch.float32)
        for _ in range(400):
            f=kin.fk(qi,base,yaw);ep=target-f['t_flange'];er=.5*sum(torch.cross(f['R_flange'][:,:,j],R[:,:,j],dim=-1) for j in range(3))
            jp=kin.point_jacobian(f,f['t_flange'][:,None],torch.tensor([kin.dof]))[:,0];J=torch.cat([jp,f['z'].transpose(-1,-2)],dim=1)
            d=J.transpose(-1,-2)@torch.linalg.solve(J@J.transpose(-1,-2)+torch.eye(6)*1e-4,torch.cat([ep,er],dim=1)[:,:,None])
            qi+=.7*d.squeeze(-1).clamp(-.02,.02);qi=torch.maximum(torch.minimum(qi,bounds[:,1]-.03),bounds[:,0]+.03)
        f=kin.fk(qi,base,yaw);pe=float((f['t_flange']-target).norm());oe=float(torch.acos(((f['R_flange'].transpose(-1,-2)@R).diagonal(dim1=-2,dim2=-1).sum(-1)-1).mul(.5).clamp(-1,1)))
        delta=qi.numpy()-seed[a+'_native_q_seed'][:,ids]
        seed[a+'_native_q_seed'][:,ids]+=delta;seed[a+'_actual_full_position_target_seed'][:,ids]+=delta
        seed[a+'_actual_full_position_target_tape'][:,:,ids]+=delta[None]
        IK.append({'arm':a,'position_error_m':pe,'orientation_error_rad':oe,'passed':bool(pe<=1e-4 and oe<=.001)})
    poses=g.transforms({a:seed[a+'_native_q_seed'] for a in ARMS});ex=json.loads((H/'all_hand_object_mesh_v3/ALL_HAND_EXTRACTION_V1.json').read_text());vertices=dict(np.load(H/'all_hand_object_mesh_v3/ALL_HAND_COMPONENT_HULLS_V1.npz'))
    s=seed['object_seed'];R=quaternion_matrix(s[3:7][None])[0];center=s[:3]-p['scene_origin'][0]
    beam=np.array([[x,y,z] for x in [-.02,.02] for y in [-.35,.35] for z in [-.04,.04]])@R.T+center;rows=[]
    for c in ex['records']:
        if not c['arm'].startswith('F'):continue
        T=poses[c['arm']][c['body']][0];cloud=vertices[c['array_key']]@T[:3,:3].T+T[:3,3]
        lower=float(np.linalg.norm(np.maximum(0,np.maximum(cloud.min(0)-beam.max(0),beam.min(0)-cloud.max(0)))))
        if lower<.002:lower=float(convex_distance(cloud,beam,tolerance=1e-8,max_iterations=96)['lower_bound_m'])
        rows.append({'arm':c['arm'],'component':c['array_key'],'body':c['body'],'lower_gap_m':lower})
    original=dict(np.load(K/'PAIR_TASK_NATIVE_SEED_V1.npz'));paths={a:np.repeat(seed[a+'_native_q_seed'],101,axis=0) for a in ARMS}
    for a in ['F_L','F_R']:
        for i,u in enumerate(np.linspace(0,1,101)):paths[a][i]=(1-u)*seed[a+'_native_q_seed'][0]+u*original[a+'_native_q_seed'][0]
    minimum=float(g.gaps(paths).min());admitted=all(v['passed'] for v in IK) and min(v['lower_gap_m'] for v in rows)>=.002 and minimum>=.0005
    report={'status':'RAISED_PREGRASP_SOURCE_ADMISSION','height_m':height,'admitted':admitted,'IK':IK,
        'source_initial_hand_beam_min_gap_m':min(v['lower_gap_m'] for v in rows),'source_approach101_states_min9021_m':minimum,
        'hand_component_rows':rows,'all_hand_coordinates_and_all_velocities_unchanged':True,'physical_controller_unchanged_from118':True,
        'startup_only_arm_pose_and_arm_target_changed':True,'runtime_state_mutations':0,'complete_task_successes':0,
        'fullSystem0_accepted':False,'source_sha256':{str(K/'PAIR_TASK_NATIVE_SEED_V1.npz'):sha(K/'PAIR_TASK_NATIVE_SEED_V1.npz'),str(Path(__file__)):sha(__file__)}}
    assert not (K/'PAIR_TASK_RAISED_ADMISSION_V3.json').exists();np.savez_compressed(K/'PAIR_TASK_RAISED_NATIVE_SEED_V3.npz',**seed)
    (K/'PAIR_TASK_RAISED_ADMISSION_V3.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['hand_component_rows','source_sha256']}),flush=True)


if __name__=='__main__':main()
