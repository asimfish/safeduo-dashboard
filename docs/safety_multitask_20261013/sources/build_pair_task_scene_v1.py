"""Prospective source IK path and real native105 terminal-state seed.

Finite source geometry is admission evidence only. No native task success is
credited here; no runtime state writes or attachment joints are generated.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch

K=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
E=Path('/mnt/nas/data/lyf/double_hand/safeduo_full50_geometry_20261013')
sys.path[:0]=[str(E),str(H/'fourhand_native_probe_v115')]
from full74_geometry_v2 import Full74Geometry
from multitask_phases_v1 import TaskSpec
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    raw=Path(json.loads((H/'fourhand_native_probe_v105/RESOURCE_EXECUTION_V1.json').read_text())['out'])
    source=raw/'physics_08160_08192.npz';z=dict(np.load(source));p=dict(np.load(raw/'native_parameters.npz'))
    reg=json.loads((H/'fourhand_native_probe_v115/REGISTRATION_V1.json').read_text())
    assets=json.loads(Path(reg['coupled_assets']).read_text())['assets'];geom=Full74Geometry(p,assets)
    seed={'object_seed':z['object_state'][-1].copy(),'finger_fraction':z['feedback_grip_finger_fraction'][-1].copy(),
        'thumb_fraction':z['feedback_grip_thumb_fraction'][-1].copy()}
    q={a:z[a+'_native_q'][-1].copy() for a in geom.names}
    for a in geom.names:
        seed[a+'_native_q_seed']=q[a];seed[a+'_native_qd_seed']=z[a+'_native_qd'][-1].copy()
        seed[a+'_actual_full_position_target_seed']=z[a+'_actual_full_position_target'][-1].copy()
        seed[a+'_actual_full_position_target_tape']=np.repeat(seed[a+'_actual_full_position_target_seed'][None],8,axis=0)
        seed[a+'_hand_nominal_position_target_tape']=np.repeat(z[a+'_hand_nominal_position_target'][-1][None],8,axis=0)
    initial=seed['object_seed'][:3]-p['scene_origin'][0];goal=initial+np.array([.22,0,0])
    spec=TaskSpec('pair_colift_native118','pair_colift','beam_common',('F_L','F_R'),tuple(map(float,initial)),tuple(map(float,goal)),(.04,.7,.08),.45);spec.validate()
    endpoints={};rows=[]
    shifts=np.array([[0,0,0],[0,0,.14],[.22,0,.14],[.22,0,0],[.22,0,.08]])
    for a in geom.names:
        ids=geom.armids[a];kin=geom.provider.kin[a[0]];base,yaw=geom.provider.layout.base_pose(a)
        start=torch.tensor(q[a][:,ids],dtype=torch.float32);fk=kin.fk(start,base,yaw)
        targets=fk['t_flange']+torch.tensor(shifts if a in spec.roles else np.zeros((5,3)),dtype=torch.float32)
        rotations=fk['R_flange'].repeat(5,1,1);qi=start.repeat(5,1)
        bounds=torch.tensor(p[a+'_hard_limits'][0,ids],dtype=torch.float32)
        for _ in range(500):
            f=kin.fk(qi,base,yaw);ep=targets-f['t_flange'];er=.5*sum(torch.cross(f['R_flange'][:,:,j],rotations[:,:,j],dim=-1) for j in range(3))
            jp=kin.point_jacobian(f,f['t_flange'][:,None],torch.tensor([kin.dof]))[:,0]
            J=torch.cat([jp,f['z'].transpose(-1,-2)],dim=1);delta=J.transpose(-1,-2)@torch.linalg.solve(J@J.transpose(-1,-2)+torch.eye(6)*1e-4,torch.cat([ep,er],dim=1)[:,:,None])
            qi+=.7*delta.squeeze(-1).clamp(-.02,.02);qi=torch.maximum(torch.minimum(qi,bounds[:,1]-.03),bounds[:,0]+.03)
        f=kin.fk(qi,base,yaw);pe=(f['t_flange']-targets).norm(dim=-1).numpy()
        oe=torch.acos(((f['R_flange'].transpose(-1,-2)@rotations).diagonal(dim1=-2,dim2=-1).sum(-1)-1).mul(.5).clamp(-1,1)).numpy()
        endpoints[a]=qi.numpy()
        rows.extend({'arm':a,'stage':stage,'position_error_m':float(pe[i]),'orientation_error_rad':float(oe[i]),'reachable':bool(pe[i]<=1e-4 and oe[i]<=.001)} for i,stage in enumerate(['CLOSE','LIFT','CARRY','PLACE','RETREAT']))
    trajectories={a:[] for a in geom.names};labels=[]
    for segment in range(4):
        for u in np.linspace(0,1,101):
            for a in geom.names:
                qa=q[a].copy();qa[:,geom.armids[a]]=(1-u)*endpoints[a][segment]+u*endpoints[a][segment+1];trajectories[a].append(qa[0])
            labels.append(['LIFT','CARRY','PLACE','RETREAT'][segment])
    trajectories={a:np.array(v) for a,v in trajectories.items()};gaps=geom.gaps(trajectories)
    plan={'status':'FINITE_SOURCE_TASK_IK_ADMISSION','task':spec.__dict__,'IK_rows':rows,'all_reachable':all(r['reachable'] for r in rows),
        'finite_source_path_states':len(labels),'all9021_minimum_m':float(gaps.min()),'all9021_samples_clear':bool(gaps.min()>=0),
        'hands_at_source_native_seed_for_this_screen':True,'future_pressure_finger_geometry_not_prequalified':True,
        'source_seed_event':8191,'initialization_only_state_restore':True,'new_physical_episodes':0,'complete_task_successes':0,
        'source_sha256':{str(source):sha(source),str(raw/'native_parameters.npz'):sha(raw/'native_parameters.npz'),str(Path(__file__)):sha(__file__)},
        'destination_support_posts':[{'center_m':[float(goal[0]),y,.89],'size_m':[.08,.04,.18]} for y in [-.18,.18]],
        'fullSystem0_accepted':False}
    assert not (K/'PAIR_TASK_SCENE_ADMISSION_V1.json').exists()
    # Preserve and compare the deterministic NPZ outputs of the pre-native
    # float32 JSON serialization failure; never overwrite those artifacts.
    for name,arrays in [('PAIR_TASK_NATIVE_SEED_V1.npz',seed),('PAIR_TASK_IK_ENDPOINTS_V1.npz',endpoints),
        ('PAIR_TASK_FINITE_GEOMETRY_PATH_V1.npz',dict(trajectories,phase=np.asarray(labels),all9021=gaps))]:
        if (K/name).exists():
            with np.load(K/name) as old:
                assert set(old.files)==set(arrays)
                for key,value in arrays.items():np.testing.assert_array_equal(old[key],value)
        else:np.savez_compressed(K/name,**arrays)
    (K/'PAIR_TASK_SCENE_ADMISSION_V1.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({k:v for k,v in plan.items() if k not in ['IK_rows','source_sha256']}),flush=True)


if __name__=='__main__':main()
