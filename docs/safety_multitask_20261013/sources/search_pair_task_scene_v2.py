"""Prospective eight-direction task admission; failed source paths retained."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch

K=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
sys.path[:0]=['/mnt/nas/data/lyf/double_hand/safeduo_full50_geometry_20261013',str(H/'fourhand_native_probe_v115')]
from full74_geometry_v2 import Full74Geometry
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    old=json.loads((K/'PAIR_TASK_SCENE_ADMISSION_V1.json').read_text());raw=Path(json.loads((H/'fourhand_native_probe_v105/RESOURCE_EXECUTION_V1.json').read_text())['out'])
    p=dict(np.load(raw/'native_parameters.npz'));seed=dict(np.load(K/'PAIR_TASK_NATIVE_SEED_V1.npz'))
    reg=json.loads((H/'fourhand_native_probe_v115/REGISTRATION_V1.json').read_text());assets=json.loads(Path(reg['coupled_assets']).read_text())['assets']
    g=Full74Geometry(p,assets);roles=old['task']['roles'];directions=np.array([[np.cos(t),np.sin(t)] for t in np.arange(8)*np.pi/4])*.22
    endpoint={};reachable=np.ones(8,bool);errors=[]
    for a in g.names:
        ids=g.armids[a];kin=g.provider.kin[a[0]];base,yaw=g.provider.layout.base_pose(a)
        initial=torch.tensor(seed[a+'_native_q_seed'][:,ids],dtype=torch.float32)
        fk=kin.fk(initial,base,yaw);base_t=fk['t_flange'];base_R=fk['R_flange'];qs=[]
        # Initialize each successive stage from its preceding solution to keep
        # the original posture branch and avoid discontinuous IK jumps.
        qi=initial.repeat(8,1)
        for stage in range(5):
            shifts=np.zeros((8,3))
            if a in roles:
                if stage in [1,2]:shifts[:,2]=.14
                elif stage==4:shifts[:,2]=.08
                if stage>=2:shifts[:,:2]=directions
            targets=base_t+torch.tensor(shifts,dtype=torch.float32);rotations=base_R.repeat(8,1,1)
            bounds=torch.tensor(p[a+'_hard_limits'][0,ids],dtype=torch.float32)
            for _ in range(400):
                f=kin.fk(qi,base,yaw);ep=targets-f['t_flange'];er=.5*sum(torch.cross(f['R_flange'][:,:,j],rotations[:,:,j],dim=-1) for j in range(3))
                jp=kin.point_jacobian(f,f['t_flange'][:,None],torch.tensor([kin.dof]))[:,0]
                J=torch.cat([jp,f['z'].transpose(-1,-2)],dim=1);d=J.transpose(-1,-2)@torch.linalg.solve(J@J.transpose(-1,-2)+torch.eye(6)*1e-4,torch.cat([ep,er],dim=1)[:,:,None])
                qi+=.7*d.squeeze(-1).clamp(-.02,.02);qi=torch.maximum(torch.minimum(qi,bounds[:,1]-.03),bounds[:,0]+.03)
            f=kin.fk(qi,base,yaw);pe=(f['t_flange']-targets).norm(dim=-1).numpy()
            oe=torch.acos(((f['R_flange'].transpose(-1,-2)@rotations).diagonal(dim1=-2,dim2=-1).sum(-1)-1).mul(.5).clamp(-1,1)).numpy()
            reachable&=(pe<=1e-4)&(oe<=.001);qs.append(qi.numpy().copy())
            errors.append({'arm':a,'stage':stage,'position_error_m':pe.tolist(),'orientation_error_rad':oe.tolist()})
        endpoint[a]=np.stack(qs,axis=1)
    rows=[];selected=None;selected_path=None
    for candidate in range(8):
        paths={a:[] for a in g.names}
        for segment in range(4):
            for u in np.linspace(0,1,101):
                for a in g.names:
                    q=seed[a+'_native_q_seed'][0].copy();q[g.armids[a]]=(1-u)*endpoint[a][candidate,segment]+u*endpoint[a][candidate,segment+1];paths[a].append(q)
        paths={a:np.asarray(v) for a,v in paths.items()};gaps=g.gaps(paths)
        row={'candidate':candidate,'xy_displacement_m':directions[candidate].tolist(),'IK_reachable':bool(reachable[candidate]),
            'minimum_proxy_gap_m':float(gaps.min()),'negative_geometry_states':int((gaps.min(1)<0).sum()),'admitted':bool(reachable[candidate] and gaps.min()>=.0005)}
        rows.append(row)
        if row['admitted'] and selected is None:selected=candidate;selected_path=paths
    result={'status':'PROSPECTIVE_EIGHT_DIRECTION_SOURCE_TASK_SEARCH','candidates':rows,'IK_errors':errors,
        'selected_candidate':selected,'selection_rule':'First predeclared angular-order candidate satisfying every IK and original9021 finite geometry gate with0.5mm source reserve',
        'sampled_source_states_per_candidate':404,'continuous_or_native_contact_certificate':False,
        'objects_pose_updated_during_operation':False,'native_episodes':0,'fullSystem0_accepted':False,
        'source_sha256':{str(K/'PAIR_TASK_NATIVE_SEED_V1.npz'):sha(K/'PAIR_TASK_NATIVE_SEED_V1.npz'),str(Path(__file__)):sha(__file__)}}
    if selected is not None:
        task=dict(old['task']);task['goal_position']=(np.asarray(task['initial_position'])+np.r_[directions[selected],0]).tolist()
        result['task']=task;result['destination_support_posts']=[{'center_m':[task['goal_position'][0],y,.89],'size_m':[.08,.04,.18]} for y in [-.18+directions[selected,1],.18+directions[selected,1]]]
        np.savez_compressed(K/'PAIR_TASK_IK_ENDPOINTS_V2.npz',**{a:endpoint[a][selected] for a in g.names})
        np.savez_compressed(K/'PAIR_TASK_FINITE_GEOMETRY_PATH_V2.npz',**selected_path)
    assert not (K/'PAIR_TASK_SCENE_ADMISSION_V2.json').exists();(K/'PAIR_TASK_SCENE_ADMISSION_V2.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['IK_errors','source_sha256']}),flush=True)


if __name__=='__main__':main()
