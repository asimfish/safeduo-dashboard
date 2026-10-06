"""Explicit local-frame ground-truth task input and pickup-reference binding.

This module is an isolated development candidate. It does not modify the
frozen SafeDuo safety observation, controller, model or existing task files.
"""
import numpy as np
import torch

ARMS=('F_L','F_R','U_L','U_R')
FIELDS=('x','y','z','qw','qx','qy','qz','vx','vy','vz','wx','wy','wz',
        'size_x','size_y','size_z','goal_x','goal_y','goal_z','age_s','valid')

def pack_task_input(native_states,origins,sizes,goals,state_time_s,now_s,valid,max_age_s):
    if native_states.ndim!=3 or native_states.shape[1:]!=(2,13):raise ValueError('expected N,2,13 native states')
    n=native_states.shape[0]
    if origins.shape!=(n,3) or valid.shape!=(n,2):raise ValueError('frame/validity shape')
    if sizes.shape!=(2,3) or goals.shape!=(2,3):raise ValueError('object geometry/target shape')
    if not torch.isfinite(native_states).all() or not torch.isfinite(origins).all():raise ValueError('nonfinite state/frame')
    if not torch.isfinite(sizes).all() or not (sizes>0).all() or not torch.isfinite(goals).all():raise ValueError('invalid geometry/target')
    if valid.dtype!=torch.bool or not valid.all():raise ValueError('invalid object observation')
    age=float(now_s-state_time_s)
    if not np.isfinite(age) or age<0 or age>max_age_s:raise ValueError('stale/future observation')
    norm=native_states[:,:,3:7].norm(dim=-1)
    if not torch.allclose(norm,torch.ones_like(norm),atol=1e-4,rtol=0):raise ValueError('invalid native quaternion')
    local=native_states.clone();local[:,:,:3]-=origins[:,None,:]
    out=torch.cat((local,sizes[None].expand(n,-1,-1),goals[None].expand(n,-1,-1),
        torch.full((n,2,1),age,device=local.device,dtype=local.dtype),valid[:,:,None].to(local.dtype)),dim=-1)
    assert out.shape==(n,2,len(FIELDS)) and torch.isfinite(out).all()
    return out

def pickup_weight(time_s,carry_end_s):
    up=np.clip((np.asarray(time_s)-.5)/3.,0,1)
    down=1-np.clip((np.asarray(time_s)-7.2)/(carry_end_s-7.2),0,1)
    return (.5-.5*np.cos(np.pi*up))*(.5-.5*np.cos(np.pi*down))

def bind_reference(base_q,dt,task_input,enabled,nominal_centers,nominal_yaws_rad):
    """Plan from measured initial poses; no live object pose writes or following."""
    from safeduo.delta.task_record_s9 import make_s9_provider
    n=task_input.shape[0];length=next(iter(base_q.values())).shape[0]
    assert enabled.shape==(n,) and enabled.dtype==torch.bool
    # Initial upright yaw layouts only; unhandled tilted inputs fail explicitly.
    if torch.max(task_input[:,:,4:6].abs()).item()>1e-5:raise ValueError('initial roll/pitch unsupported')
    device='cpu';input_cpu=task_input.detach().cpu().numpy()
    provider=make_s9_provider(1,device=device,spheres='r16')
    out={a:np.broadcast_to(base_q[a],(n,*base_q[a].shape)).copy() for a in ARMS};audit=[]
    for ai,a in enumerate(ARMS):
        oi=ai//2;kin=provider.kin[a[0]];pos,yaw=provider.layout.base_pose(a)
        q0=torch.tensor(base_q[a],dtype=torch.float32)
        f0=kin.fk(q0,pos,yaw);w=pickup_weight(np.arange(length)*dt,10.2 if oi==0 else 11.7)
        for e in range(n):
            center=input_cpu[e,oi,:3];qw,qx,qy,qz=input_cpu[e,oi,3:7]
            dyaw=float(2*np.arctan2(qz,qw)-nominal_yaws_rad[oi]);shift=center-np.asarray(nominal_centers[oi])
            # Exact no-op for nominal layout, independent of the enabled flag.
            if not bool(enabled[e]) or (np.max(np.abs(shift))<1e-7 and abs(dyaw)<1e-7):
                audit.append(dict(env=e,arm=a,enabled=bool(enabled[e]),max_position_error_m=0.,max_orientation_error_rad=0.,exact_nominal=True));continue
            angles=torch.tensor(w*dyaw,dtype=torch.float32);c,s=torch.cos(angles),torch.sin(angles)
            Rz=torch.zeros((length,3,3));Rz[:,0,0]=c;Rz[:,0,1]=-s;Rz[:,1,0]=s;Rz[:,1,1]=c;Rz[:,2,2]=1
            pivot=torch.tensor(nominal_centers[oi],dtype=torch.float32)
            goal=pivot+(Rz@(f0['t_flange']-pivot).unsqueeze(-1)).squeeze(-1)+torch.tensor(w[:,None]*shift,dtype=torch.float32)
            rotation=Rz@f0['R_flange'];q=q0.clone()
            for _ in range(35):
                fk=kin.fk(q,pos,yaw);dp=goal-fk['t_flange']
                rot=.5*sum(torch.cross(fk['R_flange'][:,:,j],rotation[:,:,j],dim=-1) for j in range(3))
                Jp=kin.point_jacobian(fk,fk['t_flange'][:,None],torch.tensor([kin.dof]))[:,0]
                J=torch.cat((Jp,fk['z'].transpose(-1,-2)),dim=1);err=torch.cat((dp,rot),dim=1)
                dq=J.transpose(-1,-2)@torch.linalg.solve(J@J.transpose(-1,-2)+torch.eye(6)*1e-5,err[:,:,None])
                q+=dq.squeeze(-1).clamp(-.02,.02)*.8
            fk=kin.fk(q,pos,yaw)
            error=float((fk['t_flange']-goal).norm(dim=-1).max())
            orient=float(torch.acos(((fk['R_flange'].transpose(-1,-2)@rotation).diagonal(dim1=-2,dim2=-1).sum(-1)-1).mul(.5).clamp(-1,1)).max())
            if error>.001 or orient>.01:raise ValueError(f'IK rejected {e}/{a}: position {error} orientation {orient}')
            out[a][e]=q.numpy()
            if not np.array_equal(out[a][e,0],base_q[a][0]):raise ValueError('binding changed registered q0')
            audit.append(dict(env=e,arm=a,enabled=True,max_position_error_m=error,max_orientation_error_rad=orient,exact_nominal=False,translation_m=shift.tolist(),yaw_delta_rad=dyaw))
    return out,audit
