"""All9021 raw rows, bounded queue-centred nominal model; no safety proof."""
from dataclasses import replace
import torch
ARMS=('F_L','F_R','U_L','U_R')
def construct(env,provider,data,centre,settings):
    n=64;D=26;device=env.device
    velocities={};sensitivities={};offsets={};lower=[];upper=[]
    va=[];vb=[];ta=[];tb=[];offset=0
    for a in ARMS:
        art=env._arms[a];view=art.root_physx_view;idx=env._joint_idx[a]
        def get(name):return getattr(view,name)().to(device).clone()
        q=get('get_dof_positions');v=get('get_dof_velocities');dof=q.shape[-1];width=len(idx)
        kp=get('get_dof_stiffnesses');kd=get('get_dof_dampings');mass=get('get_generalized_mass_matrices')
        di=torch.arange(dof,device=device);mass[:,di,di]+=get('get_dof_armatures')
        bias=get('get_coriolis_and_centrifugal_compensation_forces')
        if not art.cfg.spawn.rigid_props.disable_gravity:bias+=get('get_gravity_compensation_forces')
        target=get('get_dof_position_targets');target[:,idx]=centre[a]
        dt=float(env.cfg.sim.dt);C=mass.clone();C[:,di,di]+=dt*kd+dt*dt*kp
        rhs=torch.einsum('nij,nj->ni',mass,v)+dt*kp*(target-q)-dt*bias
        vn=torch.linalg.solve(C,rhs[...,None])[...,0]
        basis=torch.eye(dof,device=device)[:,idx][None].expand(n,-1,-1)
        S=torch.linalg.solve(C,dt*kp[...,None]*basis)
        V=torch.zeros(n,dof,D,device=device);V[:,:,offset:offset+width]=S
        vmax=get('get_dof_max_velocities');va.extend([V,-V]);vb.extend([-vmax-vn,vn-vmax])
        tau=kp*(target-q)-(dt*kp+kd)*vn
        T=torch.zeros_like(V);T[:,:,offset:offset+width]=kp[...,None]*basis-(dt*kp+kd)[...,None]*S
        effort=get('get_dof_max_forces');ta.extend([T,-T]);tb.extend([-effort-tau,tau-effort])
        limits=art.data.soft_joint_pos_limits[:,idx];cap=settings['queued_target_step_cap_rad']
        lower.append(torch.maximum(limits[...,0]-centre[a],torch.full_like(centre[a],-cap)))
        upper.append(torch.minimum(limits[...,1]-centre[a],torch.full_like(centre[a],cap)))
        velocities[a]=(v,vn);sensitivities[a]=S;offsets[a]=(offset,width);offset+=width
    assert offset==26 and data.dists.shape==(64,9021)
    geometry_A=[];geometry_b=[];measured=[];predicted=[];arrivals=[]
    # Same complete scoring rows. Chunking only bounds temporary GPU storage.
    for start in range(0,9021,settings['geometry_chunk_rows']):
        end=min(9021,start+settings['geometry_chunk_rows'])
        selected=torch.arange(start,end,device=device)[None].expand(n,-1)
        one=replace(data,active_idx=selected,active_mask=torch.ones_like(selected,dtype=torch.bool),
            active_pairs=torch.stack([data.dists.gather(1,selected),data.closing.gather(1,selected),env._sph.class_id[selected],env._sph.pair_id[selected]],-1),
            active_dmin=data.full_dmin.gather(1,selected),viol_exempt=data.full_viol_exempt.gather(1,selected))
        rows=provider.rows_from(one,env._body_pos_cache)
        m=torch.zeros(n,end-start,device=device);p=torch.zeros_like(m);coeff=torch.zeros(n,end-start,D,device=device)
        for robot,aa in [('F',ARMS[:2]),('U',ARMS[2:])]:
            native_offset=0
            for a in aa:
                v,vn=velocities[a];dof=v.shape[-1];J=rows.J[robot][:,:,native_offset:native_offset+dof];native_offset+=dof
                m+=torch.einsum('nmd,nd->nm',J,v);p+=torch.einsum('nmd,nd->nm',J,vn)
                off,width=offsets[a];coeff[:,:,off:off+width]=torch.einsum('nmd,ndk->nmk',J,sensitivities[a])
        arrival=rows.d+m.clamp_max(0)*settings['arrival_delay_s']
        desired=(-arrival/settings['response_timescale_s']).clamp(-settings['inward_velocity_cap_m_s'],settings['outward_velocity_cap_m_s'])
        geometry_A.append(coeff);geometry_b.append(desired-p);measured.append(m);predicted.append(p);arrivals.append(arrival)
    A=torch.cat([torch.cat(geometry_A,1),*va,*ta],1);b=torch.cat([torch.cat(geometry_b,1),*vb,*tb],1)
    tol=torch.cat([torch.full((9021,),1e-4,device=device),torch.full((148,),1e-3,device=device),torch.full((148,),1e-3,device=device)])
    assert A.shape==(64,9317,26) and b.shape==(64,9317) and tol.shape==(9317,)
    return dict(A=A,b=b,lower=torch.cat(lower,-1),upper=torch.cat(upper,-1),tol=tol,
        reference=torch.cat([centre[a] for a in ARMS],-1),measured_geometry_rate=torch.cat(measured,1),predicted_geometry_rate=torch.cat(predicted,1),passive_arrival_gap=torch.cat(arrivals,1),offsets=offsets,
        velocity_ids=[(a+'/'+name,side) for a in ARMS for side in ['lower','upper'] for name in env._arms[a].joint_names],
        effort_ids=[(a+'/'+name,side) for a in ARMS for side in ['lower','upper'] for name in env._arms[a].joint_names],
        native_ids=[a+'/'+name for a in ARMS for name in env._arms[a].joint_names],controlled_ids=[a+'/'+env._arms[a].joint_names[j] for a in ARMS for j in env._joint_idx[a].tolist()])
