"""Coverage counts from measured trajectories, without inventing reachable volume."""
import numpy as np

ARM_KEYS=('F_L','F_R','U_L','U_R')
SLICES=(slice(0,7),slice(7,14),slice(14,20),slice(20,26))
PAIRS=tuple((i,j) for i in range(4) for j in range(i+1,4))
PAIR_NAMES=tuple(f'{ARM_KEYS[i]}-{ARM_KEYS[j]}' for i,j in PAIRS)


def joint_occupancy(q,limits):
    q=np.asarray(q)
    lo,hi=limits[...,0],limits[...,1]
    if q.ndim==2:q=q[None]
    f=(q-lo)/(hi-lo)
    valid=(f>=0)&(f<=1)&np.isfinite(f)
    bins=np.minimum(np.floor(np.where(valid,f,0)*10).astype(int),9)
    counts=np.zeros((26,10),dtype=np.int64)
    for j in range(26):
        counts[j]=np.bincount(bins[...,j][valid[...,j]],minlength=10)
    return dict(counts=counts.tolist(),visited_bins_per_joint=(counts>0).sum(-1).tolist(),
                mean_visited_fraction=float((counts>0).mean()),
                outside_soft_limit_samples=int((~valid).sum()))


def ee_occupancy(ee,voxel=.10):
    counts=[];lo=[];hi=[]
    for a in range(4):
        values=ee[...,a,:].reshape(-1,3)
        if not len(values):counts.append(0);lo.append(None);hi.append(None);continue
        cells=np.floor(values/voxel).astype(np.int64)
        counts.append(len(np.unique(cells,axis=0)))
        lo.append(values.min(0).tolist());hi.append(values.max(0).tolist())
    return dict(voxel_m=voxel,visited_voxels_per_arm=counts,aabb_min_m=lo,aabb_max_m=hi,
                interpretation='observed local EE cells only; no kinematic or collision-free reachable-volume denominator')


def measured_coverage(q_initial,q,ee_initial,ee,limits,mask=None):
    n=q.shape[1]
    if mask is None:mask=np.ones(n,dtype=bool)
    initial=q_initial[mask];motion=q[:,mask];lim=limits[mask]
    initial_ee=ee_initial[None,mask];motion_ee=ee[:,mask]
    if not mask.any():return dict(episodes=0,initial_joint=None,visited_joint=None,initial_ee=None,visited_ee=None,
                                 mean_within_window_joint_range=None,mean_joint_path_rad=None)
    all_q=np.concatenate([initial[None],motion])
    all_ee=np.concatenate([initial_ee,motion_ee])
    span=lim[...,1]-lim[...,0]
    ranges=(all_q.max(0)-all_q.min(0))/span
    return dict(episodes=int(mask.sum()),initial_joint=joint_occupancy(initial,lim),
                visited_joint=joint_occupancy(all_q,lim),initial_ee=ee_occupancy(initial_ee),visited_ee=ee_occupancy(all_ee),
                mean_within_window_joint_range=float(ranges.mean()),
                mean_within_window_joint_range_per_arm=[float(ranges[:,s].mean()) for s in SLICES],
                mean_joint_path_rad=float(np.abs(np.diff(all_q,axis=0)).sum((0,2)).mean()))


def pair_exposure(pair_margin,ee,mask=None):
    if mask is None:mask=np.ones(pair_margin.shape[1],dtype=bool)
    pm=pair_margin[:,mask];positions=ee[:,mask]
    rows=[]
    for p,(i,j) in enumerate(PAIRS):
        vec=positions[:-1,:,i]-positions[:-1,:,j]
        delta=np.diff(positions[:,:,i]-positions[:,:,j],axis=0)
        norm=np.linalg.norm(vec,axis=-1)
        movement=np.linalg.norm(delta,axis=-1)
        radial=np.einsum('tnd,tnd->tn',vec,delta)/np.maximum(norm,1e-8)
        active=movement>.0001
        tangent=active&(np.abs(radial)<=movement*.2)
        approach=active&~tangent&(radial<0)
        recede=active&~tangent&(radial>0)
        rows.append(dict(pair=PAIR_NAMES[p],episodes=int(mask.sum()),
                         exposed_80mm=int(((pm[...,p]<.080).sum(0)>=3).sum()),
                         near_5mm=int((pm[...,p]<.005).any(0).sum()),
                         overlap=int((pm[...,p]<0).any(0).sum()),
                         min_margin_mm=None if not mask.any() else float(pm[...,p].min()*1000),
                         approach_ee_steps=int(approach.sum()),recede_ee_steps=int(recede.sum()),
                         tangent_ee_steps=int(tangent.sum()),stationary_ee_steps=int((~active).sum())))
    return rows
