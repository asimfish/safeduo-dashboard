"""Measured operating range before the first failure, and 2D marginal visits."""
import numpy as np

SLICES=(slice(0,7),slice(7,14),slice(14,20),slice(20,26))
ARMS=('F_L','F_R','U_L','U_R')

def safe_operation(q0,q,margins,limits,dt):
    bad=(margins<0).any(-1)
    end=np.where(bad.any(0),bad.argmax(0),len(q))
    span=limits[...,1]-limits[...,0]
    ranges=[];paths=[]
    for e,t in enumerate(end):
        states=np.concatenate([q0[e:e+1],q[:t,e]],0)
        ranges.append(float(np.mean(np.ptp(states,axis=0)/span[e])))
        paths.append(float(np.abs(np.diff(states,axis=0)).sum()))
    return dict(safe_transitions_before_first_violation=end.tolist(),
                mean_safe_duration_s=float(np.mean(end)*dt),median_safe_duration_s=float(np.median(end)*dt),
                mean_safe_prefix_joint_range=float(np.mean(ranges)),mean_safe_prefix_joint_path_rad=float(np.mean(paths)),
                note='Includes q0 and measured states strictly before the first negative official margin; offending transition excluded. Unequal prefix durations are reported, not normalized away. No IID probability claim.')

def pairwise_joint_visits(q0,q,limits):
    # Each 10x10 occupancy is only a 2D marginal; never a joint-volume fraction.
    normalized=(np.concatenate([q0[None],q],0)-limits[...,0])/(limits[...,1]-limits[...,0])
    points=normalized.reshape(-1,26)
    in_range=(points>=0)&(points<=1)
    bins=np.floor(np.clip(points,0,1)*10).astype(np.int64).clip(0,9)
    arm_idx=np.repeat(np.arange(4),[7,7,6,6]);rows=[];groups={}
    for i in range(26):
        for j in range(i+1,26):
            ok=in_range[:,i]&in_range[:,j]
            count=int(len(np.unique(bins[ok,i]*10+bins[ok,j])))
            key=(int(arm_idx[i]),int(arm_idx[j]))
            rows.append(dict(joint_i=i,joint_j=j,arm_i=ARMS[key[0]],arm_j=ARMS[key[1]],cells=count,possible_cells=100))
            groups.setdefault(key,[]).append(count)
    return dict(grid_bins=10,joint_pairs=325,rows=rows,
                mean_observed_cells=float(np.mean([r['cells'] for r in rows])),
                minimum_observed_cells=min(r['cells'] for r in rows),maximum_observed_cells=max(r['cells'] for r in rows),
                outside_soft_joint_sample_fraction=float((~in_range).mean()),
                groups=[dict(arm_i=ARMS[a],arm_j=ARMS[b],joint_pairs=len(v),mean_cells=float(np.mean(v)),minimum_cells=min(v),maximum_cells=max(v)) for (a,b),v in groups.items()],
                note='All q0 and all complete-window physical states including failures. Outside-soft-limit samples excluded for that joint pair. UR periodic states not deduplicated. 2D marginal cell counts are not 26D joint-space coverage, reachable volume, or collision-free volume.')
