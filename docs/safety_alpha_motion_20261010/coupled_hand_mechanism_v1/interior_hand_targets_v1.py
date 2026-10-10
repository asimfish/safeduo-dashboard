"""Bound target changes and retain distal thumb targets away from lower stops.

Target limits are operating choices, not an actual-state safety certificate.
"""
import numpy as np


def filtered_hand_targets(joint_names, nominal, previous_native_target, hard_limits, control_dt_s,
                          loaded, max_target_rate_rad_s=.6, distal_lower_reserve_rad=.10):
    nominal=np.asarray(nominal,dtype=float);previous=np.asarray(previous_native_target,dtype=float)
    limits=np.asarray(hard_limits,dtype=float);names=list(joint_names)
    if nominal.ndim!=1 or len(names)!=len(nominal) or previous.shape!=nominal.shape or limits.shape!=(len(names),2):
        raise ValueError('invalid bound hand target shapes')
    if not all(np.isfinite(v).all() for v in [nominal,previous,limits]) or not np.isfinite(control_dt_s) or control_dt_s<=0:
        raise ValueError('invalid hand target values')
    if not np.all(limits[:,1]>limits[:,0]):raise ValueError('invalid hand limits')
    if np.any(previous<limits[:,0]-1e-6) or np.any(previous>limits[:,1]+1e-6):raise ValueError('previous target outside original hard limits')
    distal=np.array([('thumb_3_joint' in n or 'thumb_4_joint' in n) for n in names])
    desired=np.clip(nominal,limits[:,0],limits[:,1])
    if loaded:
        floor=limits[:,0]+np.minimum(distal_lower_reserve_rad,.2*(limits[:,1]-limits[:,0]))
        desired=np.where(distal,np.maximum(desired,floor),desired)
    step=max_target_rate_rad_s*control_dt_s
    result=previous+np.clip(desired-previous,-step,step)
    return np.clip(result,limits[:,0],limits[:,1])
