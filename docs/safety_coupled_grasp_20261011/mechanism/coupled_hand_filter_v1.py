"""Rate-bounded, limit-consistent motor targets expanded to original URDF followers."""
import numpy as np
from coupled_hand_targets_v1 import CoupledHand,Relation

def coupled_filtered_hand_targets(names,nominal,previous,hard,dt,loaded,relations,
                                  max_target_rate_rad_s=.6,distal_lower_reserve_rad=.10):
    names=tuple(names);nominal=np.asarray(nominal,dtype=float);previous=np.asarray(previous,dtype=float);hard=np.asarray(hard,dtype=float)
    assert nominal.shape==previous.shape==(len(names),) and hard.shape==(len(names),2)
    assert np.isfinite(np.r_[nominal,previous,hard.ravel(),dt,max_target_rate_rad_s,distal_lower_reserve_rad]).all()
    assert dt>0 and max_target_rate_rad_s>0 and distal_lower_reserve_rad>=0
    hand=CoupledHand(names,hard,[Relation(r['slave'],r['master'],r['urdf_multiplier'],r['urdf_offset_rad']) for r in relations])
    motors=[names.index(n) for n in hand.motors];old=previous[motors];desired=nominal[motors]
    assert np.max(abs(hand.expand(desired)-nominal))<1e-6,'Nominal target must already satisfy original coupling'
    assert np.max(abs(hand.expand(old)-previous))<1e-6,'Previous native target must already satisfy original coupling'
    assert (previous>=hard[:,0]-1e-6).all() and (previous<=hard[:,1]+1e-6).all()
    operational=hard.copy()
    if loaded:
        for i,n in enumerate(names):
            if 'thumb_3_joint' in n or 'thumb_4_joint' in n:
                operational[i,0]+=min(distal_lower_reserve_rad,.2*(hard[i,1]-hard[i,0]))
    bounded=CoupledHand(names,operational,hand.relations);limits=bounded.feasible_motor_intervals()
    desired=np.clip(desired,limits[:,0],limits[:,1])
    amplification=np.array([max(abs(s) for root,s,b in hand.affine.values() if root==motor) for motor in hand.motors])
    delta=max_target_rate_rad_s*dt/amplification
    result=hand.expand(old+np.clip(desired-old,-delta,delta))
    assert np.max(abs(result-previous))<=max_target_rate_rad_s*dt+1e-6
    assert (result>=hard[:,0]-1e-6).all() and (result<=hard[:,1]+1e-6).all()
    return result
