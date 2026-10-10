"""Force and actual-joint feedback for an experimental hand task controller.

This emits bounded position-target progress. It cannot certify a stop or a
friction wrench; the independent native force, joint and task gates are required.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class GripDecision:
    finger_fraction: float
    thumb_fraction: float
    mode: str
    thumb_guard: bool
    finger_guard: bool
    predicted_lower_margin_rad: float
    predicted_upper_margin_rad: float


class FeedbackGrip:
    def __init__(self, control_dt_s, caps, seek_rate=.15, relax_rate=.75,
                 hold_above_N=4., resume_below_N=2.5, relax_above_N=6.,
                 joint_reserve_rad=.03, prediction_s=.10):
        values=(control_dt_s,seek_rate,relax_rate,hold_above_N,resume_below_N,
                relax_above_N,joint_reserve_rad,prediction_s)
        if not all(math.isfinite(v) and v>0 for v in values):raise ValueError('invalid grip settings')
        if not resume_below_N<hold_above_N<relax_above_N:raise ValueError('invalid force hysteresis')
        if not caps or any(len(v)!=2 or not all(math.isfinite(x) and 0<=x<=1 for x in v) for v in caps.values()):
            raise ValueError('invalid registered fractions')
        self.dt=control_dt_s;self.caps=dict(caps);self.seek=seek_rate;self.relax=relax_rate
        self.hold=hold_above_N;self.resume=resume_below_N;self.over=relax_above_N
        self.reserve=joint_reserve_rad;self.horizon=prediction_s;self.previous={};self.mode={}

    def update(self, arm, sequence, state_time_s, now_s, q, qd, hard_limits, thumb_mask,
               current_finger, current_thumb, normal_N):
        if arm not in self.caps:raise ValueError('unregistered hand')
        q=np.asarray(q,dtype=float);v=np.asarray(qd,dtype=float);lim=np.asarray(hard_limits,dtype=float)
        mask=np.asarray(thumb_mask,dtype=bool)
        if q.ndim!=1 or v.shape!=q.shape or lim.shape!=(len(q),2) or mask.shape!=q.shape or not len(q):
            raise ValueError('invalid hand observation shape')
        if not mask.any() or mask.all():raise ValueError('hand group binding incomplete')
        scalars=(state_time_s,now_s,current_finger,current_thumb,normal_N)
        if not all(math.isfinite(x) for x in scalars) or not all(np.isfinite(x).all() for x in (q,v,lim)):
            raise ValueError('nonfinite hand observation')
        if normal_N<0 or np.any(lim[:,0]>=lim[:,1]):raise ValueError('invalid force or limits')
        if not 0<=now_s-state_time_s<=1.5*self.dt:raise ValueError('stale or future hand observation')
        if not 0<=current_finger<=self.caps[arm][0]+1e-6 or not 0<=current_thumb<=self.caps[arm][1]+1e-6:
            raise ValueError('progress outside registered stroke')
        old=self.previous.get(arm)
        if old is not None:
            if sequence<=old[0] or state_time_s<=old[1]:raise ValueError('duplicate or nonmonotonic hand observation')
            elapsed=state_time_s-old[1]
            if elapsed>1.5*self.dt:raise ValueError('missing hand feedback')
            finite_difference=(q-old[2])/elapsed
            inward=np.minimum(v,finite_difference);outward=np.maximum(v,finite_difference)
        else:inward=v.copy();outward=v.copy()
        lower=q-lim[:,0]+np.minimum(inward,0)*self.horizon
        upper=lim[:,1]-q-np.maximum(outward,0)*self.horizon
        # A stationary open joint near its lower limit is not itself a request
        # to retract. React to measured motion toward a boundary, including
        # position drift that a native velocity snapshot does not represent.
        endangered=((lower<self.reserve)&(inward<-.02))|((upper<self.reserve)&(outward>.02))
        thumb_guard=bool(endangered[mask].any());finger_guard=bool(endangered[~mask].any())
        if thumb_guard or finger_guard:
            mode='JOINT_GUARD';df=-self.relax*self.dt if finger_guard else 0.;dt=-self.relax*self.dt if thumb_guard else 0.
        elif normal_N>=self.over:
            mode='RELAX';df=dt=-self.relax*self.dt
        elif normal_N>=self.hold or (self.mode.get(arm)=='HOLD' and normal_N>=self.resume):
            mode='HOLD';df=dt=0.
        else:
            mode='SEEK';df=dt=self.seek*self.dt
        finger=min(self.caps[arm][0],max(0.,current_finger+df))
        thumb=min(self.caps[arm][1],max(0.,current_thumb+dt))
        self.previous[arm]=(sequence,state_time_s,q.copy());self.mode[arm]=mode
        return GripDecision(finger,thumb,mode,thumb_guard,finger_guard,float(lower.min()),float(upper.min()))
