"""Bounded, per-clone clearance interlock; no physics state writes."""
import torch

METHODS=('scheduled_debt','stationary_release','clearance_feedback')

def zero_held_command(cmd,held):
    # Both nominal/debt deltas and any precomputed command must be covered.
    for a in cmd.delta_q:
        cmd.delta_q[a]=torch.where(held[:,None],torch.zeros_like(cmd.delta_q[a]),cmd.delta_q[a])
        if hasattr(cmd,'_r37_nominal'):
            cmd._r37_nominal[a]=torch.where(held[:,None],torch.zeros_like(cmd._r37_nominal[a]),cmd._r37_nominal[a])
    return cmd

class ClearanceGate:
    def __init__(self,methods,dt,params,device='cpu'):
        self.dt=dt;self.p=params
        self.method=torch.tensor([METHODS.index(m) for m in methods],device=device)
        self.phase=torch.zeros(len(methods),dtype=torch.float64,device=device)
        # 0 regular, 1 waiting, 2 admitted clearance, 3 aborted.
        self.stage=torch.zeros_like(self.method)
        self.wait=torch.zeros_like(self.phase)
        self.stable=torch.zeros_like(self.method);self.bad=torch.zeros_like(self.method)
        self.reason=torch.zeros_like(self.method)
    def step(self,x):
        p=self.p
        regular=(self.stage==0)|(self.stage==2)
        next_phase=self.phase+regular*self.dt
        entry=(self.method==2)&(self.stage==0)&(next_phase>=p['clearance_start_s'])
        self.stage[entry]=1;self.phase[entry]=p['clearance_start_s']
        waiting=self.stage==1
        if x is not None:
            finite=torch.stack([torch.isfinite(v).reshape(len(self.phase),-1).all(-1) for v in x.values()]).all(0)
            support=(x['table_n']>p['support_normal_min_n']).all(-1)
            bottom=(x['bottom_delta_m'].abs()<=p['bottom_abs_max_m']).all(-1)
            slow=(x['speed_m_s']<=p['settled_speed_max_m_s']).all(-1)&(x['angular_rad_s']<=p['settled_angular_max_rad_s']).all(-1)
            upright=(x['tilt_deg']<=p['tilt_max_deg']).all(-1)
            detached=(x['hand_n']<=p['hand_normal_max_n']).reshape(len(self.phase),-1).all(-1)
            opened=(x['hand_open_max_error_rad']<=p['open_error_max_rad']).all(-1)
            good=finite&support&bottom&slow&upright&detached&opened
            self.stable=torch.where(waiting&good,self.stable+1,torch.zeros_like(self.stable))
            ready=waiting&(self.stable>=p['stable_steps'])
            self.stage[ready]=2
            # Post-contact reaction; this is not predictive collision avoidance.
            bad=((x['hand_n']>p['hand_normal_max_n']).reshape(len(self.phase),-1).any(-1)|
                 (x['speed_m_s']>p['reaction_speed_max_m_s']).any(-1)|
                 (x['angular_rad_s']>p['reaction_angular_max_rad_s']).any(-1))
            active=(self.stage==2)&~ready
            self.bad=torch.where(active&bad,self.bad+1,torch.zeros_like(self.bad))
            reaction=active&(self.bad>=p['reaction_steps'])
            self.stage[reaction]=3;self.reason[reaction]=2
            poisoned=(self.method==2)&(self.phase>=p['clearance_start_s'])&~finite
            self.stage[poisoned]=3;self.reason[poisoned]=3
        self.wait+=waiting*self.dt
        expired=(self.stage==1)&(self.wait>=p['max_wait_s'])
        self.stage[expired]=3;self.reason[expired]=1
        # Admission takes effect next control step, preserving explicit entry hold.
        move=regular&~entry&(self.stage!=3)
        self.phase=torch.where(move,next_phase,self.phase)
        hold=((self.method!=0)&(self.phase>=p['release_start_s'])&(self.phase<p['clearance_start_s']))|waiting|(self.stage==3)|entry
        return hold
