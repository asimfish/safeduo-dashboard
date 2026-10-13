"""Prospective slow approach and measured settling before pressure.

The native119 force violation remains a failure. This policy has software tests
only; lower target rates and reactive force intents are not a force/stop bound.
"""
import numpy as np
from pair_task_operator_v1 import PairTaskOperator
from multitask_phases_v1 import ARMS


class PairTaskOperatorV2(PairTaskOperator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.settled_start=None;self.last_request_time=None
        self.mode='UNQUALIFIED_WAIT';self.stop_intent_required=False
        self.physical_force_bound_qualified=False

    def request(self,previous_target,actual_q,actual_qd,time_s):
        if not np.isfinite(time_s) or (self.last_request_time is not None and time_s<=self.last_request_time):
            raise ValueError('fresh monotonic native request time required')
        q={};v={}
        for a in ARMS:
            q[a]=np.asarray(actual_q[a]);v[a]=np.asarray(actual_qd[a])
            shape=(len(self.parameters[a+'_joint_names']),)
            if q[a].shape!=shape or v[a].shape!=shape or not np.isfinite(q[a]).all() or not np.isfinite(v[a]).all():
                raise ValueError('all original74 actual coordinates and velocities required')
        if self.last_request_time is not None and time_s-self.last_request_time>.05:self.settled_start=None
        self.last_request_time=time_s
        if any(self.force[ARMS.index(a)]>=12. for a in self.task.spec.roles):
            self.mode='UNQUALIFIED_CONTACT_LOAD_STOP_INTENT';self.stop_intent_required=True
            # Caller must route this outside the intent FIFO into the actual
            # brake layer. The returned request alone does not stop a robot.
            return {a:q[a][self.rootids[a]].copy() for a in ARMS}
        self.stop_intent_required=False
        result=super().request(previous_target)
        if self.task.phase!='CLOSE':self.mode='UNQUALIFIED_TASK_PHASE';return result
        settled=True
        for a in self.task.spec.roles:
            ids=self.parameters[a+'_controlled_indices'].ravel();pos=self.arm_positions[a];goal=self.endpoints[a][0]
            settled &= np.max(np.abs(q[a][ids]-goal))<=.005 and np.max(np.abs(v[a][ids]))<=.02
            settled &= np.max(np.abs(np.asarray(previous_target[a])[pos]-goal))<=1e-6
            result[a][pos]=np.asarray(previous_target[a])[pos]+np.clip(goal-np.asarray(previous_target[a])[pos],-.02*self.dt,.02*self.dt)
        if not settled:self.settled_start=None
        elif self.settled_start is None:self.settled_start=time_s
        ready=self.settled_start is not None and time_s-self.settled_start>=.3-1e-9
        self.mode='UNQUALIFIED_PRESSURE_AFTER_ACTUAL_SETTLE' if ready else 'UNQUALIFIED_APPROACH_HOLD_HANDS'
        for a in self.task.spec.roles:
            hand=[j for j,i in enumerate(self.rootids[a]) if int(i) in self.open[a]]
            prior=np.asarray(previous_target[a])
            if not ready:result[a][hand]=prior[hand]
            else:result[a][hand]=prior[hand]+np.clip(result[a][hand]-prior[hand],-.005*self.dt,.005*self.dt)
        return result
