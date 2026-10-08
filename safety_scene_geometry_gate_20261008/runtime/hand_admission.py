"""Opt-in experimental hand gate. Exact measured paths only; rejection is not success.

This module controls hand position targets, not the arm System0 action or an
emergency stop. Abort remains latched. A separate empty-hand return controller may command neutral; physical motion can persist.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Decision:
    admitted: bool
    reason: str
    aborted: bool

class HandAdmission:
    def __init__(self, passport, arm):
        self.arm=arm; self.p=passport; self.latched=False
        self.opened=np.asarray(passport['open_rad'][arm],dtype=np.float32)
        self.limits=np.asarray(passport['limits_rad'][arm],dtype=np.float32)
        self.allowed=[np.asarray(x,dtype=np.float32) for x in passport['allowed_goals_rad'][arm]]
        self.active=False

    def observe(self, state, current_step):
        if self.latched:return Decision(False,'abort_latched',True)
        try:
            q=np.asarray(state['q'],dtype=np.float32);qd=np.asarray(state['qd'],dtype=np.float32)
            force=np.asarray(state['self_normal_n'],dtype=np.float32)
            external=np.asarray(state['unintended_normal_n'],dtype=np.float32)
            if q.shape!=self.opened.shape or qd.shape!=q.shape or force.size==0 or external.size==0:raise ValueError('shape')
            if not np.isfinite(q).all() or not np.isfinite(qd).all() or not np.isfinite(force).all() or not np.isfinite(external).all():reason='nonfinite_state'
            elif state['step']!=current_step-1:reason='stale_state'
            elif np.max(force)>np.float32(self.p['max_self_normal_n']):reason='self_contact'
            elif np.max(external)>np.float32(self.p['max_self_normal_n']):reason='unintended_contact'
            elif np.max(np.abs(qd))>self.p['max_speed_rad_s']:reason='speed_limit'
            elif np.any(q<self.limits[:,0]-.02) or np.any(q>self.limits[:,1]+.02):reason='state_bounds'
            else:return Decision(True,'state_valid',False)
        except (KeyError,TypeError,ValueError):reason='missing_or_invalid_state'
        self.latched=True;self.active=False
        return Decision(False,reason,True)

    def request(self, goal, ramp_s, reference_time_s, state, current_step):
        observed=self.observe(state,current_step)
        if not observed.admitted:return observed
        try:
            goal=np.asarray(goal,dtype=np.float32)
            if goal.shape!=self.opened.shape or not np.isfinite(goal).all():return Decision(False,'invalid_target',False)
            if np.any(goal<self.limits[:,0]) or np.any(goal>self.limits[:,1]):return Decision(False,'target_bounds',False)
            if ramp_s!=self.p['ramp_s']:return Decision(False,'unqualified_path_duration',False)
            if reference_time_s not in self.p['registered_reference_times_s']:return Decision(False,'unregistered_arm_reference',False)
            if np.max(abs(np.asarray(state['q'],np.float32)-self.opened))>self.p['max_start_error_rad']:return Decision(False,'unqualified_start',False)
            if not any(np.array_equal(goal,x) for x in self.allowed):return Decision(False,'unqualified_exact_target',False)
        except (TypeError,ValueError):return Decision(False,'invalid_target',False)
        for exclusion in self.p['context_exclusions']:
            if exclusion['arm']==self.arm and exclusion['reference_time_s']==reference_time_s and np.array_equal(goal,np.asarray(exclusion['goal_rad'],np.float32)):
                return Decision(False,'known_unsafe_context',False)
        self.active=True
        return Decision(True,'qualified_exact_path',False)
