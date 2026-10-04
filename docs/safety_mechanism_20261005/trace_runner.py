"""Observe the actual dynamic solver; no control or geometry mutation."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'safety_risk_strata_20261004'))
import risk_runner as risk
from safeduo.safety.types import ARM_KEYS

OriginalEpisodeTrace=risk.RiskTrace.__mro__[2]
ROW_KEYS={'pre_struct_exempt','solver_cap','delivered_row_backlog','issued_row_backlog',
          'projected_margin_delta','applied_margin_delta'}

class MechanismTrace(risk.RiskTrace):
    def start(self,env):
        super().start(env)
        assert self.causal and self.kinematic,'complete causal kinematic trace required'
        self.original_project=env._backstop.project
        keys=['pre_row_J_F','pre_row_J_U','pre_row_arm_mask','pre_pending_targets','pre_actual_queue',
              'pre_struct_exempt','pre_bypass_arm','solver_cap','solver_priority_p','delivered_row_backlog',
              'issued_row_backlog','full_table_distance','full_table_exempt','post_class_min_row_id']
        for k in keys:self.frames[k]=[]
        sphere=env._sph
        cls=sphere.class_id.long().clone();cls[cls==2]=3
        cls[sphere._slice_self]=1+sphere.self_robot.long()
        self.full_class=cls
        self.row_meta=dict(class_index=cls.cpu().tolist(),conditional=sphere.pair_conditional.cpu().tolist(),
                           nominal_dmin=sphere.pair_dmin.cpu().tolist(),table_start=sphere._slice_table.start,
                           observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (self.out/'mechanism_metadata.json').write_text(json.dumps(self.row_meta,indent=2)+'\n')
        def observe(*args,**kwargs):
            result=self.original_project(*args,**kwargs)
            self.calls+=1
            self.observed=dict(solver_cap=result[2]['cap'],solver_priority_p=args[3],
                               pre_struct_exempt=kwargs['struct_exempt'],pre_bypass_arm=kwargs['bypass_arm'])
            return result
        env._backstop.project=observe

    def before_step(self,env,t):
        super().before_step(env,t)
        # WideTrace overrides the generic causal hook; invoke its original ancestor.
        OriginalEpisodeTrace.before_step(self,env,t)
        self.calls=0;self.q_before=self.frames['pre_q'][-1];self.j_before=self.pre_rows.J
        history=env._pending_target_history;assert history is not None
        values=dict(pre_row_J_F=self.pre_rows.J['F'],pre_row_J_U=self.pre_rows.J['U'],
                    pre_row_arm_mask=self.pre_rows.arm_mask,
                    pre_pending_targets=torch.stack([torch.cat([p[a] for a in ARM_KEYS],-1) for p in history.targets],1),
                    pre_actual_queue=torch.stack([torch.cat([p[a] for a in ARM_KEYS],-1) for p in env._evaluation_actuator_delay.queue.pending],1))
        for k,v in values.items():self.frames[k].append(v.detach().clone())

    def step(self,env,t):
        super().step(env,t)
        assert self.calls==1,self.calls
        values=dict(self.observed)
        if values['pre_struct_exempt'] is None:values['pre_struct_exempt']=torch.zeros_like(self.pre_rows.valid)
        if values['pre_bypass_arm'] is None:values['pre_bypass_arm']=torch.zeros((env.num_envs,4),device=env.device,dtype=torch.bool)
        for k,target in [('delivered_row_backlog',env._evaluation_actuator_delay.applied),('issued_row_backlog',env._targets)]:
            backlog=torch.cat([target[a] for a in ARM_KEYS],-1)-self.q_before
            values[k]=torch.einsum('nmd,nd->nm',self.j_before['F'],backlog[:,:14])+torch.einsum('nmd,nd->nm',self.j_before['U'],backlog[:,14:])
        values['full_table_distance']=env._sph.last_table_margin
        values['full_table_exempt']=env._sph.last_table_viol_exempt
        full=env._last_out;masked=full.dists.masked_fill(full.full_viol_exempt,float('inf'))
        ids=[]
        for c in range(4):ids.append(masked.masked_fill(self.full_class!=c,float('inf')).argmin(-1))
        values['post_class_min_row_id']=torch.stack(ids,-1)
        for k,v in values.items():self.frames[k].append(v.detach().clone())
        # Bound observer GPU memory; copied tensors never feed control.
        for frames in self.frames.values():
            if frames:frames[-1]=frames[-1].cpu()

    def write(self,*args,**kwargs):
        widths=[v.shape[1] for v in self.frames['pre_row_d']];width=max(widths);assert width<=512
        for k,values in self.frames.items():
            if not (k.startswith('pre_row_') or k in ROW_KEYS):continue
            for i,v in enumerate(values):
                pad=width-v.shape[1]
                if pad:
                    padding=(0,pad) if v.ndim==2 else (0,0,0,pad)
                    values[i]=F.pad(v,padding,value=-1 if k in ['pre_row_id','pre_row_cls'] else 0)
        (self.out/'trace_padding.json').write_text(json.dumps(dict(width=width,frame_widths=widths,
                                                                  control_width_unchanged=True,storage_padding_only=True))+'\n')
        try:return super().write(*args,**kwargs)
        finally:self.env._backstop.project=self.original_project

risk.base.battery.EpisodeTrace=MechanismTrace
if __name__=='__main__':risk.base.battery.main()
