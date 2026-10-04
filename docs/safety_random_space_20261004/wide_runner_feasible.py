"""Wide-space evaluation only; fixed policy and original safety implementation."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'dependencies'))
from critical_rows import merge_critical_rows
from safeduo.eval import research_battery as battery
from safeduo.eval.random_battery import PairMarginProbe
from safeduo.safety.types import ARM_KEYS,DeltaCmd
from wide_random import make_initial,make_tape

FLOWS=('feasible_burst',)


class WideSource:
    def __init__(self,env,kind,amp,seed,steps):
        self.kind=kind
        self.n=env.num_envs
        temporal='iid' if kind=='wide_iid' else 'mixed_hold'
        tape,self.info=make_tape(self.n,steps,amp,seed,temporal)
        self.tape=tape.to(env.device);self.index=0
        baseline=torch.cat([env._arms[a].data.default_joint_pos[:,env._joint_idx[a]] for a in ARM_KEYS],-1)
        limits=torch.cat([env._q_soft_limits[a] for a in ARM_KEYS],-2)
        master=int(sys.argv[sys.argv.index('--seeds')+1])
        self.initial_seed=master
        bank_path=Path(os.environ['SAFEDUO_INITIAL_BANK_NPZ'])
        with np.load(bank_path,allow_pickle=False) as bank:
            accepted=bank['accepted_q'].copy()
            bank_limits=bank['joint_soft_limits'].copy()
        if accepted.shape!=(self.n,26):raise ValueError('bank must provide exactly one pose per environment')
        if not np.array_equal(bank_limits,limits.cpu().numpy()):raise ValueError('bank soft limits differ from evaluation')
        if not np.isfinite(accepted).all():raise ValueError('bank contains nonfinite positions')
        initial=torch.from_numpy(accepted).to(env.device)
        difference=initial-baseline
        self.initial_meta=dict(kind='conditioned_global_bank_0.1mm',
            clipped_joints=[0]*self.n,effective_rms_rad=difference.square().mean(-1).sqrt().cpu().tolist(),
            bank_path=str(bank_path),bank_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest(),
            conditioning='initial nonexempt margins >=0.1mm and no initial violation; first sampling order',
            sampled_joint_fraction=[.025,.975],policy_outcomes_used=False)
        self.positions=dict(zip(ARM_KEYS,initial.split([7,7,6,6],-1)))
        self.limits=limits
        self.baseline=baseline

    def reset(self,env_ids,generator=None):
        if env_ids.numel()!=self.n:raise ValueError('full-window reset only')
        self.index=0

    def sample(self,state):
        if self.index>=len(self.tape):raise ValueError('random tape exhausted')
        value=self.tape[self.index];self.index+=1
        return DeltaCmd(delta_q=dict(zip(ARM_KEYS,value.split([7,7,6,6],-1))))

    def initial_positions(self,ids):
        return ids,{a:q[ids] for a,q in self.positions.items()}

    def coverage_metadata(self):
        return [dict(effective_flow=self.kind,initial_sampling=self.initial_meta['kind'],
                     initial_sampler_seed=self.initial_seed,
                     initial_clipped_joints=self.initial_meta['clipped_joints'][e],
                     initial_effective_rms_rad=self.initial_meta['effective_rms_rad'][e],
                     command_tape_sha256=self.info['tape_sha256']) for e in range(self.n)]


def make_flow(kind,env,amp,env_yaml):
    if kind not in FLOWS:raise ValueError('only registered wide random flows accepted')
    duration=float(sys.argv[sys.argv.index('--duration-s')+1]);dt=env.cfg.sim.dt*env.cfg.decimation
    master=int(sys.argv[sys.argv.index('--seeds')+1])
    return WideSource(env,kind,amp,battery.cell_seed(master,kind,amp),round(duration/dt))


class WideTrace(battery.EpisodeTrace):
    def start(self,env):
        super().start(env)
        self.env=env;self.probe=PairMarginProbe(env._sph)
        self.original_safety=env.safety_dist_out
        self.union_enabled=sys.argv[sys.argv.index('--methods')+1]=='system0'
        self.last_count=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
        self.t=0
        if self.union_enabled:
            def safety():
                selected=self.original_safety()
                try:
                    out=merge_critical_rows(env._last_out,selected,env._sph.class_id,env._sph.pair_id,band=.010,capacity=None)
                    if out.active_idx.shape[1]>512:
                        raise ValueError(f'critical-row union exceeds dynamic budget: {out.active_idx.shape[1]} > 512')
                except ValueError as e:
                    (self.out/'capacity_failure.json').write_text(json.dumps(dict(step=self.t,error=str(e),capacity=512,policy='abort; no row dropping'))+'\n')
                    raise
                self.last_count=out.active_mask.sum(-1)
                return out
            env.safety_dist_out=safety
        self.out=Path(sys.argv[sys.argv.index('--out')+1])
        src=env._delta_src
        info={k:v for k,v in src.info.items() if not isinstance(v,np.ndarray)}
        manifest=dict(schema='safeduo.wide_random_source.v1',flow=src.kind,initial=src.initial_meta,temporal=info,
                      initial_sampler_seed=src.initial_seed,state_feedback=False,collision_rejection=True,selection_stage='initial geometry only; all candidates retained in bank',
                      controlled_joints=26,controlled_arms=4,hand_randomization=False,
                      critical_union=self.union_enabled,band_m=.010,capacity=512,allocation='exact required rows per frame; hard budget512; no dropping',
                      effective_initial='custom sampler; CLI initial-jitter-rad=0 disables legacy jitter only',
                      ee_coordinates='scene_state local positions; environment origins already subtracted',
                      checkpoint_frozen=True,probe_only=False,posthoc_stop=False,
                      source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'wide_random.py',HERE/'dependencies/critical_rows.py']})
        (self.out/'random_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        np.savez_compressed(self.out/'input_recipe.npz',tape=src.tape.cpu().numpy(),updates=src.info['updates'],
                            holds=src.info['holds'],segment_amplitudes=src.info['segment_amplitudes'],
                            q_initial=self.q0.cpu().numpy(),sampled_initial=torch.cat([src.positions[a] for a in ARM_KEYS],-1).cpu().numpy(),
                            baseline_initial=src.baseline.cpu().numpy(),
                            initial_violation=self.initial_violation.cpu().numpy(),joint_soft_limits=self.limits.cpu().numpy(),
                            ee_initial=torch.stack([env.scene_state().ee_pos[a] for a in ARM_KEYS],1).cpu().numpy())
        for k in ['pre_qd_compact','pair_margin','critical_selected_count']:self.frames[k]=[]

    def before_step(self,env,t):
        self.t=t
        state=env.scene_state()
        self.frames['pre_qd_compact'].append(torch.cat([state.qd[a] for a in ARM_KEYS],-1).clone())

    def step(self,env,t):
        super().step(env,t)
        self.frames['pair_margin'].append(self.probe.pair_min().clone())
        self.frames['critical_selected_count'].append(self.last_count.clone())

    def write(self,*args,**kwargs):
        try:return super().write(*args,**kwargs)
        finally:self.env.safety_dist_out=self.original_safety


original_design=battery.build_design
original_summary=battery.summarize_window
original_aggregate=battery.aggregate

def without_iid_intervals(value):
    if isinstance(value,dict):return {k:without_iid_intervals(v) for k,v in value.items() if 'cp95' not in k}
    if isinstance(value,list):return [without_iid_intervals(v) for v in value]
    return value

def summarize_window(*args,**kwargs):
    return without_iid_intervals(original_summary(*args,**kwargs))

def aggregate(*args,**kwargs):
    return without_iid_intervals(original_aggregate(*args,**kwargs))

def build_design(args):
    result=original_design(args)
    for c in result:
        c['initial_sampling']='conditioned_global_bank_0.1mm'
        c['temporal_sampling']='iid_each_step' if c['flow']=='wide_iid' else 'independent_per_arm_mixed_hold'
        c['initial_sampler_seed']=c['seed']
    return result

battery.FLOWS=FLOWS
battery.make_flow=make_flow
battery.build_design=build_design
battery.summarize_window=summarize_window
battery.aggregate=aggregate
battery.EpisodeTrace=WideTrace
if __name__=='__main__':battery.main()
