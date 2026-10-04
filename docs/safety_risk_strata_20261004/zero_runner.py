"""Zero-command diagnostics on exactly the previous registered initial banks."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'safety_random_space_20261004'
sys.path.insert(0,str(BASE))
import wide_runner_feasible as base

original_design=base.battery.build_design
base.FLOWS=('feasible_zero',)


def make_flow(kind,env,amp,env_yaml):
    assert amp==0.,'zero input diagnostic only'
    source=base.make_flow(kind,env,.05,env_yaml)
    source.tape.zero_()
    length,n,_=source.tape.shape
    updates=np.zeros((length,n,4),bool);updates[0]=True
    holds=np.zeros((length,n,4),np.int16);holds[0]=length
    scales=np.zeros((length,n,4),np.float32)
    source.info=dict(kind='zero_hold',seed=None,hold_choices=[length],amplitude_choices=[0.],
                     updates=updates,holds=holds,segment_amplitudes=scales,
                     tape_sha256=hashlib.sha256(source.tape.cpu().numpy().tobytes()).hexdigest(),
                     streams='no random commands; all 26 joints exactly zero')
    return source


def snapshot(env):
    state=env.scene_state();table=env._sph.last_table_margin;exempt=env._sph.last_table_viol_exempt
    official=torch.stack([env._last_out.min_margin[k] for k in ['cross','self_F','self_U','table']],-1)
    official[:,-1]=table.masked_fill(exempt,float('inf')).amin(-1)
    return dict(q=torch.cat([state.q[a] for a in base.ARM_KEYS],-1).clone(),
                qd=torch.cat([state.qd[a] for a in base.ARM_KEYS],-1).clone(),
                official_margins=official.clone(),violation=env._last_out.violation.clone(),
                centers=env._sph.last_centers.clone(),table_margin=table.clone(),table_exempt=exempt.clone())


class ZeroTrace(base.WideTrace):
    def start(self,env):
        super().start(env)
        before=snapshot(env);counter=env._sim_step_counter
        env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
        env.compute_dist(need_full=True)
        after=snapshot(env)
        assert env._sim_step_counter==counter,'zero-time probe advanced physics counter'
        np.savez_compressed(self.out/'zero_time_geometry.npz',**{f'{stage}_{k}':v.cpu().numpy() for stage,d in [('before',before),('after',after)] for k,v in d.items()})
        (self.out/'zero_probe.json').write_text(json.dumps(dict(
            source='zero input, no safety policy',sim_counter_before=counter,sim_counter_after=env._sim_step_counter,
            sphere_names=env._sph.qualified_names,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))+'\n')
        self.initial_snapshot=after
        for key in ['table_margin_all','table_exempt_mask']:self.frames[key]=[]

    def step(self,env,t):
        super().step(env,t)
        self.frames['table_margin_all'].append(env._sph.last_table_margin.clone())
        self.frames['table_exempt_mask'].append(env._sph.last_table_viol_exempt.clone())


def build_design(args):
    result=original_design(args)
    for cell in result:cell['temporal_sampling']='zero_hold'
    return result


base.battery.FLOWS=base.FLOWS
base.battery.make_flow=make_flow
base.battery.build_design=build_design
base.battery.EpisodeTrace=ZeroTrace
if __name__=='__main__':base.battery.main()
