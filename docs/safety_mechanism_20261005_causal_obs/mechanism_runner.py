"""Evaluation-only queue-envelope admission, with original actor and projection."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'safety_risk_strata_20261004'))
import risk_runner as risk
from safeduo.baselines.base import stack_robot
from safeduo.safety.types import ARM_KEYS

CAPACITY = 1024


def union_mask(full, selected, forecast, band=.010):
    admitted = torch.zeros_like(full.dists, dtype=torch.long)
    admitted.scatter_add_(1, selected.active_idx.clamp_min(0), selected.active_mask.long())
    return (admitted > 0) | (full.dists <= full.full_dmin + band) | (forecast <= full.full_dmin + band)


def select(full, selected, mask, sph, capacity=CAPACITY):
    width = max(1, int(mask.sum(-1).max().item()))
    if width > capacity:
        raise ValueError(f'queue-envelope union exceeds declared capacity: {width}>{capacity}; no dropping')
    n, p = mask.shape
    ids = torch.arange(p, device=mask.device).expand(n,-1)
    chosen = ids.masked_fill(~mask,p).topk(width,dim=1,largest=False).values
    valid = chosen < p
    idx = chosen.clamp_max(p-1)
    features = torch.stack([full.dists.gather(1,idx),full.closing.gather(1,idx),sph.class_id[idx],sph.pair_id[idx]],-1)
    features = features.masked_fill(~valid.unsqueeze(-1),0)
    features[...,3] = features[...,3].masked_fill(~valid,-1)
    return replace(selected, active_pairs=features,active_mask=valid,active_idx=idx.masked_fill(~valid,-1),
                   active_dmin=full.full_dmin.gather(1,idx).masked_fill(~valid,0),
                   viol_exempt=full.full_viol_exempt.gather(1,idx)&valid)


class MechanismTrace(risk.RiskTrace):
    def start(self, env):
        super().start(env)
        kind = os.environ['SAFEDUO_MECHANISM']
        if kind not in ['baseline','queue_envelope']:
            raise ValueError('unregistered mechanism')
        self.extra = {key:[] for key in ['forecast_added_count','forecast_only_min','required_rows']}
        self.last_added = torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
        self.last_only = torch.zeros(env.num_envs,device=env.device)
        self.kind = kind
        if kind == 'queue_envelope':
            def safety():
                full = env._last_out
                base = self.original_safety()
                p = full.dists.shape[1]
                ids = torch.arange(p,device=env.device).expand(env.num_envs,-1)
                all_rows = replace(full,active_idx=ids,active_mask=torch.ones_like(ids,dtype=torch.bool),
                                   active_pairs=torch.stack([full.dists,full.closing,
                                                           env._sph.class_id.expand_as(full.dists),
                                                           env._sph.pair_id.expand_as(full.dists)],-1),
                                   active_dmin=full.full_dmin,viol_exempt=full.full_viol_exempt)
                rows = env._provider.rows_from(all_rows,env._body_pos_cache)
                state = env.scene_state()
                forecast = full.dists.clone()
                targets = list(env._evaluation_actuator_delay.queue.pending)+[env._targets]
                cmd = env._pending_cmd
                if cmd is None:
                    raise ValueError('state-independent pending command required')
                box = env._backstop.cfg.vmax*state.dt
                proposal = {a:(env._targets[a]+cmd.delta_q[a].clamp(-box,box)).clamp(
                                  env._q_soft_limits[a][...,0],env._q_soft_limits[a][...,1]) for a in ARM_KEYS}
                targets.append(proposal)
                for target in targets:
                    displacement = {a:target[a]-state.q[a] for a in ARM_KEYS}
                    delta = sum(torch.einsum('nmd,nd->nm',rows.J[r],stack_robot(displacement,r)) for r in ('F','U'))
                    forecast = torch.minimum(forecast,full.dists+delta)
                baseline_mask = union_mask(full,base,full.dists)
                mask = union_mask(full,base,forecast)
                added = mask & ~baseline_mask
                self.last_added = added.sum(-1)
                self.last_only = forecast.masked_fill(~added,float('inf')).min(-1).values
                self.last_only = torch.where(added.any(-1),self.last_only,torch.zeros_like(self.last_only))
                try:
                    result = select(full,base,mask,env._sph)
                except ValueError as e:
                    (self.out/'capacity_failure.json').write_text(json.dumps(dict(step=self.t,capacity=CAPACITY,error=str(e)))+'\n')
                    raise
                self.last_count = result.active_mask.sum(-1)
                return result
            env.safety_dist_out = safety
        manifest = json.loads((self.out/'random_manifest.json').read_text())
        manifest.update(mechanism=kind,capacity=CAPACITY if kind=='queue_envelope' else 512,
                        forecast='min current full d and same-row linear d at actual pending targets, current issued target, and next boxed/soft-limited proposal',
                        projection_unchanged=True,actor_observation_rows=32,strict_fifo_steps=6,
                        capacity_reason='candidate includes all additional forecast-dangerous rows; abort on >1024, no truncation',
                        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (self.out/'random_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

    def step(self,env,t):
        super().step(env,t)
        for k,v in [('forecast_added_count',self.last_added),('forecast_only_min',self.last_only),('required_rows',self.last_count)]:
            self.extra[k].append(v.cpu().numpy().copy())

    def write(self,*args,**kwargs):
        np.savez_compressed(self.out/'mechanism.npz',**{k:np.stack(v) for k,v in self.extra.items()})
        return super().write(*args,**kwargs)


risk.base.battery.EpisodeTrace = MechanismTrace
if __name__ == '__main__':risk.base.battery.main()
