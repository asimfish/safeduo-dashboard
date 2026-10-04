"""Separate v2 diagnostic: fail closed on nonfinite predictions, stable-row receipt.

The frozen v1 holdout is untouched. This short development prefix adds validation
and observation only; no admission threshold or valid numeric output is changed.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

import mechanism_runner as v1
from safeduo.safety.types import ARM_KEYS

ROWS=[8974,7310]
SLOTS=[14,35,47]


def require_finite(label, tensors):
    for key,value in tensors.items():
        if not torch.isfinite(value).all():
            raise ValueError(f'nonfinite {label}.{key}; abort before projection/physics')


class GuardedTrace(v1.MechanismTrace):
    def start(self,env):
        super().start(env)
        if self.kind!='queue_envelope':raise ValueError('guard diagnostic only')
        self.receipt={}
        self.original_safety_v1=env.safety_dist_out
        self.original_rows=env._provider.rows_from
        self.original_union=v1.union_mask
        self.slots=torch.tensor(SLOTS,device=env.device)
        self.row_ids=torch.tensor(ROWS,device=env.device)
        def union(full,selected,forecast,band=.010):
            require_finite('geometry',dict(d=full.dists,dmin=full.full_dmin,closing=full.closing,forecast=forecast))
            self.forecast=forecast
            return self.original_union(full,selected,forecast,band)
        def rows(out,body):
            result=self.original_rows(out,body)
            require_finite('jacobian',result.J)
            return result
        def safety():
            state=env.scene_state()
            require_finite('q',state.q);require_finite('qd',state.qd)
            require_finite('issued',env._targets)
            require_finite('proposal',env._pending_cmd.delta_q)
            for i,target in enumerate(env._evaluation_actuator_delay.queue.pending):
                require_finite(f'pending_{i}',target)
            result=self.original_safety_v1()
            self.receipt.setdefault('row_forecast',[]).append(self.forecast[self.slots][:,self.row_ids].cpu().numpy().copy())
            admitted=(result.active_idx[:,:,None]==self.row_ids[None,None,:])&result.active_mask[:,:,None]
            self.receipt.setdefault('row_admitted',[]).append(admitted.any(1)[self.slots].cpu().numpy().copy())
            self.receipt.setdefault('row_distance',[]).append(env._last_out.dists[self.slots][:,self.row_ids].cpu().numpy().copy())
            for name,targets in [('actual_queue',list(env._evaluation_actuator_delay.queue.pending)),('issued',[env._targets])]:
                self.receipt.setdefault(name,[]).append(torch.stack([torch.cat([t[a] for a in ARM_KEYS],-1) for t in targets],1)[self.slots].cpu().numpy().copy())
            return result
        v1.union_mask=union
        env._provider.rows_from=rows
        env.safety_dist_out=safety
        (self.out/'guard_metadata.json').write_text(json.dumps(dict(schema='safeduo.finite_guard_diagnostic.v2',
          row_ids=ROWS,env_ids=SLOTS,all_full_rows_checked=True,steps=120,
          no_parameter_change=True,frozen_v1_holdout_untouched=True,
          scope='development prefix; not additional complete16s windows',
          source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')

    def write(self,*args,**kwargs):
        try:
            np.savez_compressed(self.out/'row_receipt.npz',**{k:np.stack(v) for k,v in self.receipt.items()})
            return super().write(*args,**kwargs)
        finally:
            self.env._provider.rows_from=self.original_rows
            v1.union_mask=self.original_union


OriginalSource=v1.risk.RiskSource
class PrefixSource(OriginalSource):
    def __init__(self,env,kind,amp,seed,steps):
        if steps!=120:raise ValueError('registered prefix is120 steps')
        super().__init__(env,kind,amp,seed,960)

v1.risk.RiskSource=PrefixSource
v1.risk.base.battery.EpisodeTrace=GuardedTrace
if __name__=='__main__':v1.risk.base.battery.main()
