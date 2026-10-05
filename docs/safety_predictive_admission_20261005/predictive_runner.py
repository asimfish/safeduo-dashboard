"""Frozen factorial runner: online full-body admission and original envelope."""
import hashlib
import json
import os
from pathlib import Path
import sys
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'safety_mechanism_20261005'))
from mechanism_runner import MechanismTrace,risk
from reference_envelope import install
from predictive_rows import merge_predictive_rows,HORIZONS,CAPACITY

class PredictiveTrace(MechanismTrace):
    def start(self,env):
        self.factor=os.environ['SAFEDUO_ADMISSION_MODE']
        assert self.factor in ('raw','baseline','envelope_050','predictive','predictive_envelope')
        # Reuse unchanged original wrapper for the fixed envelope factor.
        os.environ['SAFEDUO_MECHANISM']='envelope_050' if self.factor in ('envelope_050','predictive_envelope') else 'baseline'
        super().start(env)
        self.predictive=self.factor in ('predictive','predictive_envelope')
        self.frozen_union=env.safety_dist_out
        self.admission_stats={k:torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
                              for k in ('required','additional','predicted_missing','legacy_missing')}
        self.actor_width=env._last_out.active_idx.shape[1]
        assert self.actor_width==32
        if self.predictive:
            def admission():
                selected=self.frozen_union()
                try:
                    out,stats=merge_predictive_rows(env._last_out,selected,env._sph.class_id,env._sph.pair_id)
                except ValueError as e:
                    (self.out/'predictive_capacity_failure.json').write_text(json.dumps(dict(step=self.t,error=str(e),capacity=CAPACITY,policy='abort; no truncation'))+'\n')
                    raise
                self.admission_stats=stats;self.last_count=out.active_mask.sum(-1)
                assert env._last_out.active_idx.shape[1]==32,'actor geometry width changed'
                return out
            env.safety_dist_out=admission
        for k in self.admission_stats:self.frames['admission_'+k]=[]
        (self.out/'admission_manifest.json').write_text(json.dumps(dict(
            schema='safeduo.predictive_admission.v1',mode=self.factor,predictive=self.predictive,
            envelope_gap_rad=.050 if self.factor in ('envelope_050','predictive_envelope') else None,
            horizons_s=HORIZONS,band_m=.010,capacity=CAPACITY,allocation='exact dynamic union; abort on overflow',
            admission_rate='online full body velocity; full.closing; no future targets/commands',
            original_actor_observation_rows=32,original_projection_and_exemptions=True,
            strict_fifo_steps=6,queue_preemption=False,production_promoted=False,
            source_sha256={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('predictive_runner.py','predictive_rows.py')}
        ),indent=2)+'\n')

    def step(self,env,t):
        super().step(env,t)
        for k,v in self.admission_stats.items():self.frames['admission_'+k].append(v.detach().clone())

    def write(self,*args,**kwargs):
        try:return super().write(*args,**kwargs)
        finally:self.env.safety_dist_out=self.original_safety

risk.base.battery.EpisodeTrace=PredictiveTrace
if __name__=='__main__':risk.base.battery.main()
