"""Frozen-policy evaluation on preregistered stable risk strata; no production edits."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'safety_random_space_20261004'))
import wide_runner_feasible as base
from risk_recipe import PREFIX,prepend_zero,validate_bank

class RiskSource(base.WideSource):
    def __init__(self,env,kind,amp,seed,steps):
        super().__init__(env,kind,amp,seed,steps-PREFIX)
        bank_path=Path(os.environ['SAFEDUO_INITIAL_BANK_NPZ'])
        meta=json.loads(bank_path.with_name('metadata.json').read_text())
        with np.load(bank_path,allow_pickle=False) as z:
            self.labels=z['risk_pair_index'].copy()
            validate_bank(z['accepted_q'],self.labels,meta)
        assert hashlib.sha256(bank_path.read_bytes()).hexdigest()==meta['bank_sha256']
        self.initial_meta.update(kind='risk_stable_bank',conditioning=meta['conditioning'],
                                 selection='six targeted risk strata8 each and16 global LHS survivors; first pre-policy qualified sampling order',
                                 risk_pair_index=self.labels.tolist(),policy_outcomes_used=False)
        self.tape,self.info=prepend_zero(self.tape,self.info)
    def coverage_metadata(self):
        rows=super().coverage_metadata()
        for row,label in zip(rows,self.labels):row['risk_pair_index']=int(label)
        return rows

def make_flow(kind,env,amp,env_yaml):
    if kind!='risk_burst':raise ValueError('only registered risk_burst accepted')
    duration=float(sys.argv[sys.argv.index('--duration-s')+1]);dt=env.cfg.sim.dt*env.cfg.decimation
    master=int(sys.argv[sys.argv.index('--seeds')+1])
    return RiskSource(env,kind,amp,base.battery.cell_seed(master,kind,amp),round(duration/dt))

class RiskTrace(base.WideTrace):
    def start(self,env):
        super().start(env)
        manifest=json.loads((self.out/'random_manifest.json').read_text())
        manifest.update(schema='safeduo.risk_strata_source.v1',selection_stage='initial geometry and raw zero-input qualification only; no random-policy outcomes',
                        primary_endpoint='all16s including60 zero-prefix steps; retain every failure',
                        source_sha256={**manifest['source_sha256'],**{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'risk_recipe.py']}})
        (self.out/'random_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        np.savez_compressed(self.out/'bank_assignment.npz',risk_pair_index=env._delta_src.labels,
                            initial_table_raw_min=env._sph.last_table_margin.amin(-1).cpu().numpy())
        self.frames['table_raw_min']=[]
    def step(self,env,t):
        super().step(env,t)
        self.frames['table_raw_min'].append(env._sph.last_table_margin.amin(-1).clone())

def build_design(args):
    result=base.original_design(args)
    for c in result:
        c.update(initial_sampling='risk_stable_bank',temporal_sampling='zero60_then_independent_per_arm_mixed_hold900',initial_sampler_seed=c['seed'])
    return result

base.battery.FLOWS=('risk_burst',)
base.battery.make_flow=make_flow
base.battery.build_design=build_design
base.battery.EpisodeTrace=RiskTrace
if __name__=='__main__':base.battery.main()
