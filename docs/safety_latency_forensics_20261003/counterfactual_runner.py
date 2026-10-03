"""Registered factors: backlog debit and additional critical solver rows."""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

from critical_rows import merge_critical_rows
from trace_runner import ForensicTrace, battery


class FactorTrace(ForensicTrace):
    def start(self, env):
        self.union_enabled = os.environ['SAFEDUO_FORENSIC_CRITICAL_ROWS'] == '1'
        self.original_safety = env.safety_dist_out
        if self.union_enabled:
            def safety():
                selected = self.original_safety()
                self.baseline_rows = selected
                return merge_critical_rows(env._last_out, selected,
                    env._sph.class_id, env._sph.pair_id, band=.010, capacity=128)
            env.safety_dist_out = safety
        super().start(env)
        for key in ['solver_priority_p', 'solver_selected_count', 'solver_added_count']:
            self.frames[key] = []
        assert env._last_out.active_idx.shape[1] == 32
        out=Path(sys.argv[sys.argv.index('--out')+1])
        manifest=dict(schema='safeduo.evaluation_factor.v1',
            original_actor_observation_rows=32,
            critical_union=self.union_enabled,critical_band_m=.010,
            union_capacity=128,capacity_overflow='abort; no silent row dropping',
            backlog_aware=env._backstop.cfg.backlog_aware,
            pending_target_steps=env._backstop.cfg.pending_target_steps,
            actuator_delay_steps=env._evaluation_actuator_delay.queue.steps,
            safety_execution='before unchanged target FIFO',
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__),Path(__file__).with_name('critical_rows.py'),
                                     Path(__file__).with_name('trace_runner.py')]})
        (out/'factor_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

    def step(self, env, t):
        super().step(env, t)
        self.frames['solver_priority_p'].append(env._step_cache['p'].detach().clone())
        selected=self.pre_rows.valid.sum(-1)
        baseline=self.baseline_rows.active_mask.sum(-1) if self.union_enabled else selected
        self.frames['solver_selected_count'].append(selected.clone())
        self.frames['solver_added_count'].append((selected-baseline).clone())

    def write(self,*args,**kwargs):
        try:
            return super().write(*args,**kwargs)
        finally:
            self.env.safety_dist_out=self.original_safety


battery.EpisodeTrace=FactorTrace
if __name__=='__main__': battery.main()
