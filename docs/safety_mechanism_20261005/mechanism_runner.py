"""Isolated mechanism runner; original actor, physics and six-step FIFO."""
import hashlib
import json
import os
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'safety_risk_strata_20261004'))
import risk_runner as risk
from reference_envelope import install, reference_bounds
from safeduo.safety.types import ARM_KEYS


class MechanismTrace(risk.RiskTrace):
    def start(self, env):
        super().start(env)
        self.mode = os.environ.get('SAFEDUO_MECHANISM', 'baseline')
        self.governor_original = None
        if self.mode != 'baseline':
            self.governor_original = install(env, self.mode)
        for name in ('pre_target_debt', 'reference_envelope_unreachable'):
            self.frames[name] = []
        (self.out / 'mechanism_manifest.json').write_text(json.dumps(dict(
            schema='safeduo.reference_governor.v1', mode=self.mode,
            gap_rad=.050 if self.mode == 'envelope_050' else None,
            original_actor_observation_rows=32, original_damper_unchanged=True,
            original_conditional_exemptions=True, queue_preemption=False,
            actuator_delay_steps=6, production_promoted=False,
            target_updates='original integral targets; no rebase; original speed and soft limits',
            envelope_failure='slew to nearest reachable endpoint; no snap; not a safety guarantee',
            governor_input='current actual q and issued integral target; no future commands',
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (Path(__file__), HERE / 'reference_envelope.py')}
        ), indent=2) + '\n')

    def before_step(self, env, t):
        super().before_step(env, t)
        state = env.scene_state()
        debt = torch.cat([env._targets[a] - state.q[a] for a in ARM_KEYS], -1)
        self.frames['pre_target_debt'].append(debt.clone())
        box = env._backstop.cfg.vmax * state.dt
        unreachable = []
        for a in ARM_KEYS:
            lim = env._q_soft_limits[a]
            lo, hi = reference_bounds(state.q[a], env._targets[a], lim[..., 0], lim[..., 1], box)
            unreachable.append(((state.q[a] + .050 < env._targets[a] + lo) |
                                (state.q[a] - .050 > env._targets[a] + hi)).any(-1))
        self.frames['reference_envelope_unreachable'].append(torch.stack(unreachable, -1))

    def write(self, *args, **kwargs):
        try:
            return super().write(*args, **kwargs)
        finally:
            if self.governor_original is not None:
                self.env._backstop.project = self.governor_original


risk.base.battery.EpisodeTrace = MechanismTrace
if __name__ == '__main__':
    risk.base.battery.main()
