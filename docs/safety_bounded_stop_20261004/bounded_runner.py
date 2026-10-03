"""Bounded post-hoc hold diagnostics, before the unchanged target FIFO."""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'dependencies'))
from counterfactual_runner import FactorTrace, battery
from safeduo.safety.types import ARM_KEYS, DOF_OF
from bounded_stop import BoundedStopLatch


def stack(target):
    return torch.cat([target[a] for a in ARM_KEYS], -1)


class BoundedTrace(FactorTrace):
    def start(self, env):
        super().start(env)
        schedule_path = Path(os.environ['SAFEDUO_STOP_SCHEDULE'])
        recipe = json.loads(schedule_path.read_text())
        dt = env.cfg.sim.dt * env.cfg.decimation
        limit = env._backstop.cfg.vmax * dt
        self.latch = BoundedStopLatch(recipe['trigger_steps'], limit, self.limits[..., 0], self.limits[..., 1])
        delay = env._evaluation_actuator_delay
        assert delay.queue.steps == 6, 'unchanged strict 6-step FIFO required'
        self.original_delay_pre = delay.original_pre
        for key in ['stop_active', 'stop_fixed_target', 'stop_proposed_target', 'stop_sent_target']:
            self.frames[key] = []

        def intervene(actions):
            self.original_delay_pre(actions)
            proposed = stack(env._targets)
            prior = stack(self.pre_targets)
            sent = self.latch.update(self.t, prior, self.q_before, proposed)
            assert torch.isfinite(sent).all()
            assert (torch.abs(sent - prior) <= limit + 5e-7).all(), 'actual output exceeds ordinary increment box'
            assert ((sent >= self.limits[..., 0] - 5e-7) & (sent <= self.limits[..., 1] + 5e-7)).all()
            for a, q in zip(ARM_KEYS, sent.split([DOF_OF[a] for a in ARM_KEYS], dim=-1)):
                env._targets[a] = q.clone()
            env._pending_target_history.targets[-1] = env._pending_target_history.clone(env._targets)
            self.proposed, self.sent = proposed.detach().clone(), sent.detach().clone()

        delay.original_pre = intervene
        out = Path(sys.argv[sys.argv.index('--out') + 1])
        manifest = dict(schema='safeduo.bounded_hold_diagnostic.v1',
                        trigger_recipe=recipe, schedule_sha256=hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
                        target='once-sampled measured pose clipped to soft limits; bounded approach from last issued target',
                        increment_box_rad=limit, numeric_tolerance_rad=5e-7,
                        distance_projection='probe target is not projected against original distance rows',
                        fifo_steps=6, queue_preemption=False, actuator_side_control=False,
                        hand_control='unchanged; uncommanded hand joints are not held',
                        interpretation='post-hoc diagnostic; no prospective prevention claim or independent safety rate',
                        exec_trace='solver exec before override; controller_target/stop_sent_target are actual issued targets',
                        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in [Path(__file__), HERE / 'bounded_stop.py']})
        (out / 'stop_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def before_step(self, env, t):
        self.t = t
        super().before_step(env, t)

    def step(self, env, t):
        super().step(env, t)
        self.frames['stop_active'].append(self.latch.active.clone())
        self.frames['stop_fixed_target'].append(self.latch.fixed.clone())
        self.frames['stop_proposed_target'].append(self.proposed.clone())
        self.frames['stop_sent_target'].append(self.sent.clone())

    def write(self, *args, **kwargs):
        try:
            return super().write(*args, **kwargs)
        finally:
            self.env._evaluation_actuator_delay.original_pre = self.original_delay_pre


battery.EpisodeTrace = BoundedTrace
if __name__ == '__main__':
    battery.main()
