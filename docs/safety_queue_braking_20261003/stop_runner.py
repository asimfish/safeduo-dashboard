"""Post-hoc diagnostic holds; all hold targets traverse the unchanged FIFO."""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

OLD = Path(__file__).resolve().parent.parent / 'safety_latency_forensics_20261003'
sys.path.insert(0, str(OLD))
from counterfactual_runner import FactorTrace, battery
from safeduo.safety.types import ARM_KEYS, DOF_OF
from stop_latch import StopLatch


def stack(target):
    return torch.cat([target[a] for a in ARM_KEYS], -1)


class StopTrace(FactorTrace):
    def start(self, env):
        super().start(env)
        schedule_path = Path(os.environ['SAFEDUO_STOP_SCHEDULE'])
        recipe = json.loads(schedule_path.read_text())
        mode = os.environ['SAFEDUO_STOP_MODE']
        self.latch = StopLatch(recipe['trigger_steps'], mode)
        delay = env._evaluation_actuator_delay
        assert delay.queue.steps == 6, 'diagnostic hold requires unchanged strict 6-step FIFO'
        self.original_delay_pre = delay.original_pre
        for key in ['stop_active', 'stop_fixed_target', 'stop_proposed_target', 'stop_sent_target']:
            self.frames[key] = []

        def intervene(actions):
            self.original_delay_pre(actions)
            proposed = stack(env._targets)
            sent = self.latch.update(self.t, stack(self.pre_targets), self.q_before, proposed)
            for a, q in zip(ARM_KEYS, sent.split([DOF_OF[a] for a in ARM_KEYS], dim=-1)):
                env._targets[a] = q.clone()
            # The latest issue-history entry must describe the actual issued target.
            # Neither the real FIFO nor any older pending entry is touched.
            env._pending_target_history.targets[-1] = env._pending_target_history.clone(env._targets)
            self.proposed = proposed.detach().clone()
            self.sent = sent.detach().clone()

        delay.original_pre = intervene
        out = Path(sys.argv[sys.argv.index('--out') + 1])
        manifest = dict(schema='safeduo.fixed_hold_diagnostic.v1', mode=mode,
                        trigger_recipe=recipe, schedule_sha256=hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
                        hold='one sample at pre-step, fixed thereafter; all four controlled arms in selected environments',
                        fifo_steps=6, queue_preemption=False, actuator_side_control=False,
                        hand_control='unchanged; hand joints are not held by this probe',
                        interpretation='post-hoc diagnostic only; no prevention success rate',
                        exec_trace='solver exec is before target override; applied_margin_delta and sent_target record actual issue',
                        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in [Path(__file__), Path(__file__).with_name('stop_latch.py')]})
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


battery.EpisodeTrace = StopTrace
if __name__ == '__main__':
    battery.main()
