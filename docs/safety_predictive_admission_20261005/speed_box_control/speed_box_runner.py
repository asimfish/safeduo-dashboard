"""Isolated speed-limit control; no actor gating or geometry projection."""
import hashlib
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'safety_risk_strata_20261004'))
import risk_runner as risk
from safeduo.safety.types import ARM_KEYS, DeltaCmd


class SpeedBoxShim:
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def project(self, cmd, rows, alpha, p, dt, **kwargs):
        box = self.cfg.vmax * dt
        if not (box > 0 and box < float('inf')):
            raise ValueError('speed box must be finite and positive')
        value = {a: cmd.delta_q[a].clamp(-box, box) for a in ARM_KEYS}
        active = torch.stack([(value[a] != cmd.delta_q[a]).any(-1) for a in ARM_KEYS], -1)
        zero = torch.zeros(alpha.shape[0], dtype=alpha.dtype, device=alpha.device)
        return DeltaCmd(delta_q=value), active, {'residual_F': zero, 'residual_U': zero.clone()}


class SpeedTrace(risk.RiskTrace):
    def start(self, env):
        assert sys.argv[sys.argv.index('--methods') + 1] == 'backstop_only'
        super().start(env)
        self.original_backstop = env._backstop
        env._backstop = SpeedBoxShim(env._backstop)
        assert not self.union_enabled
        (self.out / 'speed_box_manifest.json').write_text(json.dumps(dict(
            schema='safeduo.speed_box_control.v1', speed_box_only=True,
            original_vmax_rad_s=env._backstop.cfg.vmax, original_soft_target_limits=True,
            actor_gating=False, geometry_projection=False, no_geometry_residual_evaluated=True,
            strict_fifo_steps=6, queue_preemption=False, production_promoted=False,
            runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ), indent=2) + '\n')

    def write(self, *args, **kwargs):
        try:
            return super().write(*args, **kwargs)
        finally:
            self.env._backstop = self.original_backstop


risk.base.battery.EpisodeTrace = SpeedTrace
if __name__ == '__main__':
    risk.base.battery.main()
