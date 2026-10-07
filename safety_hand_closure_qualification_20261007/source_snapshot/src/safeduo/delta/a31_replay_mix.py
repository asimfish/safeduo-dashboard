"""A31 joint-space rehearsal mixed with the existing conflict curriculum.

Coupling intervals are explicit scripted task-intent labels. These episodes
do NOT simulate grasped objects and must not count as physical task success.
Physical co-holding evaluation supplies measured grasp state separately.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import torch

from safeduo.delta.skill_replay import SkillTrajectory, SkillReplayDelta, SkillNoiseParams
from safeduo.safety.types import ARM_KEYS, DeltaCmd
from safeduo.safety.coupling import coupling_from_grasps


class A31ReplayMix:
    def __init__(self, base, n_envs, manifest, device):
        self.base = base
        self.device = device
        self.n = n_envs
        path = Path(manifest)
        self.entries = json.loads(path.read_text())["trajectories"]
        trajectories = []
        for entry in self.entries:
            trajectory_path = path.parent / entry["path"]
            if hashlib.sha256(trajectory_path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"a31 trajectory digest mismatch: {trajectory_path}")
            trajectories.append(SkillTrajectory.load(trajectory_path))
        self.replay = SkillReplayDelta(n_envs, trajectories,
            params=SkillNoiseParams(), device=device, auto_advance=False)
        self.selected = torch.arange(n_envs, device=device) % 4 == 0
        self.clock = torch.zeros(n_envs, device=device)
        self.last_graph = torch.zeros(n_envs, 4, 4, dtype=torch.bool, device=device)
        self._traj = trajectories

    @property
    def mix_fractions(self):
        fraction = float(self.selected.float().mean())
        return {"s9_joint_rehearsal": fraction,
                **{name: value * (1 - fraction)
                   for name, value in self.base.mix_fractions.items()}}

    def set_progress(self, fraction):
        self.base.set_progress(fraction)

    def reset(self, env_ids, generator=None):
        self.base.reset(env_ids, generator)
        self.replay.reset(env_ids, generator)
        durations = self.replay._n_steps[ self.replay.assignment[env_ids]] * self.replay._dt_traj[self.replay.assignment[env_ids]]
        self.clock[env_ids] = torch.rand(len(env_ids), device=self.device,
                                        generator=generator) * durations * .8
        self.last_graph[env_ids] = False

    def initial_positions(self, env_ids):
        ids = env_ids[self.selected[env_ids]]
        return ids, self.reference(self.clock, ids)

    def reference(self, times, ids):
        which = self.replay.assignment[ids]
        tick = times[ids] / self.replay._dt_traj[which]
        lo = torch.minimum(tick.long(), self.replay._n_steps[which])
        hi = torch.minimum(lo + 1, self.replay._n_steps[which])
        frac = (tick - lo).clamp(0, 1).unsqueeze(-1)
        return {a: torch.lerp(self.replay._qlib[a][which, lo],
                              self.replay._qlib[a][which, hi], frac) for a in ARM_KEYS}

    def sample(self, state):
        base = self.base.sample(state)
        ids = torch.arange(self.n, device=self.device)
        q0 = self.reference(self.clock, ids)
        q1 = self.reference(self.clock + state.dt, ids)
        object_ids = torch.full((self.n, 4), -1, dtype=torch.long, device=self.device)
        for k, entry in enumerate(self.entries):
            for interval in entry.get("coupling_intervals", []):
                active = (self.selected & (self.replay.assignment == k)
                          & (self.clock >= interval["start_s"])
                          & (self.clock < interval["end_s"]))
                for arm in interval["arms"]:
                    object_ids[active, ARM_KEYS.index(arm)] = interval["object_id"]
        present = object_ids >= 0
        self.last_graph = coupling_from_grasps(object_ids, present, present)
        self.clock += state.dt
        return DeltaCmd(delta_q={a: torch.where(self.selected[:, None],
                    q1[a] - q0[a], base.delta_q[a]) for a in ARM_KEYS})
