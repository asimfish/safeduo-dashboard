"""Opt-in diagnostic driver. Preserve original absolute closed endpoints.

Each registered event linearly interpolates from the current target to its
absolute joint target over ramp_s. Only the zero-fraction open target is replaced.
This changes intermediate timing; it does not assert equivalence of the path.
"""
import torch
from safeduo.safety.types import ARM_KEYS


class AbsoluteHandTargets:
    def __init__(self, env, dt, legacy_metadata, open_thumb):
        self.env, self.dt = env, dt
        self.ids, self.original_open, self.open_q, self.far_q = {}, {}, {}, {}
        self.target, self.start, self.goal, self.elapsed, self.duration = {}, {}, {}, {}, {}
        self.cur = {}
        for arm in ARM_KEYS:
            art = env._arms[arm]
            ids = legacy_metadata['arms'][arm]['hand_ids']
            assert [art.joint_names[i] for i in ids] == [legacy_metadata['arms'][arm]['joint_names'][i] for i in ids]
            old = torch.tensor(legacy_metadata['arms'][arm]['hand_default_rad'][0], device=env.device)
            limits = art.data.soft_joint_pos_limits[0, ids]
            far = torch.where((limits[:, 1]-old).abs() >= (limits[:, 0]-old).abs(), limits[:, 1], limits[:, 0])
            opened = old.clone()
            for j, jid in enumerate(ids):
                if arm.startswith('U') and 'thumb_1_joint' in art.joint_names[jid]:
                    opened[j] = open_thumb
            self.ids[arm] = (ids, torch.tensor(ids, device=env.device))
            self.original_open[arm], self.open_q[arm], self.far_q[arm] = old, opened, far
            self.target[arm] = opened.clone()
            self.start[arm] = opened.clone()
            self.goal[arm] = opened.clone()
            self.elapsed[arm], self.duration[arm], self.cur[arm] = 0., dt, 0.

    def command(self, arm, fraction, ramp_s, fraction_thumb=None):
        if not 0 <= fraction <= 1 or (fraction_thumb is not None and not 0 <= fraction_thumb <= 1):
            raise ValueError('fraction outside registered stroke')
        if ramp_s <= 0:
            raise ValueError('nonpositive ramp')
        names = [self.env._arms[arm].joint_names[i] for i in self.ids[arm][0]]
        fractions = torch.tensor([fraction_thumb if fraction_thumb is not None and 'thumb' in n else fraction for n in names], device=self.env.device)
        # Closed targets use the legacy open reference, never the new neutral reference.
        goal = self.original_open[arm] + fractions * (self.far_q[arm]-self.original_open[arm])
        goal = torch.where(fractions == 0, self.open_q[arm], goal)
        self.start[arm], self.goal[arm] = self.target[arm].clone(), goal
        self.elapsed[arm], self.duration[arm], self.cur[arm] = 0., ramp_s, fraction

    def step(self):
        for arm in self.ids:
            self.elapsed[arm] = min(self.duration[arm], self.elapsed[arm]+self.dt)
            alpha = self.elapsed[arm]/self.duration[arm]
            # Exact endpoint assignment avoids endpoint floating-point cancellation.
            self.target[arm] = self.goal[arm].clone() if alpha == 1 else self.start[arm] + alpha*(self.goal[arm]-self.start[arm])
            self.env._arms[arm].set_joint_position_target(self.target[arm][None].expand(self.env.num_envs, -1), joint_ids=self.ids[arm][0])
