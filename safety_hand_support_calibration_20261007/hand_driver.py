"""Unmodified HandDriver class extracted from the sealed v4 recorder."""
import math
import torch
from safeduo.safety.types import ARM_KEYS
HAND_RE=".*(thumb|index|middle|ring|pinky|little).*"

class HandDriver:
    """Ramped open/close position targets on the hand actuator joints.

    'closed' per joint = the soft-limit endpoint farthest from the default
    (open) pose; events carry a fraction of that stroke. Arm joints are
    untouched (duo_env only ever writes its own arm joint ids)."""

    def __init__(self, env, ctrl_dt: float):
        self.env = env
        self.dt = ctrl_dt
        (self.ids, self.open_q, self.far_q) = ({}, {}, {})
        (self.cur, self.goal, self.rate) = ({}, {}, {})
        (self.is_thumb, self.cur_thumb, self.goal_thumb) = ({}, {}, {})
        for arm in ARM_KEYS:
            art = env._arms[arm]
            (ids, names) = art.find_joints(HAND_RE)
            if not ids:
                continue
            t_ids = torch.tensor(ids, dtype=torch.long, device=env.device)
            lim = art.data.soft_joint_pos_limits[0, t_ids, :]
            openq = art.data.default_joint_pos[0, t_ids].clone()
            far = torch.where((lim[:, 1] - openq).abs() >= (lim[:, 0] - openq).abs(), lim[:, 1], lim[:, 0])
            self.ids[arm] = (ids, t_ids)
            self.open_q[arm] = openq
            self.far_q[arm] = far
            self.cur[arm] = 0.0
            self.goal[arm] = 0.0
            self.rate[arm] = 1.0 / 0.6
            self.is_thumb[arm] = torch.tensor(['thumb' in n for n in names], device=env.device)
            self.cur_thumb[arm] = 0.0
            self.goal_thumb[arm] = None
            gains = ''
            for (_an, _act) in getattr(art, 'actuators', {}).items():
                if 'hand' in _an:
                    try:
                        gains = f' hand_gains={_an}:kp={float(_act.stiffness[0, 0]):.1f}/kd={float(_act.damping[0, 0]):.2f}/eff={float(_act.effort_limit[0, 0]):.1f}'
                    except Exception as _ex:
                        gains = f' hand_gains={_an}:unavailable({type(_ex).__name__})'
            print(f'HAND_GROUP {arm}: {len(ids)} joints stroke_mean={(far - openq).abs().mean():.2f} rad ({names[0]}..){gains}', flush=True)

    def command(self, arm: str, frac: float, ramp_s: float, frac_thumb: 'float | None'=None):
        if arm in self.goal:
            self.goal[arm] = float(frac)
            self.rate[arm] = 1.0 / max(ramp_s, 0.001)
            self.goal_thumb[arm] = None if frac_thumb is None else float(frac_thumb)

    def step(self):
        for arm in self.ids:
            d = self.goal[arm] - self.cur[arm]
            if abs(d) > 1e-06:
                self.cur[arm] += math.copysign(min(abs(d), self.rate[arm] * self.dt), d)
            gt = self.goal_thumb.get(arm)
            if gt is None:
                self.cur_thumb[arm] = self.cur[arm]
            else:
                dt_ = gt - self.cur_thumb[arm]
                if abs(dt_) > 1e-06:
                    self.cur_thumb[arm] += math.copysign(min(abs(dt_), self.rate[arm] * self.dt), dt_)
            frac_vec = torch.where(self.is_thumb[arm], torch.full_like(self.open_q[arm], self.cur_thumb[arm]), torch.full_like(self.open_q[arm], self.cur[arm]))
            tgt = self.open_q[arm] + frac_vec * (self.far_q[arm] - self.open_q[arm])
            self.env._arms[arm].set_joint_position_target(tgt.unsqueeze(0).expand(self.env.num_envs, -1), joint_ids=self.ids[arm][0])
