"""Observe a frozen research battery; return every original control output unchanged."""
import torch

from safeduo.eval import research_battery as battery
from safeduo.safety.types import ARM_KEYS, DOF_OF


class ForensicTrace(battery.EpisodeTrace):
    def start(self, env):
        super().start(env)
        keys = ('pre_row_J_F', 'pre_row_J_U', 'pre_row_arm_mask',
                'pre_pending_targets', 'pre_struct_exempt', 'pre_bypass_arm',
                'solver_cap', 'delivered_row_backlog', 'issued_row_backlog',
                'full_table_distance', 'full_table_exempt')
        for key in keys:
            self.frames[key] = []
        self.env = env
        self.original_project = env._backstop.project

        def observe(*args, **kwargs):
            result = self.original_project(*args, **kwargs)
            self.calls += 1
            self.observed = {
                'solver_cap': result[2]['cap'],
                'pre_struct_exempt': kwargs['struct_exempt'],
                'pre_bypass_arm': kwargs['bypass_arm'],
            }
            return result

        env._backstop.project = observe

    def before_step(self, env, t):
        super().before_step(env, t)
        self.calls = 0
        self.q_before = self.frames['pre_q'][-1]
        self.j_before = self.pre_rows.J
        history = env._pending_target_history
        assert history is not None
        targets = torch.stack([
            torch.cat([past[a] for a in ARM_KEYS], -1)
            for past in history.targets
        ], 1)
        for key, value in {
            'pre_row_J_F': self.pre_rows.J['F'],
            'pre_row_J_U': self.pre_rows.J['U'],
            'pre_row_arm_mask': self.pre_rows.arm_mask,
            'pre_pending_targets': targets,
        }.items():
            self.frames[key].append(value.detach().clone())

    def step(self, env, t):
        super().step(env, t)
        assert self.calls == 1, self.calls
        values = dict(self.observed)
        if values['pre_struct_exempt'] is None:
            values['pre_struct_exempt'] = torch.zeros_like(self.pre_rows.valid)
        if values['pre_bypass_arm'] is None:
            values['pre_bypass_arm'] = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.bool)
        delay = env._evaluation_actuator_delay
        delivered = delay.applied if delay.queue.steps else env._targets
        split = DOF_OF['F_L'] + DOF_OF['F_R']
        for key, target in [('delivered_row_backlog', delivered), ('issued_row_backlog', env._targets)]:
            backlog = torch.cat([target[a] for a in ARM_KEYS], -1) - self.q_before
            values[key] = (torch.einsum('nmd,nd->nm', self.j_before['F'], backlog[:, :split])
                           + torch.einsum('nmd,nd->nm', self.j_before['U'], backlog[:, split:]))
        values['full_table_distance'] = env._sph.last_table_margin
        values['full_table_exempt'] = env._sph.last_table_viol_exempt
        for key, value in values.items():
            self.frames[key].append(value.detach().clone())

    def write(self, *args, **kwargs):
        try:
            return super().write(*args, **kwargs)
        finally:
            self.env._backstop.project = self.original_project


battery.EpisodeTrace = ForensicTrace
if __name__ == '__main__':
    battery.main()
