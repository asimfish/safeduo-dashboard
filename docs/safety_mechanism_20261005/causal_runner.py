"""Isolated passive observer of the original dynamic512/FIFO experiment."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'safety_risk_strata_20261004'))
import risk_runner as risk
from safeduo.baselines.base import stack_robot
from safeduo.safety.types import ARM_KEYS

DETAIL = [0, 14, 20, 29, 35, 39, 47, 54]


class CausalTrace(risk.RiskTrace):
    def __init__(self, causal=False, kinematic=False, spheres=False):
        # The inherited selected-row width is dynamic. Keep its compact logger and
        # record a separate, fixed-width causal schema rather than stack ragged rows.
        super().__init__(causal=False, kinematic=False, spheres=spheres)

    def save(self, key, value):
        self.details.setdefault(key, []).append(value.detach().cpu().numpy().copy())

    def start(self, env):
        super().start(env)
        self.details = {}
        self.detail = torch.tensor(DETAIL, device=env.device)
        sph = env._sph
        classes = sph.class_id.long().clone()
        classes[sph._slice_self] = 1 + sph.self_robot.long()
        classes[sph._slice_table] = 3
        self.classes = classes
        self.original_union = env.safety_dist_out
        self.original_project = env._backstop.project

        def safety():
            result = self.original_union()
            self.selection = result
            return result

        def project(cmd, rows, alpha, p, dt, **kwargs):
            selected = self.selection
            self.rows = rows
            self.save('selected_id', self.pad(selected.active_idx[self.detail], -1))
            self.save('selected_valid', self.pad(rows.valid[self.detail], False))
            self.save('selected_J', self.pad(torch.cat([rows.J['F'], rows.J['U']], -1)[self.detail], 0))
            for key, value in [('selected_d', rows.d), ('selected_dmin', rows.d_min),
                               ('selected_exempt', selected.viol_exempt),
                               ('selected_structural', kwargs['struct_exempt'])]:
                self.save(key, self.pad(value[self.detail], 0))
            state = env.scene_state()
            rate = sum(torch.einsum('nmd,nd->nm', rows.J[r], stack_robot(state.qd, r)) for r in ('F', 'U'))
            self.save('selected_Jqd', self.pad(rate[self.detail], 0))
            backlog = kwargs['backlog']
            stored = sum(torch.einsum('nmd,nd->nm', rows.J[r], stack_robot(backlog, r)) for r in ('F', 'U'))
            self.save('selected_backlog', self.pad(stored[self.detail], 0))
            self.save('pending_targets', torch.stack([
                torch.cat([target[a] for a in ARM_KEYS], -1)
                for target in env._evaluation_actuator_delay.queue.pending], 1)[self.detail])
            self.pre_targets = {a: env._targets[a].clone() for a in ARM_KEYS}
            result = self.original_project(cmd, rows, alpha, p, dt, **kwargs)
            self.save('solver_residual', torch.stack([result[2]['residual_F'], result[2]['residual_U']], -1))
            self.save('solver_passes', torch.tensor([result[2]['passes_F'], result[2]['passes_U']]))
            self.save('selected_cap', self.pad(result[2]['cap'][self.detail], 0))
            delta = sum(torch.einsum('nmd,nd->nm', rows.J[r], stack_robot(result[0].delta_q, r)) for r in ('F', 'U'))
            self.save('selected_projected_delta', self.pad(delta[self.detail], 0))
            return result

        env.safety_dist_out = safety
        env._backstop.project = project
        meta = dict(schema='safeduo.passive_causal.v1', detail_env_ids=DETAIL,
                    slots_chosen_before_replay=True, all_envs=env.num_envs,
                    raw_full_row_count=sph.n_pairs, detail_capacity=512,
                    class_names=['cross', 'self_F', 'self_U', 'table'],
                    row_class=classes.cpu().tolist(), pair_table=sph.pair_table.cpu().tolist(),
                    pair_arms=sph.pair_arms.cpu().tolist(), sphere_names=sph.qualified_names,
                    sphere_radii=sph.radii.cpu().tolist(), table_start=sph._slice_table.start,
                    jacobian_reference=env._provider.jacobian_reference,
                    observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    candidate=os.environ.get('SAFEDUO_MECHANISM', 'baseline'))
        (self.out / 'causal_metadata.json').write_text(json.dumps(meta, indent=2) + '\n')

    @staticmethod
    def pad(value, fill):
        shape = list(value.shape)
        shape[1] = 512
        if value.shape[1] > 512:
            raise ValueError('causal capacity exceeded, no row dropping')
        result = torch.full(shape, fill, dtype=value.dtype, device=value.device)
        result[:, :value.shape[1]] = value
        return result

    def _before_step(self, env, t):
        super().before_step(env, t)
        full = env._last_out
        state = env.scene_state()
        self.save('pre_q', torch.cat([state.q[a] for a in ARM_KEYS], -1))
        self.save('pre_target', torch.cat([env._targets[a] for a in ARM_KEYS], -1))
        self.save('pre_full_d', full.dists[self.detail])
        self.save('pre_full_dmin', full.full_dmin[self.detail])
        self.save('pre_full_closing', full.closing[self.detail])
        self.save('pre_full_exempt', full.full_viol_exempt[self.detail])
        self.save('pre_sphere_centers', env._sph.last_centers[self.detail])
        mins, ids = [], []
        for cls in range(4):
            d = full.dists.masked_fill((self.classes != cls)[None] | full.full_viol_exempt, float('inf'))
            m, i = d.min(-1)
            mins.append(m)
            ids.append(i)
        self.save('pre_class_margin', torch.stack(mins, -1))
        self.save('pre_class_row_id', torch.stack(ids, -1))

    def step(self, env, t):
        super().step(env, t)
        full = env._last_out
        self.save('post_full_d', full.dists[self.detail])
        self.save('post_full_dmin', full.full_dmin[self.detail])
        self.save('post_full_exempt', full.full_viol_exempt[self.detail])
        self.save('post_sphere_centers', env._sph.last_centers[self.detail])
        actual = {a: env.scene_state().q[a] - self.prev_q[a] for a in ARM_KEYS}
        observed = sum(torch.einsum('nmd,nd->nm', self.rows.J[r], stack_robot(actual, r)) for r in ('F', 'U'))
        self.save('selected_observed_linear_delta', self.pad(observed[self.detail], 0))
        target_delta = {a: env._targets[a] - self.pre_targets[a] for a in ARM_KEYS}
        target = sum(torch.einsum('nmd,nd->nm', self.rows.J[r], stack_robot(target_delta, r)) for r in ('F', 'U'))
        self.save('selected_actual_target_delta', self.pad(target[self.detail], 0))

    def before_step(self, env, t):
        self._before_step(env, t)
        self.prev_q = {a: env.scene_state().q[a].clone() for a in ARM_KEYS}

    def write(self, *args, **kwargs):
        try:
            np.savez_compressed(self.out / 'causal.npz', **{key: np.stack(values) for key, values in self.details.items()})
            return super().write(*args, **kwargs)
        finally:
            self.env._backstop.project = self.original_project


risk.base.battery.EpisodeTrace = CausalTrace

if __name__ == '__main__':
    risk.base.battery.main()
