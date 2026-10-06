"""Three-mode tracking reserve experiment; original projection, actor and FIFO."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.append(str(HERE.parent / "safety_joint_guard_20261005_1005"))
sys.path.insert(0, str(HERE.parent / 'safety_mechanism_20261005_causal_obs'))
import mechanism_runner as admission
from target_forecast import full_forecast, require_finite
from tracking_reserve import reserve_forecast, gap_from_margin, MODES
from reference_envelope import install, reference_bounds
from safeduo.baselines.base import stack_robot
from safeduo.safety.types import ARM_KEYS, DeltaCmd

CHUNK_STEPS = 32
DIAGNOSTIC_STEPS = [0, 60, 62, 66, 71, 75, 180, 480, 959]


class GuardTrace(admission.risk.RiskTrace):
    def start(self, env):
        super().start(env)
        self.mode = os.environ['SAFEDUO_JOINT_MODE']
        if self.mode not in MODES:
            raise ValueError('unregistered joint guard mode')
        self.admission_on = True
        self.reference_on = True
        self.reserve_gap = torch.full((env.num_envs,1),.050,device=env.device)
        self.command_scale = 1.
        original_sample = env._delta_src.sample
        env._delta_src.sample = lambda state: DeltaCmd({a: v * self.command_scale for a,v in original_sample(state).delta_q.items()})
        self.frames['external_unscaled_cmd'] = []
        self.capacity = 9021
        self.original_selected = self.original_safety if self.admission_on else env.safety_dist_out
        self.original_rows = env._provider.rows_from
        self.original_project = env._backstop.project
        self.original_pre = env._pre_physics_step
        self.project_records = []
        self.first_failure_seen = torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        self.first_failure_receipts = []
        self.forecast_chunks = []
        self.chunk = []
        self.chunk_start = 0
        self.guard_calls = 0
        self.extra = {key: [] for key in ['forecast_added_count', 'forecast_only_min',
                       'required_rows', 'full_forecast_min', 'full_rows_checked',
                       'guard_calls_before_step', 'reference_envelope_unreachable',
                       'shadow_reference_missed_count', 'shadow_reference_min']}
        self.frames['pre_target_debt'] = []
        self.frames['pre_pending_actuator_targets'] = []
        self.frames['pre_pending_project_history'] = []
        self.frames['effective_target_delta'] = []
        self.frames['governor_changed'] = []
        (self.out / 'full_row_identity.json').write_text(json.dumps(dict(
            rows=9021, class_id=env._sph.class_id.cpu().tolist(), pair_id=env._sph.pair_id.cpu().tolist(),
            sphere_names=env._sph.qualified_names, sphere_radii_m=env._sph.radii.cpu().tolist(),
            sphere_arm_id=env._sph.arm_id.cpu().tolist(), pair_sphere_idx=env._sph.pair_table.cpu().tolist(),
            scope='stable full-row/sphere identity; forecast uses actual COM-referenced controlled-joint J'), indent=2) + '\n')

        def abort(error, stage):
            (self.out / 'guard_abort.json').write_text(json.dumps(dict(
                step=self.t, mode=self.mode, error=f'{type(error).__name__}: {error}',
                abort_stage=stage, partial_windows_not_safe=True), indent=2) + '\n')

        def checked_rows(out, body):
            rows = self.original_rows(out, body)
            require_finite('selected_jacobian', rows.J)
            return rows

        def safety():
            try:
                full = env._last_out
                base = self.original_selected()
                state = env.scene_state()
                p = full.dists.shape[1]
                if full.dists.shape != (64, 9021):
                    raise ValueError('full geometry identity changed before physics')
                ids = torch.arange(p, device=env.device).expand(env.num_envs, -1)
                all_rows = replace(full, active_idx=ids,
                    active_mask=torch.ones_like(ids, dtype=torch.bool),
                    active_pairs=torch.stack([full.dists, full.closing,
                        env._sph.class_id.expand_as(full.dists), env._sph.pair_id.expand_as(full.dists)], -1),
                    active_dmin=full.full_dmin, viol_exempt=full.full_viol_exempt)
                rows = checked_rows(all_rows, env._body_pos_cache)
                if env._pending_cmd is None:
                    raise ValueError('missing state-independent external command')
                forecast = full_forecast(full, rows, state, env._targets,
                    env._evaluation_actuator_delay.queue.pending, env._pending_cmd,
                    env._q_soft_limits, env._backstop.cfg.vmax * state.dt)
                risk_forecast = reserve_forecast(full.dists, full.full_dmin, full.full_viol_exempt,
                    rows.J, state.q, state.qd, env._evaluation_actuator_delay.queue.pending, float(state.dt))
                self.reserve_gap = gap_from_margin(risk_forecast['risk_margin'], self.mode)[:,None]
                unreachable = []
                for a in ARM_KEYS:
                    lim = env._q_soft_limits[a]
                    lo, hi = reference_bounds(state.q[a], env._targets[a], lim[...,0], lim[...,1], env._backstop.cfg.vmax * state.dt)
                    unreachable.append(((state.q[a] + self.reserve_gap < env._targets[a] + lo) |
                                        (state.q[a] - self.reserve_gap > env._targets[a] + hi)).any(-1))
                self.last_unreachable = torch.stack(unreachable,-1)
                motion = dict(reserve_forecast=risk_forecast['distance'],
                    reserve_margin=risk_forecast['risk_margin'], reserve_gap=self.reserve_gap[:,0],
                    target_forecast=forecast)
                # Keep observed distances and exemptions. Forecast only controls
                # additional admission, never the original measured geometry.
                baseline = admission.union_mask(full, base, full.dists)
                mask = admission.union_mask(full, base, forecast)
                # A passive reference-proposal diagnostic. This NEVER changes
                # the registered raw-proposal admission factor or its selection.
                bounds = {a: reference_bounds(state.q[a], env._targets[a],
                    env._q_soft_limits[a][..., 0], env._q_soft_limits[a][..., 1],
                    env._backstop.cfg.vmax * state.dt, .050) for a in ARM_KEYS}
                require_finite('shadow_bounds', {a + side: v[i] for a, v in bounds.items()
                                               for i, side in enumerate(('lo', 'hi'))})
                shadow = {a: env._targets[a] + env._pending_cmd.delta_q[a].maximum(bounds[a][0]).minimum(bounds[a][1])
                          - state.q[a] for a in ARM_KEYS}
                require_finite('shadow_displacement', shadow)
                shadow_delta = sum(torch.einsum('nmd,nd->nm', rows.J[r], stack_robot(shadow, r))
                                   for r in ('F', 'U'))
                require_finite('shadow_linear_delta', dict(delta=shadow_delta))
                shadow_d = full.dists + shadow_delta
                require_finite('shadow_prediction', dict(d=shadow_d))
                missed = (shadow_d <= full.full_dmin + .010) & ~mask
                self.last_shadow_count = missed.sum(-1).cpu().numpy()
                self.last_shadow_min = shadow_d.min(-1).values.cpu().numpy()
                added = mask & ~baseline if self.admission_on else torch.zeros_like(mask)
                result = admission.select(full, base, mask, env._sph, self.capacity) if self.admission_on else base
                self.last_count = result.active_mask.sum(-1)
                if int(self.last_count.max()) > self.capacity:
                    raise ValueError('row budget exceeded; no dropping')
                self.last_added = added.sum(-1)
                only = forecast.masked_fill(~added, float('inf')).min(-1).values
                self.last_only = torch.where(added.any(-1), only, torch.zeros_like(only))
                self.last_forecast = forecast.detach().cpu().numpy().copy()
                def padded_ids(out):
                    value = np.full((64, self.capacity), -1, dtype=np.int32)
                    ids = out.active_idx.masked_fill(~out.active_mask, -1).cpu().numpy()
                    value[:, :ids.shape[1]] = ids
                    return value
                self.last_chunk = dict(forecast=self.last_forecast,
                    measured_d=full.dists.cpu().numpy().copy(),
                    dmin=full.full_dmin.cpu().numpy().copy(),
                    exempt=np.packbits(full.full_viol_exempt.cpu().numpy(), axis=-1),
                    selected_ids=padded_ids(result), baseline_ids=padded_ids(admission.select(full, base, baseline, env._sph, self.capacity) if self.admission_on else base),
                    shadow_missed=np.packbits(missed.cpu().numpy(), axis=-1),
                    **{k: v.detach().cpu().numpy().copy() for k,v in motion.items() if k != 'forecast'})
                self.pre_targets = {a: v.clone() for a, v in env._targets.items()}
                self.guard_calls += 1
                return result
            except BaseException as error:
                abort(error, 'before projection/physics')
                raise

        env._provider.rows_from = checked_rows
        env.safety_dist_out = safety
        from projection_diagnostics import install as install_diagnostics, checked_project
        actual_project = env._backstop.project
        env._backstop.project = lambda cmd, rows, alpha, p, dt, **kwargs: checked_project(
            actual_project, cmd, rows, alpha, p, dt, **kwargs)
        self.last_project_record = {}
        install_diagnostics(env, self.last_project_record, snapshot=lambda: self.t in DIAGNOSTIC_STEPS)
        if self.reference_on:
            install(env, 'envelope_050', gap_provider=lambda: self.reserve_gap)
        observed_project = env._backstop.project

        def project(cmd, rows, alpha, p, dt, **kwargs):
            try:
                require_finite('project_input', dict(alpha=alpha, p=p, dt=torch.as_tensor(dt),
                    box=torch.as_tensor(env._backstop.cfg.vmax * dt)))
                require_finite('project_command', cmd.delta_q)
                for a, bound in (kwargs.get('delta_bounds') or {}).items():
                    require_finite('project_bounds_' + a, dict(lo=bound[0], hi=bound[1]))
                result, active, info = checked_project(observed_project, cmd, rows, alpha, p, dt,
                    diagnostic_sink=self.last_project_record, **kwargs)
                require_finite('effective_project_output', result.delta_q)
                require_finite('project_info', {k: v for k, v in info.items() if torch.is_tensor(v)})
                self.last_project_record.update({k: v.clone() for k,v in info.items() if k.startswith('repair_') and torch.is_tensor(v)})
                require_finite('projection_diagnostics', self.last_project_record)
                actual_kwargs = dict(kwargs)
                if self.reference_on:
                    actual_kwargs['delta_bounds'] = dict(zip(ARM_KEYS, zip(
                        self.last_project_record['bounds_lower'].split([7, 7, 6, 6], -1),
                        self.last_project_record['bounds_upper'].split([7, 7, 6, 6], -1))))
                inside_cmd = DeltaCmd(delta_q=dict(zip(ARM_KEYS,
                    self.last_project_record['project_input_cmd'].split([7, 7, 6, 6], -1))))
                self.last_project_context = (inside_cmd, rows, alpha, p, dt, actual_kwargs, active, info)
                if kwargs.get('past_backlogs') is not None:
                    self.last_project_record['pending_history_backlogs'] = torch.stack([
                        torch.cat([target[a] for a in ARM_KEYS], -1) for target in kwargs['past_backlogs']], 1)
                self.last_changed = info.get('reference_governor_changed', torch.zeros(64, 4, device=env.device, dtype=torch.bool))
                if self.t in DIAGNOSTIC_STEPS:
                    root = self.out / 'projection_snapshots'
                    root.mkdir(exist_ok=True)
                    values = {k: v.detach().cpu().numpy() for k, v in self.last_project_record.items() if torch.is_tensor(v)}
                    values.update({'J_' + r: v.detach().cpu().numpy() for r, v in rows.J.items()})
                    values.update(selected_ids=self.last_chunk['selected_ids'], raw_cmd=cmd.stacked().detach().cpu().numpy())
                    np.savez_compressed(root / f'step_{self.t:04d}.npz', **values)
                return result, active, info
            except BaseException as error:
                abort(error, 'project input/output; before physics')
                raise

        env._backstop.project = project

        def pre(actions):
            try:
                self.original_pre(actions)
                require_finite('integrated_issued_target', env._targets)
                require_finite('delivered_actuator_target', env._evaluation_actuator_delay.applied)
            except BaseException as error:
                if not (self.out / 'guard_abort.json').exists():
                    abort(error, 'after target integration; before physics')
                raise
        env._pre_physics_step = pre
        metadata = dict(schema='safeduo.tracking_reserve.v1', mode=self.mode, command_scale=self.command_scale,
            joint_repair=False, unavoidable_slack_is_safety=False,
            admission=self.admission_on, reference=self.reference_on,
            full_forecast_gate=True, finite_intermediates_checked=True,
            full_forecast_chunks=True, forecast_stage='pre-step; q=post[t-1] or q_initial',
            forecast_chunk_steps=CHUNK_STEPS, capacity=self.capacity,
            admission_band_m=.010, reference_gap_rad={'joint_reference':.050,'tight_reference':.010,'delay_reserve':'perenv .010..050'}[self.mode],
            original_actor_rows=32, strict_fifo_steps=6, queue_preemption=False,
            structural_and_conditional_exemptions_unchanged=True, production_promoted=False,
            sources={str(HERE / name): hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                     for name in ['guard_runner.py', 'target_forecast.py', 'reference_envelope.py', 'projection_diagnostics.py', 'tracking_reserve.py', 'RANDOM_EXPERIMENT_DESIGN.json']},
            shadow_reference='passive .050 proposal; omissions retained, does not affect control',
            projection_snapshots=DIAGNOSTIC_STEPS,
            motion_horizons=[18], pd_scope='no PD model used',
            cv_scope='CV18 and six pending endpoint frozen-J risk; heuristic no bound', additional_rows_only=False,
            future_assumption='risk: only measured q/qd plus exact first6 pending, no current proposal; target admission unchanged')
        (self.out / 'guard_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        manifest = json.loads((self.out / 'random_manifest.json').read_text())
        manifest.update(mechanism=self.mode, capacity=self.capacity,
            allocation=f'exact required width, hard budget{self.capacity}; abort, no dropping',
            full_finite_guard=True, guard_metadata='guard_metadata.json')
        (self.out / 'random_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def before_step(self, env, t):
        super().before_step(env, t)
        self.frames['external_unscaled_cmd'].append(env._delta_src.tape[t].clone())
        self.guard_calls = 0
        state = env.scene_state()
        self.frames['pre_target_debt'].append(torch.cat([env._targets[a] - state.q[a] for a in ARM_KEYS], -1).clone())
        box = env._backstop.cfg.vmax * state.dt
        require_finite('control_box', dict(box=torch.as_tensor(box), dt=torch.as_tensor(state.dt)))
        self.frames['pre_pending_actuator_targets'].append(torch.stack([
            torch.cat([target[a] for a in ARM_KEYS], -1) for target in env._evaluation_actuator_delay.queue.pending]).clone())
        self.frames['pre_pending_project_history'].append(torch.stack([
            torch.cat([target[a] for a in ARM_KEYS], -1) for target in env._pending_target_history.targets]).clone())
        unreachable = []
        for a in ARM_KEYS:
            lim = env._q_soft_limits[a]
            lo, hi = reference_bounds(state.q[a], env._targets[a], lim[..., 0], lim[..., 1], box)
            unreachable.append(((state.q[a] + .050 < env._targets[a] + lo) |
                                (state.q[a] - .050 > env._targets[a] + hi)).any(-1))
        self.last_unreachable = torch.stack(unreachable, -1)

    def flush_forecast(self):
        if not self.chunk:
            return
        root = self.out / 'full_forecast'
        root.mkdir(exist_ok=True)
        stop = self.chunk_start + len(self.chunk)
        path = root / f'steps_{self.chunk_start:04d}_{stop:04d}.npz'
        np.savez_compressed(path, **{k: np.stack([value[k] for value in self.chunk]) for k in self.chunk[0]})
        self.forecast_chunks.append(dict(path=str(path.relative_to(self.out)),
            start=self.chunk_start, stop=stop, shape=[len(self.chunk), 64, 9021],
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        self.chunk = []
        self.chunk_start = stop

    def step(self, env, t):
        super().step(env, t)
        if self.guard_calls < 1:
            raise ValueError('full finite gate was not executed for this physics step')
        if self.last_forecast.shape != (64, 9021):
            raise ValueError('full forecast shape changed')
        self.chunk.append(self.last_chunk)
        if len(self.chunk) == CHUNK_STEPS:
            self.flush_forecast()
        values = dict(forecast_added_count=self.last_added.cpu().numpy(),
            forecast_only_min=self.last_only.cpu().numpy(), required_rows=self.last_count.cpu().numpy(),
            full_forecast_min=self.last_forecast.min(-1),
            full_rows_checked=np.full(64, 9021), guard_calls_before_step=np.full(64, self.guard_calls),
            reference_envelope_unreachable=self.last_unreachable.cpu().numpy(),
            shadow_reference_missed_count=self.last_shadow_count, shadow_reference_min=self.last_shadow_min)
        for key, value in values.items():
            self.extra[key].append(value.copy())
        # These post-target increments include the original soft-limit clamp.
        self.frames['effective_target_delta'].append(torch.cat([
            env._targets[a] - self.pre_targets[a] for a in ARM_KEYS], -1).clone())
        self.frames['governor_changed'].append(self.last_changed.clone())
        from projection_diagnostics import projection_record
        inside_cmd, rows, alpha, p, dt, kwargs, active, info = self.last_project_context
        target_cmd = DeltaCmd(delta_q={a: env._targets[a] - self.pre_targets[a] for a in ARM_KEYS})
        target_record = projection_record(env._backstop, inside_cmd, rows, alpha, p, dt,
                                         kwargs, (target_cmd, active, info))
        full = env._last_out
        bad = ((full.dists < 0) & ~full.full_viol_exempt).any(-1)
        first = bad & ~self.first_failure_seen
        if first.any():
            ids = torch.where(first)[0]
            detailed = projection_record(env._backstop, inside_cmd, rows, alpha, p, dt,
                kwargs, (target_cmd, active, info), snapshot=True)
            values = {k: v[ids].detach().cpu().numpy() for k,v in detailed.items()
                      if torch.is_tensor(v) and v.ndim and v.shape[0] == 64}
            pre_q = self.q0 if t == 0 else self.frames['q'][-2]
            values.update(env_ids=ids.cpu().numpy(), step=np.array(t),
                pre_q=pre_q[ids].cpu().numpy(),
                pre_issued_target=torch.cat([self.pre_targets[a] for a in ARM_KEYS],-1)[ids].cpu().numpy(),
                pre_pending_actuator_targets=self.frames['pre_pending_actuator_targets'][-1][:,ids].permute(1,0,2).cpu().numpy(),
                external_raw_cmd=self.last_project_record['external_raw_cmd'][ids].cpu().numpy(),
                actual_project_return=self.last_project_record['outer_returned_cmd'][ids].cpu().numpy(),
                post_q=torch.cat([env.scene_state().q[a] for a in ARM_KEYS],-1)[ids].cpu().numpy(),
                post_qd=torch.cat([env.scene_state().qd[a] for a in ARM_KEYS],-1)[ids].cpu().numpy(),
                post_full_d=full.dists[ids].cpu().numpy(), post_full_dmin=full.full_dmin[ids].cpu().numpy(),
                post_full_exempt=full.full_viol_exempt[ids].cpu().numpy(),
                selected_ids=self.last_chunk['selected_ids'][ids.cpu().numpy()])
            root=self.out/'first_failures'; root.mkdir(exist_ok=True)
            path=root/f'step_{t:04d}.npz'; np.savez_compressed(path,**values)
            self.first_failure_receipts.append(dict(step=t,env_ids=ids.cpu().tolist(),
                path=str(path.relative_to(self.out)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                scope='exact first negative post geometry with actual preceding selected G/h/J, alpha, bounds and FIFO; snapshot returned_cmd is effective target increment'))
        self.first_failure_seen |= bad
        for robot in ('F', 'U'):
            for kind in ('safety', 'alpha', 'bound'):
                self.last_project_record[f'target_{kind}_residual_{robot}'] = target_record[f'returned_{kind}_residual_{robot}']
        self.project_records.append({k: v.detach().cpu().numpy().copy()
            for k, v in self.last_project_record.items() if torch.is_tensor(v) and v.ndim <= 2
            and (v.shape[0] == 64 if v.ndim else False)})

    def write(self, *args, **kwargs):
        try:
            self.flush_forecast()
            (self.out / 'first_failure_receipts.json').write_text(json.dumps(dict(envs_with_failure=int(self.first_failure_seen.sum()),receipts=self.first_failure_receipts),indent=2)+'\n')
            np.savez_compressed(self.out / 'mechanism.npz', **{k: np.stack(v) for k, v in self.extra.items()})
            np.savez_compressed(self.out / 'project_diagnostics.npz', **{k: np.stack([
                record[k] for record in self.project_records]) for k in self.project_records[0]
                if all(k in record and record[k].shape == self.project_records[0][k].shape for record in self.project_records)})
            (self.out / 'forecast_receipts.json').write_text(json.dumps(dict(
                schema='safeduo.full_forecast_receipts.v1', steps=self.chunk_start,
                rows=9021, envs=64, chunks=self.forecast_chunks,
                jacobian_finite='all rows checked by producer before every forecast; full J not persisted'), indent=2) + '\n')
            return super().write(*args, **kwargs)
        finally:
            self.env._provider.rows_from = self.original_rows
            self.env._backstop.project = self.original_project
            self.env._pre_physics_step = self.original_pre
            self.env.safety_dist_out = self.original_safety


admission.risk.base.battery.EpisodeTrace = GuardTrace
if __name__ == '__main__':
    admission.risk.base.battery.main()
