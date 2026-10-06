"""Passive selected-row diagnostics for the frozen VelocityDamperBackstop.

No Isaac imports, simulation calls, target writes, row selection, or solver changes.
install() returns the original callable and returns every solver result unchanged.
All sink values are detached tensor copies. A callable sink receives one record;
a mapping sink is overwritten per call. snapshot_* fields are optional/variable
width. Use summary_record(record) for the fixed-width per-frame stream.

Parent composition (all four methods, reference installed only for E=1):
    actual = env._backstop.project
    env._backstop.project = partial(checked_project, actual)  # common inner gate
    restore = install(env, sink, snapshot=lambda: wanted_step())
    reference_envelope.install(env, 'envelope_050')          # only if E=1
    outer = env._backstop.project
    env._backstop.project = partial(checked_project, outer,
                                   diagnostic_sink=sink)   # common outer gate

The inner gate sees reference-computed bounds/limited cmd before the real solver.
The outer gate sees external cmd and the final reference info after the call.
The passive observer itself is not the finite-input/output policy gate.

returned_* means project-returned increment, including the solver's R19 override.
It does NOT mean the subsequent target-clamped/controller/applied FIFO increment.
Worst row indices are positions in the selected matrix, NOT global row IDs.
The parent binds row IDs and saves sink before the next project invocation.
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import math

import torch

from safeduo.baselines.base import stack_robot
from safeduo.safety.types import ARM_KEYS, ARMS_OF_ROBOT, CLASS_CROSS

SCHEMA = 'safeduo.passive_projection_diagnostics.v1'


def require_finite_tree(label, value):
    """Fail closed on numeric leaves; never clamp or replace a value."""
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError(f'nonfinite {label}; abort before physics')
    elif isinstance(value, Mapping):
        for key, item in value.items():
            require_finite_tree(f'{label}.{key}', item)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            require_finite_tree(f'{label}[{index}]', item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'nonfinite {label}; abort before physics')


def check_project_inputs(cmd, rows, alpha, p, dt, kwargs):
    """Check the actual call seam, including computed reference delta_bounds."""
    require_finite_tree('input', dict(cmd=cmd.delta_q, d=rows.d, J=rows.J,
        cls=rows.cls, arm_mask=rows.arm_mask, valid=rows.valid,
        row_dmin=rows.d_min, alpha=alpha, p=p, dt=dt, kwargs=kwargs))


def check_project_result(result):
    """Check returned exec/active/all numeric info before targets reach physics."""
    cmd, active, info = result
    require_finite_tree('output', dict(cmd=cmd.delta_q, active=active, info=info))


def checked_project(project, cmd, rows, alpha, p, dt, *, diagnostic_sink=None, **kwargs):
    """Independent common gate, returning the exact original tuple/objects.

    Use both inside reference (to check computed bounds/limited cmd) and outside
    reference (to annotate governor_changed and the external raw command).
    This checks the selected call seam; the parent's full-row guard remains
    responsible for all9021 rows, q/qd/limits/config and actual FIFO identities.
    """
    check_project_inputs(cmd, rows, alpha, p, dt, kwargs)
    result = project(cmd, rows, alpha, p, dt, **kwargs)
    check_project_result(result)
    if diagnostic_sink is not None:
        if 'project_input_cmd' not in diagnostic_sink:
            raise ValueError('diagnostic sink was not populated by the inner observer')
        output, active, info = result
        diagnostic_sink['external_raw_cmd'] = _copy(_stack_arms(cmd.delta_q))
        diagnostic_sink['outer_returned_cmd'] = _copy(_stack_arms(output.delta_q))
        diagnostic_sink['outer_active'] = _copy(active)
        present = 'reference_governor_changed' in info
        diagnostic_sink['governor_changed_available'] = torch.full_like(active, present)
        diagnostic_sink['governor_changed'] = _copy(info['reference_governor_changed']) if present else torch.zeros_like(active)
    return result


def _copy(value):
    return value.detach().clone()


def _stack_arms(values):
    return torch.cat([values[arm] for arm in ARM_KEYS], -1)


def _effective_distance(backstop, rows, kwargs):
    """Repeat production d_eff ONLY when its strict engage predicate is needed.

    cap is reused from original info for budgets. Recovering d_eff by dividing
    cap would risk changing a strict threshold at float rounding boundaries.
    """
    cfg = backstop.cfg
    qd = kwargs.get('qd')
    d_eff = rows.d
    if any(t is not None for t in (cfg.lookahead_s, cfg.self_lookahead_s,
                                  cfg.table_lookahead_s)) and qd is not None:
        ddot = torch.zeros_like(rows.d)
        for robot in ('F', 'U'):
            ddot = ddot + torch.einsum('nmd,nd->nm', rows.J[robot], stack_robot(qd, robot))
        horizon = cfg.lookahead_s or 0.0
        if cfg.self_lookahead_s is not None:
            horizon = torch.where(rows.cls == 1.0, max(horizon, cfg.self_lookahead_s), horizon)
        if cfg.table_lookahead_s is not None:
            table = rows.cls == 2.0
            structural = kwargs.get('struct_exempt')
            contact = kwargs.get('contact_exempt')
            if structural is not None:
                permanent = structural if contact is None else (structural & ~contact)
                table = table & ~permanent
            horizon = torch.where(table, max(cfg.lookahead_s or 0.0, cfg.table_lookahead_s), horizon)
        d_eff = rows.d + horizon * ddot.clamp(max=0.0)
    if cfg.predict_backlog:
        backlog = kwargs.get('backlog')
        if backlog is None:
            raise ValueError('stored-target prediction requires backlog')
        past = kwargs.get('past_backlogs')
        if cfg.pending_target_steps and (past is None or len(past) != cfg.pending_target_steps):
            raise ValueError('pending target prediction requires the declared target history')
        stored = sum(torch.einsum('nmd,nd->nm', rows.J[r], stack_robot(backlog, r)) for r in ('F', 'U'))
        for previous in past or []:
            stored = torch.minimum(stored, sum(torch.einsum('nmd,nd->nm', rows.J[r], stack_robot(previous, r)) for r in ('F', 'U')))
        d_eff = torch.minimum(d_eff, rows.d + stored.clamp(max=0.0))
    return d_eff


def _gate(backstop, rows, cap, kwargs, d_eff=None):
    cfg = backstop.cfg
    gate = None
    if cfg.engage_dist is not None:
        if d_eff is None:
            d_eff = _effective_distance(backstop, rows, kwargs)
        gate = (d_eff < cfg.engage_dist) | (cap < 0.0)
    structural = kwargs.get('struct_exempt')
    if cfg.exempt_structural_rows and structural is not None:
        structural = structural.to(rows.d.device)
        if cfg.retain_conditional_rows and kwargs.get('contact_exempt') is not None:
            structural = structural & ~kwargs['contact_exempt']
        drop = structural & (cap >= 0.0)
        if cfg.struct_engage_dist is not None:
            drop = drop & (rows.d >= cfg.struct_engage_dist)
        gate = ~drop if gate is None else (gate & ~drop)
    return rows.valid if gate is None else (rows.valid & gate)


def _worst(values, mask):
    """Per-env max and selected-position index, with0/-1 for no relevant row."""
    n, width = values.shape
    if width == 0:
        return values.new_zeros(n), torch.full((n,), -1, device=values.device, dtype=torch.long)
    value, index = values.masked_fill(~mask, -torch.inf).max(-1)
    any_row = mask.any(-1)
    return torch.where(any_row, value, torch.zeros_like(value)), index.masked_fill(~any_row, -1)


def _at(values, index):
    n = values.shape[0]
    if values.shape[1] == 0:
        return values.new_zeros((n,) + values.shape[2:])
    value = values[torch.arange(n, device=values.device), index.clamp_min(0)]
    present = (index >= 0).reshape((n,) + (1,) * (value.ndim - 1))
    return torch.where(present, value, torch.zeros_like(value))


def _per_env(value, n, like, *, integer=False):
    value = torch.as_tensor(value, device=like.device, dtype=torch.long if integer else like.dtype)
    if value.ndim == 0:
        value = value.expand(n)
    if value.shape != (n,):
        raise ValueError('unexpected original residual/passes shape')
    return value


def projection_record(backstop, cmd, rows, alpha, p, dt, kwargs, result, *, snapshot=False):
    """Observe the production result; no iterative solve and no in-place writes."""
    output, active, info = result
    cfg = backstop.cfg
    cap = info['cap']
    n = rows.d.shape[0]
    box = cfg.vmax * dt
    d_eff = _effective_distance(backstop, rows, kwargs) if snapshot or cfg.engage_dist is not None else None
    gate = _gate(backstop, rows, cap, kwargs, d_eff)
    bounds = kwargs.get('delta_bounds')
    backlog = kwargs.get('backlog')
    bypass = kwargs.get('bypass_arm')
    dm = kwargs.get('dmin')
    if dm is None:
        dm = rows.d_min
    if dm is None:
        dm = torch.full_like(rows.d, cfg.d_min)
    record = dict(project_input_cmd=_stack_arms(cmd.delta_q), returned_cmd=_stack_arms(output.delta_q),
        alpha=alpha, p=p, original_active=active,
        bypass_arm=torch.zeros_like(active) if bypass is None else bypass,
        governor_changed_available=torch.zeros_like(active), governor_changed=torch.zeros_like(active),
        detailed_snapshot=torch.full((n,), snapshot, dtype=torch.bool, device=rows.d.device))
    lower_all, upper_all = [], []
    for robot in ('F', 'U'):
        c = stack_robot(cmd.delta_q, robot).clamp(-box, box)
        u = stack_robot(output.delta_q, robot)
        if bounds is None:
            lower, upper = torch.full_like(c, -box), torch.full_like(c, box)
        else:
            lower = stack_robot({a: bounds[a][0] for a in ARM_KEYS}, robot).clamp_min(-box)
            upper = stack_robot({a: bounds[a][1] for a in ARM_KEYS}, robot).clamp_max(box)
        lower_all.append(lower)
        upper_all.append(upper)
        involved = rows.arm_mask[..., [ARM_KEYS.index(a) for a in ARMS_OF_ROBOT[robot]]].any(-1)
        rel = gate & involved
        G = -rows.J[robot]
        budget = (1.0 + p) * .5 if robot == 'F' else (1.0 - p) * .5
        cross_h = torch.where(cap >= 0, budget.unsqueeze(-1) * cap, cap)
        raw_h = torch.where(rows.cls == CLASS_CROSS, cross_h, cap)
        debit = torch.zeros_like(raw_h)
        if cfg.backlog_aware and backlog is not None:
            b_eff = stack_robot(backlog, robot).clamp(-box, box)
            debit = torch.einsum('nmd,nd->nm', G, b_eff)
            if cfg.predict_backlog:
                for past in kwargs.get('past_backlogs') or []:
                    past_eff = stack_robot(past, robot).clamp(-box, box)
                    debit = torch.maximum(debit, torch.einsum('nmd,nd->nm', G, past_eff))
        before_h = raw_h - debit if cfg.backlog_aware and backlog is not None else raw_h
        min_Gu = (-(G.abs().sum(-1)) * box if bounds is None else
                  torch.where(G >= 0, G * lower.unsqueeze(1), G * upper.unsqueeze(1)).sum(-1))
        authority_on = cfg.row_authority_clamp or (cfg.backlog_aware and backlog is not None)
        h = torch.maximum(before_h, .9 * min_Gu) if authority_on else before_h
        aG, ah, arel = backstop._alpha_rows(c, alpha, robot)
        signed = (G @ u.unsqueeze(-1)).squeeze(-1) - h
        positive = signed.clamp_min(0.0)
        alpha_error = ((aG @ u.unsqueeze(-1)).squeeze(-1) - ah).clamp_min(0.0)
        max_error, worst = _worst(positive, rel)
        max_alpha, worst_alpha = _worst(alpha_error, arel)
        minimum_error = (min_Gu - h).clamp_min(0.0)
        max_infeasible, worst_infeasible = _worst(minimum_error, rel)
        record.update({f'original_safety_residual_{robot}': _per_env(info[f'residual_{robot}'], n, rows.d),
            f'original_passes_{robot}': _per_env(info[f'passes_{robot}'], n, rows.d, integer=True),
            f'passes_hit_limit_{robot}': _per_env(info[f'passes_{robot}'], n, rows.d, integer=True) >= cfg.max_passes,
            f'returned_safety_residual_{robot}': max_error,
            f'returned_alpha_residual_{robot}': max_alpha,
            f'returned_bound_residual_{robot}': torch.maximum((lower-u).clamp_min(0), (u-upper).clamp_min(0)).amax(-1),
            f'relevant_rows_{robot}': rel.sum(-1), f'alpha_relevant_{robot}': arel,
            f'alpha_residual_by_arm_{robot}': alpha_error * arel,
            f'worst_safety_position_{robot}': worst, f'worst_alpha_arm_local_{robot}': worst_alpha,
            f'individual_infeasibility_lower_bound_{robot}': max_infeasible,
            f'individually_infeasible_row_count_{robot}': ((min_Gu > h) & rel).sum(-1),
            f'authority_changed_row_count_{robot}': ((h != before_h) & rel).sum(-1),
            f'worst_infeasible_position_{robot}': worst_infeasible,
            f'zero_outside_bounds_{robot}': ((lower > 0) | (upper < 0)).any(-1),
            f'singleton_joint_count_{robot}': (lower == upper).sum(-1),
            f'worst_G_{robot}': _at(G, worst), f'worst_d_{robot}': _at(rows.d, worst),
            f'worst_dmin_{robot}': _at(dm, worst), f'worst_class_{robot}': _at(rows.cls, worst),
            f'worst_cap_{robot}': _at(cap, worst), f'worst_raw_h_{robot}': _at(raw_h, worst),
            f'worst_debit_{robot}': _at(debit, worst), f'worst_h_before_authority_{robot}': _at(before_h, worst),
            f'worst_h_after_authority_{robot}': _at(h, worst), f'worst_min_Gu_{robot}': _at(min_Gu, worst)})
        if snapshot:
            record.update({f'snapshot_G_{robot}': G, f'snapshot_raw_h_{robot}': raw_h, f'snapshot_backlog_debit_{robot}': debit,
                f'snapshot_h_before_authority_{robot}': before_h, f'snapshot_h_after_authority_{robot}': h,
                f'snapshot_min_Gu_{robot}': min_Gu, f'snapshot_rel_{robot}': rel,
                f'snapshot_alpha_G_{robot}': aG, f'snapshot_alpha_h_{robot}': ah, f'snapshot_alpha_rel_{robot}': arel})
    record['bounds_lower'] = torch.cat(lower_all, -1)
    record['bounds_upper'] = torch.cat(upper_all, -1)
    if snapshot:
        past = kwargs.get('past_backlogs') or []
        record.update(snapshot_d=rows.d, snapshot_dmin=dm, snapshot_cls=rows.cls,
            snapshot_arm_mask=rows.arm_mask, snapshot_valid=rows.valid, snapshot_cap=cap,
            snapshot_d_eff=d_eff, snapshot_J_F=rows.J['F'], snapshot_J_U=rows.J['U'],
            snapshot_backlog=_stack_arms(backlog) if backlog is not None else record['returned_cmd'].new_zeros((n, record['returned_cmd'].shape[-1])),
            snapshot_backlog_available=torch.full((n,), backlog is not None, device=rows.d.device, dtype=torch.bool),
            snapshot_past_backlogs=torch.stack([_stack_arms(b) for b in past], 1) if past else record['returned_cmd'].new_zeros((n, 0, record['returned_cmd'].shape[-1])))
        record['snapshot_row_gate'] = gate
        qd = kwargs.get('qd')
        record['snapshot_qd'] = _stack_arms(qd) if qd is not None else torch.zeros_like(record['returned_cmd'])
        record['snapshot_qd_available'] = torch.full((n,), qd is not None, device=rows.d.device, dtype=torch.bool)
        for name in ('struct_exempt', 'contact_exempt'):
            value = kwargs.get(name)
            record['snapshot_' + name] = torch.zeros_like(rows.valid) if value is None else value
            record['snapshot_' + name + '_available'] = torch.full((n,), value is not None, device=rows.d.device, dtype=torch.bool)
    return {key: _copy(value) for key, value in record.items()}


def summary_record(record):
    """Fixed-width fields only; optional snapshots must not enter all-frame stacks."""
    return {key: value for key, value in record.items() if not key.startswith('snapshot_')}


def install(env, sink, *, snapshot=False):
    """Install before reference; sink(record) callback or per-call scratch dict.

    snapshot is a bool or zero-argument predicate evaluated once per call.
    Store/clone the record before the next call. Restore using the returned
    callable. No files are written and no queue/state/config is changed.
    """
    if not callable(sink) and not isinstance(sink, MutableMapping):
        raise TypeError('sink must be a callable or mutable mapping of tensor fields')
    original = env._backstop.project

    def project(cmd, rows, alpha, p, dt, **kwargs):
        if not callable(sink):
            sink.clear()
        result = original(cmd, rows, alpha, p, dt, **kwargs)
        detailed = bool(snapshot() if callable(snapshot) else snapshot)
        with torch.no_grad():
            record = projection_record(env._backstop, cmd, rows, alpha, p, dt, kwargs, result, snapshot=detailed)
        if callable(sink):
            sink(record)
        else:
            sink.update(record)
        return result

    env._backstop.project = project
    return original
