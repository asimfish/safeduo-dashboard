"""Evaluation-only adaptive zero-inclusive .010 reference governor.

API (run Python with -B; safeduo/src must already be importable)::

    restore = install_adaptive(env, lambda: trace.reserve_gap, trace.last_project_record)
    result, active, info = env._backstop.project(cmd, rows, alpha, p, dt, **kwargs)
    env._backstop.project = restore

Install IN PLACE OF reference_envelope.install, before the outer checked_project
gate. The captured callable must be the original backstop, optionally wrapped by
finite gates/passive diagnostics, never an already installed reference governor.
gap_provider is zero-argument and returns the registered .010 scalar/batch gap;
None selects .010. diagnostic_sink is a mutable mapping or a one-record callback.

Only first-call RETURNED safety/alpha/bound residual > 1e-6 triggers a CPU LP for
that env/robot. LP uses selected/gated safety AND alpha rows and actual bounds.
Only actual scipy status 2 triggers a second original project call, with all four
arms of that env using the original speed/soft-limit box. Bounds of other envs
are unchanged. Every invocation starts again at .010; widening is not persistent.
The second call receives freshly limited RAW cmd, so authority, alpha rows and
gates are recomputed by production. It does not receive any stored h or LP x.

Return remains (DeltaCmd, active, info). info and sink contain repair_* tensors:
repair_lp_status/classification have shape (N,2), robot order F,U. Raw statuses
are SciPy integers, -2=not checked, -3=exception/malformed status. Classification
is -2=not checked, -1=UNKNOWN, 0=numerically feasible, 2=infeasible. Status 0 also
requires a finite witness satisfying the actual inequalities/bounds to 1e-6.
*_count are scalar counts for this invocation; *_env flags have shape (N,).
repair_widened_env records the status-2 decision; repair_bounds_changed_env
records whether that decision changed any bound. reference_governor_changed
retains the old (N,4) FINAL command-prelimit attribution. sink's unprefixed fields
and all returned production info belong to the FINAL actual project call.

repair_queued_future_status is always -1 (UNKNOWN). Neither an LP nor a small
current residual certifies physical safety, full-row coverage, or queued future.
No queues, targets, measured state, p, alpha, config, scoring or files are written.
An existing mapping observer's detailed_snapshot flag controls publication of
snapshot_* fields. This preserves the original sparse snapshot schedule; without
that observer, full snapshots are published. LP reconstruction always uses an
internal full snapshot. This synchronous wrapper is for serial evaluation calls,
not training or concurrent/reentrant use. Snapshots cost memory and CPU LPs
synchronize device tensors; no real-time claim is made.
"""
from __future__ import annotations

from collections.abc import MutableMapping
import importlib.util
from numbers import Integral
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
import torch

from safeduo.safety.types import ARM_KEYS, ARMS_OF_ROBOT, DeltaCmd


def _sealed_module(name):
    path = Path(__file__).resolve().parent.parent / 'safety_zero_intent_20261006_2225' / (name + '.py')
    spec = importlib.util.spec_from_file_location('_adaptive_sealed_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_reference = _sealed_module('reference_envelope')
_diagnostics = _sealed_module('projection_diagnostics')
RESIDUAL_TOL = 1e-6
NOT_CHECKED, LP_ERROR, UNKNOWN, FEASIBLE, INFEASIBLE = -2, -3, -1, 0, 2


def _current_residual(record):
    return torch.stack([
        torch.stack([record[f'returned_{kind}_residual_{robot}']
                     for kind in ('safety', 'alpha', 'bound')], -1).amax(-1)
        for robot in ('F', 'U')], -1)


def _current_lp(record, robot, env_index, joint_slice):
    """Return raw solver status and conservative classification; never repair x."""
    relevant = record[f'snapshot_rel_{robot}'][env_index]
    alpha_relevant = record[f'snapshot_alpha_rel_{robot}'][env_index]
    G = torch.cat((record[f'snapshot_G_{robot}'][env_index][relevant],
                   record[f'snapshot_alpha_G_{robot}'][env_index][alpha_relevant]), 0)
    h = torch.cat((record[f'snapshot_h_after_authority_{robot}'][env_index][relevant],
                   record[f'snapshot_alpha_h_{robot}'][env_index][alpha_relevant]), 0)
    lower = record['bounds_lower'][env_index, joint_slice]
    upper = record['bounds_upper'][env_index, joint_slice]
    A, b, lo, hi = [value.detach().to(device='cpu', dtype=torch.float64).numpy()
                    for value in (G, h, lower, upper)]
    # Explicit signed bounds avoid linprog's default x >= 0. No unbounded probe.
    try:
        result = linprog(np.zeros(lo.size), A_ub=A if b.size else None,
                         b_ub=b if b.size else None, bounds=list(zip(lo, hi)),
                         method='highs', options={'primal_feasibility_tolerance': 1e-9,
                                                  'dual_feasibility_tolerance': 1e-9})
        status = getattr(result, 'status', None)
        if not isinstance(status, Integral) or isinstance(status, (bool, np.bool_)):
            return LP_ERROR, UNKNOWN
        status = int(status)
        if status == INFEASIBLE:
            return status, INFEASIBLE
        if status != FEASIBLE:
            return status, UNKNOWN
        x = np.asarray(getattr(result, 'x', None), dtype=np.float64)
        if x.shape != lo.shape or not np.isfinite(x).all():
            return status, UNKNOWN
        residual = max(float(np.max(A @ x - b, initial=0.0)),
                       float(np.max(lo - x, initial=0.0)),
                       float(np.max(x - hi, initial=0.0)))
        return status, FEASIBLE if residual <= RESIDUAL_TOL else UNKNOWN
    except Exception:
        # Solver failures are explicit UNKNOWN, never permission to widen.
        # Control-flow exceptions (KeyboardInterrupt/SystemExit) still propagate.
        return LP_ERROR, UNKNOWN


def install_adaptive(env, gap_provider, diagnostic_sink):
    """Install the scoped governor and return the captured project for restoration."""
    if gap_provider is not None and not callable(gap_provider):
        raise TypeError('gap_provider must be a zero-argument callable or None')
    if not callable(diagnostic_sink) and not isinstance(diagnostic_sink, MutableMapping):
        raise TypeError('diagnostic_sink must be a callable or mutable mapping')
    original = env._backstop.project
    if getattr(original, '_adaptive_reference_installed', False):
        raise ValueError('adaptive reference is already installed')

    def project(cmd, rows, alpha, p, dt, **kwargs):
        # Invalidate the scratch record on failure; never leave a previous frame.
        if not callable(diagnostic_sink):
            diagnostic_sink.clear()
        _diagnostics.check_project_inputs(cmd, rows, alpha, p, dt, kwargs)
        state = env.scene_state()
        gap = .010 if gap_provider is None else gap_provider()
        if gap is None or not torch.all(torch.as_tensor(gap) == .010):
            raise ValueError('adaptive first projection requires registered .010 gap')
        box = env._backstop.cfg.vmax * dt
        bounds = {arm: _reference.reference_bounds(
            state.q[arm], env._targets[arm], env._q_soft_limits[arm][..., 0],
            env._q_soft_limits[arm][..., 1], box, gap, zero_inclusive=True)
            for arm in ARM_KEYS}

        def run(actual_bounds):
            limited = DeltaCmd({arm: cmd.delta_q[arm].maximum(actual_bounds[arm][0])
                               .minimum(actual_bounds[arm][1]) for arm in ARM_KEYS})
            actual_kwargs = dict(kwargs, delta_bounds=actual_bounds)
            result = _diagnostics.checked_project(original, limited, rows, alpha, p, dt,
                                                   **actual_kwargs)
            observed_detail = (diagnostic_sink.get('detailed_snapshot')
                               if isinstance(diagnostic_sink, MutableMapping) else None)
            with torch.no_grad():
                record = _diagnostics.projection_record(env._backstop, limited, rows,
                    alpha, p, dt, actual_kwargs, result, snapshot=True)
                _diagnostics.require_finite_tree('adaptive current snapshot', record)
            return result, record, limited, observed_detail

        result, first_record, limited, observed_detail = run(bounds)
        residual = _current_residual(first_record)
        checked = residual > RESIDUAL_TOL
        status = torch.full_like(residual, NOT_CHECKED, dtype=torch.long)
        classification = status.clone()
        offset = 0
        for robot_index, robot in enumerate(('F', 'U')):
            width = sum(cmd.delta_q[a].shape[-1] for a in ARMS_OF_ROBOT[robot])
            joint_slice = slice(offset, offset + width)
            for env_index in checked[:, robot_index].nonzero().flatten().tolist():
                raw, label = _current_lp(first_record, robot, env_index, joint_slice)
                status[env_index, robot_index] = raw
                classification[env_index, robot_index] = label
            offset += width
        widened = (classification == INFEASIBLE).any(-1)
        bounds_changed = torch.zeros_like(widened)
        record = first_record
        calls = 1
        if widened.any():
            final_bounds = {}
            for arm in ARM_KEYS:
                reachable = _reference.reference_bounds(state.q[arm], env._targets[arm],
                    env._q_soft_limits[arm][..., 0], env._q_soft_limits[arm][..., 1],
                    box, gap=None, zero_inclusive=True)
                final_bounds[arm] = tuple(torch.where(widened[:, None], wide, narrow)
                                         for wide, narrow in zip(reachable, bounds[arm]))
                bounds_changed |= torch.stack([new != old for new, old in
                                               zip(final_bounds[arm], bounds[arm])], -1).any(-1).any(-1)
            # A whole-batch production call preserves the original solver/gates.
            # Its batch-global early stop can also affect unchanged env iterates.
            result, record, limited, observed_detail = run(final_bounds)
            bounds = final_bounds
            calls = 2

        output, active, production_info = result
        for arm in ARM_KEYS:
            if ((output.delta_q[arm] < bounds[arm][0] - RESIDUAL_TOL) |
                    (output.delta_q[arm] > bounds[arm][1] + RESIDUAL_TOL)).any():
                raise ValueError('adaptive output exceeds final reachable bounds')
        changed = torch.stack([(limited.delta_q[a] - cmd.delta_q[a]).abs().amax(-1)
                               > RESIDUAL_TOL for a in ARM_KEYS], -1)
        active = active | changed
        info = dict(production_info)
        info['reference_governor_changed'] = changed
        metadata = dict(
            repair_lp_checked=checked, repair_lp_status=status,
            repair_lp_classification=classification,
            repair_lp_checks_count=checked.sum(),
            repair_lp_feasible_count=(classification == FEASIBLE).sum(),
            repair_lp_infeasible_count=(classification == INFEASIBLE).sum(),
            repair_lp_unknown_count=(classification == UNKNOWN).sum(),
            repair_lp_error_count=((status == LP_ERROR) & checked).sum(),
            repair_lp_unknown_env=(classification == UNKNOWN).any(-1),
            repair_widened_env=widened, repair_widened_env_count=widened.sum(),
            repair_bounds_changed_env=bounds_changed,
            repair_bounds_changed_env_count=bounds_changed.sum(),
            repair_projection_calls=torch.tensor(calls, device=rows.d.device),
            repair_first_current_residual=residual,
            repair_final_current_residual=_current_residual(record),
            repair_final_residual_env=(_current_residual(record) > RESIDUAL_TOL).any(-1),
            repair_queued_future_status=torch.full_like(widened, UNKNOWN, dtype=torch.long),
            repair_physical_safety_certified_env=torch.zeros_like(widened))
        info.update(metadata)
        record.update({key: value.detach().clone() for key, value in metadata.items()})
        record.update(external_raw_cmd=cmd.stacked().detach().clone(),
                      outer_returned_cmd=output.stacked().detach().clone(),
                      outer_active=active.detach().clone(),
                      governor_changed_available=torch.ones_like(changed),
                      governor_changed=changed.detach().clone())
        final_result = output, active, info
        _diagnostics.check_project_result(final_result)
        if observed_detail is not None:
            record['detailed_snapshot'] = observed_detail.detach().clone()
            if not observed_detail.any():
                record = _diagnostics.summary_record(record)
        if callable(diagnostic_sink):
            diagnostic_sink(record)
        else:
            diagnostic_sink.clear()
            diagnostic_sink.update(record)
        return final_result

    project._adaptive_reference_installed = True
    env._backstop.project = project
    return original
