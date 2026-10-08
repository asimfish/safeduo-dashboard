"""CPU environment stubs around REAL production projection and scipy HiGHS.

Run: CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
     OPENBLAS_NUM_THREADS=1 <safeduo-python> -B test_adaptive_reference.py -v
No Isaac, rendering, filesystem fixtures, GPU calls or subprocesses in this suite.
The parent records the actual child exit code, source hashes and combined log.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog as real_linprog
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(HERE))

import adaptive_reference as adaptive
from safeduo.baselines.base import ConstraintRows
from safeduo.safety.backstop import BackstopConfig, VelocityDamperBackstop
from safeduo.safety.types import ARM_KEYS, ARMS_OF_ROBOT, DOF_OF, DeltaCmd

DT = .1


def environment(n=1, **config):
    """Only scene/targets are stubbed; numeric production code is imported intact."""
    q = {arm: torch.zeros(n, DOF_OF[arm]) for arm in ARM_KEYS}
    env = SimpleNamespace(
        num_envs=n, device='cpu', q=q,
        _targets={arm: value.clone() for arm, value in q.items()},
        _q_soft_limits={arm: torch.stack((torch.full_like(value, -1),
                                         torch.full_like(value, 1)), -1)
                        for arm, value in q.items()},
        _backstop=VelocityDamperBackstop(BackstopConfig(
            gamma=1, d_min=0, vmax=1, **config)))
    env.scene_state = lambda: SimpleNamespace(q=env.q)
    # Opaque real-FIFO stand-in: mechanism must never dereference or mutate it.
    env._evaluation_actuator_delay = SimpleNamespace(
        queue=SimpleNamespace(pending=tuple(object() for _ in range(6))))
    env._pending_target_history = object()
    return env


def rows_for(env, width=1):
    n = env.num_envs
    return ConstraintRows(
        d=torch.ones(n, width),
        J={robot: torch.zeros(n, width, sum(DOF_OF[a] for a in ARMS_OF_ROBOT[robot]))
           for robot in ('F', 'U')},
        cls=torch.full((n, width), 2.),
        arm_mask=torch.zeros(n, width, 4, dtype=torch.bool),
        valid=torch.zeros(n, width, dtype=torch.bool))


def inputs(env):
    return (DeltaCmd({arm: torch.zeros_like(value) for arm, value in env.q.items()}),
            torch.ones(env.num_envs, 4), torch.zeros(env.num_envs))


def add_row(rows, env_id, row, robot, G, cap):
    rows.J[robot][env_id, row, :len(G)] = -torch.tensor(G)
    rows.d[env_id, row] = cap / DT
    rows.arm_mask[env_id, row, 0 if robot == 'F' else 2] = True
    rows.valid[env_id, row] = True


def recovering_case(n=1):
    env = environment(n)
    rows = rows_for(env)
    cmd, alpha, p = inputs(env)
    add_row(rows, 0, 0, 'F', [1.], -.020)
    cmd.delta_q['F_L'][0, 0] = -.080
    alpha[0, 0] = .5
    return env, rows, cmd, alpha, p


def spy_on_original(env):
    calls = []
    original = env._backstop.project

    def project(cmd, rows, alpha, p, dt, **kwargs):
        result = original(cmd, rows, alpha, p, dt, **kwargs)
        record = adaptive._diagnostics.projection_record(
            env._backstop, cmd, rows, alpha, p, dt, kwargs, result, snapshot=True)
        calls.append(dict(cmd=cmd.clone(), rows=rows, alpha=alpha, p=p,
                          kwargs=kwargs, result=result, record=record))
        return result

    env._backstop.project = project
    return calls, project


def solve_snapshot(record, robot='F', *, bounds='actual', h_override=None):
    """Independent LP oracle from recorded actual inequalities (single env)."""
    rel = record[f'snapshot_rel_{robot}'][0]
    arel = record[f'snapshot_alpha_rel_{robot}'][0]
    h = record[f'snapshot_h_after_authority_{robot}'] if h_override is None else h_override
    A = torch.cat((record[f'snapshot_G_{robot}'][0][rel],
                   record[f'snapshot_alpha_G_{robot}'][0][arel])).double().numpy()
    b = torch.cat((h[0][rel], record[f'snapshot_alpha_h_{robot}'][0][arel])).double().numpy()
    sl = slice(0, 14) if robot == 'F' else slice(14, 26)
    lo, hi = (record[f'bounds_{side}'][0, sl].double().numpy() for side in ('lower', 'upper'))
    lp_bounds = list(zip(lo, hi)) if bounds == 'actual' else [(None, None)] * len(lo)
    return real_linprog(np.zeros(len(lo)), A_ub=A, b_ub=b, bounds=lp_bounds,
                        method='highs', options={'primal_feasibility_tolerance': 1e-9,
                                                 'dual_feasibility_tolerance': 1e-9})


class AdaptiveTests(unittest.TestCase):
    def assert_tensor_equal(self, actual, expected):
        self.assertTrue(torch.equal(actual, expected), f'{actual}\n!=\n{expected}')

    def assert_final_record(self, sink, calls, output, active, info):
        final = calls[-1]
        for key, value in final['record'].items():
            if key not in ('governor_changed', 'governor_changed_available'):
                self.assert_tensor_equal(sink[key], value)
        self.assert_tensor_equal(sink['returned_cmd'], output.stacked())
        self.assert_tensor_equal(sink['outer_returned_cmd'], output.stacked())
        self.assert_tensor_equal(sink['outer_active'], active)
        for key, value in final['result'][2].items():
            if torch.is_tensor(value):
                self.assert_tensor_equal(info[key], value)
            else:
                self.assertEqual(info[key], value)
        for key, value in info.items():
            if key.startswith('repair_'):
                self.assert_tensor_equal(sink[key], value)
        self.assertTrue((info['repair_queued_future_status'] == adaptive.UNKNOWN).all())
        self.assertFalse(info['repair_physical_safety_certified_env'].any())

    def test_first_projection_exact_zero_inclusive_reference_and_feasible_unchanged(self):
        # Multiple tracking debts, soft-limit clipping, mixed alpha and R19.
        old, new = environment(3), environment(3)
        for env in (old, new):
            env._targets['F_L'][1] = .95
            env.q['F_L'][1] = .80
            env.q['U_R'][2] = -.08
        rows = rows_for(new, 2)
        add_row(rows, 0, 0, 'F', [1.], .1)
        cmd, alpha, p = inputs(new)
        for i, a in enumerate(ARM_KEYS):
            cmd.delta_q[a].fill_(.4 if i % 2 else -.4)
        alpha[0] = .5
        p[:] = torch.tensor([-.7, .0, .8])
        bypass = torch.zeros(3, 4, dtype=torch.bool)
        bypass[1:] = True
        old_sink, sink = {}, {}
        adaptive._diagnostics.install(old, old_sink, snapshot=True)
        adaptive._reference.install(old, 'envelope_050', lambda: .010, zero_inclusive=True)
        calls, original = spy_on_original(new)
        self.assertIs(adaptive.install_adaptive(new, lambda: .010, sink), original)
        expected = old._backstop.project(cmd, rows, alpha, p, DT, bypass_arm=bypass)
        with patch.object(adaptive, 'linprog', side_effect=AssertionError('unexpected LP')):
            actual = new._backstop.project(cmd, rows, alpha, p, DT, bypass_arm=bypass)
        self.assert_tensor_equal(actual[0].stacked(), expected[0].stacked())
        self.assert_tensor_equal(actual[1], expected[1])
        for key in old_sink:
            if key not in ('governor_changed', 'governor_changed_available'):
                self.assert_tensor_equal(sink[key], old_sink[key])
        self.assertEqual(len(calls), 1)
        self.assertEqual(int(actual[2]['repair_lp_checks_count']), 0)
        self.assertFalse(actual[2]['repair_widened_env'].any())
        self.assert_final_record(sink, calls, *actual)

    def test_status2_reprojects_raw_command_recomputes_alpha_and_authority(self):
        env, rows, cmd, alpha, p = recovering_case()
        calls, _ = spy_on_original(env)
        sink = {}
        adaptive.install_adaptive(env, lambda: torch.full((1, 1), .010), sink)
        output, active, info = env._backstop.project(cmd, rows, alpha, p, DT)
        self.assertEqual(len(calls), 2)
        self.assertEqual(info['repair_lp_status'].tolist(), [[2, -2]])
        self.assertTrue(info['repair_bounds_changed_env'][0])
        first, final = (call['record'] for call in calls)
        self.assertAlmostEqual(float(first['project_input_cmd'][0, 0]), -.010, places=7)
        self.assertAlmostEqual(float(final['project_input_cmd'][0, 0]), -.080, places=7)
        self.assertAlmostEqual(float(first['snapshot_alpha_h_F'][0, 0]), .005, places=7)
        self.assertAlmostEqual(float(final['snapshot_alpha_h_F'][0, 0]), .040, places=7)
        self.assertAlmostEqual(float(first['snapshot_h_after_authority_F'][0, 0]), -.009, places=7)
        self.assertAlmostEqual(float(final['snapshot_h_after_authority_F'][0, 0]), -.020, places=7)
        self.assertAlmostEqual(float(output.delta_q['F_L'][0, 0]), -.040, places=6)
        self.assertFalse(info['repair_final_residual_env'].any())
        self.assertEqual(solve_snapshot(final).status, 0)
        self.assertIs(calls[1]['alpha'], alpha)
        self.assertIs(calls[1]['p'], p)
        self.assert_final_record(sink, calls, output, active, info)

    def test_unbounded_and_frozen_h_negative_controls_do_not_certify_wider_box(self):
        env = environment()
        rows = rows_for(env, 2)
        add_row(rows, 0, 0, 'F', [1., 0.], -1.)
        add_row(rows, 0, 1, 'F', [-1., 1.], -1.)
        cmd, alpha, p = inputs(env)
        calls, _ = spy_on_original(env)
        sink = {}
        adaptive.install_adaptive(env, None, sink)
        result = env._backstop.project(cmd, rows, alpha, p, DT)
        first, final = (call['record'] for call in calls)
        self.assertEqual(solve_snapshot(first).status, 2)
        self.assertEqual(solve_snapshot(first, bounds='unbounded').status, 0)
        self.assertEqual(solve_snapshot(final, h_override=first['snapshot_h_after_authority_F']).status, 0)
        self.assertEqual(solve_snapshot(final).status, 2)
        torch.testing.assert_close(final['snapshot_h_after_authority_F'],
                                   torch.tensor([[-.09, -.18]]), rtol=0, atol=2e-8)
        self.assertGreater(float(result[2]['repair_final_current_residual'][0, 0]), 1e-6)
        self.assertTrue(result[2]['repair_final_residual_env'][0])
        self.assert_final_record(sink, calls, *result)

    def test_individually_feasible_rows_jointly_impossible(self):
        env = environment()
        rows = rows_for(env, 2)
        add_row(rows, 0, 0, 'F', [1.], -.5)
        add_row(rows, 0, 1, 'F', [-1.], -.5)
        cmd, alpha, p = inputs(env)
        calls, _ = spy_on_original(env)
        adaptive.install_adaptive(env, None, {})
        result = env._backstop.project(cmd, rows, alpha, p, DT)
        first = calls[0]['record']
        self.assertEqual(int(first['individually_infeasible_row_count_F'][0]), 0)
        for row in range(2):
            single = real_linprog(np.zeros(14),
                A_ub=first['snapshot_G_F'][0, row:row+1].numpy(),
                b_ub=first['snapshot_h_after_authority_F'][0, row:row+1].numpy(),
                bounds=[(-.01, .01)] * 14, method='highs')
            self.assertEqual(single.status, 0)
        self.assertEqual(solve_snapshot(first).status, 2)
        self.assertEqual(int(result[2]['repair_lp_infeasible_count']), 1)
        self.assertTrue(result[2]['repair_final_residual_env'][0])

    def test_feasible_lp_with_iteration_residual_keeps_original_box(self):
        env = environment()
        rows = rows_for(env, 2)
        add_row(rows, 0, 0, 'F', [1., 0.], -.001)
        add_row(rows, 0, 1, 'F', [-1., .1], 0.)
        cmd, alpha, p = inputs(env)
        calls, _ = spy_on_original(env)
        adaptive.install_adaptive(env, None, {})
        result = env._backstop.project(cmd, rows, alpha, p, DT)
        self.assertGreater(float(result[2]['repair_first_current_residual'][0, 0]), 1e-6)
        self.assertEqual(result[2]['repair_lp_status'].tolist(), [[0, -2]])
        self.assertEqual(int(result[2]['repair_lp_feasible_count']), 1)
        self.assertFalse(result[2]['repair_widened_env'].any())
        self.assertEqual(len(calls), 1)
        self.assertEqual(result[2]['passes_F'], 30)

    def test_unknown_statuses_and_solver_errors_never_widen_or_become_feasible(self):
        cases = [SimpleNamespace(status=s, x=np.zeros(14)) for s in (1, 3, 4, 99, None, '2', True)]
        cases += [RuntimeError('synthetic solver failure'),
                  SimpleNamespace(status=0, x=None),
                  SimpleNamespace(status=0, x=np.full(14, np.nan)),
                  SimpleNamespace(status=0, x=np.zeros(14))]
        for response in cases:
            with self.subTest(response=repr(response)):
                env, rows, cmd, alpha, p = recovering_case()
                calls, _ = spy_on_original(env)
                sink = {}
                adaptive.install_adaptive(env, None, sink)
                with patch.object(adaptive, 'linprog', **(
                    {'side_effect': response} if isinstance(response, Exception) else {'return_value': response})):
                    result = env._backstop.project(cmd, rows, alpha, p, DT)
                info = result[2]
                self.assertEqual(int(info['repair_lp_checks_count']), 1)
                self.assertEqual(int(info['repair_lp_unknown_count']), 1)
                self.assertEqual(int(info['repair_lp_feasible_count']), 0)
                self.assertEqual(int(info['repair_lp_infeasible_count']), 0)
                self.assertEqual(info['repair_lp_classification'].tolist(), [[-1, -2]])
                self.assertFalse(info['repair_widened_env'].any())
                self.assertTrue(info['repair_lp_unknown_env'][0])
                self.assertEqual(len(calls), 1)
                self.assert_final_record(sink, calls, *result)

    def test_only_status2_environments_widen_and_next_call_restarts_narrow(self):
        env, rows, cmd, alpha, p = recovering_case(3)
        add_row(rows, 1, 0, 'U', [1.], -.020)
        cmd.delta_q['U_L'][1, 0] = -.080
        alpha[1, 2] = .5
        calls, _ = spy_on_original(env)
        sink = {}
        adaptive.install_adaptive(env, None, sink)
        # F/env0 real infeasible; U/env1 solver UNKNOWN; env2 needs no LP.
        with patch.object(adaptive, 'linprog', side_effect=[
                SimpleNamespace(status=2), SimpleNamespace(status=4)]):
            result = env._backstop.project(cmd, rows, alpha, p, DT)
        info = result[2]
        self.assertEqual(info['repair_widened_env'].tolist(), [True, False, False])
        self.assertEqual(info['repair_lp_unknown_env'].tolist(), [False, True, False])
        self.assertEqual(int(info['repair_lp_checks_count']), 2)
        self.assertEqual(int(info['repair_lp_unknown_count']), 1)
        self.assertEqual(int(info['repair_lp_infeasible_count']), 1)
        first, final = (call['record'] for call in calls)
        self.assert_tensor_equal(first['bounds_lower'][1:], final['bounds_lower'][1:])
        self.assert_tensor_equal(first['bounds_upper'][1:], final['bounds_upper'][1:])
        self.assert_tensor_equal(final['bounds_lower'][0], torch.full((26,), -.1))
        self.assert_tensor_equal(final['bounds_upper'][0], torch.full((26,), .1))
        self.assert_final_record(sink, calls, *result)
        # Real next invocation rechecks both robots; no sticky relaxed mode.
        result = env._backstop.project(cmd, rows, alpha, p, DT)
        self.assert_tensor_equal(calls[2]['record']['bounds_lower'], first['bounds_lower'])
        self.assertEqual(result[2]['repair_widened_env'].tolist(), [True, True, False])

    def test_final_sink_callback_only_once_detached_and_outer_gate_compatible(self):
        env, rows, cmd, alpha, p = recovering_case()
        calls, _ = spy_on_original(env)
        records = []
        adaptive.install_adaptive(env, None, records.append)
        result = env._backstop.project(cmd, rows, alpha, p, DT)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(calls), 2)
        self.assert_final_record(records[0], calls, *result)
        saved = records[0]['returned_cmd'].clone()
        result[0].delta_q['F_L'].fill_(.6)
        self.assert_tensor_equal(records[0]['returned_cmd'], saved)
        env, rows, cmd, alpha, p = recovering_case()
        sink = {'stale': torch.tensor(123)}
        adaptive._diagnostics.install(env, sink, snapshot=False)
        adaptive.install_adaptive(env, None, sink)
        result = adaptive._diagnostics.checked_project(
            env._backstop.project, cmd, rows, alpha, p, DT, diagnostic_sink=sink)
        self.assertNotIn('stale', sink)
        self.assertEqual(int(sink['repair_projection_calls']), 2)
        self.assert_tensor_equal(sink['returned_cmd'], result[0].stacked())
        self.assertFalse(sink['detailed_snapshot'].any())
        self.assertFalse(any(key.startswith('snapshot_') for key in sink))
        self.assertAlmostEqual(float(sink['project_input_cmd'][0, 0]), -.08, places=7)

    def test_original_observer_sparse_snapshot_schedule_is_preserved(self):
        env, rows, cmd, alpha, p = recovering_case()
        sink = {}
        want_snapshot = False
        adaptive._diagnostics.install(env, sink, snapshot=lambda: want_snapshot)
        adaptive.install_adaptive(env, None, sink)
        for want_snapshot in (False, True, False):
            result = env._backstop.project(cmd, rows, alpha, p, DT)
            self.assertEqual(bool(sink['detailed_snapshot'].all()), want_snapshot)
            self.assertEqual(any(key.startswith('snapshot_') for key in sink), want_snapshot)
            self.assert_tensor_equal(sink['returned_cmd'], result[0].stacked())
            self.assertAlmostEqual(float(sink['project_input_cmd'][0, 0]), -.08, places=7)
            if want_snapshot:
                self.assertAlmostEqual(float(sink['snapshot_alpha_h_F'][0, 0]), .04, places=7)
                self.assertAlmostEqual(float(sink['snapshot_h_after_authority_F'][0, 0]), -.02, places=7)

    def test_soft_limit_speed_zero_inclusion_and_R19_survive_widening(self):
        env, rows, cmd, alpha, p = recovering_case()
        # Restrict a bypassed arm with original soft limits and shifted q.
        env._targets['F_R'].fill_(.98)
        env.q['F_R'].fill_(.93)
        cmd.delta_q['F_R'].fill_(.50)
        bypass = torch.tensor([[False, True, True, True]])
        calls, _ = spy_on_original(env)
        sink = {}
        cfg_before = asdict(env._backstop.cfg)
        target_before = {a: value.clone() for a, value in env._targets.items()}
        q_before = {a: value.clone() for a, value in env.q.items()}
        alpha_before, p_before = alpha.clone(), p.clone()
        pending = env._evaluation_actuator_delay.queue.pending
        history = env._pending_target_history
        adaptive.install_adaptive(env, None, sink)
        result = env._backstop.project(cmd, rows, alpha, p, DT, bypass_arm=bypass)
        for call in calls:
            record = call['record']
            self.assertTrue((record['bounds_lower'] <= 0).all())
            self.assertTrue((record['bounds_upper'] >= 0).all())
            self.assertTrue((record['project_input_cmd'] >= record['bounds_lower']).all())
            self.assertTrue((record['project_input_cmd'] <= record['bounds_upper']).all())
            self.assertIs(call['kwargs']['bypass_arm'], bypass)
            for index, a in enumerate(ARM_KEYS):
                if bypass[0, index]:
                    self.assert_tensor_equal(call['result'][0].delta_q[a], call['cmd'].delta_q[a])
                    self.assertFalse(call['result'][1][0, index])
        for a in ARM_KEYS:
            self.assertTrue((result[0].delta_q[a].abs() <= .100001).all())
            target = env._targets[a] + result[0].delta_q[a]
            self.assertTrue((target >= env._q_soft_limits[a][..., 0] - 1e-6).all())
            self.assertTrue((target <= env._q_soft_limits[a][..., 1] + 1e-6).all())
            self.assert_tensor_equal(env._targets[a], target_before[a])
            self.assert_tensor_equal(env.q[a], q_before[a])
        self.assertAlmostEqual(float(result[0].delta_q['F_R'][0, 0]), .02, places=6)
        self.assert_tensor_equal(alpha, alpha_before)
        self.assert_tensor_equal(p, p_before)
        self.assertEqual(asdict(env._backstop.cfg), cfg_before)
        self.assertEqual(env._backstop.cfg.max_passes, 30)
        self.assertIs(env._evaluation_actuator_delay.queue.pending, pending)
        self.assertEqual(len(pending), 6)
        self.assertIs(env._pending_target_history, history)
        self.assert_final_record(sink, calls, *result)

    def test_zero_raw_command_does_not_receive_governor_injection(self):
        env = environment()
        env.q['F_L'].fill_(.4)
        rows = rows_for(env)
        cmd, alpha, p = inputs(env)
        sink = {}
        adaptive.install_adaptive(env, None, sink)
        result = env._backstop.project(cmd, rows, alpha, p, DT)
        self.assert_tensor_equal(result[0].stacked(), torch.zeros(1, 26))
        self.assertFalse(result[2]['reference_governor_changed'].any())
        self.assert_tensor_equal(sink['project_input_cmd'], torch.zeros(1, 26))
        self.assertTrue((sink['bounds_lower'] <= 0).all())
        self.assertTrue((sink['bounds_upper'] >= 0).all())

    def test_post_R19_actual_residual_is_observed_even_when_pre_bypass_info_is_zero(self):
        env = environment()
        rows = rows_for(env)
        add_row(rows, 0, 0, 'F', [1.], .005)
        cmd, alpha, p = inputs(env)
        cmd.delta_q['F_L'][0, 0] = .01
        calls, _ = spy_on_original(env)
        sink = {}
        adaptive.install_adaptive(env, None, sink)
        result = env._backstop.project(cmd, rows, alpha, p, DT,
                                      bypass_arm=torch.tensor([[True, False, False, False]]))
        self.assertAlmostEqual(float(result[2]['residual_F'][0]), 0., places=7)
        self.assertGreater(float(sink['returned_safety_residual_F'][0]), .0049)
        self.assertEqual(int(result[2]['repair_lp_checks_count']), 1)
        self.assertEqual(int(result[2]['repair_lp_feasible_count']), 1)
        self.assertFalse(result[2]['repair_widened_env'].any())
        self.assertEqual(len(calls), 1)

    def test_final_gates_backlog_and_priority_match_direct_original_project(self):
        env, _, cmd, alpha, p = recovering_case()
        env._backstop.cfg.engage_dist = .04
        env._backstop.cfg.exempt_structural_rows = True
        env._backstop.cfg.retain_conditional_rows = True
        env._backstop.cfg.predict_backlog = True
        env._backstop.cfg.backlog_aware = True
        env._backstop.cfg.pending_target_steps = 6
        env._backstop.cfg.lookahead_s = .02
        rows = rows_for(env, 4)
        add_row(rows, 0, 0, 'F', [1.], -.020)
        add_row(rows, 0, 1, 'F', [-1.], .01)  # far positive row: disengaged
        add_row(rows, 0, 2, 'U', [1.], .001)  # near structural: exempt
        add_row(rows, 0, 3, 'U', [-1.], .002) # conditional contact: retained
        rows.cls[0, 3] = 0.
        p.fill_(.6)
        zero = {a: torch.zeros_like(v) for a, v in cmd.delta_q.items()}
        kwargs = dict(backlog=zero, past_backlogs=[zero for _ in range(6)], qd=zero,
                      struct_exempt=torch.tensor([[False, False, True, True]]),
                      contact_exempt=torch.tensor([[False, False, False, True]]))
        calls, _ = spy_on_original(env)
        sink = {}
        adaptive.install_adaptive(env, None, sink)
        result = env._backstop.project(cmd, rows, alpha, p, DT, **kwargs)
        self.assertEqual(len(calls), 2)
        self.assertEqual(sink['snapshot_row_gate'].tolist(), [[True, False, False, True]])
        self.assertAlmostEqual(float(sink['snapshot_raw_h_U'][0, 3]), .0004, places=7)
        for call in calls:
            for name, value in kwargs.items():
                self.assertIs(call['kwargs'][name], value)
        self.assert_final_record(sink, calls, *result)

    def test_invalid_inputs_fail_before_original_and_sink_has_no_stale_frame(self):
        for gap in (.05, None, float('nan')):
            with self.subTest(gap=gap):
                env = environment()
                rows = rows_for(env)
                cmd, alpha, p = inputs(env)
                calls, _ = spy_on_original(env)
                sink = {'old': torch.tensor(1.)}
                adaptive.install_adaptive(env, lambda: gap, sink)
                with self.assertRaises(ValueError):
                    env._backstop.project(cmd, rows, alpha, p, DT)
                self.assertEqual(calls, [])
                self.assertEqual(sink, {})
        env = environment()
        rows = rows_for(env)
        cmd, alpha, p = inputs(env)
        cmd.delta_q['F_L'][0, 0] = float('nan')
        calls, _ = spy_on_original(env)
        adaptive.install_adaptive(env, None, {})
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            env._backstop.project(cmd, rows, alpha, p, DT)
        self.assertEqual(calls, [])
        with self.assertRaisesRegex(ValueError, 'already installed'):
            adaptive.install_adaptive(env, None, {})


if __name__ == '__main__':
    torch.set_num_threads(1)
    print('CPU_ONLY real VelocityDamperBackstop + scipy HiGHS; no Isaac/GPU/rendering', flush=True)
    print(f'RUNTIME python={sys.version.split()[0]} torch={torch.__version__}', flush=True)
    unittest.main(verbosity=2)
