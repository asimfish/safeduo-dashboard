"""Independent CPU-only review checks; no simulator imports or disk fixtures.

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python3 astra_motion_runner_tests.py
Runner methods are compiled verbatim from their AST, using a CPU scene fixture
and an in-memory output directory. Model coefficients used for behavioral
oracles are synthetic and are never fitted to experiment outcomes.
"""
import ast
from dataclasses import dataclass, replace
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import torch

H = Path(__file__).resolve().parent
REPO = Path('/home/liyufeng/safeduo')
OLD = H.parent / 'safety_feasible_guard_20261005_2100'
sys.path[:0] = [str(H), str(REPO/'src')]
torch.set_num_threads(1)

import motion_forecast as motion
import projection_diagnostics as diagnostics
import reference_envelope as reference
import target_forecast as target
from astra_fifo_oracle import DiagonalDynamics, affine_horizons
from safeduo.baselines.base import ConstraintRows, stack_robot
from safeduo.safety.backstop import BackstopConfig, VelocityDamperBackstop
from safeduo.safety.types import ARM_KEYS, DeltaCmd


def arm_dict(value):
    return dict(zip(ARM_KEYS, value.split([7, 7, 6, 6], -1)))


def coeff(dtype=torch.float64):
    cq = torch.tensor([[.01, .23, .0001]], dtype=dtype).repeat(26, 1)
    cv = torch.tensor([[.7, 1.2, -.002]], dtype=dtype).repeat(26, 1)
    cq[3, 1], cv[4, 1] = -.04, -.2
    return cq, cv


def extract(path, names, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


ADMISSION_PATH = H.parent/'safety_mechanism_20261005_causal_obs/mechanism_runner.py'
admission = extract(ADMISSION_PATH, ('union_mask', 'select'),
                    {'torch': torch, 'replace': replace, 'CAPACITY': 1024})


@dataclass
class Geometry:
    dists: torch.Tensor
    closing: torch.Tensor
    full_dmin: torch.Tensor
    full_viol_exempt: torch.Tensor
    active_idx: torch.Tensor
    active_mask: torch.Tensor
    active_pairs: torch.Tensor
    active_dmin: torch.Tensor
    viol_exempt: torch.Tensor


def scene(n=2, p=9, dtype=torch.float64):
    q = torch.zeros((n, 26), dtype=dtype)
    qd = torch.linspace(-.6, .6, 26, dtype=dtype).expand(n, -1).clone()
    state = SimpleNamespace(q=arm_dict(q), qd=arm_dict(qd), dt=.016666)
    issued = arm_dict(q + .015)
    pending = [arm_dict(q + i*.004) for i in range(6)]
    cmd = DeltaCmd(arm_dict(torch.linspace(-.1, .1, 26, dtype=dtype).expand(n, -1)))
    limits = {a: torch.stack((torch.full_like(v, -2.), torch.full_like(v, 2.)), -1)
              for a, v in state.q.items()}
    d = torch.full((n, p), .08, dtype=dtype)
    dm = torch.full_like(d, .02)
    ids = torch.arange(min(p, 32)).expand(n, -1).clone()
    valid = torch.ones_like(ids, dtype=torch.bool)
    sph = SimpleNamespace(class_id=torch.full((p,), 2., dtype=dtype),
                          pair_id=torch.arange(p, dtype=dtype),
                          qualified_names=['sphere'], radii=torch.ones(1, dtype=dtype),
                          arm_id=torch.zeros(1, dtype=torch.long),
                          pair_table=torch.zeros((p, 2), dtype=torch.long))
    features = torch.stack((d[:, :ids.shape[1]], torch.zeros_like(ids, dtype=dtype),
                            sph.class_id[ids], sph.pair_id[ids]), -1)
    full = Geometry(d, torch.zeros_like(d), dm, torch.zeros_like(d, dtype=torch.bool),
                    ids, valid, features, dm[:, :ids.shape[1]], torch.zeros_like(valid))
    J = torch.zeros((n, p, 26), dtype=dtype)
    J[:, :, 0] = torch.linspace(-1., 1., p, dtype=dtype)
    J[:, :, 14] = -.3
    rows = SimpleNamespace(J={'F': J[:, :, :14], 'U': J[:, :, 14:]})
    return state, issued, pending, cmd, limits, full, rows, sph


class MemoryPath:
    def __init__(self, name='memory', files=None):
        self.name = name
        self.files = {} if files is None else files

    def __truediv__(self, child):
        return MemoryPath(self.name+'/'+str(child), self.files)

    def __str__(self):
        return self.name

    def write_text(self, value):
        self.files[self.name] = value.encode()

    def read_text(self):
        return self.files[self.name].decode()

    def read_bytes(self):
        return self.files[self.name]

    def exists(self):
        return self.name in self.files

    def mkdir(self, **kwargs):
        pass

    def relative_to(self, root):
        return self.name[len(str(root))+1:]


class RiskStandIn:
    def start(self, env):
        self.out = MemoryPath()
        (self.out/'random_manifest.json').write_text('{}')
        self.original_safety = env.safety_dist_out
        self.frames = {}
        self.t = 1


def runner_class():
    helpers = SimpleNamespace(union_mask=admission.union_mask, select=admission.select,
                              risk=SimpleNamespace(RiskTrace=RiskStandIn))
    namespace = dict(replace=replace, hashlib=hashlib, json=json, os=os, Path=Path,
                     sys=sys, np=np, torch=torch, HERE=H, admission=helpers,
                     full_forecast=target.full_forecast, require_finite=target.require_finite,
                     distance_forecasts=motion.distance_forecasts, MODES=motion.MODES,
                     install=reference.install, reference_bounds=reference.reference_bounds,
                     stack_robot=stack_robot, ARM_KEYS=ARM_KEYS, DeltaCmd=DeltaCmd,
                     CHUNK_STEPS=32, DIAGNOSTIC_STEPS=[0, 60, 62, 66, 71, 75, 180, 480, 959])
    return extract(H/'guard_runner.py', ('GuardTrace',), namespace).GuardTrace


def start_runner(mode, all_admitted=False):
    state, issued, pending, cmd, limits, full, rows, sph = scene(64, 9021, torch.float32)
    if all_admitted:
        full.dists[:] = .015
        full.active_pairs[:, :, 0] = .015
    def rows_from(out, body):
        idx = out.active_idx.clamp_min(0)
        matrices = {r: value.gather(1, idx[:, :, None].expand(-1, -1, value.shape[-1]))
                    for r, value in rows.J.items()}
        return ConstraintRows(out.active_pairs[:, :, 0], matrices, out.active_pairs[:, :, 2],
                              out.active_mask[:, :, None].expand(-1, -1, 4),
                              out.active_mask, out.active_dmin)
    backstop = VelocityDamperBackstop(BackstopConfig())
    env = SimpleNamespace(num_envs=64, device=torch.device('cpu'),
        _delta_src=SimpleNamespace(sample=lambda _: cmd), _sph=sph,
        _provider=SimpleNamespace(rows_from=rows_from), _backstop=backstop,
        _pre_physics_step=lambda _: None, _last_out=full, _body_pos_cache={},
        _pending_cmd=cmd, _targets=issued, _q_soft_limits=limits,
        _evaluation_actuator_delay=SimpleNamespace(queue=SimpleNamespace(pending=pending), applied=pending[0]),
        safety_dist_out=lambda: full, scene_state=lambda: state)
    trace = runner_class()()
    with patch.dict(os.environ, {'SAFEDUO_JOINT_MODE': mode}):
        trace.start(env)
    return trace, env, rows


class RecurrenceTests(unittest.TestCase):
    def test_simultaneous_old_q_v_matches_independent_affine_oracle(self):
        rng = np.random.default_rng(7201)
        q, v = torch.tensor(rng.normal(size=(3, 26))), torch.tensor(rng.normal(size=(3, 26)))
        pending, proposal = torch.tensor(rng.normal(size=(6, 3, 26))), torch.tensor(rng.normal(size=(3, 26)))
        cq, cv = coeff()
        horizons = (1, 6, 7, 12, 18)
        cv_d, pd_d = motion.motion_displacements(q, v, pending, proposal, cq, cv, .016666, horizons)
        for env in range(3):
            oracle = affine_horizons(DiagonalDynamics(cq.numpy(), cv.numpy(), .016666),
                q[env].numpy(), v[env].numpy(), pending[:, env].numpy(), horizons)
            expected, _ = oracle.evaluate(proposal[env].numpy())
            assert_allclose(pd_d[:, env], expected, atol=3e-14, rtol=3e-14)
        assert_allclose(cv_d, np.stack([v.numpy()*.016666*h for h in horizons]), atol=1e-15)

    def test_first_six_fixed_seventh_sensitive_and_no_input_mutation(self):
        q = torch.zeros((2, 26), dtype=torch.float64)
        pending = torch.arange(6., dtype=q.dtype)[:, None, None].expand(6, 2, 26).clone()
        cq, cv = coeff()
        saved = pending.clone()
        a = motion.motion_displacements(q, q, pending, torch.ones_like(q), cq, cv, .016666, tuple(range(1, 8)))[1]
        b = motion.motion_displacements(q, q, pending, torch.zeros_like(q), cq, cv, .016666, tuple(range(1, 8)))[1]
        assert_array_equal(a[:6], b[:6])
        assert_allclose(a[6]-b[6], cq[:, 1].expand(2, -1), atol=2e-15)
        assert_array_equal(pending, saved)
        assert_array_equal(q, torch.zeros_like(q))

    def test_reference_governed_proposal_uses_reachable_envelope(self):
        s, issued, pending, cmd, limits, full, rows, sph = scene()
        # Envelope unreachable: target must slew at the existing speed bound.
        for arm in ARM_KEYS:
            issued[arm][:] = .5
        box = 1.5*s.dt
        actual = motion.governed_proposal(s, issued, cmd, limits, box)
        for arm in ARM_KEYS:
            assert_allclose(actual[arm], .5-box, atol=1e-15)
        for arm in ARM_KEYS:
            issued[arm][:] = .015
        actual = motion.governed_proposal(s, issued, cmd, limits, box)
        for arm in ARM_KEYS:
            lo = np.maximum(-box, -.05-issued[arm].numpy())
            hi = np.minimum(box, .05-issued[arm].numpy())
            expected = issued[arm].numpy()+np.clip(cmd.delta_q[arm].numpy(), lo, hi)
            assert_allclose(actual[arm], expected, atol=1e-15)

    def test_nonfinite_state_queue_coefficients_and_overflow_rejected(self):
        q = torch.zeros((2, 26))
        cq, cv = coeff(torch.float32)
        arguments = [q, q, torch.zeros((6, 2, 26)), q, cq, cv]
        for i in range(len(arguments)):
            damaged = [x.clone() for x in arguments]
            damaged[i].flatten()[0] = float('nan')
            with self.subTest(index=i), self.assertRaises(ValueError):
                motion.motion_displacements(*damaged, .016666)
        huge = cq.clone(); huge[:, 0] = 1e38
        with self.assertRaises(ValueError):
            motion.motion_displacements(q, q+10, arguments[2], q, huge, cv, .016666)


class ModeAndAdmissionTests(unittest.TestCase):
    def test_each_mode_matches_explicit_min_and_retains_old_target_rows(self):
        s, issued, pending, cmd, limits, full, rows, sph = scene()
        old = target.full_forecast(full, rows, s, issued, pending, cmd, limits, 1.5*s.dt)
        old_mask = admission.union_mask(full, full, old)
        cq, cv = coeff()
        with patch.object(motion, 'model_coefficients', return_value=(cq, cv, s.dt)):
            for mode in motion.MODES:
                actual, data = motion.distance_forecasts(full, rows, s, issued, pending, cmd, limits, 1.5*s.dt, old, mode)
                expected = old.clone()
                if mode in ('velocity_admission', 'motion_admission'):
                    expected = torch.minimum(expected, data['cv_forecast'])
                if mode in ('pd_admission', 'motion_admission'):
                    expected = torch.minimum(expected, data['pd_forecast'])
                assert_array_equal(actual, expected)
                new_mask = admission.union_mask(full, full, actual)
                self.assertTrue(torch.all(~old_mask | new_mask))
                if mode == 'joint_reference':
                    self.assertIs(actual, old)

    def test_handcrafted_rows_distinguish_cv_pd_both_and_reference(self):
        s, issued, pending, cmd, limits, full, rows, sph = scene(1, 5)
        full.active_mask[:] = False
        old = full.dists.clone(); old[0, 0] = 0.
        J = torch.zeros((1, 5, 26), dtype=torch.float64)
        J[0, 1, 0] = 1; J[0, 2, 1] = 1
        rows.J = {'F': J[:, :, :14], 'U': J[:, :, 14:]}
        cv_d, pd_d = torch.zeros((4, 1, 26), dtype=torch.float64), torch.zeros((4, 1, 26), dtype=torch.float64)
        cv_d[0, 0, 0] = -.1; pd_d[0, 0, 1] = -.1
        cq, cv = coeff()
        with patch.object(motion, 'model_coefficients', return_value=(cq, cv, s.dt)), \
             patch.object(motion, 'motion_displacements', return_value=(cv_d, pd_d)):
            for mode, ids in zip(motion.MODES, ([0], [0, 1], [0, 2], [0, 1, 2])):
                result, _ = motion.distance_forecasts(full, rows, s, issued, pending, cmd, limits, 1.5*s.dt, old, mode)
                mask = admission.union_mask(full, full, result)
                assert_array_equal(torch.where(mask[0])[0], ids)

    def test_full_9021_rows_retained_and_capacity_overflow_aborts(self):
        s, issued, pending, cmd, limits, full, rows, sph = scene(2, 9021, torch.float32)
        mask = torch.ones_like(full.dists, dtype=torch.bool)
        full.full_viol_exempt[:, -1] = True
        output = admission.select(full, full, mask, sph, capacity=9021)
        assert_array_equal(output.active_idx, torch.arange(9021).expand(2, -1))
        assert_array_equal(output.active_pairs[:, :, 0], full.dists)
        assert_array_equal(output.viol_exempt, full.full_viol_exempt)
        with self.assertRaisesRegex(ValueError, 'no dropping'):
            admission.select(full, full, mask, sph, capacity=9020)

    def test_invalid_mode_and_wrong_model_dt_rejected(self):
        s, issued, pending, cmd, limits, full, rows, sph = scene()
        args = (full, rows, s, issued, pending, cmd, limits, 1.5*s.dt, full.dists)
        with self.assertRaises(ValueError):
            motion.distance_forecasts(*args, 'joint_repair')
        cq, cv = coeff()
        with patch.object(motion, 'model_coefficients', return_value=(cq, cv, .02)):
            with self.assertRaises(ValueError):
                motion.distance_forecasts(*args, 'motion_admission')


class RunnerSeamTests(unittest.TestCase):
    def test_real_safety_closure_preserves_geometry_actor_rows_queue_and_all_rows(self):
        trace, env, rows = start_runner('motion_admission', all_admitted=True)
        full = env._last_out
        original_pairs = full.active_pairs.clone()
        original_targets = {a: x.clone() for a, x in env._targets.items()}
        pending = [{a: x.clone() for a, x in p.items()} for p in env._evaluation_actuator_delay.queue.pending]
        cq, cv = coeff(torch.float32)
        with patch.object(motion, 'model_coefficients', return_value=(cq, cv, .016666)):
            result = env.safety_dist_out()
        self.assertIs(env._last_out, full)
        assert_array_equal(full.active_pairs, original_pairs)
        self.assertEqual(full.active_idx.shape, (64, 32))
        self.assertTrue(torch.all(result.active_mask.sum(-1) == 9021))
        self.assertEqual(trace.last_chunk['pd_q_horizons'].shape, (4, 64, 26))
        self.assertEqual(trace.last_chunk['cv_q_horizons'].shape, (4, 64, 26))
        self.assertEqual(trace.last_chunk['pd_h6'].shape, (64, 9021))
        self.assertEqual(trace.last_chunk['selected_ids'].shape, (64, 9021))
        for a in ARM_KEYS:
            assert_array_equal(env._targets[a], original_targets[a])
            for before, after in zip(pending, env._evaluation_actuator_delay.queue.pending):
                assert_array_equal(after[a], before[a])
        meta = json.loads((trace.out/'guard_metadata.json').read_text())
        self.assertEqual(meta['motion_horizons'], [1, 6, 12, 18])
        self.assertFalse(meta['joint_repair'])
        self.assertEqual(meta['reference_gap_rad'], .050)

    def test_unselected_last_row_nan_aborts_before_projection(self):
        for kind in ('J', 'distance', 'closing', 'dmin'):
            with self.subTest(kind=kind):
                trace, env, rows = start_runner('joint_reference')
                if kind == 'J':
                    rows.J['U'][0, -1, 0] = float('nan')
                else:
                    value = {'distance': env._last_out.dists, 'closing': env._last_out.closing,
                             'dmin': env._last_out.full_dmin}[kind]
                    value[0, -1] = float('nan')
                with self.assertRaises(ValueError):
                    env.safety_dist_out()
                abort = json.loads((trace.out/'guard_abort.json').read_text())
                self.assertEqual(abort['abort_stage'], 'before projection/physics')
                self.assertTrue(abort['partial_windows_not_safe'])

    def test_h6_diagnostic_uses_horizon_axis_one_not_step_six_index(self):
        s, issued, pending, cmd, limits, full, rows, sph = scene(64, 7)
        cq, cv = coeff()
        with patch.object(motion, 'model_coefficients', return_value=(cq, cv, s.dt)):
            _, data = motion.distance_forecasts(full, rows, s, issued, pending, cmd, limits, 1.5*s.dt, full.dists, 'motion_admission')
        q = torch.cat([s.q[a] for a in ARM_KEYS], -1)
        J = torch.cat([rows.J[r] for r in ('F', 'U')], -1)
        for name in ('cv', 'pd'):
            expected = full.dists + torch.einsum('nmd,nd->nm', J, data[name+'_q_horizons'][1]-q)
            assert_allclose(data[name+'_h6'], expected, atol=3e-15)

    def test_verbatim_chunk_writer_preserves_time_horizon_env_joint_axes(self):
        trace = runner_class()()
        trace.out, trace.chunk_start, trace.forecast_chunks = MemoryPath(), 60, []
        frames = []
        for t in range(2):
            q = np.arange(4*64*26).reshape(4, 64, 26) + 100000*t
            frames.append(dict(pd_q_horizons=q, cv_q_horizons=-q, forecast=np.zeros((64, 9021))))
        trace.chunk = frames
        real_save = np.savez_compressed
        def save_in_memory(path, **values):
            stream = io.BytesIO(); real_save(stream, **values)
            path.files[path.name] = stream.getvalue()
        with patch.object(np, 'savez_compressed', side_effect=save_in_memory):
            trace.flush_forecast()
        receipt = trace.forecast_chunks[0]
        with np.load(io.BytesIO((trace.out/receipt['path']).read_bytes()), allow_pickle=False) as saved:
            self.assertEqual(saved['pd_q_horizons'].shape, (2, 4, 64, 26))
            for time in range(2):
                for hi in range(4):
                    assert_array_equal(saved['pd_q_horizons'][time, hi, 38], frames[time]['pd_q_horizons'][hi, 38])
        self.assertEqual(trace.chunk_start, 62)
        self.assertEqual(trace.chunk, [])


class OriginalProjectionTests(unittest.TestCase):
    def test_copied_helpers_are_identical_to_sealed_prior(self):
        for new, old in (('target_forecast.py', 'full_finite_guard.py'),
                         ('reference_envelope.py', 'reference_envelope.py'),
                         ('projection_diagnostics.py', 'projection_diagnostics.py')):
            self.assertEqual((H/new).read_bytes(), (OLD/old).read_bytes())

    def test_checked_project_preserves_original_tuple_and_passive_install(self):
        q = torch.zeros((2, 26))
        command = DeltaCmd(arm_dict(q+.01))
        rows = ConstraintRows(torch.full((2, 1), .04),
            {'F': torch.ones((2, 1, 14))*.01, 'U': torch.ones((2, 1, 12))*.01},
            torch.zeros((2, 1)), torch.ones((2, 1, 4), dtype=torch.bool),
            torch.ones((2, 1), dtype=torch.bool), torch.full((2, 1), .02))
        alpha, priority = torch.ones((2, 4)), torch.zeros(2)
        bs = VelocityDamperBackstop(BackstopConfig())
        original = bs.project
        expected = original(command, rows, alpha, priority, .016666)
        self.assertIs(diagnostics.checked_project(lambda *a, **kw: expected, command, rows, alpha, priority, .016666), expected)
        env = SimpleNamespace(_backstop=bs)
        sink = {}
        diagnostics.install(env, sink, snapshot=True)
        actual = bs.project(command, rows, alpha, priority, .016666)
        for a in ARM_KEYS:
            assert_array_equal(actual[0].delta_q[a], expected[0].delta_q[a])
        assert_array_equal(actual[1], expected[1])
        for key in expected[2]:
            assert_array_equal(actual[2][key], expected[2][key])
        self.assertIn('snapshot_J_F', sink)


if __name__ == '__main__':
    unittest.main(verbosity=2)
