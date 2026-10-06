"""CPU tests: run PYTHONDONTWRITEBYTECODE=1 python3 astra_fifo_tests.py.

Tests use analytic solutions, the source evaluator queue, and an independent
direct recurrence. Coefficients are synthetic, not estimates from any block.
Floating point tolerances below test arithmetic only, never physical safety.
"""
import ast
from collections import deque
import hashlib
import io
import json
from pathlib import Path
import sys
import types
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from astra_fifo_oracle import (DiagonalDynamics, affine_horizons,
                               displacement_halfspaces, fifo_targets, rollout)

SOURCE = Path('/home/liyufeng/safeduo/src/safeduo/eval/perturbations.py')
SOURCE_SHA = '5c7ade79753a557474a6e4778ecf90b4ca9f80df9b8b6d123247e1ad513bd59e'


class TensorStandIn(np.ndarray):
    """Only detach/clone are needed by the literal evaluator FIFO class."""
    def detach(self):
        return self

    def clone(self):
        return self.copy()


def source_queue():
    source = SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA:
        raise AssertionError('source evaluator changed since block0 registration')
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.ClassDef) and n.name == 'TargetDelayQueue')
    namespace = {'deque': deque}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['TargetDelayQueue']


class FifoTests(unittest.TestCase):
    def test_append_pop_order_and_first_new_arrival(self):
        pending = np.arange(6.)[:, None]
        issued = (100 + np.arange(9.))[:, None]
        applied, tail = fifo_targets(pending, issued)
        assert_array_equal(applied[:, 0], [0, 1, 2, 3, 4, 5, 100, 101, 102])
        assert_array_equal(tail[:, 0], [103, 104, 105, 106, 107, 108])

    def test_actual_source_evaluator_delay_zero_one_six(self):
        Queue = source_queue()
        rng = np.random.default_rng(623)
        for delay in (0, 1, 6):
            initial, issued = rng.normal(size=26), rng.normal(size=(24, 26))
            queue = Queue(delay)
            queue.reset({'arm': initial.view(TensorStandIn)})
            expected = np.array([queue.push({'arm': t.view(TensorStandIn)})['arm']
                                 for t in issued])
            applied, tail = fifo_targets(np.tile(initial, (delay, 1)), issued)
            assert_array_equal(applied, expected)
            if delay:
                assert_array_equal(tail, np.stack([t['arm'] for t in queue.pending]))
            else:
                self.assertEqual(tail.shape, (0, 26))

    def test_short_empty_and_no_aliasing(self):
        pending = np.arange(12.).reshape(6, 2)
        issued = np.array([[91., 92.]])
        before = pending.copy()
        applied, tail = fifo_targets(pending, issued)
        assert_array_equal(applied, before[:1])
        assert_array_equal(tail, np.concatenate((before[1:], issued)))
        applied[:] = 99; tail[:] = 99
        assert_array_equal(pending, before)
        assert_array_equal(issued, [[91, 92]])
        applied, tail = fifo_targets(pending, np.empty((0, 2)))
        self.assertEqual(applied.shape, (0, 2))
        assert_array_equal(tail, pending)


class AffineTests(unittest.TestCase):
    def setUp(self):
        self.model = DiagonalDynamics([[0., .5, 0.]], [[0., 0., 0.]], .016666)
        self.q = np.zeros(1)
        self.pending = np.zeros((6, 1))

    def test_closed_form_hold_and_exact_fixed_prefix(self):
        horizons = (1, 6, 7, 12, 18)
        a = affine_horizons(self.model, self.q, self.q, self.pending, horizons)
        assert_array_equal(a.displacement_base, np.zeros((5, 1)))
        assert_array_equal(a.displacement_sensitivity[:, 0],
                           [0, 0, .5, 1 - 2.**-6, 1 - 2.**-12])
        assert_array_equal(a.velocity_sensitivity, np.zeros((5, 1)))

    def test_perturb_current_and_future_issues_cannot_change_first_six_states(self):
        rng = np.random.default_rng(715)
        model = DiagonalDynamics(np.tile([.01, .2, .001], (26, 1)),
                                 np.tile([.7, 1.5, -.002], (26, 1)), .016666)
        q, v, pending = rng.normal(size=26), rng.normal(size=26), rng.normal(size=(6, 26))
        aa, _ = fifo_targets(pending, rng.normal(size=(18, 26)))
        bb, _ = fifo_targets(pending, rng.normal(size=(18, 26)))
        qa, va = rollout(model, q, v, aa)
        qb, vb = rollout(model, q, v, bb)
        assert_array_equal(qa[:6], qb[:6]); assert_array_equal(va[:6], vb[:6])
        self.assertTrue(np.any(qa[6] != qb[6]))

    def test_velocity_can_respond_at_seven_position_only_at_eight(self):
        model = DiagonalDynamics([[.01, 0., 0.]], [[.8, 2., 0.]], .016666)
        a = affine_horizons(model, self.q, self.q, self.pending, (6, 7, 8))
        assert_array_equal(a.displacement_sensitivity[:, 0], [0, 0, .02])
        assert_array_equal(a.velocity_sensitivity[:2, 0], [0, 2])

    def test_target_independent_model_never_responds(self):
        model = DiagonalDynamics([[.01, 0., .001]], [[.8, 0., .002]], .016666)
        a = affine_horizons(model, self.q, np.ones(1), self.pending)
        assert_array_equal(a.displacement_sensitivity, np.zeros((4, 1)))
        assert_array_equal(a.velocity_sensitivity, np.zeros((4, 1)))

    def test_single_issue_versus_hold_future_assumption(self):
        offsets, gains = np.zeros((12, 1)), np.zeros((12, 1))
        gains[0] = 1
        pulse = affine_horizons(self.model, self.q, self.q, self.pending,
                                (7, 12, 18), future_target_offsets=offsets,
                                future_target_gains=gains)
        assert_array_equal(pulse.displacement_sensitivity[:, 0], [.5, 2.**-6, 2.**-12])
        hold = affine_horizons(self.model, self.q, self.q, self.pending, (7, 12, 18))
        assert_array_equal(pulse.displacement_sensitivity[0], hold.displacement_sensitivity[0])
        self.assertTrue(np.all(pulse.displacement_sensitivity[1:] < hold.displacement_sensitivity[1:]))

    def test_affine_equals_independent_direct_recurrence_across_26_joints(self):
        rng = np.random.default_rng(906)
        for delay in (0, 1, 6, 20):
            cq = rng.uniform([0, -.1, -.01], [.02, .5, .01], size=(26, 3))
            cv = rng.uniform([.3, -2, -.02], [.9, 2, .02], size=(26, 3))
            model = DiagonalDynamics(cq, cv, .016666)
            q, v, pending = rng.normal(size=26), rng.normal(size=26), rng.normal(size=(delay, 26))
            count = max(0, 18-delay)
            offsets, gains = rng.normal(size=(count, 26)), rng.normal(size=(count, 26))
            if count:
                offsets[0], gains[0] = 0, 1
            horizons = (18, 1, 6, 12, 6)
            affine = affine_horizons(model, q, v, pending, horizons,
                                     future_target_offsets=offsets, future_target_gains=gains)
            for target in (np.zeros(26), rng.normal(size=26), rng.normal(size=26)):
                applied = np.concatenate((pending, offsets + gains*target))[:18]
                qs, vs = rollout(model, q, v, applied)
                dq, vel = affine.evaluate(target)
                assert_allclose(dq, qs[np.array(horizons)-1]-q, atol=3e-14, rtol=3e-14)
                assert_allclose(vel, vs[np.array(horizons)-1], atol=3e-14, rtol=3e-14)

    def test_known_pending_prefix_sets_base(self):
        a = affine_horizons(self.model, self.q, self.q, np.arange(1., 7.)[:, None], (1, 6, 7))
        # Scalar geometric convolution: q6=sum_{i=1}^6 i/2^(7-i).
        q6 = sum(i / 2.**(7-i) for i in range(1, 7))
        assert_array_equal(a.displacement_base[:, 0], [.5, q6, q6/2])

    def test_bias_and_centered_means_are_not_dropped(self):
        cq, cv = np.array([[.02, .3, .4]]), np.array([[.7, 1.2, -.2]])
        means, qm, vm = np.array([[2., 3., 1.]]), np.array([.6]), np.array([-.8])
        model = DiagonalDynamics.from_centered(cq, cv, .016666, means, qm, vm)
        q, v, u = np.array([.5]), np.array([2.5]), np.array([[1.]])
        features = np.array([[2.5, .5, 1.]])
        qs, vs = rollout(model, q, v, u)
        assert_allclose(qs[0]-q, np.sum(cq*(features-means), axis=1)+qm, atol=1e-15)
        assert_allclose(vs[0], np.sum(cv*(features-means), axis=1)+vm, atol=1e-15)

    def test_no_implicit_extra_dt_or_coefficient_mutation(self):
        cq = np.array([[.02, .3, .4]])
        model = DiagonalDynamics(cq, [[.7, 1.2, -.2]], .016666)
        cq[:] = 999
        qs, vs = rollout(model, [1.], [2.], [[4.]])
        assert_allclose(qs[0], [1 + .02*2 + .3*3 + .4])
        assert_allclose(vs[0], [.7*2 + 1.2*3 - .2])
        self.assertFalse(model.dq_coeff.flags.writeable)


class ConstraintTests(unittest.TestCase):
    def test_full_coupled_geometry_residual_equals_distance_shortfall(self):
        rng = np.random.default_rng(122)
        J, d, floor = rng.normal(size=(7, 26)), rng.normal(size=7), rng.normal(size=7)
        base, sensitivity, origin, u = rng.normal(size=(4, 26))
        G, h = displacement_halfspaces(J, d, floor, base, sensitivity, origin)
        predicted = d + J @ (base + sensitivity*(origin+u))
        assert_allclose(G @ u - h, floor - predicted, atol=2e-14, rtol=2e-14)

    def test_nonzero_target_origin_and_negative_sensitivity(self):
        G, h = displacement_halfspaces([[2., -3.]], [.1], [.02], [.3, -.2], [-.5, .25], [1., -2.])
        assert_array_equal(G, [[1., .75]])
        assert_allclose(h, [1.78], atol=1e-15)

    def test_fixed_prefix_negative_margin_is_uncontrollable_without_epsilon(self):
        G, h = displacement_halfspaces([[1.]], [0.], [0.], [-1e-15], [0.], [3.])
        assert_array_equal(G, [[0.]])
        self.assertLess(h[0], 0.)
        self.assertGreater((G @ np.array([-1e12])-h)[0], 0.)

    def test_units_rad_to_degrees_preserve_meter_residual(self):
        k = 180/np.pi
        J, d, floor = np.array([[.1, -.3]]), [.05], [.02]
        base, sens, origin, u = (np.array(v) for v in ([.2, -.1], [.4, -.7], [1., 2.], [.01, -.02]))
        G, h = displacement_halfspaces(J, d, floor, base, sens, origin)
        Gdeg, hdeg = displacement_halfspaces(J/k, d, floor, base*k, sens, origin*k)
        assert_allclose(G@u-h, Gdeg@(u*k)-hdeg, atol=1e-15)


class InvalidInputTests(unittest.TestCase):
    def test_nonfinite_shape_horizon_and_schedule_rejected(self):
        model = DiagonalDynamics([[.01, .2, 0.]], [[.8, 2., 0.]], .016666)
        invalid = [
            lambda: DiagonalDynamics([[0., np.nan, 0.]], [[0., 0., 0.]], .01),
            lambda: DiagonalDynamics([[0., 1.]], [[0., 1.]], .01),
            lambda: DiagonalDynamics([[0., 1., 0.]], [[0., 1., 0.]], 0),
            lambda: fifo_targets([[0.]], [[np.inf]]),
            lambda: fifo_targets([], [[1.]]),
            lambda: rollout(model, [0.], [0.], [[1., 2.]]),
            lambda: affine_horizons(model, [0.], [0.], np.zeros((6, 1)), (0,)),
            lambda: affine_horizons(model, [0.], [0.], np.zeros((6, 1)), (1.5,)),
            lambda: affine_horizons(model, [0.], [0.], np.zeros((6, 1)), (True,)),
            lambda: affine_horizons(model, [0.], [0.], np.zeros((6, 1)), (),),
            lambda: affine_horizons(model, [0.], [0.], np.zeros((6, 1)), (7,), future_target_gains=[[1.]]),
            lambda: affine_horizons(model, [0.], [0.], np.zeros((6, 1)), (7,), future_target_gains=[[0.]], future_target_offsets=[[0.]]),
            lambda: displacement_halfspaces([[1.]], [0.], [0.], [0.], [np.nan], [0.]),
        ]
        for call in invalid:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()

    def test_numerical_overflow_fails_instead_of_emitting_infinity(self):
        model = DiagonalDynamics([[1e308, 1e308, 1e308]], [[0., 0., 0.]], .01)
        with self.assertRaises(FloatingPointError):
            rollout(model, [0.], [2.], [[2.]])
        with self.assertRaises(FloatingPointError):
            affine_horizons(model, [0.], [2.], [[2.]], (1,))

    def test_einsum_silent_overflow_is_explicitly_rejected(self):
        # np.einsum can emit inf without raising even under errstate(over=raise).
        model = DiagonalDynamics([[0., 0., 0.]], [[1e308, 0., 0.]], .016666)
        with self.assertRaises(FloatingPointError):
            affine_horizons(model, [0.], [2.], [[0.]], (1,))


def mutation_checks():
    """In-memory red-state checks; never rewrite oracle or source evaluator."""
    source = Path(__file__).with_name('astra_fifo_oracle.py').read_text()
    changes = {
        'drop_oldest_pending_off_by_one':
            ('steps, delay = max(horizons), len(pending)',
             'pending = pending[1:]\n    steps, delay = max(horizons), len(pending)'),
        'wrong_position_target_sign':
            ('A[:, 0, 0], A[:, 0, 1] = 1 - b, a', 'A[:, 0, 0], A[:, 0, 1] = 1 + b, a'),
        'omit_affine_bias': (' + B * offset[:, None] + bias', ' + B * offset[:, None]'),
        'omit_target_origin': ('J @ (base + sensitivity * origin)', 'J @ base'),
        'invert_geometry_sign': ('return -J * sensitivity[None, :]', 'return J * sensitivity[None, :]'),
        'fifo_lifo_error': ('stream[:len(issued)].copy()', 'stream[-len(issued):].copy()'),
    }
    names = ('DiagonalDynamics', 'affine_horizons', 'displacement_halfspaces', 'fifo_targets', 'rollout')
    original = {name: globals()[name] for name in names}
    results = {}
    try:
        for label, (old, new) in changes.items():
            assert old in source, label
            module = types.ModuleType('astra_fifo_mutant')
            sys.modules[module.__name__] = module
            exec(compile(source.replace(old, new, 1), label, 'exec'), module.__dict__)
            globals().update({name: getattr(module, name) for name in names})
            suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
            result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
            results[label] = dict(killed=not result.wasSuccessful(), tests=result.testsRun,
                                 failures=[str(t) for t, _ in result.failures],
                                 errors=[str(t) for t, _ in result.errors])
    finally:
        globals().update(original)
        sys.modules.pop('astra_fifo_mutant', None)
    assert all(r['killed'] for r in results.values())
    return results


if __name__ == '__main__':
    if sys.argv[1:] == ['--mutations']:
        print(json.dumps(mutation_checks(), indent=2))
    else:
        unittest.main(verbosity=2)
