"""Finite CPU fixtures for metadata amendment and termination, no camera outcomes."""
import ast
import copy
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import astra_zero_camera_amendment as A

H = A.H
ORIGINAL = json.loads((H / 'visual_plan.json').read_text())
AMENDED = json.loads((H / 'visual_recovery_plan.json').read_text())
SELECTION = json.loads((H / 'astra_zero_camera_selection.json').read_text())
TERMINATION = json.loads((H / 'astra_zero_termination_source_snapshot_01.json').read_text())


def execution():
    return dict(status='complete', jobs=[dict(mode=j['mode'], status='complete', exit_code=0,
        argv=copy.deepcopy(j['argv']), out=str(Path(AMENDED['actual_visual_root']) / j['mode']))
        for j in AMENDED['jobs']])


def binding():
    changes = A.validate_delta(ORIGINAL, AMENDED)
    return dict(status='PASS_ACTUAL_CAMERA_DEVICE_AMENDMENT_BINDING',
        original_plan_sha256=A.PINS['visual_plan.json'],
        executed_recovery_plan_sha256=A.PINS['visual_recovery_plan.json'],
        registered_recovery_plan_sha256=A.PINS['visual_recovery_plan.json'],
        amendment_sha256=A.PINS['PLANS_RECOVERY_AMENDMENT.json'], actual_camera_execution_sha256='actual-execution',
        device_only_deltas=[dict(mode=c['mode'], argument_index=c['argv_index'], before=c['before'], after=c['after']) for c in changes],
        original_receipt_not_modified=True, original_plan_not_relabelled_as_executed=True,
        original_selection_and_math_unchanged=True)


def extract(text, name, namespace):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<captured pure fixture>', 'exec'), namespace)
    return namespace[name]


class AmendmentContract(unittest.TestCase):
    def test_current_registered_input_bytes_and_only_two_device_tokens(self):
        values, changes = A.bound_documents()
        self.assertEqual(len(changes), 2)
        self.assertEqual(values['astra_zero_camera_selection.json'], SELECTION)

    def test_only_one_device_changed_or_none_rejects(self):
        for count in [0, 1]:
            value = copy.deepcopy(ORIGINAL)
            for job in value['jobs'][:count]:
                job['argv'][job['argv'].index('--device') + 1] = 'cuda:1'
            with self.subTest(count=count), self.assertRaises(ValueError):
                A.validate_delta(ORIGINAL, value)

    def test_argv_seed_duration_device_extra_and_root_mutants_reject(self):
        for key, replacement in [('--seeds', '999'), ('--duration-s', '12'),
                                 ('--device', 'cuda:0'), ('--out', '/different'),
                                 ('--ckpt', '/different-actor'), ('--env-yaml', 'different.yaml')]:
            value = copy.deepcopy(AMENDED)
            value['jobs'][0]['argv'][value['jobs'][0]['argv'].index(key) + 1] = replacement
            with self.subTest(argument=key), self.assertRaises(ValueError):
                A.validate_delta(ORIGINAL, value)
        value = copy.deepcopy(AMENDED)
        value['jobs'][0]['argv'].append('--different')
        with self.assertRaises(ValueError):
            A.validate_delta(ORIGINAL, value)

    def test_env_bank_source_views_and_actor_mutants_reject(self):
        for name in ['CUDA_VISIBLE_DEVICES', 'SAFEDUO_INITIAL_BANK_NPZ', 'SAFEDUO_JOINT_MODE']:
            value = copy.deepcopy(AMENDED)
            value['jobs'][0]['env'][name] = 'changed'
            with self.subTest(environment=name), self.assertRaises(ValueError):
                A.validate_delta(ORIGINAL, value)
        for field in ['sources', 'original_source_sha256', 'views_per_group', 'scheduled_groups', 'actor_sha256']:
            value = copy.deepcopy(AMENDED)
            value[field] = {} if isinstance(value[field], dict) else 'changed'
            with self.subTest(field=field), self.assertRaises(ValueError):
                A.validate_delta(ORIGINAL, value)

    def test_missing_duplicate_or_reordered_plan_jobs_reject(self):
        for jobs in [AMENDED['jobs'][:1], [AMENDED['jobs'][0]] * 2, list(reversed(AMENDED['jobs']))]:
            value = copy.deepcopy(AMENDED)
            value['jobs'] = jobs
            with self.subTest(jobs=jobs), self.assertRaises(ValueError):
                A.validate_delta(ORIGINAL, value)

    def test_each_plan_amendment_and_selection_byte_anchor_rejects_mutation(self):
        real = Path.read_bytes
        for name in A.PINS:
            def altered(path, _name=name):
                data = real(path)
                return data + b' ' if path == H / _name else data
            with self.subTest(name=name), patch.object(Path, 'read_bytes', altered), self.assertRaisesRegex(ValueError, 'registered input changed'):
                A.bound_documents()

    def test_real_cuda1_receipt_is_accepted_without_fabricating_legacy_field(self):
        value = execution()
        original = copy.deepcopy(value)
        A.validate_execution(value, AMENDED)
        self.assertEqual(value, original)
        self.assertNotIn('plan_sha256', value)

    def test_incomplete_unknown_nonzero_and_boolean_waits_reject(self):
        for code in ['UNRECORDED', None, 1, False]:
            value = execution()
            value['jobs'][0]['exit_code'] = code
            with self.subTest(code=code), self.assertRaises(ValueError):
                A.validate_execution(value, AMENDED)
        value = execution()
        del value['jobs'][0]['exit_code']
        with self.assertRaises(ValueError):
            A.validate_execution(value, AMENDED)
        for status in ['running', 'failed', 'FAIL', 'complete_with_failures']:
            value = execution()
            value['status'] = status
            with self.subTest(status=status), self.assertRaises(ValueError):
                A.validate_execution(value, AMENDED)

    def test_actual_argv_or_false_original_plan_execution_rejects(self):
        value = execution()
        value['jobs'][0]['argv'] = ORIGINAL['jobs'][0]['argv']
        with self.assertRaises(ValueError):
            A.validate_execution(value, AMENDED)
        value = execution()
        value['plan_sha256'] = A.PINS['visual_plan.json']
        with self.assertRaises(ValueError):
            A.validate_execution(value, AMENDED)

    def test_duplicate_actual_job_and_wrong_output_rejects(self):
        value = execution()
        value['jobs'][1] = copy.deepcopy(value['jobs'][0])
        with self.assertRaises(ValueError):
            A.validate_execution(value, AMENDED)
        value = execution()
        value['jobs'][1]['out'] += '/wrong'
        with self.assertRaises(ValueError):
            A.validate_execution(value, AMENDED)

    def test_standalone_binding_checks_all_hashes_and_delta_without_mutating(self):
        value = binding()
        original = copy.deepcopy(value)
        changes = A.validate_delta(ORIGINAL, AMENDED)
        A.validate_binding(value, 'actual-execution', changes)
        self.assertEqual(value, original)
        for field in ['original_plan_sha256', 'executed_recovery_plan_sha256', 'amendment_sha256',
                      'actual_camera_execution_sha256', 'registered_recovery_plan_sha256', 'device_only_deltas']:
            bad = copy.deepcopy(value)
            bad[field] = 'wrong'
            with self.subTest(field=field), self.assertRaises(ValueError):
                A.validate_binding(bad, 'actual-execution', changes)

    def test_compiled_invocation_reuses_exact_original_body_and_72_queue(self):
        code = (H / 'astra_zero_camera_prepare.py').read_text()
        node, header_sha, body_sha = A.invocation_body(code)
        self.assertEqual(body_sha, A.digest(ast.dump(ast.Module(body=node.body[3:], type_ignores=[]), include_attributes=False).encode()))
        views = SELECTION['views']
        ledger = types.SimpleNamespace(hashes={str(H / 'astra_zero_camera_selection.json'): A.PINS['astra_zero_camera_selection.json'],
            str(H / 'visual_plan.json'): A.PINS['visual_plan.json']}, recheck=lambda: [])

        def audit(ledger, plan, selection, record, mode):
            groups = []
            for index, spec in enumerate(SELECTION['scheduled_groups']):
                identifier = mode + ':' + str(index)
                groups.append(dict(group_id=identifier, mode=mode, env=spec['env'], step=spec['step'],
                    kind='scheduled', selected_for_actual_view_image=True, state_path=identifier + '.json',
                    native_before_path=identifier + '.before', native_after_path=identifier + '.after',
                    views={v: dict(path=identifier + '/' + v, sha256=identifier + v) for v in views}))
            return dict(groups=groups, actual_group_count=4, actual_image_count=36)

        scope = dict(H=H, require=A.require, VIEWS=views, all_source_bindings=lambda *a: None,
            audit_mode=audit, now=lambda: 'synthetic-time', json=json, print=lambda *a, **k: None)
        baseline_output = {}
        original_execution = execution()
        original_execution['plan_sha256'] = A.PINS['visual_plan.json']
        ledger.json = lambda p: original_execution
        baseline = extract(code, 'run', dict(scope, write=lambda n, v: baseline_output.update({n: v})))
        baseline(ledger, ORIGINAL, SELECTION)
        actual_output = {}
        frozen = types.SimpleNamespace(**scope)
        invoke = A.compile_invocation(frozen, dict(original_run_header_ast_sha256=header_sha,
            unchanged_run_body_ast_sha256=body_sha), lambda n, v: actual_output.update({n: v}))
        self.assertIs(invoke.__globals__['audit_mode'], audit)
        value = execution()
        invoke(ledger, AMENDED, SELECTION, value)
        self.assertEqual(actual_output, baseline_output)
        self.assertEqual(actual_output['astra_zero_camera_image_selection.json']['image_count'], 72)
        self.assertNotIn('plan_sha256', value)
        with self.assertRaises(ValueError):
            A.compile_invocation(frozen, dict(original_run_header_ast_sha256=header_sha,
                unchanged_run_body_ast_sha256='changed'), lambda *a: None)


class FiniteTerminationFixtures(unittest.TestCase):
    def test_failed_record_preserves_known_wait_and_marks_missing_unrecorded(self):
        fn = extract(TERMINATION['sources']['finalize_recovery_termination.py']['text'], 'failed_record', dict(copy=copy))
        original = dict(status='running', jobs=[dict(status='complete', exit_code=0),
            dict(status='running', exit_code=7), dict(status='running')])
        before = copy.deepcopy(original)
        result = fn(original, 1)
        self.assertEqual(original, before)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['jobs'][0], original['jobs'][0])
        self.assertEqual(result['jobs'][1]['exit_code'], 7)
        self.assertEqual(result['jobs'][1]['observed_child_wait_code'], 7)
        self.assertEqual(result['jobs'][2]['observed_child_wait_code'], 'UNRECORDED')
        self.assertNotIn('exit_code', result['jobs'][2])

    def test_parent_camera_binding_accepts_only_declared_delta_and_known_wait(self):
        hash_values = {str(H / n): s for n, s in A.PINS.items()}
        hash_values[str(H / 'visual_execution.json')] = 'actual-execution'
        fn = extract(TERMINATION['sources']['finalize_recovery_termination.py']['text'], 'camera_binding',
            dict(copy=copy, H=H, sha=lambda p: hash_values[str(p)]))
        amendment = dict(visual_recovery_plan_sha256=A.PINS['visual_recovery_plan.json'])
        value = execution()
        result = fn(ORIGINAL, AMENDED, value, amendment)
        A.validate_binding(result, 'actual-execution', A.validate_delta(ORIGINAL, AMENDED))
        self.assertNotIn('plan_sha256', value)
        bad = copy.deepcopy(AMENDED)
        bad['jobs'][0]['env']['OMP_NUM_THREADS'] = '2'
        with self.assertRaises(AssertionError):
            fn(ORIGINAL, bad, execution(), amendment)
        value['jobs'][0].pop('exit_code')
        with self.assertRaises(KeyError):
            fn(ORIGINAL, AMENDED, value, amendment)

    def test_original_wait_closed_raises_immediately_on_terminal_failed(self):
        class FailedPath:
            def __truediv__(self, name):
                return self
            def exists(self):
                return True
            def read_text(self):
                return json.dumps(dict(status='failed'))
        fn = extract(TERMINATION['sources']['follow_analysis.py']['text'], 'wait_closed',
            dict(H=FailedPath(), json=json, time=types.SimpleNamespace(sleep=lambda n:
                (_ for _ in ()).throw(AssertionError('must not sleep after terminal failure')))))
        with self.assertRaisesRegex(RuntimeError, 'failed; no completion fabricated'):
            fn('visual_execution.json', 'complete')


if __name__ == '__main__':
    unittest.main(verbosity=2)
