"""Captured original-CUDA0 driver with in-memory IO and synthetic child waits."""
import ast
import copy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import types
import unittest

H = Path(__file__).resolve().parent
SNAPSHOT = json.loads((H / 'astra_zero_original_camera_source_snapshot_01.json').read_text())
SOURCE = SNAPSHOT['sources']['recover_original_camera.py']['text']
PLAN = json.loads(SNAPSHOT['sources']['visual_plan.json']['text'])
REG = json.loads(SNAPSHOT['sources']['CAMERA_ORIGINAL_RECOVERY_REGISTRATION.json']['text'])


class FakePath:
    fixture = None

    def __init__(self, value):
        self.value = str(value)

    def __str__(self):
        return self.value

    def __truediv__(self, part):
        return FakePath(self.value + '/' + str(part))

    def exists(self):
        return self.fixture.scenario == 'existing_root'

    def is_file(self):
        if self.fixture.scenario == 'missing_npz' and self.value.endswith('/cell_001.npz'):
            return False
        if self.fixture.scenario == 'missing_capture_receipt' and self.value.endswith('/camera_receipts.json'):
            return False
        return True

    def open(self, *args, **kwargs):
        return io.StringIO()

    def read_text(self):
        if self.value.startswith('/proc/'):
            return ' '.join(['0'] * 21 + ['synthetic-start-tick'])
        raise AssertionError('unexpected real-file read')


class DriverFixture:
    def __init__(self, scenario):
        self.scenario = scenario
        self.events = []
        self.writes = []
        self.launches = []
        self.waited = False
        self.first_source = next(iter(PLAN['sources']))
        FakePath.fixture = self

    def read(self, path):
        path = str(path)
        self.events.append(('read', path))
        if path.endswith('/CAMERA_ORIGINAL_RECOVERY_REGISTRATION.json'):
            return copy.deepcopy(REG)
        if path.endswith('/visual_plan.json'):
            return copy.deepcopy(PLAN)
        if path.endswith('/visual_protocol.json'):
            if self.scenario == 'missing_protocol':
                raise FileNotFoundError('synthetic missing protocol')
            return dict(status='planned' if self.scenario == 'planned_protocol' else 'complete',
                        steps=959 if self.scenario == 'wrong_steps' else 960,
                        completed_windows=63 if self.scenario == 'wrong_count' else 64,
                        actor_sha256='changed' if self.scenario == 'protocol_actor' else PLAN['actor_sha256'])
        if path.endswith('/camera_receipts.json'):
            return dict(scheduled_groups=20 if self.scenario == 'wrong_scheduled' else 21,
                        groups=21, PNG=188 if self.scenario == 'wrong_image_count' else 189)
        raise AssertionError('unmodelled read: ' + path)

    def sha(self, path):
        path = str(path)
        if path == str(H / 'recover_original_camera.py'):
            return REG['driver_sha256']
        if path == str(H / 'visual_plan.json'):
            return REG['original_plan_sha256']
        if path == str(H / 'CAMERA_ORIGINAL_RECOVERY_REGISTRATION.json'):
            return SNAPSHOT['sources']['CAMERA_ORIGINAL_RECOVERY_REGISTRATION.json']['sha256']
        if path in PLAN['sources']:
            if path == self.first_source and (self.scenario == 'source_before' or
                    self.scenario == 'source_after' and self.waited):
                return 'changed-source'
            return PLAN['sources'][path]
        for relative, digest in PLAN['original_source_sha256'].items():
            if path == str(Path(PLAN['cwd']) / relative):
                return digest
        if path == PLAN['jobs'][0]['argv'][PLAN['jobs'][0]['argv'].index('--ckpt') + 1]:
            if self.scenario == 'actor_before' or self.scenario == 'actor_after' and self.waited:
                return 'changed-actor'
            return PLAN['actor_sha256']
        if path.endswith('/visual_protocol.json'):
            return 'synthetic-protocol-hash'
        raise AssertionError('unmodelled hash: ' + path)

    def write(self, path, value):
        self.writes.append((str(path), copy.deepcopy(value)))
        self.events.append(('write', str(path), copy.deepcopy(value)))

    def popen(self, argv, **kwargs):
        self.launches.append(dict(argv=copy.deepcopy(argv), kwargs=kwargs))
        self.events.append(('launch', argv))

        def wait():
            self.waited = True
            code = 7 if self.scenario == 'nonzero' else (-9 if self.scenario == 'signal_exit' else 0)
            self.events.append(('actual_synthetic_wait_return', code))
            return code

        return types.SimpleNamespace(pid=999000 + len(self.launches), wait=wait)

    def run(self):
        tree = ast.parse(SOURCE)
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ['verify_sources', 'actual_capacity', 'main']]
        namespace = dict(H=FakePath(H), Path=FakePath, __file__=str(H / 'recover_original_camera.py'),
            read=self.read, sha=self.sha, write=self.write, datetime=datetime, timezone=timezone,
            print=lambda *a, **k: None, os=types.SimpleNamespace(environ={}),
            subprocess=types.SimpleNamespace(STDOUT=-2, Popen=self.popen,
                run=lambda *a, **k: types.SimpleNamespace(stdout='0, %d\n1, 30000\n' %
                    (25599 if self.scenario == 'capacity' else 25600))))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<captured original camera driver>', 'exec'), namespace)
        namespace['main']()

    def final(self):
        return next(value for path, value in reversed(self.writes) if path.endswith('/visual_execution.json'))


class OriginalCameraContracts(unittest.TestCase):
    def test_success_uses_exact_original_plan_argv_env_and_two_per_job_gates(self):
        f = DriverFixture('success')
        f.run()
        out = f.final()
        self.assertEqual(out['status'], 'complete')
        self.assertEqual(out['plan_sha256'], REG['original_plan_sha256'])
        self.assertEqual(len(f.launches), 2)
        self.assertEqual(len([p for p, v in f.writes if p.endswith('_capacity.json')]), 2)
        for launch, job, row in zip(f.launches, PLAN['jobs'], out['jobs']):
            self.assertEqual(launch['argv'], job['argv'])
            self.assertEqual(launch['kwargs']['env'], job['env'])
            self.assertEqual(row['exit_code'], 0)
            self.assertTrue(row['child_reaped'])
            self.assertEqual(row['status'], 'complete')

    def test_wait_is_persisted_before_every_product_read(self):
        f = DriverFixture('success')
        f.run()
        last_wait = None
        persisted = False
        for event in f.events:
            if event[0] == 'actual_synthetic_wait_return':
                last_wait = event[1]
                persisted = False
            if event[0] == 'write' and event[1].endswith('/visual_execution.json'):
                row = event[2]['jobs'][-1] if event[2]['jobs'] else {}
                if row.get('stage') == 'closed_product_validation':
                    self.assertEqual(row['exit_code'], last_wait)
                    persisted = row['child_reaped'] is True
            if event[0] == 'read' and event[1].endswith('/visual_protocol.json'):
                self.assertTrue(persisted)

    def test_nonzero_and_known_signal_wait_persist_and_skip_product_validation(self):
        for scenario, expected in [('nonzero', 7), ('signal_exit', -9)]:
            f = DriverFixture(scenario)
            with self.subTest(scenario=scenario), self.assertRaises(AssertionError):
                f.run()
            out = f.final()
            self.assertEqual(out['status'], 'failed')
            self.assertEqual(out['jobs'][0]['exit_code'], expected)
            self.assertTrue(out['jobs'][0]['child_reaped'])
            self.assertEqual(out['jobs'][0]['failed_validation_stage'], 'closed_product_validation')
            self.assertFalse(any(e[0] == 'read' and e[1].endswith('/visual_protocol.json') for e in f.events))

    def test_child0_planned_missing_and_wrong_protocol_fail_with_known_zero_retained(self):
        for scenario in ['planned_protocol', 'missing_protocol', 'wrong_steps', 'wrong_count']:
            f = DriverFixture(scenario)
            with self.subTest(scenario=scenario), self.assertRaises((AssertionError, FileNotFoundError)):
                f.run()
            out = f.final()
            self.assertEqual(out['status'], 'failed')
            self.assertEqual(out['jobs'][0]['exit_code'], 0)
            self.assertTrue(out['jobs'][0]['child_reaped'])
            self.assertIn('error', out)
            self.assertEqual(len(f.launches), 1)

    def test_missing_npz_capture_receipt_or_wrong_capture_counts_fail_with_wait(self):
        for scenario in ['missing_npz', 'missing_capture_receipt', 'wrong_scheduled', 'wrong_image_count']:
            f = DriverFixture(scenario)
            with self.subTest(scenario=scenario), self.assertRaises(AssertionError):
                f.run()
            out = f.final()
            self.assertEqual(out['status'], 'failed')
            self.assertEqual(out['jobs'][0]['exit_code'], 0)
            self.assertTrue(out['jobs'][0]['child_reaped'])

    def test_actor_and_source_change_after_child_wait_fail_without_erasing_zero(self):
        for scenario in ['actor_after', 'protocol_actor', 'source_after']:
            f = DriverFixture(scenario)
            with self.subTest(scenario=scenario), self.assertRaises(AssertionError):
                f.run()
            out = f.final()
            self.assertEqual(out['status'], 'failed')
            self.assertEqual(out['jobs'][0]['exit_code'], 0)
            self.assertTrue(out['jobs'][0]['child_reaped'])

    def test_prelaunch_actor_source_capacity_or_existing_output_reject_without_invented_wait(self):
        for scenario in ['actor_before', 'source_before', 'capacity', 'existing_root']:
            f = DriverFixture(scenario)
            with self.subTest(scenario=scenario), self.assertRaises(AssertionError):
                f.run()
            out = f.final()
            self.assertEqual(out['status'], 'failed')
            self.assertEqual(out['jobs'][0]['failed_validation_stage'], 'prelaunch')
            self.assertNotIn('exit_code', out['jobs'][0])
            self.assertEqual(f.launches, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
