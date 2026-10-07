"""CPU-only captured-source diagnostics. No simulation, raw scoring or parent writes."""
import ast
import copy
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest

H = Path(__file__).resolve().parent
SNAPSHOT = json.loads((H / 'ASTRA_ZERO_FAILURE_EVIDENCE_SNAPSHOT_01.json').read_text())
FILES = SNAPSHOT['files']


def source(name):
    return FILES[str(H / name)]['text']


def document(path):
    return json.loads(FILES[str(path)]['text'])


def function(name, symbol, scope):
    node = next(n for n in ast.parse(source(name)).body
                if isinstance(n, ast.FunctionDef) and n.name == symbol)
    env = dict(scope)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<captured ' + name + '>', 'exec'), env)
    return env[symbol]


def require(condition, message):
    if not condition:
        raise ValueError(message)


class NoRawLedger:
    def json(self, *args, **kwargs):
        raise AssertionError('unexpected input read beyond completion gate')


class MemoryPath:
    """Only an in-memory seam; no method delegates IO to pathlib."""
    documents = {}

    def __init__(self, value):
        self.value = str(value)

    def __str__(self):
        return self.value

    def __truediv__(self, value):
        return MemoryPath(self.value + '/' + str(value))

    def __eq__(self, other):
        return str(self) == str(other)

    def exists(self):
        return False

    def is_file(self):
        return True

    def open(self, *args, **kwargs):
        return io.StringIO()

    def mkdir(self, *args, **kwargs):
        pass

    def rename(self, *args):
        pass

    def write_bytes(self, data):
        pass

    def read_bytes(self):
        return json.dumps(self.documents[str(self)]).encode()

    def read_text(self):
        return json.dumps(self.documents[str(self)])

    def iterdir(self):
        return iter([self / 'protocol.json'])


def camera_fixture(missing_protocol=False):
    original = document(H / 'visual_plan.json')
    amended = document(H / 'visual_recovery_plan.json')
    # Source/actor checks are outside this fault-injection seam.
    original['sources'] = amended['sources'] = {}
    original['original_source_sha256'] = amended['original_source_sha256'] = {}
    writes = []

    def read(path):
        if str(path).endswith('/visual_plan.json'):
            return original
        if str(path).endswith('/visual_recovery_plan.json'):
            return amended
        if str(path).endswith('/visual_protocol.json'):
            if missing_protocol:
                raise FileNotFoundError('synthetic missing visual protocol after child0')
            return dict(status='complete', steps=960, completed_windows=64,
                        actor_sha256=amended['actor_sha256'])
        raise AssertionError(str(path))

    env = dict(H=MemoryPath(H), Path=MemoryPath, read=read,
               sha=lambda p: amended['actor_sha256'],
               write=lambda p, d: writes.append((str(p), copy.deepcopy(d))),
               subprocess=types.SimpleNamespace(STDOUT=-2, Popen=lambda *a, **k:
                   types.SimpleNamespace(pid=-1, wait=lambda: 0)),
               os=types.SimpleNamespace(environ={}), datetime=datetime, timezone=timezone,
               print=lambda *a, **k: None)
    return function('recover_unstarted.py', 'camera', env), writes


class ExitContractDiagnostics(unittest.TestCase):
    def test_real_numeric_wrapper_returns_zero_for_captured_failed_campaign(self):
        # Execute the captured wrapper with only its campaign import replaced.
        # All parent inputs are read-only and the replacement launches nothing.
        child_code = '''
import json,sys,types
from pathlib import Path
h=Path(sys.argv[1]);s=json.loads((h/'ASTRA_ZERO_FAILURE_EVIDENCE_SNAPSHOT_01.json').read_text())
campaign=json.loads(s['files'][s['initial_campaign_paths']['1']]['text'])
assert campaign['status']=='complete_with_failures'
def captured(plan,out):
    assert plan==campaign['plan'] and out==plan['output_root']
    return campaign
stub=types.ModuleType('isolated_campaign_v2');stub.run_campaign=captured
sys.modules['isolated_campaign_v2']=stub
path=h/'execute_numeric.py';sys.argv=[str(path),'1']
exec(compile(s['files'][str(path)]['text'],str(path),'exec'),{'__file__':str(path),'__name__':'__main__'})
'''
        child = subprocess.Popen([sys.executable, '-B', '-c', child_code, str(H)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''})
        stdout, stderr = child.communicate(timeout=15)
        print(json.dumps(dict(diagnostic='ORIGINAL_WRAPPER_FALSE_ZERO_REPRODUCED',
            actual_child_pid=child.pid, actual_child_exit_code=child.returncode,
            child_reaped=True, stdout=stdout, stderr=stderr)), flush=True)
        self.assertEqual(child.returncode, 0)
        self.assertIn('CLOSED 1 complete_with_failures', stdout)

    def test_original_aggregate_accepts_false_zero_without_campaign_validation(self):
        expression = next(n for n in ast.walk(ast.parse(source('execute_all.py')))
            if isinstance(n, ast.IfExp) and isinstance(n.body, ast.Constant)
            and n.body.value == 'PASS_ALL_NUMERIC_CLOSED')
        rows = document(H / 'initial_attempts/NUMERIC_EXECUTION.json')['jobs']
        result = eval(compile(ast.Expression(expression), '<captured status predicate>', 'eval'),
                      dict(rows=rows))
        campaigns = [document(p) for p in SNAPSHOT['initial_campaign_paths'].values()]
        self.assertEqual(result, 'PASS_ALL_NUMERIC_CLOSED')
        self.assertFalse(all(c['status'] == 'complete' for c in campaigns))

    def test_frozen_raw_entry_rejects_failed_child_zero_before_any_array_read(self):
        fn = function('astra_zero_raw_score.py', 'score_cell', dict(Path=Path, require=require))
        for block in ['1', '2']:
            campaign = document(SNAPSHOT['initial_campaign_paths'][block])
            record = campaign['jobs'][2]
            self.assertEqual(record['exit_code'], 0)
            with self.assertRaisesRegex(ValueError, 'condition did not complete'):
                fn(NoRawLedger(), campaign['plan'], campaign['plan']['jobs'][2], record)

    def test_recovery_camera_success_receipt_omits_frozen_plan_binding(self):
        fn, writes = camera_fixture()
        fn()
        execution = writes[-1][1]
        self.assertEqual(execution['status'], 'complete')
        self.assertNotIn('plan_sha256', execution)
        legacy = function('astra_zero_camera_prepare.py', 'run', dict(H=H, require=require))
        ledger = types.SimpleNamespace(json=lambda p: execution,
            hashes={str(H / 'visual_plan.json'): FILES[str(H / 'visual_plan.json')]['sha256']})
        with self.assertRaisesRegex(KeyError, 'plan_sha256'):
            legacy(ledger, document(H / 'visual_plan.json'), {})

    def test_frozen_camera_mode_rejects_amended_device_before_any_array_read(self):
        fn = function('astra_zero_camera_prepare.py', 'audit_mode', dict(Path=Path, require=require))
        original = document(H / 'visual_plan.json')
        amended = document(H / 'visual_recovery_plan.json')
        for job in amended['jobs']:
            record = dict(status='complete', exit_code=0, argv=job['argv'],
                          out=str(Path(amended['actual_visual_root']) / job['mode']))
            with self.assertRaisesRegex(ValueError, 'camera actual argv differs'):
                fn(NoRawLedger(), original, {}, record, job['mode'])

    def test_recovery_camera_missing_protocol_leaves_persisted_running_without_exit(self):
        fn, writes = camera_fixture(missing_protocol=True)
        with self.assertRaises(FileNotFoundError):
            fn()
        persisted = writes[-1][1]
        self.assertEqual(persisted['status'], 'running')
        self.assertEqual(persisted['jobs'][0]['status'], 'running')
        self.assertNotIn('exit_code', persisted['jobs'][0])

    def test_recovery_numeric_failed_validation_leaves_persisted_running_without_exit(self):
        campaign = document(SNAPSHOT['initial_campaign_paths']['1'])
        plan = campaign['plan']
        original_protocol_path = next(r['evidence_root'] + '/protocol.json'
            for r in SNAPSHOT['numeric_classification'] if r['block'] == 1 and r['child_status'] == 'failed')
        protocol = document(original_protocol_path)
        writes = []
        MemoryPath.documents = {
            str(Path(plan['output_root']) / 'campaign.json'): campaign,
            str(Path(plan['output_root']) / plan['jobs'][2]['id'] / 'protocol.json'): protocol,
        }

        def read(path):
            if str(path).endswith('_plan.json'):
                return plan
            if str(path).endswith('/campaign.json'):
                return copy.deepcopy(campaign)
            if str(path).endswith('/protocol.json'):
                return protocol
            raise AssertionError(str(path))

        helper = H.parent / 'safety_random_space_20261004/isolated_campaign_v2.py'
        node = next(n for n in ast.parse(FILES[str(helper)]['text']).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'verify_child')
        namespace = dict(json=json)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<captured verify_child>', 'exec'), namespace)

        fn = function('recover_unstarted.py', 'numeric', dict(H=MemoryPath(H),
            RAW=MemoryPath('/synthetic-raw'), Path=MemoryPath, read=read, frozen=lambda p: None,
            copy=copy, write=lambda p, d: writes.append(copy.deepcopy(d)),
            subprocess=types.SimpleNamespace(STDOUT=-2, Popen=lambda *a, **k:
                types.SimpleNamespace(pid=-1, wait=lambda: 0)),
            os=types.SimpleNamespace(environ={}), verify_child=namespace['verify_child']))
        with self.assertRaisesRegex(ValueError, 'complete exactly one cell'):
            fn(1)
        self.assertEqual(writes[-1]['status'], 'recovery_running')
        self.assertEqual(writes[-1]['jobs'][2]['status'], 'running')
        self.assertNotIn('exit_code', writes[-1]['jobs'][2])

    def test_amended_visual_plan_changes_exactly_two_device_tokens(self):
        original = document(H / 'visual_plan.json')
        expected = copy.deepcopy(original)
        for job in expected['jobs']:
            self.assertEqual(job['argv'][job['argv'].index('--device') + 1], 'cuda:0')
            job['argv'][job['argv'].index('--device') + 1] = 'cuda:1'
        self.assertEqual(expected, document(H / 'visual_recovery_plan.json'))


class RecoveryPresentationFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location('owned_presentation_fixtures',
            H / 'astra_zero_presentation_source_tests.py')
        cls.fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.fixture)
        cls.presentation = json.loads((H / 'ASTRA_ZERO_RECOVERY_PRESENTATION_SNAPSHOT_01.json').read_text())
        cls.fixture.SOURCES['build_report.py'] = cls.presentation['sources']['build_report.py']['text']

    def test_report_rejects_missing_recovery_and_initial_448_128(self):
        data = self.fixture.report_docs()
        with self.assertRaises(KeyError):
            self.fixture.run_report(data)
        data['RECOVERY_EXECUTION.json'] = dict(status='PASS_REGISTERED_UNSTARTED_RECOVERY_CLOSED')
        data['holdout_results.json'].update(completed_method_windows=448, invalid_method_windows=128)
        with self.assertRaises(AssertionError):
            self.fixture.run_report(data)

    def test_synthetic_complete_report_keeps_all_new_and_initial_deviations(self):
        data = self.fixture.report_docs()
        data['RECOVERY_EXECUTION.json'] = dict(status='PASS_REGISTERED_UNSTARTED_RECOVERY_CLOSED')
        report = self.fixture.run_report(data)['REPORT.md']
        for required in ['448完成', '128', 'Kit退出0', 'CUDA0改CUDA1', '传递USD',
                         '初始GPU空闲内存门禁实际失败', 'SIGSTOP', 'SIGCONT',
                         '相对初始测量q的最大已发目标位移rad', 'FAIL_EXACT_FORWARD']:
            self.assertIn(required, report)

    def test_follower_rejects_claimed_pass_with_failed_campaign_child(self):
        text = self.presentation['sources']['follow_recovery_analysis.py']['text']
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'wait_recovered')

        class FollowerPath(MemoryPath):
            documents = {}

            def __truediv__(self, value):
                return FollowerPath(self.value + '/' + str(value))

            def exists(self):
                return str(self) in self.documents

        docs = {str(H / 'NUMERIC_EXECUTION.json'): dict(status='PASS_ALL_NUMERIC_CLOSED', recovery_jobs=[{}, {}])}
        for block, path in SNAPSHOT['initial_campaign_paths'].items():
            campaign = document(path)
            docs[str(H / 'plans' / ('holdout_' + block + '_plan.json'))] = campaign['plan']
            # Even with a forged overall complete status, the failed child must reject.
            campaign['status'] = 'complete'
            docs[str(Path(campaign['plan']['output_root']) / 'campaign.json')] = campaign
        FollowerPath.documents = docs
        namespace = dict(H=FollowerPath(H), Path=FollowerPath, json=json,
            time=types.SimpleNamespace(monotonic=lambda: 0,
                sleep=lambda n: (_ for _ in ()).throw(AssertionError('unexpected wait'))))
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<captured follow_recovery_analysis>', 'exec'), namespace)
        with self.assertRaises(AssertionError):
            namespace['wait_recovered']()


if __name__ == '__main__':
    unittest.main(verbosity=2)
