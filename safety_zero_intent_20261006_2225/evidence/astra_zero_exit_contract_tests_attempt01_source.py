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
               subprocess=types.SimpleNamespace(Popen=lambda *a, **k:
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
        MemoryPath.documents = {str(Path(plan['output_root']) / 'campaign.json'): campaign}

        def read(path):
            if str(path).endswith('_plan.json'):
                return plan
            if str(path).endswith('/campaign.json'):
                return copy.deepcopy(campaign)
            if str(path).endswith('/protocol.json'):
                return protocol
            raise AssertionError(str(path))

        def invalid(*args):
            raise ValueError('each fresh process must complete exactly one cell')

        fn = function('recover_unstarted.py', 'numeric', dict(H=MemoryPath(H),
            RAW=MemoryPath('/synthetic-raw'), Path=MemoryPath, read=read, frozen=lambda p: None,
            copy=copy, write=lambda p, d: writes.append(copy.deepcopy(d)),
            subprocess=types.SimpleNamespace(Popen=lambda *a, **k:
                types.SimpleNamespace(pid=-1, wait=lambda: 0)),
            os=types.SimpleNamespace(environ={}), verify_child=invalid))
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
