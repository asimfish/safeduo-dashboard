"""Regression against installed Kit's actual bootstrap environment function."""
import ast
import contextlib
import io
import unittest
from unittest.mock import patch
from repair_common_v2 import *
import parent_runtime_v2 as runtime

SDK = Path('/home/liyufeng/miniforge3/envs/safeduo/lib/python3.11/site-packages/isaacsim/kit/kit_app.py')
OLD = H / 'native_camera_repair_v3'

def sdk_checker():
    raw = SDK.read_bytes()
    tree = ast.parse(raw, str(SDK))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'check_eula')
    module = ast.Module(body=[fn], type_ignores=[])
    namespace = dict(os=os, __file__=str(SDK))
    exec(compile(module, str(SDK), 'exec'), namespace)
    return namespace['check_eula']

class BootstrapContract(unittest.TestCase):
    def test_real_child_environment_passes_installed_check_without_prompt_or_write(self):
        env = runtime.child_environment(BASE / 'cpu_environment_fixture', native=True)
        self.assertEqual(env['OMNI_KIT_ACCEPT_EULA'], 'YES')
        self.assertEqual(env['PRIVACY_CONSENT'], 'Y')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0')
        with patch.dict(os.environ, env, clear=True), patch('builtins.input', side_effect=AssertionError('prompt forbidden')), \
             patch('builtins.open', side_effect=AssertionError('SDK receipt writes forbidden')):
            sdk_checker_cached()

    def test_old_misnamed_flag_reproduces_observed_EOF(self):
        with patch.dict(os.environ, {'ACCEPT_EULA': 'Y'}, clear=True), \
             patch('os.path.isfile', return_value=False), patch('builtins.input', side_effect=EOFError('no stdin')), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(EOFError):
                sdk_checker_cached()

    def test_runtime_change_is_only_bootstrap_flag_and_new_namespace(self):
        names = ['repair_common_v2.py', 'parent_runtime_v2.py', 'parent_app_integration_v2.py',
                 'request_review_builder_v2.py', 'repair_phase_trace_v2.py',
                 'native_camera_repair_entry_v2.py', 'native_measurements_v2.py']
        for name in names:
            old = (OLD / name).read_text()
            new = (HERE / name).read_text()
            expected = old.replace('native_camera_repair_CPU_v3', 'native_camera_repair_CPU_v4') \
                          .replace('env002_FR_native_entry_v3_attempt1', 'env002_FR_native_entry_v4_attempt1') \
                          .replace('parent_camera_repair_review_v3', 'parent_camera_repair_review_v4')
            if name == 'parent_runtime_v2.py':
                expected = expected.replace("ACCEPT_EULA='Y'", "OMNI_KIT_ACCEPT_EULA='YES'")
            self.assertEqual(ast.dump(ast.parse(new)), ast.dump(ast.parse(expected)), name)
        self.assertEqual(CANDIDATE, H / 'astra/full74_camera_upgrade_v3')
        core.verify(read(HERE / 'PRESERVED_V3_BYTES_V4.json')['files'])

sdk_checker_cached = sdk_checker()

if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BootstrapContract))
    write(CPU / 'CPU_BOOTSTRAP_ENVIRONMENT_REPORT_V4.json', dict(
        status='PASS_CPU_ONLY' if result.wasSuccessful() else 'FAILED_CPU_ONLY',
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        official_installed_bootstrap_source=dict(path=str(SDK), sha256=sha(SDK)),
        sources={str(p):sha(p) for p in HERE.glob('*.py')}, old_V3_sources_unchanged=True,
        native_started=False, physical_safety_certified=False))
    sys.exit(0 if result.wasSuccessful() else 2)
