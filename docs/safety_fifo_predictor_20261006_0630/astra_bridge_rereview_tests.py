"""Read-only bridge binding checks; never invokes main/copy or writes parent files."""
from pathlib import Path
import copy,hashlib,json,types,unittest
H=Path(__file__).resolve().parent
SOURCE=H/'finalize_analysis_effective.py'

def module():
    m=types.ModuleType('astra_bridge_readonly_test');m.__file__=str(SOURCE)
    exec(compile(SOURCE.read_bytes(),str(SOURCE),'exec'),m.__dict__)
    return m

class BridgeBinding(unittest.TestCase):
    def test_current_actual_compatible_bindings(self):
        m=module();reg,lp,result,path,process=m.compatible_binding()
        self.assertEqual(result['cases'],79);self.assertEqual(process['observed_tool_returncode'],0)
        self.assertEqual(path,H/'lp_backend_compatible/first_failure_audit.json')
    def test_receipt_or_registration_mutations_rejected(self):
        mutations=[('LP_COMPATIBLE_EXECUTION.json','numpy','bad'),('LP_COMPATIBLE_EXECUTION.json','interpreter','/wrong'),('LP_COMPATIBLE_EXECUTION.json','wrapper_sha256','bad'),('LP_COMPATIBLE_EXECUTION.json','source_sha256','bad'),('LP_COMPATIBLE_EXECUTION.json','result_path','/wrong'),('LP_COMPATIBLE_EXECUTION.json','cases',78),('LP_COMPATIBLE_EXECUTION.json','utc','2000-01-01T00:00:00+00:00'),('LP_COMPATIBLE_PROCESS_EXIT.json','observed_tool_returncode',1),('LP_COMPATIBLE_PROCESS_EXIT.json','log_sha256','bad'),('LP_BACKEND_REGISTRATION.json','geometry_threshold_epsilon_added',True)]
        for name,key,value in mutations:
            with self.subTest(name=name,key=key):
                m=module();original=m.load
                def altered(n):
                    d=copy.deepcopy(original(n))
                    if n==name:d[key]=value
                    return d
                m.load=altered
                with self.assertRaises(AssertionError):m.compatible_binding()
    def test_current_source_mutation_rejected(self):
        m=module();original=m.sha
        m.sha=lambda p:'bad' if Path(p)==H/'run_lp_compatible.py' else original(p)
        with self.assertRaises(AssertionError):m.compatible_binding()
if __name__=='__main__':unittest.main(verbosity=2)
