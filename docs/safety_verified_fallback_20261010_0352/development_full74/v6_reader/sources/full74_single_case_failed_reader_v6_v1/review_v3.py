"""Small failure-route contract regressions; no native fixture can become PASS."""
from evidence_io import cpu_limits
cpu_limits()
import argparse,copy,os,resource,unittest
from evidence_io import OUTPUT,contained,write_new
from failed_binding import failed_wait
from single_reader import diagnostic_envelope

def fixture():
    return dict(schema='astra.full74.owned_cpu_wait.v1',backend='PARENT_SUPERVISING_NATIVE_SINGLE_CASE',child_pid=100,waited_pid=100,waitpid_observed=True,
        raw_wait_status=15,actual_wait_exit=-15,wait_mechanism='os.wait4(owned_pid, WNOHANG)',resource_abort='ValueError: descendant ancestry changed',error='ValueError: descendant ancestry changed',
        owned_identities=[dict(pid=100,ppid=99,startticks=123,live=False,enrollment=dict(kind='direct_fork_handshake',parent_pid=99))])
class FailureRoute(unittest.TestCase):
    def test_negative_outer_is_not_success(self):
        self.assertEqual(failed_wait(fixture())['pid'],100)
        for k,v in [('actual_wait_exit',0),('raw_wait_status',0),('waited_pid',101),('waitpid_observed',False),('resource_abort',None),('backend','CPU_ONLY')]:
            with self.assertRaises(ValueError):failed_wait(dict(fixture(),**{k:v}))
    def test_live_and_missing_enrollment_denied(self):
        for kind in ('live','enrollment'):
            bad=fixture()
            if kind=='live':bad['owned_identities'][0]['live']=True
            else:bad['owned_identities'][0]['enrollment']['kind']='unverified'
            with self.assertRaises(ValueError):failed_wait(bad)
    def test_unknown_device_and_failure_are_unconditional(self):
        report=diagnostic_envelope();report['audit_completed']=True
        for key in ('native_pass','data_pass','qualification','stage_pass','safety_acceptance','prefix_qualified'):self.assertIs(report[key],False)
        self.assertEqual(report['status'],'FAILED_DIAGNOSTIC');self.assertEqual(report['actual_CUDA_identity'],'UNKNOWN');self.assertEqual(report['actual_Kit_identity'],'UNKNOWN')
    def test_real_CPU_limits(self):
        self.assertEqual(resource.getrlimit(resource.RLIMIT_AS),(1024**3,1024**3));self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'')
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):self.assertEqual(os.environ[key],'1')
def main():
    p=argparse.ArgumentParser();p.add_argument('--proof-dir',required=True);a=p.parse_args();out=contained(a.proof_dir,OUTPUT);out.mkdir(parents=True,exist_ok=False)
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FailureRoute))
    write_new(out/'CPU_TESTS.json',dict(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),native_pass=False,fixtures_never_native=True))
    return 0 if r.wasSuccessful() else 2
if __name__=='__main__':raise SystemExit(main())
