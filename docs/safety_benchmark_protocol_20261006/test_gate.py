import unittest
from gate import COMMON,TASK,registration_gate
class Tests(unittest.TestCase):
    def test_old_random_count_cannot_substitute_for_task_evidence(self):
        r=registration_gate('real_task',{'windows':100000,'tests_passed':100})
        self.assertEqual(r['status'],'NOT_READY_TO_REGISTER')
        self.assertIn('real_grasp_release',r['missing']);self.assertIn('independent_hazard_oracle',r['missing'])
    def test_failed_receipt_still_blocks(self):
        e={k:dict(status='PASS',evidence_path='synthetic-unit-fixture',sha256='0'*64) for k in COMMON+TASK}
        e['real_grasp_release']['status']='FAIL'
        self.assertIn('real_grasp_release',registration_gate('real_task',e)['missing'])
    def test_completeness_is_not_certification(self):
        e={k:dict(status='PASS',evidence_path='synthetic-unit-fixture',sha256='0'*64) for k in COMMON}
        r=registration_gate('random_motion',e)
        self.assertEqual(r['status'],'READY_TO_REGISTER');self.assertFalse(r['physical_safety_certified'])
if __name__=='__main__':unittest.main()
