"""Captured bad launch rejected; good camera and GPU1 numerical paths allowed."""
from pathlib import Path
import json,unittest
from camera_launch_contract import require_supported_camera_device
HERE=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
    def test_captured_gpu1_camera_failure_is_rejected(self):
        p=json.loads((HERE/'visual_plan_v3.json').read_text())
        job=next(j for j in p['jobs'] if j['mode']=='motion_admission')
        with self.assertRaisesRegex(ValueError,'logical cuda:0'):require_supported_camera_device(job['argv'])
    def test_same_source_gpu0_reference_and_registered_retry_allowed(self):
        p=json.loads((HERE/'visual_plan_v3.json').read_text())
        require_supported_camera_device(next(j for j in p['jobs'] if j['mode']=='joint_reference')['argv'])
        p=json.loads((HERE/'visual_retry_plan.json').read_text())
        require_supported_camera_device(p['jobs'][0]['argv'])
    def test_gpu1_numerical_plan_not_blocked(self):
        p=json.loads((HERE/'plans/holdout_1_plan.json').read_text())
        for j in p['jobs']:require_supported_camera_device(j['argv'])
if __name__=='__main__':unittest.main(verbosity=2)
