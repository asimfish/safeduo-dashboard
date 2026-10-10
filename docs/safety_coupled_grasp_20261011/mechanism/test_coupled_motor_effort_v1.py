import json,unittest,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from coupled_motor_effort_v1 import motor_effort_projection
H=Path(__file__).resolve().parent
class MotorEffortTests(unittest.TestCase):
    def assets(self):
        meta=json.loads((H/'coupled_hand_assets_v2/COUPLED_ASSETS_V1.json').read_text())
        for arm,a in meta['assets'].items():yield arm,a
    def test_virtual_work_all24_original_relations_and_zero_follower_actuation(self):
        rng=np.random.default_rng(20261011)
        for arm,a in self.assets():
            names=['synthetic_arm_axis',*a['hand_joint_names']];M=motor_effort_projection(names,a['relations'])
            roots=[i for i,n in enumerate(names) if n not in {r['slave'] for r in a['relations']}]
            for _ in range(100):
                velocity=np.zeros(len(names));velocity[roots]=rng.normal(size=len(roots))
                # Expand velocities by an independent original-XML traversal.
                joints={j.get('name'):j for j in ET.parse(a['original_URDF']).findall('joint')}
                def v(n):
                    if n=='synthetic_arm_axis' or joints[n].find('mimic') is None:return velocity[names.index(n)]
                    m=joints[n].find('mimic');return float(m.get('multiplier','1'))*v(m.get('joint'))
                full=np.array([v(n) for n in names]);effort=rng.normal(size=len(names));motors=M@effort
                self.assertAlmostEqual(float(effort@full),float(motors@velocity),places=12)
                for r in a['relations']:self.assertEqual(motors[names.index(r['slave'])],0.)
                self.assertEqual(motors[0],effort[0])
    def test_offsets_do_not_change_velocity_or_effort_projection(self):
        names=['root','middle','tip'];r=[dict(slave='middle',master='root',urdf_multiplier=2.,urdf_offset_rad=-.15),dict(slave='tip',master='middle',urdf_multiplier=-.5,urdf_offset_rad=.1)]
        np.testing.assert_array_equal(motor_effort_projection(names,r),np.array([[1,2,-1],[0,0,0],[0,0,0]]))
        shifted=[dict(x,urdf_offset_rad=1e4) for x in r];np.testing.assert_array_equal(motor_effort_projection(names,r),motor_effort_projection(names,shifted))
    def test_cycles_rejected(self):
        with self.assertRaisesRegex(AssertionError,'Cyclic'):motor_effort_projection(['a','b'],[dict(slave='a',master='b',urdf_multiplier=1.),dict(slave='b',master='a',urdf_multiplier=1.)])
if __name__=='__main__':unittest.main()
