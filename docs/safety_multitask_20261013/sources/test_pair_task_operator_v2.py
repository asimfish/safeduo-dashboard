from pathlib import Path
import json,unittest
import numpy as np
from pair_task_operator_v2 import PairTaskOperatorV2
from multitask_phases_v1 import ARMS,PayloadTask,TaskSpec

H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
K=Path(__file__).resolve().parent


class OperatorTests(unittest.TestCase):
    def setUp(self):
        reg=json.loads((H/'fourhand_native_probe_v119/REGISTRATION_V1.json').read_text())
        raw=Path(json.loads((H/'fourhand_native_probe_v119/RESOURCE_EXECUTION_V1.json').read_text())['out'])
        self.p=dict(np.load(raw/'native_parameters.npz'));refs=dict(np.load(reg['coupled_reference_endpoints']));bank=dict(np.load(reg['motion_test']['command_bank']))
        self.ids={a:bank[a+'_full50_root_indices'] for a in ARMS};endpoints=dict(np.load(reg['pair_task']['IK_endpoints']))
        self.op=PairTaskOperatorV2(PayloadTask(TaskSpec(**reg['pair_task']['spec'])),self.p,self.ids,endpoints,refs)
        self.q={a:self.p[a+'_initial_q'][0].copy() for a in ARMS};self.v={a:np.zeros_like(self.q[a]) for a in ARMS}
        self.prior={a:self.q[a][self.ids[a]].copy() for a in ARMS}

    def settle_pose(self):
        for a in self.op.task.spec.roles:
            self.q[a][self.p[a+'_controlled_indices']]=self.op.endpoints[a][0]
            self.prior[a][self.op.arm_positions[a]]=self.op.endpoints[a][0]

    def test_raised_actual_pose_cannot_start_pressure(self):
        result=self.op.request(self.prior,self.q,self.v,0.)
        for a in self.op.task.spec.roles:
            hand=[j for j,i in enumerate(self.ids[a]) if int(i) in self.op.open[a]]
            np.testing.assert_array_equal(result[a][hand],self.prior[a][hand])
            self.assertLessEqual(np.max(abs(result[a][self.op.arm_positions[a]]-self.prior[a][self.op.arm_positions[a]])),.02/60+1e-6)

    def test_actual_speed_and_FIFO_arrival_required(self):
        self.settle_pose();a='F_L';self.v[a][0]=.021
        self.op.request(self.prior,self.q,self.v,0.);self.op.request(self.prior,self.q,self.v,.3)
        self.assertEqual(self.op.mode,'UNQUALIFIED_APPROACH_HOLD_HANDS')
        self.v[a][0]=0;self.prior[a][0]+=.001
        self.op.request(self.prior,self.q,self.v,.31);self.assertIsNone(self.op.settled_start)

    def test_continuous_actual_settle_then_bounded_pressure(self):
        self.settle_pose()
        for t in np.arange(19)/60:self.op.request(self.prior,self.q,self.v,float(t))
        result=self.op.request(self.prior,self.q,self.v,.32)
        self.assertEqual(self.op.mode,'UNQUALIFIED_PRESSURE_AFTER_ACTUAL_SETTLE')
        a='F_L';index=list(self.ids[a]).index(self.p[a+'_joint_names'].tolist().index('left_index_1_joint'))
        self.assertGreater(result[a][index],self.prior[a][index]);self.assertLessEqual(result[a][index]-self.prior[a][index],.005/60+1e-6)

    def test_native119_63N_example_requests_stop_without_certifying_it(self):
        self.op.force[1]=63.2855224609375
        result=self.op.request(self.prior,self.q,self.v,0.)
        for a in ARMS:np.testing.assert_array_equal(result[a],self.q[a][self.ids[a]])
        self.assertTrue(self.op.stop_intent_required);self.assertFalse(self.op.physical_force_bound_qualified)

    def test_nonfinite_or_partial_actual_state_rejected_and_inputs_unchanged(self):
        before={a:self.q[a].copy() for a in ARMS};self.op.request(self.prior,self.q,self.v,0.)
        for a in ARMS:np.testing.assert_array_equal(before[a],self.q[a])
        self.q['F_R']=self.q['F_R'][:-1]
        with self.assertRaises(ValueError):self.op.request(self.prior,self.q,self.v,.02)


if __name__=='__main__':unittest.main()
