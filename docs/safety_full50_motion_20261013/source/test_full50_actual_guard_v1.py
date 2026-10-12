from pathlib import Path
import json,unittest
import numpy as np
from full74_geometry_v2 import Full74Geometry,ARMS
from full50_actual_guard_v1 import Full50ActualGuard

H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')


class GuardContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=H/'fourhand_native_probe_v113';reg=json.loads((p/'REGISTRATION_V1.json').read_text())
        raw=Path(json.loads((p/'RESOURCE_EXECUTION_V1.json').read_text())['out'])
        with np.load(raw/'native_parameters.npz') as z:cls.params={k:z[k] for k in z.files}
        meta=json.loads(Path(reg['coupled_assets']).read_text())['assets']
        cls.ids={a:np.array([i for i,n in enumerate(cls.params[a+'_joint_names']) if n not in {r['slave'] for r in meta[a]['relations']}]) for a in ARMS}
        cls.geometry=Full74Geometry(cls.params,meta)
        cls.guard=Full50ActualGuard(cls.geometry,cls.params,cls.ids)
        with np.load(reg['contact_reproducer']['fixture']) as z:cls.q={a:z[a+'_native_q_seed'][0].copy() for a in ARMS}

    def inputs(self):
        q={a:x.copy() for a,x in self.q.items()};v={a:np.zeros_like(x) for a,x in q.items()}
        roots={a:q[a][self.ids[a]].copy() for a in ARMS};zero={a:np.zeros_like(x) for a,x in roots.items()}
        return q,v,roots,[{a:x.copy() for a,x in roots.items()} for _ in range(6)],self.geometry.gaps({a:x[None] for a,x in q.items()})[0],roots,zero

    def test_all50_outputs_follow_previous_command_rate_and_acceleration(self):
        args=list(self.inputs());args[2]={a:x+.2 for a,x in args[2].items()}
        d=self.guard.decide(*args)
        self.assertNotEqual(d.mode,'UNQUALIFIED_BRAKE')
        self.assertEqual(sum(len(x) for x in d.target.values()),50)
        for a in ARMS:self.assertLessEqual(float(abs(d.target[a]-args[5][a]).max()),.5*(1/60)**2+2e-7)

    def test_negative_native_gap_never_admits_proposal(self):
        args=list(self.inputs());args[4]=args[4].copy();args[4][5]=-.000001
        d=self.guard.decide(*args);self.assertEqual(d.mode,'UNQUALIFIED_BRAKE');self.assertFalse(d.prediction_qualified)

    def test_finger_speed_limit_violation_is_not_ignored(self):
        args=list(self.inputs());a='U_R';i=self.ids[a][-1];args[1][a][i]=float(self.params[a+'_max_velocity'][0,i])+1
        self.assertEqual(self.guard.decide(*args).mode,'UNQUALIFIED_BRAKE')

    def test_missing_finger_fifo_coordinates_rejected_without_state_mutation(self):
        args=list(self.inputs());saved={a:x.copy() for a,x in args[0].items()};args[3][2]['F_L']=args[3][2]['F_L'][:7]
        with self.assertRaises(ValueError):self.guard.decide(*args)
        for a in ARMS:np.testing.assert_array_equal(args[0][a],saved[a])


if __name__=='__main__':unittest.main()
