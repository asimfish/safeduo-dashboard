from pathlib import Path
import json,unittest
import numpy as np
from full74_geometry_v2 import Full74Geometry,ARMS
from hand_kinematic_replay_v1 import quaternion_matrix

H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')


class FullStateGeometryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=H/'fourhand_native_probe_v113';reg=json.loads((p/'REGISTRATION_V1.json').read_text())
        raw=Path(json.loads((p/'RESOURCE_EXECUTION_V1.json').read_text())['out'])
        with np.load(raw/'native_parameters.npz') as z:cls.params={k:z[k] for k in z.files}
        cls.assets=json.loads(Path(reg['coupled_assets']).read_text())['assets']
        cls.fk=Full74Geometry(cls.params,cls.assets)
        with np.load(reg['contact_reproducer']['fixture']) as z:
            cls.seed={a:z[a+'_native_q_seed'].copy() for a in ARMS}
        cls.raw=raw

    def test_ur_base_frame_does_not_include_internal_fixed_joint_yaw(self):
        c=json.loads((self.raw/'physics_chunks.json').read_text())[0]
        with np.load(self.raw/c['path']) as z:
            q={a:z[a+'_native_q'][:1,0] for a in ARMS};pred=self.fk.transforms(q)
            for a in ('U_L','U_R'):
                i=self.fk.bodies[a].index('base_link')
                native=quaternion_matrix(z[a+'_native_body_transforms_after_xyzw'][:1,0,i,3:])
                np.testing.assert_allclose(pred[a]['base_link'][:,:3,:3],native,atol=1e-6,rtol=0)

    def test_each_finger_root_changes_geometry_and_only_its_arm(self):
        before=self.fk.centers(self.seed)
        for a in ARMS:
            followers={r['slave'] for r in self.assets[a]['relations']}
            roots=[self.fk.names[a].index(n) for n in self.assets[a]['independent_motor_joint_names']]
            self.assertEqual(len(roots),6)
            for i in roots:
                self.assertNotIn(self.fk.names[a][i],followers)
                q={arm:x.copy() for arm,x in self.seed.items()};q[a][0,i]+=.25
                after=self.fk.centers(q);s=self.fk.sph._arm_slices[a]
                self.assertGreater(float(abs(after[:,s]-before[:,s]).max()),.0001,(a,self.fk.names[a][i]))
                self.assertGreater(float(abs(self.fk.gaps(q)-self.fk.gaps(self.seed)).max()),.0001)
                for other in ARMS:
                    if other!=a:np.testing.assert_array_equal(after[:,self.fk.sph._arm_slices[other]],before[:,self.fk.sph._arm_slices[other]])

    def test_measured_passive_follower_is_not_synthesized_from_root(self):
        q={a:x.copy() for a,x in self.seed.items()};a='F_L'
        slave=self.fk.names[a].index('left_thumb_3_joint');q[a][0,slave]+=.2
        one=self.fk.transforms(self.seed)[a];two=self.fk.transforms(q)[a]
        np.testing.assert_array_equal(one['left_thumb_2'],two['left_thumb_2'])
        self.assertGreater(float(abs(one['left_thumb_4']-two['left_thumb_4']).max()),.01)

    def test_all74_required_and_nonfinite_rejected(self):
        q={a:x.copy() for a,x in self.seed.items()};q['F_R']=q['F_R'][:,:7]
        with self.assertRaises(ValueError):self.fk.gaps(q)
        q={a:x.copy() for a,x in self.seed.items()};q['U_L'][0,-1]=np.nan
        with self.assertRaises(ValueError):self.fk.gaps(q)
        with self.assertRaises(ValueError):self.fk.gaps({'F_L':self.seed['F_L']})

    def test_batch_chunking_equivalent_and_inputs_not_mutated(self):
        q={a:np.repeat(x,3,axis=0) for a,x in self.seed.items()};q['U_R'][1,-1]+=.1
        saved={a:x.copy() for a,x in q.items()};batch=self.fk.gaps(q)
        split=np.concatenate([self.fk.gaps({a:x[i:i+1] for a,x in q.items()}) for i in range(3)])
        np.testing.assert_allclose(batch,split,atol=2e-7,rtol=0)
        for a in ARMS:np.testing.assert_array_equal(q[a],saved[a])

    def test_original_pair_count_names_and_radii_preserved(self):
        legacy=self.fk.provider.sph
        self.assertEqual(self.fk.sph.qualified_names,legacy.qualified_names)
        np.testing.assert_array_equal(self.fk.radii,legacy.radii.numpy())
        for original,current in zip([legacy.pairs_cross,legacy.pairs_self,legacy.pairs_table],self.fk.pairs):np.testing.assert_array_equal(current,original.numpy())
        self.assertEqual(self.fk.gaps(self.seed).shape,(1,9021))


if __name__=='__main__':unittest.main()
