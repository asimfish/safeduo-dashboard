"""All four original hands: independent XML oracle and physical-step slew."""
from pathlib import Path
import json
import unittest
import numpy as np
from coupled_hand_filter_v1 import coupled_filtered_hand_targets
from audit_coupled_hand_targets_native_v3 import matrix, expected_target

H = Path(__file__).resolve().parent
P = H / 'fourhand_native_probe_v54'


class MicrostepSlewTests(unittest.TestCase):
    def assets(self):
        reg = json.loads((P/'REGISTRATION_V1.json').read_text())
        raw = Path(json.loads((P/'RESOURCE_EXECUTION_V1.json').read_text())['out'])
        meta = json.loads(Path(reg['coupled_assets']).read_text())
        with np.load(raw/'native_parameters.npz', allow_pickle=False) as params:
            for arm, asset in meta['assets'].items():
                names = asset['hand_joint_names']
                ids = [params[arm+'_joint_names'].tolist().index(n) for n in names]
                roots, A, b = matrix(names, asset['original_URDF'])
                rootids = [names.index(n) for n in roots]
                opened=A@params[arm+'_initial_q'][0, ids][rootids].astype(float)+b
                yield asset, names, A, b, rootids, params[arm+'_hard_limits'][0, ids].astype(float), opened, reg

    def test_abrupt_nominal_commands_match_independent_oracle_each_microstep(self):
        rng = np.random.default_rng(55261011)
        for asset, names, A, b, roots, hard, opened, reg in self.assets():
            q = opened.copy()
            for _ in range(20):
                # Use convex combinations of known feasible endpoints; testing
                # targets is distinct from native reachable-space validation.
                with np.load(reg['coupled_reference_endpoints'], allow_pickle=False) as refs:
                    arm = next(a for a, x in json.loads(Path(reg['coupled_assets']).read_text())['assets'].items() if x == asset)
                    far = A@refs[arm+'_hand_far_q'][roots].astype(float)+b
                nominal = opened + rng.uniform(0., 1.)*(far-opened)
                for _ in range(reg['physics_substeps_per_control']):
                    expected = expected_target(names,A,b,roots,nominal,q,hard,reg['physics_dt_s'],True,reg['interior_hand_target_settings'])
                    actual = coupled_filtered_hand_targets(names,nominal,q,hard,reg['physics_dt_s'],True,asset['relations'],**reg['interior_hand_target_settings'])
                    np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=0)
                    self.assertLessEqual(abs(actual-q).max(),.6*reg['physics_dt_s']+1e-14)
                    np.testing.assert_allclose(A@actual[roots]+b,actual,atol=1e-14,rtol=0)
                    q = actual

    def test_one_control_budget_is_spread_across_sixteen_updates(self):
        for asset,names,A,b,roots,hard,opened,reg in self.assets():
            nominal = A@(opened[roots]+.05)+b
            macro = coupled_filtered_hand_targets(names,nominal,opened,hard,reg['control_dt_s'],False,asset['relations'],**reg['interior_hand_target_settings'])
            q = opened.copy()
            jumps = []
            for _ in range(16):
                out = coupled_filtered_hand_targets(names,nominal,q,hard,reg['physics_dt_s'],False,asset['relations'],**reg['interior_hand_target_settings'])
                jumps.append(float(abs(out-q).max()));q=out
            np.testing.assert_allclose(q,macro,atol=1e-13,rtol=0)
            self.assertLessEqual(max(jumps),float(abs(macro-opened).max())/16.+1e-13)

    def test_release_reaches_original_open_endpoint_without_jump(self):
        for asset,names,A,b,roots,hard,opened,reg in self.assets():
            q = coupled_filtered_hand_targets(names,opened,opened,hard,reg['control_dt_s'],True,asset['relations'],**reg['interior_hand_target_settings'])
            for _ in range(200):
                out = coupled_filtered_hand_targets(names,opened,q,hard,reg['physics_dt_s'],False,asset['relations'],**reg['interior_hand_target_settings'])
                self.assertLessEqual(abs(out-q).max(),.6*reg['physics_dt_s']+1e-14);q=out
            np.testing.assert_allclose(q,opened,atol=1e-14,rtol=0)


if __name__ == '__main__':
    unittest.main()
