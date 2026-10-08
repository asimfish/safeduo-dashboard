"""CPU numerical/identity tests; no Isaac, torch, GPU, rendering or fixture writes.

Contract: all six plane distances enclose the .06m link spheres plus .05m;
native XYZW rotates directions; exact rigidpath/bodyname matching has no drops;
whole-64 native snapshots are detached and compared by bytes. Invalid inputs
fail closed. Frozen raw native_smoke_04 is read-only compatibility evidence,
NOT live camera or live native-link observation. Run with Python -B; the parent
process records the actual child wait exit code and source SHA256 in its receipt.
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

import hand_views as h

RAW = Path('/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/native_smoke_04')
IDENTITY = RAW/'native_contact_identity.json'
CAMERA_RECEIPTS = RAW/'camera_receipts.json'


def raw_fixture():
    identity = json.loads(IDENTITY.read_text())
    names, paths = {}, {}
    for arm in h.ARM_KEYS:
        # Reconstruct metadata from real rigid inventory; reverse it to ensure
        # lookup uses exact indices, never sensor order. NOT live PhysX metadata.
        env0 = list(reversed([p for p in identity['partners_env0'] if f'/env_0/{arm}/' in p]))
        names[arm] = [p.rsplit('/', 1)[-1] for p in env0]
        paths[arm] = [[p.replace('/env_0/', f'/env_{i}/') for p in env0] for i in range(64)]
    return identity, names, paths


class FrustumTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[640., 0, 640], [0, 360, 360], [0, 0, 1]])
        self.world = np.eye(4)
        self.clip = [.1, 10.]

    def margins(self, points, radius=.06):
        return h.sphere_frustum_margins(points, radius, self.world, self.K, self.clip)

    def test_all_six_planes_have_independent_numeric_oracles(self):
        got = self.margins([[.2, .3, -2]])[0]
        expected = np.array([2.2/np.sqrt(2), 1.8/np.sqrt(2), 1.7/np.sqrt(2),
                             2.3/np.sqrt(2), 1.9, 8.])-.06
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-14)

    def test_each_plane_rejects_a_sphere_whose_center_is_inside(self):
        points = [[-.99, 0, -1], [.99, 0, -1], [0, .99, -1], [0, -.99, -1],
                  [0, 0, -.11], [0, 0, -9.99]]
        center_margins = self.margins(points, 0)
        self.assertTrue((center_margins > 0).all())
        sphere_margins = self.margins(points)
        for i in range(6):
            with self.subTest(plane=h.PLANE_NAMES[i]):
                self.assertLess(sphere_margins[i, i], 0.)
                self.assertEqual(np.flatnonzero(sphere_margins[i] < 0).tolist(), [i])

    def test_additional_five_cm_is_not_confused_with_radius(self):
        point = [[0, 0, -.19]]
        got = self.margins(point)[0]
        self.assertTrue((got > 0).all())
        self.assertLess(got.min(), h.CLEARANCE_M)
        self.assertAlmostEqual(got[4], .03)

    def test_far_and_near_behind_camera_negative_controls(self):
        self.assertLess(self.margins([[0, 0, .2]])[0, 4], 0)
        self.assertLess(self.margins([[0, 0, -10.2]])[0, 5], 0)

    def test_off_center_intrinsics_against_corner_cross_product_planes(self):
        K = np.array([[1100., 0, 610.], [0, 1030., 345.], [0, 0, 1.]])
        points = np.array([[-.2, .1, -2], [.9, -.4, -3], [.01, .02, -.4]])
        def ray(u, v):
            return np.array([(u-610)/1100, (345-v)/1030, -1.])
        tl, tr, bl, br = ray(0, 0), ray(1280, 0), ray(0, 720), ray(1280, 720)
        normals = [np.cross(bl, tl), np.cross(tr, br), np.cross(tl, tr), np.cross(br, bl)]
        optical_axis = np.array([0., 0, -1])
        normals = [n/np.linalg.norm(n) * (1 if n @ optical_axis > 0 else -1) for n in normals]
        expected = np.column_stack([points @ n for n in normals] + [-points[:, 2]-.1, 10+points[:, 2]])-.06
        got = h.sphere_frustum_margins(points, .06, self.world, K, self.clip)
        np.testing.assert_allclose(got, expected, atol=1e-14, rtol=0)

    def test_row_transform_world_origins_and_rotation_invariance(self):
        points = np.array([[.1, .2, -1], [-.2, -.1, -2]])
        world = np.eye(4)
        world[:3, :3] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        world[3, :3] = [-10, -14, .7]  # Actual first group slot 48 XY origin.
        transformed = (np.column_stack((points, np.ones(2))) @ world)[:, :3]
        got = h.sphere_frustum_margins(transformed, .06, world, self.K, self.clip)
        np.testing.assert_allclose(got, self.margins(points), atol=3e-15, rtol=0)

    def test_fit_solves_sides_radius_margin_and_far_interval(self):
        points = np.array([[-.5, .2, -.3], [.4, -.3, -.4], [0, 0, -.2]])
        self.assertLess(self.margins(points).min(), .05)
        delta = h.fit_distance_delta(points, .06, self.world, self.K, self.clip)
        moved = self.world.copy()
        moved[3, 2] += delta
        margins = h.sphere_frustum_margins(points, .06, moved, self.K, self.clip)
        self.assertGreaterEqual(margins.min(), .05)
        closer = moved.copy()
        closer[3, 2] -= .001
        self.assertLess(h.sphere_frustum_margins(points, .06, closer, self.K, self.clip).min(), .05)

    def test_fit_far_clip_impossibility_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no distance'):
            h.fit_distance_delta([[0, 0, -.2], [0, 0, -2]], .06, self.world, self.K, [.1, 1.])

    def test_fit_can_move_closer_when_far_plane_is_initially_violated(self):
        points = [[0, 0, -10.2]]
        delta = h.fit_distance_delta(points, .06, self.world, self.K, self.clip)
        self.assertLess(delta, 0)
        moved = self.world.copy()
        moved[3, 2] += delta
        self.assertGreaterEqual(h.sphere_frustum_margins(points, .06, moved, self.K, self.clip).min(), .05)

    def test_invalid_scaled_transform_nan_optics_and_radius_rejected(self):
        bad_world = self.world.copy()
        bad_world[0, 0] = 2
        with self.assertRaisesRegex(ValueError, 'rigid'):
            h.sphere_frustum_margins([[0, 0, -1]], .06, bad_world, self.K, self.clip)
        for radius in [-.01, np.nan]:
            with self.assertRaises(ValueError):
                self.margins([[0, 0, -1]], radius)
        for clip in [[1, .1], [0, 1], [.1, np.inf]]:
            with self.assertRaises(ValueError):
                h.sphere_frustum_margins([[0, 0, -1]], .06, self.world, self.K, clip)


class QuaternionTests(unittest.TestCase):
    def test_physx_xyzw_z90_and_sign_invariance(self):
        q = np.array([0, 0, np.sqrt(.5), np.sqrt(.5)])
        expected = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        np.testing.assert_allclose(h.rotate_xyzw(q, np.eye(3)), expected, atol=4e-16, rtol=0)
        np.testing.assert_array_equal(h.rotate_xyzw(q, np.eye(3)), h.rotate_xyzw(-q, np.eye(3)))

    def test_actual_local_opposition_and_above_below_survive_rotation(self):
        vectors = np.array(list(h.LOCAL_DIRECTIONS.values()))
        q = np.array([np.sqrt(.5), 0, 0, np.sqrt(.5)])
        rotated = h.rotate_xyzw(q, vectors)
        np.testing.assert_allclose(rotated[0], [1, -.8, -1], atol=1e-15, rtol=0)
        np.testing.assert_allclose(rotated[1], -rotated[0], atol=1e-15, rtol=0)
        np.testing.assert_allclose(np.linalg.norm(rotated, axis=1), np.linalg.norm(vectors, axis=1))

    def test_invalid_handroot_quaternion_rejected(self):
        for q in [[0, 0, 0, 0], [0, 0, 0, 2], [0, 0, np.nan, 1]]:
            with self.assertRaises(ValueError):
                h.rotate_xyzw(q, [1, 0, 0])


class RealIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.identity, cls.names, cls.paths = raw_fixture()

    def test_real_3328_sensors_map_to_exact_native_metadata_order_without_drop(self):
        mapped = h.map_contact_sensors(self.identity, self.names, self.paths)
        seen = set()
        for arm in h.ARM_KEYS:
            view = next(v for v in self.identity['views'] if v['arm'] == arm)
            self.assertEqual(len(view['sensors']), 64*13)
            for slot, rows in enumerate(mapped[arm]):
                self.assertEqual(len(rows), 13)
                for row in rows:
                    path = self.paths[arm][slot][row['body_index']]
                    self.assertEqual(row['rigid_path'], path)
                    self.assertEqual(view['sensors'][row['sensor_index']], path)
                    self.assertEqual(self.names[arm][row['body_index']], path.rsplit('/', 1)[-1])
                    self.assertEqual(view['env_ids'][row['sensor_index']], slot)
                    self.assertNotIn(path, seen)
                    seen.add(path)
        self.assertEqual(seen, {s for v in self.identity['views'] for s in v['sensors']})
        self.assertEqual(len(seen), 3328)

    def test_unmatched_same_basename_wrong_rigid_path_is_rejected(self):
        identity = copy.deepcopy(self.identity)
        identity['views'][0]['sensors'][0] = identity['views'][0]['sensors'][0].replace('/f2_hand/', '/wrong_parent/')
        with self.assertRaisesRegex(ValueError, 'unmatched or dropped'):
            h.map_contact_sensors(identity, self.names, self.paths)

    def test_duplicate_sensor_and_duplicate_bodyname_are_rejected(self):
        identity = copy.deepcopy(self.identity)
        identity['views'][0]['sensors'][1] = identity['views'][0]['sensors'][0]
        with self.assertRaisesRegex(ValueError, 'duplicate sensors'):
            h.map_contact_sensors(identity, self.names, self.paths)
        names = copy.deepcopy(self.names)
        names['F_L'][1] = names['F_L'][0]
        with self.assertRaisesRegex(ValueError, 'duplicate/empty body_names'):
            h.map_contact_sensors(self.identity, names, self.paths)

    def test_silent_sensor_drop_even_consistent_across_all64_is_rejected(self):
        identity = copy.deepcopy(self.identity)
        view = identity['views'][0]
        keep = [i for i, p in enumerate(view['sensors']) if not p.endswith('/left_thumb_4')]
        view['sensors'] = [view['sensors'][i] for i in keep]
        view['env_ids'] = [view['env_ids'][i] for i in keep]
        with self.assertRaisesRegex(ValueError, 'unmatched or dropped'):
            h.map_contact_sensors(identity, self.names, self.paths)

    def test_wrong_env_row_env_id_body_order_and_arm_duplication_rejected(self):
        paths = copy.deepcopy(self.paths)
        paths['U_L'][1], paths['U_L'][2] = paths['U_L'][2], paths['U_L'][1]
        with self.assertRaisesRegex(ValueError, 'environment/arm row'):
            h.map_contact_sensors(self.identity, self.names, paths)
        identity = copy.deepcopy(self.identity)
        identity['views'][2]['env_ids'][0] = 63
        with self.assertRaisesRegex(ValueError, 'env_id'):
            h.map_contact_sensors(identity, self.names, self.paths)
        names = copy.deepcopy(self.names)
        names['F_R'][0], names['F_R'][1] = names['F_R'][1], names['F_R'][0]
        with self.assertRaisesRegex(ValueError, 'bodyname mismatch'):
            h.map_contact_sensors(self.identity, names, self.paths)
        identity = copy.deepcopy(self.identity)
        identity['views'][1]['arm'] = identity['views'][0]['arm']
        with self.assertRaisesRegex(ValueError, 'duplicate contact arm'):
            h.map_contact_sensors(identity, self.names, self.paths)


class NativeAndRawGroupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        receipt = json.loads(CAMERA_RECEIPTS.read_text())['receipts'][0]
        cls.group_path = RAW/receipt['state']
        cls.group = json.loads(cls.group_path.read_text())
        cls.native_path = RAW/cls.group['fresh_native_snapshot']
        cls.native = {}
        with np.load(cls.native_path, allow_pickle=False) as data:
            for arm in h.ARM_KEYS:
                for old, new in [('q', 'q'), ('qd', 'qd'), ('root', 'root_xyzw'), ('root_vel', 'root_velocity')]:
                    cls.native[arm+'_native_'+new] = data[arm+'_'+old].copy()

    def test_first_real_group_macro_native_q_qd_binding_and_sha(self):
        group = self.group
        result = h.bind_parent_group(RAW, group['env_id'], group['step'], group['capture_kind'], self.native)
        self.assertEqual(result['status'], 'matched_parent_macro_native_all64')
        self.assertEqual(result['sha256'], h.sha256(self.group_path))
        self.assertEqual(result['cell_oracle_status'], 'pending_parent_integration')
        self.assertEqual(group['origin_world_m'], [-10., -14., 0.])

    def test_real_group_native_mismatch_hidden_env_is_rejected(self):
        native = {k: v.copy() for k, v in self.native.items()}
        native['U_R_native_q'][63, -1] += .1
        with self.assertRaisesRegex(ValueError, 'native state'):
            h.bind_parent_group(RAW, self.group['env_id'], 0, 'first_native_contact', native)

    def test_missing_group_remains_pending_not_pass(self):
        result = h.bind_parent_group(RAW, 0, 999999, 'missing_hand_test_group', self.native)
        self.assertEqual(result['status'], 'pending_parent_group')

    def test_real_group_actual_usd_optics_and_sphere_geometry(self):
        group = self.group
        view = group['views']['overview']
        K = h.intrinsic_from_usd(view['actual_usd_optics'])
        np.testing.assert_allclose(K, view['actual_intrinsic_matrix'], atol=2e-4, rtol=1e-6)
        margins = h.sphere_frustum_margins(group['sphere_centers_world_m'], group['sphere_radii_m'],
            view['actual_camera_to_world_row_matrix'], K, view['actual_clipping_range_m'])
        self.assertGreaterEqual(margins.min(), .05)
        np.testing.assert_allclose(margins.min(0), view['observed_sphere_frustum']['minimum_by_plane_m'], atol=2e-6, rtol=0)
        # Known stale SDK pose is never used as the actual USD transform.
        self.assertFalse(view['sdk_position_matches_actual'])
        self.assertFalse(np.allclose(view['sdk_position_world_m'], view['actual_camera_to_world_row_matrix'][3][:3]))

    def test_unsupported_nonzero_usd_offsets_fail_closed(self):
        optics = copy.deepcopy(self.group['views']['overview']['actual_usd_optics'])
        optics['horizontal_aperture_offset'] = .001
        with self.assertRaisesRegex(ValueError, 'offsets'):
            h.intrinsic_from_usd(optics)

    def test_bytes_detect_signed_zero_dtype_shape_and_each_native_field(self):
        before = {'q': np.array([0., 1.], dtype=np.float32)}
        after = {'q': np.array([-0., 1.], dtype=np.float32)}
        self.assertTrue(np.array_equal(before['q'], after['q']))
        with self.assertRaises(ValueError):
            h.assert_native_bitwise(before, after)
        with self.assertRaises(ValueError):
            h.assert_native_bitwise(before, {'q': before['q'].astype(np.float64)})
        with self.assertRaises(ValueError):
            h.assert_native_bitwise(before, {'q': before['q'].reshape(1, 2)})
        for key in self.native:
            after = {k: v.copy() for k, v in self.native.items()}
            after[key][63, -1] += .01
            with self.subTest(field=key), self.assertRaises(ValueError):
                h.assert_native_bitwise(self.native, after)

    def test_native_snapshot_copies_all64_targets_links_and_clock(self):
        # Getter-storage double only; comparison and snapshot implementation real.
        fields = {'get_dof_positions': (64, 2), 'get_dof_velocities': (64, 2),
            'get_root_transforms': (64, 7), 'get_root_velocities': (64, 6),
            'get_link_transforms': (64, 3, 7), 'get_dof_position_targets': (64, 2)}
        sources, arms = {}, {}
        for arm in h.ARM_KEYS:
            sources[arm] = {k: np.zeros(s, dtype=np.float32) for k, s in fields.items()}
            view = SimpleNamespace(**{k: (lambda value=v: value) for k, v in sources[arm].items()})
            arms[arm] = SimpleNamespace(root_physx_view=view, joint_names=['j0', 'j1'], body_names=['a', 'b', 'c'])
        env = SimpleNamespace(num_envs=64, _arms=arms, scene=SimpleNamespace(env_origins=np.zeros((64, 3))),
            sim=SimpleNamespace(current_time=.125, current_time_step_index=15))
        before = h.native_snapshot(env)
        for getter, native_key in [('get_link_transforms', 'native_link_transforms_xyzw'),
                                   ('get_dof_position_targets', 'native_position_targets')]:
            sources['F_L'][getter].flat[-1] = 1
            self.assertEqual(before['F_L_'+native_key].flat[-1], 0.)
            with self.assertRaisesRegex(ValueError, native_key):
                h.assert_native_bitwise(before, h.native_snapshot(env))
            sources['F_L'][getter].flat[-1] = 0
        env.sim.current_time_step_index += 1
        with self.assertRaisesRegex(ValueError, 'simulation_time_step_index'):
            h.assert_native_bitwise(before, h.native_snapshot(env))

    def test_module_contains_no_physics_step_or_articulation_writes(self):
        tree = ast.parse(Path(h.__file__).read_text())
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        forbidden = ('write_joint', 'write_root', 'set_joint', 'set_dof', 'set_root', 'set_link')
        self.assertFalse([name for name in calls if name.startswith(forbidden)])
        self.assertNotIn('step', calls)
        self.assertNotIn('forward', calls)
        self.assertNotIn('reset', calls)
        self.assertNotIn('AppLauncher', [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)])


class CameraContractTests(unittest.TestCase):
    def test_update_period_resolution_render_mode_and_camera_rows(self):
        camera = SimpleNamespace(image_shape=(720, 1280), cfg=SimpleNamespace(update_period=0.),
                                 frame=np.array([10]))
        h.validate_camera_contract(camera, 'PARTIAL_RENDERING')
        camera.cfg.update_period = .1
        with self.assertRaisesRegex(ValueError, 'update_period'):
            h.validate_camera_contract(camera, 'PARTIAL_RENDERING')
        camera.cfg.update_period = 0.
        with self.assertRaisesRegex(ValueError, 'does not render'):
            h.validate_camera_contract(camera, 'NO_RENDERING')
        camera.image_shape = (360, 640)
        with self.assertRaisesRegex(ValueError, '1280x720'):
            h.validate_camera_contract(camera, 'PARTIAL_RENDERING')
        camera.image_shape = (720, 1280)
        camera.frame = np.zeros(64)
        with self.assertRaisesRegex(ValueError, 'one shared'):
            h.validate_camera_contract(camera, 'PARTIAL_RENDERING')

    def test_stale_sensor_frame_is_rejected_without_advancing_time(self):
        h.assert_sensor_refresh(np.array([19]), np.array([20]))
        with self.assertRaisesRegex(ValueError, 'new frame'):
            h.assert_sensor_refresh(np.array([19]), np.array([19]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
