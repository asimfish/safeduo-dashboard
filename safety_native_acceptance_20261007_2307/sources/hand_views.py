"""Passive native hand closeups, supplementary to the parent's nine views.

API (inside the parent's held macro-post boundary, after its nine-view capture)::

    receipts = capture_hand_views(env, out, t, slots, kind,
                                  Path(out) / 'native_contact_identity.json')

Returns one group receipt per slot, each with 4 arms x 3 original 1280x720 PNGs.
No Isaac/torch imports or writes happen at module import. Runtime needs the
already initialized shared env._viz_cam. Only visibility, camera view poses,
env.sim.render(), and camera.update() are mutated. Caller must own the physics
hold; concurrent stepping is unsupported and detected by native byte checks.
The .06m link-center spheres plus .05m clearance are a conservative framing
proxy, NOT certification of full mesh coverage, silhouette or occlusion.
Cell/microstep oracle binding remains the parent's responsibility.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np

ARM_KEYS = ('F_L', 'F_R', 'U_L', 'U_R')
WIDTH, HEIGHT = 1280, 720
RADIUS_M, CLEARANCE_M = .06, .05
PLANE_NAMES = ('left', 'right', 'top', 'bottom', 'near', 'far')
LOCAL_DIRECTIONS = {
    'oblique_above': (1., -1., .8),
    'opposite_below': (-1., 1., -.8),
    'cross_above': (.2, 1., 1.4),
}
SCOPE = ('Conservative .06m spheres at every exact native hand sensor link center; '
         '.05m additional six-plane clearance. Full mesh, silhouette, occlusion, '
         'microstep contact-state imagery and hardware safety are not certified.')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().clone().cpu().numpy()
    result = np.array(value, copy=True)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite native/camera data')
    return result


def _hand_path(path, arm):
    # Same inventory rule as native_contacts.hand_path; matching below is EXACT.
    return '/' + arm + '/' in path and (
        'f2_hand/' in path or any(x in path.rsplit('/', 1)[-1] for x in
                                 ('thumb', 'index', 'middle', 'ring', 'little', 'pinky', 'wrist_3_link')))


def map_contact_sensors(identity, body_names, link_paths, environment_count=64):
    """Bijection from actual contact sensors to native PhysX link indices.

    Input metadata must come from env._arms[a].body_names and
    env._arms[a].root_physx_view.link_paths, in getter tensor order. Every
    environment and every hand link is checked, including unrendered slots.
    No suffix guessing, dropped sensors, duplicate names, or row reordering.
    """
    if identity.get('schema') != 'safeduo.native_hand_contact.v1':
        raise ValueError('unsupported contact identity schema')
    if identity.get('environment_count') != environment_count:
        raise ValueError('contact environment count mismatch')
    views = identity['views']
    if len(views) != 4 or {v['arm'] for v in views} != set(ARM_KEYS):
        raise ValueError('missing or duplicate contact arm')
    result = {}
    all_sensors = set()
    for arm in ARM_KEYS:
        names = list(body_names[arm])
        paths = link_paths[arm]
        if not names or len(set(names)) != len(names):
            raise ValueError('duplicate/empty body_names: ' + arm)
        if len(paths) != environment_count:
            raise ValueError('native link_paths environment count mismatch')
        exact = {}
        expected_hand = set()
        for slot, row in enumerate(paths):
            if len(row) != len(names):
                raise ValueError('link_paths/body_names length mismatch')
            prefix = f'/World/envs/env_{slot}/{arm}/'
            for index, path in enumerate(row):
                if not isinstance(path, str) or not path.startswith(prefix):
                    raise ValueError('native path does not match environment/arm row')
                if path.rsplit('/', 1)[-1] != names[index] or path in exact:
                    raise ValueError('native rigidpath/bodyname mismatch or duplicate')
                exact[path] = (slot, index)
                if _hand_path(path, arm):
                    expected_hand.add(path)
        view = next(v for v in views if v['arm'] == arm)
        sensors, env_ids = view['sensors'], view['env_ids']
        if len(sensors) != len(env_ids) or len(set(sensors)) != len(sensors):
            raise ValueError('duplicate sensors or sensor/env_ids length mismatch')
        if not sensors or set(sensors) != expected_hand:
            raise ValueError('unmatched or dropped hand sensors: ' + arm)
        if all_sensors.intersection(sensors):
            raise ValueError('sensor duplicated across arms')
        all_sensors.update(sensors)
        rows = [[] for _ in range(environment_count)]
        for sensor_index, (path, env_id) in enumerate(zip(sensors, env_ids)):
            slot, index = exact[path]
            if isinstance(env_id, bool) or not isinstance(env_id, int) or env_id != slot:
                raise ValueError('sensor env_id differs from exact native rigidpath')
            rows[slot].append(dict(sensor_index=sensor_index, rigid_path=path,
                                   body_name=names[index], body_index=index))
        signature = [(r['body_name'], r['body_index']) for r in sorted(rows[0], key=lambda r: r['body_index'])]
        for row in rows:
            if not row or sorted((r['body_name'], r['body_index']) for r in row) != sorted(signature):
                raise ValueError('hand coverage differs across environments')
            anchor = 'wrist_3_link' if arm.startswith('U') else (
                'left_hand_base' if arm.endswith('L') else 'right_hand_base')
            if sum(r['body_name'] == anchor for r in row) != 1:
                raise ValueError('exact native hand-root anchor missing')
        result[arm] = rows
    return result


def rotate_xyzw(quaternion, directions):
    """PhysX native XYZW quaternion; never reinterpret it as WXYZ."""
    q, v = np.asarray(quaternion, dtype=np.float64), np.asarray(directions, dtype=np.float64)
    if q.shape != (4,) or v.shape[-1:] != (3,) or not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError('invalid quaternion/directions')
    norm = np.linalg.norm(q)
    if abs(norm - 1.) > 1e-3:
        raise ValueError('native hand quaternion is not unit length')
    q = q / norm
    twice = 2 * np.cross(q[:3], v)
    return v + q[3] * twice + np.cross(q[:3], twice)


def intrinsic_from_usd(optics, width=WIDTH, height=HEIGHT):
    """Current parent's centered USD perspective optics; fail on offsets.

    IsaacLab's SDK offset convention differs from USD film-offset units. The
    existing camera has zero offsets; do not silently certify other optics.
    """
    f, h, v = (float(optics[k]) for k in ('focal_length', 'horizontal_aperture', 'vertical_aperture'))
    offsets = [float(optics[k]) for k in ('horizontal_aperture_offset', 'vertical_aperture_offset')]
    if not np.isfinite([f, h, v, *offsets]).all() or min(f, h, v, width, height) <= 0:
        raise ValueError('invalid USD optics')
    if any(offset != 0 for offset in offsets):
        raise ValueError('nonzero USD aperture offsets require independent renderer calibration')
    return np.array([[width*f/h, 0, width/2], [0, height*f/v, height/2], [0, 0, 1.]])


def _camera_coordinates(points_world, camera_to_world_row):
    points, world = np.asarray(points_world, dtype=np.float64), np.asarray(camera_to_world_row, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError('invalid hand link centers')
    if world.shape != (4, 4) or not np.isfinite(world).all():
        raise ValueError('invalid USD camera transform')
    if (not np.allclose(world[:, 3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(world[:3, :3] @ world[:3, :3].T, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(world[:3, :3]), 1., atol=1e-6, rtol=0)):
        raise ValueError('USD camera transform must be rigid and right handed')
    return (np.column_stack((points, np.ones(len(points)))) @ np.linalg.inv(world))[:, :3]


def _projection(K, clip, width, height):
    k = np.asarray(K, dtype=np.float64)
    clip = np.asarray(clip, dtype=np.float64)
    if k.shape != (3, 3) or clip.shape != (2,) or not np.isfinite(k).all() or not np.isfinite(clip).all():
        raise ValueError('invalid K/clipping range')
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    if (min(fx, fy, width, height) <= 0 or not (0 < cx < width and 0 < cy < height)
            or k[0, 1] != 0 or k[1, 0] != 0 or not np.array_equal(k[2], [0, 0, 1])
            or not 0 < clip[0] < clip[1]):
        raise ValueError('unsupported K/clipping range')
    return fx, fy, cx, cy, clip


def sphere_frustum_margins(points_world, radii, camera_to_world_row, K, clip,
                           width=WIDTH, height=HEIGHT):
    """Exact signed Euclidean sphere clearance to ALL SIX optical planes.

    USD uses row-vector camera-to-world, with +X right, +Y up, -Z forward.
    A center inside the pixel rectangle alone does not imply a sphere fits.
    """
    p = _camera_coordinates(points_world, camera_to_world_row)
    fx, fy, cx, cy, clip = _projection(K, clip, width, height)
    r = np.broadcast_to(np.asarray(radii, dtype=np.float64), (len(p),))
    if not np.isfinite(r).all() or np.any(r < 0):
        raise ValueError('invalid sphere radii')
    x, y, z = p[:, 0], p[:, 1], -p[:, 2]
    return np.column_stack(((fx*x+cx*z)/np.hypot(fx, cx),
                            ((width-cx)*z-fx*x)/np.hypot(fx, width-cx),
                            (cy*z-fy*y)/np.hypot(fy, cy),
                            ((height-cy)*z+fy*y)/np.hypot(fy, height-cy),
                            z-clip[0], clip[1]-z)) - r[:, None]


def fit_distance_delta(points_world, radii, camera_to_world_row, K, clip,
                       clearance=CLEARANCE_M, minimum_delta=-np.inf):
    """Solve a feasible optical-axis translation interval, including far plane.

    Positive delta moves the eye backward along its actual +Z axis. This
    analytic fit also detects impossible near/far intervals instead of
    indefinitely backing away and clipping the far plane.
    """
    if not np.isfinite(clearance) or clearance < 0 or np.isnan(minimum_delta):
        raise ValueError('invalid clearance/minimum delta')
    margins = sphere_frustum_margins(points_world, radii, camera_to_world_row, K, clip)
    fx, fy, cx, cy, _ = _projection(K, clip, WIDTH, HEIGHT)
    slopes = np.array([cx/np.hypot(fx, cx), (WIDTH-cx)/np.hypot(fx, WIDTH-cx),
                       cy/np.hypot(fy, cy), (HEIGHT-cy)/np.hypot(fy, HEIGHT-cy), 1.])
    lower = max(float(np.max((clearance-margins[:, :5])/slopes)), float(minimum_delta))
    upper = float(np.min(margins[:, 5]-clearance))
    if lower > upper:
        raise ValueError('no distance satisfies all six sphere-frustum planes')
    # Leave a small floating-point cushion, without exceeding the far bound.
    return lower + min(1e-4, (upper-lower)/2)


def native_snapshot(env):
    """Detached copies from getters, all joints/all links/all 64 environments."""
    if env.num_envs != 64:
        raise ValueError('hand acceptance requires all 64 native environments')
    result = dict(environment_origins=_array(env.scene.env_origins),
                  simulation_time_s=np.asarray(float(env.sim.current_time)),
                  simulation_time_step_index=np.asarray(int(env.sim.current_time_step_index)))
    if result['environment_origins'].shape != (64, 3):
        raise ValueError('environment origin shape')
    fields = {'native_q': 'get_dof_positions', 'native_qd': 'get_dof_velocities',
              'native_root_xyzw': 'get_root_transforms', 'native_root_velocity': 'get_root_velocities',
              'native_link_transforms_xyzw': 'get_link_transforms',
              'native_position_targets': 'get_dof_position_targets'}
    for arm in ARM_KEYS:
        actor = env._arms[arm]
        for name, getter in fields.items():
            value = _array(getattr(actor.root_physx_view, getter)())
            shape = (64, len(actor.joint_names))
            if name == 'native_root_xyzw':
                shape = (64, 7)
            elif name == 'native_root_velocity':
                shape = (64, 6)
            elif name == 'native_link_transforms_xyzw':
                shape = (64, len(actor.body_names), 7)
            if value.shape != shape:
                raise ValueError('native getter shape mismatch: ' + arm + '/' + name)
            result[arm+'_'+name] = value
    return result


def assert_native_bitwise(before, after):
    """Use dtype/shape/bytes, so even +0 to -0 changes fail."""
    if before.keys() != after.keys():
        raise ValueError('native snapshot keys changed')
    changed = [k for k in before if before[k].shape != after[k].shape
               or before[k].dtype != after[k].dtype or before[k].tobytes() != after[k].tobytes()]
    if changed:
        raise ValueError('render changed all64 native state: ' + ', '.join(changed))
    return {'all_64_bitwise_equal': True, 'comparison': 'dtype+shape+C-order bytes',
            'fields': list(before)}


def bind_parent_group(out, slot, t, kind, native):
    """Read-only optional macro-state binding; a missing parent is PENDING."""
    path = Path(out) / 'multiview' / kind / f'env_{slot:03d}' / f'step_{t:04d}_state.json'
    if not path.exists():
        return dict(status='pending_parent_group', cell_oracle_status='pending_parent_integration')
    state = json.loads(path.read_text())
    if (state['env_id'], state['step'], state['capture_kind']) != (slot, t, kind):
        raise ValueError('parent macro group identity mismatch')
    source = Path(out) / state['fresh_native_snapshot']
    if sha256(source) != state['fresh_native_before_sha256']:
        raise ValueError('parent native snapshot hash mismatch')
    with np.load(source, allow_pickle=False) as data:
        for arm in ARM_KEYS:
            for old, new in [('q', 'q'), ('qd', 'qd'), ('root', 'root_xyzw'), ('root_vel', 'root_velocity')]:
                assert_native_bitwise({arm: data[arm+'_'+old]}, {arm: native[arm+'_native_'+new]})
            indices = np.asarray(state['fresh_native_controlled_joint_indices'][arm], dtype=int)
            for name in ('q', 'qd'):
                if not np.array_equal(native[arm+'_native_'+name][slot, indices], state['arms'][arm][name]):
                    raise ValueError('parent macro controlled q/qd mismatch')
    return dict(status='matched_parent_macro_native_all64', path=str(path.relative_to(out)),
                sha256=sha256(path), state_time_s=state['state_time_s'],
                cell_oracle_status='pending_parent_integration',
                contact_scope='macro post state; microstep peak is not relabelled as image state')


def _read_camera(env, stage):
    from pxr import Usd, UsdGeom

    path = env._viz_cam.cfg.prim_path
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
        raise ValueError('actual shared USD camera prim missing')
    camera = UsdGeom.Camera(prim)
    if str(camera.GetProjectionAttr().Get()) != 'perspective':
        raise ValueError('actual camera is not perspective')
    if float(UsdGeom.GetStageMetersPerUnit(stage)) != 1.:
        raise ValueError('native meters require meter-scaled USD stage')
    optics = {key: float(getter().Get()) for key, getter in {
        'focal_length': camera.GetFocalLengthAttr,
        'horizontal_aperture': camera.GetHorizontalApertureAttr,
        'vertical_aperture': camera.GetVerticalApertureAttr,
        'horizontal_aperture_offset': camera.GetHorizontalApertureOffsetAttr,
        'vertical_aperture_offset': camera.GetVerticalApertureOffsetAttr}.items()}
    world = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64)
    K = intrinsic_from_usd(optics)
    sdk_K = _array(env._viz_cam.data.intrinsic_matrices[0])
    if not np.allclose(K, sdk_K, atol=2e-4, rtol=1e-6):
        raise ValueError('actual USD optics and SDK K disagree')
    return dict(prim_path=path, actual_pose_source='USD ComputeLocalToWorldTransform after camera operation',
                actual_camera_to_world_row_matrix=world.tolist(), actual_usd_optics=optics,
                actual_intrinsic_matrix=K.tolist(), sdk_intrinsic_matrix=sdk_K.tolist(),
                actual_clipping_range_m=list(map(float, camera.GetClippingRangeAttr().Get())),
                actual_clipping_source='USD clippingRange',
                K_source='actual centered USD focal length/apertures; SDK cross-check',
                image_size=[WIDTH, HEIGHT])


def _check_view_pose(camera, eye, target):
    world = np.asarray(camera['actual_camera_to_world_row_matrix'])
    direction = np.asarray(eye) - target
    direction = direction / np.linalg.norm(direction)
    if (not np.allclose(world[3, :3], eye, atol=2e-5, rtol=0)
            or not np.allclose(world[2, :3], direction, atol=2e-5, rtol=0)):
        raise ValueError('actual USD camera pose disagrees with requested eye/target')


def validate_camera_contract(camera, render_mode_name):
    if tuple(camera.image_shape) != (HEIGHT, WIDTH):
        raise ValueError('original camera image must be 1280x720')
    # In the installed IsaacLab, force_recompute only updates OUTDATED rows.
    # At dt=0 only update_period=0 guarantees a refresh on every held render.
    if float(camera.cfg.update_period) != 0.:
        raise ValueError('held-time camera refresh requires update_period=0')
    if render_mode_name not in ('PARTIAL_RENDERING', 'FULL_RENDERING'):
        raise ValueError('simulation mode does not render camera images')
    if _array(camera.frame).shape != (1,):
        raise ValueError('expected one shared world camera row')


def assert_sensor_refresh(before, after):
    if before.shape != (1,) or after.shape != (1,) or not np.array_equal(after, before+1):
        raise ValueError('camera sensor did not fetch exactly one new frame')


def _save_npz(path, values, out):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **values)
    return dict(path=str(path.relative_to(out)), sha256=sha256(path))


def _write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def capture_hand_views(env, out, t, slots, kind, contact_identity):
    """See module docstring. Must run synchronously at an actual macro-post hold.

    contact_identity: native_contact_identity.json path or parsed identity dict.
    Output is isolated under out/hand_views/kind/step_NNNN (exclusive creation).
    Existing nine-view files and sealed runs are never rewritten.
    """
    if not isinstance(t, int) or isinstance(t, bool) or t < 0:
        raise ValueError('nonnegative macro step required')
    if not isinstance(kind, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', kind):
        raise ValueError('unsafe capture kind')
    slots = list(slots)
    if (not slots or len(slots) != len(set(slots)) or
            any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) or not 0 <= s < 64 for s in slots)):
        raise ValueError('unique valid environment slots required')
    slots = [int(s) for s in slots]
    if isinstance(contact_identity, (str, Path)):
        identity_path = Path(contact_identity)
        identity_bytes = identity_path.read_bytes()
        identity = json.loads(identity_bytes)
        identity_source = dict(path=str(identity_path), sha256=hashlib.sha256(identity_bytes).hexdigest(),
                               hash_encoding='original file bytes')
    else:
        identity = contact_identity
        identity_source = dict(sha256=hashlib.sha256(json.dumps(identity, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest(), hash_encoding='canonical JSON dict')
    names = {a: list(env._arms[a].body_names) for a in ARM_KEYS}
    paths = {a: [list(p) for p in env._arms[a].root_physx_view.link_paths] for a in ARM_KEYS}
    mapping = map_contact_sensors(identity, names, paths)
    before = native_snapshot(env)
    parent = {s: bind_parent_group(out, s, t, kind, before) for s in slots}
    for binding in parent.values():
        if 'state_time_s' in binding and not np.isclose(binding['state_time_s'], (t+1)*env.step_dt, rtol=0, atol=1e-9):
            raise ValueError('parent logical macro time mismatch')
    validate_camera_contract(env._viz_cam, env.sim.render_mode.name)

    # Import runtime dependencies only after cheap validation; never start Isaac.
    import torch
    from PIL import Image
    from pxr import UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    out = Path(out)
    root = out / 'hand_views' / kind / f'step_{t:04d}'
    root.mkdir(parents=True, exist_ok=False)
    before_ref = _save_npz(root/'native_before_all64.npz', before, out)
    coverage = dict(source=identity_source, native_body_names=names,
                    native_link_paths=paths, mapping=mapping,
                    sensor_count=sum(len(v['sensors']) for v in identity['views']),
                    matched_sensor_count=sum(len(row) for a in mapping.values() for row in a),
                    unmatched=[], dropped=[], duplicates=[], all_64_exact_match=True)
    coverage_path = root/'identity_mapping.json'
    _write_json(coverage_path, coverage)
    coverage_ref = dict(path=str(coverage_path.relative_to(out)), sha256=sha256(coverage_path))
    visuals = [UsdGeom.Imageable(stage.GetPrimAtPath(f'/World/envs/env_{i}')) for i in range(64)]
    if any(not v.GetPrim().IsValid() for v in visuals):
        raise ValueError('environment visibility prim missing')
    saved_visibility = [(v.GetVisibilityAttr().HasAuthoredValueOpinion(), v.GetVisibilityAttr().Get()) for v in visuals]
    receipts = []
    try:
        for slot in slots:
            for index, visual in enumerate(visuals):
                visual.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if index == slot else UsdGeom.Tokens.invisible)
            slot_root = root/f'env_{slot:03d}'
            slot_root.mkdir()
            origin = before['environment_origins'][slot]
            selected = {}
            for arm in ARM_KEYS:
                rows = mapping[arm][slot]
                indices = [r['body_index'] for r in rows]
                selected[arm] = {k[len(arm)+1:]: v[slot].tolist() for k, v in before.items() if k.startswith(arm+'_')}
                selected[arm].update(joint_names=list(env._arms[arm].joint_names), body_names=names[arm],
                    hand_sensor_mapping=rows,
                    hand_link_positions_world_m=before[arm+'_native_link_transforms_xyzw'][slot, indices, :3].tolist())
            images = []
            for arm in ARM_KEYS:
                rows = mapping[arm][slot]
                indices = [r['body_index'] for r in rows]
                links = before[arm+'_native_link_transforms_xyzw'][slot]
                points = links[indices, :3].astype(np.float64)
                target = (points.min(0)+points.max(0))/2
                anchor_name = 'wrist_3_link' if arm.startswith('U') else ('left_hand_base' if arm.endswith('L') else 'right_hand_base')
                anchor = next(r for r in rows if r['body_name'] == anchor_name)
                quat = links[anchor['body_index'], 3:7]
                for view_name, local_direction in LOCAL_DIRECTIONS.items():
                    direction = rotate_xyzw(quat, local_direction)
                    direction /= np.linalg.norm(direction)
                    image_before = native_snapshot(env)
                    assert_native_bitwise(before, image_before)
                    distance = .5
                    for attempt in range(4):
                        eye = target+distance*direction
                        # Both are already WORLD coordinates; never add origins again.
                        env._viz_cam.set_world_poses_from_view(
                            torch.as_tensor(eye, device=env.device, dtype=torch.float32)[None],
                            torch.as_tensor(target, device=env.device, dtype=torch.float32)[None])
                        observed = _read_camera(env, stage)
                        _check_view_pose(observed, eye, target)
                        margins = sphere_frustum_margins(points, RADIUS_M,
                            observed['actual_camera_to_world_row_matrix'], observed['actual_intrinsic_matrix'],
                            observed['actual_clipping_range_m'])
                        if np.all(margins >= CLEARANCE_M):
                            break
                        distance += fit_distance_delta(points, RADIUS_M,
                            observed['actual_camera_to_world_row_matrix'], observed['actual_intrinsic_matrix'],
                            observed['actual_clipping_range_m'], minimum_delta=.01-distance)
                    else:
                        raise ValueError('actual USD six-plane distance fit did not converge')
                    env.sim.render()
                    # Sensor refresh at held physical time; no artificial camera time advance.
                    sensor_frame_before = _array(env._viz_cam.frame)
                    env._viz_cam.update(0., force_recompute=True)
                    sensor_frame_after = _array(env._viz_cam.frame)
                    assert_sensor_refresh(sensor_frame_before, sensor_frame_after)
                    observed = _read_camera(env, stage)
                    _check_view_pose(observed, eye, target)
                    margins = sphere_frustum_margins(points, RADIUS_M,
                        observed['actual_camera_to_world_row_matrix'], observed['actual_intrinsic_matrix'],
                        observed['actual_clipping_range_m'])
                    if not np.all(margins >= CLEARANCE_M):
                        raise ValueError('post-render actual USD camera clips hand spheres/clearance')
                    rgb = _array(env._viz_cam.data.output['rgb'][0])
                    if rgb.dtype != np.uint8 or rgb.shape not in [(HEIGHT, WIDTH, 3), (HEIGHT, WIDTH, 4)]:
                        raise ValueError('expected original 1280x720 uint8 RGB/RGBA; no rescale/crop allowed')
                    stem = f'{arm}_{view_name}'
                    path = slot_root/(stem+'.png')
                    # Preserve original channels and pixels, including alpha if supplied.
                    with path.open('xb') as stream:
                        Image.fromarray(rgb).save(stream, format='PNG')
                    after = native_snapshot(env)
                    invariant = assert_native_bitwise(image_before, after)
                    assert_native_bitwise(before, after)
                    after_ref = _save_npz(slot_root/(stem+'_native_after_all64.npz'), after, out)
                    record = dict(arm=arm, view=view_name, path=str(path.relative_to(out)), sha256=sha256(path),
                        resolution=[WIDTH, HEIGHT], channels=rgb.shape[2], raw_pixels_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                        camera=observed, target_world_m=target.tolist(), target_env_local_m=(target-origin).tolist(),
                        requested_eye_world_m=eye.tolist(), local_direction=list(local_direction),
                        world_direction=direction.tolist(), native_handroot_quat_xyzw=quat.tolist(),
                        handroot_sensor=anchor, distance_m=distance, fit_attempts=attempt+1,
                        sphere_radius_m=RADIUS_M, required_plane_clearance_m=CLEARANCE_M,
                        sphere_count=len(points), plane_names=list(PLANE_NAMES),
                        plane_margins_m=margins.tolist(), minimum_by_plane_m=margins.min(0).tolist(),
                        minimum_plane_margin_m=float(margins.min()), actual_usd_six_plane_spheres_contained=True,
                        native_before_all64=before_ref, native_after_all64=after_ref, native_invariant=invariant,
                        native_selected_all_four_hands=selected,
                        simulation_time_s=float(before['simulation_time_s']),
                        simulation_time_step_index=int(before['simulation_time_step_index']),
                        sensor_frame_before=int(sensor_frame_before[0]), sensor_frame_after=int(sensor_frame_after[0]),
                        sensor_update_dt_s=0., native_link_coordinate_frame='world; IsaacLab articulation subspace root /',
                        step=t, logical_macro_post_time_s=(t+1)*float(env.step_dt), scope=SCOPE)
                    images.append(record)
            state = dict(schema='safeduo.native_hand_closeups.v1', env_id=slot, step=t, capture_kind=kind,
                environment_count=64, images=images, image_count=12, visibility_only=True,
                native_invariant=assert_native_bitwise(before, native_snapshot(env)),
                native_before_all64=before_ref, native_selected_all_four_hands=selected,
                sensor_identity_mapping=coverage_ref, parent_macro_binding=parent[slot],
                cell_oracle_status='pending_parent_integration',
                origin_world_m=origin.tolist(), state_time_s=(t+1)*float(env.step_dt),
                simulation_time_s=float(before['simulation_time_s']),
                simulation_time_step_index=int(before['simulation_time_step_index']),
                source_sha256=sha256(__file__), scope=SCOPE)
            state_path = slot_root/'state.json'
            _write_json(state_path, state)
            receipts.append(dict(env_id=slot, step=t, capture_kind=kind, state=str(state_path.relative_to(out)),
                sha256=sha256(state_path), images=[{k: v[k] for k in ('arm', 'view', 'path', 'sha256')} for v in images],
                parent_macro_binding=parent[slot], cell_oracle_status='pending_parent_integration'))
    finally:
        for visual, (authored, value) in zip(visuals, saved_visibility):
            if authored:
                visual.GetVisibilityAttr().Set(value)
            else:
                visual.GetVisibilityAttr().Clear()
    final = native_snapshot(env)
    final_invariant = assert_native_bitwise(before, final)
    final_ref = _save_npz(root/'native_final_all64.npz', final, out)
    _write_json(root/'receipts.json', dict(schema='safeduo.native_hand_closeup_receipts.v1',
        groups=len(receipts), PNG=12*len(receipts), receipts=receipts,
        native_before_all64=before_ref, native_final_all64=final_ref,
        native_invariant=final_invariant, visibility_restored=True, scope=SCOPE))
    return receipts
