"""Independent closed-cell and actual USD/pixel oracle for 12 native hand views."""
from pathlib import Path
import json, sys, hashlib, datetime, copy
import numpy as np
from PIL import Image

H = Path(__file__).resolve().parent
ARMS = ('F_L', 'F_R', 'U_L', 'U_R')
VIEWS = ('oblique_above', 'opposite_below', 'cross_above')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def exact(a, b, why):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), why

def referenced(root, ref, cache):
    p = root/ref['path']
    assert sha(p) == ref['sha256'], str(p)
    if str(p) not in cache:
        cache[str(p)] = load(p)
    return cache[str(p)]

def unchanged(a, b):
    assert set(a) == set(b)
    for k in a:
        exact(a[k], b[k], 'render changed '+k)

def bound(before, packet, cell, t):
    exact(before['environment_origins'], packet['post_environment_origins'], 'origins')
    q = []
    for arm in ARMS:
        for field in ('q', 'qd', 'root_xyzw', 'root_velocity'):
            exact(before[arm+'_native_'+field], packet['post_'+arm+'_native_'+field], 'native post '+field)
        idx = packet['post_'+arm+'_controlled_joint_indices']
        q.append(before[arm+'_native_q'][:, idx])
        exact(before[arm+'_native_position_targets'][:, idx], cell['actuator_target'][t, :, sum([7 if a.startswith('F') else 6 for a in ARMS[:ARMS.index(arm)]]):sum([7 if a.startswith('F') else 6 for a in ARMS[:ARMS.index(arm)+1]])], 'actual position targets')
    exact(np.concatenate(q, -1), cell['q'][t], 'closed cell q')

def planes(points, camera, radius):
    W = np.asarray(camera['actual_camera_to_world_row_matrix'], float)
    K = np.asarray(camera['actual_intrinsic_matrix'], float)
    near, far = camera['actual_clipping_range_m']
    assert W.shape == (4, 4) and K.shape == (3, 3) and np.isfinite(W).all() and np.isfinite(K).all()
    assert np.allclose(W[:3, :3]@W[:3, :3].T, np.eye(3), atol=1e-6, rtol=0)
    assert 0 < near < far and camera['image_size'] == [1280, 720]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    assert fx > 0 and fy > 0 and 0 < cx < 1280 and 0 < cy < 720
    p = np.c_[points, np.ones(len(points))]@np.linalg.inv(W)
    x, y, z = p[:, 0], p[:, 1], -p[:, 2]
    return np.column_stack([(fx*x+cx*z)/np.hypot(fx, cx), ((1280-cx)*z-fx*x)/np.hypot(fx, 1280-cx), (cy*z-fy*y)/np.hypot(fy, cy), ((720-cy)*z+fy*y)/np.hypot(fy, 720-cy), z-near, far-z])-radius

def mapping_check(mapping, identity):
    assert mapping['sensor_count'] == mapping['matched_sensor_count'] == 3328
    assert not mapping['unmatched'] and not mapping['dropped'] and not mapping['duplicates']
    for arm in ARMS:
        sensors = next(v['sensors'] for v in identity['views'] if v['arm'] == arm)
        rows = mapping['mapping'][arm]
        assert len(rows) == 64 and len(sensors) == 64*13
        seen = set()
        for e, hand in enumerate(rows):
            assert len(hand) == 13
            bodies = set()
            for row in hand:
                si, bi = row['sensor_index'], row['body_index']
                assert si not in seen and bi not in bodies
                seen.add(si); bodies.add(bi)
                assert sensors[si] == row['rigid_path'] and row['rigid_path'].startswith(f'/World/envs/env_{e}/{arm}/')
                assert mapping['native_body_names'][arm][bi] == row['body_name'] == Path(row['rigid_path']).name
                assert Path(mapping['native_link_paths'][arm][e][bi]).name == row['body_name']
        assert seen == set(range(832))

def audit(root):
    root = Path(root)
    assert json.loads((root/'visual_protocol.json').read_text())['status'] == 'complete'
    cell = load(root/'cell_001.npz')
    overview = json.loads((root/'camera_receipts.json').read_text())
    hands = json.loads((root/'hand_camera_receipts.json').read_text())
    identity = json.loads((root/'native_contact_identity.json').read_text())
    native = json.loads((root/'native_receipts.json').read_text())
    needed = {r['step'] for r in hands['receipts']}
    packets = {}
    for ch in native['chunks']:
        if not any(ch['start'] <= t < ch['stop'] for t in needed):
            continue
        z = referenced(root, ch, {})
        for t in needed:
            if ch['start'] <= t < ch['stop']:
                packets[t] = {k: v[t-ch['start']] for k, v in z.items()}
    key = lambda x: (x['step'], x['env_id'], x['capture_kind'])
    assert {key(x) for x in hands['receipts']} == {key(x) for x in overview['receipts']}
    assert len(hands['receipts']) == hands['groups'] == overview['groups'] and hands['PNG'] == 12*hands['groups']
    minimum, count, caught = np.inf, 0, []
    tested = False
    for receipt in hands['receipts']:
        sp = root/receipt['state']; assert sha(sp) == receipt['sha256']
        s = json.loads(sp.read_text()); e, t = s['env_id'], s['step']; assert key(s) == key(receipt)
        cache = {}; before = referenced(root, s['native_before_all64'], cache)
        bound(before, packets[t], cell, t)
        macro = s['parent_macro_binding']; mp = root/macro['path']; assert sha(mp) == macro['sha256']
        parent = json.loads(mp.read_text()); assert key(parent) == key(s) and s['state_time_s'] == parent['state_time_s']
        principal = load(root/parent['fresh_native_snapshot'])
        for arm in ARMS:
            for old, new in [('q','q'),('qd','qd'),('root','root_xyzw'),('root_vel','root_velocity')]:
                exact(principal[arm+'_'+old], before[arm+'_native_'+new], 'principal native binding')
        for k in ('simulation_time_s', 'simulation_time_step_index'):
            assert s[k] == before[k].item()
        ref = s['sensor_identity_mapping']; p = root/ref['path']; assert sha(p) == ref['sha256']
        mapping = json.loads(p.read_text()); assert mapping['source']['sha256'] == sha(root/'native_contact_identity.json')
        mapping_check(mapping, identity)
        group = json.loads(sp.parent.parent.joinpath('receipts.json').read_text())
        assert group['visibility_restored']; unchanged(before, referenced(root, group['native_final_all64'], cache))
        assert len(s['images']) == s['image_count'] == 12
        assert {(im['arm'], im['view']) for im in s['images']} == {(a,v) for a in ARMS for v in VIEWS}
        for im in s['images']:
            arm = im['arm']; rows = mapping['mapping'][arm][e]; indices = [v['body_index'] for v in rows]
            points = before[arm+'_native_link_transforms_xyzw'][e, indices, :3].astype(float)
            assert im['sphere_count'] == 13 and im['sphere_radius_m'] == .06 and im['required_plane_clearance_m'] == .05
            margins = planes(points, im['camera'], .06)
            assert margins.min() >= .05, 'sphere clipped'
            assert np.allclose(margins, im['plane_margins_m'], atol=1e-7, rtol=0)
            assert np.allclose(margins.min(0), im['minimum_by_plane_m'], atol=1e-7, rtol=0)
            assert abs(float(margins.min())-im['minimum_plane_margin_m']) <= 1e-7
            minimum = min(minimum, float(margins.min()))
            unchanged(before, referenced(root, im['native_after_all64'], cache))
            assert im['native_before_all64'] == s['native_before_all64']
            for a in ARMS:
                for field in ('q','qd','root_xyzw','root_velocity','link_transforms_xyzw','position_targets'):
                    observed = np.asarray(im['native_selected_all_four_hands'][a]['native_'+field], dtype=before[a+'_native_'+field].dtype)
                    exact(observed, before[a+'_native_'+field][e], 'selected native JSON')
            assert im['step'] == t and im['logical_macro_post_time_s'] == s['state_time_s']
            assert im['simulation_time_s'] == before['simulation_time_s'].item() and im['simulation_time_step_index'] == before['simulation_time_step_index'].item()
            assert im['sensor_frame_after'] == im['sensor_frame_before']+1 and im['sensor_update_dt_s'] == 0
            p = root/im['path']; assert sha(p) == im['sha256']
            with Image.open(p) as image:
                image.load(); pixels = np.asarray(image)
                assert image.size == (1280,720) and pixels.dtype == np.uint8 and pixels.shape == (720,1280,im['channels']) and im['channels'] in (3,4)
                assert hashlib.sha256(pixels.tobytes()).hexdigest() == im['raw_pixels_sha256'] and pixels.std() > 1
            count += 1
        if not tested:
            for field in ('F_L_native_q','U_R_native_root_xyzw','simulation_time_s','U_L_native_position_targets'):
                bad = {k: v.copy() for k,v in before.items()}; bad[field].flat[0] += .125
                try:
                    unchanged(before,bad)
                except AssertionError:
                    caught.append(field)
                else:
                    raise AssertionError('negative control escaped')
            bad = copy.deepcopy(mapping); bad['mapping']['F_L'][0].pop()
            try:
                mapping_check(bad,identity)
            except AssertionError:
                caught.append('dropped_sensor')
            else:
                raise AssertionError('mapping negative escaped')
            assert not np.allclose(planes(points, im['camera'], 0), margins, atol=1e-7, rtol=0)
            caught.append('ignored_sphere_radius'); tested=True
    assert count == hands['PNG']
    return dict(status='PASS_CLOSED_CELL_NATIVE_HAND_CAMERA_BINDING', root=str(root), frames=len(cell['q']), windows=64, groups=hands['groups'], hand_PNG=count, total_PNG=count+overview['PNG'], minimum_hand_sphere_frustum_clearance_m=minimum, all64_native_bitwise_render_invariant=True, exact_contact_sensor_bijection=3328, negative_controls_caught=caught, cell_sha256=sha(root/'cell_001.npz'), source_sha256=sha(__file__), full_mesh_certified=False, occlusion_certified=False, hardware_certified=False, utc=datetime.datetime.now(datetime.timezone.utc).isoformat())

if __name__ == '__main__':
    result = audit(sys.argv[1])
    with (H/(sys.argv[2]+'.json')).open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
