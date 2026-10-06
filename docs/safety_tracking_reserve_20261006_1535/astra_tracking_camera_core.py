"""Independent CPU camera receipt mathematics; never a substitute for vision."""
import numpy as np

from astra_tracking_score_core import require

ARMS = ('F_L', 'F_R', 'U_L', 'U_R')
DOFS = (7, 7, 6, 6)
VIEWS = ('overview', 'front', 'reverse', 'f_pair', 'f_opposite_low',
         'f_opposite_high', 'u_pair', 'u_opposite_low', 'u_opposite_high')
PLANES = ('left', 'right', 'top', 'bottom', 'near', 'far')


def exact(a, b, label):
    require(np.array_equal(a, b), label)


def compact(values):
    arrays = [np.asarray(values[a], dtype=np.float32) for a in ARMS]
    require(all(v.shape == (n,) for v, n in zip(arrays, DOFS)), 'compact arm ordering/dimensions')
    require(all(np.isfinite(v).all() for v in arrays), 'nonfinite compact state')
    return np.concatenate(arrays)


def frustum_planes(world_centers, radii, camera_to_world, intrinsic, clipping,
                   width=1280, height=720):
    """Signed metric sphere clearance to normalized camera halfspaces.

    USD uses row-vector camera-to-world transforms and camera forward is -z.
    Construct normals explicitly; subtract each sphere's radius after normalizing.
    Containment uses >0. No physical safety tolerance is introduced.
    """
    centers, radii = np.asarray(world_centers, float), np.asarray(radii, float)
    W, K, clip = (np.asarray(v, float) for v in (camera_to_world, intrinsic, clipping))
    require(centers.ndim == 2 and centers.shape[1] == 3 and radii.shape == centers.shape[:1], 'sphere shape')
    require(W.shape == (4, 4) and K.shape == (3, 3) and clip.shape == (2,), 'camera matrix shape')
    require(all(np.isfinite(a).all() for a in (centers, radii, W, K, clip)), 'nonfinite camera input')
    require((radii >= 0).all() and 0 <= clip[0] < clip[1], 'invalid radius/clipping')
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    require(fx > 0 and fy > 0 and 0 < cx < width and 0 < cy < height, 'invalid intrinsics')
    normals = np.array([[fx, 0, -cx], [-fx, 0, -(width-cx)],
                        [0, -fy, -cy], [0, fy, -(height-cy)],
                        [0, 0, -1], [0, 0, 1]], float)
    offsets = np.array([0, 0, 0, 0, -clip[0], clip[1]], float)
    camera = np.column_stack((centers, np.ones(len(centers)))) @ np.linalg.inv(W)
    norms = np.linalg.norm(normals, axis=1)
    margins = (camera[:, :3] @ normals.T + offsets) / norms - radii[:, None]
    return margins


def select_groups(selection, receipts, mode):
    require(mode in selection['modes'], 'unknown selected mode')
    chosen = []
    for point in selection['scheduled_groups']:
        matches = [r for r in receipts if r['capture_kind'] == 'scheduled'
                   and r['env_id'] == point['env'] and r['step'] == point['step']]
        require(len(matches) == 1, 'preselected scheduled group missing/duplicated')
        chosen.append(matches[0])
    failures = [r for r in receipts if r['capture_kind'] == 'first_failure']
    if failures:
        chosen.append(min(failures, key=lambda r: (r['step'], r['env_id'])))
    return chosen


def native_state_binding(state, before, after, dense):
    """Bind every native before/after array and selected post-state to its own run."""
    e, t = state['env_id'], state['step']
    steps, envs, joints = dense['q'].shape
    require(joints == 26 and 0 <= e < envs and 0 <= t < steps, 'camera state identity outside dense run')
    keys = {a+'_'+k for a in ARMS for k in ('q', 'qd', 'root', 'root_vel')}
    require(set(before) == keys and set(after) == keys, 'native snapshot field set')
    for name in sorted(keys):
        a, b = before[name], after[name]
        require(a.dtype == np.float32 and a.ndim == 2 and a.shape[0] == envs
                and b.shape == a.shape and b.dtype == a.dtype, 'native shape/dtype: '+name)
        require(np.isfinite(a).all() and np.isfinite(b).all(), 'nonfinite native: '+name)
        exact(a, b, 'render changed native all-env state: '+name)
    all_q, all_v = [], []
    for arm, dofs in zip(ARMS, DOFS):
        idx = np.asarray(state['fresh_native_controlled_joint_indices'][arm])
        require(idx.shape == (dofs,) and idx.dtype.kind in 'iu' and len(np.unique(idx)) == dofs,
                'controlled joint identity')
        n = before[arm+'_q'].shape[1]
        require((idx >= 0).all() and (idx < n).all(), 'controlled joint index range')
        require(len(state['fresh_native_all_joint_names'][arm]) == n, 'native joint name count')
        for key in ('q', 'qd', 'root', 'root_vel'):
            a = before[arm+'_'+key]
            exact(a[e], np.asarray(state['fresh_native_selected'][arm][key], dtype=a.dtype),
                  'selected fresh native state: '+arm+'/'+key)
        q = np.asarray(state['arms'][arm]['q'], np.float32)
        v = np.asarray(state['arms'][arm]['qd'], np.float32)
        exact(before[arm+'_q'][e, idx], q, 'controlled native q')
        exact(before[arm+'_qd'][e, idx], v, 'controlled native qd')
        all_q.append(q); all_v.append(v)
    q, v = np.concatenate(all_q), np.concatenate(all_v)
    exact(q, dense['q'][t, e], 'own dense post q')
    exact(compact(state['controller_target']), dense['controller_target'][t, e], 'current issued target')
    exact(compact(state['actuator_target']), dense['actuator_target'][t, e], 'actual applied target')
    pre_queue = dense['pre_pending_actuator_targets'][t, :, e]
    exact(compact(state['actuator_target']), pre_queue[0], 'actual applied target is pre FIFO head')
    pending = np.stack([compact(x) for x in state['pending_actuator_targets']])
    expected_post_queue = np.concatenate((pre_queue[1:], dense['controller_target'][t, e][None]))
    exact(pending, expected_post_queue, 'post FIFO shift and append')
    if t+1 < steps:
        exact(v, dense['pre_qd_compact'][t+1, e], 'post velocity equals next pre velocity')
        exact(pending, dense['pre_pending_actuator_targets'][t+1, :, e], 'post FIFO equals next pre FIFO')
    return dict(native_all_env_fields_bound=len(keys), native_all_env_count=envs,
        native_q_and_qd_exact=True, actual_issued_applied_and_post_fifo_exact=True,
        post_qd_next_pre_available=t+1 < steps, final_post_qd_only_native_when_no_next_pre=t+1 == steps,
        q=q.tolist(), qd=v.tolist(), controller_target=compact(state['controller_target']).tolist(),
        applied_target=compact(state['actuator_target']).tolist(), post_pending=pending.tolist())
