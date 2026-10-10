"""Independent every-microstep force, geometry, FIFO, and matched-pair audit."""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

H = Path(__file__).resolve().parent
ARMS = ('F_L', 'F_R', 'U_L', 'U_R')
REG = json.loads((H / 'REGISTRATION_V1.json').read_text())
ROOT = Path(REG['raw'])


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def write(p, d):
    p.write_text(json.dumps(d, indent=2) + '\n')


def audit(batch, method):
    root = ROOT / f'paired_batch{batch}_{method}_v8'
    protocol = json.loads((root / 'response_protocol.json').read_text())
    assert protocol['status'] == 'complete' and protocol['completed_steps'] == 480 and protocol['physics_events'] == 960
    p = load(root / 'resolved_native_parameters.npz')
    stream = load(root / 'response_stream.npz')
    identity = json.loads((root / 'native_contact_identity.json').read_text())
    receipt = json.loads((root / 'point_contact_receipts.json').read_text())
    assert receipt['status'] == 'complete' and receipt['physics_events'] == 960
    expected_owners = {x['rigid_owner'] for x in identity['inventory'] if x['collision_enabled'] and x['rigid_owner'] and any('/'+a+'/' in x['rigid_owner'] for a in ARMS)}
    observed_owners = {s for view in identity['views'] for s in view['sensors'] if '/env_0/' in s}
    assert expected_owners <= observed_owners and len(observed_owners) == 82
    force, hard, velocity, targets, event_order = [], [], [], [], []
    point_observations = 0
    peak_detail = None
    for chunk in receipt['chunks']:
        path = root / chunk['path']
        assert sha(path) == chunk['sha256']
        with np.load(path) as z:
            frames, subs = z['frame'], z['substep']
            n = len(frames)
            peak = np.zeros((n, 64))
            hard_max, velocity_max = np.zeros_like(peak), np.zeros_like(peak)
            applied = []
            for arm in ARMS:
                spec = next(v for v in identity['views'] if v['arm'] == arm)
                owners = np.asarray(spec['env_ids'])
                key = arm + '_points_'
                counts, starts = z[key+'counts'], z[key+'starts']
                sensor, partner, point = [z[key+k] for k in ['sensor_indices','partner_indices','point_indices']]
                offsets = z[key+'event_offsets']
                normal = z[key+'normal_forces'][:, 0].astype(float)
                assert np.isfinite(normal).all(), 'nonfinite native point force'
                stored = z[key+'partner_abs_normal_sum_N']
                for event in range(n):
                    lo, hi = offsets[event:event+2]
                    si, pi, ii = sensor[lo:hi], partner[lo:hi], point[lo:hi]
                    flat = si * counts.shape[-1] + pi
                    reconstructed = np.bincount(flat, minlength=counts[event].size).reshape(counts[event].shape)
                    assert np.array_equal(reconstructed, counts[event])
                    assert (ii >= starts[event, si, pi]).all()
                    assert (ii < starts[event, si, pi] + counts[event, si, pi]).all()
                    assert len(np.unique(ii)) == len(ii)
                    scalar = np.bincount(flat, weights=abs(normal[lo:hi]), minlength=counts[event].size).reshape(counts[event].shape)
                    assert np.array_equal(scalar, stored[event])
                    np.maximum.at(peak[event], owners, scalar.max(-1))
                    point_observations += int(hi-lo)
                    if peak_detail is None or scalar.max() > peak_detail['scalar_N']:
                        s, partner_index = np.unravel_index(scalar.argmax(), scalar.shape)
                        peak_detail = dict(arm=arm, macro=int(frames[event]), substep=int(subs[event]),
                            lane=int(owners[s]), scalar_N=float(scalar[s, partner_index]),
                            sensor=spec['sensors'][s], partner_index=int(partner_index))
                q, v = z[arm+'_native_q'], z[arm+'_native_qd']
                assert np.isfinite(q).all() and np.isfinite(v).all(), 'nonfinite native joint state'
                bounds = p[arm+'_hard_limits']
                hard_max = np.maximum(hard_max, np.maximum(bounds[None,...,0]-q, q-bounds[None,...,1]).max(-1))
                vmax = p[arm+'_max_velocity']
                velocity_max = np.maximum(velocity_max, (abs(v)-vmax[None]).max(-1))
                idx = p[arm+'_controlled_joint_indices']
                hand = np.ones(q.shape[-1], bool); hand[idx] = False
                actual_targets = z[arm+'_native_position_targets']
                assert np.isfinite(actual_targets).all(), 'nonfinite native position target'
                assert np.array_equal(actual_targets[...,hand], np.broadcast_to(p[arm+'_full_hold_target'][None,...,hand], actual_targets[...,hand].shape))
                applied.append(actual_targets[...,idx])
                assert np.array_equal(q[1::2], stream['post_'+arm+'_q'][frames[1::2]])
                assert np.array_equal(v[1::2], stream['post_'+arm+'_qd'][frames[1::2]])
            force.append(peak)
            hard.append(hard_max.clip(0))
            velocity.append(velocity_max.clip(0))
            targets.append(np.concatenate(applied, -1))
            event_order.extend(zip(frames.tolist(), subs.tolist()))
    assert event_order == [(t, s) for t in range(480) for s in range(2)]
    force, hard, velocity, targets = [np.concatenate(x) for x in [force, hard, velocity, targets]]
    assert np.array_equal(force.reshape(480,2,64).max(1), stream['scalar_normal_max_N'])
    assert np.array_equal(targets, np.repeat(stream['applied_target'], 2, axis=0))
    initial_arm_q = np.concatenate([p[a+'_initial_q'][:,p[a+'_controlled_joint_indices']] for a in ARMS], -1)
    assert np.array_equal(stream['applied_target'][:6], np.broadcast_to(initial_arm_q, (6,64,26)))
    assert np.isfinite(stream['issued_target']).all() and np.isfinite(stream['reference_target']).all()
    assert np.array_equal(stream['applied_target'][6:], stream['issued_target'][:-6])
    assert np.array_equal(stream['pre_pending'][0], np.broadcast_to(initial_arm_q, (6,64,26)))
    for t in range(1,480):
        assert np.array_equal(stream['pre_pending'][t,:-1], stream['pre_pending'][t-1,1:])
        assert np.array_equal(stream['pre_pending'][t,-1], stream['issued_target'][t-1])
    gr = json.loads((root / 'all_raw_geometry_receipts.json').read_text())
    assert gr['status'] == 'complete' and gr['raw_rows'] == 9021 and gr['physics_events'] == 960 and gr['exemptions'] == 0
    gaps, first_macro = [], 0
    for chunk in gr['chunks']:
        path = root / chunk['path']
        assert sha(path) == chunk['sha256']
        with np.load(path) as z:
            d = z['distances']
            assert d.shape == (chunk['events'],64,9021)
            assert np.isfinite(d).all(), 'nonfinite geometry invalidates scoring; never counted safe'
            assert int(z['first_macro']) == first_macro
            assert np.array_equal(z['global_input_id'], p['global_input_id'])
            gaps.append(d.min(-1))
            first_macro += len(d)//2
    gap = np.concatenate(gaps)
    assert gap.shape == (960,64)
    assert np.array_equal(gap.reshape(480,2,64), stream['substep_min_raw_gap_m'])
    initial = load(root / 'initial_geometry.npz')
    assert (initial['d'].min(-1) >= .0001).all()
    geo_bad, force_bad = gap.min(0)<0, force.max(0)>.1
    prefix = (gap[:12].min(0)>=0) & (force[:12].max(0)<=.1)
    controlled = np.concatenate([stream['post_'+a+'_q'][...,p[a+'_controlled_joint_indices']] for a in ARMS],-1)
    controlled = np.concatenate([initial_arm_q[None],controlled])
    steps = abs(np.diff(controlled,axis=0))
    paths = steps.sum((0,2))
    arms_moved = np.stack([steps[:,:,s].sum((0,2))>1e-4 for s in [slice(0,7),slice(7,14),slice(14,20),slice(20,26)]]).all(0)
    state_rows = []
    for lane in range(64):
        bad = np.flatnonzero((gap[:,lane]<0)|(force[:,lane]>.1))
        state_rows.append(dict(lane=lane,input_id=int(p['global_input_id'][lane]),cell=int(p['cell_id'][lane]),
            replay_prefix_qualified=bool(prefix[lane]),geometry_failure=bool(geo_bad[lane]),force_failure=bool(force_bad[lane]),
            primary_failure=bool(geo_bad[lane] or force_bad[lane]), minimum_raw_gap_m=float(gap[:,lane].min()),
            peak_all_arm_scalar_N=float(force[:,lane].max()),
            first_bad_macro=int(bad[0]//2) if len(bad) else None,first_bad_substep=int(bad[0]%2) if len(bad) else None,
            controlled_path_rad=float(paths[lane]),four_arms_moved=bool(arms_moved[lane]),
            supplementary_max_all74_hard_violation_rad=float(hard[:,lane].max()),
            supplementary_max_all74_velocity_exceedance_rad_s=float(velocity[:,lane].max())))
    result = dict(status='PASS_INDEPENDENT_ALL960_RAW_POINT_GEOMETRY_FIFO_ORACLE',batch=batch,method=method,
        points_observed=point_observations,rigid_owners_env0=82,all_point_counts_and_scalar_sums_exact=True,
        geometry_rows_per_microstep=9021,physics_events=960,all_FIFO6_targets_exact=True,
        all_hand_targets_held_exact=True,all_post_macro_native_q_qd_exact=True,states=state_rows,
        global_peak_point_detail=peak_detail,source_protocol_sha256=sha(root/'response_protocol.json'),
        source_stream_sha256=sha(root/'response_stream.npz'),analysis_source_sha256=sha(Path(__file__)),safety_acceptance=False)
    write(H/f'PAIRED_ORACLE_batch{batch}_{method}_V1.json',result)
    np.savez_compressed(H/f'PAIRED_METRICS_batch{batch}_{method}_V1.npz',
        global_input_id=p['global_input_id'],cell_id=p['cell_id'],gap_m=gap,force_N=force,
        hard_violation_rad=hard,velocity_exceedance_rad_s=velocity,controlled_q=controlled)
    print('AUDIT_PASS',batch,method,'prefix',int(prefix.sum()),'primary failures',int((geo_bad|force_bad).sum()),flush=True)


if __name__=="__main__":audit(int(sys.argv[1]),sys.argv[2])
