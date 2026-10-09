"""Independent raw point bincount oracle, without importing the recorder helper."""
import json,sys,hashlib
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent
reg = json.loads((H / 'DYNAMIC_REGISTRATION.json').read_text())
root = Path(sys.argv[1]) if Path(sys.argv[1]).is_absolute() else Path(reg['raw']) / sys.argv[1]
output_name = root.parent.name+'_constructor_probe' if root.name=='constructor_probe' else root.name
receipt = json.loads((root / 'point_contact_receipts.json').read_text())
steps=receipt['control_steps']
assert steps in (1,8,480)
constructor = sys.argv[1].endswith("/constructor_probe")
assert (steps == 1) == constructor
all_clock, all_indices, all_frames, all_substeps = [], [], [], []
identity = json.loads((root / 'native_contact_identity.json').read_text())
views = identity['views'] if 'views' in identity else identity['identities']
lane_peak, partners, total_points = [], [], 0
for chunk in receipt['chunks']:
    assert hashlib.sha256((root/chunk['path']).read_bytes()).hexdigest()==chunk['sha256']
    with np.load(root / chunk['path'], allow_pickle=False) as z:
        frames = z['frame']; substeps = z['substep']
        all_frames.append(frames); all_substeps.append(substeps)
        all_clock.append(z['simulation_time_s']); all_indices.append(z['simulation_time_step_index'])
        env_peak = np.zeros((len(frames), 64))
        for a in ('F_L', 'F_R', 'U_L', 'U_R'):
            spec = next(x for x in views if x['arm'] == a)
            env_ids = np.asarray(spec['env_ids'])
            prefix = a + '_points_'
            force = z[prefix + 'normal_forces'][:, 0].astype(float)
            sensor, partner = z[prefix + 'sensor_indices'], z[prefix + 'partner_indices']
            points = z[prefix + 'point_indices']; offsets = z[prefix + 'event_offsets']
            counts, starts = z[prefix + 'counts'], z[prefix + 'starts']
            stored = z[prefix + 'partner_abs_normal_sum_N']
            for e in range(len(frames)):
                lo, hi = offsets[e:e+2]; total_points += int(hi-lo)
                si, pi, ii = sensor[lo:hi], partner[lo:hi], points[lo:hi]
                flat = si * counts.shape[-1] + pi
                reconstructed_counts = np.bincount(flat, minlength=counts.shape[-2]*counts.shape[-1]).reshape(counts[e].shape)
                assert np.array_equal(reconstructed_counts, counts[e])
                assert np.all(ii >= starts[e, si, pi]) and np.all(ii < starts[e, si, pi] + counts[e, si, pi])
                assert len(np.unique(ii)) == len(ii)
                scalar = np.bincount(flat, weights=abs(force[lo:hi]), minlength=stored.shape[-2]*stored.shape[-1]).reshape(stored[e].shape)
                assert np.array_equal(scalar, stored[e])
                for lane in range(64):
                    owned = scalar[env_ids == lane]
                    env_peak[e, lane] = max(env_peak[e, lane], owned.max())
                if scalar.max() > .1:
                    si_peak, pi_peak = np.unravel_index(scalar.argmax(), scalar.shape)
                    partners.append(dict(arm=a, macro=int(frames[e]), substep=int(substeps[e]),
                        lane=int(env_ids[si_peak]), sensor_index=int(si_peak), partner_index=int(pi_peak),
                        scalar_N=float(scalar.max()), valid_points=int(counts[e, si_peak, pi_peak])))
        lane_peak.append(env_peak)
peak = np.concatenate(lane_peak)
frames=np.concatenate(all_frames); subs=np.concatenate(all_substeps)
clock=np.concatenate(all_clock); indices=np.concatenate(all_indices)
assert len(peak)==2*steps and np.array_equal(subs,np.tile([0,1],steps))
assert np.array_equal(frames,np.full(2,-1) if constructor else np.repeat(np.arange(steps),2))
assert np.all(np.diff(indices)==1)
assert np.allclose(np.diff(clock),.008333333767950535,atol=1e-10,rtol=0)
if not constructor:
    with np.load(root / 'response_stream.npz') as z:
        assert np.array_equal(peak.reshape(steps, 2, 64).max(1), z['scalar_normal_max_N'])
np.savez_compressed(H/('MICRO_PEAK_'+output_name+'.npz'),scalar_N=peak,frame=frames,substep=subs,simulation_time_s=clock,simulation_time_step_index=indices)
maximum = max(partners, key=lambda x:x['scalar_N'], default=None)
expected_owners={p['rigid_owner'] for p in identity['inventory'] if p['collision_enabled'] and p['rigid_owner'] and any('/'+a+'/' in p['rigid_owner'] for a in ('F_L','F_R','U_L','U_R'))}
observed_owners={p for spec in views for p in spec['sensors'] if '/env_0/' in p}
assert expected_owners <= observed_owners
(H / ('FORCE_ORACLE_'+output_name+'.json')).write_text(json.dumps(dict(status='PASS_RAW_SCALAR_POINT_FORCE_ALL_REGISTERED_MICROSTEPS',
    point_receipt_sha256=hashlib.sha256((root/'point_contact_receipts.json').read_bytes()).hexdigest(),
    events=2*steps, lanes=64, arms=4, valid_point_observations=total_points,
    all_enabled_arm_collider_owners_covered=True, covered_owners_env0=len(observed_owners), enabled_owners_env0=len(expected_owners), raw_native_point_owner_counts_exact=True, original_indices_inside_owned_intervals=True,
    every_scalar_summary_exact=True, macro_peak_exact=not constructor, native_clock_and_microstep_order_exact=True, constructor_explicit_two_step_probe=constructor, maximum=maximum,
    safety_acceptance=False, observations_above_gate=partners), indent=2) + '\n')
print('PASS', maximum, flush=True)
