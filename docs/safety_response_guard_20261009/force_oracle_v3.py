"""Independent raw point bincount oracle, without importing the recorder helper."""
import json
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent
reg = json.loads((H / 'REGISTRATION_V2.json').read_text())
root = Path(reg['raw']) / 'multirow_pilot_v3'
receipt = json.loads((root / 'point_contact_receipts.json').read_text())
identity = json.loads((root / 'native_contact_identity.json').read_text())
views = identity['views'] if 'views' in identity else identity['identities']
lane_peak, partners, total_points = [], [], 0
for chunk in receipt['chunks']:
    with np.load(root / chunk['path'], allow_pickle=False) as z:
        frames = z['frame']; substeps = z['substep']
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
with np.load(root / 'response_stream.npz') as z:
    assert np.array_equal(peak.reshape(64, 2, 64).max(1), z['scalar_normal_max_N'])
maximum = max(partners, key=lambda x:x['scalar_N'], default=None)
(H / 'FORCE_ORACLE_RESULT_V3.json').write_text(json.dumps(dict(status='PASS_RAW_SCALAR_BINCOUT_ALL128_PHYSICS_EVENTS',
    events=128, lanes=64, arms=4, valid_point_observations=total_points,
    raw_native_point_owner_counts_exact=True, original_indices_inside_owned_intervals=True,
    every_scalar_summary_exact=True, macro_peak_exact=True, maximum=maximum,
    safety_acceptance=False, observations_above_gate=partners), indent=2) + '\n')
print('PASS', maximum, flush=True)
