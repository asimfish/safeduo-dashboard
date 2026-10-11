"""Prospective development-distribution comparison; no physical safety inference."""
from pathlib import Path
import datetime
import hashlib
import itertools
import json
import numpy as np
from scipy.spatial import cKDTree

H = Path(__file__).resolve().parent
P = H / 'full_motor_operating_domain_v2'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pairs(x):
    assert x.ndim == 2 and x.shape[1] == 50
    assert np.isfinite(x).all() and x.min() >= -1e-12 and x.max() <= 1 + 1e-12
    b = np.minimum(7, np.floor(np.clip(x, 0, 1) * 8).astype(int))
    c = np.corrcoef(x, rowvar=False)
    rows = []
    for i, j in itertools.combinations(range(50), 2):
        counts = np.bincount(8 * b[:, i] + b[:, j], minlength=64)
        rows.append(dict(left_index=i, right_index=j, occupied_bins=int(np.count_nonzero(counts)),
                         minimum_count=int(counts.min()), correlation=float(c[i, j])))
    return dict(proposals=len(x), pairs=rows, minimum_pair_bin_occupancy=min(r['occupied_bins'] / 64 for r in rows),
                minimum_pair_cell_count=min(r['minimum_count'] for r in rows),
                maximum_absolute_pair_correlation=max(abs(r['correlation']) for r in rows),
                minimum_marginal_span=float(np.ptp(x, axis=0).min()))


def velocity_dependence(v):
    upper = np.triu_indices(50, 1)
    signed = np.corrcoef(v, rowvar=False)[upper]
    squared = np.corrcoef(v * v, rowvar=False)[upper]
    return dict(mean_signed_correlation=float(signed.mean()), max_absolute_signed_correlation=float(abs(signed).max()),
                mean_squared_velocity_correlation=float(squared.mean()),
                max_absolute_squared_velocity_correlation=float(abs(squared).max()),
                squared_velocity_correlations=squared.tolist(),
                interpretation='Magnitude correlations expose shared speed strata; low empirical correlation alone does not prove statistical independence.')


def main():
    registration = P / 'JOINT_SAMPLING_SUPPORT_REGISTRATION_V2.json'
    target = P / 'JOINT_SAMPLING_SUPPORT_AUDIT_V2.json'
    assert not registration.exists() and not target.exists()
    files = [H / 'full_motor_operating_domain_v1' / 'DEVELOPMENT_FULL50_PROPOSALS_V1.npz',
             P / 'DEVELOPMENT_FULL50_PROPOSALS_V2.npz']
    for file, version in zip(files, [1, 2]):
        report = json.loads((file.parent / f'FULL_MOTOR_DOMAIN_V{version}.json').read_text())
        assert sha(file) == report['proposal_SHA256']
    registration.write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={str(f): sha(f) for f in [*files, Path(__file__)]},
        pair_grid=[8, 8], pure_v2_rows=32000, velocity_magnitude_comparison=True,
        probe_seed=734195820113, probe_count=128, diagnostic_probes_unexecuted_development_only=True,
        native_trajectories_added=0, formal_holdout_count=0, fullSystem0_accepted=False), indent=2) + '\n')
    with np.load(files[0], allow_pickle=False) as old, np.load(files[1], allow_pickle=False) as new:
        np.testing.assert_array_equal(old['independent_names'], new['independent_names'])
        for key in ['independent_lower_rad', 'independent_upper_rad', 'independent_speed_cap_rad_s']:
            np.testing.assert_array_equal(old[key], new[key])
        mask = new['proposal_stratum'] == 'pure_iid_position_velocity_target'
        assert mask.sum() == 32000 and len(mask) == 40000
        lower, upper, cap = [new[k] for k in ['independent_lower_rad', 'independent_upper_rad', 'independent_speed_cap_rad_s']]
        q = (new['independent_q'] - lower) / (upper - lower)
        v = new['independent_qd'] / cap
        t = (new['independent_target'] - lower) / (upper - lower)
        mixed = new['mixed_velocity_scale']
        assert mixed.shape == (4000, 50) and set(np.unique(mixed)) == {0., .01, .1, 1.}
        assert np.unique(mixed, axis=0).shape[0] == 4000
        distance, nearest = cKDTree(q).query(np.random.default_rng(734195820113).uniform(size=(128, 50)), p=np.inf, workers=1)
        result = dict(status='CLOSED_DEVELOPMENT_FULL50_V2_SUPPORT_AND_MAGNITUDE_DEPENDENCE_COMPARISON',
            positions=pairs(q[mask]), velocities=pairs((v[mask] + 1) / 2), commands=pairs(t[mask]),
            previous_v1_uniform_with_shared_row_scales=velocity_dependence(old['independent_qd'][:16000] / cap),
            new_v2_pure_independent_velocities=velocity_dependence(v[mask]),
            new_v2_mixed_per_motor_scales=velocity_dependence(v[new['proposal_stratum'] == 'mixed_independent_motor_speed_scales']),
            per_motor_mixed_scale_counts=[[int(np.sum(mixed[:, i] == scale)) for scale in [0., .01, .1, 1.]] for i in range(50)],
            unexecuted_development_probe_distances=distance.tolist(), nearest_proposals=nearest.tolist(),
            nearest_probe_median=float(np.median(distance)), within_25percent_each_coordinate=int(np.sum(distance <= .25)),
            normalized_50D_union_upper_bounds=[dict(radius=r, upper_bound=min(1., 40000 * (2*r)**50)) for r in [.1, .25]],
            safe_or_reachable_volume_not_measured=True, object_noise_stale_fault_strata_not_exercised=True,
            producer_actual_outer_exit=None, producer_receipt_unknown=True, native_trajectories_added=0,
            formal_holdout_count=0, fullSystem0_accepted=False,
            source_sha256={str(f): sha(f) for f in [*files, Path(__file__), registration, P / 'INDEPENDENT_PROPOSAL_ORACLE_V2.json']})
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(result['status'], 'squared_velocity_mean_correlation',
          result['previous_v1_uniform_with_shared_row_scales']['mean_squared_velocity_correlation'],
          result['new_v2_pure_independent_velocities']['mean_squared_velocity_correlation'],
          'nearest_50D_probe_median', result['nearest_probe_median'], 'NATIVE_0_FORMAL_0', flush=True)


if __name__ == '__main__':
    main()
