"""Finite recorded-episode gate. Does not certify a safe workspace or all motion.

Trace is an observer contract; qualifying capture custody must be established by a
separate native auditor. CPU fixtures and metadata flags cannot establish it.
"""
import numpy as np

GATES = {'hard_excess_rad': (0., 1e-5), 'speed_excess_rad_s': (0., 1e-5),
         'minimum_proxy_gap_m': (0., np.inf),
         'mimic_position_error_rad': (0., 1e-4),
         'mimic_velocity_error_rad_s': (0., 1e-4),
         'unintended_contact_force_N': (0., .1),
         'payload_hand_force_N': (0., 30.)}


def judge(trace, domain, planned_duration_s, coverage_contract=None, physics_hz=960):
    """Safety and mobility stay separate. Never treats an early prefix as completion.

    Native runner must emit initial state index0, then every original microstep.
    Coverage requirements must be predeclared against a qualified reachable domain,
    not chosen from outcomes. Without one, broad-motion acceptance remains pending.
    """
    if (not np.isfinite(planned_duration_s) or planned_duration_s <= 0 or physics_hz != 960 or
            abs(planned_duration_s * physics_hz - round(planned_duration_s * physics_hz)) > 1e-8):
        raise ValueError('invalid planned clock')
    required = ['sample_index', 'time_s', 'actual_q', 'actual_qd', 'mount_xyz',
                'issued_q', 'applied_q', 'held'] + list(GATES)
    if set(required) - set(trace):
        raise ValueError('missing measured fields: '+str(sorted(set(required)-set(trace))))
    a = {k: np.asarray(trace[k]) for k in required}
    n = len(a['time_s'])
    shapes = {'sample_index': (n,), 'time_s': (n,), 'actual_q': (n, 50),
              'actual_qd': (n, 50), 'mount_xyz': (n, 4, 3),
              'issued_q': (n, 50), 'applied_q': (n, 50), 'held': (n,)}
    shapes.update({k: (n,) for k in GATES})
    if n < 2 or any(a[k].shape != shape for k, shape in shapes.items()):
        raise ValueError('invalid full50 trace shape')
    if (a['sample_index'].dtype.kind not in 'iu' or a['held'].dtype.kind != 'b' or
            not all(np.isfinite(v).all() for v in a.values())):
        raise ValueError('invalid measured values')
    if any(np.any(a[k] < 0.) for k in GATES if k != 'minimum_proxy_gap_m'):
        raise ValueError('magnitudes cannot be negative')
    continuous = bool(np.array_equal(a['sample_index'], np.arange(n)) and
                      np.allclose(a['time_s'], np.arange(n)/physics_hz, rtol=0., atol=1e-9))
    expected = int(round(planned_duration_s*physics_hz))+1
    complete = continuous and n == expected
    passed = {k: bool(np.all((a[k] >= lo) & (a[k] <= hi))) for k, (lo, hi) in GATES.items()}
    passed['observed_full50_bounds'] = bool(np.all(a['actual_q'] >= domain.lower-1e-5) and
                                          np.all(a['actual_q'] <= domain.upper+1e-5))
    passed['observed_full50_speeds'] = bool(np.all(np.abs(a['actual_qd']) <= domain.cap+1e-5))
    first_bad = np.flatnonzero(np.logical_or.reduce([
        (a[k] < lo) | (a[k] > hi) for k, (lo, hi) in GATES.items()] + [
        ((a['actual_q'] < domain.lower-1e-5) | (a['actual_q'] > domain.upper+1e-5)).any(1),
        (np.abs(a['actual_qd']) > domain.cap+1e-5).any(1)]))
    span = domain.upper-domain.lower
    measured_span = np.ptp(a['actual_q'], axis=0)/span
    path = np.abs(np.diff(a['actual_q'], axis=0)).sum(0)
    mount_path = np.linalg.norm(np.diff(a['mount_xyz'], axis=0), axis=2).sum(0)
    target_span = np.ptp(a['issued_q'], axis=0)/span
    # Useful observed movement threshold is a reporting definition, not safety limit.
    moving = (np.abs(a['actual_qd']) >= .01).any(1)
    coverage_pass = None
    if coverage_contract is not None:
        req = np.asarray(coverage_contract['minimum_span_fraction'], dtype=float)
        mount_req = np.asarray(coverage_contract['minimum_mount_path_m'], dtype=float)
        if (req.shape != (50,) or mount_req.shape != (4,) or
                not np.isfinite(req).all() or not np.isfinite(mount_req).all() or
                np.any(req <= 0.) or np.any(req > 1.) or np.any(mount_req <= 0.) or
                not coverage_contract.get('predeclared_contract_sha256') or
                not coverage_contract.get('qualified_reachable_domain_sha256')):
            raise ValueError('invalid predeclared coverage contract')
        coverage_pass = bool(complete and np.all(measured_span >= req) and
                             np.all(mount_path >= mount_req))
    return {'continuous_clock': continuous, 'planned_duration_s': float(planned_duration_s),
            'observed_duration_s': float(a['time_s'][-1]), 'complete_episode': complete,
            'recorded_original_gates_pass': bool(all(passed.values())), 'gate_results': passed,
            'first_bad_index': int(first_bad[0]) if len(first_bad) else None,
            'actual_span_fraction_per_motor': measured_span.tolist(),
            'issued_span_fraction_per_motor': target_span.tolist(),
            'actual_path_rad_per_motor': path.tolist(), 'mount_path_m_per_arm': mount_path.tolist(),
            'hold_fraction': float(a['held'].mean()),
            'observed_moving_fraction_speed_above_0_01_rad_s': float(moving.mean()),
            'maximum_actual_vs_applied_target_error_rad': float(np.abs(a['actual_q']-a['applied_q']).max()),
            'actual_coverage_contract_pass': coverage_pass,
            'finite_recorded_long_motion_candidate_pass': bool(complete and all(passed.values()) and coverage_pass is True),
            'native_capture_custody_qualified_here': False,
            'learned_system0_qualified_here': False, 'hardware_qualified_here': False,
            'all_arbitrary_motion_safety_proven': False}
