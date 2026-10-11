"""Reject incompatible phase-admission clocks before starting native simulation."""
import math


def validate_settle_clock(reg):
    physics = reg['physics_dt_s']
    control = reg['control_dt_s']
    substeps = reg['physics_substeps_per_control']
    gate = reg['grasp_settle_gate']
    assert type(substeps) is int and substeps > 0
    assert math.isfinite(physics) and physics > 0
    assert math.isfinite(control) and abs(physics * substeps - control) <= 1e-12
    assert math.isfinite(gate['physics_dt_s']) and abs(gate['physics_dt_s'] - physics) <= 1e-12, 'Phase admission clock must equal actual native physics sampling clock'
    assert gate['position_error_max_rad'] == .005
    assert gate['speed_max_rad_s'] == .02
    assert gate['applied_error_max_rad'] == 1e-6
    assert gate['dwell_s'] == .3 and gate['timeout_s'] == 8.
    return True
