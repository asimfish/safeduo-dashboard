"""Keep the native control clock distinct from the stored trajectory grid."""
import math

def control_clock(actual_dt_s,registered_dt_s,trajectory_dt_s):
    for value in (actual_dt_s,registered_dt_s,trajectory_dt_s):
        if not math.isfinite(value) or value<=0:raise ValueError('invalid clock')
    if abs(actual_dt_s-registered_dt_s)>1e-12:raise ValueError('native control clock differs from registration')
    return float(actual_dt_s)
