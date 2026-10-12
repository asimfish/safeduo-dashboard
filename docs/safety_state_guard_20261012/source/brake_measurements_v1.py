"""Observe a native episode; a temporary low speed does not prove settling."""
import numpy as np


def settling_window(times, speeds, trigger_time, *, speed_cap=.02, dwell=.3):
    times=np.asarray(times,float);speeds=np.asarray(speeds,float)
    if times.ndim!=1 or speeds.shape!=times.shape or not len(times):
        raise ValueError('one speed for each physical timestamp is required')
    if not np.isfinite(times).all() or not np.isfinite(speeds).all() or (speeds<0).any():
        raise ValueError('invalid measurement')
    if (np.diff(times)<=0).any() or speed_cap<=0 or dwell<=0:
        raise ValueError('physical clock must advance')
    start=None
    for i,(t,v) in enumerate(zip(times,speeds)):
        if t<trigger_time or v>speed_cap:
            start=None
        elif start is None:
            start=i
        if start is not None and t-times[start]>=dwell-1e-12:
            return {'settled':True,'start_index':start,'confirmation_index':i,
                    'low_speed_started_s':float(times[start]),
                    'confirmed_s':float(t),'confirmation_latency_s':float(t-trigger_time),
                    'continuous_dwell_s':float(t-times[start]),
                    'speed_cap_rad_s':speed_cap}
    return {'settled':False,'start_index':None,'confirmation_index':None,
            'low_speed_started_s':None,'confirmed_s':None,'confirmation_latency_s':None,
            'continuous_dwell_s':0.,'speed_cap_rad_s':speed_cap}
