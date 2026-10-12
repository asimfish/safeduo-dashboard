"""Bind registered timestamps and controller dt to native integer-rate scene metadata."""
import math
def validate_native_clock(reg,scene_hz,runtime_dt_s):
    assert type(scene_hz) is int and scene_hz>0,'Native scene rate must be a positive integer'
    h=reg['physics_dt_s'];dt=reg['control_dt_s'];n=reg['physics_substeps_per_control']
    assert type(n) is int and n==16 and math.isfinite(runtime_dt_s)
    assert abs(runtime_dt_s-1/scene_hz)<=1e-12,'Runtime getter and USD scene Hz disagree'
    assert abs(h-runtime_dt_s)<=1e-12,'Registered dt differs from native integer-rate dt'
    assert abs(n*runtime_dt_s-dt)<=1e-12,'Control clock differs from native16-step clock'
    assert abs(dt-1/60)<=1e-12,'Control clock must be exact60Hz'
    assert reg['position_target_FIFO_controls']==6
    return dict(qualified=True,native_scene_Hz=scene_hz,native_getter_dt_s=runtime_dt_s,
                registered_physics_dt_s=h,control_dt_s=dt,physics_substeps_per_control=n,
                original_FIFO_controls=6,FIFO_s=6*dt,requires_physical_model_and_task_qualification=True,
                fullSystem0_accepted=False)
