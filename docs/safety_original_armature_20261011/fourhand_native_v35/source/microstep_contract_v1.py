"""Strict complete-control clock and event contracts for numerical refinement."""
import math
import numpy as np

def microsteps(reg):
    n=reg.get('physics_substeps_per_control',2)
    assert type(n) is int and n>0
    physics=reg['physics_dt_s'];control=reg['control_dt_s']
    assert math.isfinite(physics) and math.isfinite(control) and physics>0 and control>0
    assert abs(n*physics-control)<=1e-12
    return n

def constant_control_targets(target,n):
    x=np.asarray(target);assert len(x)%n==0
    grouped=x.reshape((len(x)//n,n,*x.shape[1:]))
    assert np.array_equal(grouped,np.repeat(grouped[:,0:1],n,axis=1))

def event_indices(frame,substep,n):
    frame=np.asarray(frame);substep=np.asarray(substep)
    assert frame.shape==substep.shape and np.issubdtype(frame.dtype,np.integer) and np.issubdtype(substep.dtype,np.integer)
    assert (frame>=0).all() and ((substep>=0)&(substep<n)).all()
    return frame*n+substep

def complete_event_layout(frame,substep,controls,n):
    assert type(controls) is int and controls>0
    np.testing.assert_array_equal(event_indices(frame,substep,n),np.arange(controls*n))

def control_fifo_targets(applied,issued,initial,n,delay_controls):
    assert type(delay_controls) is int and delay_controls>0
    applied=np.asarray(applied);issued=np.asarray(issued)
    assert applied.shape==issued.shape
    constant_control_targets(applied,n);constant_control_targets(issued,n)
    delay=n*delay_controls;first=min(delay,len(applied))
    np.testing.assert_array_equal(applied[:first],np.broadcast_to(initial,applied[:first].shape))
    if len(applied)>delay:np.testing.assert_array_equal(applied[delay:],issued[:-delay])
