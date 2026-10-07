"""Sum native scalar normal magnitudes per sensor/filter; no vector cancellation."""
import numpy as np

def scalar_contacts(cv, dt, capacity):
    raw=cv.get_contact_data(dt)
    force=raw[0].detach().clone().cpu().numpy().reshape(-1)
    count=raw[-2].detach().clone().cpu().numpy()
    start=raw[-1].detach().clone().cpu().numpy()
    assert (count>=0).all()
    assert int(count.sum())<capacity
    assert np.all((count==0)|((start>=0)&(start+count<=capacity)))
    summed=np.zeros(count.shape,np.float32);max_valid=0.;valid_force=np.zeros(capacity,np.float32)
    for idx in zip(*np.nonzero(count)):
        first=int(start[idx]);n=int(count[idx]);values=force[first:first+n];valid_force[first:first+n]=values;assert np.isfinite(values).all();summed[idx]=np.abs(values).sum(dtype=np.float64);max_valid=max(max_valid,float(np.abs(values).max()))
    return summed,count,max_valid,valid_force,start
