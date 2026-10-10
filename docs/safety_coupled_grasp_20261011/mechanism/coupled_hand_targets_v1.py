"""URDF affine hand forest: feasible motor intervals and consistent followers.

These bounds are intersections of original limits under original coupling.
They do not change the original hard or speed limits observed by an auditor.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Relation:
    slave: str
    master: str
    multiplier: float
    offset_rad: float

class CoupledHand:
    def __init__(self,names,limits,relations):
        self.names=tuple(names);self.limits=np.asarray(limits,dtype=float).copy()
        assert len(set(self.names))==len(self.names)
        assert self.limits.shape==(len(names),2) and np.isfinite(self.limits).all()
        assert (self.limits[:,1]>self.limits[:,0]).all()
        self.relations=tuple(relations);self.by_slave={r.slave:r for r in relations}
        assert len(self.by_slave)==len(relations)
        assert all(r.slave in self.names and r.master in self.names and r.slave!=r.master and np.isfinite([r.multiplier,r.offset_rad]).all() and r.multiplier!=0 for r in relations)
        self.motors=tuple(n for n in self.names if n not in self.by_slave)
        assert self.motors
        self.affine={}
        def visit(n,seen):
            if n in self.affine:return self.affine[n]
            assert n not in seen,'Cyclic mimic relation'
            if n not in self.by_slave:value=(n,1.,0.)
            else:
                r=self.by_slave[n];root,s,b=visit(r.master,seen|{n});value=(root,r.multiplier*s,r.multiplier*b+r.offset_rad)
            self.affine[n]=value;return value
        for n in self.names:visit(n,set())

    def feasible_motor_intervals(self,reserve_rad=0.):
        assert np.isfinite(reserve_rad) and reserve_rad>=0
        bounds=np.tile([-np.inf,np.inf],(len(self.motors),1))
        for i,n in enumerate(self.names):
            root,s,b=self.affine[n];j=self.motors.index(root)
            lo,hi=self.limits[i]+np.array([reserve_rad,-reserve_rad])
            assert lo<=hi,'Reserve consumes original joint interval'
            transformed=sorted([(lo-b)/s,(hi-b)/s])
            bounds[j,0]=max(bounds[j,0],transformed[0]);bounds[j,1]=min(bounds[j,1],transformed[1])
        assert np.isfinite(bounds).all() and (bounds[:,0]<=bounds[:,1]).all(),'No feasible coupled motor interval'
        return bounds

    def expand(self,motor_q):
        q=np.asarray(motor_q,dtype=float)
        assert q.shape[-1]==len(self.motors) and np.isfinite(q).all()
        return np.stack([s*q[...,self.motors.index(root)]+b for root,s,b in (self.affine[n] for n in self.names)],axis=-1)

    def reference_endpoints(self,preferred_q,reserve_rad=.03):
        preferred=np.asarray(preferred_q,dtype=float);assert preferred.shape==(len(self.names),) and np.isfinite(preferred).all()
        bounds=self.feasible_motor_intervals(reserve_rad)
        current=np.array([preferred[self.names.index(n)] for n in self.motors])
        opened=np.clip(current,bounds[:,0],bounds[:,1])
        far=np.where(abs(bounds[:,1]-opened)>=abs(bounds[:,0]-opened),bounds[:,1],bounds[:,0])
        return self.expand(opened),self.expand(far)
