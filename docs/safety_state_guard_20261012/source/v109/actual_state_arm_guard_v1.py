"""Online simulation candidate after the real FIFO, not a safety certificate.

Uses actual arm q/qd, all74 measured limits, actual9021 proxy gaps and the six
pending commands. Forecast uses persistent measured velocity plus bounded future
target changes, anchored to current native proxy gaps. Hand motion and nonlinear
tracking/braking uncertainty are NOT certified by this model. Every native
microstep must still be audited against the unchanged original gates.
"""
from dataclasses import dataclass
import numpy as np

ARMS = ('F_L', 'F_R', 'U_L', 'U_R')


@dataclass(frozen=True)
class Decision:
    target: dict
    mode: str
    alpha: float
    forecast_minimum_m: float
    actual_minimum_m: float
    measured_max_speed_fraction: float
    queue_maximum_departure_rad: float
    prediction_qualified: bool = False


class ActualStateArmGuard:
    def __init__(self, gap_function, bounds, speed_caps, controlled_indices, retreat,
                 command_rate_rad_s=.25, backup_rate_rad_s=.15, reserve_m=.0005,
                 horizon_s=.12, control_dt_s=1./60., fifo_controls=6):
        self.gap_function=gap_function
        self.bounds={a:np.asarray(bounds[a],float) for a in ARMS}
        self.speed_caps={a:np.asarray(speed_caps[a],float) for a in ARMS}
        self.ids={a:np.asarray(controlled_indices[a],int) for a in ARMS}
        self.retreat={a:np.asarray(retreat[a],float) for a in ARMS}
        if (command_rate_rad_s<=0 or backup_rate_rad_s<=0 or reserve_m<0 or
                horizon_s < fifo_controls*control_dt_s or control_dt_s != 1./60. or fifo_controls != 6):
            raise ValueError('invalid registered clock/reserve')
        self.rate,self.backup_rate,self.reserve,self.horizon=command_rate_rad_s,backup_rate_rad_s,reserve_m,horizon_s
        self.dt,self.fifo=control_dt_s,fifo_controls
        for a in ARMS:
            n=len(self.speed_caps[a]);k=7 if a.startswith('F') else 6
            if (self.bounds[a].shape!=(n,2) or self.ids[a].shape!=(k,) or
                    self.retreat[a].shape!=(k,) or len(set(self.ids[a]))!=k or
                    np.any(self.ids[a]<0) or np.any(self.ids[a]>=n) or
                    np.any(self.speed_caps[a]<=0) or np.any(self.bounds[a][:,0]>=self.bounds[a][:,1])):
                raise ValueError('invalid original all74 limits and26 selection')
            if not all(np.isfinite(v).all() for v in (self.bounds[a],self.speed_caps[a],self.retreat[a])):
                raise ValueError('nonfinite original configuration')

    def decide(self, full_q, full_qd, applied_from_fifo, pending_fifo, actual_gaps):
        if len(pending_fifo)!=self.fifo:
            raise ValueError('exact pending FIFO6 required')
        q,v,applied={},{},{}
        measured_max_speed=0.; measured_bad=False;queue_departure=0.
        for a in ARMS:
            q[a],v[a]=np.asarray(full_q[a],float),np.asarray(full_qd[a],float)
            applied[a]=np.asarray(applied_from_fifo[a],float)
            n=len(self.speed_caps[a]);k=len(self.ids[a])
            if (q[a].shape!=(n,) or v[a].shape!=(n,) or applied[a].shape!=(k,) or
                    not all(np.isfinite(x).all() for x in (q[a],v[a],applied[a]))):
                raise ValueError('complete measured all74 state required')
            measured_max_speed=max(measured_max_speed,float((abs(v[a])/self.speed_caps[a]).max()))
            measured_bad |= bool((q[a]<self.bounds[a][:,0]-1e-5).any() or
                                 (q[a]>self.bounds[a][:,1]+1e-5).any() or
                                 (abs(v[a])>self.speed_caps[a]+1e-5).any())
            for queued in pending_fifo:
                z=np.asarray(queued[a],float)
                if z.shape!=(k,) or not np.isfinite(z).all():raise ValueError('invalid queued packet')
                queue_departure=max(queue_departure,float(abs(z-q[a][self.ids[a]]).max()))
        actual_gaps=np.asarray(actual_gaps,float)
        if actual_gaps.shape!=(9021,) or not np.isfinite(actual_gaps).all():
            raise ValueError('complete actual9021 gap vector required')
        actual_min=float(actual_gaps.min())
        root={a:q[a][self.ids[a]].copy() for a in ARMS}
        velocity={a:v[a][self.ids[a]].copy() for a in ARMS}
        source_now=np.asarray(self.gap_function({a:root[a][None] for a in ARMS}),float)
        if source_now.shape!=(1,9021) or not np.isfinite(source_now).all():
            raise ValueError('invalid source proxy')
        offset=actual_gaps-source_now[0]
        times=np.linspace(0.,self.horizon,9)
        candidates=[];modes=[];alphas=[];paths={a:[] for a in ARMS}
        for fraction in (1.,.5,.25,.125,.0625,0.):
            target={a:root[a]+fraction*np.clip(applied[a]-root[a],-self.rate*self.dt,self.rate*self.dt) for a in ARMS}
            candidates.append(target);modes.append('REQUEST' if fraction==1 else 'SCALED' if fraction else 'BRAKE');alphas.append(fraction)
        candidates.append({a:root[a]+np.clip(self.retreat[a]-root[a],-self.backup_rate*self.dt,self.backup_rate*self.dt) for a in ARMS})
        modes.append('RETREAT');alphas.append(-1.)
        # Each future FIFO packet is individually bounded by the same post-FIFO gate.
        # This is a diagnostic envelope, not a dynamical/stopping proof.
        for ci,target in enumerate(candidates):
            for a in ARMS:
                path=root[a][None]+times[:,None]*velocity[a][None]
                path+=times[:,None]/self.horizon*(target[a]-root[a])[None]
                if modes[ci] not in ('BRAKE','RETREAT'):
                    for ti in range(1,len(times)):
                        packet=pending_fifo[min(self.fifo-1,int(times[ti]/self.dt))][a]
                        future=np.clip(np.asarray(packet)-root[a],-self.rate*self.dt,self.rate*self.dt)*alphas[ci]
                        path[ti]+=(times[ti]/self.horizon)*future
                paths[a].append(path)
        predictions=np.asarray(self.gap_function({a:np.concatenate(paths[a]) for a in ARMS}),float)
        if predictions.shape!=(len(candidates)*9,9021) or not np.isfinite(predictions).all():
            raise ValueError('invalid full forecast')
        minima=(predictions+offset[None]).reshape(len(candidates),9,9021).min((1,2))
        accepted=None
        for ci,target in enumerate(candidates):
            interior=True
            for a in ARMS:
                b=self.bounds[a][self.ids[a]]
                interior &= bool((target[a]>=b[:,0]+.03).all() and (target[a]<=b[:,1]-.03).all())
            if interior and minima[ci]>=self.reserve and not measured_bad and actual_min>=0:
                accepted=ci;break
        if accepted is None:
            target=root;mode='UNQUALIFIED_BRAKE';alpha=0.;forecast=float(minima[5])
        else:
            target=candidates[accepted];mode=modes[accepted];alpha=alphas[accepted];forecast=float(minima[accepted])
        return Decision({a:target[a].astype(np.float32) for a in ARMS},mode,alpha,
                        forecast,actual_min,measured_max_speed,queue_departure)
