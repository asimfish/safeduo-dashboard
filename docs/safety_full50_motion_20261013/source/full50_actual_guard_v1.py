"""Full50 intent filter over measured74 state, FIFO and original hand geometry.

The forecast is a development heuristic. Captured original physical gates and
q/qd relations are the outcome oracle; no empirical stopping certificate yet.
"""
from dataclasses import dataclass
import numpy as np

ARMS=('F_L','F_R','U_L','U_R')


@dataclass(frozen=True)
class Decision:
    target:dict
    mode:str
    alpha:float
    forecast_minimum_m:float
    actual_minimum_m:float
    prediction_qualified:bool=False


class Full50ActualGuard:
    def __init__(self,geometry,parameters,rootids,*,rate=.1,acceleration=.5,reserve_m=.0005,horizon_s=.12):
        self.geometry=geometry;self.rootids={a:np.asarray(rootids[a],int) for a in ARMS}
        self.bounds={a:parameters[a+'_hard_limits'][0].astype(float) for a in ARMS}
        self.caps={a:parameters[a+'_max_velocity'][0].astype(float) for a in ARMS}
        self.maps={a:parameters[a+'_full_motor_velocity_mapping'].astype(float) for a in ARMS}
        self.dt=1/60.;self.rate=rate;self.acceleration=acceleration;self.reserve=reserve_m;self.horizon=horizon_s
        if not (0<rate and 0<acceleration and reserve_m>=0 and horizon_s>=6*self.dt):raise ValueError('invalid filter configuration')
        if sum(len(i) for i in self.rootids.values())!=50:raise ValueError('full50 original roots required')
        for a in ARMS:
            n=len(self.bounds[a]);k=len(self.rootids[a])
            if self.maps[a].shape!=(n,k) or len(set(self.rootids[a]))!=k:raise ValueError('original root mapping required')
            np.testing.assert_array_equal(self.maps[a][self.rootids[a]],np.eye(k))

    def decide(self,q,qd,request,pending,actual_gaps,previous,previous_velocity):
        if len(pending)!=6:raise ValueError('exact FIFO6 required')
        state={};velocity={};root={};prior={};priorv={};bad=False
        for a in ARMS:
            n=len(self.bounds[a]);k=len(self.rootids[a]);state[a]=np.asarray(q[a],float);velocity[a]=np.asarray(qd[a],float)
            prior[a]=np.asarray(previous[a],float);priorv[a]=np.asarray(previous_velocity[a],float)
            for x,shape in [(state[a],(n,)),(velocity[a],(n,)),(prior[a],(k,)),(priorv[a],(k,)),(np.asarray(request[a]),(k,))]:
                if x.shape!=shape or not np.isfinite(x).all():raise ValueError('finite all74 state and full50 command required')
            for packet in pending:
                x=np.asarray(packet[a])
                if x.shape!=(k,) or not np.isfinite(x).all():raise ValueError('invalid full50 pending packet')
            root[a]=state[a][self.rootids[a]]
            bad |= bool((state[a]<self.bounds[a][:,0]-1e-5).any() or (state[a]>self.bounds[a][:,1]+1e-5).any() or (abs(velocity[a])>self.caps[a]+1e-5).any())
        gaps=np.asarray(actual_gaps,float)
        if gaps.shape!=(9021,) or not np.isfinite(gaps).all():raise ValueError('all9021 native gaps required')
        actual_min=float(gaps.min())
        source=self.geometry.gaps({a:state[a][None] for a in ARMS})[0]
        offset=gaps-source
        times=np.linspace(0,self.horizon,9);choices=[];modes=[];alphas=[];paths={a:[] for a in ARMS}
        for fraction in (1.,.5,.25,.125,.0625,0.):
            target={}
            for a in ARMS:
                goal=prior[a]+fraction*(np.asarray(request[a])-prior[a]) if fraction else root[a]
                wanted=np.clip((goal-prior[a])/self.dt,-self.rate,self.rate)
                speed=np.clip(wanted,priorv[a]-self.acceleration*self.dt,priorv[a]+self.acceleration*self.dt)
                target[a]=prior[a]+np.clip(speed,-self.rate,self.rate)*self.dt
                # Preserve actual follower coordinates/residuals, including their
                # measured velocity, and propagate root command changes through
                # the ORIGINAL native full-motor velocity map.
                path=state[a][None]+times[:,None]*velocity[a][None]
                path+=(times/self.horizon)[:,None]*((target[a]-root[a])@self.maps[a].T)[None]
                if fraction:
                    for i,t in enumerate(times[1:],1):
                        queued=np.asarray(pending[min(5,int(t/self.dt))][a])
                        delta=np.clip(queued-root[a],-self.rate*self.dt,self.rate*self.dt)*fraction
                        path[i]+=(t/self.horizon)*(delta@self.maps[a].T)
                paths[a].append(path)
            choices.append(target);modes.append('REQUEST' if fraction==1 else 'SCALED' if fraction else 'BRAKE');alphas.append(fraction)
        predicted={a:np.concatenate(paths[a]) for a in ARMS}
        forecast=self.geometry.gaps(predicted)
        if forecast.shape!=(54,9021) or not np.isfinite(forecast).all():raise ValueError('invalid full74 forecast')
        minima=(forecast+offset).reshape(6,9,9021).min(axis=(1,2))
        chosen=None
        for i,target in enumerate(choices):
            interior=True
            for a in ARMS:
                b=self.bounds[a][self.rootids[a]]
                interior &= bool((target[a]>=b[:,0]+.03).all() and (target[a]<=b[:,1]-.03).all())
                p=paths[a][i]
                interior &= bool((p>=self.bounds[a][:,0]).all() and (p<=self.bounds[a][:,1]).all())
            if interior and minima[i]>=self.reserve and not bad and actual_min>=0:chosen=i;break
        i=5 if chosen is None else chosen
        return Decision({a:choices[i][a].astype(np.float32) for a in ARMS},'UNQUALIFIED_BRAKE' if chosen is None else modes[i],0. if chosen is None else alphas[i],float(minima[i]),actual_min)
