"""Contact-driven task intents; native dynamics and independent success judge remain required."""
from dataclasses import dataclass
import numpy as np

ARMS=('F_L','F_R','U_L','U_R')
FAMILIES={'pick_place':1,'pair_colift':2,'four_shared_colift':4}
SECONDS={'CLOSE':.25,'LIFT':2.,'CARRY':.5,'PLACE':.5,'RELEASE':.5,'RETREAT':1.}


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    family: str
    object_id: str
    roles: tuple
    initial_position: tuple
    goal_position: tuple
    size_m: tuple
    mass_kg: float

    def validate(self):
        if self.family not in FAMILIES or len(self.roles)!=FAMILIES[self.family]:
            raise ValueError('unsupported family or incorrect holder count')
        if len(set(self.roles))!=len(self.roles) or any(a not in ARMS for a in self.roles):
            raise ValueError('unique original arm roles required')
        for vector in [self.initial_position,self.goal_position,self.size_m]:
            if len(vector)!=3 or not np.isfinite(vector).all():raise ValueError('invalid geometry')
        if min(self.size_m)<=0 or not np.isfinite(self.mass_kg) or self.mass_kg<=0:
            raise ValueError('positive actual object size and mass required')
        if np.linalg.norm(np.asarray(self.goal_position)[:2]-self.initial_position[:2])<.2:
            raise ValueError('original carry acceptance requires at least200mm')


class PayloadTask:
    def __init__(self,spec,*,max_age_s=.025,phase_timeout_s=30.):
        spec.validate();self.spec=spec;self.ids=[ARMS.index(a) for a in spec.roles]
        self.phase='CLOSE';self.last_time=None;self.last_seq=-1;self.phase_start=None
        self.window_start=None;self.release_mounts=None;self.transitions=[];self.failure=None;self.contact_lost_start=None
        self.max_age=max_age_s;self.timeout=phase_timeout_s

    def _transition(self,phase,t):
        self.transitions.append({'from':self.phase,'to':phase,'time_s':float(t)})
        self.phase=phase;self.phase_start=t;self.window_start=None

    def _fail(self,reason,t):
        self.failure=reason;self._transition('FAILED',t)

    def update(self,o):
        if self.phase in ['DONE','FAILED']:return self.intent()
        required={'object_id','sequence','time_s','now_s','position','quaternion_wxyz','linear_velocity',
            'angular_velocity','size_m','mass_kg','hand_force_N','opposed','support_N',
            'hand_open_error_rad','hand_speed_rad_s','mounts_m','conservative_mu','physical_observation_valid'}
        t=float(o.get('time_s',0))
        if not required.issubset(o):self._fail('missing_native_observation',t);return self.intent()
        shapes={'position':(3,),'quaternion_wxyz':(4,),'linear_velocity':(3,),'angular_velocity':(3,),
            'size_m':(3,),'hand_force_N':(4,),'opposed':(4,),'hand_open_error_rad':(4,),
            'hand_speed_rad_s':(4,),'mounts_m':(4,3)}
        x={k:np.asarray(o[k]) for k in shapes}
        if any(x[k].shape!=shape for k,shape in shapes.items()):self._fail('native_observation_shape',t);return self.intent()
        scalars=[t,o['now_s'],o['mass_kg'],o['support_N'],o['conservative_mu']]
        if not all(np.isfinite(a).all() for a in [*x.values(),scalars]):self._fail('nonfinite_native_observation',t);return self.intent()
        if o['object_id']!=self.spec.object_id or not o['physical_observation_valid']:
            self._fail('invalid_or_wrong_native_object',t);return self.intent()
        if o['sequence']<=self.last_seq or (self.last_time is not None and t<=self.last_time) or not 0<=o['now_s']-t<=self.max_age:
            self._fail('stale_or_nonmonotonic_native_observation',t);return self.intent()
        if self.last_time is not None and t-self.last_time>self.max_age+1e-9:
            self._fail('native_observation_gap',t);return self.intent()
        if not np.allclose(x['size_m'],self.spec.size_m,rtol=0,atol=1e-6) or abs(o['mass_kg']-self.spec.mass_kg)>1e-6:
            self._fail('native_geometry_or_mass_changed',t);return self.intent()
        if abs(np.dot(x['quaternion_wxyz'],x['quaternion_wxyz'])-1)>2e-4 or x['opposed'].dtype.kind!='b':
            self._fail('invalid_native_orientation_or_contact_label',t);return self.intent()
        if min(o['support_N'],o['conservative_mu'],*x['hand_force_N'],*x['hand_open_error_rad'],*x['hand_speed_rad_s'])<0:
            self._fail('negative_native_magnitude',t);return self.intent()
        self.last_seq=o['sequence'];self.last_time=t
        if self.phase_start is None:self.phase_start=t
        if t-self.phase_start>=self.timeout:self._fail('timeout_'+self.phase,t);return self.intent()
        force=x['hand_force_N'][self.ids];opposed=x['opposed'][self.ids].all()
        contact=(force>.1).all() and opposed
        lift=x['position'][2]-self.spec.initial_position[2]
        if self.phase in ['LIFT','CARRY'] and lift>=.05 and not contact:
            if self.contact_lost_start is None:self.contact_lost_start=t
            elif t-self.contact_lost_start>=.2-1e-9:
                self._fail('lost_opposed_holder_contacts_while_airborne',t);return self.intent()
        else:self.contact_lost_start=None
        xy_error=np.linalg.norm(x['position'][:2]-self.spec.goal_position[:2])
        displaced=np.linalg.norm(x['position'][:2]-self.spec.initial_position[:2])
        q=x['quaternion_wxyz'];tilt=np.degrees(np.arccos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)))
        rest=np.linalg.norm(x['linear_velocity'])<=.01 and np.linalg.norm(x['angular_velocity'])<=.1 and tilt<=10
        held=contact and lift>=.1 and o['support_N']<=.1 and rest
        supported=.8*self.spec.mass_kg*9.81<=o['support_N']<=1.5*self.spec.mass_kg*9.81
        at_goal=xy_error<=.02 and abs(x['position'][2]-self.spec.goal_position[2])<=.012
        detached=(force<=.1).all();opened=(x['hand_open_error_rad'][self.ids]<=.05).all()
        capacity=o['conservative_mu']*force.sum()
        ready=contact and capacity>=1.25*self.spec.mass_kg*9.81 and (x['hand_speed_rad_s'][self.ids]<=.05).all()
        retreat=False
        if self.release_mounts is not None:
            retreat=(np.linalg.norm(x['mounts_m'][self.ids]-self.release_mounts,axis=1)>=.02).all()
        predicate={'CLOSE':ready,'LIFT':held,'CARRY':held and xy_error<=.02 and displaced>=.2,
            'PLACE':supported and rest and at_goal,'RELEASE':supported and rest and at_goal and detached and opened,
            'RETREAT':supported and rest and at_goal and detached and retreat}[self.phase]
        if not predicate:self.window_start=None
        elif self.window_start is None:self.window_start=t
        elif t-self.window_start>=SECONDS[self.phase]-1e-9:
            if self.phase=='RELEASE':self.release_mounts=x['mounts_m'][self.ids].copy()
            next_phase={'CLOSE':'LIFT','LIFT':'CARRY','CARRY':'PLACE','PLACE':'RELEASE','RELEASE':'RETREAT','RETREAT':'DONE'}[self.phase]
            self._transition(next_phase,t)
        return self.intent()

    def intent(self):
        p=np.array(self.spec.initial_position if self.phase in ['CLOSE','LIFT'] else self.spec.goal_position,float)
        if self.phase in ['LIFT','CARRY']:p[2]+=.14
        return {'task_id':self.spec.task_id,'object_id':self.spec.object_id,'roles':self.spec.roles,
            'phase':self.phase,'object_target_m':p.tolist(),'open_hands':self.phase in ['RELEASE','RETREAT','DONE'],
            'halt_intent':self.phase in ['FAILED','DONE'],'failure':self.failure,'native_task_completion_certified':False}


class MultiObjectTasks:
    """Compose two-pair or four parallel tasks; an arm cannot own two objects."""
    def __init__(self,specs):
        if not specs:raise ValueError('tasks required')
        objects=[s.object_id for s in specs];roles=[a for s in specs for a in s.roles]
        if len(set(objects))!=len(objects) or len(set(roles))!=len(roles):
            raise ValueError('duplicate object or conflicting arm ownership')
        self.tasks={s.object_id:PayloadTask(s) for s in specs}

    def update(self,observations):
        if set(observations)!=set(self.tasks):raise ValueError('fresh observations for every assigned object required')
        intents={k:task.update(observations[k]) for k,task in self.tasks.items()}
        return {'object_intents':intents,'all_done':all(x['phase']=='DONE' for x in intents.values()),
            'any_failed':any(x['phase']=='FAILED' for x in intents.values()),'native_task_completion_certified':False}
