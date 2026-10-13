"""Full50 position intents for a friction-only payload task; no state writes."""
import numpy as np
from multitask_phases_v1 import ARMS
from opposed_contact_v1 import opposed_contact


def native_opposed_flags(packet,partners):
    flags=[]
    for a in ARMS:
        ids=[j for j,v in enumerate(packet['partner_indices']) if '/'+a+'/' in partners[int(v)]
            and any(s in partners[int(v)].lower() for s in ['thumb','index','middle','ring','little','pinky','hand_base'])]
        flags.append(opposed_contact([partners[int(packet['partner_indices'][j])] for j in ids],
            packet['normals'][ids].reshape(-1,3),packet['points'][ids].reshape(-1,3),packet['normal_forces'][ids,0]))
    return np.asarray(flags,bool)


class PairTaskOperator:
    def __init__(self,task,parameters,rootids,endpoints,coupled_refs,*,dt=1/60,seek_rate=.06):
        self.task=task;self.parameters=parameters;self.rootids=rootids;self.endpoints=endpoints
        self.dt=dt;self.rate=seek_rate;self.force=np.zeros(4);self.last_observation=None
        mu=float(parameters['prelift_conservative_dynamic_mu'])
        # Original capacity reserve stays1.25mg. Command10% additional normal
        # reserve, within the unchanged30N measured hand-force gate.
        self.seek_force_N=min(16.,max(6.,1.375*task.spec.mass_kg*9.81/max(mu*len(task.spec.roles),1e-9)))
        self.open={};self.far={};self.seed={};self.arm_positions={}
        for a in ARMS:
            ids=rootids[a];names=parameters[a+'_joint_names'].tolist();hi=coupled_refs[a+'_hand_indices'].astype(int)
            self.open[a]=dict(zip(hi,coupled_refs[a+'_hand_open_q']))
            self.far[a]=dict(zip(hi,coupled_refs[a+'_hand_far_q']))
            self.seed[a]=parameters[a+'_initial_q'][0,ids].copy()
            self.arm_positions[a]=np.asarray([list(ids).index(int(i)) for i in parameters[a+'_controlled_indices']],int)

    def request(self,previous_target):
        intent=self.task.intent();phase=intent['phase'];result={a:np.asarray(previous_target[a]).copy() for a in ARMS}
        stage={'LIFT':1,'CARRY':2,'PLACE':3,'RELEASE':3,'RETREAT':4,'DONE':4}.get(phase,0)
        for ai,a in enumerate(ARMS):
            if a not in self.task.spec.roles or phase=='FAILED':continue
            result[a][self.arm_positions[a]]=self.endpoints[a][stage]
            names=self.parameters[a+'_joint_names'].tolist();ids=self.rootids[a]
            for j,index in enumerate(ids):
                if int(index) not in self.open[a]:continue
                name=names[int(index)]
                if intent['open_hands']:result[a][j]=self.open[a][int(index)]
                elif 'thumb_1' in name:continue  # keep registered opposition orientation
                else:
                    direction=np.sign(self.far[a][int(index)]-self.open[a][int(index)])
                    pressure=1. if self.force[ai]<self.seek_force_N else -1. if self.force[ai]>self.seek_force_N+2. else 0.
                    result[a][j]+=pressure*direction*self.rate*self.dt
            bounds=self.parameters[a+'_hard_limits'][0,ids]
            result[a]=np.clip(result[a],bounds[:,0]+.03,bounds[:,1]-.03)
        return result

    def observe(self,observation):
        self.force=np.asarray(observation['hand_force_N']).copy();self.last_observation=observation
        return self.task.update(observation)
