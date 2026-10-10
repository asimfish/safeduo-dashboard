"""Bound raw camera pixels and all 27 held fields to the short-screen microstate."""
import json
import re
import numpy as np
from evidence_io import ARMS, require, equal_bits, exact, contained, parse_json
from camera_optics import frustum

FIELDS = ('native_q','native_qd','native_root_xyzw','native_root_velocity','native_link_transforms_xyzw','native_position_targets')
CHUNK_FIELDS = FIELDS
CLOCK = ('simulation_time_s','simulation_time_step_index')


def exact_snapshot(a,b):
    require(set(a)==set(b) and len(a)==27,'all27 snapshot field set')
    for key in a: exact(a[key],b[key],'held27 '+key)


def hand_path(path):
    return 'f2_hand/' in path or any(s in path.rsplit('/',1)[-1] for s in ('thumb','index','middle','ring','little','pinky','wrist_3_link'))


class CameraAudit:
    def __init__(self,root,evidence,body,identity,layout):
        self.root=root; self.evidence=evidence; self.body=body; self.identity=identity
        self.layout=layout; self.environment_count=32; self.snapshots=0

    def ref(self,ref):
        require(isinstance(ref,dict) and set(('path','sha256'))<=set(ref),'camera SHA reference')
        p=contained(self.root/ref['path'],self.root)
        return self.evidence.raw(p,ref['sha256'])

    def npz(self,ref):
        p=contained(self.root/ref['path'],self.root)
        return self.evidence.npz(p,ref['sha256'])

    def snapshot(self,ref):
        state=self.npz(ref)
        require(set(state)=={'environment_origins',*CLOCK,*(a+'_'+f for a in ARMS for f in FIELDS)},'native27 inventory')
        require(state['environment_origins'].shape==(32,3) and state['environment_origins'].dtype==np.float32,'all32 origins')
        require(state[CLOCK[0]].shape==state[CLOCK[1]].shape==() and state[CLOCK[0]].dtype==np.float64 and state[CLOCK[1]].dtype==np.int64,'native clock')
        for arm in ARMS:
            lo,hi=self.layout['slices'][arm]
            for f in FIELDS:
                shape=(32,hi-lo)
                if f=='native_root_xyzw':shape=(32,7)
                if f=='native_root_velocity':shape=(32,6)
                if f=='native_link_transforms_xyzw':shape=(32,len(self.body[arm]),7)
                require(state[arm+'_'+f].shape==shape and state[arm+'_'+f].dtype==np.float32,'complete held native field '+arm+f)
        require(all(np.isfinite(v).all() for v in state.values()),'nonfinite held camera snapshot')
        self.snapshots+=1
        return state

    def load_inventory(self,ref):
        inventory=parse_json(self.ref(ref));indices=set()
        sensors={p for v in self.identity['views'] for p in v['sensors']}
        require(set(inventory)==sensors and len(inventory)==32*82,'all native camera inventory')
        for p,m in inventory.items():
            match=re.fullmatch(r'/World/envs/env_(\d+)/(F_L|F_R|U_L|U_R)/(.+)',p)
            require(match is not None,'camera native path')
            slot,arm=int(match[1]),match[2];idx=m['body_index']
            require(0<=slot<32 and type(idx) is int and 0<=idx<len(self.body[arm]),'camera index bounds')
            require((m['env_id'],m['arm'],m['body_name'],m['hand'])==(slot,arm,self.body[arm][idx],hand_path(p)),'camera native metadata')
            require(p.rsplit('/',1)[-1]==self.body[arm][idx] and (slot,arm,idx) not in indices,'camera body identity')
            indices.add((slot,arm,idx))
        return inventory


def micro_native(fields,micro,layout,origins):
    result={'environment_origins':origins,'simulation_time_s':fields['simulation_time_s'][micro],
        'simulation_time_step_index':fields['simulation_step'][micro]}
    for a in ARMS:
        lo,hi=layout['slices'][a]
        for dst,src in [('native_q','q74'),('native_qd','qd74'),('native_position_targets','target74')]:
            result[a+'_'+dst]=fields[src][micro,:,lo:hi]
        for dst,src in [('native_root_xyzw','root_xyzw'),('native_root_velocity','root_velocity'),('native_link_transforms_xyzw','link_xyzw')]:
            result[a+'_'+dst]=fields[a+'_'+src][micro]
    return result


def supplement(group,inventory,slot):
    gaps=[]
    for arm in ARMS:
        body={p for p,m in inventory.items() if m['env_id']==slot and m['arm']==arm and m['hand']}
        counts=group['arms'][arm]['per_native_body_peak_pixels']
        require(body and set(counts)==body,'hand64 denominator')
        gaps.extend(dict(path=p,pixels=n,required=64) for p,n in counts.items() if n<64)
    critical={p for p,m in inventory.items() if m['env_id']==slot and m['body_name'] in ('forearm_link','wrist_2_link')}
    require(len(critical)==4 and set(group['critical_body_pixels'])==critical,'two critical body types in both U arms')
    gaps.extend(dict(path=p,pixels=n,required=128) for p,n in group['critical_body_pixels'].items() if n<128)
    return dict(observed_coverage_sufficient=not gaps,gaps=gaps,
        scope='64 fixed initial/final groups; no alarm-triggered camera captures in this screen',
        full_mesh_coverage_certified=False,physical_safety_certified=False)
