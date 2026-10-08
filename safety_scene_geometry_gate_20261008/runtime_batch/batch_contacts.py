"""Batch native contact reads; preserve per-environment native point tuples."""
import numpy as np
import torch

def pack(raw,ids,capacity):
    counts=raw[-2][ids].copy();starts=raw[-1][ids];newstarts=np.zeros_like(counts)
    arrays=[np.zeros((capacity,)+x.shape[1:],x.dtype) for x in raw[:-2]];used=0
    assert (counts>=0).all() and int(counts.sum())<capacity
    for si,fi in zip(*np.nonzero(counts)):
        n=int(counts[si,fi]);st=int(starts[si,fi]);assert 0<=st and st+n<=len(raw[0])
        newstarts[si,fi]=used
        for arr,src in zip(arrays,raw[:-2]):arr[used:used+n]=src[st:st+n]
        used+=n
    return arrays+[counts,newstarts]

class LocalView:
    def __init__(self,pool,group,legacy):
        self.pool=pool;self.group=group;self.sensor_paths=list(legacy.sensor_paths);self.filter_paths=[list(f) for f in legacy.filter_paths]
        self.ids=[group['rowmap'][p] for p in self.sensor_paths];self.cached=None;self.epoch=None
    def _data(self):
        if self.epoch!=self.pool.epoch:
            r=self.pool.read(self.group);self.cached=dict(net=r['net'][self.ids].copy(),matrix=r['matrix'][self.ids].copy(),normal=pack(r['normal'],self.ids,self.pool.capacity),friction=pack(r['friction'],self.ids,self.pool.capacity));self.epoch=self.pool.epoch
        return self.cached
    def get_net_contact_forces(self,dt):assert dt==self.pool.dt;return torch.from_numpy(self._data()['net'])
    def get_contact_force_matrix(self,dt):assert dt==self.pool.dt;return torch.from_numpy(self._data()['matrix'])
    def get_contact_data(self,dt):assert dt==self.pool.dt;return tuple(torch.from_numpy(x) for x in self._data()['normal'])
    def get_friction_data(self,dt):assert dt==self.pool.dt;return tuple(torch.from_numpy(x) for x in self._data()['friction'])

class BatchPool:
    def __init__(self,sv,hands,objects,capacity,dt):
        self.capacity=capacity;self.dt=dt;self.epoch=0;self.groups=[];self.cache={};self.identities=[]
        replacements={}
        for arm in ['F_L','F_R','U_L','U_R']:
            rows=[(e,a,v,p) for e,a,v,p in hands if a==arm];self._group(sv,'hand_'+arm,rows,replacements)
        for obj in ['beam700','beam300']:
            rows=[(e,o,v,p) for e,o,v,p in objects if o==obj];self._group(sv,'object_'+obj,rows,replacements)
        self.hands=[(e,a,replacements[id(v)],p) for e,a,v,p in hands]
        self.objects=[(e,o,replacements[id(v)],p) for e,o,v,p in objects]

    def _group(self,sv,name,rows,replacements):
        paths=[p for e,k,v,fs in rows for p in v.sensor_paths];filters=[list(f) for e,k,v,fs in rows for f in v.filter_paths]
        view=sv.create_rigid_contact_view(paths,filter_patterns=filters,max_contact_data_count=self.capacity*len(rows));assert view.check() and set(view.sensor_paths)==set(paths)
        rowmap={path:i for i,path in enumerate(view.sensor_paths)};actual=list(view.filter_paths)
        for e,k,v,fs in rows:
            for path,ff in zip(v.sensor_paths,v.filter_paths):assert list(actual[rowmap[path]])==list(ff)
        g=dict(name=name,view=view,rowmap=rowmap);self.groups.append(g)
        for e,k,v,fs in rows:
            local=LocalView(self,g,v);replacements[id(v)]=local
            self.identities.append(dict(group=name,env=e,key=k,sensor_paths=local.sensor_paths,sensor_indices=local.ids,filter_paths=local.filter_paths))

    def invalidate(self):self.epoch+=1;self.cache={}
    def read(self,g):
        if g['name'] not in self.cache:
            cp=lambda x:x.detach().clone().cpu().numpy();v=g['view']
            net=cp(v.get_net_contact_forces(self.dt));matrix=cp(v.get_contact_force_matrix(self.dt));normal=[cp(x) for x in v.get_contact_data(self.dt)];friction=[cp(x) for x in v.get_friction_data(self.dt)]
            self.cache[g['name']]=dict(net=net,matrix=matrix,normal=normal,friction=friction)
        return self.cache[g['name']]
    def originals(self):
        out={}
        for g in self.groups:
            data=self.read(g)
            for kind in ['normal','friction']:
                raw=data[kind];cnt,st=raw[-2:];force=np.zeros_like(raw[0]);assert (cnt>=0).all() and int(cnt.sum())<len(force)
                for si,fi in zip(*np.nonzero(cnt)):
                    n=int(cnt[si,fi]);start=int(st[si,fi]);assert 0<=start and start+n<=len(force);force[start:start+n]=raw[0][start:start+n];assert np.isfinite(force[start:start+n]).all()
                key='batch:'+g['name']+':'+kind+':'
                out[key+'force']=force;out[key+'count']=cnt.copy();out[key+'start']=st.copy()
        return out
