"""Full nonlinear independent-coordinate FK using source-bound joint frames.

Data bundle must supply actual-frame evidence/calibration; this file does not
infer active native mimic/tendon/gear constraints or fabricate their absence.
"""
import numpy as np
from io_v2 import need,science
ARMS=('F_L','F_R','U_L','U_R')
def transform(v):
    x=np.asarray(v,np.float64);need(x.shape==(4,4) and np.isfinite(x).all(),'finite4x4 frame')
    need(np.array_equal(x[3],np.array([0.,0.,0.,1.])) and np.max(abs(x[:3,:3].T@x[:3,:3]-np.eye(3)))<=1e-9 and abs(np.linalg.det(x[:3,:3])-1)<=1e-9,'rigid proper transform')
    return x.copy()
def rotation(axis,theta):
    a=np.asarray(axis,np.float64);need(a.shape==(3,) and np.isfinite(a).all() and abs(np.linalg.norm(a)-1)<=1e-12,'unit joint axis; no guessed normalization')
    k=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]],np.float64)
    return np.eye(3)[None]+np.sin(theta)[:,None,None]*k+(1-np.cos(theta))[:,None,None]*(k@k)
def quaternion(r):
    """Matrix->xyzw for exact reused pose-to-9021 geometry, float64 throughout."""
    out=np.empty((len(r),4),np.float64)
    for i,a in enumerate(r):
        tr=float(np.trace(a))
        if tr>0:
            s=2*np.sqrt(tr+1);q=np.array([(a[2,1]-a[1,2])/s,(a[0,2]-a[2,0])/s,(a[1,0]-a[0,1])/s,s/4])
        else:
            j=int(np.argmax(np.diag(a)));u=(j+1)%3;v=(j+2)%3;s=2*np.sqrt(1+a[j,j]-a[u,u]-a[v,v]);q=np.empty(4)
            q[j]=s/4;q[u]=(a[j,u]+a[u,j])/s;q[v]=(a[j,v]+a[v,j])/s;q[3]=(a[v,u]-a[u,v])/s
        need(np.isfinite(q).all() and abs(np.linalg.norm(q)-1)<1e-9,'predicted quaternion');out[i]=q
    return out

class Model:
    def __init__(self,bundle,geometry,layout,body_names):
        need(bundle['schema']=='astra.full74.fk.frame_model.v2' and bundle['coordinates']=='ALL74_INDEPENDENT_NO_MIMIC_PROJECTION' and bundle['units']=={'position':'m','angle':'rad'},'typed full-coordinate model')
        need(set(bundle['arms'])==set(ARMS) and bundle['joint_names74']==layout['names'].tolist(),'all74 actual DOF identity')
        self.geometry=geometry;self.table_centers=np.asarray(bundle['table_centers_world_no_env_origin'],np.float64);self.table_half=np.asarray(bundle['table_half_extents'],np.float64)
        need(self.table_centers.shape==self.table_half.shape==(2,3) and np.isfinite(self.table_centers).all() and np.isfinite(self.table_half).all() and np.all(self.table_half>0),'original two table boxes')
        self.arms={};used=[]
        for arm in ARMS:
            spec=bundle['arms'][arm];names=body_names[arm].tolist();need(spec['body_names']==names and len(set(names))==len(names),'original82 body order')
            root=spec['root_frame'];known={root};instructions=[];lo,hi=layout['slices'][arm]
            for j in spec['joints']:
                need(j['parent'] in known and j['child'] not in known,'acyclic topologically ordered native frame tree')
                known.add(j['child']);kind=j['kind'];need(kind in ('fixed','revolute','prismatic'),'supported rigid FK joint only')
                idx=j['q_index']
                if kind=='fixed':need(idx is None,'fixed joint cannot consume a sampled DOF')
                else:
                    need(type(idx) is int and lo<=idx<hi and j['native_joint_name']==layout['names'][idx],'matched native DOF axis')
                    used.append(idx);axis=np.asarray(j['axis'],np.float64);rotation(axis,np.array([0.]))
                axis=np.asarray(j['axis'],np.float64);need(axis.shape==(3,) and np.isfinite(axis).all(),'finite axis metadata')
                instructions.append((j['parent'],j['child'],kind,idx,np.asarray(j['axis'],np.float64),transform(j['parent_to_joint']),np.linalg.inv(transform(j['child_to_joint']))))
            need(set(spec['body_frames'])==set(names) and all(f in known for f in spec['body_frames'].values()),'all body frames accounted')
            self.arms[arm]=(root,transform(spec['root_world_no_env_origin']),instructions,[spec['body_frames'][n] for n in names])
        need(sorted(used)==list(range(74)),'every original74 DOF once; no arm-only/zero-hand shortcut')
        self.geometry_fn=science().geometry

    def predict_poses(self,q74):
        need(q74.shape==(32,74) and q74.dtype==np.float32 and np.isfinite(q74).all(),'all32 complete unchanged q74')
        fields={}
        for arm,(root,base,instructions,frames) in self.arms.items():
            tf={root:np.repeat(base[None],32,axis=0)}
            for parent,child,kind,idx,axis,a,b in instructions:
                motion=np.repeat(np.eye(4)[None],32,axis=0)
                if kind=='revolute':motion[:,:3,:3]=rotation(axis,q74[:,idx].astype(np.float64))
                elif kind=='prismatic':motion[:,:3,3]=q74[:,idx,None].astype(np.float64)*axis
                tf[child]=tf[parent]@a@motion@b
            poses=np.empty((32,len(frames),7),np.float64)
            for i,f in enumerate(frames):poses[:,i,:3]=tf[f][:,:3,3];poses[:,i,3:]=quaternion(tf[f][:,:3,:3])
            fields[arm+'_link_xyzw']=poses
        return fields
    def predict_geometry(self,q74,*,global_ids):
        need(global_ids.shape==(32,) and global_ids.dtype==np.int64,'registered32 identity')
        poses=self.predict_poses(q74);g,centers=self.geometry_fn(poses,self.geometry,self.table_centers,self.table_half,np.zeros((32,3),np.float64))
        need(g.shape==(32,9021) and g.dtype==np.float64,'all9021 model rows');return g
