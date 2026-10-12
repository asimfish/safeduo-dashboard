"""Predict original arm and finger spheres from all actual native coordinates.

Same original9021 pairs and local sphere definitions. Fixed bases; no online
native body/wrist anchor, pose correction, q projection or follower synthesis.
Accuracy on recorded states does not prove future tracking or stopping bounds.
"""
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,'/home/liyufeng/safeduo/src')
from safeduo.baselines.real_geometry_v7 import make_v7_provider
from safeduo.baselines.real_geometry import F2_MOUNT_Z
from safeduo.configs import load_config,repo_root
from safeduo.safety.semantics import semantics_with_safety_overrides
from safeduo.safety.sphere_specs import real_v7_arm_specs
from safeduo.safety.sphere_distance import SphereDistanceModule
from hand_kinematic_replay_v1 import HandFK

ARMS=('F_L','F_R','U_L','U_R')


class Full74Geometry:
    def __init__(self,parameters,assets,*,config='duo_env_a31_pending_guard.yaml'):
        self.names={a:parameters[a+'_joint_names'].tolist() for a in ARMS}
        self.bodies={a:parameters[a+'_body_names'].tolist() for a in ARMS}
        self.armids={a:parameters[a+'_controlled_indices'].astype(int).ravel() for a in ARMS}
        assert sum(len(x) for x in self.names.values())==74
        assert sum(len(x) for x in self.armids.values())==26
        self.provider=make_v7_provider(1,device='cpu')
        cfg=load_config(config)
        assert cfg['assets']['spheres']=='real_v7'
        semantics=semantics_with_safety_overrides(str(repo_root()/cfg['assets']['semantics_yaml']),cfg['safety'])
        self.specs=real_v7_arm_specs(repo_root())
        self.sph=SphereDistanceModule(self.specs,semantics=semantics,device='cpu')
        self.sph.bind_bodies(self.bodies)
        self.tables=np.asarray(self.provider.layout.table_centers,np.float32)
        self.halves=np.asarray(self.provider.layout.table_half_extents,np.float32)
        self.radii=self.sph.radii.numpy().copy()
        self.offsets=self.sph.offsets.numpy().copy()
        self.pairs=[self.sph.pairs_cross.numpy().copy(),self.sph.pairs_self.numpy().copy(),self.sph.pairs_table.numpy().copy()]
        assert sum(len(p) for p in self.pairs)==9021
        self.hand={}
        for a in ARMS:
            anchor=('left_hand_base' if a.endswith('L') else 'right_hand_base') if a.startswith('F') else 'wrist_3_link'
            self.hand[a]=HandFK(assets[a]['original_URDF'],anchor)

    def _validate(self,full_q):
        if set(full_q)!=set(ARMS):raise ValueError('all4 full native coordinate arrays required')
        q={a:np.asarray(full_q[a]) for a in ARMS}
        n=None
        for a in ARMS:
            if q[a].ndim!=2 or q[a].shape[1]!=len(self.names[a]) or not len(q[a]):
                raise ValueError(f'{a}: need N by all native DOFs')
            if n is None:n=len(q[a])
            if len(q[a])!=n or not np.isfinite(q[a]).all():raise ValueError('mismatched or nonfinite states')
        return q,n

    def transforms(self,full_q):
        q,n=self._validate(full_q);out={}
        for a in ARMS:
            kin=self.provider.kin[a[0]];base,yaw=self.provider.layout.base_pose(a)
            fko=kin.fk(torch.as_tensor(q[a][:,self.armids[a]],dtype=torch.float32),base,yaw)
            R=fko['R'].numpy();t=fko['t'].numpy();Rf=fko['R_flange'].numpy();tf=fko['t_flange'].numpy()
            transforms={}
            for b,i in kin.frame_of_link.items():
                T=np.broadcast_to(np.eye(4),(n,4,4)).copy();T[:,:3,:3]=R[:,i+1];T[:,:3,3]=t[:,i+1];transforms[b]=T
            anchor=np.broadcast_to(np.eye(4),(n,4,4)).copy();anchor[:,:3,:3]=Rf;anchor[:,:3,3]=tf
            if a.startswith('F'):
                transforms['fr3_link8']=anchor.copy()
                anchor[:,:3,3]+=np.einsum('nij,j->ni',Rf,np.array([0.,0.,F2_MOUNT_Z]))
            if a.startswith('U'):
                # URDF base_link -> base_link_inertia applies a fixed pi rotation.
                # That rotation belongs to the moving chain, never to base_link.
                cy,sy=np.cos(yaw),np.sin(yaw)
                transforms['base_link'][:,:3,:3]=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
            hand=self.hand[a].transforms({name:q[a][:,i] for i,name in enumerate(self.names[a])},anchor)
            transforms.update(hand)
            out[a]=transforms
        return out

    def centers_from_transforms(self,transforms):
        pieces=[]
        for a in ARMS:
            for link in self.specs[a].links:
                T=transforms[a][link.link]
                offsets=np.asarray(link.offsets,np.float32)
                pieces.append((T[:,:3,3][:,None]+np.einsum('nij,sj->nsi',T[:,:3,:3],offsets)).astype(np.float32))
        result=np.concatenate(pieces,axis=1)
        assert result.shape[1:]==(self.sph.n_spheres,3)
        return result

    def centers(self,full_q):return self.centers_from_transforms(self.transforms(full_q))

    def gaps_from_centers(self,centers):
        centers=np.asarray(centers,np.float32)
        if centers.ndim!=3 or centers.shape[1:]!=(self.sph.n_spheres,3) or not np.isfinite(centers).all():
            raise ValueError('finite N by original sphere count by3 required')
        rows=[]
        for pairs in self.pairs[:2]:
            rows.append(np.linalg.norm(centers[:,pairs[:,0]]-centers[:,pairs[:,1]],axis=-1)-(self.radii[pairs[:,0]]+self.radii[pairs[:,1]]))
        pair=self.pairs[2];box=np.abs(centers[:,pair[:,0]]-self.tables[pair[:,1]])-self.halves[pair[:,1]]
        rows.append(np.linalg.norm(np.maximum(box,0),axis=-1)+np.minimum(box.max(axis=-1),0)-self.radii[pair[:,0]])
        return np.concatenate(rows,axis=1)

    def gaps(self,full_q):return self.gaps_from_centers(self.centers(full_q))
