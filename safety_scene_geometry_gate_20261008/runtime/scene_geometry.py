"""Read-only USD collision/FK prediction. Never writes simulator state."""
import itertools
import numpy as np

def pose_matrix(pose):
    p=np.asarray(pose,dtype=np.float64)
    if p.shape!=(7,) or not np.isfinite(p).all():raise ValueError('pose')
    x,y,z,w=p[3:]; n=np.linalg.norm(p[3:])
    if abs(n-1)>1e-3:raise ValueError('quaternion norm')
    x,y,z,w=p[3:]/n
    t=np.eye(4);t[:3,3]=p[:3]
    t[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    return t

def corners(lo,hi):
    lo=np.asarray(lo,float);hi=np.asarray(hi,float)
    if not np.isfinite([lo,hi]).all() or np.any(lo>hi) or np.max(abs(np.r_[lo,hi]))>1000:
        raise ValueError('empty/nonfinite/unreasonable collider bounds')
    return np.asarray(list(itertools.product(*zip(lo,hi))))

def transform(t,xyz):
    return np.matmul(np.asarray(xyz),np.swapaxes(t[...,:3,:3],-1,-2))+t[...,None,:3,3]

def joint_matrix(j):
    return pose_matrix(j)

def extract(stage, arm_meta, prefix='/World/envs/env_0'):
    from pxr import Usd,UsdGeom,UsdPhysics
    xf=UsdGeom.XformCache(Usd.TimeCode.Default())
    out=dict(colliders=[],joints={a:[] for a in arm_meta},purposes=['default','guide','render','proxy'])
    def jp(v,q):return list(v)+list(q.GetImaginary())+[q.GetReal()]
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        path=str(prim.GetPath())
        if not path.startswith(prefix+'/'):continue
        if prim.IsA(UsdPhysics.Joint):
            j=UsdPhysics.Joint(prim);b0=j.GetBody0Rel().GetTargets();b1=j.GetBody1Rel().GetTargets()
            if not b0 or not b1:continue
            parent=str(b0[0]).rsplit('/',1)[-1];child=str(b1[0]).rsplit('/',1)[-1]
            for arm,m in arm_meta.items():
                if arm not in ('U_L','U_R'):continue
                if path.startswith(prefix+'/'+arm+'/') and child in m['body_names'][7:]:
                    name=prim.GetName()
                    if name not in m['joint_names']:raise ValueError(('unknown hand joint',path))
                    rev=UsdPhysics.RevoluteJoint(prim)
                    if not rev:raise ValueError(('unsupported joint',path))
                    out['joints'][arm].append(dict(name=name,parent=parent,child=child,axis=str(rev.GetAxisAttr().Get()),
                        local0=jp(j.GetLocalPos0Attr().Get(),j.GetLocalRot0Attr().Get()),local1=jp(j.GetLocalPos1Attr().Get(),j.GetLocalRot1Attr().Get())))
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():continue
        owner=prim
        while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):owner=owner.GetParent()
        ownerpath=str(owner.GetPath()) if owner else None
        points=[];mismatches=[]
        # Authored extent can be stale after asset unit conversion (KLT: 100x).
        # Read Mesh points or physical primitive dimensions; do not trust extent.
        for geo in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()):
            kind=geo.GetTypeName();v=None
            if kind=='Mesh':
                pp=np.asarray(UsdGeom.Mesh(geo).GetPointsAttr().Get(),float)
                if pp.ndim!=2 or pp.shape[1]!=3 or not len(pp) or not np.isfinite(pp).all():raise ValueError(('mesh points',str(geo.GetPath())))
                v=corners(pp.min(0),pp.max(0))
            elif kind=='Cube':
                size=float(UsdGeom.Cube(geo).GetSizeAttr().Get());v=corners(np.full(3,-size/2),np.full(3,size/2))
            elif kind=='Sphere':
                rad=float(UsdGeom.Sphere(geo).GetRadiusAttr().Get());v=corners(np.full(3,-rad),np.full(3,rad))
            elif kind in ('Cylinder','Capsule','Cone'):
                shape=getattr(UsdGeom,kind)(geo);rad=float(shape.GetRadiusAttr().Get());h=float(shape.GetHeightAttr().Get());axis='XYZ'.index(str(shape.GetAxisAttr().Get()));half=np.full(3,rad);half[axis]=h/2+(rad if kind=='Capsule' else 0);v=corners(-half,half)
            elif geo.IsA(UsdGeom.Boundable):raise ValueError(('unsupported physical shape',str(geo.GetPath()),kind))
            if v is None:continue
            attr=geo.GetAttribute('extent');auth=np.asarray(attr.Get(),float) if attr and attr.Get() is not None else None
            if auth is not None and (auth.shape!=(2,3) or not np.allclose(auth,np.stack([v.min(0),v.max(0)]),atol=1e-5)):
                mismatches.append(str(geo.GetPath()))
            mat=xf.ComputeRelativeTransform(geo,owner)[0] if owner else xf.GetLocalToWorldTransform(geo)
            points.append(transform(np.asarray(mat).T,v))
        if not points:raise ValueError(('no physical geometry',path))
        pp=np.concatenate(points);xyz=corners(pp.min(0),pp.max(0))
        offset=prim.GetAttribute('physxCollision:contactOffset').Get()
        if offset is None or offset<0:offset=.020
        arm=next((a for a in arm_meta if path.startswith(prefix+'/'+a+'/')),None)
        body=ownerpath.rsplit('/',1)[-1] if ownerpath else None
        if arm and body not in arm_meta[arm]['body_names']:raise ValueError(('missing native body',ownerpath))
        out['colliders'].append(dict(path=path,owner=ownerpath,arm=arm,body=body,corners=xyz.tolist(),contact_offset_m=float(offset),
            authored_extent_mismatches=mismatches,geometry_source='mesh_points_or_primitive_dimensions',
            hand=bool(arm and (body=='wrist_3_link' or body in arm_meta[arm]['body_names'][7:]))))
    for arm in ('U_L','U_R'):
        pending=list(out['joints'][arm]);ordered=[];seen={'wrist_3_link'}
        while pending:
            ready=[j for j in pending if j['parent'] in seen]
            if not ready:raise ValueError(('disconnected hand joint tree',arm,pending))
            for j in ready:ordered.append(j);seen.add(j['child']);pending.remove(j)
        out['joints'][arm]=ordered
        if len(ordered)!=12:raise ValueError(('hand joint count',arm,len(ordered)))
    return out

def hand_fk(schema,arm,wrist_pose,q,joint_names):
    q=np.asarray(q,float)
    batch=q.ndim==2
    root=pose_matrix(wrist_pose)
    poses={'wrist_3_link':np.broadcast_to(root,(q.shape[0],4,4)).copy() if batch else root}
    for j in schema['joints'][arm]:
        v=q[...,joint_names.index(j['name'])];axis='XYZ'.index(j['axis']);rot=np.zeros(v.shape+(4,4));rot[...,3,3]=1;rot[...,axis,axis]=1
        b=(axis+1)%3;c=(axis+2)%3;co=np.cos(v);si=np.sin(v)
        rot[...,b,b]=co;rot[...,c,c]=co;rot[...,b,c]=-si;rot[...,c,b]=si
        poses[j['child']]=poses[j['parent']]@joint_matrix(j['local0'])@rot@np.linalg.inv(joint_matrix(j['local1']))
    return poses

def fk_error(schema,arm,link_poses,q,meta):
    from scipy.spatial.transform import Rotation
    poses=hand_fk(schema,arm,link_poses[meta['body_names'].index('wrist_3_link')],q,meta['joint_names'])
    pe=re=0.
    for body,t in poses.items():
        native=pose_matrix(link_poses[meta['body_names'].index(body)])
        pe=max(pe,float(abs(t[:3,3]-native[:3,3]).max()))
        aq=Rotation.from_matrix(t[:3,:3]).as_quat();nq=link_poses[meta['body_names'].index(body),3:]
        re=max(re,float(min(abs(aq-nq).max(),abs(aq+nq).max())))
    return pe,re

def interpolation_padding(schema,arm,body,xyz,delta,hand_names,samples):
    bychild={j['child']:j for j in schema['joints'][arm]};chain=[]
    while body in bychild:
        j=bychild[body];chain.append(j);body=j['parent']
    # Radius bound about each ancestor: collider radius + downstream anchor norms.
    radius=float(np.linalg.norm(xyz,axis=1).max());bound=0.
    for j in chain:
        radius+=float(np.linalg.norm(j['local1'][:3]))
        bound+=radius*abs(float(delta[hand_names.index(j['name'])]))
        radius+=float(np.linalg.norm(j['local0'][:3]))
    return bound/(samples-1)

def path_boxes(schema,arm,wrist,q0,goal,hand_names,joint_names,reserve,samples=121):
    q0=np.asarray(q0,float);goal=np.asarray(goal,float)
    q=np.zeros((samples,len(joint_names)));alpha=np.linspace(0,1,samples)
    q[:,[joint_names.index(n) for n in hand_names]]=q0+alpha[:,None]*(goal-q0)
    poses=hand_fk(schema,arm,wrist,q,joint_names);boxes=[]
    for c in schema['colliders']:
        if c['arm']!=arm or not c['hand']:continue
        xyz=np.asarray(c['corners']);p=transform(poses[c['body']],xyz)
        padding=reserve+c['contact_offset_m']+interpolation_padding(schema,arm,c['body'],xyz,goal-q0,hand_names,samples)
        boxes.append((c,p.min(1)-padding,p.max(1)+padding,padding))
    if not boxes:raise ValueError('no hand collider geometry')
    return boxes

def separation(lo1,hi1,lo2,hi2):
    # Positive axis gap is a sufficient separation certificate for outer boxes.
    return np.maximum(lo2-hi1,lo1-hi2).max(axis=-1)

def predict(schema,arm,links,meta,opened,goals,object_poses,reserve=.020,samples=121):
    paths={}
    for a in ('U_L','U_R'):
        paths[a]=path_boxes(schema,a,links[a][meta[a]['body_names'].index('wrist_3_link')],opened[a],goals[a],
            [meta[a]['joint_names'][i] for i in meta[a]['hand_ids']],meta[a]['joint_names'],reserve,samples)
    obs=[]
    for c in schema['colliders']:
        if c['arm']==arm:continue
        if c['arm'] in paths and c['hand']:
            b=next(b for b in paths[c['arm']] if b[0]['path']==c['path'])
            obs.append((c['path'],b[1].min(0),b[2].max(0)));continue
        if c['arm']:
            p=links[c['arm']][meta[c['arm']]['body_names'].index(c['body'])]
            xyz=transform(pose_matrix(p),np.asarray(c['corners']))
        elif c['owner']:
            p=object_poses[c['owner']];xyz=transform(pose_matrix(p),np.asarray(c['corners']))
        else:xyz=np.asarray(c['corners'])
        obs.append((c['path'],xyz.min(0)-c['contact_offset_m']-reserve,xyz.max(0)+c['contact_offset_m']+reserve))
    olo=np.asarray([o[1] for o in obs]);ohi=np.asarray([o[2] for o in obs]);minimum=float('inf');witness=None
    for c,lo,hi,padding in paths[arm]:
        gaps=separation(lo[:,None,:],hi[:,None,:],olo[None],ohi[None])
        i,j=np.unravel_index(np.argmin(gaps),gaps.shape)
        if gaps[i,j]<minimum:
            minimum=float(gaps[i,j]);witness=dict(hand_collider=c['path'],obstacle=obs[j][0],alpha=float(i/(samples-1)),padding_m=float(padding))
        ground=float(lo[:,2].min())
        if ground<minimum:minimum=ground;witness=dict(hand_collider=c['path'],obstacle='/World/ground',alpha=float(np.argmin(lo[:,2])/(samples-1)),padding_m=float(padding))
    return dict(admitted=minimum>0,reason='scene_path_separated' if minimum>0 else 'predicted_scene_overlap',minimum_axis_gap_m=minimum,witness=witness,samples=samples)

class NativeSceneGuard:
    def __init__(self,env,registration,sv,schema,meta):
        self.env=env;self.r=registration;self.schema=schema;self.meta=meta;self.sv=sv
        self.views={p:sv.create_rigid_body_view(p) for p in sorted({c['owner'] for c in schema['colliders'] if c['owner'] and not c['arm']})}
        if any(not v.check() or v.count!=1 for v in self.views.values()):raise ValueError('obstacle view identity')

    def check(self,e,arm,goal,profile_index):
        try:return self._check(e,arm,goal,profile_index)
        except (KeyError,ValueError,TypeError,IndexError) as err:
            return dict(admitted=False,reason='missing_or_invalid_scene_geometry',error=str(err))

    def _check(self,e,arm,goal,profile_index):
        npv=lambda x:x.detach().clone().cpu().numpy()
        links={a:npv(self.env._arms[a].root_physx_view.get_link_transforms()[e]) for a in self.meta}
        poses={p:npv(v.get_transforms()[0]) for p,v in self.views.items()}
        origin=npv(self.env.scene.env_origins[e]);origin0=npv(self.env.scene.env_origins[0])
        # Move all native poses to env0 coordinates, which own the collision template.
        for x in links.values():x[:,:3]+=origin0-origin
        for p in poses:
            v=self.views[p]
            actual=p.replace('/env_0/',f'/env_{e}/')
            if e:
                ev=getattr(self,'env_views',None)
                if ev is None:self.env_views={};ev=self.env_views
                if actual not in ev:
                    ev[actual]=self.sv.create_rigid_body_view(actual)
                    if not ev[actual].check() or ev[actual].count!=1:raise ValueError('obstacle env identity')
                poses[p]=npv(ev[actual].get_transforms()[0]);poses[p][:3]+=origin0-origin
        goals={a:np.asarray(self.r['passport']['allowed_goals_rad'][a][profile_index],np.float32) for a in ('U_L','U_R')}
        goals[arm]=np.asarray(goal,np.float32)
        opened={a:np.asarray(self.r['passport']['open_rad'][a],np.float32) for a in ('U_L','U_R')}
        errors={}
        for a in ('U_L','U_R'):
            q=npv(self.env._arms[a].root_physx_view.get_dof_positions()[e])
            pe,re=fk_error(self.schema,a,links[a],q,self.meta[a]);errors[a]=dict(position_error_m=pe,quaternion_error=re)
            if pe>self.r['geometry']['max_fk_position_error_m'] or re>self.r['geometry']['max_fk_quaternion_component_error']:
                return dict(admitted=False,reason='invalid_geometry_kinematics',fk_errors=errors)
        out=predict(self.schema,arm,links,self.meta,opened,goals,poses,self.r['geometry']['reserve_m'],self.r['geometry']['samples'])
        out['fk_errors']=errors
        return out
