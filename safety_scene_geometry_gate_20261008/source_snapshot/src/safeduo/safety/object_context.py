"""Versioned simulation object state and sphere/OBB safety geometry.

This is an explicit geometric branch, not a feature-compatible replacement for
an a31b policy or a camera-based perception system. Sizes are FULL extents.
"""
import math
from dataclasses import dataclass
import torch

SCHEMA = 'safeduo.object_context.v1'

@dataclass
class ObjectContext:
    names: tuple[str, ...]
    position: torch.Tensor       # N,K,3 world metres
    quaternion: torch.Tensor     # N,K,4 wxyz
    size: torch.Tensor           # N,K,3 full local extent
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor

    def validate(self):
        if self.position.ndim != 3 or self.position.shape[-1] != 3:
            raise ValueError("Object positions must have shape N,K,3")
        n,k,_ = self.position.shape
        if len(self.names)!=k or len(set(self.names))!=k:
            raise ValueError('Object names must be unique and match K')
        for tensor,width in [(self.position,3),(self.quaternion,4),(self.size,3),(self.linear_velocity,3),(self.angular_velocity,3)]:
            if tensor.shape!=(n,k,width) or not torch.isfinite(tensor).all():
                raise ValueError('Invalid object state shape or nonfinite value')
        if (self.size<=0).any() or (self.quaternion.norm(dim=-1)<1e-8).any():
            raise ValueError('Object dimensions and quaternion norm must be positive')

    def features(self):
        self.validate()
        return torch.cat([self.position,self.quaternion/self.quaternion.norm(dim=-1,keepdim=True),self.size,self.linear_velocity,self.angular_velocity],dim=-1)


def sphere_object_margin(centers, radii, objects: ObjectContext):
    """Signed N,S,K margin; negative means sphere intersects object OBB."""
    objects.validate()
    if centers.ndim!=3 or centers.shape[-1]!=3 or radii.shape!=(centers.shape[1],):
        raise ValueError('Expected centers N,S,3 and radii S')
    if centers.shape[0] != objects.position.shape[0]:
        raise ValueError("Sphere/object batch dimensions differ")
    if not torch.isfinite(centers).all() or not torch.isfinite(radii).all() or (radii<0).any():
        raise ValueError('Invalid sphere geometry')
    q=objects.quaternion/objects.quaternion.norm(dim=-1,keepdim=True)
    v=centers[:,:,None,:]-objects.position[:,None,:,:]
    xyz=-q[:,None,:,1:].expand_as(v);w=q[:,None,:,:1]
    twice=2*torch.cross(xyz,v,dim=-1)
    local=v+w*twice+torch.cross(xyz,twice,dim=-1)
    d=local.abs()-objects.size[:,None,:,:]/2
    sdf=d.clamp_min(0).norm(dim=-1)+d.amax(dim=-1).clamp_max(0)
    return sdf-radii[None,:,None]


def object_risk(current, predicted, radii, objects, predicted_objects, allowed_contact, clearance_m=0.01):
    """No-risk-exposure rows stay free; approaching forbidden rows brake.

    Separating motion is permitted only when predicted clearance increases.
    This endpoint check does not prove continuous collision avoidance.
    """
    if not math.isfinite(clearance_m) or clearance_m<0: raise ValueError('clearance must be nonnegative')
    now=sphere_object_margin(current,radii,objects)
    future=sphere_object_margin(predicted,radii,predicted_objects)
    if allowed_contact.dtype!=torch.bool or allowed_contact.shape!=now.shape:
        raise ValueError('Explicit N,S,K boolean contact allowance required')
    risk=(torch.minimum(now,future)<clearance_m)&(future<=now+1e-6)&~allowed_contact
    return {'margin':now,'predicted_margin':future,'risk':risk}


def from_env(env):
    """Read simulator truth; object names must resolve to configured geometry."""
    specs={s['name']:s for s in env.cfg.table_objects}
    names=tuple(sorted(env._objects))
    if not names: raise ValueError('No task objects configured')
    pos=[];quat=[];sizes=[];lin=[];ang=[]
    for name in names:
        obj=env._objects[name];spec=specs[name]
        # parse_table_objects resolves catalog sizes into the scene spec.
        size=spec.get('size')
        if size is None: raise ValueError(f'Missing full object extent: {name}')
        pos.append(obj.data.root_pos_w);quat.append(obj.data.root_quat_w)
        lin.append(obj.data.root_lin_vel_w);ang.append(obj.data.root_ang_vel_w)
        sizes.append(torch.tensor(size,device=env.device,dtype=obj.data.root_pos_w.dtype).expand(env.num_envs,3))
    context=ObjectContext(names,torch.stack(pos,1),torch.stack(quat,1),torch.stack(sizes,1),torch.stack(lin,1),torch.stack(ang,1))
    context.validate();return context

class SimulationObjectGuard:
    """Explicit object-state consumer around a legacy neural policy.

    Uses instantaneous rigid-body Jacobians and the pending joint command for
    a short linear lookahead. No contact allowance is inferred from proximity.
    Contact assignments must be task-authorized. Does not certify payload/table
    or payload/payload collisions, which require separate checks.
    """
    def __init__(self, env, assignments, clearance_m=0.01, horizon_s=0.08):
        if not math.isfinite(horizon_s) or not math.isfinite(clearance_m) or horizon_s<=0 or clearance_m<0:
            raise ValueError('Invalid guard configuration')
        self.env=env;self.assignments=dict(assignments)
        self.clearance_m=clearance_m;self.horizon_s=horizon_s
        self.object_names=tuple(sorted(env._objects))
        from safeduo.safety.types import ARM_KEYS
        self.arms=ARM_KEYS
        if any(a not in ARM_KEYS or o not in self.object_names for a,o in self.assignments.items()):
            raise ValueError('Invalid intentional contact assignment')
        self.allowed=torch.zeros((env.num_envs,env._sph.n_spheres,len(self.object_names)),dtype=torch.bool,device=env.device)
        for s,name in enumerate(env._sph.qualified_names):
            arm=name.split('/')[0]
            if arm in self.assignments and any(x in name for x in ('thumb','index','middle','ring','pinky','little','hand_base')):
                self.allowed[:,s,self.object_names.index(self.assignments[arm])]=True

    def assess(self, alpha):
        env=self.env;obj=from_env(env)
        if obj.names!=self.object_names:raise ValueError('Object inventory changed')
        obj.features()  # Validate full state branch independently of policy ABI.
        pos,quat,lin,ang={},{},{},{}
        for a in self.arms:pos[a],quat[a],lin[a],ang[a]=env._body_data(a)
        centers,_=env._sph._centers_vels(pos,quat,lin,ang)
        predicted=centers.clone()
        for ai,a in enumerate(self.arms):
            sl=env._sph._arm_slices[a];body=env._sph._body_idx[a];art=env._arms[a]
            jac=art.root_physx_view.get_jacobians()
            # Fixed-base PhysX Jacobians omit the root row; root spheres stay fixed.
            rows=body-1 if jac.shape[1]==len(art.body_names)-1 else body
            j=jac[:,rows.clamp_min(0),:,:][:,:,:,env._joint_idx[a]]
            dq=env._pending_cmd.delta_q[a]*alpha[:,ai:ai+1]*(self.horizon_s/env.step_dt)
            twist=(j@dq[:,None,:,None]).squeeze(-1)
            offset=centers[:,sl]-pos[a][:,body]
            displacement=twist[:,:,:3]+torch.cross(twist[:,:,3:],offset,dim=-1)
            displacement=torch.where((rows>=0)[None,:,None],displacement,torch.zeros_like(displacement))
            predicted[:,sl]+=displacement
        # Translational prediction with angular-motion envelope inflation.
        extent=obj.size+2*obj.angular_velocity.norm(dim=-1,keepdim=True)*self.horizon_s*(obj.size/2).norm(dim=-1,keepdim=True)
        future=ObjectContext(obj.names,obj.position+obj.linear_velocity*self.horizon_s,obj.quaternion,extent,obj.linear_velocity,obj.angular_velocity)
        out=object_risk(centers,predicted,env._sph.radii,obj,future,self.allowed,self.clearance_m)
        locks=torch.stack([out['risk'][:,env._sph._arm_slices[a]].flatten(1).any(1) for a in self.arms],dim=1)
        # Shared rigid payload: stopping only one carrier can wrench the grasp.
        for name in set(self.assignments.values()):
            ids=[self.arms.index(a) for a,o in self.assignments.items() if o==name]
            locks[:,ids]=locks[:,ids].any(1,keepdim=True)
        forbidden=out['margin'].masked_fill(self.allowed,float('inf'))
        flat_index=int(forbidden[0].argmin())
        min_pair=[env._sph.qualified_names[flat_index//len(obj.names)],obj.names[flat_index%len(obj.names)]]
        return locks, {'guard_version':'object_guard.v2_assigned_gripper','min_pair_env0':min_pair,'schema':SCHEMA,'source':'simulator_ground_truth','min_forbidden_margin_m':float(forbidden.min()),'object_features':obj.features().detach().cpu().tolist(),'locked':locks.detach().cpu().tolist()}
