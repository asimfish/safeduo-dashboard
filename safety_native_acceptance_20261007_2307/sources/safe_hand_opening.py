"""Evaluation-local common U-thumb .35rad reset and actual actuator goal.

No runtime pose write, retreat, force exemption or feedback adaptation. Default
reset pose is changed before window reset, and every reset writes the matching
hand target buffer. Constructor internal contacts remain unobserved/unqualified.
"""
import torch
OPEN_THUMB_RAD=.35

def install_open_hand_defaults(env):
 if getattr(env,'_native_open_hand_installed',False):raise ValueError('opening already installed')
 plans={}
 for arm,name in [('U_L','left_thumb_1_joint'),('U_R','right_thumb_1_joint')]:
  art=env._arms[arm];names=art.joint_names
  if names.count(name)!=1:raise ValueError('exact thumb joint missing/duplicate '+arm)
  idx=names.index(name)
  if idx in env._joint_idx[arm].tolist():raise ValueError('thumb overlaps original controlled arm')
  limits=art.data.soft_joint_pos_limits[:,idx];native=art.root_physx_view.get_dof_limits()[:,idx]
  for x in [limits,native]:
   if not torch.isfinite(x).all() or (x[:,0]>OPEN_THUMB_RAD).any() or (x[:,1]<OPEN_THUMB_RAD).any():raise ValueError('calibrated opening outside hand limits')
  plans[arm]=idx
 original=env._reset_idx
 for arm,idx in plans.items():
  art=env._arms[arm];art.data.default_joint_pos[:,idx]=OPEN_THUMB_RAD;art.data.default_joint_vel[:,idx]=0.
  art.set_joint_position_target(torch.full((env.num_envs,1),OPEN_THUMB_RAD,device=env.device,dtype=art.data.default_joint_pos.dtype),joint_ids=[idx])
 def reset(ids):
  result=original(ids)
  for arm,idx in plans.items():
   art=env._arms[arm];target=torch.full((len(ids),1),OPEN_THUMB_RAD,device=env.device,dtype=art.data.default_joint_pos.dtype)
   art.set_joint_position_target(target,joint_ids=[idx],env_ids=ids)
  return result
 env._reset_idx=reset;env._native_open_hand_installed=True
 return dict(open_thumb_rad=OPEN_THUMB_RAD,joint_indices=plans,requested_joint_names={'U_L':'left_thumb_1_joint','U_R':'right_thumb_1_joint'},all_modes_common=True,controlled_arm_targets_changed=False,original_actor_FIF06_scores_exemptions_gains_self_collision_unchanged=True,constructor_contacts_certified=False,init_scope='default pose/velocity and target applied by reset before scoring; never q/root rebase inside a window')
