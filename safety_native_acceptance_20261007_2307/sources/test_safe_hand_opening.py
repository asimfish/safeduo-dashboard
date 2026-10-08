import torch,copy
from types import SimpleNamespace as S
from safe_hand_opening import install_open_hand_defaults
class Arm:
 def __init__(self,names):
  self.joint_names=names;self.data=S(default_joint_pos=torch.zeros(2,12),default_joint_vel=torch.ones(2,12),soft_joint_pos_limits=torch.stack([torch.zeros(2,12),torch.ones(2,12)],-1));self.root_physx_view=S(get_dof_limits=lambda:self.data.soft_joint_pos_limits);self.targets=torch.zeros(2,12)
 def set_joint_position_target(self,goal,joint_ids,env_ids=None):
  ids=torch.arange(2) if env_ids is None else env_ids
  self.targets[ids[:,None],torch.tensor(joint_ids)[None,:]]=goal

def env():
 arms={a:Arm([str(i) for i in range(10)]+[name,'other']) for a,name in [('U_L','left_thumb_1_joint'),('U_R','right_thumb_1_joint')]}
 e=S(_arms=arms,_joint_idx={a:torch.arange(6) for a in arms},num_envs=2,device='cpu');e._reset_idx=lambda ids:None;return e
x=env();before={a:art.data.default_joint_pos.clone() for a,art in x._arms.items()};report=install_open_hand_defaults(x)
for a,art in x._arms.items():
 assert torch.equal(art.data.default_joint_pos[:,:6],before[a][:,:6]);assert (art.data.default_joint_pos[:,10]==torch.tensor(.35)).all();assert (art.targets[:,10]==torch.tensor(.35)).all()
 art.targets[:,10]=0
x._reset_idx(torch.tensor([1]));assert all(art.targets[1,10]==torch.tensor(.35) and art.targets[0,10]==0 for art in x._arms.values())
for kind in ['missing','limit','controlled']:
 x=env();old=x._arms['U_L'].data.default_joint_pos.clone()
 if kind=='missing':x._arms['U_R'].joint_names[10]='wrong'
 if kind=='limit':x._arms['U_R'].data.soft_joint_pos_limits[:,10,1]=.3
 if kind=='controlled':x._joint_idx['U_R']=torch.tensor([10])
 try:install_open_hand_defaults(x)
 except ValueError:pass
 else:raise AssertionError('invalid plan accepted '+kind)
 assert torch.equal(x._arms['U_L'].data.default_joint_pos,old),'partial mutation before validation'
print('PASS unchanged arm6, actual reset target, subset reset, missing/limits/overlap reject atomically; no actual Isaac assertion')
