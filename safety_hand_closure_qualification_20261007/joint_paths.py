"""Registered diagnostic joint paths; no production admission claim."""
import torch
from safeduo.safety.types import ARM_KEYS

class JointPaths:
    def __init__(self, env, registration):
        self.env, self.r = env, registration
        self.ids, self.open_q, self.far_q = {}, {}, {}
        self.original_open = {}; self.cur={}
        for arm in ARM_KEYS:
            art=env._arms[arm]
            ids=registration['hand_ids'][arm]
            names=[art.joint_names[i] for i in ids]
            assert names==registration['hand_names'][arm]
            far=art.data.soft_joint_pos_limits[0,ids,1].clone()
            opened=torch.zeros_like(far)
            if arm.startswith('U'): opened[names.index(('left' if arm=='U_L' else 'right')+'_thumb_1_joint')]=.35
            self.ids[arm]=(ids,torch.tensor(ids,device=env.device))
            self.open_q[arm]=opened;self.far_q[arm]=far;self.original_open[arm]=torch.zeros_like(far)
            self.cur[arm]=0.
        self.goal={};self.target={}

    def step(self, step):
        cycle=step//self.r['cycle_steps']; phase=step%self.r['cycle_steps']
        dt=self.r['physics_dt_s']; time=phase*dt
        for arm in ARM_KEYS:
            art=self.env._arms[arm];ids=self.ids[arm][0]
            opened=self.open_q[arm][None].expand(self.env.num_envs,-1)
            goal=opened.clone()
            for e,c in enumerate(self.r['cases']):
                profile=self.r['profiles'][cycle*self.r['profiles_per_cycle']+c['profile_slot']]
                if arm.startswith('U'):
                    names=self.r['hand_names'][arm]
                    fraction=profile['finger_fraction']
                    g=self.far_q[arm]*fraction
                    if profile['kind']=='original_task':
                        thumb=.7569 if arm=='U_L' else .7546
                        g=torch.where(torch.tensor(['thumb' in x for x in names],device=self.env.device),self.far_q[arm]*thumb,g)
                        g=torch.where(torch.tensor(['thumb' not in x for x in names],device=self.env.device),self.far_q[arm]*(.4769 if arm=='U_L' else .4746),g)
                    else:
                        g[names.index(('left' if arm=='U_L' else 'right')+'_thumb_1_joint')]=profile['thumb1_fraction']*self.far_q[arm][names.index(('left' if arm=='U_L' else 'right')+'_thumb_1_joint')]
                        g[names.index(('left' if arm=='U_L' else 'right')+'_thumb_2_joint')]=profile['thumb2_fraction']*self.far_q[arm][names.index(('left' if arm=='U_L' else 'right')+'_thumb_2_joint')]
                    if profile['kind']=='open_control':g=opened[e].clone()
                    goal[e]=g
            # 1s open; 1s close; 1.5s closed; 1s open; 1.5s open.
            if time<1:target=opened
            elif time<2:target=opened+min(1.,(time-1+dt)/self.r['ramp_s'])*(goal-opened)
            elif time<3.5:target=goal
            elif time<4.5:target=goal+min(1.,(time-3.5+dt)/self.r['ramp_s'])*(opened-goal)
            else:target=opened
            self.goal[arm]=goal;self.target[arm]=target.clone()
            art.set_joint_position_target(target,joint_ids=ids)
