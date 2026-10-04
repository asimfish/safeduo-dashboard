"""Measure all candidate initial poses; build an explicitly conditioned bank."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from wide_random import make_initial
from safeduo.eval.research_battery import cell_seed


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--ckpt',required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    app=AppLauncher(args).app
    from safeduo.envs.duo_env import DuoEnv,make_duo_env_cfg
    from safeduo.eval.endurance_eval import ckpt_arm_aware,ckpt_p2_obs
    from safeduo.safety.types import ARM_KEYS,zeros_delta
    cfg=make_duo_env_cfg(num_envs=64,device=args.device,yaml_name='duo_env_a31_pending_guard.yaml',
                         coordinator=True,arm_aware_obs=ckpt_arm_aware(args.ckpt),p2_obs=ckpt_p2_obs(args.ckpt))
    cfg.coordinator['terminate_on_violation']=False
    env=DuoEnv(cfg)
    class PoseSource:
        def reset(self,ids,generator=None):pass
        def sample(self,state):return zeros_delta(64,device=env.device)
        def initial_positions(self,ids):return ids,{a:q[ids] for a,q in self.positions.items()}
    source=PoseSource();env._delta_src=source
    baseline=torch.cat([env._arms[a].data.default_joint_pos[:,env._joint_idx[a]] for a in ARM_KEYS],-1)
    limits=torch.cat([env._q_soft_limits[a] for a in ARM_KEYS],-2)
    try:
        for seed in [13447771,27180353,48921161]:
            all_q=[];all_margin=[];all_bad=[];chosen=[];selected=0
            for batch in range(200):
                q,_=make_initial(baseline,limits,'global_lhs',cell_seed(seed,'feasible_pose_batch',batch))
                source.positions=dict(zip(ARM_KEYS,q.split([7,7,6,6],-1)))
                env.reset()
                state=env.scene_state();actual=torch.cat([state.q[a] for a in ARM_KEYS],-1)
                assert torch.equal(actual,q),'initial reset pose differs from candidate'
                table=env._sph.last_table_margin.masked_fill(env._sph.last_table_viol_exempt,float('inf')).amin(-1)
                margins=torch.stack([env._last_out.min_margin[k] for k in ['cross','self_F','self_U','table']],-1)
                margins[:,-1]=table
                bad=env._last_out.violation
                valid=(margins.amin(-1)>=.0001)&~bad
                idx=torch.where(valid)[0].cpu().tolist()
                use=idx[:64-selected]
                chosen.extend([batch*64+i for i in use]);selected+=len(use)
                all_q.append(actual.cpu().numpy());all_margin.append(margins.cpu().numpy());all_bad.append(bad.cpu().numpy())
                print(f'BANK seed={seed} batch={batch+1} selected={selected}/64',flush=True)
                if selected==64:break
            q=np.concatenate(all_q);margins=np.concatenate(all_margin);bad=np.concatenate(all_bad)
            mask=np.zeros(len(q),dtype=bool);mask[chosen]=True
            dest=args.out/str(seed);dest.mkdir()
            np.savez_compressed(dest/'bank.npz',accepted_q=q[mask],accepted_margins=margins[mask],
                                all_candidate_q=q,all_candidate_margins=margins,all_candidate_violation=bad,
                                selected_mask=mask,selected_indices=np.array(chosen),joint_soft_limits=limits.cpu().numpy())
            meta=dict(seed=seed,status='complete' if selected==64 else 'failed',candidate_count=len(q),
                      initial_violation_count=int(bad.sum()),eligible_count=int(((margins.min(-1)>=.0001)&~bad).sum()),
                      selected_count=selected,conditioning='all nonexempt initial sphere margins >=0.1mm; no initial violation',
                      sampled_joint_fraction=[.025,.975],policy_outcomes_used=False,
                      bank_sha256=hashlib.sha256((dest/'bank.npz').read_bytes()).hexdigest())
            (dest/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
            if selected!=64:raise ValueError(f'bank seed={seed} insufficient accepted poses: {selected}/64')
    finally:
        env.close();app.close()


if __name__=='__main__':main()
