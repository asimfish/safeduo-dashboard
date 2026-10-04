"""Geometry-only risk searches plus preregistered zero-input admission."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'safety_random_space_20261004'))
from wide_random import generator,make_initial
from safeduo.eval.research_battery import cell_seed

SEEDS=[60317411,80692357,109441003]
PAIR_INDICES=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
SLICES=[slice(0,7),slice(7,14),slice(14,20),slice(20,26)]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--ckpt',required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    app=AppLauncher(args).app
    from safeduo.envs.duo_env import DuoEnv,make_duo_env_cfg
    from safeduo.eval.endurance_eval import ckpt_arm_aware,ckpt_p2_obs,_RawShim
    from safeduo.eval.random_battery import PairMarginProbe
    from safeduo.safety.types import ARM_KEYS,zeros_delta
    cfg=make_duo_env_cfg(num_envs=64,device=args.device,yaml_name='duo_env_a31_pending_guard.yaml',coordinator=True,
                         arm_aware_obs=ckpt_arm_aware(args.ckpt),p2_obs=ckpt_p2_obs(args.ckpt))
    cfg.coordinator['terminate_on_violation']=False;cfg.episode_length_s=30;cfg.seed=0
    env=DuoEnv(cfg);env._backstop=_RawShim(env._backstop);probe=PairMarginProbe(env._sph)
    class Source:
        def reset(self,ids,generator=None):pass
        def sample(self,state):return zeros_delta(64,device=env.device)
        def initial_positions(self,ids):return ids,{a:q[ids] for a,q in self.positions.items()}
    source=Source();env._delta_src=source
    baseline=torch.cat([env._arms[a].data.default_joint_pos[:,env._joint_idx[a]] for a in ARM_KEYS],-1)
    limits=torch.cat([env._q_soft_limits[a] for a in ARM_KEYS],-2)
    lo=limits.cpu()[...,0];span=limits.cpu()[...,1]-lo
    actions=torch.ones((64,5),device=env.device)
    history=[];settling=[]

    def geometry(q,identity):
        source.positions=dict(zip(ARM_KEYS,q.to(env.device).split([7,7,6,6],-1)));env._pending_cmd=None;env.reset()
        state=env.scene_state();actual=torch.cat([state.q[a] for a in ARM_KEYS],-1)
        assert torch.equal(actual,q.to(env.device))
        margins=torch.stack([env._last_out.min_margin[k] for k in ['cross','self_F','self_U','table']],-1)
        raw_table=env._sph.last_table_margin.amin(-1)
        margins[:,-1]=env._sph.last_table_margin.masked_fill(env._sph.last_table_viol_exempt,float('inf')).amin(-1)
        pairs=probe.pair_min();minimum=torch.minimum(margins.amin(-1),raw_table)
        valid=(minimum>=.001)&~env._last_out.violation
        record=dict(q=q.cpu().numpy(),margins=margins.cpu().numpy(),table=raw_table.cpu().numpy(),pairs=pairs.cpu().numpy(),
                    eligible=valid.cpu().numpy(),identity=identity)
        history.append(record)
        return valid.cpu(),pairs.cpu(),minimum.cpu(),len(history)-1

    def settle(q,eligible,identity):
        seen_bad=torch.zeros(64,dtype=torch.bool,device=env.device);drift=torch.zeros(64,device=env.device)
        minimum=torch.full((64,),float('inf'),device=env.device)
        expected=q.to(env.device)
        for t in range(60):
            env.step(actions)
            current=torch.cat([env.scene_state().q[a] for a in ARM_KEYS],-1)
            assert torch.equal(torch.cat([env._targets[a] for a in ARM_KEYS],-1),expected),'zero target changed'
            assert all(not env._step_cache['cmd'].delta_q[a].any() and not env._step_cache['exec'].delta_q[a].any() for a in ARM_KEYS)
            drift=torch.maximum(drift,(current-expected).abs().amax(-1))
            table=env._sph.last_table_margin.amin(-1)
            seen_bad|=env._last_out.violation|(table<0)
            minimum=torch.minimum(minimum,table)
        velocity=torch.cat([env.scene_state().qd[a] for a in ARM_KEYS],-1).abs().amax(-1)
        valid=eligible&~seen_bad.cpu()&(drift.cpu()<=.05)&(velocity.cpu()<=.1)
        settling.append(dict(identity=identity,eligible=eligible.numpy(),valid=valid.numpy(),bad=seen_bad.cpu().numpy(),
                             drift=drift.cpu().numpy(),velocity=velocity.cpu().numpy(),min_table=minimum.cpu().numpy()))
        return valid

    try:
        for seed in SEEDS:
            selected=[];labels=[];refs=[];quotas=[]
            for pair,(a,b) in enumerate(PAIR_INDICES):
                found=0
                for restart in range(12):
                    gen=generator(cell_seed(seed,f'risk_{pair}_{restart}',0),'risk_search')
                    other=(baseline.cpu()[0]+(torch.rand(26,generator=gen)*2-1)*.3).clamp(lo[0],lo[0]+span[0])
                    mean=((baseline.cpu()[0]-lo[0])/span[0]).clone();sigma=torch.full((26,),.25)
                    active=torch.zeros(26,dtype=torch.bool);active[SLICES[a]]=True;active[SLICES[b]]=True
                    chosen=False
                    for iteration in range(40):
                        u=(mean+torch.randn((64,26),generator=gen)*sigma).clamp(.025,.975)
                        u[:16]=(torch.rand((16,26),generator=gen)*.95+.025)
                        q=lo+u*span;q[:,~active]=other[~active]
                        identity=[seed,pair,restart,iteration]
                        valid,pairs,minimum,ref=geometry(q,identity)
                        band=(pairs[:,pair]>=.020)&(pairs[:,pair]<=.060)
                        eligible=valid&band
                        if eligible.any():
                            accepted=settle(q,eligible,identity)
                            if accepted.any():
                                idx=int(torch.where(accepted)[0][0]);selected.append(q[idx].numpy());labels.append(pair);refs.append([ref,idx]);found+=1;chosen=True
                                print(f'RISK seed={seed} pair={pair} restart={restart} iter={iteration} found={found}/8',flush=True)
                                break
                        score=(pairs[:,pair]-.040).abs()+20*(.001-minimum).clamp_min(0)
                        elite=u[torch.topk(score,8,largest=False).indices]
                        mean=.25*mean+.75*elite.mean(0);sigma=(.25*sigma+.75*elite.std(0,unbiased=False)).clamp_min(.01)
                    if not chosen:print(f'RISK seed={seed} pair={pair} restart={restart} no admissible pose',flush=True)
                    if found==8:break
                quotas.append(found)
            general=0
            for batch in range(100):
                q,_=make_initial(baseline,limits,'global_lhs',cell_seed(seed,'general_stable',batch));q=q.cpu()
                identity=[seed,-1,batch,0];eligible,_,_,ref=geometry(q,identity)
                if eligible.any():
                    accepted=settle(q,eligible,identity)
                    for idx in torch.where(accepted)[0].tolist()[:16-general]:
                        selected.append(q[idx].numpy());labels.append(-1);refs.append([ref,idx]);general+=1
                if general==16:break
            dest=args.out/str(seed);dest.mkdir()
            np.savez_compressed(dest/'bank.npz',accepted_q=np.stack(selected) if selected else np.empty((0,26),np.float32),
                                risk_pair_index=np.array(labels),selected_refs=np.array(refs),joint_soft_limits=limits.cpu().numpy())
            meta=dict(seed=seed,status='complete' if quotas==[8]*6 and general==16 else 'quota_failed',
                      risk_quotas=quotas,general_count=general,selected_count=len(selected),
                      policy_outcomes_used=False,conditioning='all table and robot margins >=1mm; raw zero hold1s no violation, drift<=.05rad, final velocity<=.1rad/s',
                      risk_band_m=[.020,.060],bank_sha256=hashlib.sha256((dest/'bank.npz').read_bytes()).hexdigest())
            (dest/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
            print(json.dumps(meta),flush=True)
        np.savez_compressed(args.out/'all_geometry.npz',q=np.concatenate([r['q'] for r in history]),
                            margins=np.concatenate([r['margins'] for r in history]),table=np.concatenate([r['table'] for r in history]),
                            pairs=np.concatenate([r['pairs'] for r in history]),eligible=np.concatenate([r['eligible'] for r in history]),
                            batch_identity=np.array([r['identity'] for r in history]))
        np.savez_compressed(args.out/'all_settling.npz',identity=np.array([r['identity'] for r in settling]),
                            **{key:np.stack([r[key] for r in settling]) for key in ['eligible','valid','bad','drift','velocity','min_table']})
        (args.out/'sampling_summary.json').write_text(json.dumps(dict(geometry_candidates=len(history)*64,settling_candidates=len(settling)*64,
                                                                      random_policy_windows=0,all_candidates_saved=True))+'\n')
    finally:env.close();app.close()


if __name__=='__main__':main()
