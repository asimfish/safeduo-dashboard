"""Prospective full-soft-limit proposals; retain every geometry rejection."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

H = Path(__file__).resolve().parent
OLD = Path('/home/liyufeng/safeduo/artifacts/safety_velocity_arrival_20261008_1519')
sys.path.insert(0, str(OLD))
ARMS = ('F_L', 'F_R', 'U_L', 'U_R')

class ZeroSource:
    def reset(self, ids, generator=None): pass
    def sample(self, state):
        from safeduo.safety.types import DeltaCmd
        return DeltaCmd({a: torch.zeros_like(state.q[a]) for a in ARMS})

def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=False)
    reg = json.loads((H/'REGISTRATION.json').read_text())
    p = Path(reg['raw'])/'hand_root_fixture.npz'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == reg['fixture_sha256']
    with np.load(p) as z: fixture = {k:z[k] for k in z.files}
    app = AppLauncher(args).app
    env = None
    receipt = dict(status='running', policy_outcomes_used=False, physics_steps_after_reset=0,
                   constructor_contacts_qualified=False, safety_acceptance=False)
    try:
        from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg
        from safe_hand_opening import install_open_hand_defaults
        from native_state_capture import array
        cfg = make_duo_env_cfg(num_envs=64, device=args.device,
            yaml_name='duo_env_a31_pending_guard.yaml', coordinator=True, arm_aware_obs=True, p2_obs=True)
        cfg.coordinator['terminate_on_violation'] = False
        env = DuoEnv(cfg); install_open_hand_defaults(env)
        env._delta_src = ZeroSource(); env.reset()
        base, limits, names = {}, {}, {}
        for a in ARMS:
            art = env._arms[a]; idx = env._joint_idx[a]
            assert np.array_equal(np.asarray(art.joint_names), fixture[a+'_native_joint_names'])
            q = torch.as_tensor(fixture[a+'_native_q'][0],device=env.device)[None].expand(64,-1).clone()
            root = torch.as_tensor(fixture[a+'_native_root_xyzw'][0],device=env.device)[None].expand(64,-1).clone()
            root[:,:3] += env.scene.env_origins - torch.as_tensor(fixture['source_origin'][0],device=env.device)
            art.write_root_pose_to_sim(torch.cat([root[:,:3],root[:,6:7],root[:,3:6]],-1))
            art.write_root_velocity_to_sim(torch.zeros((64,6),device=env.device))
            base[a] = q
            limits[a] = art.data.soft_joint_pos_limits[:,idx].clone()
            names[a] = np.asarray(art.body_names)
        rng = np.random.default_rng(reg['proposal_seed'])
        proposed, margins, class_margins, selected, body_positions, body_quats = [], [], [], [], [], []
        selected_native = {a:[] for a in ARMS}
        selected_d = []; all_eligible = []
        selected_count = 0
        for batch in range(reg['maximum_batches']):
            unit = rng.uniform(-1,1,(64,26)).astype(np.float32)
            controlled = []; offset = 0
            for a in ARMS:
                art = env._arms[a]; idx=env._joint_idx[a]; n=len(idx)
                q=base[a].clone(); l=limits[a]
                q[:,idx] = l.mean(-1) + torch.as_tensor(unit[:,offset:offset+n],device=env.device)*(l[...,1]-l[...,0])/2
                art.write_joint_state_to_sim(q,torch.zeros_like(q))
                art.set_joint_position_target(q[:,idx],joint_ids=idx)
                controlled.append(array(q[:,idx]));offset+=n
            env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
            d=env.compute_dist(); gaps=array(d.dists); mins=gaps.min(1)
            class_id=array(env._sph.class_id)
            cm=np.stack([gaps[:,class_id==k].min(1) for k in sorted(set(class_id.tolist()))],-1)
            qactual=np.concatenate([array(env._arms[a].root_physx_view.get_dof_positions()[:,env._joint_idx[a]]) for a in ARMS],-1)
            assert np.array_equal(qactual,np.concatenate(controlled,-1))
            eligible=np.flatnonzero(mins>=.0001); use=eligible[:reg['requested_selected']-selected_count]
            proposed.append(qactual);margins.append(mins);class_margins.append(cm);all_eligible.extend((batch*64+eligible).tolist())
            if len(use):
                selected.extend((batch*64+use).tolist());selected_d.append(gaps[use])
                for a in ARMS:selected_native[a].append(array(env._arms[a].root_physx_view.get_dof_positions())[use])
                body_positions.append({a:array(env._arms[a].data.body_pos_w)[use]-array(env.scene.env_origins)[use,None,:] for a in ARMS})
                body_quats.append({a:array(env._arms[a].data.body_quat_w)[use] for a in ARMS})
                selected_count+=len(use)
            print('BANK',batch+1,'proposals',(batch+1)*64,'selected',selected_count,'/',reg['requested_selected'],flush=True)
            if selected_count==reg['requested_selected']:break
        qall=np.concatenate(proposed);selected=np.asarray(selected,dtype=np.int64)
        fields=dict(all_proposal_q=qall,all_min_raw_gap_m=np.concatenate(margins),all_class_min_raw_gap_m=np.concatenate(class_margins),
            selected_indices=selected,all_eligible_indices=np.asarray(all_eligible),accepted_q=qall[selected],
            soft_limits=np.concatenate([array(limits[a][0]) for a in ARMS]),class_ids=np.unique(class_id),
            accepted_d=np.concatenate(selected_d) if len(selected) else np.empty((0,9021)))
        for a in ARMS:
            fields[a+'_native_q']=np.concatenate(selected_native[a]) if selected_native[a] else np.empty((0,len(base[a][0])))
            fields[a+'_body_names']=names[a]
            fields[a+'_body_pos_local']=np.concatenate([x[a] for x in body_positions]) if body_positions else np.empty((0,len(names[a]),3))
            fields[a+'_body_quat_wxyz']=np.concatenate([x[a] for x in body_quats]) if body_quats else np.empty((0,len(names[a]),4))
        np.savez_compressed(out/'bank.npz',**fields)
        receipt.update(status='complete' if selected_count==reg['requested_selected'] else 'insufficient_eligible',
            candidate_count=len(qall),eligible_count=len(all_eligible),selected_count=selected_count,
            invalid_raw_geometry_count=int((fields['all_min_raw_gap_m']<0).sum()),
            bank_sha256=hashlib.sha256((out/'bank.npz').read_bytes()).hexdigest(),sampling_support='full native soft limits in all 26 arm coordinates',
            selection='first qualifying raw geometry proposals, no outcome or exemption filtering',
            accepted_min_raw_gap_m=float(fields['accepted_d'].min()) if len(selected) else None)
    except BaseException as e:
        import traceback
        receipt.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        (out/'sampling_summary.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if env is not None:env.close()
        app.close()

if __name__=='__main__':main()
