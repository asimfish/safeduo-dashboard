"""Every saved frame/window: first alarm, predecessor timing and exposure."""
from pathlib import Path
import hashlib,json
import numpy as np
from queue_guard import issuance_status
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def first(mask):
    ids=np.flatnonzero(mask);return int(ids[0]) if len(ids) else None


def main():
    reg=json.loads((HERE/'DIAGNOSTIC_REGISTRATION.json').read_text());old=Path(reg['prior_root'])
    for p,v in reg['sources'].items():assert sha(p)==v
    records=[];bindings={};traces=[]
    for plan_path in sorted((old/'plans').glob('holdout_*_plan.json')):
        plan=json.loads(plan_path.read_text());root=Path(plan['output_root'])
        for job in plan['jobs']:
            path=root/job['id'];c=path/'cell_001.npz';d=path/'project_diagnostics.npz'
            bindings[str(c)]=sha(c);bindings[str(d)]=sha(d)
            with np.load(c,allow_pickle=False) as z:
                keys=['margins','pre_qd_compact','pre_target_debt','pre_pending_actuator_targets','q','q_initial']
                cell={k:z[k].copy() for k in keys}
            with np.load(d,allow_pickle=False) as z:
                keys=[f'{prefix}_{r}' for prefix in ['target_safety_residual','target_alpha_residual','target_bound_residual','individual_infeasibility_lower_bound'] for r in ['F','U']]
                diag={k:z[k].copy() for k in keys}
            assert cell['margins'].shape==(960,64,4)
            for e in range(64):
                channels={prefix:np.maximum(diag[prefix+'_F'][:,e],diag[prefix+'_U'][:,e]) for prefix in ['target_safety_residual','target_alpha_residual','target_bound_residual','individual_infeasibility_lower_bound']}
                status=issuance_status(*channels.values());warning=status['unmet_selected_constraints'] | status['individual_box_impossibility']
                margin=cell['margins'][:,e].min(axis=-1);fail=first(margin<np.float32(0));alarm=first(warning)
                before=warning[:fail+1] if fail is not None else warning
                safety_alarm=first(channels['target_safety_residual']>1e-6)
                row=dict(condition=job['id'],mode=job['env']['SAFEDUO_JOINT_MODE'],seed=job['expected']['seed'],env=e,
                         first_strict_step=fail,first_alarm_step=alarm,first_safety_residual_step=safety_alarm,
                         alarm_by_first_strict=bool(fail is not None and before.any()),
                         lead_steps=None if fail is None or not before.any() else fail-int(np.flatnonzero(before)[0]),
                         warned_env_steps=int(warning.sum()),strict_env_steps=int((margin<0).sum()),
                         warning_only_window=bool(fail is None and warning.any()))
                records.append(row)
                if fail is not None:
                    ids=set(np.linspace(0,959,65,dtype=int).tolist())
                    for t in [fail,alarm]:
                        if t is not None:ids.update(range(max(0,t-12),min(960,t+13)))
                    ix=np.array(sorted(ids));preq=np.concatenate([cell['q_initial'][None,e],cell['q'][:-1,e]],axis=0)
                    debt=np.abs(cell['pre_pending_actuator_targets'][:,:,e]-preq[:,None]).max(axis=(1,2))
                    traces.append(dict(**row,indices=ix.tolist(),margin_mm=(margin[ix]*1000).tolist(),
                        max_qd_rad_s=np.abs(cell['pre_qd_compact'][ix,e]).max(-1).tolist(),
                        max_issued_debt_rad=np.abs(cell['pre_target_debt'][ix,e]).max(-1).tolist(),
                        max_pending_debt_rad=debt[ix].tolist(),
                        target_safety_residual_m=channels['target_safety_residual'][ix].tolist(),
                        individual_lower_bound_m=channels['individual_infeasibility_lower_bound'][ix].tolist(),
                        alarm=warning[ix].tolist()))
            assert sha(c)==bindings[str(c)] and sha(d)==bindings[str(d)]
            print('CLOSED_WARNINGS',job['id'],flush=True)
    summary=[]
    for mode in reg['modes']:
        group=[r for r in records if r['mode']==mode];failed=[r for r in group if r['first_strict_step'] is not None]
        summary.append(dict(mode=mode,windows=len(group),strict_windows=len(failed),
                       warned_by_first_strict=sum(r['alarm_by_first_strict'] for r in failed),
                       warned_before_first_strict=sum(r['lead_steps'] is not None and r['lead_steps']>0 for r in failed),
                       warning_only_windows=sum(r['warning_only_window'] for r in group),
                       warned_env_steps=sum(r['warned_env_steps'] for r in group),
                       minimum_lead_steps=min((r['lead_steps'] for r in failed if r['lead_steps'] is not None),default=None)))
    assert len(records)==576 and len(traces)==211
    for p,v in reg['sources'].items():assert sha(p)==v
    with (HERE/'WARNING_AUDIT.json').open('x') as f:
        json.dump(dict(status='PASS_ALL_SAVED_WINDOWS_AND_FRAMES',windows=576,env_frames=576*960,summary=summary,records=records,
                       bindings=bindings,projection_tolerance=1e-6,geometry_strict_epsilon=0,
                       scope='Retrospective shadow alarm; pre-step residual before post[t] margin. Lead0 means same step before physics, not an earlier frame. Warning-only windows are not proven false positives. No interventions executed.',physical_safety_certified=False),f,indent=2);f.write('\n')
    with (HERE/'FAILURE_TRACES.json').open('x') as f:json.dump(dict(traces=traces),f,separators=(',',':'));f.write('\n')


if __name__=='__main__':main()
