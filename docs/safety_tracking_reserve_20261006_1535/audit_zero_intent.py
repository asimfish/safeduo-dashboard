"""Exploratory post-observation audit, no control tuning or new method denominator."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
from reference_envelope import reference_bounds


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert json.loads((HERE/'NUMERIC_EXECUTION.json').read_text())['status']=='PASS_ALL_NUMERIC_CLOSED'
    rows=[];replays=[]
    for plan_path in sorted((HERE/'plans').glob('holdout_*_plan.json')):
        plan=json.loads(plan_path.read_text());root=Path(plan['output_root'])
        assert json.loads((root/'campaign.json').read_text())['status']=='complete'
        for job in plan['jobs']:
            path=root/job['id'];file=path/'cell_001.npz';digest=sha(file)
            with np.load(file,allow_pickle=False) as source:
                data={k:source[k].copy() for k in ['cmd','exec','effective_target_delta','q','q_initial','controller_target','official_margins']}
            assert not data['cmd'][:60].any()
            with np.load(path/'project_diagnostics.npz',allow_pickle=False) as source:
                diag={k:source[k].copy() for k in ['bounds_lower','bounds_upper','project_input_cmd']}
            forced=((diag['bounds_lower'][:60]>0)|(diag['bounds_upper'][:60]<0)).any(-1)
            moved=data['effective_target_delta'][:60].any(-1)
            bad=(data['official_margins'][:60]<0).any(-1)
            rows.append(dict(mode=job['env']['SAFEDUO_JOINT_MODE'],seed=job['expected']['seed'],zero_prefix_frames=60,windows=64,
                zero_prefix_strict_windows=int(bad.any(0).sum()),zero_prefix_strict_env_steps=int(bad.sum()),
                bounds_excluding_zero_env_steps=int(forced.sum()),nonzero_effective_target_env_steps=int(moved.sum()),
                max_zero_prefix_target_increment_rad=float(np.abs(data['effective_target_delta'][:60]).max()),
                zero_prefix_target_displacement_from_initial_max_rad=float(np.abs(data['controller_target'][:60]-data['q_initial']).max()),
                raw_cell_sha256=digest,classification='observational actual closed own trajectory; no unique physical cause'))
            receipt=json.loads((path/'first_failure_receipts.json').read_text())
            for cap in receipt['receipts']:
                if cap['step']>=60:continue
                snapshot=path/cap['path'];assert sha(snapshot)==cap['sha256']
                with np.load(snapshot,allow_pickle=False) as source:s={k:source[k].copy() for k in source.files}
                p=json.loads((path/'protocol.json').read_text());box=float(p['effective_backstop']['vmax'])*p['dt']
                with np.load(Path(job['env']['SAFEDUO_INITIAL_BANK_NPZ']),allow_pickle=False) as source:limits=source['joint_soft_limits'].copy()
                for index,e in enumerate(s['env_ids']):
                    assert not s['external_raw_cmd'][index].any()
                    q=torch.from_numpy(s['pre_q'][index:index+1]);target=torch.from_numpy(s['pre_issued_target'][index:index+1]);lo=torch.from_numpy(limits[int(e):int(e)+1,:,0]);hi=torch.from_numpy(limits[int(e):int(e)+1,:,1])
                    old=reference_bounds(q,target,lo,hi,box,.050)
                    tight=reference_bounds(q,target,lo,hi,box,.010)
                    raw=torch.zeros_like(q)
                    old_cmd=raw.maximum(old[0]).minimum(old[1]);tight_cmd=raw.maximum(tight[0]).minimum(tight[1])
                    changed=bool(tight_cmd.any());old_zero=not bool(old_cmd.any())
                    replays.append(dict(mode=job['env']['SAFEDUO_JOINT_MODE'],seed=job['expected']['seed'],env=int(e),step=cap['step'],snapshot_sha256=cap['sha256'],
                        actual_bounds_exclude_zero=bool(((s['bounds_lower'][index]>0)|(s['bounds_upper'][index]<0)).any()),
                        same_saved_state_050_prelimited_rawzero_unchanged=old_zero,same_saved_state_010_prelimited_rawzero_nonzero=changed,
                        tight_prelimit_max_abs_rad=float(tight_cmd.abs().max()),scope='exact saved pre-state CPU helper .050 versus.010 command-bound replay; no alternate physical simulation or unique cause'))
            assert sha(file)==digest
    summary=[]
    for mode in ['joint_reference','tight_reference','delay_reserve']:
        group=[r for r in rows if r['mode']==mode]
        summary.append(dict(mode=mode,windows=192,zero_prefix_strict_windows=sum(r['zero_prefix_strict_windows'] for r in group),
            zero_prefix_strict_env_steps=sum(r['zero_prefix_strict_env_steps'] for r in group),bounds_excluding_zero_env_steps=sum(r['bounds_excluding_zero_env_steps'] for r in group),
            nonzero_effective_target_env_steps=sum(r['nonzero_effective_target_env_steps'] for r in group),maximum_target_drift_rad=max(r['zero_prefix_target_displacement_from_initial_max_rad'] for r in group)))
    with (HERE/'ZERO_INTENT_AUDIT.json').open('x') as f:
        json.dump(dict(status='PASS_CLOSED_EXPLORATORY_ZERO_INTENT_TRACE_AUDIT',summary=summary,rows=rows,replays=replays,
            physical_unique_cause_proven=False,policy_changed=False,new_random_windows=0,
            scope='post-observation diagnostic registered in ZERO_INTENT_DIAGNOSTIC_CONTRACT; original strict criteria and complete192x3 denominators unchanged; no heldout-tuning claim'),f,indent=2,allow_nan=False)
    print('ZERO_INTENT_DIAGNOSIS',summary,flush=True)


if __name__=='__main__':main()
