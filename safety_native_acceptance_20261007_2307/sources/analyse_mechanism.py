"""Read-only attribution of the frozen candidate's actual widening and residuals."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    reg=json.loads((H/'ACCEPTANCE_REGISTRATION.json').read_text());rows=[]
    assert json.loads((H/(reg['tag']+'_execution.json')).read_text())['status']=='complete'
    for job in reg['jobs']:
        p=Path(job['out'])/'project_diagnostics.npz'
        with np.load(p,allow_pickle=False) as src:z={k:src[k].copy() for k in src.files}
        assert z['returned_cmd'].shape==(960,64,26)
        row=dict(job_id=job['id'],mode=job['env']['SAFEDUO_JOINT_MODE'],source_sha256=sha(p),controlled_frames=960*64,final_project_returned_safety_residual_over_1e6_frames=int(((z['returned_safety_residual_F']>1e-6)|(z['returned_safety_residual_U']>1e-6)).sum()),target_increment_safety_residual_over_1e6_frames=int(((z['target_safety_residual_F']>1e-6)|(z['target_safety_residual_U']>1e-6)).sum()),final_project_bound_residual_peak=float(max(z['returned_bound_residual_F'].max(),z['returned_bound_residual_U'].max())))
        if row['mode']=='adaptive_joint':
            checked=z['repair_lp_checked'];status=z['repair_lp_status'];classification=z['repair_lp_classification'];widened=z['repair_widened_env'];first=z['repair_first_current_residual'];final=z['repair_final_current_residual']
            assert checked.shape==status.shape==classification.shape==first.shape==final.shape==(960,64,2)
            assert np.array_equal(checked,first>1e-6)
            assert np.array_equal(widened,(classification==2).any(-1))
            assert (status[~checked]==-2).all() and (classification[~checked]==-2).all()
            assert (status[classification==2]==2).all() and (status[classification==0]==0).all()
            assert np.isin(classification,[-2,-1,0,2]).all()
            assert not (z['repair_bounds_changed_env']&~widened).any()
            assert np.array_equal(z['repair_final_residual_env'],(final>1e-6).any(-1))
            assert (z['repair_queued_future_status']==-1).all() and not z['repair_physical_safety_certified_env'].any()
            row.update(current_LP_checks=int(checked.sum()),current_LP_infeasible_checks=int((classification==2).sum()),current_LP_feasible_checks=int((classification==0).sum()),current_LP_unknown_checks=int((classification==-1).sum()),widened_env_frames=int(widened.sum()),widened_cases=int(widened.any(0).sum()),actual_bounds_changed_env_frames=int(z['repair_bounds_changed_env'].sum()),remaining_final_current_residual_frames=int(z['repair_final_residual_env'].sum()),first_current_residual_peak=float(first.max()),final_current_residual_peak=float(final.max()),queued_future_status='UNKNOWN',physical_safety_certified=False)
        rows.append(row)
    out=dict(status='COMPLETE_FROZEN_MECHANISM_ATTRIBUTION_ONLY',rows=rows,source_sha256=sha(__file__),scope='Observed original current linear solver statuses and actual projection residuals; separate from original physical geometry and native contact scoring. No future nonlinear/dynamic/queue/hardware safety certificate.',utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (H/'MECHANISM_RESULT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(out['status'],flush=True)
if __name__=='__main__':main()
