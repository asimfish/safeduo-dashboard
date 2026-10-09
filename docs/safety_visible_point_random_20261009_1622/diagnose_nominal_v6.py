"""Descriptive reasons for model rejection, from closed native issued-command records."""
from pathlib import Path
import hashlib,json,datetime,numpy as np
H=Path(__file__).resolve().parent
def main():
    active=json.loads((H/'ACTIVE_ATTEMPT_V6.json').read_text());root=Path(active['primary_roots']['multirow'])
    assert json.loads((root/'response_protocol.json').read_text())['status']=='complete'
    result=json.loads((H/'FRESH_RANDOM_RESULT.json').read_text());eligible=np.asarray(result['states']['raw']['prefix_eligible'],dtype=bool)
    with np.load(root/'response_stream.npz') as z:
        reasons={};values={}
        for key,tol in [('geometry_velocity_residual',1e-4),('controlled_velocity_limit_residual',1e-3),('full_drive_effort_residual',1e-3)]:
            a=z['multi_'+key];assert a.shape==(480,64) and np.isfinite(a).all();reasons[key+'_exceeds_registered_tolerance']=a>tol
            values[key]=dict(tolerance=tol,minimum=float(a.min()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),maximum=float(a.max()))
        for key in ['missing_direction','invalid_box','uncontrolled_nearby_raw_negative']:
            a=z['multi_'+key];assert a.shape==(480,64) and a.dtype==np.bool_;reasons[key]=a
        satisfied=z['multi_model_constraints_satisfied'];assert satisfied.dtype==np.bool_ and satisfied.shape==(480,64)
        reconstructed=~np.logical_or.reduce(list(reasons.values()));assert np.array_equal(reconstructed,satisfied)
        per_state={k:a.sum(0).tolist() for k,a in reasons.items()}
    counts={k:dict(all_issued=int(a.sum()),qualified_issued=int(a[:,eligible].sum()),actually_actuated_new_issues=int(a[:-6].sum())) for k,a in reasons.items()}
    evidence=dict(status='PASS_CLOSED_NOMINAL_REJECTION_FLAGS_RECONSTRUCTED_EXACT',source_stream_sha256=result['states']['multirow']['stream_sha256'],source_code_sha256=hashlib.sha256((H/'multirow_response_v3.py').read_bytes()).hexdigest(),diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),all_issued_denominator=30720,qualified_issued_denominator=480*int(eligible.sum()),actuated_new_issue_denominator=30336,model_satisfied_issued=int(satisfied.sum()),failure_reason_counts=counts,residual_quantiles=values,per_state_failure_counts=per_state,counts_overlap=True,reasons_not_causal_explanations=True,no_strategy_change=True,raw_reference_resource_verdict='FAILED_UNCHANGED',physical_safety_certified=False,next_mechanism_frontier='Do not treat an infeasible nominal output as a certified action. A future runtime fallback must preserve FIFO6 and native full-arm dynamics, and be independently paired on new frozen inputs; no adoption claim from this diagnostic.',utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (H/'NOMINAL_REJECTION_DIAGNOSTIC_V6.json').open('x') as f:json.dump(evidence,f,indent=2);f.write('\n')
    print(evidence['status'],json.dumps(counts),flush=True)
if __name__=='__main__':main()
