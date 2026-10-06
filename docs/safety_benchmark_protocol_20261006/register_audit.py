"""Fix retrospective metric and file identities before scoring; no new physics."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'safety_feasible_guard_20261005_2100'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    expected=json.loads((OLD/'holdout_results.json').read_text());rows=[]
    for file in sorted((OLD/'holdout_v2').glob('holdout_*_plan.json')):
        plan=json.loads(file.read_text())
        for job in plan['jobs']:
            root=Path(plan['output_root'])/job['id'];r=next(r for r in expected['rows'] if r['id']==job['id'])
            rows.append(dict(id=job['id'],root=str(root),mode=r['mode'],seed=r['seed'],dense_sha256=sha(root/'cell_001.npz'),input_sha256=sha(root/'input_recipe.npz'),expected_violations=r['violations'],expected_deep=r['deep']))
    names=['PROTOCOL.md','coverage.py','test_coverage.py','audit.py','register_audit.py','software_tests_green.log']
    out=dict(status='retrospective_audit_registered',created_utc=datetime.now(timezone.utc).isoformat(),rows=rows,
        source_sha256={str(HERE/n):sha(HERE/n) for n in names},closed_results_sha256=sha(OLD/'holdout_results.json'),
        bins=10,ee_voxel_m=.1,active_arm_pre_qd_rms_rad_s=.01,
        new_physical_windows=0,outcomes_previously_known=True,no_new_conservative_claim=True)
    with (HERE/'AUDIT_REGISTRATION.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print('FROZEN OLD768 COVERAGE AUDIT',flush=True)
if __name__=='__main__':main()
