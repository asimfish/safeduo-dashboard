"""Registered geometry/zero-input qualification; complete source checks before/after."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, subprocess
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent.parent
RAW=Path('/mnt/nas/data/lyf/double_hand')/HERE.name
OLD=HERE.parent/'safety_feasible_guard_20261005_2100'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,d):
    with (HERE/n).open('x') as f:f.write(json.dumps(d,indent=2)+'\n')
def main():
    prior=json.loads((OLD/'holdout_v2/holdout_0_plan.json').read_text())
    argv=[prior['jobs'][0]['argv'][0],str(HERE/'fresh_bank.py'),'--out',str(RAW/'banks'),'--ckpt',prior['checkpoint_path'],'--device','cuda:1','--headless']
    sources={str(PROJECT/n):v for n,v in prior['source_sha256'].items()}
    sources.update({str(p):sha(p) for p in [HERE/'fresh_bank.py',HERE/'execute_bank.py',HERE/'RANDOM_EXPERIMENT_DESIGN.json',HERE.parent/'safety_risk_strata_20261004/risk_bank.py',HERE.parent/'safety_random_space_20261004/wide_random.py']})
    sources[prior['checkpoint_path']]=prior['checkpoint_sha256']
    assert all(sha(p)==v for p,v in sources.items())
    plan=dict(status='REGISTERED_BEFORE_QUALIFICATION',argv=argv,cwd=str(PROJECT),env=prior['env'],source_sha256=sources,utc=datetime.now(timezone.utc).isoformat(),policy_outcomes_used=False,hardware_approved=False)
    write('BANK_LAUNCH_REGISTRATION.json',plan)
    env={**os.environ,**prior['env'],'PYTHONDONTWRITEBYTECODE':'1'}
    with (HERE/'bank.log').open('w') as f:
        result=subprocess.run(argv,cwd=PROJECT,env=env,stdout=f,stderr=subprocess.STDOUT)
    source_after={p:sha(p) for p in sources};assert source_after==sources
    design=json.loads((HERE/'RANDOM_EXPERIMENT_DESIGN.json').read_text())
    metadata=[json.loads((RAW/'banks'/str(r['initial_seed'])/'metadata.json').read_text()) for r in design['rows']] if result.returncode==0 else []
    complete=result.returncode==0 and len(metadata)==3 and all(r['status']=='complete' and r['selected_count']==64 and not r['policy_outcomes_used'] for r in metadata)
    write('BANK_EXECUTION.json',dict(status='PASS_COMPLETE_POLICY_BLIND_QUALIFICATION' if complete else 'FAIL',argv=argv,returncode=result.returncode,source_after_sha256=source_after,banks=metadata,utc=datetime.now(timezone.utc).isoformat(),new_random_policy_windows=0))
    assert complete
    print('BANK_COMPLETE',len(metadata),sum(m['selected_count'] for m in metadata),flush=True)
if __name__=='__main__':main()
