"""Freeze all source, input and exact command bindings before policy outcomes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
HERE=Path(__file__).resolve().parent
RAW=Path('/mnt/nas/data/lyf/double_hand')/HERE.name
OLD=HERE.parent/'safety_feasible_guard_20261005_2100'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')

def main():
    d=json.loads((HERE/'RANDOM_EXPERIMENT_DESIGN.json').read_text())
    prior=json.loads((OLD/'holdout_v2/holdout_0_plan.json').read_text())
    assert json.loads((HERE/'BANK_EXECUTION.json').read_text())['status']=='PASS_COMPLETE_POLICY_BLIND_QUALIFICATION'
    assert json.loads((HERE/'bank_audit.json').read_text())['status']=='PASS_ENUMERATED_BANK_FRESHNESS'
    assert (HERE/'motion_tests.log').read_text().rstrip().endswith('OK')
    astra=json.loads((HERE/'ASTRA_MOTION_RUNNER_REVIEW.json').read_text())
    assert astra['status'].startswith('PASS'),astra
    assert all(sha(Path(prior['cwd'])/p)==v for p,v in prior['source_sha256'].items())
    assert sha(prior['checkpoint_path'])==prior['checkpoint_sha256']
    assert sha(HERE/'MODEL_FIT.json')==d['prior_model_fit_sha256']
    helpers=dict(prior['research_source_sha256'])
    names=['RANDOM_EXPERIMENT_DESIGN.json','MODEL_FIT.json','guard_runner.py','motion_forecast.py',
           'target_forecast.py','reference_envelope.py','projection_diagnostics.py','test_motion_forecast.py',
           'dense_audit.py','analyze.py','register_numeric.py','execute_numeric.py','audit_prediction_h6.py',
           'ASTRA_MOTION_RUNNER_REVIEW.json','astra_fifo_oracle.py','bank_audit.json']
    for name in names:helpers[str(HERE/name)]=sha(HERE/name)
    banks=[]
    for r in d['rows']:
        bank=RAW/'banks'/str(r['initial_seed'])/'bank.npz'
        meta=json.loads(bank.with_name('metadata.json').read_text())
        assert meta['status']=='complete' and meta['selected_count']==64 and not meta['policy_outcomes_used']
        for p in (bank,bank.with_name('metadata.json')):helpers[str(p)]=sha(p)
        banks.append(bank)
    dest=HERE/'plans';dest.mkdir(exist_ok=False)
    plans=[]
    for block,(r,bank) in enumerate(zip(d['rows'],banks)):
        device='cuda:0' if block in (0,2) else 'cuda:1'
        jobs=[]
        for mode in d['modes']:
            old=prior['jobs'][0];argv=list(old['argv']);argv[1]=str(HERE/'guard_runner.py')
            argv[argv.index('--seeds')+1]=str(r['command_seed'])
            argv[argv.index('--device')+1]=device
            jobs.append(dict(id=f'{mode}_{r["command_seed"]}',argv=argv,
                expected={**old['expected'],'seed':r['command_seed']},
                expected_args={**old['expected_args'],'device':device},
                env=dict(SAFEDUO_INITIAL_BANK_NPZ=str(bank),SAFEDUO_JOINT_MODE=mode)))
        plan={k:prior[k] for k in ('cwd','source_sha256','checkpoint_path','checkpoint_sha256')}
        plan.update(research_source_sha256=helpers,env={**prior['env'],'PYTHONDONTWRITEBYTECODE':'1'},
            output_root=str(RAW/f'holdout_{block}'),continue_after_child_failure=True,jobs=jobs,
            registered_utc=datetime.now(timezone.utc).isoformat(),scope='fresh randomized paired nominal motion admission',
            production_promoted=False,sources_frozen=True)
        p=dest/f'holdout_{block}_plan.json';write(p,plan);plans.append(p)
    write(HERE/'NUMERIC_REGISTRATION.json',dict(status='FROZEN_BEFORE_ALL_NEW_POLICY_OUTCOMES',
          plans={str(p):sha(p) for p in plans},method_windows=768,paired_cases=192,primary=d['primary_comparison']))
    print('FROZEN 12 conditions / 768 windows',flush=True)
if __name__=='__main__':main()
