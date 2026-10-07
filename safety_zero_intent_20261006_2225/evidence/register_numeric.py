"""Freeze all source, input and exact command bindings before policy outcomes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
HERE=Path(__file__).resolve().parent
RAW=Path('/mnt/nas/data/lyf/double_hand')/HERE.name
OLD=HERE.parent/'safety_tracking_reserve_20261006_1535'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')

def main():
    d=json.loads((HERE/'RANDOM_EXPERIMENT_DESIGN.json').read_text())
    prior=json.loads((OLD/'plans/holdout_0_plan.json').read_text())
    assert json.loads((HERE/'BANK_EXECUTION.json').read_text())['status']=='PASS_COMPLETE_POLICY_BLIND_QUALIFICATION'
    assert json.loads((HERE/'bank_audit.json').read_text())['status']=='PASS_ENUMERATED_BANK_FRESHNESS'
    assert (HERE/'zero_intent_cpu_tests_v2.log').read_text().rstrip().endswith('OK')
    astra=json.loads((HERE/'ASTRA_ZERO_SOURCE_REVIEW.json').read_text())
    assert astra['status'].startswith('PASS'),astra
    for source in ('zero_intent.py','reference_envelope.py','guard_runner.py'):
        assert sha(HERE/source) in str(astra), 'independent source binding missing: '+source
    assert all(sha(Path(prior['cwd'])/p)==v for p,v in prior['source_sha256'].items())
    assert sha(prior['checkpoint_path'])==prior['checkpoint_sha256']
    helpers=dict(prior['research_source_sha256'])
    names=['RANDOM_EXPERIMENT_DESIGN.json','guard_runner.py','zero_intent.py','tracking_reserve.py',
           'target_forecast.py','reference_envelope.py','projection_diagnostics.py','test_zero_intent.py',
           'dense_audit.py','analyze.py','register_numeric.py','execute_numeric.py','execute_analysis.py',
           'ASTRA_ZERO_SOURCE_REVIEW.json','execute_all.py','SCHEDULING_REGISTRATION.json','LP_BACKEND_REGISTRATION.json','audit_fixed_feasibility.py','FIXED_LP_REGISTRATION.json','bank_audit.json','failure_audit.py','audit_commands.py','audit_zero_intent.py','ZERO_INTENT_AUDIT_REGISTRATION.json']
    helpers[str(HERE.parent/'safety_random_space_20261004/coverage_metrics.py')]=sha(HERE.parent/'safety_random_space_20261004/coverage_metrics.py')
    for name in names:helpers[str(HERE/name)]=sha(HERE/name)
    banks=[]
    for r in d['rows']:
        bank=RAW/'banks'/str(r['initial_seed'])/'bank.npz'
        meta=json.loads(bank.with_name('metadata.json').read_text())
        assert meta['status']=='complete' and meta['selected_count']==64 and not meta['policy_outcomes_used']
        for p in (bank,bank.with_name('metadata.json')):helpers[str(p)]=sha(p)
        banks.append(bank)
    dest=HERE/'plans';dest.mkdir(exist_ok=True);assert not list(dest.glob('holdout_*_plan.json'))
    plans=[]
    for block,(r,bank) in enumerate(zip(d['rows'],banks)):
        device='cuda:0' if block==0 else 'cuda:1'
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
            registered_utc=datetime.now(timezone.utc).isoformat(),scope='fresh randomized paired zero-inclusive target reference',
            production_promoted=False,sources_frozen=True)
        p=dest/f'holdout_{block}_plan.json';write(p,plan);plans.append(p)
    write(HERE/'NUMERIC_REGISTRATION.json',dict(status='FROZEN_BEFORE_ALL_NEW_POLICY_OUTCOMES',
          plans={str(p):sha(p) for p in plans},method_windows=576,paired_cases=192,primary=d['primary_comparison']))
    print('FROZEN 9 conditions / 576 windows',flush=True)
if __name__=='__main__':main()
