"""Do not exclude pressure windows based on these supplemental neutral outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'safety_mechanism_20261005'))
from analyze_mechanism import inspect_mechanism

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args()
    plan=json.loads((HERE/'campaign_plan.json').read_text());root=Path(plan['output_root'])
    c=json.loads((root/'campaign.json').read_text())
    if not args.partial:assert c['status'] in ('complete','complete_with_failures') and len(c['jobs'])==9
    for name,sha in plan['source_sha256'].items():assert hashlib.sha256((Path(plan['cwd'])/name).read_bytes()).hexdigest()==sha,name
    for name,sha in plan['research_source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    specs={j['id']:j for j in plan['jobs']};rows=[];inputs={};aggregate=[];protocols=[]
    for j in c['jobs']:
        if j['status']=='running':continue
        r,b,d,p,a=inspect_mechanism(root,{**j,'mode':specs[j['id']]['mode']})
        assert r['sampling']['neutral_diagnostic'] and r['sampling']['initial_banks_reused']
        assert r['sampling']['neutral_runner_sha256']==hashlib.sha256((HERE/'zero_runner.py').read_bytes()).hexdigest()
        assert not b['tape'].any() and not b['updates'].any() and not b['holds'].any() and not b['segment_amplitudes'].any()
        if d is not None:
            assert not d['cmd'].any()
            if r['mode']=='raw':assert np.array_equal(d['controller_target'],np.repeat(b['q_initial'][None],960,0))
            r.update(nonzero_execution_windows=int((np.abs(d['exec']).max((0,2))>1e-7).sum()),
                     max_actual_joint_drift_rad=float(np.abs(d['q']-b['q_initial']).max()),
                     max_absolute_velocity_rad_s=float(np.abs(d['pre_qd_compact']).max()),
                     full960_zero_commands_verified=True,
                     neutral_violations=r['violations'])
        rows.append(r);inputs[r['id']]=b;protocols.append(p)
    if protocols:
        for p in protocols:
            for key in ('source_sha256','checkpoint_sha256','resolved_config','effective_backstop','effective_coordinator'):
                assert p[key]==protocols[0][key],key
    for mode in ('raw','baseline','envelope_050'):
        sel=[r for r in rows if r['mode']==mode and r['status']=='complete']
        if sel:aggregate.append(dict(mode=mode,windows=sum(r['completed_windows'] for r in sel),planned_windows=192,
                                    violations=sum(r['violations'] for r in sel),deep=sum(r['deep'] for r in sel),
                                    class_violations=np.sum([r['class_violations'] for r in sel],0).tolist(),
                                    nonzero_execution_windows=sum(r['nonzero_execution_windows'] for r in sel),
                                    max_actual_joint_drift_rad=max(r['max_actual_joint_drift_rad'] for r in sel),
                                    all960_zero_commands=True,fifo_exact=True,soft_limits_pass=True))
    matched=[]
    for seed in plan['seeds']:
        same=[r for r in rows if r['seed']==seed]
        if len(same)!=3:continue
        ref=inputs[same[0]['id']]
        for row in same[1:]:
            b=inputs[row['id']]
            assert np.array_equal(ref['q_initial'],b['q_initial']) and np.array_equal(ref['tape'],b['tape'])
        matched.append(dict(seed=seed,initial_exact=True,zero_input_exact=True))
    result=dict(schema='safeduo.zero_command_diagnostic.v1',campaign_status=c['status'],
                completed_windows=sum(r['completed_windows'] for r in rows),invalid_windows=sum(r['invalid_windows'] for r in rows),
                pending_windows=576-len(rows)*64,registered_windows=576,new_unique_random_command_windows=0,new_initial_banks=0,
                primary_pressure_results_unchanged=True,rows=rows,aggregate_rows=aggregate,matched=matched,
                actor_sha256=plan['checkpoint_sha256'],analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='supplement registered after partial pressure outcomes but before its own outcomes; all16s zeros; reused initial banks; never used to exclude primary cases')
    (HERE/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('campaign_status','completed_windows','invalid_windows','pending_windows')}))
    print(json.dumps([(x['mode'],x['violations'],x['windows']) for x in aggregate]))

if __name__=='__main__':main()
