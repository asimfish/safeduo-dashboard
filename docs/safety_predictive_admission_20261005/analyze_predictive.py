"""Full-velocity admission factorial audit, all registered new initial/command pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'safety_risk_strata_20261004'))
from analyze_risk import inspect, load, measured_coverage, pair_exposure, PAIR_NAMES
from risk_recipe import validate_bank
from operation_metrics import safe_operation,pairwise_joint_visits
sys.path.insert(0,str(HERE.parent / 'safety_mechanism_20261005'))
from analyze_mechanism import inspect_mechanism as inspect_original_envelope

SEEDS = (331047829, 389116237, 451902773)
MODES = ('raw','baseline','envelope_050','predictive','predictive_envelope')


def audit_bank():
    root = Path('/mnt/nas/data/lyf/double_hand/safety_predictive_admission_20261005/holdout_banks')
    geo, settle = load(root / 'all_geometry.npz'), load(root / 'all_settling.npz')
    assert len(geo['q']) == len(geo['batch_identity'])*64
    indices = {tuple(x):i for i,x in enumerate(settle['identity'])}
    assert len(indices) == len(settle['identity'])
    rows = []
    for seed in SEEDS:
        bank = root / str(seed)
        meta = json.loads((bank / 'metadata.json').read_text())
        b = load(bank / 'bank.npz')
        validate_bank(b['accepted_q'], b['risk_pair_index'], meta)
        restart = {i:set() for i in range(6)}
        for q,label,(ref,e) in zip(b['accepted_q'], b['risk_pair_index'], b['selected_refs']):
            idx = int(ref)*64+int(e)
            ident = tuple(geo['batch_identity'][ref]); s = indices[ident]
            assert ident[0] == seed and ident[1] == label
            assert np.array_equal(q, geo['q'][idx])
            assert geo['eligible'][idx] and geo['margins'][idx].min() >= .001-1e-7
            assert geo['table'][idx] >= .001-1e-7
            assert settle['eligible'][s,e] and settle['valid'][s,e] and not settle['bad'][s,e]
            assert settle['drift'][s,e] <= .05 and settle['velocity'][s,e] <= .1
            assert settle['min_table'][s,e] >= 0
            if label >= 0:
                assert .020-1e-7 <= geo['pairs'][idx,label] <= .060+1e-7
                assert int(e) == int(np.flatnonzero(settle['valid'][s])[0])
                assert ident[2] not in restart[int(label)]
                restart[int(label)].add(ident[2])
        rows.append({**meta, 'selection_references_checked':True,
                     'risk_restarts_per_pair':[len(restart[i]) for i in range(6)]})
    return dict(rows=rows, sampling=json.loads((root / 'sampling_summary.json').read_text()),
                geometry_pass_candidates=int(geo['eligible'].sum()),
                settling_pass_candidates=int(settle['valid'].sum()),
                selected_before_policy=True,
                geometry_sha256=hashlib.sha256((root / 'all_geometry.npz').read_bytes()).hexdigest(),
                settling_sha256=hashlib.sha256((root / 'all_settling.npz').read_bytes()).hexdigest())


def inspect_mechanism(root,job):
    factor=job['mode']
    inherited='envelope_050' if factor in ('envelope_050','predictive_envelope') else 'raw' if factor=='raw' else 'baseline'
    r,b,d,p,a=inspect_original_envelope(root,{**job,'mode':inherited})
    m=json.loads((Path(r['path'])/'admission_manifest.json').read_text())
    assert m['mode']==factor and m['predictive']==(factor in ('predictive','predictive_envelope'))
    assert m['horizons_s']==[.16,.4,.3] and m['band_m']==.010 and m['capacity']==2048
    assert m['strict_fifo_steps']==6 and not m['queue_preemption'] and not m['production_promoted']
    assert m['original_actor_observation_rows']==32 and m['original_projection_and_exemptions']
    assert m['source_sha256']=={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('predictive_runner.py','predictive_rows.py')}
    r.update(mode=factor,admission=m)
    if d is not None:
        for k in ('required','additional','predicted_missing','legacy_missing'):
            assert d['admission_'+k].shape==(960,64)
        assert not d['admission_predicted_missing'].any() and not d['admission_legacy_missing'].any()
        if m['predictive']:
            assert np.array_equal(d['admission_required'],d['critical_selected_count'])
            assert d['admission_required'].max()<=2048
        r.update(predicted_required_missing=0,legacy_rows_missing=0,
                 mean_additional_rows=float(d['admission_additional'].mean()),
                 additional_rows_env_steps=int((d['admission_additional']>0).sum()),
                 maximum_additional_rows=int(d['admission_additional'].max()),
                 finite_full_geometry_checked=m['predictive'])
        r.update(safe_operation(b['q_initial'],d['q'],d['official_margins'],d['joint_soft_limits'],p['dt']))
    return r,b,d,p,a

def paired(left, right, banks, data):
    a,b = banks[left['id']], banks[right['id']]
    assert np.array_equal(a['q_initial'], b['q_initial'])
    assert np.array_equal(a['tape'], b['tape'])
    assert np.array_equal(a['initial_violation'], b['initial_violation'])
    result = dict(seed=left['seed'], left=left['id'], right=right['id'],
                  left_mode=left['mode'], right_mode=right['mode'],
                  initial_exact=True, command_exact=True, initial_flags_exact=True)
    if left['status'] != 'complete' or right['status'] != 'complete':
        return result
    da,db = data[left['id']], data[right['id']]
    fa,fb = (da['official_margins']<0).any((0,2)), (db['official_margins']<0).any((0,2))
    result.update(left_only_failure=int((fa&~fb).sum()), right_only_failure=int((fb&~fa).sum()),
                  both_failure=int((fa&fb).sum()), neither_failure=int((~fa&~fb).sum()),
                  actual_q_exact=bool(np.array_equal(da['q'], db['q'])),
                  actual_q_prefix66_exact=bool(np.array_equal(da['q'][:66],db['q'][:66])),
                  actual_q_prefix66_max_diff_rad=float(np.abs(da['q'][:66]-db['q'][:66]).max()),
                  exec_exact=bool(np.array_equal(da['exec'],db['exec'])))
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--partial', action='store_true'); args=parser.parse_args()
    rows,banks,data,assignments,protocols = [],{},{},{},[]
    statuses = {}
    for phase in ('development','holdout'):
        plan = json.loads((HERE / (phase+'_plan.json')).read_text())
        root = Path(plan['output_root']); campaign=json.loads((root / 'campaign.json').read_text())
        statuses[phase] = campaign['status']
        if not args.partial:
            assert campaign['status'] in ('complete','complete_with_failures')
            assert len(campaign['jobs']) == plan['planned_cells']
        for name,sha in plan['source_sha256'].items():
            assert hashlib.sha256((Path(plan['cwd'])/name).read_bytes()).hexdigest()==sha,name
        for name,sha in plan['research_source_sha256'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
        assert hashlib.sha256(Path(plan['checkpoint_path']).read_bytes()).hexdigest()==plan['checkpoint_sha256']
        assert len(set(j['pid'] for j in campaign['jobs'])) == len(campaign['jobs'])
        specs = {j['id']:j for j in plan['jobs']}
        for j in campaign['jobs']:
            if j['status']=='running':
                continue
            r,b,d,p,a = inspect_mechanism(root,{**j,'mode':specs[j['id']]['mode']})
            r['phase']=phase;rows.append(r);banks[r['id']]=b;data[r['id']]=d;assignments[r['id']]=a;protocols.append(p)
    first = protocols[0]
    for p in protocols:
        for key in ('source_sha256','checkpoint_sha256','resolved_config','effective_backstop','effective_coordinator'):
            assert p[key]==first[key],key
    strata,groups,pairs,aggregate,operation_joint = [],[],[],[],[]
    for mode in MODES:
        selected = [r for r in rows if r['phase']=='holdout' and r['mode']==mode and r['status']=='complete']
        if not selected:
            continue
        d = {k:np.concatenate([data[r['id']][k] for r in selected],1)
             for k in ('q','ee','official_margins','pair_margin','table_raw_min')}
        q0 = np.concatenate([banks[r['id']]['q_initial'] for r in selected])
        ee0 = np.concatenate([banks[r['id']]['ee_initial'] for r in selected])
        limits = np.concatenate([banks[r['id']]['joint_soft_limits'] for r in selected])
        labels = np.concatenate([assignments[r['id']]['risk_pair_index'] for r in selected])
        margin = d['official_margins']
        aggregate.append(dict(mode=mode,windows=len(q0),planned_windows=192,
                              violations=int((margin<0).any((0,2)).sum()),
                              deep=int((margin<-.005).any((0,2)).sum()),
                              class_violations=(margin<0).any(0).sum(0).tolist(),
                              zero_prefix_violations=sum(r['zero_prefix_violations'] for r in selected),
                              before_first_random_delivery=sum(r['before_first_random_delivery'] for r in selected),
                              four_arms_moving_fraction=float(np.mean([r['four_arms_moving_fraction'] for r in selected])),
                              exec_command_l2_ratio=float(np.mean([r['exec_command_l2_ratio'] for r in selected])),
                              pre_target_debt_abs_p95_rad=float(np.mean([r['pre_target_debt_abs_p95_rad'] for r in selected])),
                              over_nominal_box_env_steps=sum(r['over_nominal_box_env_steps'] for r in selected),
                              fifo_exact=True,soft_target_limits_pass=True))
        safe=safe_operation(q0,d['q'],margin,limits,first['dt'])
        aggregate[-1].update(safe)
        operation_joint.append(dict(mode=mode,**pairwise_joint_visits(q0,d['q'],limits)))
        for label,name in [(-2,'全部风险分层'),(-1,'广域稳定初态'),*enumerate(PAIR_NAMES)]:
            mask = labels>=0 if label==-2 else labels==label
            margins = margin[:,mask]
            pairs_all = pair_exposure(d['pair_margin'],d['ee'],mask)
            pressure = pair_exposure(d['pair_margin'][66:],d['ee'][66:],mask)
            c = measured_coverage(q0,d['q'],ee0,d['ee'],limits,mask)
            strata.append(dict(mode=mode,stratum=label,label=name,windows=int(mask.sum()),
                               planned_windows=144 if label==-2 else 48 if label==-1 else 24,
                               violations=int((margins<0).any((0,2)).sum()),
                               deep=int((margins<-.005).any((0,2)).sum()),
                               class_violations=(margins<0).any(0).sum(0).tolist(),
                               target_pair_exposure=None if label<0 else pairs_all[label],
                               target_pair_pressure_exposure=None if label<0 else pressure[label],
                               mean_within_window_joint_range=c['mean_within_window_joint_range'],
                               mean_joint_path_rad=c['mean_joint_path_rad']))
            if label<0:
                groups.append(dict(mode=mode,stratum=label,label=name,coverage=c,
                                   pairs=pairs_all,pressure_pairs=pressure))
    hold = [r for r in rows if r['phase']=='holdout']
    for seed in SEEDS:
        available = {r['mode']:r for r in hold if r['seed']==seed}
        for left,right in [('raw','baseline'),('baseline','envelope_050'),('baseline','predictive'),('envelope_050','predictive_envelope'),('predictive','predictive_envelope')]:
            if left in available and right in available:
                pairs.append(paired(available[left],available[right],banks,data))
    for a in aggregate:
        same=[g['coverage'] for g in groups if g['mode']==a['mode']]
        n=sum(c['episodes'] for c in same)
        a['mean_within_window_joint_range']=sum(c['mean_within_window_joint_range']*c['episodes'] for c in same)/n
        a['mean_joint_path_rad']=sum(c['mean_joint_path_rad']*c['episodes'] for c in same)/n
        a['visited_joint_bins']=int((sum(np.array(c['visited_joint']['counts']) for c in same)>0).sum())
    development = [r for r in rows if r['phase']=='development']
    results = dict(schema='safeduo.predictive_admission_evidence.v1',campaign_status=statuses,
                   completed_windows=sum(r['completed_windows'] for r in hold),
                   invalid_windows=sum(r['invalid_windows'] for r in hold),
                   pending_windows=960-len(hold)*64,registered_windows=960,
                   development_completed_windows=sum(r['completed_windows'] for r in development),
                   development_registered_windows=128,development_new_command_windows=0,
                   new_unique_command_windows=192,rows=rows,aggregate_rows=aggregate,strata=strata,
                   group_coverage=groups,paired_inputs=pairs,bank_audit=audit_bank(),operation_pairwise=operation_joint,
                   diagnostic=json.loads((HERE/'admission_audit.json').read_text()),
                   actor_sha256=plan['checkpoint_sha256'],source_sha256=plan['source_sha256'],dt=first['dt'],
                   candidate_status='experimental_not_promoted',production_source_unchanged=True,
                   no_holdout_tuning=True,cpu_contract_tests=6,
                   statistical_unit='conditioned matched windows with shared physics; no IID interval',
                   analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    results['prior_queue_evidence']=dict(path='/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005_causal_obs/final_evidence',baseline_failures=120,candidate_failures=40,windows_per_method=192,new_commands=192,new_initial_banks=False,scope='different candidate/command seeds/initial banks; historical evidence only, no pooled denominator')
    (HERE/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:results[k] for k in ('campaign_status','completed_windows','invalid_windows','pending_windows')}))
    print(json.dumps([(r['mode'],r['violations'],r['windows']) for r in aggregate]))


if __name__=='__main__':
    main()
