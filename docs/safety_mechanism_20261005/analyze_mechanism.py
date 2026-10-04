"""Audited full-window, fixed-factor development and independent holdout evidence."""
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

SEEDS = (152684921, 198470327, 237901613)
MODES = ('raw', 'baseline', 'box_only', 'envelope_050')


def audit_bank():
    root = Path('/mnt/nas/data/lyf/double_hand/safety_mechanism_20261005/holdout_banks')
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


def inspect_mechanism(root, job):
    r,b,d,p,a = inspect(root, job)
    manifest = json.loads((Path(r['path']) / 'mechanism_manifest.json').read_text())
    expected = 'baseline' if job['mode'] == 'raw' else job['mode']
    assert manifest['mode'] == expected and manifest['actuator_delay_steps'] == 6
    assert not manifest['queue_preemption'] and not manifest['production_promoted']
    assert manifest['original_actor_observation_rows']==32 and manifest['original_damper_unchanged']
    assert manifest['gap_rad']==(.050 if expected=='envelope_050' else None)
    assert manifest['source_sha256']=={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                       for name in ('mechanism_runner.py','reference_envelope.py')}
    r.update(mode=job['mode'], mechanism=manifest)
    if d is None:
        return r,b,d,p,a
    pre_q = np.concatenate([b['q_initial'][None], d['q'][:-1]])
    pre_target = np.concatenate([b['q_initial'][None], d['controller_target'][:-1]])
    assert np.max(np.abs(d['pre_target_debt'] - (pre_target-pre_q))) < 5e-7
    box = p['effective_backstop']['vmax']*p['dt']
    lower = np.maximum(d['joint_soft_limits'][...,0]-pre_target, -box)
    upper = np.minimum(d['joint_soft_limits'][...,1]-pre_target, box)
    assert (lower <= upper+1e-6).all()
    issue = d['controller_target']-pre_target
    lo = np.minimum(np.maximum(pre_q-.050-pre_target, lower), upper)
    hi = np.minimum(np.maximum(pre_q+.050-pre_target, lower), upper)
    unreachable = (pre_q+.050 < pre_target+lower) | (pre_q-.050 > pre_target+upper)
    if r['mode'] == 'envelope_050':
        assert (issue >= lo-1e-6).all() and (issue <= hi+1e-6).all(), 'envelope oracle failed'
    if r['mode'] in ('box_only','envelope_050'):
        assert (np.abs(issue) <= box+5e-7).all(), 'governor actual target violates speed box'
    r.update(reference_bounds_pass=True if r['mode'] in ('box_only','envelope_050') else None,
             envelope_unreachable_joint_env_step_fraction=float(unreachable.mean()),
             envelope_unreachable_arm_env_step_fraction=float(d['reference_envelope_unreachable'].mean()),
             pre_target_debt_abs_max_rad=float(np.abs(d['pre_target_debt']).max()),
             pre_target_debt_abs_p95_rad=float(np.percentile(np.abs(d['pre_target_debt'][60:]),95)),
             issued_reference_gap_abs_max_rad=float(np.abs(d['controller_target']-pre_q).max()),
             issued_reference_gap_abs_p95_rad=float(np.percentile(np.abs(d['controller_target'][60:]-pre_q[60:]),95)),
             outside_050_issued_reference_joint_fraction=float((np.abs(d['controller_target'][60:]-pre_q[60:])>.050001).mean()))
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
    strata,groups,pairs,aggregate = [],[],[],[]
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
        for left,right in [('raw','baseline'),('baseline','box_only'),('baseline','envelope_050'),('box_only','envelope_050')]:
            if left in available and right in available:
                pairs.append(paired(available[left],available[right],banks,data))
    for a in aggregate:
        same=[g['coverage'] for g in groups if g['mode']==a['mode']]
        n=sum(c['episodes'] for c in same)
        a['mean_within_window_joint_range']=sum(c['mean_within_window_joint_range']*c['episodes'] for c in same)/n
        a['mean_joint_path_rad']=sum(c['mean_joint_path_rad']*c['episodes'] for c in same)/n
        a['visited_joint_bins']=int((sum(np.array(c['visited_joint']['counts']) for c in same)>0).sum())
    development = [r for r in rows if r['phase']=='development']
    results = dict(schema='safeduo.mechanism_evidence.v1',campaign_status=statuses,
                   completed_windows=sum(r['completed_windows'] for r in hold),
                   invalid_windows=sum(r['invalid_windows'] for r in hold),
                   pending_windows=768-len(hold)*64,registered_windows=768,
                   development_completed_windows=sum(r['completed_windows'] for r in development),
                   development_registered_windows=128,development_new_command_windows=0,
                   new_unique_command_windows=192,rows=rows,aggregate_rows=aggregate,strata=strata,
                   group_coverage=groups,paired_inputs=pairs,bank_audit=audit_bank(),
                   diagnostic=json.loads((HERE/'causal_results.json').read_text()),
                   actor_sha256=plan['checkpoint_sha256'],source_sha256=plan['source_sha256'],dt=first['dt'],
                   candidate_status='experimental_not_promoted',production_source_unchanged=True,
                   no_holdout_tuning=True,cpu_contract_tests=4,
                   statistical_unit='conditioned matched windows with shared physics; no IID interval',
                   analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    candidate_path=HERE/'candidate_causal_results.json'
    if candidate_path.exists():
        candidate=json.loads(candidate_path.read_text())
        assert all(candidate['baseline_observables_exact'].values()) and candidate['history_equals_actual_fifo']
        candidate['opening_delivered_target_with_closing_measured_rate_first_events']=sum(
            e['context'][-1].get('delivered_linear_backlog_mm',-1)>0 and e['context'][-1].get('full_rate_mm_s',0)<0
            for e in candidate['events'])
        results['candidate_diagnostic']=candidate
        results['additional_repeated_observation_windows']=64
        trace_plan=json.loads((HERE/'candidate_trace_plan_v2.json').read_text())
        for name,sha in trace_plan['research_source_sha256'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    results['registration_audit']=json.loads((HERE/'REGISTRATION_AUDIT.json').read_text())
    (HERE/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:results[k] for k in ('campaign_status','completed_windows','invalid_windows','pending_windows')}))
    print(json.dumps([(r['mode'],r['violations'],r['windows']) for r in aggregate]))


if __name__=='__main__':
    main()
