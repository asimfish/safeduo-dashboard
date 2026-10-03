"""Audit every environment, actual target bounds, FIFO and observed prefixes."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
OLD = OUT.parent / 'safety_queue_braking_20261003'


def load(path):
    with np.load(path) as n:
        keys = ['q', 'pre_q', 'pre_qd', 'pre_target', 'cmd', 'exec', 'official_margins',
                'sphere_centers', 'q_initial', 'initial_violation', 'joint_soft_limits',
                'controller_target', 'actuator_target', 'pre_pending_targets']
        keys += [k for k in n.files if k.startswith('stop_')]
        d = {k:n[k] for k in keys}
        d['meta'] = json.loads(str(n['meta_json']))
        for k in keys:
            assert np.isfinite(d[k]).all(), (path,k)
    return d


def bad_steps(d, e):
    return np.flatnonzero((d['official_margins'][:,e] < 0).any(-1))


def first(x):
    return int(x[0]) if len(x) else None


def audit(d, episodes, cap):
    sent, delivered = d['controller_target'], d['actuator_target']
    lag = d['meta']['actuator_delay_steps']
    assert lag == 6
    expected = np.concatenate([np.repeat(d['pre_target'][0:1],lag,axis=0),sent[:-lag]])
    assert np.array_equal(expected,delivered), 'FIFO changed'
    assert np.array_equal(d['pre_target'][1:],sent[:-1]), 'last issued target differs'
    increment = np.abs(sent-d['pre_target'])
    assert increment.max() <= cap+5e-7, 'ordinary output box exceeded'
    limits=d['joint_soft_limits']
    assert (sent >= limits[...,0]-5e-7).all() and (sent <= limits[...,1]+5e-7).all()
    bad=np.flatnonzero((d['official_margins']<0).any((0,2))).tolist()
    assert bad == [e['env_id'] for e in episodes if e['violation']]
    assert not d['initial_violation'].any()
    assert not any(e['initial_violation'] for e in episodes)
    return dict(episodes=len(episodes),initially_safe=len(episodes),initial_violations=0,
                violations=len(bad),failed_envs=bad,
                damaging_envs=np.flatnonzero((d['official_margins'] < -.005).any((0,2))).tolist(),
                min_nonexempt_mm=float(d['official_margins'].min()*1000),
                max_actual_increment_rad=float(increment.max()),increment_box_rad=cap,
                increment_box_pass=True,soft_limit_pass=True,fifo_exact=True,fifo_steps=lag,
                first_failures={str(e):first(bad_steps(d,e)) for e in bad},
                mean_joint_path_rad=float(np.mean([e['measured_joint_path_rad'] for e in episodes])),
                mean_joint_range=float(np.mean([e['joint_range_fraction_mean'] for e in episodes])))


def main():
    campaign=json.loads((OUT/'cells/campaign.json').read_text())
    assert campaign['status']=='complete' and len(campaign['jobs'])==4
    assert len({j['pid'] for j in campaign['jobs']})==4
    plan=json.loads((OUT/'campaign_plan.json').read_text())
    assert campaign['plan']==plan
    for name,sha in plan['research_files_sha256'].items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==sha, name
    datasets={j['id']:load(OUT/'cells'/j['id']/'cell_001.npz') for j in campaign['jobs']}
    control=datasets['control']
    p=json.loads((OUT/'cells/control/protocol.json').read_text())
    cap=p['effective_backstop']['vmax']*p['dt']
    rows=[]
    all_records=[]
    for job in campaign['jobs']:
        name=job['id']; path=OUT/'cells'/name
        protocol=json.loads((path/'protocol.json').read_text())
        assert protocol['status']=='complete' and protocol['completed_cells']==1 and len(protocol['design'])==1
        for k in ['source_sha256','resolved_config','effective_backstop','effective_coordinator','checkpoint_sha256','dt','steps']:
            assert protocol[k]==p[k], (name,k)
        assert protocol['source_sha256']==plan['source_sha256']
        assert protocol['checkpoint_sha256']==plan['checkpoint_sha256']
        d=datasets[name]
        ep=json.loads((path/'cell_001.json').read_text())['episodes']
        assert len(ep)==32 and len(d['q'])==600
        row=dict(id=name,lead_steps=None if name=='control' else int(name.split('_')[1]),pid=job['pid'],**audit(d,ep,cap))
        assert np.array_equal(d['q_initial'],control['q_initial'])
        assert np.array_equal(d['cmd'],control['cmd'])
        row['inputs_exact']=True
        row['new_failed_envs']=sorted(set(row['failed_envs'])-set(np.flatnonzero((control['official_margins']<0).any((0,2))).tolist()))
        row['trajectories']=[]
        if name!='control':
            manifest=json.loads((path/'stop_manifest.json').read_text())
            assert not manifest['queue_preemption'] and not manifest['actuator_side_control']
            assert manifest['increment_box_rad']==cap
            assert np.array_equal(d['stop_sent_target'],d['controller_target'])
            recipe=json.loads((OUT/f'schedule_{row["lead_steps"]}.json').read_text())
            row['triggered_envs']=[e for e,t in enumerate(recipe['trigger_steps']) if t>=0]
            row['non_intervened_failed_envs']=sorted(set(row['failed_envs'])-set(row['triggered_envs']))
            row['intervened_violations']=len(set(row['failed_envs'])&set(row['triggered_envs']))
            for e,t in enumerate(recipe['trigger_steps']):
                if t<0:
                    assert not d['stop_active'][:,e].any()
                    continue
                arrival=t+6
                assert np.array_equal(d['stop_active'][:,e],np.arange(len(d['q']))>=t)
                goal=np.clip(d['pre_q'][t,e],d['joint_soft_limits'][e,:,0],d['joint_soft_limits'][e,:,1])
                assert np.all(d['stop_fixed_target'][t:,e]==goal)
                assert np.array_equal(d['stop_sent_target'][:t,e],d['stop_proposed_target'][:t,e])
                distance=np.abs(d['controller_target'][t:,e]-goal).max(-1)
                converged=first(np.flatnonzero(distance<=5e-7))
                if converged is not None:
                    assert (distance[converged:]<=5e-7).all()
                prefix={k:np.array_equal(d[k][:t+1,e],control[k][:t+1,e]) for k in ['pre_q','pre_qd','pre_target','pre_pending_targets']}
                prefix.update({k:np.array_equal(d[k][:t,e],control[k][:t,e]) for k in ['sphere_centers','cmd','controller_target']})
                prefix['actuator_target']=np.array_equal(d['actuator_target'][:arrival,e],control['actuator_target'][:arrival,e])
                bad=bad_steps(d,e)
                item=dict(env=e,trigger_step=t,first_stop_message_delivery_step=arrival,
                          observed_prefix_exact=all(prefix.values()),prefix_fields=prefix,
                          prefix_max_q_difference_rad=float(np.abs(d['pre_q'][:t+1,e]-control['pre_q'][:t+1,e]).max()),
                          old_target_offset_rad=float(np.abs(d['pre_target'][t,e]-goal).max()),
                          goal_reached_issue_step=None if converged is None else t+converged,
                          goal_reached_delivery_step=None if converged is None else t+converged+6,
                          first_failure_step=first(bad),violation_before_first_delivery=bool((bad<arrival).any()),
                          violation_after_first_delivery=bool((bad>=arrival).any()),
                          min_nonexempt_mm=float(d['official_margins'][:,e].min()*1000),
                          max_actual_increment_after_trigger_rad=float(np.abs(d['controller_target'][t:,e]-d['pre_target'][t:,e]).max()))
                row['trajectories'].append(item)
        else:
            row.update(triggered_envs=[],non_intervened_failed_envs=row['failed_envs'],intervened_violations=0)
        rows.append(row)
        for e in range(32):
            all_records.append(dict(condition=name,env=e,violation=e in row['failed_envs'],
                                    triggered=e in row['triggered_envs'],first_failure_step=first(bad_steps(d,e)),
                                    min_nonexempt_mm=float(d['official_margins'][:,e].min()*1000)))
    old=load(OLD/'reverse_controls/cell_001.npz')
    old_exact={k:np.array_equal(control[k],old[k]) for k in ['q','pre_qd','exec','official_margins','controller_target','sphere_centers']}
    results=dict(schema='safeduo.bounded_stop_evidence.v1',candidate_status='experimental_not_promoted',
                 dt=p['dt'],rows=rows,windows=128,control_windows=32,posthoc_diagnostic_windows=96,
                 independent_safety_windows_added=0,distinct_processes=True,one_cell_per_process=True,
                 command_exact=True,initial_exact=True,
                 old_control_replay_exact_fields=old_exact,
                 old_control_max_q_difference_rad=float(np.abs(control['q']-old['q']).max()),
                 source_changes_from_previous=json.loads((OUT/'source_changes.json').read_text()),
                 checkpoint_sha256=p['checkpoint_sha256'],source_sha256=p['source_sha256'],
                 ordinary_increment_box_rad=cap,tests_passed=17)
    (OUT/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    with (OUT/'all_environments.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_records[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(all_records)
    print(json.dumps({'violations':{r['id']:r['violations'] for r in rows},
                      'fifo_and_bounds':all(r['fifo_exact'] and r['increment_box_pass'] for r in rows),
                      'observed_prefix_exact':sum(x['observed_prefix_exact'] for r in rows for x in r['trajectories']),
                      'old_control_exact':all(old_exact.values())}))


if __name__=='__main__':
    main()
