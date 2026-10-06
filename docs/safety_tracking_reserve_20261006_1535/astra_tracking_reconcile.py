"""Compare already closed independent raw results with parent results; no simulation.

Parent results never supply the independent score. Native minima are converted
to displayed millimetres in float32, exactly as in the reviewed parent source.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import numpy as np

H = Path(__file__).resolve().parent
CLASSES = ['cross', 'self_F', 'self_U', 'table']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compare(own, parent):
    failures = []
    checks = 0
    def check(ok, label):
        nonlocal checks
        checks += 1
        if not ok:
            failures.append(label)

    check(own['status'] == 'PASS_COMPLETE_INDEPENDENT_TRACKING_NUMERIC', 'own closure')
    check((parent['registered_method_windows'], parent['completed_method_windows'], parent['invalid_method_windows']) == (576, 576, 0), 'parent window contract')
    oc = {(c['command_seed'], c['env']): c for c in own['cases']}
    pc = {(c['seed'], c['env']): c for c in parent['cases']}
    check(len(oc) == len(own['cases']) == len(pc) == len(parent['cases']) == 192, 'unique 192 case identities')
    check(set(oc) == set(pc), 'case identity sets')
    modes = own['modes']
    records = []
    for key in sorted(set(oc) & set(pc)):
        a, b = oc[key], pc[key]
        check(set(b['methods']) == set(modes), f'{key}: mode inventory')
        for mode in modes:
            w, m = a['windows'][mode], b['methods'][mode]
            metric = w['metrics']
            tag = f'{a["case_id"]}:{mode}'
            check(w['status'] == 'VERIFIED', tag + ': own status')
            check(w['risk_pair_index'] == b['stratum'], tag + ': stratum')
            check(m['failed'] == metric['strict']['failed'], tag + ': strict flag')
            check(m['deep'] == metric['deep']['failed'], tag + ': deep flag')
            check(m['first_failure_step'] == metric['strict']['first_step'], tag + ': strict first step')
            minima = np.asarray([metric['minimum_nonexempt_margin_m'][k] for k in CLASSES], np.float32)
            expected_mm = (minima * np.float32(1000)).tolist()
            check(m['min_mm'] == expected_mm, tag + ': exact native float32 class minima displayed mm')
            check([v < 0 for v in m['min_mm']] == [metric['strict']['class_failed'][k] for k in CLASSES], tag + ': strict class signs')
            records.append(dict(case_id=a['case_id'], mode=mode, strict=metric['strict']['failed'], deep=metric['deep']['failed'], first_step=metric['strict']['first_step'], class_minima_mm=expected_mm))

    totals = {r['mode']: r for r in parent['totals']}
    check(len(totals) == len(parent['totals']) == 3 and set(totals) == set(modes), 'totals mode inventory')
    for mode in modes:
        a, b = own['counts'][mode], totals[mode]
        for k, expected in [('completed_windows',192), ('violations',a['strict']['failed_windows']), ('deep',a['deep']['failed_windows']), ('strict_env_steps',a['strict']['env_steps']), ('deep_env_steps',a['deep']['env_steps'])]:
            check(b[k] == expected, mode + ': total ' + k)
        check(b['class_violations'] == [a['strict']['class_failed_windows'][k] for k in CLASSES], mode + ': total strict classes')
        check(b['min_nonexempt_mm'] == float(np.float32(min(a['minimum_nonexempt_margin_m'].values())) * np.float32(1000)), mode + ': total minimum')
    orows = {c['id']: c for c in own['conditions']}
    prows = {c['id']: c for c in parent['rows']}
    check(len(orows) == len(prows) == len(parent['rows']) == 9 and set(orows) == set(prows), '9 condition identities')
    hash_bindings = 0
    for cid in sorted(set(orows) & set(prows)):
        a, b = orows[cid], prows[cid]
        check(b['status'] == 'complete' and b['completed_windows'] == 64 and b['invalid_windows'] == 0, cid + ': parent condition valid')
        check((b['violations'], b['deep']) == (a['counts']['strict']['failed_windows'], a['counts']['deep']['failed_windows']), cid + ': condition flags')
        check(b['class_violations'] == [a['counts']['strict']['class_failed_windows'][k] for k in CLASSES], cid + ': condition classes')
        check(b['path'] == a['root'], cid + ': raw root')
        for rel, digest in b['input_sha256'].items():
            path = str(Path(b['path']) / rel)
            check(own['input_sha256'].get(path) == digest, cid + ': same consumed raw hash ' + rel)
            hash_bindings += 1
    paired_rows = []
    seen = set()
    for b in parent['paired']:
        key = (b['seed'], b['a'], b['b'])
        check(key not in seen, 'duplicate paired block'); seen.add(key)
        cases = [c for c in own['cases'] if c['command_seed'] == b['seed']]
        counts = dict(rescued=0, new_failures=0, both=0, neither=0)
        for c in cases:
            af = c['windows'][b['a']]['metrics']['strict']['failed']
            bf = c['windows'][b['b']]['metrics']['strict']['failed']
            counts['both' if af and bf else 'rescued' if af else 'new_failures' if bf else 'neither'] += 1
        check(len(cases) == 64 and all(b[k] == v for k,v in counts.items()), f'{key}: strict paired partition')
        paired_rows.append(dict(seed=b['seed'], a=b['a'], b=b['b'], **counts))
    expected_pairs = {(seed, p['reference'], p['candidate']) for seed,_ in oc for p in own['paired'].values()}
    check(seen == expected_pairs and len(seen) == 9, 'all registered paired block comparisons')
    check(parent['unique_actual_tapes'] == len({c['windows'][modes[0]]['tape_sha256'] for c in own['cases']}) == 192, 'unique actual tapes')
    return dict(checks=checks, failures=failures, cases=records, paired_blocks=paired_rows, matched_raw_hash_bindings=hash_bindings)


def main():
    files = ['ASTRA_TRACKING_FINAL_SCORE.json','holdout_results.json','ANALYSIS_EXECUTION.json','analyze.py','dense_audit.py',Path(__file__).name]
    before = {n:sha(H/n) for n in files}
    own = json.loads((H/files[0]).read_text()); parent = json.loads((H/files[1]).read_text())
    r = compare(own, parent)
    execution = json.loads((H/'ANALYSIS_EXECUTION.json').read_text())
    assert execution['status'].startswith('PASS') and execution['source_after_verified']
    assert execution['result_sha256'] == before['holdout_results.json']
    assert parent['analysis_source_sha256'] == before['analyze.py']
    assert parent['dense_audit_source_sha256'] == before['dense_audit.py']
    mutations = []
    changes = [lambda p:p['cases'][0]['methods']['joint_reference'].__setitem__('failed',True),
               lambda p:p['cases'][0]['methods']['joint_reference']['min_mm'].__setitem__(0,float(np.nextafter(np.float32(p['cases'][0]['methods']['joint_reference']['min_mm'][0]),np.float32(np.inf)))),
               lambda p:p['totals'][0].__setitem__('deep_env_steps',98),
               lambda p:p['paired'][0].__setitem__('rescued',1),
               lambda p:p['cases'].append(p['cases'][0]),
               lambda p:p['rows'][0]['input_sha256'].__setitem__('cell_001.npz','0'*64)]
    for i, mutate in enumerate(changes):
        altered = copy.deepcopy(parent); mutate(altered)
        bad = compare(own, altered)['failures']; assert bad, f'undetected reconciliation mutation {i}'
        mutations.append(dict(mutation=i, rejected=True, first_finding=bad[0]))
    after = {n:sha(H/n) for n in files}; assert before == after
    receipt = dict(schema='astra.tracking.parent_reconciliation.v1', status='PASS_EXACT_PARENT_RECONCILIATION' if not r['failures'] else 'FAIL_PARENT_RECONCILIATION', utc=datetime.now(timezone.utc).isoformat(),
        case_identities=192, method_windows=576, input_sha256=before, inputs_before_after_equal=True,
        **r, negative_controls=mutations, parent_results_used_for_independent_scoring=False,
        arithmetic='Exact float32 m*float32(1000) for parent presentation; native strict <0 and deep <-float32(.005), no epsilon.',
        unavailable_parent_fields=['per-case deep first step','per-case duration','per-case deep class flags','per-class deep totals and duration'],
        unavailable_scope='These remain independently raw-scored in FINAL_SCORE, but have no corresponding parent JSON fields to reconcile.',
        motion_coverage_scope='Parent L2 path and moving-fraction are source-reviewed parent measurements; independent scorer L1 path is a different metric and is not compared.',
        physical_safety_certified=False)
    target = H/'ASTRA_TRACKING_FINAL_RECONCILIATION.json'
    with target.open('x') as f: json.dump(receipt,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({k:receipt[k] for k in ['status','case_identities','method_windows','checks','failures','matched_raw_hash_bindings']}))
    assert not r['failures']


if __name__ == '__main__': main()
