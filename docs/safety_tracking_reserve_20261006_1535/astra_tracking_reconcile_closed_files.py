"""Supplement only the 18 previously unconsumed file identities.

Keep the initial FAIL receipt and independent frozen scoring files unchanged.
No NPZ arrays are evaluated here; this is byte identity, not metric validation.
"""
import copy
from datetime import datetime, timezone
import json
import numpy as np
from astra_tracking_reconcile import H, sha, compare


def main():
    names = ['ASTRA_TRACKING_FINAL_SCORE.json','holdout_results.json','ASTRA_TRACKING_FINAL_RECONCILIATION.json','astra_tracking_reconcile.py',__file__.split('/')[-1]]
    inputs = {n:sha(H/n) for n in names}
    own = json.loads((H/names[0]).read_text()); parent = json.loads((H/names[1]).read_text())
    initial = json.loads((H/names[2]).read_text())
    assert initial['status'] == 'FAIL_PARENT_RECONCILIATION' and len(initial['failures']) == 18
    extras = {}
    for row in parent['rows']:
        for rel, digest in row['input_sha256'].items():
            path = row['path'] + '/' + rel
            if path not in own['input_sha256']:
                assert rel in ('mechanism.npz','project_diagnostics.npz')
                extras[path] = digest
    assert len(extras) == 18
    before = {p:sha(p) for p in extras}
    assert before == extras, 'actual supplemental file bytes differ from parent readback'
    # An explicitly separate in-memory identity view; never written over SCORE.
    enriched = copy.deepcopy(own)
    enriched['input_sha256'].update(before)
    result = compare(enriched, parent)
    assert not result['failures'], result['failures']
    mutations = []
    changes = [lambda p:p['cases'][0]['methods']['joint_reference'].__setitem__('failed',True),
               lambda p:p['cases'][0]['methods']['joint_reference']['min_mm'].__setitem__(0,float(np.nextafter(np.float32(p['cases'][0]['methods']['joint_reference']['min_mm'][0]),np.float32(np.inf)))),
               lambda p:p['totals'][0].__setitem__('deep_env_steps',98),
               lambda p:p['paired'][0].__setitem__('rescued',1),
               lambda p:p['cases'].append(p['cases'][0]),
               lambda p:p['rows'][0]['input_sha256'].__setitem__('cell_001.npz','0'*64),
               lambda p:p['rows'][0]['input_sha256'].__setitem__('project_diagnostics.npz','0'*64)]
    for i, mutate in enumerate(changes):
        altered=copy.deepcopy(parent); mutate(altered)
        bad=compare(enriched,altered)['failures']; assert bad
        mutations.append(dict(mutation=i, rejected=True, findings=bad))
    after = {p:sha(p) for p in extras}
    assert before == after == extras
    assert inputs == {n:sha(H/n) for n in names}
    utc=datetime.now(timezone.utc).isoformat()
    receipt=dict(status='PASS_SUPPLEMENTAL_CLOSED_FILE_BYTE_IDENTITY',utc=utc,
        files=18,before_sha256=before,after_sha256=after,parent_readback_sha256=extras,
        raw_before_after_equal=True,inputs_sha256=inputs,
        scope='Only byte identity for files absent from original independent scorer consumption ledger. No score arithmetic rerun, no diagnostic array semantics verified by this receipt.',
        missing_hash_is_not_treated_as_match=True,original_score_unchanged=True,original_failure_retained=True)
    with (H/'astra_tracking_supplemental_file_identity.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    output=dict(status='PASS_EXACT_PARENT_RECONCILIATION_WITH_SEPARATE_FILE_IDENTITY',utc=utc,
        **result,case_identities=192,method_windows=576,input_sha256=inputs,
        supplemental_identity_sha256=sha(H/'astra_tracking_supplemental_file_identity.json'),
        parent_fields_absent=initial['unavailable_parent_fields'],motion_coverage_scope=initial['motion_coverage_scope'],
        unchanged_frozen_independent_score_sha256=inputs[names[0]],initial_failure_receipt=names[2],
        explanation='Initial failure compared all parent consumed files to a narrower independent scorer ledger. Eighteen actual file hashes were missing, not unequal. They have now been read twice and exactly match the parent consumed-byte bindings.',
        negative_controls=mutations,negative_control_scope='Each mutated comparison fails from the now PASS baseline; initial negative controls were confounded by the 18 missing-file failures and are not relied upon.',
        physical_safety_certified=False)
    with (H/'ASTRA_TRACKING_FINAL_RECONCILIATION_REREVIEW.json').open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps({k:output[k] for k in ['status','checks','failures','matched_raw_hash_bindings','case_identities','method_windows']}))


if __name__ == '__main__':main()
