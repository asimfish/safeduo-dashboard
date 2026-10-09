"""Every fresh proposal and every archived raw9021 geometry row; no safety inference."""
import hashlib
import json
from pathlib import Path

import numpy as np

P=Path(__file__).resolve().parent
H=P.parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reg=json.loads((P/'NATIVE_GEOMETRY_REG_V1.json').read_text())
    execution=json.loads((H/'strong_geometry32000_v2_execution.json').read_text())
    assert execution['status']=='complete' and len(execution['jobs'])==1 and execution['jobs'][0]['actual_exit']==0
    leaf=Path(execution['jobs'][0]['out'])
    summary=json.loads((leaf/'sampling_summary.json').read_text())
    assert summary['candidate_count']==32000 and summary['status']=='complete'
    assert sha(leaf/'bank.npz')==summary['bank_sha256']
    with np.load(reg['proposal_input']) as z:expected=z['controlled_q'];box=z['soft_box_rad']
    with np.load(leaf/'bank.npz') as z:bank={k:z[k] for k in z.files}
    assert np.array_equal(bank['all_proposal_q'],expected)
    assert np.array_equal(bank['soft_limits'],box)
    receipt=json.loads((leaf/'geometry_chunk_receipts.json').read_text())
    assert len(receipt['chunks'])==500 and receipt['proposal_rows']==32000
    minimum=[];selected=[]
    for i,chunk in enumerate(receipt['chunks']):
        path=leaf/chunk['path'];assert sha(path)==chunk['sha256']
        with np.load(path) as z:
            assert np.array_equal(z['proposal_id'],np.arange(i*64,(i+1)*64))
            raw=z['raw_geometry_m'];assert raw.shape==(64,9021) and np.isfinite(raw).all()
            gaps=raw.min(-1);minimum.extend(gaps.tolist())
            for lane in np.flatnonzero(gaps>=.0001):
                if len(selected)<512:selected.append(raw[lane].copy())
    minimum=np.asarray(minimum,dtype=bank['all_min_raw_gap_m'].dtype)
    assert np.array_equal(minimum,bank['all_min_raw_gap_m'])
    eligible=np.flatnonzero(minimum>=.0001);ids=eligible[:512]
    assert len(ids)==512 and np.array_equal(ids,bank['selected_indices'])
    assert np.array_equal(eligible,bank['all_eligible_indices'])
    assert np.array_equal(bank['accepted_q'],expected[ids])
    assert np.array_equal(bank['accepted_d'],np.stack(selected))
    roots=json.loads((leaf/'actual_solver_roots.json').read_text())
    assert len(roots)==4 and all(x['position_iterations']==64 and x['velocity_iterations']==0 for x in roots)
    span=(expected.max(0)-expected.min(0))/(box[:,1]-box[:,0])
    selected_span=(bank['accepted_q'].max(0)-bank['accepted_q'].min(0))/(box[:,1]-box[:,0])
    result=dict(status='PASS_FRESH32000_NATIVE_GEOMETRIC_INPUT_AUDIT_ONLY',all32000_proposals_exact_to_frozen_inputs=True,all500_raw9021_chunks_finite_hash_verified=True,all_rejects_retained=True,proposals=32000,raw_negative=int((minimum<0).sum()),nonnegative_below_buffer=int(((minimum>=0)&(minimum<.0001)).sum()),geometrically_eligible=int(len(eligible)),first_geometric_selected=512,initial_marginal_span_fraction=[float(span.min()),float(span.max())],selected_marginal_span_fraction=[float(selected_span.min()),float(selected_span.max())],physical_controls_after_reset=0,physical_contacts_qualified=False,random_velocity_or_targets_executed=False,formal_holdout=False,fullSystem0_accepted=False,interpretation='Initial-position geometry qualification only; next12-step native joint/contact/velocity prefix gate still required before controller comparison',bank_sha256=sha(leaf/'bank.npz'),native_execution_sha256=sha(H/'strong_geometry32000_v2_execution.json'),analysis_source_sha256=sha(Path(__file__)))
    (P/'GEOMETRIC32000_RESULT_V1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)


if __name__=='__main__':main()
