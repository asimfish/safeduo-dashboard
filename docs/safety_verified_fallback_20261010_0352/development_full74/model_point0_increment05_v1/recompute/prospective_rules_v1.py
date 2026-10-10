"""Deterministic MODEL_ONLY accounting. No RNG, physics launch or source edits.

No FK model is silently supplied: a separately source-reviewed, recorded-native
calibrated full nonlinear q74->9021 evaluator is a REQUIRED future input.
These functions accept arrays; they never authorize reading the unseen cohort.
"""
import hashlib,json
from pathlib import Path
import numpy as np

def need(v,s):
    if not v:raise ValueError(s)

def known_or_reserved():
    ids={(0,i) for i in range(32)}
    for seed,start in enumerate((64,512,672,1120)):ids.update((seed,i) for i in range(start,start+32))
    ids.update((0,i) for i in range(92,108))
    return frozenset(ids)

def model_rows(first_global_id,gaps):
    """Every original row kept, no dynamic exemption removes a distance row."""
    need(type(first_global_id) is int and first_global_id%32==0 and 0<=first_global_id<=8160,'exact registered32 block')
    need(gaps.shape==(32,9021) and gaps.dtype==np.float64,'full float64 MODEL_ONLY9021 predictions')
    rows=[]
    for lane,g in enumerate(gaps):
        global_id=first_global_id+lane;seed,local=divmod(global_id,2048);finite=np.isfinite(g)
        negative=finite&(g<0);kind='MODEL_UNKNOWN' if not finite.all() else 'MODEL_NEGATIVE' if negative.any() else 'MODEL_PREDICTED_INITIAL_CLEAR'
        rows.append(dict(global_id=global_id,seed=seed,proposal_id=local,family=local%4,model_status=kind,
            finite_distance_count=int(finite.sum()),negative_rows=np.flatnonzero(negative).tolist(),
            minimum_model_gap_m=float(g.min()) if finite.all() else None,
            known_or_reserved=(seed,local) in known_or_reserved(),native_status='NOT_TESTED',camera_status='NOT_TESTED',eligible=False))
    return rows

def exhaustive_accounting(rows):
    need(len(rows)==8192 and [r['global_id'] for r in rows]==list(range(8192)),'full8192 once, ordered, no outcome truncation')
    cells=[]
    for seed in range(4):
        for family in range(4):
            members=[r for r in rows if r['seed']==seed and r['family']==family]
            need(len(members)==512,'all512 per seed/family')
            for r in members:
                need(divmod(r['global_id'],2048)==(seed,r['proposal_id']) and r['proposal_id']%4==family,'unaltered identity')
                need(r['model_status'] in ('MODEL_UNKNOWN','MODEL_NEGATIVE','MODEL_PREDICTED_INITIAL_CLEAR') and r['native_status']==r['camera_status']=='NOT_TESTED' and r['eligible'] is False,'model cannot impersonate native eligibility')
                need(r['known_or_reserved']==((seed,r['proposal_id']) in known_or_reserved()),'fixed previously seen/reserved exclusions')
            cells.append(dict(seed=seed,family=family,denominator=512,counts={k:sum(r['model_status']==k for r in members) for k in ('MODEL_UNKNOWN','MODEL_NEGATIVE','MODEL_PREDICTED_INITIAL_CLEAR')},reserved=sum(r['known_or_reserved'] for r in members)))
    return dict(status='MODEL_ONLY_ALL8192_ACCOUNTED',denominator=8192,cells=cells,native_tested=0,qualified=0,bank_count=0)

def prospective_first32(rows):
    """Fixed proposal: earliest2 unseen MODEL_CLEAR/cell =32, no borrowing.

    This is a prospective schedule, never bank selection. Native true admission
    still needs all physical gates/strong cameras/actual waits/raw audit.
    """
    counts=exhaustive_accounting(rows);chosen=[];short=[]
    for seed in range(4):
        for family in range(4):
            options=[r for r in rows if r['seed']==seed and r['family']==family and not r['known_or_reserved'] and r['model_status']=='MODEL_PREDICTED_INITIAL_CLEAR']
            if len(options)<2:short.append(dict(seed=seed,family=family,available=len(options),required=2))
            chosen.extend(dict(seed=r['seed'],proposal_id=r['proposal_id'],global_id=r['global_id'],family=r['family']) for r in options[:2])
    return dict(status='MODEL_QUOTA_SHORTFALL_NO_NATIVE_SCHEDULE' if short else 'PROSPECTIVE_NATIVE32_NOT_ADOPTED',
        candidates=[] if short else chosen,shortfall=short,counts=counts,bank_count=0,eligible=0,requires_new_parent_adoption=True)

def development_calibration(model_gaps,native_gaps,ids):
    """Observed replay error only; 5e-5 is model-release diagnostic, not a bound.

    No additive error margin, sign epsilon or data-dependent threshold tuning.
    Model calibration and verification halves are predeclared per group-of4
    local pilot positions; every seed/family has four states in each half.
    """
    expected=[(s,i) for s,start in enumerate((64,512,672,1120)) for i in range(start,start+32)]
    need(ids==expected and model_gaps.shape==native_gaps.shape==(128,9021),'all128 declared native point0 replay')
    need(model_gaps.dtype==native_gaps.dtype==np.float64 and np.isfinite(model_gaps).all() and np.isfinite(native_gaps).all(),'finite actual model/native distances')
    mask=np.array([((i-(64,512,672,1120)[s])//4)%2==0 for s,i in ids]);error=np.abs(model_gaps-native_gaps);sign=(model_gaps<0)!=(native_gaps<0)
    output={}
    for name,m in [('calibration',mask),('development_verification',~mask)]:
        output[name]=dict(states=int(m.sum()),max_abs_error_m=float(error[m].max()),RMS_error_m=float(np.sqrt(np.mean(error[m]**2))),sign_disagreements=int(sign[m].sum()))
    ready=all(x['max_abs_error_m']<=5e-5 and x['sign_disagreements']==0 for x in output.values())
    return dict(status='EMPIRICAL_MODEL_REPLAY_READY_FOR_REVIEW' if ready else 'MODEL_REPLAY_MISMATCH_DO_NOT_SCREEN8192',
        replay=output,diagnostic_tolerance_m=5e-5,formal_holdout=False,global_error_bound='UNKNOWN',native_admission=False)

def stream_registered_geometry(q74,qd74,model,write_block,write_rows):
    """Integrable bounded loop; caller MUST bind/preregister model and authority.

    Model returns only geometry, not a PD rollout. Original qd is retained and
    hashed; never reset/clipped/projected to improve the model/native outcome.
    No parent runner or future authority is created by this callable.
    """
    need(q74.shape==qd74.shape==(8192,74) and q74.dtype==qd74.dtype==np.float32,'original cohort typed all8192')
    for first in range(0,8192,32):
        q=q74[first:first+32].copy();v=qd74[first:first+32].copy();qb=q.tobytes();vb=v.tobytes()
        q.setflags(write=False);v.setflags(write=False)
        g=model.predict_geometry(q,global_ids=np.arange(first,first+32,dtype=np.int64))
        need(q.tobytes()==qb and v.tobytes()==vb,'no proposal mutation by model')
        records=model_rows(first,g)
        # Write sequential float64 NPY payload/chunks on NAS; do not mmap a
        # 591MB array into the 1GiB worker just to hold model outputs.
        write_block(first,g);write_rows(first,records,qb,vb)
    return dict(status='MODEL_ONLY_STREAM_FINISHED_PENDING_EXTERNAL_WAIT',blocks=256,denominator=8192,native_pass=False)
