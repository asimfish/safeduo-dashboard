"""Actual960-frame input uniqueness and fixed scaling, against enumerated studies."""
from pathlib import Path
import json,hashlib,numpy as np
HERE=Path(__file__).resolve().parent;RAW=Path('/mnt/nas/data/lyf/double_hand')/HERE.name
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes(p):
    with np.load(p,allow_pickle=False) as z:tape=z['tape'][:960].copy()
    assert tape.shape==(960,64,26) and np.isfinite(tape).all()
    return [hashlib.sha256(np.ascontiguousarray(tape[:,e]).tobytes()).hexdigest() for e in range(64)]
if __name__=='__main__':
    fresh=[];rows=[]
    for i,r in enumerate(json.loads((HERE/'RANDOM_EXPERIMENT_DESIGN.json').read_text())['rows']):
        p=RAW/f'holdout_{i}'/f'joint_reference_{r["command_seed"]}'/'input_recipe.npz'
        h=hashes(p);fresh.extend(h);rows.append(dict(path=str(p),sha256=sha(p),case_sha256=h))
    prior=set();refs=[]
    for namespace in ('safety_mechanism_20261005_causal_obs','safety_joint_guard_20261005_1005','safety_feasible_guard_20261005_2100'):
        for p in sorted((RAW.parent/namespace).rglob('input_recipe.npz')):
            prior.update(hashes(p));refs.append(dict(path=str(p),sha256=sha(p)))
    assert len(set(fresh))==192 and not set(fresh)&prior
    receipt=dict(status='PASS_ACTUAL_960_INPUT_UNIQUENESS',fresh_unique_tapes=len(set(fresh)),prior_unique_tapes=len(prior),
        exact_repeats_with_prior=0,rows=rows,prior_references=refs,source_sha256=sha(Path(__file__)),
        scope='current actual960 unscaled external inputs vs stored first960 recipes in enumerated prior namespaces, including aborted prospective recipes; no claim prior recipes all executed or IID independence')
    (HERE/'command_audit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['status'],len(set(fresh)),len(prior))
