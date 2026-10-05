"""Check uniqueness of the actual 960-frame inputs, without drawing new tapes."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
RAW=Path('/mnt/nas/data/lyf/double_hand')/HERE.name
OLD=RAW.parent/'safety_mechanism_20261005_causal_obs'

def digest(data):return hashlib.sha256(data).hexdigest()

def main():
    rows=[];fresh=[];old=set();references=[]
    for seed in [1701627244,1651261960,779659058]:
        file=RAW/'holdout'/f'baseline_guard_{seed}'/'input_recipe.npz'
        with np.load(file,allow_pickle=False) as z:
            tape=z['tape'][:960].copy()
        assert tape.shape==(960,64,26) and np.isfinite(tape).all() and not tape[:60].any()
        hashes=[digest(np.ascontiguousarray(tape[:,e]).tobytes()) for e in range(64)]
        fresh.extend(hashes)
        rows.append(dict(seed=seed,path=str(file),sha256=digest(file.read_bytes()),case_sha256=hashes))
    for file in sorted(OLD.rglob('input_recipe.npz')):
        with np.load(file,allow_pickle=False) as z:
            tape=z['tape'][:960].copy()
        assert tape.shape==(960,64,26)
        old.update(digest(np.ascontiguousarray(tape[:,e]).tobytes()) for e in range(64))
        references.append(dict(path=str(file),sha256=digest(file.read_bytes())))
    assert len(set(fresh))==192 and not (set(fresh)&old)
    receipt=dict(status='PASS_ACTUAL_INPUT_UNIQUENESS',fresh_cases=192,fresh_unique_960_frame_tapes=len(set(fresh)),
        prior_unique_960_frame_tapes=len(old),exact_repeats_with_prior=0,rows=rows,references=references,
        verification_source_sha256=digest(Path(__file__).read_bytes()),
        scope='exact first960-frame byte uniqueness within fresh batch and against immediate prior causal-observation namespace; not IID independence or global stochastic coverage proof')
    with (HERE/'command_audit.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
    print(receipt['status'],len(set(fresh)),len(old),flush=True)

if __name__=='__main__':main()
