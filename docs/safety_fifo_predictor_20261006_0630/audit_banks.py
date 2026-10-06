"""All first-qualified samples, freshness relative to enumerated earlier banks."""
from pathlib import Path
import numpy as np,json,hashlib
HERE=Path(__file__).resolve().parent
RAW=Path('/mnt/nas/data/lyf/double_hand')/HERE.name
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
    d=json.loads((HERE/'RANDOM_EXPERIMENT_DESIGN.json').read_text());fresh=[];rows=[]
    geom=load(RAW/'banks/all_geometry.npz');settle=load(RAW/'banks/all_settling.npz')
    for r in d['rows']:
        p=RAW/'banks'/str(r['initial_seed'])/'bank.npz';z=load(p);meta=json.loads(p.with_name('metadata.json').read_text())
        assert sha(p)==meta['bank_sha256'];assert meta['status']=='complete' and not meta['policy_outcomes_used']
        q=z['accepted_q'];labels=z['risk_pair_index'];assert q.shape==(64,26) and [(labels==i).sum() for i in range(6)]==[8]*6 and (labels==-1).sum()==16
        for qrow,label,(gidx,e) in zip(q,labels,z['selected_refs']):
            ident=geom['batch_identity'][gidx];batchq=geom['q'][gidx*64:(gidx+1)*64]
            assert np.array_equal(qrow,batchq[e]) and geom['eligible'][gidx*64+e]
            if label>=0:assert .020<=geom['pairs'][gidx*64+e,label]<=.060
            found=np.flatnonzero((settle['identity']==ident).all(-1));assert len(found)==1
            s=found[0];assert settle['valid'][s,e] and not settle['bad'][s,e] and settle['drift'][s,e]<=.05 and settle['velocity'][s,e]<=.1
        fresh.append(q);rows.append(dict(seed=r['initial_seed'],sha256=sha(p),qualified=64))
    fresh=np.concatenate(fresh);prior=[];refs=[]
    for root in sorted(RAW.parent.glob('safety*')):
        if root==RAW or not root.is_dir():continue
        for p in sorted(root.rglob('bank.npz')):
            z=load(p)
            if 'accepted_q' in z and z['accepted_q'].ndim==2 and z['accepted_q'].shape[1]==26:
                prior.append(z['accepted_q']);refs.append(dict(path=str(p),sha256=sha(p),count=len(z['accepted_q'])))
    assert prior;prior=np.concatenate(prior)
    hashes=[hashlib.sha256(q.tobytes()).hexdigest() for q in fresh]
    oldhash={hashlib.sha256(q.tobytes()).hexdigest() for q in prior}
    assert len(set(hashes))==192 and not set(hashes)&oldhash
    dist=np.linalg.norm(fresh[:,None,:].astype(np.float64)-prior[None,:,:],axis=-1)
    receipt=dict(status='PASS_ENUMERATED_BANK_FRESHNESS',fresh=192,prior_compared=len(prior),exact_duplicates=0,
        minimum_l2_distance_rad=float(dist.min()),rows=rows,prior_references=refs,
        full_joint_volume_certified=False,scope='only explicitly enumerated bank paths; source sampling order unchanged and selected references qualified')
    (HERE/'bank_audit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ('rows','prior_references')}))
