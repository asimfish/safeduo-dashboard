"""Recompute registered MODEL_ONLY geometry, never physics/camera/admission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--out');parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--first-block',type=int,default=0);parser.add_argument('--blocks',type=int,default=256)
    args=parser.parse_args()
    from io_v2 import need
    need(sys.dont_write_bytecode,'launch with -B')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):need(os.environ.get(key)=='1','threads1')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU only')
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    root=Path(__file__).resolve().parent;manifest=json.loads((root/'RECOMPUTE_BUNDLE_V1.json').read_bytes())
    for name,item in manifest['files'].items():
        need(Path(name).name==name,'local declared leaf')
        p=root/name;need(p.is_file() and not p.is_symlink() and p.stat().st_size==item['bytes'],'declared local bytes')
        digest=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024**2),b''):digest.update(chunk)
        need(digest.hexdigest()==item['sha256'],'input/source SHA '+name)
    if args.verify_only:
        need(args.out is None,'verify-only creates no output');print('SHA verified; no model/native evaluated',flush=True);return 0
    need(args.out and 0<=args.first_block<256 and 1<=args.blocks<=256-args.first_block,'bounded explicit block interval')
    out=Path(args.out).absolute();need(not out.exists(),'exclusive output; preserve prior results');out.mkdir()
    import numpy as np
    import science
    from fk_v2 import Model
    from prospective_rules_v1 import model_rows,exhaustive_accounting,prospective_first32
    with np.load(root/'parameters.npz',allow_pickle=False) as z:layout=science.parameters_layout({k:z[k] for k in z.files})
    with np.load(root/'body_identity.npz',allow_pickle=False) as z:body={arm:z[arm] for arm in science.ARMS}
    model=Model(json.loads((root/'model.json').read_bytes()),json.loads((root/'geometry.json').read_bytes()),layout,body)
    q=np.load(root/'q74.npy',mmap_mode='r',allow_pickle=False);qd=np.load(root/'qd74.npy',mmap_mode='r',allow_pickle=False)
    need(q.shape==qd.shape==(8192,74) and q.dtype==qd.dtype==np.float32,'registered all8192 full74')
    compact=[]
    for block in range(args.first_block,args.first_block+args.blocks):
        first=32*block;batch=q[first:first+32].copy();before=batch.tobytes()
        g=model.predict_geometry(batch,global_ids=np.arange(first,first+32,dtype=np.int64));need(before==batch.tobytes(),'proposal bytes unchanged')
        records=model_rows(first,g);np.save(out/f'model_raw9021_{first:04d}.npy',g,allow_pickle=False)
        (out/f'ROWS_{first:04d}.json').write_text(json.dumps(records,allow_nan=False)+'\n')
        compact.extend({k:v for k,v in row.items() if k!='negative_rows'} for row in records)
        print('MODEL_ONLY_BLOCK',first,flush=True)
    result=dict(scope='RECOMPUTED_MODEL_ONLY_NOT_NATIVE',blocks=args.blocks,registered_denominator=8192,rows_computed=32*args.blocks,native_tested=0,camera_tested=0,eligible=0,bank_count=0,safety_pass=False,global_error_bound='UNKNOWN')
    if args.first_block==0 and args.blocks==256:result.update(accounting=exhaustive_accounting(compact),schedule=prospective_first32(compact))
    (out/'RECOMPUTED_MODEL_ONLY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return 0

if __name__=='__main__':raise SystemExit(main())
