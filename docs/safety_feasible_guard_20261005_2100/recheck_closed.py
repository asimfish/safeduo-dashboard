"""Read-only endpoint/FIFO reproduction; --dense adds registered full-forecast audit."""
from pathlib import Path
import argparse,json,hashlib,types
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main(dense=False):
    result=json.loads((HERE/'holdout_results.json').read_text());rows=[];totals={};outcomes={}
    plans=[json.loads(p.read_text()) for p in sorted((HERE/'holdout_v2').glob('*_plan.json'))]
    helper=HERE/'dense_audit.py';code=helper.read_bytes();expected=plans[0]['research_source_sha256'][str(helper)]
    assert hashlib.sha256(code).hexdigest()==expected
    m=types.ModuleType('read_only_dense_audit');m.__file__=str(helper);exec(compile(code,str(helper),'exec'),m.__dict__)
    for plan in plans:
        campaign=json.loads((Path(plan['output_root'])/'campaign.json').read_text());assert campaign['status']=='complete'
        for job in plan['jobs']:
            root=Path(plan['output_root'])/job['id'];protocol=json.loads((root/'protocol.json').read_text())
            assert protocol['status']=='complete' and protocol['completed_cells']==1
            assert protocol['source_sha256']==plan['source_sha256'] and protocol['checkpoint_sha256']==plan['checkpoint_sha256']
            file=root/'cell_001.npz';digest=sha(file)
            if dense:z=m.load(file)
            else:
                with np.load(file,allow_pickle=False) as data:z={k:data[k].copy() for k in ['q_initial','controller_target','actuator_target','official_margins']}
            margins=z['official_margins'];assert margins.shape==(960,64,4) and np.isfinite(margins).all()
            applied=np.concatenate([np.repeat(z['q_initial'][None],6,0),z['controller_target'][:-6]])
            assert np.array_equal(z['actuator_target'],applied)
            bad=(margins<0).any((0,2));deep=(margins<-.005).any((0,2))
            row=dict(id=job['id'],violations=int(bad.sum()),deep=int(deep.sum()),classes=(margins<0).any(0).sum(0).tolist(),cell_sha256=digest,actual_applied_fifo_exact=True)
            if dense:row['full_forecast_audit']=m.forecast_audit(root,z)
            assert sha(file)==digest
            mode=job['env']['SAFEDUO_JOINT_MODE'];totals.setdefault(mode,dict(windows=0,violations=0,deep=0))
            for key,value in [('windows',64),('violations',row['violations']),('deep',row['deep'])]:totals[mode][key]+=value
            parent=next(r for r in result['rows'] if r['id']==job['id']);assert row['violations']==parent['violations'] and row['deep']==parent['deep'] and row['classes']==parent['class_violations']
            assert digest==parent['input_sha256']['cell_001.npz'];outcomes[(mode,job['expected']['seed'])]=bad;rows.append(row)
    rescued=new=0
    for seed in {seed for _,seed in outcomes}:
        a=outcomes['joint_reference',seed];b=outcomes['joint_repair',seed];rescued+=int((a&~b).sum());new+=int((~a&b).sum())
    print(json.dumps(dict(status='PASS_READ_ONLY_REPRODUCTION',dense_audit_executed=dense,rows=rows,totals=totals,primary_rescued=rescued,primary_new_failures=new,
        scope='original stored strict/deep/class endpoints and actual appliedFIFO6; --dense additionally executes exact registered fullforecast helper; no metadata/raw writes, no safety approval'),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dense',action='store_true');a=p.parse_args();main(a.dense)
