"""HELD future exhaustive32768 CPU MODEL_ONLY; never a native driver."""
import argparse,time
import numpy as np
import common_v1 as c
from sampling_v1 import TOTAL,load_inputs
from extended_rules_v1 import model_rows,accounting
from model_adapter_v1 import authority,actual_model
STATES={'MODEL_NOT_EVALUATED':0,'MODEL_UNKNOWN':1,'MODEL_NEGATIVE':2,'MODEL_PREDICTED_INITIAL_CLEAR':3}
def main():
 c.limits();p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--check-cpu',action='store_true');p.add_argument('--authority');p.add_argument('--authority-sha256');p.add_argument('--out');a=p.parse_args();graph=c.source_graph()
 if a.check_cpu:
  c.need(not (a.authority or a.authority_sha256 or a.out),'check source only');print('SOURCE_ONLY_SUPPLEMENT_MODEL0_NATIVE0',flush=True);return 0
 c.need(a.authority and a.authority_sha256 and a.out,'future exact parent authority')
 auth=authority(dict(path=a.authority,sha256=a.authority_sha256));out=c.Path(a.out);c.need(out.resolve()==out and out.is_relative_to(c.OUT/'MODEL_ONLY') and str(out)==auth['out'] and not out.exists(),'exclusive parent-bound supplemental result')
 started=time.monotonic();deadline=started+auth['max_wall_s'];manifest,arrays,layout=load_inputs(auth['supplemental_cohort']);model=actual_model(auth);out.mkdir(parents=True,exist_ok=False);(out/'blocks').mkdir()
 dispositions=np.zeros(TOTAL,np.uint8);rows=[];error=None
 c.npy(out/'REGISTERED_ALL32768_NOT_EVALUATED.npy',dispositions);c.write(out/'REGISTRATION_V1.json',dict(registered=TOTAL,original8192='PRESERVED_SEPARATE_DENOMINATOR',supplemental_inputs=auth['supplemental_cohort'],scope='MODEL_ONLY',missing_final_receipt='INCOMPLETE_NOT_PASS'))
 report=dict(schema='astra.supplemental.model_stream.v1',status='INCOMPLETE_MODEL_ONLY',denominator=TOTAL,completed=0,native_tested=0,camera_tested=0,eligible=0,bank_count=0,global_error_bound='UNKNOWN',states=STATES)
 try:
  for first in range(0,TOTAL,32):
   c.need(time.monotonic()<deadline,'finite fixed model deadline');q=arrays['q74'][first:first+32].copy();v=arrays['qd74'][first:first+32].copy();qb=q.tobytes();vb=v.tobytes();q.setflags(write=False);v.setflags(write=False)
   gaps=model.predict_geometry(q,global_ids=arrays['global_id'][first:first+32].copy());c.need(q.tobytes()==qb and v.tobytes()==vb,'no input clipping/qdreset/projection')
   c.need(time.monotonic()<deadline,'model block deadline');rr=model_rows(first,gaps);block=out/'blocks'/f'block_{first:05d}';block.mkdir();files={}
   for name,x in [('q74',q),('qd74',v),('global_id',arrays['global_id'][first:first+32]),('model_raw9021',gaps)]:
    r=c.npy(block/(name+'.npy'),x);files[r['path']]=r['sha256']
   r=c.write(block/'ROWS.json',rr);files[r['path']]=r['sha256'];c.write(block/'COMMIT.json',dict(storage_first=first,first_global_id=8192+first,count=32,files=files,scope='SUPPLEMENTAL_MODEL_ONLY'))
   rows.extend({k:v for k,v in r.items() if k!='negative_rows'} for r in rr)
   for r in rr:dispositions[r['storage_row']]=STATES[r['model_status']]
   report['completed']=first+32
  report['accounting']=accounting(rows);report['status']='COMPLETE_SUPPLEMENTAL_MODEL_ONLY_PENDING_REAL_WAIT_AND_DATA_PEER'
 except BaseException as exc:error=exc;report['error']=type(exc).__name__+': '+str(exc)
 finally:
  c.npy(out/'ALL32768_MODEL_DISPOSITIONS.npy',dispositions);report.update(not_evaluated=int(np.count_nonzero(dispositions==0)),authority=dict(path=a.authority,sha256=a.authority_sha256),source_graph=graph,supplemental_cohort=auth['supplemental_cohort'],original_cohort=c.protocol()['original_cohort'],elapsed_s=time.monotonic()-started,artifacts={str(p):c.sha(p) for p in out.rglob('*') if p.is_file()});c.write(out/'SUPPLEMENTAL_MODEL_STREAM_V1.json',report)
 if error is not None:raise error
 c.source_graph();return 0
if __name__=='__main__':raise SystemExit(main())
