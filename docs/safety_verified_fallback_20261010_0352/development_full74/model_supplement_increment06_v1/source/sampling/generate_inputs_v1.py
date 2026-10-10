"""One fixed input generation: reproduce8192 exactly, then source-address32768."""
import argparse,datetime,time,io,hashlib
import numpy as np
import common_v1 as c
from sampling_v1 import *
def main():
 c.limits();p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--out',required=True);a=p.parse_args();graph=c.source_graph();proto=c.protocol();original,identity=validate_protocol(proto)
 root=c.Path(a.out);c.need(root==c.OUT/'inputs_v1' and root.resolve()==root and not root.exists(),'one fixed new input namespace, no overwrite/retry sampler')
 root.mkdir(parents=True,exist_ok=False);(root/'original_replay').mkdir();started=time.monotonic();deadline=started+600
 c.write(root/'REGISTRATION_V1.json',dict(protocol=c.ref(c.HERE/'PROTOCOL_V1.json'),source_graph=graph,registered=TOTAL,each_seed=PER_SEED,each_family=2048,purpose='INPUTS_ONLY',model_evaluated=0,native_launches=0,no_adaptive_resampling=True,missing_final_manifest='INCOMPLETE_INPUTS_NOT_RELEASED'))
 original_refs={p:c.sha(p) for p in original['files']};c.need(original_refs==original['files'],'old arrays exact before reproduction')
 sample,layout,_=functions(proto);error=None;completed=0;reproduction=None;manifest=None
 try:
  old={n:np.empty(shape,dt) for n,shape,dt in [('q74',(8192,74),np.float32),('qd74',(8192,74),np.float32),('family',(8192,),np.uint8)]}
  for s,seed in enumerate(proto['seed_integers']):
   for start in range(0,2048,32):
    c.need(time.monotonic()<deadline,'finite input deadline');batch=sample(layout,seed,start)
    for n,x in old.items():x[s*2048+start:s*2048+start+32]=batch[n]
  actual={};names={}
  for n,x in old.items():
   r=c.npy(root/'original_replay'/(n+'.npy'),x);names[n]=r
   original_path=str(c.Path(proto['original_cohort']['path']).parent/(n+'.npy'));actual[original_path]=r['sha256']
   y=np.load(io.BytesIO(c.raw(dict(path=original_path,sha256=original['files'][original_path]))),allow_pickle=False)
   c.need(x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes(),'every original bit '+n)
  verify_original(actual,original['files']);del old
  reproduction=c.write(root/'ORIGINAL8192_REPLAY_EXACT_V1.json',dict(status='PASS_EXACT_ALL8192_ORIGINAL_BYTES',original_manifest=proto['original_cohort'],original_files=original['files'],regenerated=names,original_array_SHA=actual,all8192_q74_qd74_family_bits_exact=True,numpy_version=np.__version__,PCG='PCG64 per-proposal SeedSequence([seed,pid])',persistent_cross_proposal_state_existed=False,source=proto['original_sampler'],supplemental_draws_started=False))
  # Replay receipt is durable BEFORE the first new address is drawn.
  (root/'blocks').mkdir();shape=(TOTAL,74)
  arrays={n:np.empty(shape if n!='family' else (TOTAL,),np.float32 if n!='family' else np.uint8) for n in ('q74','qd74','family')}
  for s,seed in enumerate(proto['seed_integers']):
   for start in range(START,STOP,32):
    c.need(time.monotonic()<deadline,'finite fixed generation deadline');batch=sample(layout,seed,start);offset=s*PER_SEED+start-START
    for n,x in arrays.items():x[offset:offset+32]=batch[n]
    c.write(root/'blocks'/f'{offset:05d}.json',dict(storage_first=offset,count=32,seed_index=s,proposal_ids=[start,start+31],global_ids=[8192+offset,8192+offset+31],family_counts=[8]*4,scope='FIXED_INPUT_DRAW_ONLY_NO_MODEL'))
    completed+=32
  ids=validate_arrays(arrays,layout);arrays.update(ids)
  for n,x in arrays.items():c.npy(root/(n+'.npy'),x)
  with (root/'INPUT_BOUNDS_V1.npz').open('xb') as f:np.savez_compressed(f,**{n:layout[n] for n in ('bounds','hard','soft','vmax','controlled','names')});f.flush();c.os.fsync(f.fileno())
  duplicates=0;seen=set()
  for i in range(TOTAL):
   digest=hashlib.sha256(arrays['q74'][i].tobytes()+arrays['qd74'][i].tobytes()).digest()
   duplicates+=digest in seen;seen.add(digest)
  manifest=dict(schema='astra.full74.supplemental_frozen_inputs.v1',status='FROZEN_INPUTS_ONLY_NO_MODEL_OUTCOMES',purpose='DEVELOPMENT_SUPPLEMENT_ONLY',protocol=c.ref(c.HERE/'PROTOCOL_V1.json'),source_graph=graph,original_cohort=proto['original_cohort'],denominator=TOTAL,original_denominator_preserved=8192,seed_integers=proto['seed_integers'],local_id_range=[START,STOP-1],global_id_rule=proto['global_id_rule'],global_id_range=[8192,40959],rng_address_extension=proto['rng'],original_reproduction=reproduction,parameter=proto['model_unchanged']['parameter'],numpy_version=np.__version__,per_seed_family_counts=[[2048]*4 for s in range(4)],duplicate_full148_bytes_observed=duplicates,duplicates_not_resampled=True,files={str(p):c.sha(p) for p in root.glob('*.npy')},input_bounds=c.ref(root/'INPUT_BOUNDS_V1.npz'),block_registration_count=1024,blocks={str(p):c.sha(p) for p in (root/'blocks').glob('*.json')},native_tested=0,model_evaluated=0,camera_tested=0,eligible=0,bank_count=0,formal_holdout_opened=False,global_error_bound='UNKNOWN',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  c.need({p:c.sha(p) for p in original_refs}==original_refs,'old input bytes unchanged after fixed generation')
  validate_manifest(manifest);c.write(root/'SUPPLEMENTAL_PROPOSALS_V1.json',manifest)
 except BaseException as exc:error=exc
 finally:
  c.write(root/'INPUT_GENERATION_TERMINAL_V1.json',dict(status='INPUTS_COMPLETE_REQUIRES_REAL_CPU_WAIT_AND_REVIEW' if error is None else 'FAILED_INPUT_GENERATION_ALL_ATTEMPTS_PRESERVED',error=None if error is None else type(error).__name__+': '+str(error),registered=TOTAL,completed_draws=completed,original_replay=reproduction,manifest=c.ref(root/'SUPPLEMENTAL_PROPOSALS_V1.json') if (root/'SUPPLEMENTAL_PROPOSALS_V1.json').exists() else None,native_launches=0,model_evaluations=0,elapsed_s=time.monotonic()-started,actual_wait='REQUIRED_SEPARATE_REAL_CHILD_WAIT'))
 if error is not None:raise error
 print('FROZEN_FIXED_32768_INPUTS_ONLY_REPLAY8192_EXACT_NATIVE0_MODEL0',flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
