"""Compile EXACT original sampler and layout AST; no RNG or arithmetic rewrite."""
import ast,io
import numpy as np
import common_v1 as c
TOTAL=32768;PER_SEED=8192;START=2048;STOP=10240

def validate_protocol(p):
 c.need(p['schema']=='astra.full74.fixed_supplemental_development_protocol.v1' and p['purpose']=='DEVELOPMENT_SUPPLEMENT_ONLY','explicit independent development supplement')
 c.need(p['supplemental_denominator']==TOTAL and p['per_original_seed']==PER_SEED and p['per_seed_family']==2048 and p['proposal_id_inclusive']==[START,STOP-1] and p['global_id_inclusive']==[8192,40959],'fixed count/range BEFORE outcomes; never loop to fill quota')
 c.need(p['original_denominator']==8192 and p['original_cohort']['sha256']=='bdc14bb135aa5b4801cae941110284aebecdad1f0a4b39d71f40e05622d47d8c','preserved original denominator identity')
 original=c.read(p['original_cohort']);identity=c.read(p['original_identity'])
 c.need(p['seed_integers']==original['identity']['seeds']==identity['seeds'] and len(set(p['seed_integers']))==4,'same four literal development seeds')
 c.need(p['family_rule']=='proposal_id % 4' and p['families']==original['identity']['sampling_families'],'same family law and exact quotas')
 c.need(p['rng']['method']=='ORIGINAL_PER_PROPOSAL_ADDRESSED_PCG64_EXTENSION' and p['rng']['sequential_four_stream_state_does_not_exist'] is True and p['rng']['not_claimed_as_persisted_PCG_state_continuation'] is True,'no invented sequential PCG state')
 c.need(p['global_id_rule']=='8192 + seed_index*8192 + (proposal_id-2048)' and p['old_global_id_rule']=='seed_index*2048 + proposal_id, ONLY original local IDs0..2047; never renumbered','no legacy ID reuse')
 c.need(p['no_zero_qd'] is True and p['no_clip'] is True and p['no_fit'] is True and p['no_outcome_conditional_sampling'] is True and p['no_ranges_or_family_changes'] is True,'unchanged full148 draws')
 c.need(p['acceptance']['formal_holdout_opened'] is False and p['execution']['native_launches_authorized'] is False,'no formal/native authority')
 return original,identity

def functions(p):
 original,identity=validate_protocol(p)
 scope=dict(np=np,N=32,D=74,ARMS=('F_L','F_R','U_L','U_R'),require=c.need,DEV_SEEDS=tuple(p['seed_integers']),RESERVED_FRESH_SEEDS=tuple(identity['reserved_holdout_seeds_excluded']))
 for r,name in [(p['original_layout'],'intervals'),(p['original_sampler'],'sample_batch')]:
  tree=ast.parse(c.raw(r,2*1024**2));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name];c.need(len(nodes)==1,'one exact source function '+name)
  exec(compile(ast.Module(body=nodes,type_ignores=[]),r['path'],'exec'),scope)
 with np.load(io.BytesIO(c.raw(c.pinned(c.R/'shard_raw_v1/batch0_shard0_raw/resolved_native_parameters.npz'))),allow_pickle=False) as z:params={k:z[k] for k in z.files}
 layout=scope['intervals'](params)
 c.need(layout['bounds'].dtype==layout['vmax'].dtype==np.float32 and len(layout['controlled'])==26 and len(layout['names'])==74,'exact native full74 layout')
 return scope['sample_batch'],layout,original

def identity_arrays():
 storage=np.arange(TOTAL,dtype=np.int64);seed=(storage//PER_SEED).astype(np.uint8);local=(START+storage%PER_SEED).astype(np.int64);global_id=8192+storage
 return dict(seed_index=seed,proposal_id=local,global_id=global_id)

def verify_original(got,expected):
 c.need(set(got)==set(expected) and got==expected,'FULL original8192 canonical NPY SHA must reproduce before any supplemental draw')

def validate_arrays(arrays,layout):
 q,v,f=arrays['q74'],arrays['qd74'],arrays['family'];c.need(q.shape==v.shape==(TOTAL,74) and q.dtype==v.dtype==np.float32 and f.shape==(TOTAL,) and f.dtype==np.uint8,'exact supplemental all32768/full148 schema')
 c.need(np.isfinite(q).all() and np.isfinite(v).all() and np.all(q>=layout['bounds'][:,0]) and np.all(q<=layout['bounds'][:,1]) and np.all(abs(v)<=layout['vmax']),'original hard∩soft26/hard48/all74 vmax')
 ids=identity_arrays();c.need(np.array_equal(f,(ids['proposal_id']%4).astype(np.uint8)),'original family IDs')
 for seed in range(4):
  for family in range(4):c.need(np.count_nonzero((ids['seed_index']==seed)&(f==family))==2048,'exact sixteen quotas, no outcome sampling')
 return ids

def validate_manifest(m):
 p=c.protocol();validate_protocol(p)
 c.need(m['schema']=='astra.full74.supplemental_frozen_inputs.v1' and m['status']=='FROZEN_INPUTS_ONLY_NO_MODEL_OUTCOMES' and m['protocol']==c.ref(c.HERE/'PROTOCOL_V1.json') and m['source_graph']==c.ref(c.HERE/'SOURCE_GRAPH_V1.json'),'exact versioned input manifest')
 c.need(m['original_cohort']==p['original_cohort'] and m['denominator']==TOTAL and m['original_denominator_preserved']==8192 and m['seed_integers']==p['seed_integers'],'source identity, both denominators')
 c.need(m['native_tested']==m['model_evaluated']==m['camera_tested']==m['eligible']==m['bank_count']==0 and m['formal_holdout_opened'] is False,'input freeze is not an outcome')
 c.need(m['rng_address_extension']==p['rng'] and m['global_id_rule']==p['global_id_rule'],'unchanged RNG/ID version')
 return m

def load_inputs(r):
 m=validate_manifest(c.read(r));root=c.Path(r['path']).parent
 a={}
 for n in ('q74','qd74','family','seed_index','proposal_id','global_id'):
  path=root/(n+'.npy');c.need(str(path) in m['files'],'frozen input path')
  a[n]=np.load(io.BytesIO(c.raw(dict(path=str(path),sha256=m['files'][str(path)]))),allow_pickle=False)
 _,layout,_=functions(c.protocol());ids=validate_arrays(a,layout)
 for n,x in ids.items():c.need(a[n].shape==x.shape and a[n].dtype==x.dtype and a[n].tobytes()==x.tobytes(),'stable no-overlap ID mapping '+n)
 return m,a,layout
