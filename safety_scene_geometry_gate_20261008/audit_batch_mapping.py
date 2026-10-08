"""Independent original-stream oracle; imports no batch adapter/controller."""
import json,hashlib,argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
pa=argparse.ArgumentParser();pa.add_argument('--name',required=True);pa.add_argument('--output',type=Path);a=pa.parse_args();p=Path(__file__).resolve().parent;raw=Path('/mnt/nas/data/lyf/double_hand/safety_scene_geometry_gate_20261008')/a.name
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
class ChunkArrays(dict):
 def __init__(self,source):super().__init__();self.source=source
 def __missing__(self,key):
  value=self.source[key];self[key]=value;return value
receipt=json.loads((raw/'recording_receipt.json').read_text());ids=json.loads((raw/'batch_contact_identities.json').read_text());legacy=json.loads((raw/'contact_identities.json').read_text());pairs=0;point_values=0;maximum=0.;group_rows=defaultdict(list)
for row in ids:
 e,key=row['env'],row['key'];is_hand=key in ['F_L','F_R','U_L','U_R'];original=next(v for v in legacy['hands' if is_hand else 'objects'] if v['env']==e and v['arm' if is_hand else 'object']==key)
 assert row['sensor_paths']==original['sensors']
 assert row['filter_paths']==(original['filters'] if is_hand else [original['filters']])
 assert len(row['sensor_indices'])==len(row['sensor_paths'])
 group_rows[row['group']].extend(row['sensor_indices'])
for group,rows in group_rows.items():assert sorted(rows)==list(range(len(rows))),('mapping_bijection',group)
for ch in receipt['chunks']:
 f=raw/ch['file'];assert sha(f)==ch['sha256']
 with np.load(f) as source:
  z=ChunkArrays(source);prefix_by_group={}
  for row in ids:
   e,key,group,rows=row['env'],row['key'],row['group'],row['sensor_indices'];base='batch:'+group+':normal:';gc=z[base+'count'][:,rows];gs=z[base+'start'][:,rows];gf=z[base+'force'].reshape(len(z['step']),-1)
   assert (gc>=0).all() and np.all((gc==0)|((gs>=0)&(gs+gc<=gf.shape[1])))
   ix=np.arange(len(gf))[:,None,None];ss=np.where(gc>0,gs,0).astype(int)
   if group not in prefix_by_group:prefix_by_group[group]=np.pad(abs(gf).astype(np.float64).cumsum(1),((0,0),(1,0)))
   prefix=prefix_by_group[group];oracle=prefix[ix,ss+gc]-prefix[ix,ss]
   if key in ['F_L','F_R','U_L','U_R']:
    c=z[f'e{e}:{key}:hand_normal_count'];assert np.array_equal(c,gc);actual=z[f'e{e}:{key}:hand_partner_scalar_normal'];err=float(abs(oracle-actual).max());maximum=max(maximum,err);assert np.all(abs(oracle-actual)<=.001+2e-6*oracle)
    if key in ['U_L','U_R']:
     st=z[f'e{e}:{key}:hand_point_start'];local_force=z[f'e{e}:{key}:hand_point_force']
     for t,s,j in zip(*np.nonzero(c)):
      n=int(c[t,s,j]);local=int(st[t,s,j]);global_start=int(gs[t,s,j]);assert np.array_equal(local_force[t,local:local+n],gf[t,global_start:global_start+n]);pairs+=1;point_values+=n
   else:
    assert np.array_equal(z[f'e{e}:{key}:normal_count'],gc[:,0]);base='batch:'+group+':friction:';fc=z[base+'count'][:,rows];fs=z[base+'start'][:,rows];ff=z[base+'force'];assert (fc>=0).all();assert np.array_equal(z[f'e{e}:{key}:friction_count'],fc[:,0]);out=np.zeros_like(z[f'e{e}:{key}:friction'],dtype=np.float64)
    for t,s,j in zip(*np.nonzero(fc)):
     n=int(fc[t,s,j]);st=int(fs[t,s,j]);assert 0<=st and st+n<=ff.shape[1];out[t,j]=ff[t,st:st+n].sum(0,dtype=np.float64)
    assert np.allclose(out,z[f'e{e}:{key}:friction'],atol=.001,rtol=2e-6)
result=dict(status='PASS_ALL_RECORDED_LOCAL_POINTS_MATCH_ORIGINAL_GLOBAL_NATIVE_STREAM',run=a.name,source_receipt_sha256=sha(raw/'recording_receipt.json'),source_identity_sha256=sha(raw/'batch_contact_identities.json'),legacy_identity_sha256=sha(raw/'contact_identities.json'),independent_sensor_filter_identity_check=True,global_sensor_mapping_bijection=True,all_four_hand_scalar_streams_checked=True,steps=receipt['steps'],nonzero_sensor_partner_frames=pairs,original_point_values_compared=point_values,max_scalar_reconstruction_error_n=maximum,object_friction_reconstructed=True)
result['analysis_source_sha256']=sha(Path(__file__))
(a.output or p/(a.name.upper()+'_MAPPING_GATE.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
