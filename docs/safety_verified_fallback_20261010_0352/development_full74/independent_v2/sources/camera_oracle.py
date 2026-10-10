"""Independent original mask/frustum/native27 audit of prospective camera v7.

Does NOT import producer hand_views_v3/qualified_views_v7 or parent validator.
Uses preserved independent V7 CPU optics/read primitives through GenericAudit.
Caller establishes CLOSED producer scope and independently computed event firsts.
"""
from pathlib import Path
import hashlib,itertools,json,sys
import numpy as np
from PIL import Image
import io
import camera_adapter as h
from evidence_io import require,parse_json

ARMS=('F_L','F_R','U_L','U_R');FINGERS=('thumb','index','middle','ring','little')

def token(prefix,path):return prefix+hashlib.sha256(path.encode()).hexdigest()
def ancestor(path,paths):
 if not isinstance(path,str) or not path.startswith('/'):return None
 while path:
  if path in paths:return path
  path=path.rstrip('/').rsplit('/',1)[0]
 return None

def family(name):
 parts=name.split('_')
 for f in FINGERS:
  if f in parts or (f=='little' and 'pinky' in parts):return f
 return None

def numeric_ids(raw):
 a=np.asarray(raw);require(a.shape==(720,1280,1) and a.dtype.kind in 'iu' and a.dtype.itemsize==4,'original integer segmentation array required')
 return a[...,0].view(np.uint32) if a.dtype.kind=='i' else a[...,0]

def raw_maps(outputs,info,inventory,context,palms,slot):
 """Independently resolve original integer IDs to exact native owner/subpart."""
 objects=dict(inventory)
 for p in context.values():
  if p.endswith(('/TableF','/TableU')):objects[p]=dict(env_id=int(p.split('/env_')[1].split('/')[0]),table=p.rsplit('/',1)[-1],hand=False)
 handtokens={token('qv4_hand_',p):p for p,m in inventory.items() if m['hand']}
 images={};maps={};partids={};errors=[]
 for name in ('instance_segmentation_fast','semantic_segmentation'):
  ids=numeric_ids(outputs[name]);labels=info[name]['idToLabels'];mapping={};pm={}
  require(isinstance(labels,dict),'original idToLabels missing')
  for key,label in labels.items():
   require(str(key).isdigit(),'invalid segmentation ID');key=int(key);require(key not in mapping,'duplicate normalized ID')
   owner=None;parts=[]
   if name=='instance_segmentation_fast':
    require(isinstance(label,str),'instance prim path required')
    owner=ancestor(label,objects)
   else:
    require(isinstance(label,dict),'semantic type map required')
    values=label.get('class','');require(isinstance(values,str),'semantic class string required')
    tokens=[s.strip() for s in values.split(',') if s.strip()]
    seen=[]
    for t in tokens:
     if t.startswith('qv4_hand_'):
      require(t in handtokens,'unregistered hand semantic token');seen.append(handtokens[t])
     elif t.startswith('qv6_context_'):
      require(t in context,'unregistered context semantic token');seen.append(context[t])
     elif t.startswith('qv6_palm_'):
      require(t in palms,'unregistered palm semantic token');parts.append(t)
    require(len(set(seen))<=1,'semantic tokens have ambiguous rigid owner');owner=seen[0] if seen else None
    for t in parts:require(owner==palms[t]['rigid_path'],'palm visual/native rigid ownership mismatch')
   mapping[key]=owner;pm[key]=parts
   if owner is not None and objects[owner]['env_id']!=slot and np.any(ids==key):errors.append('foreign_native_pixels')
  require(all(int(i) in mapping for i in np.unique(ids)),'unknown original segmentation pixel IDs')
  images[name]=ids;maps[name]=mapping;partids[name]=pm
 return objects,images,maps,partids,errors

def mask_counts(outputs,info,inventory,context,palms,slot):
 objects,ids,maps,parts,errors=raw_maps(outputs,info,inventory,context,palms,slot)
 i,s='instance_segmentation_fast','semantic_segmentation';counts={};palm_by_arm={a:0 for a in ARMS};family_by_arm={a:{f:0 for f in FINGERS} for a in ARMS}
 # Encode each mask once. A same-owner intersection cannot mix different links.
 codes={p:n+1 for n,p in enumerate(objects)}; decoded={};palm_pixels=None
 for name in (i,s):
  keys,inverse=np.unique(ids[name],return_inverse=True)
  owners=np.array([codes.get(maps[name][int(k)],0) for k in keys],np.int32)
  decoded[name]=owners[inverse].reshape(ids[name].shape)
  if name==s:
   is_palm=np.array([bool(parts[name][int(k)]) for k in keys],bool)
   palm_pixels=is_palm[inverse].reshape(ids[name].shape)
 same=(decoded[i]>0)&(decoded[i]==decoded[s])
 totals=np.bincount(decoded[i][same],minlength=len(codes)+1)
 palm_totals=np.bincount(decoded[i][same&palm_pixels],minlength=len(codes)+1)
 for path,row in objects.items():
  if row['env_id']!=slot:continue
  counts[path]=int(totals[codes[path]])
  if row.get('hand'):
   arm=row['arm'];finger=family(row['body_name'])
   if finger:family_by_arm[arm][finger]+=counts[path]
   palm_by_arm[arm]+=int(palm_totals[codes[path]])
 return dict(per_native_object_pixels=counts,palm_by_arm=palm_by_arm,family_by_arm=family_by_arm,errors=sorted(set(errors)))

def validate_palms_and_context(setup,inventory,asset_manifest):
 for path,digest in asset_manifest['source_files_sha256'].items():
  require(isinstance(digest,str) and len(digest)==64,'asset binding digest')
 n=setup['environment_count'];require(n in (32,64),'N32/64 registered inventory required')
 expected={p for p in inventory}|{f'/World/envs/env_{i}/{t}' for i in range(n) for t in ('TableF','TableU')}
 context={token('qv6_context_',p):p for p in sorted(expected)}
 require(setup['context_registry']==context,'context registry does not cover exact all-native bodies/tables')
 require(setup['semantic_token_to_native_path']=={token('qv4_hand_',p):p for p,m in inventory.items() if m['hand']},'base hand semantic identity mismatch')
 palms={}
 for p,m in inventory.items():
  entry=asset_manifest['arms'][m['arm']]
  for part in entry['parts']:
   if part['native_body_name']!=m['body_name']:continue
   visual=p+'/'+part['relative_visual_path'];key=token('qv6_palm_',visual)
   expected_meshes=sorted(p+'/'+suffix for suffix in part['relative_mesh_paths'])
   require(key in setup['palm_visual_registry'],'actual authored palm part missing')
   saved=setup['palm_visual_registry'][key]
   require(saved['rigid_path']==p and saved['visual_path']==visual and sorted(saved['actual_mesh_paths'])==expected_meshes,'palm visual is not exact independently inspected asset subtree')
   palms[key]=saved
 require(set(palms)==set(setup['palm_visual_registry']),'extra unverified palm visual token')
 require(all(any(x['rigid_path'].startswith(f'/World/envs/env_{i}/{a}/') for x in palms.values()) for i in range(n) for a in ARMS),'missing arm/lane palm parts')
 return context,palms

def bind_native_event(state,baseline,expected_native_fields,expected_event):
 event=(state['capture_kind'],state['env_id'],state['step'],state['substep'])
 require(event==tuple(expected_event),'not the independently requested exact micro event')
 required={a+'_'+f for a in ARMS for f in h.CHUNK_FIELDS}|set(h.CLOCK)
 require(required<=set(expected_native_fields)<=set(baseline),'external native micro binding incomplete')
 for key,x in expected_native_fields.items():require(h.equal_bits(baseline[key],x),'wrong actual micro/native field:'+key)
 require(state['simulation_time_s']==float(baseline['simulation_time_s']) and state['simulation_time_step_index']==int(baseline['simulation_time_step_index']),'JSON/actual native clock mismatch')
 return event

def selected_three(refs,attempts):
 require(len(refs)==3 and len({r['path'] for r in refs})==3,'three different selected original views required')
 bypath={r['rgb']['path']:r for r in attempts if 'rgb' in r};require(len(bypath)==sum('rgb' in r for r in attempts),'duplicate RGB attempt')
 result=[]
 for r in refs:
  require(r['path'] in bypath and r==bypath[r['path']]['rgb'],'selected image missing exact original attempt reference');result.append(bypath[r['path']])
 angles=[x['_independent_azimuth'] for x in result]
 require(all(abs((a-b+180)%360-180)>=30. for a,b in itertools.combinations(angles,2)),'three views not pairwise30deg')
 return result

def hand_group(records,inventory,slot,arm):
 paths=[p for p,x in inventory.items() if x['env_id']==slot and x['arm']==arm and x['hand']]
 peaks={p:max(r['_counts']['per_native_object_pixels'].get(p,0) for r in records) for p in paths}
 palm=max(r['_counts']['palm_by_arm'][arm] for r in records)
 fingers={f:max(r['_counts']['family_by_arm'][arm][f] for r in records) for f in FINGERS}
 eachbase=[sum(r['_counts']['per_native_object_pixels'].get(p,0) for p in paths) for r in records]
 passed=all(n>200 for n in eachbase) and palm>=2048 and all(n>=512 for n in fingers.values()) and all(not r['_counts']['errors'] and r['_framing_pass'] for r in records)
 return dict(qualified=bool(passed),palm_visual_peak_pixels=palm,finger_family_peak_pixels=fingers,per_native_body_peak_pixels=peaks,native_body_visibility_gaps=[p for p,n in peaks.items() if n<64],all_native_hand_bodies_observed=all(n>=64 for n in peaks.values()),full_mesh_coverage_certified=False)

def native_and_pixels(audit,rec,baseline):
 h.exact_snapshot(baseline,audit.snapshot(rec['native_before_all64']));h.exact_snapshot(baseline,audit.snapshot(rec['native_after_all64']))
 held=rec.get('held_renders',[])
 for i,row in enumerate(held):
  require(row['index']==i,'render order mismatch');h.exact_snapshot(baseline,audit.snapshot(row['native_before_all64']));h.exact_snapshot(baseline,audit.snapshot(row['native_after_all64']))
 if not held and 'original_outputs' not in rec:
  require(rec.get('render_calls',0)==0,'unrecorded render');return None,None
 require(len(held)==3 and rec['render_calls']==3,'actual three held render records required')
 before,after=np.asarray(rec['sensor_frame_before']),np.asarray(rec['sensor_frame_after'])
 require(before.shape==after.shape==(1,) and before.dtype.kind in 'iu' and after.dtype.kind in 'iu' and after[0]==before[0]+1 and rec['sensor_update_dt_s']==0.,'sensor frame increment/time mismatch')
 outputs=audit.npz(rec['original_outputs']);info=parse_json(audit.ref(rec['original_info']));rgb=outputs['rgb']
 require(rgb.dtype==np.uint8 and rgb.shape in ((720,1280,3),(720,1280,4)),'original RGB dimensions/type')
 require(hashlib.sha256(rgb.tobytes()).hexdigest()==rec['raw_rgb_sha256'],'original renderer RGB hash mismatch')
 with Image.open(io.BytesIO(audit.ref(rec['rgb']))) as im:
  require(im.format=='PNG' and im.size==(1280,720) and h.equal_bits(np.asarray(im),rgb),'PNG not exact original renderer pixels')
 return outputs,info

def wide_points_from_bound_record(record,baseline,inventory,slot,tables):
 expected={p for p,m in inventory.items() if m['env_id']==slot};require(set(record['bounds'])==expected,'wide authored bound native inventory')
 pts=[]
 for p,row in record['bounds'].items():
  m=inventory[p];pose=baseline[m['arm']+'_native_link_transforms_xyzw'][slot,m['body_index']].astype(float)
  require(np.array_equal(pose,np.asarray(row['actual_native_pose_xyzw'])),'wide body pose not captured native pose')
  lo,hi=np.asarray(row['local_lower']),np.asarray(row['local_upper']);require(lo.shape==hi.shape==(3,) and np.isfinite([lo,hi]).all() and np.all(lo<=hi),'invalid authored bound')
  corners=np.array(list(itertools.product(*zip(lo,hi))))
  x,y,z,w=pose[3:];norm=np.linalg.norm(pose[3:]);require(abs(norm-1)<1e-4,'invalid native quaternion');x,y,z,w=np.array([x,y,z,w])/norm
  R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
  pts.extend(corners@R.T+pose[:3])
 for table in tables:pts.extend(itertools.product(*zip(table['lower_world_m'],table['upper_world_m'])))
 points=np.asarray(pts)
 require(np.allclose(points,record['frustum_points_world_m'],atol=1e-12,rtol=0),'wide framing points differ from native pose reconstruction')
 return points

def audit_group(audit,state_ref,*,expected_native_fields,expected_event,expected_sources,asset_manifest):
 """GenericAudit(root,N) must already setup(); caller binds event from raw data.

 expected_event=(kind,lane,control,sub). expected_native_fields comes from actual
 independent point-chunk event (22fields), or full native initial snapshot.
 Held before/after/render snapshots always independently compare ALL27 fields.
 """
 state=parse_json(audit.ref(state_ref));slot=state['env_id']
 require(state['schema']=='safeduo.qualified_views.v7' and state['status']=='qualified','v7 qualified record required for positive strong camera evidence')
 require(state['environment_count']==audit.environment_count and state['all_registered_lanes'] is True and state['all64'] is (audit.environment_count==64),'registered lane metadata mismatch')
 require(state['physics_advanced_by_collector'] is False,'collector claims extra physics')
 require(set(expected_sources)=={'source_sha256','inherited_hand_views_sha256','frozen_native_hold_source_sha256'},'complete camera source binding required')
 for field,path in expected_sources.items():require(state[field]==audit.evidence.hash(path),'camera source hash mismatch:'+field)
 baseline=audit.snapshot(state['native_before_all64']);h.exact_snapshot(baseline,audit.snapshot(state['native_final_all64']))
 event=bind_native_event(state,baseline,expected_native_fields,expected_event)
 inventory=audit.load_inventory(state['native_path_mapping']);setup=parse_json(audit.ref(state['semantic_setup']))
 context,palms=validate_palms_and_context(setup,inventory,asset_manifest)
 attempts=[];arm_results={};allattemptrefs=set()
 for embedded in state['attempts']:
  record=parse_json(audit.ref(embedded['attempt_record']));require(record=={k:v for k,v in embedded.items() if k!='attempt_record'},'embedded raw attempt changed')
  allattemptrefs.add(embedded['attempt_record']['path']);rec=dict(record)
  outputs,info=native_and_pixels(audit,rec,baseline)
  if outputs is not None:
   arm=rec['arm'];paths=[(p,m) for p,m in inventory.items() if m['env_id']==slot and m['arm']==arm and m['hand']]
   points=baseline[arm+'_native_link_transforms_xyzw'][slot,[m['body_index'] for p,m in paths],:3].astype(float)
   require(np.array_equal(points,rec['hand_link_positions_world_m']),'hand framing not native link positions')
   framing=h.frustum(points,rec['camera']);rec['_framing_pass']=framing['pass_gate'];rec['_independent_azimuth']=framing['azimuth_deg']
   require(abs((rec['_independent_azimuth']-rec['azimuth_deg']+180)%360-180)<1e-4,'hand azimuth differs from actual camera matrix')
   rec['_counts']=mask_counts(outputs,info,inventory,context,palms,slot)
  attempts.append(rec)
 for arm in ARMS:
  armstate=state['arms'][arm];subset=[x for x in attempts if x['arm']==arm]
  require([r['attempt'] for r in subset]==list(range(1,len(subset)+1)) and len(subset)<=24,'hand attempt inventory gap')
  chosen=selected_three(armstate['qualified_images'],subset);result=hand_group(chosen,inventory,slot,arm)
  require(result['qualified'],'declared qualified hand group fails independent anatomical masks/frustum')
  for k in ('palm_visual_peak_pixels','finger_family_peak_pixels','per_native_body_peak_pixels','native_body_visibility_gaps','all_native_hand_bodies_observed'):require(armstate['group_coverage'][k]==result[k],'recorded hand coverage differs:'+k)
  def score(r):
   c=r['_counts'];f=c['family_by_arm'][arm];p=c['palm_by_arm'][arm];paths=result['per_native_body_peak_pixels']
   return (sum(f[v]>=512 for v in FINGERS),int(p>=2048),sum(c['per_native_object_pixels'].get(x,0)>=64 for x in paths),sum(f.values()),p)
  require(chosen[0]==max(chosen,key=score) and armstate['representative_image']==chosen[0]['rgb'],'representative view not highest declared anatomical score')
  arm_results[arm]=result
 wide=state['wide_context'];bounds=parse_json(audit.ref(wide['authored_bounds']));points=wide_points_from_bound_record(bounds,baseline,inventory,slot,state['actual_table_aabbs']);wide_attempts=[]
 for embedded in wide['attempts']:
  rec=parse_json(audit.ref(embedded['attempt_record']));require(rec=={k:v for k,v in embedded.items() if k!='attempt_record'},'wide embedded attempt differs')
  allattemptrefs.add(embedded['attempt_record']['path']);outputs,info=native_and_pixels(audit,rec,baseline)
  if outputs is not None:
   framing=h.frustum(points,rec['camera']);rec['_framing_pass']=framing['pass_gate'];rec['_independent_azimuth']=framing['azimuth_deg']
   require(abs((rec['_independent_azimuth']-rec['azimuth_deg']+180)%360-180)<1e-4,'wide azimuth differs from actual camera matrix')
   rec['_counts']=mask_counts(outputs,info,inventory,context,palms,slot)
  wide_attempts.append(rec)
 require([r['attempt'] for r in wide_attempts]==list(range(1,len(wide_attempts)+1)) and len(wide_attempts)<=6,'wide attempts missing/reordered')
 chosen=selected_three(wide['qualified_images'],wide_attempts);require(all(r['_framing_pass'] and not r['_counts']['errors'] for r in chosen),'wide frustum/mask failure')
 rows=[p for p,m in inventory.items() if m['env_id']==slot]+[f'/World/envs/env_{slot}/{t}' for t in ('TableF','TableU')]
 peaks={p:max(r['_counts']['per_native_object_pixels'][p] for r in chosen) for p in rows}
 armpeaks={a:max(sum(r['_counts']['per_native_object_pixels'][p] for p,m in inventory.items() if m['env_id']==slot and m['arm']==a) for r in chosen) for a in ARMS}
 tablepeaks={t:peaks[f'/World/envs/env_{slot}/{t}'] for t in ('TableF','TableU')}
 critical={p:n for p,n in peaks.items() if p in inventory and inventory[p]['body_name'] in ('forearm_link','wrist_2_link')}
 require(all(n>=1024 for n in [*armpeaks.values(),*tablepeaks.values()]) and all(n>=128 for n in critical.values()),'wide allfour/table/critical native body observation insufficient')
 for k,val in [('per_native_object_peak_pixels',peaks),('arm_peak_pixels',armpeaks),('table_peak_pixels',tablepeaks),('critical_body_peak_pixels',critical)]:require(wide['group_coverage'][k]==val,'wide reported coverage differs:'+k)
 directory=(audit.root/state_ref['path']).parent
 require(allattemptrefs=={str(p.relative_to(audit.root)) for p in directory.glob('*.attempt.json')},'orphan or omitted original attempts')
 return dict(status='PASS_STRONG_CAMERA_EVIDENCE_ONLY',event=event,arms=arm_results,wide_arm_pixels=armpeaks,wide_table_pixels=tablepeaks,critical_body_pixels=critical,
  original_raw_attempt_count=len(allattemptrefs),held_native_fields=27,external_micro_native_bound_fields=len(expected_native_fields),native_exact_micro_clock_bound=True,
  limitations=['Pixel region presence is not complete mesh/surface or freedom-from-occlusion certification.',
   'Wide local authored bounds/table AABBs are original recorded framing inputs; native pose transform and six-plane projection independently rebuilt, not independently cooked mesh bounds.',
   'All27 fields are checked bitwise in each held capture. Short-screen native rows independently bind 24 arm fields and two clock fields; environment_origins are held camera bytes, not an independently regenerated origin grid.'],
  physical_safety_certified=False,full_mesh_coverage_certified=False,contact_patch_visibility_certified=False)
