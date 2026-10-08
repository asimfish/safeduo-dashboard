import json,hashlib
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,Gf
from runtime.scene_geometry import extract, fk_error, predict, corners
p=Path(__file__).resolve().parent
old=Path('/home/liyufeng/safeduo/artifacts/safety_hand_closure_qualification_20261007')
raw=Path('/mnt/nas/data/lyf/double_hand/safety_hand_closure_qualification_20261007/validation_fresh_v3')
r=json.loads((old/'REGISTRATION_VALIDATION_FRESH_V3.json').read_text());m=json.loads((raw/'native_metadata.json').read_text())
stage=Usd.Stage.CreateInMemory()
for arm,meta in m['arms'].items():
    prim=UsdGeom.Xform.Define(stage,'/World/envs/env_0/'+arm).GetPrim();prim.GetReferences().AddReference(meta['usd_path'])
props=[('nist_board','/home/liyufeng/safeduo/assets_real/isaac/Assets/Isaac/6.1/Isaac/Props/NIST/Taskboard/nistboard.usd',[-.95,.19,.8],0),('klt_spare','/home/liyufeng/safeduo/assets_real/objects/industrial/small_KLT/small_KLT_safeduo.usd',[.86,0,.8732],90),('parts_scale','/home/liyufeng/safeduo/assets_real/isaac/Assets/Isaac/4.5/Isaac/IsaacLab/Mimic/nut_pour_task/nut_pour_assets/sorting_scale.usd',[.15,.64,.8],0)]
for name,asset,pos,yaw in props:
    x=UsdGeom.Xform.Define(stage,'/World/envs/env_0/Prop_'+name);x.GetPrim().GetReferences().AddReference(asset);x.ClearXformOpOrder();x.AddTranslateOp(opSuffix='scene').Set(Gf.Vec3d(*pos));x.AddRotateZOp(opSuffix='scene').Set(yaw)
schema=extract(stage,m['arms']);xf=UsdGeom.XformCache()
for c in schema['colliders']:
    if not c['arm'] and c['owner']:
        t=np.array(xf.GetLocalToWorldTransform(stage.GetPrimAtPath(c['owner']))).T
        from runtime.scene_geometry import transform
        c['corners']=transform(t,np.array(c['corners'])).tolist();c['owner']=None
origin=np.asarray(m['origins'][0])
for c in m['collision_inventory']:
    if 'Table' not in c['path']:continue
    xyz=corners(np.asarray(c['bbox_min_m'])-origin,np.asarray(c['bbox_max_m'])-origin)
    schema['colliders'].append(dict(path=c['path'],owner=None,arm=None,body=None,corners=xyz.tolist(),contact_offset_m=.020,hand=False))
z=np.load(raw/'dense_0000_0119.npz');ix=119;records=[];maxp=maxq=0.
for e,case in enumerate(r['cases']):
    links={a:z[a+':link_pose'][ix,e].copy() for a in m['arms']}
    for a in links:links[a][:,:3]-=np.asarray(m['origins'][e])
    errors={}
    for a in ('U_L','U_R'):
        pe,qe=fk_error(schema,a,links[a],z[a+':q'][ix,e],m['arms'][a]);maxp=max(maxp,pe);maxq=max(maxq,qe);errors[a]=[pe,qe]
    for a in ('U_L','U_R'):
        result=predict(schema,a,links,m['arms'],r['passport']['open_rad'],{a:r['passport']['allowed_goals_rad'][a][0] for a in ('U_L','U_R')},{})
        records.append(dict(env=e,reference_time_s=case['reference_time_s'],arm=a,development_profile=48,fk_errors=errors,**result))
assert maxp<=.0005 and maxq<=.002,(maxp,maxq)
fail=[x for x in records if x['env'] in (9,10) and x['arm']=='U_L']
assert len(fail)==2 and all(not x['admitted'] for x in fail),fail
out=dict(status='PASS_ARCHIVED_FAILURE_PREDICTION_AND_NATIVE_FK',scope='Development only; actual primitive/mesh dimensions for fixtures and robots; table inventory bounds. Objects omitted in this offline diagnostic; native gate includes them.',source_registration_sha256=hashlib.sha256((old/'REGISTRATION_VALIDATION_FRESH_V3.json').read_bytes()).hexdigest(),collision_boxes=len(schema['colliders']),max_fk_position_error_m=maxp,max_fk_quaternion_error=maxq,records=records)
(p/'ARCHIVED_FAILURE_DIAGNOSTIC.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
(p/'ARCHIVE_ROBOT_GEOMETRY.json').write_text(json.dumps(schema,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2));print(json.dumps(fail,indent=2))
